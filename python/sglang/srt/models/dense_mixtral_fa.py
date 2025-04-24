import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import MixtralConfig
import torch.cuda.nvtx as nvtx

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from sglang.srt.profiling import create_profiling_context, TimerContext, ENABLE_CUDA_EVENTS

from sglang.srt.models.mixtral import MixtralAttention
from sglang.srt.models.llama import LlamaMLP


# Global profiling context
_PROFILING_CTX = create_profiling_context()


class DenseMixtralFADecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.ffn = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size * config.num_experts_per_tok,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Only create profiling regions if profiling is active
        is_profiling = _PROFILING_CTX is not None and _PROFILING_CTX.profiling_active

        # Self Attention
        if residual is None:
            residual = hidden_states
            # NVTX tracker for first normalization
            if is_profiling:
                nvtx.range_push(f"layer{self.layer_id}_input_norm")
            hidden_states = self.input_layernorm(hidden_states)
            if is_profiling:
                nvtx.range_pop()
        else:
            # NVTX tracker for first normalization with residual
            if is_profiling:
                nvtx.range_push(f"layer{self.layer_id}_input_norm")
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            if is_profiling:
                nvtx.range_pop()

        # NVTX tracker for attention
        attn_timer = TimerContext(self.layer_id, "attn") if is_profiling and ENABLE_CUDA_EVENTS else None

        if is_profiling:
            nvtx.range_push(f"layer{self.layer_id}_attention")

        if attn_timer:
            with attn_timer:
                hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if is_profiling:
            nvtx.range_pop()

        # Fully Connected
        # NVTX tracker for post-attention normalization
        if is_profiling:
            nvtx.range_push(f"layer{self.layer_id}_post_attn_norm")
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if is_profiling:
            nvtx.range_pop()

        # NVTX tracker for FFN
        ffn_timer = TimerContext(self.layer_id, "ffn") if is_profiling and ENABLE_CUDA_EVENTS else None

        if is_profiling:
            nvtx.range_push(f"layer{self.layer_id}_ffn")

        if ffn_timer:
            with ffn_timer:
                hidden_states = self.ffn(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)

        if is_profiling:
            nvtx.range_pop()

        # Record timing information if CUDA events are enabled
        if attn_timer and ffn_timer and _PROFILING_CTX is not None:
            _PROFILING_CTX.record_timing(
                self.layer_id,
                attn_timer.get_elapsed_time(),
                ffn_timer.get_elapsed_time()
            )

        return hidden_states, residual


class DenseMixtralFAModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                DenseMixtralFADecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # Check if profiling is active
        is_profiling = _PROFILING_CTX is not None and _PROFILING_CTX.profiling_active

        if input_embeds is None:
            # NVTX tracker for token embedding
            if is_profiling:
                nvtx.range_push("token_embedding")
            hidden_states = self.embed_tokens(input_ids)
            if is_profiling:
                nvtx.range_pop()
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(len(self.layers)):
            # NVTX tracker for each decoder layer
            if is_profiling:
                nvtx.range_push(f"decoder_layer_{i}")
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
            if is_profiling:
                nvtx.range_pop()

        # NVTX tracker for final normalization
        if is_profiling:
            nvtx.range_push("final_norm")
        hidden_states, _ = self.norm(hidden_states, residual)
        if is_profiling:
            nvtx.range_pop()

        return hidden_states


class DenseMixtralFAForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DenseMixtralFAModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, prefix=add_prefix("lm_head", prefix)
        )
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        # Check if CUDA graph capture is completed (only relevant if profiling is enabled)
        is_capturing = False
        if _PROFILING_CTX is not None and torch.cuda.is_available():
            is_capturing = torch.cuda.is_current_stream_capturing()

        # Update profiling state if profiling is enabled
        is_profiling = False
        if _PROFILING_CTX is not None:
            is_profiling = _PROFILING_CTX.update(is_capturing)

        # NVTX tracker for model forward with iteration count
        if is_profiling:
            nvtx.range_push(f"model_forward_{_PROFILING_CTX.forward_count}")

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        if is_profiling:
            nvtx.range_pop()

        # NVTX tracker for logits processing and language modeling head
        if is_profiling:
            nvtx.range_push("lm_head_and_logits")

        result = self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

        if is_profiling:
            nvtx.range_pop()

        # Check if we should print timings at the end of this forward pass
        if _PROFILING_CTX is not None and _PROFILING_CTX.should_print_timings:
            _PROFILING_CTX.print_accumulated_timings()

        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Implement weight loading logic if needed
        pass


EntryClass = DenseMixtralFAForCausalLM
