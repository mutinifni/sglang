# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Profiling utilities for model performance analysis."""

import os
import torch
import datetime
import json

# Global profiling state controlled by environment variables
ENABLE_NSYS = os.getenv('SGL_ENABLE_NSYS', '0').lower() in ('1', 'true', 'yes')
ENABLE_CUDA_EVENTS = os.getenv('SGL_ENABLE_CUDA_EVENTS', '0').lower() in ('1', 'true', 'yes')
PROFILING_WARMUP_PASSES = int(os.getenv('SGL_PROFILING_WARMUP_PASSES', '700'))
PROFILING_ITERS = int(os.getenv('SGL_PROFILING_ITERS', '30'))
CUDA_EVENTS_LOG_FILE = os.getenv('SGL_CUDA_EVENTS_LOG_FILE', './model_profile.txt')


class ProfilingContext:
    """Manages profiling state and metrics collection."""

    def __init__(self):
        self.forward_count = 0
        # Common profiling state
        self.profiling_active = False
        self.profiling_completed = False
        self.profiling_iterations = 0
        # NSYS specific state
        self.nsys_active = False
        # CUDA events state
        self.accumulated_layer_timings = {}
        self.should_print_timings = False

        # Ensure the directory for the output file exists
        if ENABLE_CUDA_EVENTS:
            output_dir = os.path.dirname(CUDA_EVENTS_LOG_FILE)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    if torch.distributed.get_rank() == 0:
                        print(f"Created directory for CUDA events log: {output_dir}")
                except Exception as e:
                    if torch.distributed.get_rank() == 0:
                        print(f"Warning: Could not create directory for CUDA events log: {e}")

        # Print configuration at initialization
        if (ENABLE_NSYS or ENABLE_CUDA_EVENTS) and torch.distributed.get_rank() == 0:
            print(f"Profiling configuration:")
            print(f"  - NSYS profiling: {'enabled' if ENABLE_NSYS else 'disabled'}")
            print(f"  - CUDA events: {'enabled' if ENABLE_CUDA_EVENTS else 'disabled'}")
            print(f"  - Warmup passes: {PROFILING_WARMUP_PASSES}")
            print(f"  - Profile iterations: {PROFILING_ITERS}")
            if ENABLE_CUDA_EVENTS:
                print(f"  - CUDA events log file: {CUDA_EVENTS_LOG_FILE}")

    def update(self, is_capturing: bool) -> bool:
        """Update profiling state and return whether profiling is active.

        Args:
            is_capturing: Whether CUDA graph capture is in progress

        Returns:
            bool: Whether profiling is currently active
        """
        if not ENABLE_NSYS and not ENABLE_CUDA_EVENTS:
            return False

        if not is_capturing:
            # Increment forward count unconditionally
            self.forward_count += 1

            # Check if we should start profiling - only if not already completed
            if not self.profiling_active and not self.profiling_completed and self.forward_count >= PROFILING_WARMUP_PASSES:
                self.start_profiling()

            # Update profiling iterations if profiling is active
            if self.profiling_active:
                self.profiling_iterations += 1
                if self.profiling_iterations >= PROFILING_ITERS:
                    self.stop_profiling()
                    # Set flag to save timings at the end of this forward pass if CUDA event timing was enabled
                    if ENABLE_CUDA_EVENTS:
                        self.should_print_timings = True

        # Return whether profiling is active
        return self.profiling_active

    def record_timing(self, layer_id: int, attn_ms: float, ffn_ms: float):
        """Record timing information for a layer."""
        if not self.profiling_active:
            return

        if layer_id not in self.accumulated_layer_timings:
            self.accumulated_layer_timings[layer_id] = []

        self.accumulated_layer_timings[layer_id].append({
            "attn_ms": attn_ms,
            "ffn_ms": ffn_ms,
            "iteration": self.profiling_iterations
        })

    def start_profiling(self):
        """Start profiling based on enabled features."""
        self.profiling_active = True

        # Start NSYS profiling if enabled
        if ENABLE_NSYS and torch.cuda.is_available():
            if torch.distributed.get_rank() == 0:
                print(f"Starting NSYS profiling after {PROFILING_WARMUP_PASSES} warmup passes")
            torch.cuda.cudart().cudaProfilerStart()
            self.nsys_active = True

        if ENABLE_CUDA_EVENTS and torch.distributed.get_rank() == 0:
            print(f"Starting CUDA event profiling after {PROFILING_WARMUP_PASSES} warmup passes")

        # Reset profiling iterations and timing data
        self.profiling_iterations = 0
        self.accumulated_layer_timings = {}

    def stop_profiling(self):
        """Stop all active profiling."""
        self.profiling_active = False
        self.profiling_completed = True

        # Stop NSYS profiling if it was active
        if self.nsys_active and torch.cuda.is_available():
            if torch.distributed.get_rank() == 0:
                print(f"Stopping NSYS profiling after {PROFILING_ITERS} iterations")
            torch.cuda.cudart().cudaProfilerStop()
            self.nsys_active = False

        if ENABLE_CUDA_EVENTS and torch.distributed.get_rank() == 0:
            print(f"Stopping CUDA event profiling after {PROFILING_ITERS} iterations")

        if torch.distributed.get_rank() == 0:
            print(f"Total forward passes before profiling stopped: {self.forward_count}")

    def print_accumulated_timings(self):
        """Save accumulated timing statistics to a file."""
        if not self.accumulated_layer_timings:
            if torch.distributed.get_rank() == 0:
                print("No accumulated timing data available, skipping file output")
            return

        if torch.distributed.get_rank() != 0:
            return  # Only save on rank 0

        # Get output file paths
        output_file = CUDA_EVENTS_LOG_FILE
        json_output_file = f"{os.path.splitext(output_file)[0]}.json"

        try:
            with open(output_file, 'w') as f:
                total_iterations = sum(len(timings) for timings in self.accumulated_layer_timings.values()) / len(self.accumulated_layer_timings)

                f.write(f"\n===== Layer Timing Analysis (GPU-measured) =====\n")
                f.write(f"Averaged over {int(total_iterations)} iterations across {len(self.accumulated_layer_timings)} layers\n")
                f.write("\nPer-Layer Breakdown:\n")
                f.write(f"{'Layer':<6} {'Attention (ms)':<15} {'FFN (ms)':<15} {'Total (ms)':<10} {'Attn %':<10} {'FFN %':<10}\n")
                f.write("-" * 70 + "\n")

                total_attn_ms = 0
                total_ffn_ms = 0

                for layer_id, timings in sorted(self.accumulated_layer_timings.items()):
                    avg_attn_ms = sum(t["attn_ms"] for t in timings) / len(timings)
                    avg_ffn_ms = sum(t["ffn_ms"] for t in timings) / len(timings)
                    total_ms = avg_attn_ms + avg_ffn_ms
                    attn_pct = (avg_attn_ms / total_ms * 100) if total_ms > 0 else 0
                    ffn_pct = (avg_ffn_ms / total_ms * 100) if total_ms > 0 else 0

                    f.write(f"{layer_id:<6} {avg_attn_ms:<15.3f} {avg_ffn_ms:<15.3f} {total_ms:<10.3f} {attn_pct:<10.1f} {ffn_pct:<10.1f}\n")

                    total_attn_ms += avg_attn_ms
                    total_ffn_ms += avg_ffn_ms

                grand_total_ms = total_attn_ms + total_ffn_ms
                attn_pct = (total_attn_ms / grand_total_ms * 100) if grand_total_ms > 0 else 0
                ffn_pct = (total_ffn_ms / grand_total_ms * 100) if grand_total_ms > 0 else 0

                f.write("\nSummary:\n")
                f.write(f"Total Attention time: {total_attn_ms:.3f} ms ({attn_pct:.1f}%)\n")
                f.write(f"Total FFN time:       {total_ffn_ms:.3f} ms ({ffn_pct:.1f}%)\n")
                f.write(f"Total compute time:   {grand_total_ms:.3f} ms\n")
                f.write("==============================================\n")

            print(f"Saved CUDA events profiling results to {output_file}")

            # Also save raw data in JSON format for later analysis
            with open(json_output_file, 'w') as f:
                # Add timestamp for reference
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                json_data = {
                    "metadata": {
                        "timestamp": timestamp,
                        "warmup_passes": PROFILING_WARMUP_PASSES,
                        "profile_iterations": PROFILING_ITERS,
                        "total_forward_passes": self.forward_count,
                    },
                    "layer_timings": {str(layer_id): timings for layer_id, timings in self.accumulated_layer_timings.items()},
                    "summary": {
                        "total_attn_ms": total_attn_ms,
                        "total_ffn_ms": total_ffn_ms,
                        "total_ms": grand_total_ms,
                        "attn_percent": attn_pct,
                        "ffn_percent": ffn_pct
                    }
                }
                json.dump(json_data, f, indent=2)

            print(f"Saved raw CUDA events data to {json_output_file}")

        except Exception as e:
            print(f"Error saving CUDA events profiling results: {e}")

        # Reset accumulated timings after saving
        self.accumulated_layer_timings = {}
        self.should_print_timings = False


class TimerContext:
    """Context manager for GPU event timing."""

    def __init__(self, layer_id: int, timer_type: str):
        self.layer_id = layer_id
        self.timer_type = timer_type
        self.start_event = None
        self.end_event = None

        if ENABLE_CUDA_EVENTS and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.start_event is not None:
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.end_event is not None:
            self.end_event.record()

        # We'll collect the timing data in the layer's forward method
        return False

    def get_elapsed_time(self):
        """Get elapsed time in milliseconds."""
        if self.start_event is None or self.end_event is None:
            return 0.0

        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


# Create the global profiling context
def create_profiling_context():
    """Create a profiling context if profiling is enabled.

    Returns:
        ProfilingContext or None if profiling is disabled
    """
    if ENABLE_NSYS or ENABLE_CUDA_EVENTS:
        return ProfilingContext()
    return None
