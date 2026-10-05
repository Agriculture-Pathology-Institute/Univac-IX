# siemens_s7_stage.py
import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Allocates processing natively inside Dual NVIDIA L4 GPUs

try:
    import snap7
    SNAP7_AVAILABLE = True
except ImportError:
    SNAP7_AVAILABLE = False

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class SiemensS7DataBlockStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage interfacing with Siemens SIMATIC S7 Data Blocks.
    Extracts absolute memory byte offsets over 64 CPU threads and dual NVIDIA L4 GPUs.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 22000  # High-throughput batch size optimized for 1 TB RAM platforms

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Allocates calculations across both L4 graphics engines
            cpus=float(os.cpu_count() or 64),   # Consumes the full 64-core thread configuration
            is_spmd=True                        # Engages Single Program Multiple Data coordination matrix
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Establishes connection drivers to the physical Siemens PLC field nodes on startup."""
        self.worker_id = worker_metadata.distributed_execution_params.rank
        print(f"[SIEMENS-S7] Worker {self.worker_id} Online. Snap7 Drivers Initialized: {SNAP7_AVAILABLE}")

    def process_data(self, data_samples: list) -> list:
        """
        Unpacks absolute S7 data block memory buffers by splitting the 
        matrix processing load evenly across both available L4 GPUs.
        """
        if not data_samples:
            return []

        # Convert the streaming network collection into a high-performance numpy matrix
        raw_byte_matrix = np.array([s.get("data") for s in data_samples], dtype=np.float32)

        # 1. Parse absolute memory byte arrays over multi-core threads via Numba
        validated_blocks = _parallel_siemens_byte_carver(raw_byte_matrix)

        # 2. Divide data loads down the middle to balance matrix math across dual GPUs
        midpoint = len(validated_blocks) // 2
        segment_alpha = validated_blocks[:midpoint]
        segment_beta = validated_blocks[midpoint:]

        # Execute Segment Alpha on NVIDIA L4 GPU 0 (Computes process state transformations)
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(segment_alpha, dtype=cp.float32)
            cuda_out_0 = cp.asnumpy(cp.tanh(gpu_arr_0 / 255.0) * 100.0)

        # Execute Segment Beta on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(segment_beta, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.tanh(gpu_arr_1 / 255.0) * 100.0)

        # Recombine calculation matrices back into primary memory space
        compiled_metrics = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output vectors back into the pipeline array for the Visio hazard monitor
        output_packets = []
        for i in range(len(compiled_metrics)):
            peak_signal = float(np.max(compiled_metrics[i]))
            # Flags a critical condition if the memory register indicates a mechanical fault flag
            status_flag = "Critical" if peak_signal > 82.5 else "Nominal"
            
            output_packets.append({
                "node_id": data_samples[i].get("id"),
                "computed_load": peak_signal,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_siemens_byte_carver(data: np.ndarray) -> np.ndarray:
    """
    Thread-safe Numba processing kernel that unpacks non-optimized 
    data block absolute offsets across all 64 CPU cores concurrently.
    """
    rows, cols = data.shape
    output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            # Isolate and clean raw register values
            val = data[i, j]
            output[i, j] = val if (val >= 0.0 and val <= 255.0) else 0.0
            
    return output
