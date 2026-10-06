# stellar_range_stage.py
import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Executes high-speed spatial distance tracking natively inside VRAM

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    # Fail-safe abstract definitions for isolated cluster environment runtimes
    pipelines_v1 = object

class StellarRangeOrchestratorStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage modeling the Stellar RANGE Orchestrator Node.
    Processes high-volume spatial boundaries, multi-vector perimeters, and signal radius 
    matrices across 64 CPU cores and dual NVIDIA L4 GPUs.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 31000  # High-throughput batch profile optimized for 1 TB RAM platforms

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Allocates calculations across both L4 graphics engines
            cpus=float(os.cpu_count() or 64),   # Consumes the full 64-core thread configuration
            is_spmd=True                        # Engages Single Program Multiple Data coordination matrix
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Runs once per cluster worker to initialize the spatial boundary tracking structures."""
        self.worker_id = worker_metadata.distributed_execution_params.rank
        print(f"[STELLAR-RANGE] Worker {self.worker_id} Online. Spatial Range Orchestrator Core Active.")

    def process_data(self, range_samples: list) -> list:
        """
        Ingests spatial tracking signals and tracks boundary conditions.
        Splits execution blocks evenly between both available NVIDIA graphics engines.
        """
        if not range_samples:
            return []

        # Convert the streaming telemetry collection into a high-performance numpy matrix
        # Columns correspond to: [Current Latitude Delta, Current Longitude Delta, Target Vector Radius, Signal Degradation Ratio]
        raw_range_matrix = np.array([s.get("data") for s in range_samples], dtype=np.float32)

        # 1. Distribute raw matrix processing over 64 CPU threads via a compiled Numba kernel
        validated_ranges = _parallel_range_vector_preprocessor(raw_range_matrix)

        # 2. Divide data loads down the middle to balance matrix math across dual GPUs
        midpoint = len(validated_ranges) // 2
        segment_alpha = validated_ranges[:midpoint]
        segment_beta = validated_ranges[midpoint:]

        # Execute Segment Alpha on NVIDIA L4 GPU 0 (Computes dynamic boundary crossings)
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(segment_alpha, dtype=cp.float32)
            # Apply Euclidean range mapping formulas natively inside VRAM
            cuda_out_0 = cp.asnumpy(cp.sqrt(gpu_arr_0[:, 0]**2 + gpu_arr_0[:, 1]**2) - gpu_arr_0[:, 2])

        # Execute Segment Beta on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(segment_beta, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.sqrt(gpu_arr_1[:, 0]**2 + gpu_arr_1[:, 1]**2) - gpu_arr_1[:, 2])

        # Recombine calculation matrices back into primary memory space
        calculated_boundary_offsets = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output vectors back into the pipeline array for the Visio hazard monitor
        output_packets = []
        for i in range(len(calculated_boundary_offsets)):
            offset_meters = float(calculated_boundary_offsets[i])
            # Flags a critical condition if a target breaches or drifts outside the assigned containment perimeter
            status_flag = "Critical" if offset_meters > 50.0 or offset_meters < -500.0 else "Nominal"
            
            output_packets.append({
                "node_id": range_samples[i].get("id"),
                "computed_offset": offset_meters,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_range_vector_preprocessor(data: np.ndarray) -> np.ndarray:
    """
    Thread-safe Numba processing kernel that parallelizes spatial coordination calculations
    across all 64 CPU cores concurrently to prevent system telemetry locks.
    """
    rows, cols = data.shape
    output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            val = data[i, j]
            # Eliminate invalid anomalies from range tracking tools
            output[i, j] = val if (val >= -10000.0 and val <= 10000.0) else 0.0
            
    return output
