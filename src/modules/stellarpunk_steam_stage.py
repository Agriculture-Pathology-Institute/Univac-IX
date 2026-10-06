# stellarpunk_steam_stage.py
import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Executes high-speed kinetic optimizations natively inside VRAM

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    # Fail-safe abstract definitions for isolated cluster environment runtimes
    pipelines_v1 = object

class StellarPunkSteamStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage modeling a Stellar-Punk Steam Node.
    Processes high-volume thermodynamic simulations, mechanical valve layouts,
    and kinetic trajectory arrays across 64 CPU cores and dual NVIDIA L4 GPUs.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 32000  # High-throughput batch profile optimized for 1 TB RAM platforms

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Allocates calculations across both L4 graphics engines
            cpus=float(os.cpu_count() or 64),   # Consumes the full 64-core thread configuration
            is_spmd=True                        # Engages Single Program Multiple Data coordination matrix
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Runs once per cluster worker to initialize the thermodynamic valve registers."""
        self.worker_id = worker_metadata.distributed_execution_params.rank
        print(f"[STELLAR-PUNK-STEAM] Worker {self.worker_id} Online. Kinetic Pressure Matrix Activated.")

    def process_data(self, kinetic_samples: list) -> list:
        """
        Ingests raw thermodynamic input logs and calculates valve pressure limits.
        Splits execution blocks evenly between both available NVIDIA graphics engines.
        """
        if not kinetic_samples:
            return []

        # Convert the streaming network matrix into a high-performance numpy array
        # Columns correspond to: [PSI Pressure, Temperature Celsius, Valve Flow Velocity, Structural Stress Ratio]
        raw_kinetic_matrix = np.array([s.get("data") for s in kinetic_samples], dtype=np.float32)

        # 1. Distribute raw telemetry processing over 64 CPU threads via Numba
        validated_metrics = _parallel_steam_preprocessor(raw_kinetic_matrix)

        # 2. Divide data loads down the middle to balance matrix math across dual GPUs
        midpoint = len(validated_metrics) // 2
        segment_alpha = validated_metrics[:midpoint]
        segment_beta = validated_metrics[midpoint:]

        # Execute Segment Alpha on NVIDIA L4 GPU 0 (Computes dynamic thermal expansions)
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(segment_alpha, dtype=cp.float32)
            # Apply mechanical strain formulas natively inside VRAM
            cuda_out_0 = cp.asnumpy(cp.tanh(gpu_arr_0[:, 0] / 100.0) * 85.0 + gpu_arr_0[:, 1] * 0.15)

        # Execute Segment Beta on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(segment_beta, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.tanh(gpu_arr_1[:, 0] / 100.0) * 85.0 + gpu_arr_1[:, 1] * 0.15)

        # Recombine calculation matrices back into primary memory space
        calculated_pressures = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output vectors back into the pipeline array for the Visio hazard monitor
        output_packets = []
        for i in range(len(calculated_pressures)):
            peak_psi = float(calculated_pressures[i])
            # Flags a critical condition if a boiler or pipeline exceeds safe physical load margins
            status_flag = "Critical" if peak_psi > 90.0 or peak_psi < 12.0 else "Nominal"
            
            output_packets.append({
                "node_id": kinetic_samples[i].get("id"),
                "computed_psi": peak_psi,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_steam_preprocessor(data: np.ndarray) -> np.ndarray:
    """
    Thread-safe Numba processing kernel that parallelizes mechanical data calculations
    across all 64 CPU cores concurrently to prevent system telemetry locks.
    """
    rows, cols = data.shape
    output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            val = data[i, j]
            # Eliminate unphysical negative anomalies from thermal sensors
            output[i, j] = val if val >= 0.0 else 0.0
            
    return output
