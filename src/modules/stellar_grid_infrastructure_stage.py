# stellar_grid_infrastructure_stage.py
import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Executes structural strain optimizations natively inside VRAM

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    # Fail-safe abstract definitions for isolated cluster environment runtimes
    pipelines_v1 = object

class StructuralScales:
    TERRESTRIAL_BUILDING   = 0x1A  # Standard foundation loading and floor-by-floor structural stresses
    ORBITAL_SPACE_STATION  = 0x2B  # High-vacuum skin pressure, hull breaches, and atmospheric containment
    COLONIAL_HABITAT       = 0x3C  # Controlled dome pressure, shielding integrity, and airlock loops
    PLANETARY_GRID         = 0x4D  # Global tectonic baselines and continental tectonic plates matrices

class StellarGridInfrastructureStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage modeling the Stellar GRiD Structural Infrastructure Node.
    Processes high-volume atmospheric pressures, structural load balances, and habitability 
    matrices across 64 CPU cores and dual NVIDIA L4 GPUs.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 34000  # High-throughput batch profile optimized for 1 TB RAM platforms

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Allocates calculations across both L4 graphics engines
            cpus=float(os.cpu_count() or 64),   # Consumes the full 64-core thread configuration
            is_spmd=True                        # Engages Single Program Multiple Data coordination matrix
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Runs once per cluster worker to initialize the habitability tracking structures."""
        self.worker_id = worker_metadata.distributed_execution_params.rank
        print(f"[STELLAR-GRID] Worker {self.worker_id} Online. Structural Infrastructure Core Active.")

    def process_data(self, grid_samples: list) -> list:
        """
        Ingests environmental structural signals and tracks layout conditions.
        Splits execution blocks evenly between both available NVIDIA graphics engines.
        """
        if not grid_samples:
            return []

        # Convert the streaming telemetry collection into a high-performance numpy matrix
        # Columns correspond to: [Internal Pressure PSI, Mechanical Structural Strain, Radiation Shielding %, Delta-T Thermal Flux]
        raw_structural_matrix = np.array([s.get("data") for s in grid_samples], dtype=np.float32)
        structural_scale = grid_samples.get("structural_scale", StructuralScales.COLONIAL_HABITAT)

        # 1. Distribute raw grid validation over 64 CPU threads via a compiled Numba kernel
        validated_structures = _parallel_structural_grid_preprocessor(raw_structural_matrix, structural_scale)

        # 2. Divide data loads down the middle to balance matrix math across dual GPUs
        midpoint = len(validated_structures) // 2
        segment_alpha = validated_structures[:midpoint]
        segment_beta = validated_structures[midpoint:]

        # Execute Segment Alpha on NVIDIA L4 GPU 0 (Computes dynamic safety factors)
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(segment_alpha, dtype=cp.float32)
            # Apply structural integrity formulas natively inside VRAM
            cuda_out_0 = cp.asnumpy(cp.tanh(gpu_arr_0[:, 1] * 1.5 + gpu_arr_0[:, 3] * 0.05) * 100.0)

        # Execute Segment Beta on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(segment_beta, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.tanh(gpu_arr_1[:, 1] * 1.5 + gpu_arr_1[:, 3] * 0.05) * 100.0)

        # Recombine calculation matrices back into primary memory space
        calculated_structural_stresses = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output vectors back into the pipeline array for the Visio hazard monitor
        output_packets = []
        for i in range(len(calculated_structural_stresses)):
            stress_factor = float(calculated_structural_stresses[i])
            # Flags a critical condition if hull strain or structural loads exceed safe containment bounds
            status_flag = "Critical" if stress_factor > 86.5 or stress_factor < 5.0 else "Nominal"
            
            output_packets.append({
                "node_id": grid_samples[i].get("id"),
                "computed_stress": stress_factor,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_structural_grid_preprocessor(data: np.ndarray, structural_scale: int) -> np.ndarray:
    """
    Thread-safe Numba processing kernel that parallelizes architectural layout calculations
    across all 64 CPU cores concurrently to prevent system telemetry locks.
    """
    rows, cols = data.shape
    output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            val = data[i, j]
            if structural_scale == StructuralScales.ORBITAL_SPACE_STATION or structural_scale == StructuralScales.COLONIAL_HABITAT:
                # Retains absolute vacuum measurements, clipping negative sensor loops
                output[i, j] = val if val >= 0.0 else 0.0
            else:
                # Standard terrestrial scaling rules
                output[i, j] = val if (val >= -100.0 and val <= 500.0) else 20.0
            
    return output
