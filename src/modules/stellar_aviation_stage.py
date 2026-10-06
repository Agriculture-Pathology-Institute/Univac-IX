# stellar_aviation_stage.py
import os
import sys
import numpy as np
from numba import njit, prange
import cupy as cp

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class StellarAviationTransceiverStage(pipelines_v1.Stage):
    """Processes high-velocity radar vectors and air corridor trajectories."""
    @property
    def stage_batch_size(self) -> int: return 38000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=2.0, cpus=64.0, is_spmd=True)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print("[STELLAR-AVIATION] Aerospace flight tracking transceivers initialized.")

    def process_data(self, samples: list) -> list:
        if not samples: return []
        matrix = np.array([s.get("data") for s in samples], dtype=np.float32)
        validated = _parallel_aviation_preprocessor(matrix)
        
        mid = len(validated) // 2
        with cp.cuda.Device(0):
            out_0 = cp.asnumpy(cp.log2(cp.abs(cp.array(validated[:mid])) + 1.0) * 12.0)
        with cp.cuda.Device(1):
            out_1 = cp.asnumpy(cp.log2(cp.abs(cp.array(validated[mid:])) + 1.0) * 12.0)
            
        projections = np.concatenate((out_0, out_1), axis=0)
        return [{"node_id": samples[i].get("id"), "computed_aviation": float(np.mean(projections[i])), "status": "Critical" if float(np.mean(projections[i])) > 88.0 else "Nominal"} for i in range(len(projections))]

@njit(parallel=True, fastmath=True)
def _parallel_aviation_preprocessor(data: np.ndarray) -> np.ndarray:
    r, c = data.shape
    out = np.zeros((r, c), dtype=np.float32)
    for i in prange(r):
        for j in range(c):
            out[i, j] = data[i, j] if data[i, j] >= 0.0 else 0.0
    return out
