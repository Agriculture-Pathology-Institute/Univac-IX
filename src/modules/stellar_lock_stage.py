# stellar_lock_stage.py
import os
import sys
import numpy as np
from numba import njit, prange
import cupy as cp

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class StellarLockSubsystemStage(pipelines_v1.Stage):
    """Processes cryptographic perimeter lock matrices over 64 CPU threads and dual GPUs."""
    @property
    def stage_batch_size(self) -> int: return 30000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=2.0, cpus=64.0, is_spmd=True)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print("[STELLAR-LOCK] Encrypted boundary interlock arrays online.")

    def process_data(self, samples: list) -> list:
        if not samples: return []
        matrix = np.array([s.get("data") for s in samples], dtype=np.float32)
        validated = _parallel_lock_preprocessor(matrix)
        
        mid = len(validated) // 2
        with cp.cuda.Device(0):
            out_0 = cp.asnumpy(cp.tanh(cp.array(validated[:mid])) * 100.0)
        with cp.cuda.Device(1):
            out_1 = cp.asnumpy(cp.tanh(cp.array(validated[mid:])) * 100.0)
            
        projections = np.concatenate((out_0, out_1), axis=0)
        return [{"node_id": samples[i].get("id"), "computed_lock": float(np.mean(projections[i])), "status": "Critical" if float(np.mean(projections[i])) > 85.0 else "Nominal"} for i in range(len(projections))]

@njit(parallel=True, fastmath=True)
def _parallel_lock_preprocessor(data: np.ndarray) -> np.ndarray:
    r, c = data.shape
    out = np.zeros((r, c), dtype=np.float32)
    for i in prange(r):
        for j in range(c):
            out[i, j] = data[i, j] if data[i, j] >= 0.0 else 0.0
    return out
