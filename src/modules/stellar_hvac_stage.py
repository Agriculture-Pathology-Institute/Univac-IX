# stellar_hvac_stage.py
import os
import sys
import numpy as np
from numba import njit, prange
import cupy as cp

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class StellarHvacAtmosphericStage(pipelines_v1.Stage):
    """Processes thermodynamic fluid loops and atmospheric barometric data pools."""
    @property
    def stage_batch_size(self) -> int: return 25000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=2.0, cpus=64.0, is_spmd=True)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print("[STELLAR-HVAC] Life support atmospheric scrubbers online.")

    def process_data(self, samples: list) -> list:
        if not samples: return []
        matrix = np.array([s.get("data") for s in samples], dtype=np.float32)
        validated = _parallel_hvac_preprocessor(matrix)
        
        mid = len(validated) // 2
        with cp.cuda.Device(0):
            out_0 = cp.asnumpy(cp.array(validated[:mid]) * 1.05)
        with cp.cuda.Device(1):
            out_1 = cp.asnumpy(cp.array(validated[mid:]) * 1.05)
            
        projections = np.concatenate((out_0, out_1), axis=0)
        return [{"node_id": samples[i].get("id"), "computed_hvac": float(np.mean(projections[i])), "status": "Critical" if float(np.mean(projections[i])) > 90.0 or float(np.mean(projections[i])) < 15.0 else "Nominal"} for i in range(len(projections))]

@njit(parallel=True, fastmath=True)
def _parallel_hvac_preprocessor(data: np.ndarray) -> np.ndarray:
    r, c = data.shape
    out = np.zeros((r, c), dtype=np.float32)
    for i in prange(r):
        for j in range(c):
            out[i, j] = data[i, j] if data[i, j] >= 0.0 else 0.0
    return out
