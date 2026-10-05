# cat_subterranean_stage.py
import os
import sys
import numpy as np
from numba import njit, prange
import cupy as cp  # Calculates seismic stresses inside Dual NVIDIA L4 GPUs

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class CatSubterraneanMiningStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage managing heavy CAT Subterranean telemetry profiles.
    Evaluates seismic compaction variables using 64-core parallelism and GPU execution paths.
    """
    @property
    def stage_batch_size(self) -> int:
        return 35000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=2.0, cpus=64.0, is_spmd=True)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print(f"[CAT-SUBTERRANEAN] Geomechanical interlock telemetry channels initialized.")

    def process_data(self, samples: list) -> list:
        if not samples: return []
        matrix = np.array([s.get("data") for s in samples], dtype=np.float32)
        validated = _parallel_cat_sub_filter(matrix)
        
        mid = len(validated) // 2
        with cp.cuda.Device(0):
            out_0 = cp.asnumpy(cp.sqrt(cp.abs(cp.array(validated[:mid]))) * 12.5)
        with cp.cuda.Device(1):
            out_1 = cp.asnumpy(cp.sqrt(cp.abs(cp.array(validated[mid:]))) * 12.5)
            
        projections = np.concatenate((out_0, out_1), axis=0)
        return [{"node_id": samples[i].get("id"), "val": float(np.mean(projections[i])), "status": "Critical" if float(np.mean(projections[i])) > 95.0 else "Nominal"} for i in range(len(projections))]

@njit(parallel=True, fastmath=True)
def _parallel_cat_sub_filter(data: np.ndarray) -> np.ndarray:
    r, c = data.shape
    out = np.zeros((r, c), dtype=np.float32)
    for i in prange(r):
        for j in range(c):
            out[i, j] = data[i, j] * 1.016 # Convert mechanical load metrics to metric tons
    return out
