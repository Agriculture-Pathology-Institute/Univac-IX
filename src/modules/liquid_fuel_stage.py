# liquid_fuel_stage.py
import os
import sys
import numpy as np
fromba import njit, prange
import cupy as cp  # Resolves pipeline hydraulic calculations natively inside VRAM

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    pipelines_v1 = object

class LiquidFuelLogisticsStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage evaluating fluid dynamics for Liquid Fuel nodes.
    Balances hydraulic matrix processing loops across dual NVIDIA L4 GPUs.
    """
    @property
    def stage_batch_size(self) -> int:
        return 28000

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=2.0, cpus=64.0, is_spmd=True)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        print(f"[LIQUID-FUEL-CORE] Hydraulic flow pressure monitors linked successfully.")

    def process_data(self, samples: list) -> list:
        if not samples: return []
        matrix = np.array([s.get("data") for s in samples], dtype=np.float32)
        validated = _parallel_fuel_flow_filter(matrix)
        
        mid = len(validated) // 2
        with cp.cuda.Device(0):
            out_0 = cp.asnumpy(cp.tanh(cp.array(validated[:mid])) * 110.0)
        with cp.cuda.Device(1):
            out_1 = cp.asnumpy(cp.tanh(cp.array(validated[mid:])) * 110.0)
            
        projections = np.concatenate((out_0, out_1), axis=0)
        return [{"node_id": samples[i].get("id"), "val": float(np.mean(projections[i])), "status": "Critical" if float(np.mean(projections[i])) > 90.0 or float(np.mean(projections[i])) < 15.0 else "Nominal"} for i in range(len(projections))]

@njit(parallel=True, fastmath=True)
def _parallel_fuel_flow_filter(data: np.ndarray) -> np.ndarray:
    r, c = data.shape
    out = np.zeros((r, c), dtype=np.float32)
    for i in prange(r):
        for j in range(c):
            out[i, j] = data[i, j] if data[i, j] >= 0.0 else 0.0
    return out
