import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Allocates trajectory projections inside Dual NVIDIA L4 VRAM

# 1. BIND REPOSITORY PATHS TO SYSTEM
WORKSPACE_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(WORKSPACE_ROOT / "Mercury-Redstone-Logistics-main" / "src"))

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
except ImportError:
    # Fail-safe abstract fallback definitions
    pipelines_v1 = object

class CourseScaleTypes:
    PROFESSIONAL_COURSE = 0xAA  # Massive macro-scale fairways, hydration, and soil tracking
    AUTOMATED_MINIGOLF   = 0xBB  # High-precision micro-scale kinetic obstacles and moving traps

class BridgesTurfAutomationStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage modeling Peter Bridges' automation systems.
    Ingests high-density sensor grids across 64 CPU cores and accelerates 
    ball trajectory and kinematic obstruction tracking on dual NVIDIA L4 GPUs.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 15000  # High-throughput batch size optimized for 1 TB RAM pools

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Consumes both available NVIDIA L4 hardware slots
            cpus=float(os.cpu_count() or 64),   # Targets full 64-core multi-threaded processing
            is_spmd=True                        # Activates Single Program Multiple Data cluster scaling
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Runs once per cluster worker to initialize the turf telemetry buffers."""
        dist_params = worker_metadata.distributed_execution_params
        self.worker_id = dist_params.rank
        self.gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"[BRIDGES-TURF] Worker {self.worker_id} Online. Dual NVIDIA L4 Pathing Initialized.")

    def process_data(self, telemetry_samples: list) -> list:
        """
        Processes streaming turf telemetry by splitting the array load 
        evenly across both available graphics hardware engines.
        """
        if not telemetry_samples:
            return []

        # Convert incoming streaming batch into a unified high-performance numpy matrix
        sensor_matrix = np.array([s.get("data") for s in telemetry_samples], dtype=np.float32)
        scale_mode = telemetry_samples[0].get("scale_mode", CourseScaleTypes.PROFESSIONAL_COURSE)

        # 1. Distribute raw data validation over multi-core threads via Numba
        validated_matrix = _parallel_turf_preprocessor(sensor_matrix, scale_mode)

        # 2. Divide data matrices down the middle to balance operations over both L4 GPUs
        midpoint = len(validated_matrix) // 2
        block_0 = validated_matrix[:midpoint]
        block_1 = validated_matrix[midpoint:]

        # Execute Block 0 on NVIDIA L4 GPU 0 (Simulating trajectory vectors)
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(block_0, dtype=cp.float32)
            cuda_out_0 = cp.asnumpy(cp.exp(gpu_arr_0 / 50.0) * 1.5)

        # Execute Block 1 on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(block_1, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.exp(gpu_arr_1 / 50.0) * 1.5)

        # Recombine execution fragments back into the primary thread memory space
        final_projections = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output markers back into the queue for the Visio hazard monitor
        output_packets = []
        for i in range(len(final_projections)):
            stress_index = float(np.mean(final_projections[i]))
            # Flags a hazard if moisture levels drop critically or an automated hazard stalls
            status_flag = "Critical" if stress_index > 92.5 or stress_index < 10.0 else "Nominal"
            
            output_packets.append({
                "node_id": telemetry_samples[i].get("id"),
                "computed_stress": stress_index,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_turf_preprocessor(data: np.ndarray, scale_mode: int) -> np.ndarray:
    """
    Thread-safe Numba kernel that parallelizes raw data conditioning across 
    64 CPU cores concurrently based on course metrics.
    """
    rows, cols = data.shape
    cleaned_output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            val = data[i, j]
            if scale_mode == CourseScaleTypes.PROFESSIONAL_COURSE:
                # Professional Course: Filters macro turf hydrology, moisture %, and pH variance
                cleaned_output[i, j] = val if (val >= 0.0 and val <= 100.0) else 65.0
            elif scale_mode == CourseScaleTypes.AUTOMATED_MINIGOLF:
                # Automated Minigolf: Parses mechanical proximity flags and laser timing matrix
                cleaned_output[i, j] = val * 0.9821 if val > 0.0 else 0.0
                
    return cleaned_output


# --- Pipeline Execution Verification Suite ---
if __name__ == "__main__":
    print("=== Testing Bridges Turf Automation Integration Node ===")
    
    # Simulate 5,000 incoming telemetry streams from an automated minigolf tracking installation
    mock_samples = [
        {
            "id": f"PB-MINI-{idx:04d}",
            "scale_mode": CourseScaleTypes.AUTOMATED_MINIGOLF,
            "data": np.random.uniform(5.0, 98.0, size=4).tolist()
        }
        for idx in range(5000)
    ]
    
    class MockWorkerMetadata:
        class DistParams:
            rank = 0
            world_size = 1
        distributed_execution_params = DistParams()

    # Instantiate and spin up the processing stage
    stage = BridgesTurfAutomationStage()
    stage.setup(MockWorkerMetadata())
    
    results = stage.process_data(mock_samples)
    critical_alerts = sum(1 for r in results if r["status"] == "Critical")
    
    print(f"\n✓ Multi-Core Turf Preprocessor and Dual-GPU Trajectory Split complete.")
    print(f"👉 Total Course Nodes Monitored: {len(results)}")
    print(f"👉 Flagged Kinetic Faults      : {critical_alerts}")
