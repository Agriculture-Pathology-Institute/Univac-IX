import os
import sys
import pathlib
import numpy as np
from numba import njit, prange
import cupy as cp  # Allocates processing natively inside the Dual NVIDIA L4 GPUs

# 1. BIND REPOSITORY PATHS TO ENVIRONMENT
WORKSPACE_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(WORKSPACE_ROOT / "Mercury-Redstone-Logistics-main" / "src"))

try:
    import UNIVAC_logistics_node as univac_core
    LOCAL_LOGISTICS_AVAILABLE = True
except ImportError:
    LOCAL_LOGISTICS_AVAILABLE = False

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
    from cosmos_xenna import file_distribution
except ImportError:
    # Fail-safe abstract definitions for isolated testing environments
    pipelines_v1 = object
    file_distribution = None

class TurnkeyNodeTypes:
    AGRICULTURE = 0x0A  # Ingests crop, soil, and automated harvesting telemetry
    CONSTRUCTION = 0x0C # Ingests crane stress, physical loads, and hydraulic tracking


# --- HISTORICAL PARSING LAYER ---
class MercuryRedstoneHistoricalParser:
    """
    Historical data-parsing layer that ingests raw telemetry records 
    from Mercury-Redstone-Logistics source logs.
    """
    def __init__(self, target_log_dir: str = "./"):
        self.log_dir = pathlib.Path(target_log_dir)
        self.cpu_cores = os.cpu_count() or 64

    def ingest_raw_logistics_stream(self, file_name: str) -> list:
        target_file = self.log_dir / file_name
        if not target_file.exists():
            target_file = WORKSPACE_ROOT / "Mercury-Redstone-Logistics-main" / "docs" / "logistics.md"
        if not target_file.exists():
            # If no files exist, look for any salvage file in current folder
            target_file = WORKSPACE_ROOT / "salvaged_log.txt"

        if not target_file.exists():
            return []

        with open(target_file, "rb") as f:
            raw_binary_buffer = np.frombuffer(f.read(), dtype=np.uint8)

        if len(raw_binary_buffer) == 0:
            return []

        # Multi-core binary matrix carver strips zone flags
        cleansed_vectors = _parallel_binary_word_extractor(raw_binary_buffer)

        ingested_packets = []
        for idx, vector in enumerate(cleansed_vectors):
            ingested_packets.append({
                "id": f"MR-LOG-{idx:05d}",
                "node_type": TurnkeyNodeTypes.AGRICULTURE,
                "data": vector.tolist()
            })
        return ingested_packets


@njit(parallel=True, fastmath=True)
def _parallel_binary_word_extractor(raw_bytes: np.ndarray) -> np.ndarray:
    total_bytes = len(raw_bytes)
    total_rows = total_bytes // 4
    output_matrix = np.zeros((total_rows, 4), dtype=np.float32)

    for i in prange(total_rows):
        base_idx = i * 4
        output_matrix[i, 0] = float(raw_bytes[base_idx] & 0x3F)
        output_matrix[i, 1] = float(raw_bytes[base_idx + 1] & 0x3F)
        output_matrix[i, 2] = float(raw_bytes[base_idx + 2] & 0x3F)
        output_matrix[i, 3] = float(raw_bytes[base_idx + 3] & 0x3F)

    return output_matrix


# --- INTEGRATED COSMOS-XENNA PIPELINE STAGE ---
class TurnkeyIndustrialBridgeStage(pipelines_v1.Stage):
    """
    NVIDIA Cosmos-Xenna Pipeline Stage modeling historical Turnkey Automation Nodes.
    Natively auto-ingests Mercury-Redstone logistics archives on cluster setup.
    """
    
    @property
    def stage_batch_size(self) -> int:
        return 25000  # High-throughput batch size optimized for 1 TB RAM pool

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(
            gpus=2.0,                           # Consumes both available NVIDIA L4 GPUs
            cpus=float(os.cpu_count() or 64),   # Targets full 64-core thread architecture
            is_spmd=True                        # Activates Single Program Multiple Data coordination
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """
        Runs atomically on cluster startup. Initializes hardware links and
        automatically ingests local historical tracking archives into memory.
        """
        dist_params = worker_metadata.distributed_execution_params
        self.worker_id = dist_params.rank
        self.gpu_count = cp.cuda.runtime.getDeviceCount()
        
        print(f"\n[CLUSTER-STARTUP] [RANK {self.worker_id}] Initializing Turnkey Stage Core Hardware Link...")
        if LOCAL_LOGISTICS_AVAILABLE and hasattr(univac_core, 'initialize_hardware_link'):
            univac_core.initialize_hardware_link()

        # AUTOMATIC LOG INGESTION LAYER
        print(f"[CLUSTER-STARTUP] [RANK {self.worker_id}] Scanning for local Mercury-Redstone logistics logs...")
        self.parser = MercuryRedstoneHistoricalParser(target_log_dir="./")
        self.staged_historical_data = self.parser.ingest_raw_logistics_stream("salvaged_log.txt")
        
        print(f"[CLUSTER-STARTUP] [RANK {self.worker_id}] Auto-Ingest Finished. {len(self.staged_historical_data)} historical vectors cached in RAM.")

    def process_data(self, telemetry_samples: list) -> list:
        """
        Processes streaming runtime samples by combining them with auto-ingested 
        historical baselines, spreading tasks evenly across dual NVIDIA L4 GPUs.
        """
        # If no streaming input is provided, fallback to processing our auto-ingested historical cache
        active_batch = telemetry_samples if telemetry_samples else self.staged_historical_data
        if not active_batch:
            return []

        sensor_matrix = np.array([s.get("data") for s in active_batch], dtype=np.float32)
        node_type = active_batch[0].get("node_type", TurnkeyNodeTypes.AGRICULTURE)

        # 1. Distribute raw data validation over the multi-core CPU threads via Numba
        validated_matrix = _parallel_turnkey_preprocessor(sensor_matrix, node_type)

        # 2. Divide data arrays evenly to balance execution over both L4 GPUs
        midpoint = len(validated_matrix) // 2
        block_0 = validated_matrix[:midpoint]
        block_1 = validated_matrix[midpoint:]

        # Execute Block 0 on NVIDIA L4 GPU Device 0
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(block_0, dtype=cp.float32)
            cuda_out_0 = cp.asnumpy(cp.tanh(gpu_arr_0) * 100.0)

        # Execute Block 1 on NVIDIA L4 GPU Device 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(block_1, dtype=cp.float32)
            cuda_out_1 = cp.asnumpy(cp.tanh(gpu_arr_1) * 100.0)

        final_projections = np.concatenate((cuda_out_0, cuda_out_1), axis=0)

        # 3. Package output markers back into the queue for the Visio hazard monitor
        output_packets = []
        for i in range(len(final_projections)):
            load_factor = float(np.mean(final_projections[i]))
            status_flag = "Critical" if load_factor > 85.0 else "Nominal"
            
            output_packets.append({
                "node_id": active_batch[i].get("id"),
                "computed_load": load_factor,
                "status": status_flag
            })

        return output_packets


@njit(parallel=True, fastmath=True)
def _parallel_turnkey_preprocessor(data: np.ndarray, type_flag: int) -> np.ndarray:
    rows, cols = data.shape
    cleaned_output = np.zeros((rows, cols), dtype=np.float32)

    for i in prange(rows):
        for j in range(cols):
            val = data[i, j]
            if type_flag == TurnkeyNodeTypes.AGRICULTURE:
                cleaned_output[i, j] = val if (val >= 0.0 and val <= 100.0) else 50.0
            elif type_flag == TurnkeyNodeTypes.CONSTRUCTION:
                cleaned_output[i, j] = val * 1.01605 if val > 0.0 else 0.0
                
    return cleaned_output


# --- Pipeline Execution Verification Suite ---
if __name__ == "__main__":
    print("=== Simulating Cluster Startup & Auto-Ingestion Pipeline ===")
    
    # Generate mock log data if needed for test validation
    with open("salvaged_log.txt", "wb") as f_mock:
        f_mock.write(b"MERCURY_REDSTONE_RAW_BINARY_STREAM_DATA_PACKETS_FOR_TURNKEY_NODES")
        
    class MockWorkerMetadata:
        class DistParams:
            rank = 0
            world_size = 1
        distributed_execution_params = DistParams()

    # Instantiate and initialize the stage (Triggers automatic file ingestion)
    stage = TurnkeyIndustrialBridgeStage()
    stage.setup(MockWorkerMetadata())
    
    # Process the data
    results = stage.process_data([])
    print(f"\n✓ Processed {len(results)} records straight from the auto-ingested historical file buffer.")
