import os
import pathlib
import sys

# 1. BIND REPOSITORY PATHS TO ENV
# Ensures your local modules can be discovered by the distributed orchestrator
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))

try:
    # Safely load your local Mercury-Redstone structural logic files
    import logistics_node as legacy_node
    import UNIVAC_logistics_node as univac_core
except ImportError:
    # Fallback placeholders for isolated cluster worker runtimes
    legacy_node = None
    univac_core = None

# Import the NVIDIA Cosmos-Xenna pipeline engine architecture
import cosmos_xenna.pipelines.v1 as pipelines_v1
from cosmos_xenna import file_distribution

class CosmosMercuryRedstoneAdapterStage(pipelines_v1.Stage):
    """
    Adapter Node bridging legacy UNIVAC Mercury-Redstone telemetry processing
    with NVIDIA Cosmos-Xenna distributed inference and world tracking.
    """
    
    @property
    def stage_batch_size(self) -> int:
        # Prevent VRAM Out-Of-Memory (OOM) failures on cluster nodes
        return 10 

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        # Force integration with multi-core CPUs and NVIDIA GPU infrastructure
        # Turning on is_spmd allows distributed tensor execution across worker groups
        return pipelines_v1.Resources(
            gpus=1.0, 
            cpus=float(os.cpu_count() or 4),
            is_spmd=True 
        )

    @property
    def download_requests(self) -> list[file_distribution.DownloadRequest]:
        """Ensures tracking matrices are pre-downloaded using P2P artifact distribution."""
        return [
            file_distribution.DownloadRequest(
                value=file_distribution.PrefixDownloadRequest(
                    profile_name="mercury-telemetry",
                    uri="trajectories/redstone_cores/",
                    destination=pathlib.Path("/tmp/redstone_cores/")
                )
            )
        ]

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Runs once per cluster worker to initialize the hardware interfaces."""
        dist_params = worker_metadata.distributed_execution_params
        print(f"[RANK {dist_params.rank}/{dist_params.world_size}] Cosmos-Xenna Adapter Online.")
        
        # Initialize your legacy UNIVAC node structures if present
        if univac_core and hasattr(univac_core, 'initialize_hardware_link'):
            univac_core.initialize_hardware_link()

    def process_data(self, samples: list) -> list:
        """
        Processes streaming batches of tracking tokens. Translates legacy 
        UNIVAC data layers into unified Physical AI parameters.
        """
        processed_samples = []
        
        for sample in samples:
            # 1. Extract raw logistical properties from sample
            raw_payload = getattr(sample, 'payload', sample)
            
            # 2. Map legacy matrix parameters into the pipeline
            # (Simulates stripping zone headers from 36-bit inputs)
            clean_vector = [byte & 0x0F for byte in raw_payload] if isinstance(raw_payload, list) else raw_payload
            
            # 3. Stream updated vector directly into the Cosmos tracking space
            sample.processed_telemetry = clean_vector
            sample.origin_system = "UNIVAC-MERCURY-REDSTONE"
            
            processed_samples.append(sample)
            
        return processed_samples

# --- Verification & Pipeline Initialization Block ---
def run_integrated_pipeline(input_logistics_data: list):
    """Assembles the stages and launches execution across the Ray cluster cluster."""
    print("Assembling NVIDIA Cosmos-Xenna Pipeline Platform...")
    
    pipeline_spec = pipelines_v1.PipelineSpec(
        input_data=input_logistics_data,
        stages=[
            # Registers the wrapped Mercury-Redstone processing node as Stage 0
            pipelines_v1.StageSpec(
                CosmosMercuryRedstoneAdapterStage(),
                num_workers_per_node=2
            )
        ]
    )
    
    # Fire up the non-blocking execution loop
    pipelines_v1.run_pipeline(pipeline_spec)
    print("Pipeline run completed successfully.")
