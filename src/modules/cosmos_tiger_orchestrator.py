# cosmos_tiger_orchestrator.py
import os
import pathlib
import sys
import numpy as np
import cupy as cp  # Drives native execution inside Dual NVIDIA L4 GPUs

# Inject the local module directory
sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
import tiger_nodes

try:
    import cosmos_xenna.pipelines.v1 as pipelines_v1
    from cosmos_xenna import file_distribution
except ImportError:
    # Fail-safe abstract definitions if local cluster testing outside environment
    pipelines_v1 = object
    file_distribution = None

class CosmosTigerVisPipelineStage(pipelines_v1.Stage):
    """
    Distributed NVIDIA Cosmos-Xenna stage wrapping multi-core Tiger nodes.
    Balances heavy mathematical array projections over dual NVIDIA L4 hardware slots.
    """
    @property
    def stage_batch_size(self) -> int:
        return 50000  # High-throughput batch size optimized for 1 TB RAM pool

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        # Request full 64-core multi-threading allocation paired with SPMD GPU execution
        return pipelines_v1.Resources(
            gpus=2.0,  # Explicitly consumes both NVIDIA L4 GPUs
            cpus=64.0, # Targets the full 64-core node allotment
            is_spmd=True # Activates synchronized single-program multi-data scaling
        )

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Initializes internal sub-nodes once per cluster worker process."""
        self.ingest_engine = tiger_nodes.TigerVisIngestionNode()
        self.analytics_engine = tiger_nodes.TigerVisAnalyticsNode()
        self.gpu_count = cp.cuda.runtime.getDeviceCount()
        print(f"[COSMOS-WORKER] Active with {self.gpu_count} targetable NVIDIA graphics engines.")

    def process_data(self, data_batches: list) -> list:
        """Processes telemetry elements by splitting the payload evenly between both GPUs."""
        # Convert incoming pipeline collection to a raw numpy array
        raw_matrix = np.array(data_batches, dtype=np.float32)
        
        # Step 1: Execute 64-core ingestion parsing matrix
        cleansed_matrix = self.ingest_engine.parallel_carve_stream(raw_matrix.astype(np.uint8))
        
        # Step 2: Split processed array data loads evenly across dual GPUs
        split_idx = len(cleansed_matrix) // 2
        segment_alpha = cleansed_matrix[:split_idx]
        segment_beta = cleansed_matrix[split_idx:]
        
        # Run segment 1 on NVIDIA L4 GPU 0
        with cp.cuda.Device(0):
            gpu_arr_0 = cp.array(segment_alpha, dtype=cp.float32)
            processed_alpha = cp.asnumpy(cp.exp(gpu_arr_0 / 100.0))
            
        # Run segment 2 on NVIDIA L4 GPU 1
        with cp.cuda.Device(1):
            gpu_arr_1 = cp.array(segment_beta, dtype=cp.float32)
            processed_beta = cp.asnumpy(cp.exp(gpu_arr_1 / 100.0))
            
        combined_gpu_results = np.concatenate((processed_alpha, processed_beta), axis=0)
        
        # Step 3: Run final high-speed analytics evaluations over results
        anomalies_mask = self.analytics_engine.run_matrix_eval(combined_gpu_results, baseline_limit=1.5)
        
        # Package results back into the pipeline queue stream
        return [{"metric": float(combined_gpu_results[i]), "alert": int(anomalies_mask[i])} for i in range(len(anomalies_mask))]
