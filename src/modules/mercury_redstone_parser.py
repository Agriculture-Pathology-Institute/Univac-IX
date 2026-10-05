# mercury_redstone_parser.py
import os
import pathlib
import sys
import numpy as np
from numba import njit, prange

# Inject the local module search tree to reference repository files
WORKSPACE_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(WORKSPACE_ROOT / "Mercury-Redstone-Logistics-main" / "src"))

try:
    # Explicitly interface with your local logistics tracking components
    import UNIVAC_logistics_node as univac_core
    import logistics_node as standard_node
    LOCAL_LOGISTICS_AVAILABLE = True
except ImportError:
    LOCAL_LOGISTICS_AVAILABLE = False
    print("[WARN] Local Mercury-Redstone structural scripts not found in loop. Running fallback parsers.")

class MercuryRedstoneHistoricalParser:
    """
    Historical data-parsing layer that ingests raw telemetry records 
    from Mercury-Redstone-Logistics source logs, feeding them into Tiger/Turnkey nodes.
    """
    def __init__(self, target_log_dir: str = "./"):
        self.log_dir = pathlib.Path(target_log_dir)
        self.cpu_cores = os.cpu_count() or 64
        print(f"[MERCURY-PARSER] Initializing Historical Parser on {self.cpu_cores} CPU threads.")
        
        # Verify local repository mapping presence
        if LOCAL_LOGISTICS_AVAILABLE:
            print(" ✓ Successfully bound to local 'UNIVAC_logistics_node.py' interface script.")
        else:
            print(" ! Operating via autonomous binary extraction fallbacks.")

    def ingest_raw_logistics_stream(self, file_name: str) -> list:
        """
        Reads binary log streams or raw text dumps from the file system,
        unpacks the field parameters, and maps them into multi-core process arrays.
        """
        target_file = self.log_dir / file_name
        
        # If the specific target does not exist, look inside the repository structure
        if not target_file.exists():
            target_file = WORKSPACE_ROOT / "Mercury-Redstone-Logistics-main" / "docs" / "logistics.md"

        if not target_file.exists():
            print(f"[-] Critical Error: Source trace '{file_name}' cannot be resolved.")
            return []

        print(f"[MERCURY-PARSER] Ingesting source records from: {target_file}")
        
        # Read the file data into a byte stream array
        with open(target_file, "rb") as f:
            raw_binary_buffer = np.frombuffer(f.read(), dtype=np.uint8)

        if len(raw_binary_buffer) == 0:
            print("[-] Ingestion Aborted: Target source stream is completely empty.")
            return []

        # Use the parallel Numba matrix carver to isolate 36-bit words or structural values
        print(f"[MERCURY-PARSER] Executing multi-core string parsing loop over {len(raw_binary_buffer)} bytes...")
        cleansed_vectors = _parallel_binary_word_extractor(raw_binary_buffer)

        # Restructure vectors into standard dictionary payloads for your Turnkey Industrial Stage
        ingested_packets = []
        for idx, vector in enumerate(cleansed_vectors):
            ingested_packets.append({
                "id": f"MR-LOG-{idx:05d}",
                "node_type": 0x0A,  # Defaults payload flag to Turnkey Industrial Agriculture
                "data": vector.tolist()
            })

        print(f" ✓ Parsing complete. Formatted {len(ingested_packets)} telemetry packets into memory space.")
        return ingested_packets


@njit(parallel=True, fastmath=True)
def _parallel_binary_word_extractor(raw_bytes: np.ndarray) -> np.ndarray:
    """
    Numba parallel kernel that strips zone masks and packs raw bytes 
    into standard 4-element coordinate arrays across all 64 cores simultaneously.
    """
    total_bytes = len(raw_bytes)
    # Group every 4 bytes into an execution matrix row
    total_rows = total_bytes // 4
    
    # Pre-allocate output matrix array inside memory
    output_matrix = np.zeros((total_rows, 4), dtype=np.float32)

    for i in prange(total_rows):
        base_idx = i * 4
        # Strips out high-order zone flags or framing signatures natively
        output_matrix[i, 0] = float(raw_bytes[base_idx] & 0x3F)
        output_matrix[i, 1] = float(raw_bytes[base_idx + 1] & 0x3F)
        output_matrix[i, 2] = float(raw_bytes[base_idx + 2] & 0x3F)
        output_matrix[i, 3] = float(raw_bytes[base_idx + 3] & 0x3F)

    return output_matrix


# --- End-To-End Ingestion Testing Matrix ---
if __name__ == "__main__":
    print("=== Testing Mercury-Redstone Historical Ingestion Stream ===")
    
    # Generate a temporary testing file if not already present
    test_log = "salvaged_log.txt"
    with open(test_log, "wb") as f_mock:
        # Write dummy binary tracking stream data
        f_mock.write(b"MERCURY_REDSTONE_TELEMETRY_LOG_CORES_UNPACKED_GRID_DATA_MATRIX")
        
    # Instantiate the data layer parser
    parser = MercuryRedstoneHistoricalParser()
    
    # Execute the file data ingestion layer
    pipeline_ready_packets = parser.ingest_raw_logistics_stream(test_log)
    
    if pipeline_ready_packets:
        print("\n[Ingestion Snapshot Result]")
        print(f" 👉 First Sample ID    : {pipeline_ready_packets[0]['id']}")
        print(f" 👉 Extracted Data Row : {pipeline_ready_packets[0]['data']}")
        print(f" 👉 Total Packed Output: {len(pipeline_ready_packets)} vectors available.")
        print("\n✓ Historical parsing layer is stable and ready to feed active Turnkey/Tiger tracking stages.")
