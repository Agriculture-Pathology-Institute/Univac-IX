import os
import csv
import time
import sys

def ring_terminal_bell(count: int = 3):
    """Broadcasts standard hardware ASCII audio bells straight to the terminal console."""
    for _ in range(count):
        sys.stdout.write('\a')
        sys.stdout.flush()
        time.sleep(0.15)

def parse_visio_csv(file_path: str) -> dict:
    """Reads the current system layout tracking configurations from the Visio CSV file."""
    node_states = {}
    if not os.path.exists(file_path):
        return node_states

    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            # Flexible reading that ignores trailing line spaces
            reader = csv.DictReader(f)
            for row in reader:
                # Standardize tracking indices based on our Visio schema
                node_name = row.get("Node Name", "Unknown_Node")
                status = row.get("Status Indicator", "Nominal").strip()
                hardware = row.get("Hardware Mapping", "Unknown_HW")
                load = row.get("Resource Load (%)", "0.0")
                
                node_states[node_name] = {
                    "status": status,
                    "hardware": hardware,
                    "load": load
                }
    except Exception as e:
        print(f"\n[WARN] Read collision encountered during file update poll: {str(e)}")
    return node_states

def monitor_visio_hazards(target_csv: str = "visio_mapping.csv", poll_interval_seconds: float = 1.0):
    """
    Continuous runtime watch-loop parsing the target mapping document.
    Triggers visual alerts and hardware audio cues upon finding critical state flags.
    """
    print("=========================================================")
    print(f"🚀 UNIVAC-IX: Real-Time Visio Hazard Monitor Online")
    print(f"📡 Target Resource  : {os.path.abspath(target_csv)}")
    print(f"⏱  Polling Frequency: Every {poll_interval_seconds} seconds")
    print("=========================================================")
    print("[SYSTEM] Monitoring file stream... Press Ctrl+C to terminate.")

    # Establish an initial tracking frame baseline
    previous_states = parse_visio_csv(target_csv)
    last_modified_time = 0

    try:
        while True:
            if os.path.exists(target_csv):
                current_modified_time = os.path.getmtime(target_csv)
                
                # Only re-parse if the file modification timestamp updates
                if current_modified_time != last_modified_time:
                    last_modified_time = current_modified_time
                    current_states = parse_visio_csv(target_csv)
                    
                    # Evaluate status across all active Tiger Node targets
                    for node_name, info in current_states.items():
                        status = info["status"]
                        prev_info = previous_states.get(node_name, {})
                        prev_status = prev_info.get("status", "Unknown")

                        # Event Trigger A: A node transforms into a CRITICAL threat landscape
                        if status.upper() == "CRITICAL":
                            if prev_status.upper() != "CRITICAL":
                                print(f"\n🚨 [CRITICAL HAZARD DETECTED] 🚨")
                                print(f" 👉 Node Variant: {node_name}")
                                print(f" 👉 Hardware    : {info['hardware']}")
                                print(f" 👉 System Load : {info['load']}%")
                                print("-" * 40)
                                ring_terminal_bell(count=4)
                        
                        # Event Trigger B: A node recovers safely back down to a NOMINAL frame
                        elif status.upper() == "NOMINAL" and prev_status.upper() == "CRITICAL":
                            print(f"\n✅ [HAZARD RESOLVED]: Node '{node_name}' has recovered to Nominal operational boundaries.")

                    # Synchronize the tracking buffer frame
                    previous_states = current_states

            time.sleep(poll_interval_seconds)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Hazard monitor loop gracefully detached. Standing down.")

if __name__ == "__main__":
    # Fallback to generate a file if missing for immediate testing
    default_csv = "visio_mapping.csv"
    if not os.path.exists(default_csv):
        with open(default_csv, mode='w', newline='', encoding='utf-8') as init_f:
            writer = csv.writer(init_f)
            writer.writerow(["Process ID", "Node Name", "Hardware Mapping", "Resource Load (%)", "Status Indicator", "Next Node Link"])
            writer.writerow(["N-01", "Tiger_L4_GPU_0", "NVIDIA_L4_VRAM_0", "35.2", "Nominal", ""])
            print(f"[INIT] Created template placeholder file: {default_csv}")

    monitor_visio_hazards(target_csv=default_csv)
