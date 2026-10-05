# monitor_visio_hazards.py
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
            reader = csv.DictReader(f)
            for row in reader:
                node_name = row.get("Node Name", "Unknown_Node")
                status = row.get("Status Indicator", "Nominal").strip()
                hardware = row.get("Hardware Mapping", "Unknown_HW")
                load = row.get("Resource Load (%)", "0.0")
                proc_id = row.get("Process ID", "Unknown_ID")
                
                node_states[node_name] = {
                    "id": proc_id,
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
    Triggers customized visual alerts and hardware audio cues based on the node layer.
    """
    print("=========================================================")
    print(f"🚀 UNIVAC-IX: Real-Time Visio Hazard Monitor Online")
    print(f"📡 Target Resource  : {os.path.abspath(target_csv)}")
    print(f"⏱  Polling Frequency: Every {poll_interval_seconds} seconds")
    print("=========================================================")
    print("[SYSTEM] Monitoring file stream... Press Ctrl+C to terminate.")

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
                    
                    for node_name, info in current_states.items():
                        status = info["status"]
                        prev_info = previous_states.get(node_name, {})
                        prev_status = prev_info.get("status", "Unknown")

                        # Event Trigger: A node transforms into a CRITICAL threat landscape
                        if status.upper() == "CRITICAL":
                            if prev_status.upper() != "CRITICAL":
                                
                                # CUSTOM STELLAR TRAFFIC LAYER EXCEPTION HANDLER
                                if "Stellar" in node_name or "Traffic" in node_name or info["id"] == "N-06":
                                    print(f"\n🛑 [STELLAR LAYER: GRIDLOCK & SECTOR SATURATION BREACH] 🛑")
                                    print(f" 👉 Intersection ID : {info['id']} ({node_name})")
                                    print(f" 👉 Routing Engine : {info['hardware']}")
                                    print(f" 👉 Queue Load     : {info['load']}% Saturation")
                                    print(f" 👉 Status Report  : PLA Grid Blocked / ROW Traffic Signal Timeout")
                                    print(f" ⚠️  [ACTION]: Resetting programmable logic array priority gates down the line.")
                                    print("-" * 65)
                                    ring_terminal_bell(count=5) # Extra alert pattern for gridlocks
                                
                                # Default alert readout for standard terrestrial/satellite nodes
                                else:
                                    print(f"\n🚨 [CRITICAL HAZARD DETECTED] 🚨")
                                    print(f" 👉 Node Variant: {node_name}")
                                    print(f" 👉 Hardware    : {info['hardware']}")
                                    print(f" 👉 System Load : {info['load']}%")
                                    print("-" * 45)
                                    ring_terminal_bell(count=3)
                        
                        # Event Trigger: A node recovers safely back down to a NOMINAL frame
                        elif status.upper() == "NOMINAL" and prev_status.upper() == "CRITICAL":
                            if "Stellar" in node_name or "Traffic" in node_name or info["id"] == "N-06":
                                print(f"\n🟢 [STELLAR RESOLVED]: Intersection '{node_name}' clear. ROW traffic signals flowing normally.")
                            else:
                                print(f"\n✅ [HAZARD RESOLVED]: Node '{node_name}' has recovered to Nominal operational boundaries.")

                    previous_states = current_states

            time.sleep(poll_interval_seconds)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Hazard monitor loop gracefully detached. Standing down.")

if __name__ == "__main__":
    # Ensure environment settings are passed locally
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    monitor_visio_hazards(target_csv="visio_mapping.csv")
