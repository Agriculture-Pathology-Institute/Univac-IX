# watch_report_sync.py
import os
import csv
import time
import pathlib
import sys
import email_alerts

# Inject workspace directory hooks
WORKSPACE_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(WORKSPACE_ROOT))

try:
    import update_report
except ImportError:
    print("[-] Critical Error: 'update_report.py' cannot be found in workspace directories.")
    sys.exit(1)

def run_report_synchronizer(target_csv: str = "visio_mapping.csv", poll_interval: float = 0.5):
    """
    Stateless background file-watch loop monitoring telemetry row events.
    Automatically increments recovery metrics and refreshes the HTML report card.
    """
    print("=========================================================")
    print(f"📊 UNIVAC-IX: Real-Time Report Card Synchronizer Server")
    print(f"📡 Tracking Source  : {os.path.abspath(target_csv)}")
    print(f"💻 Deployment Target: {os.path.abspath('execution_report.html')}")
    print("=========================================================")
    print("[SYSTEM] Monitoring file stream adjustments... Press Ctrl+C to stop.")

    last_modified_time = 0
    
    # Internal running counters for failover override activations
    stellar_flushes = 0
    biomass_flushes = 0
    
    # Track historical peak resource indicators for all 6 infrastructure nodes
    peak_metrics = {
        "N-01": 0.0, "N-02": 0.0, "N-03": 0.0, 
        "N-04": 0.0, "N-06": 0.0, "N-07": 0.0
    }
    
    # Buffer to trace state transitions (prevents double-counting a single incident loop)
    previous_critical_states = set()

    try:
        while True:
            if os.path.exists(target_csv):
                current_modified_time = os.path.getmtime(target_csv)
                
                # Check if the stream file configuration timestamp has updated
                if current_modified_time != last_modified_time:
                    last_modified_time = current_modified_time
                    
                    # Track current loop parameters
                    trigger_report_update = False
                    current_critical_nodes = set()
                    
                    # Read the incoming tracking matrix array blocks
                    try:
                        with open(target_csv, mode='r', newline='', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                node_id = row.get("Process ID", "").strip()
                                node_name = row.get("Node Name", "").strip()
                                status = row.get("Status Indicator", "Nominal").strip().upper()
                                
                                # Extract load floating points safely depending on layout columns
                                try:
                                    raw_load = row.get("Resource Load (%)", "0.0")
                                    # Handle Bennan variance edge case column maps
                                    if "Bennan" in node_name:
                                        load_val = float(raw_load.replace(" Var", ""))
                                    else:
                                        load_val = float(raw_load)
                                except ValueError:
                                    load_val = 0.0

                                # Update peak metric ledger boundary rules
                                if node_id in peak_metrics:
                                    if load_val > peak_metrics[node_id]:
                                        peak_metrics[node_id] = load_val
                                        trigger_report_update = True

                                # Handle Active Critical Incident Evaluations
                                if status == "CRITICAL":
                                    current_critical_nodes.add(node_id)
                                    
                                    # If the incident is newly surfaced during this step cycle
                                    if node_id not in previous_critical_states:
                                        trigger_report_update = True
                                        
                                        # Increment specific structural fallback matrix records
                                        if node_id == "N-06" or "Stellar" in node_name:
                                            stellar_flushes += 1
                                            print(f"\n[COUNTER UPDATE] Stellar PLA Gridlock detected. Incrementing flushes: {stellar_flushes}")
                                        elif node_id == "N-07" or "Biomass" in node_name:
                                            biomass_flushes += 1
                                            print(f"\n[COUNTER UPDATE] Institutional Biomass Overload detected. Incrementing flushes: {biomass_flushes}")

                    except Exception as err:
                        # Fail-safe skip for read locks during telemetry file write operations
                        time.sleep(0.05)
                        continue

                    # Update internal tracking structures
                    previous_critical_states = current_critical_nodes

                    # Re-compile and deploy the HTML dashboard upon change detection
                    if trigger_report_update:
                        sys.stdout.write("\n🔄 State change detected. Compiling updated report card layout...")
                        sys.stdout.flush()
                        
                        update_report.compile_execution_report_card(
                            stellar_count=stellar_flushes,
                            biomass_count=biomass_flushes,
                            peaks=peak_metrics,
                            template_name="report_template.html",
                            output_name="execution_report.html"
                        )

# Re-compile and deploy the HTML dashboard upon change detection
if trigger_report_update:
    update_report.compile_execution_report_card(
        stellar_count=stellar_flushes,
        biomass_count=biomass_flushes,
        peaks=peak_metrics,
        template_name="report_template.html",
        output_name="execution_report.html"
    )

    # AUTOMATED EMAIL DISPATCH TRIGGER
    # Check if any new newly captured critical instance initialized this cycle change
    if len(current_critical_nodes) > len(previous_critical_states):
        email_alerts.send_critical_report_email(report_path="execution_report.html")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Synchronization watcher gracefully detached. Standing down report pipelines.")

if __name__ == "__main__":
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    run_report_synchronizer(target_csv="visio_mapping.csv", poll_interval=0.5)
