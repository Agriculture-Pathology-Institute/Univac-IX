# stream_all_nodes.py
import os
import sys
import time
import random
import pathlib

# Inject workspace directory trees
WORKSPACE_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(WORKSPACE_ROOT))

try:
    # Latch onto your custom updated layout exporter script
    import visio_exporter
except ImportError:
    print("[-] Integration Warning: 'visio_exporter.py' not found in local workspace paths.")
    sys.exit(1)

def run_real_time_telemetry_streamer(target_csv: str = "visio_mapping.csv", stream_delay: float = 1.0):
    """
    Main execution server loop simulating concurrent multi-threaded telemetry tracking.
    Continuously modulates workloads across all infrastructure software node layers.
    Includes an automated safe-mode override flushing mechanism for Stellar gridlocks.
    """
    print("=========================================================")
    print(f"📡 UNIVAC-IX: Real-Time Multi-Node Telemetry Server")
    print(f"🛰  Target Output Matrix: {os.path.abspath(target_csv)}")
    print(f"⏱  Streaming Frequency : Continual update every {stream_delay}s")
    print("=========================================================")
    print("[SYSTEM] Launching engine... Press Ctrl+C to stop streaming data.")

    # Initialize a baseline data dictionary for all infrastructure software layers
    nodes_state = [
        {
            "id": "N-01", 
            "name": "Tiger_Data_Carver_Core", 
            "hardware": "64_CORE_CPU_STAGING", 
            "load": 40.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": "N-02"
        },
        {
            "id": "N-02", 
            "name": "Turnkey_Industrial_Bridge", 
            "hardware": "NVIDIA_L4_VRAM_0", 
            "load": 50.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": "N-04"
        },
        {
            "id": "N-03", 
            "name": "Bridges_Turf_Automation", 
            "hardware": "NVIDIA_L4_VRAM_1", 
            "load": 45.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": "N-04"
        },
        {
            "id": "N-04", 
            "name": "Bennan_Satellite_Bayes_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 60.0, 
            "computed_variance": 2.50, 
            "status": "Nominal", 
            "shape_type": "Ellipse",
            "link": ""
        },
        {
            "id": "N-06", 
            "name": "Stellar_PLA_Traffic_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 55.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-07", 
            "name": "Malthusian_Biomass_Orchestrator", 
            "hardware": "64_CORE_PARALLEL_GRID", 
            "load": 50.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
            {
            "id": "N-11", 
            "name": "Siemens_S7_PLC_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 48.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-12", 
            "name": "Kirkland_Cost_Calculator", 
            "hardware": "64_CORE_CPU_STAGING", 
            "load": 38.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-13", 
            "name": "Stellar_Punk_Steam_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 52.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-14", 
            "name": "Stellar_Net_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 42.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-15", 
            "name": "Stellar_Life_Temporal_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 47.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-16", 
            "name": "Stellar_Green_World_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 44.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-17", 
            "name": "Stellar_GRiD_Infrastructure_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 49.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-18", 
            "name": "Stellar_RANGE_Orchestrator_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 41.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
        {
            "id": "N-19", 
            "name": "Qualcomm_DOC_Enforcement_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 45.0, 
            "status": "Nominal", 
            "shape_type": "Process",
            "link": ""
        }
    ]

    cycle_count = 0
    stellar_override_active = False

    try:
        while True:
            cycle_count += 1
            sys.stdout.write(f"\r🚀 [CYCLE {cycle_count:04d}] Multiplexing signal arrays across nodes...")
            sys.stdout.flush()

            # Dynamic State Modulation Loop
            for node in nodes_state:
                # AUTOMATED FALLBACK OVERRIDE EXECUTION BLOCK
                if node["id"] == "N-06" and stellar_override_active:
                    print(f"\n⚡ [AUTOMATED FALLBACK RECOVERY ENGAGED] ⚡")
                    print(f" 👉 Targeting Node   : {node['name']}")
                    print(f" 👉 Current Overload  : {node['load']}%")
                    print(f" 👉 Resolution Action : Injecting Safe-Mode Override String to Flush PLA Queue.")
                    print("-" * 65)
                    
                    # Cool down the system parameters instantly to clear congestion
                    node["load"] = 35.0
                    node["status"] = "Nominal"
                    
                    # Disengage override switch for standard tracking operation cycles
                    stellar_override_active = False
                    continue

                # Modulate CPU/GPU Hardware Resource Load randomly
                load_variance = random.uniform(-15.0, 15.0)
                node["load"] = max(10.0, min(100.0, round(node["load"] + load_variance, 1)))

                # Handle custom variance simulation hooks for the Bennan Satellite node layer
                if "Bennan" in node["name"]:
                    bayes_drift = random.uniform(-0.8, 1.2)
                    node["computed_variance"] = max(0.1, round(node["computed_variance"] + bayes_drift, 2))
                    
                    if node["computed_variance"] > 4.85:
                        node["status"] = "Critical"
                    else:
                        node["status"] = "Nominal"
                
                # Handle standard threshold calculations for Terrestrial layers
                else:
                    if node["load"] > 88.0:
                        node["status"] = "Critical"
                        # Catch if the overloaded target maps to the Stellar Traffic core footprint
                        if node["id"] == "N-06":
                            stellar_override_active = True
                    else:
                        node["status"] = "Nominal"

            # Flush parameters out to the shared Visio CSV file matrix
            visio_exporter.export_tiger_node_topology_to_visio(nodes_state, output_filename=target_csv)
            
            # Wait for next streaming telemetry block interval
            time.sleep(stream_delay)

    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Telemetry pipeline closed down cleanly. Core fabric standing down.")

if __name__ == "__main__":
    os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    run_real_time_telemetry_streamer(target_csv="visio_mapping.csv", stream_delay=1.0)
