# visio_exporter.py
import csv
import os

def export_tiger_node_topology_to_visio(node_registry_data: list, output_filename: str = "visio_mapping.csv"):
    """
    Compiles active Tiger-Vis hardware node tracking schemas into 
    Microsoft Visio Data Visualizer CSV configurations.
    """
    # Define standard Visio Data Visualizer mapping headers
    headers = [
        "Process ID", 
        "Node Name", 
        "Hardware Mapping", 
        "Resource Load (%)", 
        "Status Indicator", 
        "Next Node Link"
    ]
    
    print(f"[VISIO-EXPORTER] Generating mapping template sheet: {output_filename}")
    
    try:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write mapping header controls
            writer.writerow(headers)
            
            # Populate operational records
            for node in node_registry_data:
                writer.writerow([
                    node.get("id"),
                    node.get("name"),
                    node.get("hardware"),
                    node.get("load"),
                    node.get("status"),  # e.g., 'Nominal', 'Critical' -> sets shape color
                    node.get("link")     # Establishes structural process arrows in Visio
                ])
                
        print(f"✓ Visio export file written successfully to {os.path.abspath(output_filename)}.")
        return True
    except IOError as e:
        print(f"[-] Writing to Visio data map failed: {str(e)}")
        return False

# --- Quick Test Execution Matrix ---
if __name__ == "__main__":
    # Simulate a live multi-node grid tracking topology
    mock_system_state = [
        {"id": "N-01", "name": "Tiger_Ingest_Core", "hardware": "64_CORE_CPU_STAGING", "load": 42.5, "status": "Nominal", "link": "N-02"},
        {"id": "N-02", "name": "Tiger_L4_GPU_0", "hardware": "NVIDIA_L4_VRAM_0", "load": 88.1, "status": "Nominal", "link": "N-04"},
        {"id": "N-03", "name": "Tiger_L4_GPU_1", "hardware": "NVIDIA_L4_VRAM_1", "load": 94.2, "status": "Critical", "link": "N-04"},
        {"id": "N-04", "name": "Tiger_Analytics_Engine", "hardware": "64_CORE_MATH_UNIT", "load": 12.0, "status": "Nominal", "link": ""}
    ]
    
    export_tiger_node_topology_to_visio(mock_system_state)
