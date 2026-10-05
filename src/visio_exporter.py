# visio_exporter.py
import csv
import os

def export_tiger_node_topology_to_visio(node_registry_data: list, output_filename: str = "visio_mapping.csv"):
    """
    Compiles active Tiger-Vis, Turnkey, Turf, and Bennan Satellite node tracking 
    schemas into Microsoft Visio Data Visualizer CSV configurations with custom shape indicators.
    """
    # Expanded Visio Data Visualizer mapping headers to support physical shape overrides
    headers = [
        "Process ID", 
        "Node Name", 
        "Hardware Mapping", 
        "Resource Load (%)", 
        "Status Indicator", 
        "Shape Type",       # New field mapping to specific Visio Master Shapes
        "Next Node Link"
    ]
    
    print(f"[VISIO-EXPORTER] Generating updated mapping template sheet: {output_filename}")
    
    try:
        with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write mapping header controls
            writer.writerow(headers)
            
            # Populate operational records
            for node in node_registry_data:
                node_name = node.get("name", "Unknown_Node")
                status = node.get("status", "Nominal")
                
                # NATIVE BENNAN SATELLITE LAYER ADJUSTMENT: 
                # Automatically map custom shapes and symbols based on node classification
                if "Bennan" in node_name or "Satellite" in node_name:
                    # Enforce specialized geometric ellipse shapes for orbital tracking networks
                    shape_type = "Ellipse"
                    if node.get("computed_variance", 0.0) > 4.85:
                        status = "Critical"
                else:
                    # Default structural workflow shapes for terrestrial industrial units
                    shape_type = "Process" if status == "Nominal" else "Decision"

                # Override if explicitly set in the data payload
                shape_type = node.get("shape_type", shape_type)

                writer.writerow([
                    node.get("id"),
                    node_name,
                    node.get("hardware"),
                    node.get("load"),
                    status,       # e.g., 'Nominal', 'Critical' -> dynamically updates fill colors in Visio
                    shape_type,   # Instructs Visio to drop a custom stencil asset type
                    node.get("link")
                ])
                
        print(f"✓ Visio export file written successfully to {os.path.abspath(output_filename)}.")
        return True
    except IOError as e:
        print(f"[-] Writing to Visio data map failed: {str(e)}")
        return False

# --- Multi-Node Grid Topology Test Matrix ---
if __name__ == "__main__":
    # Simulate a live multi-node grid tracking topology across all active layers
    mock_system_state = [
        {
            "id": "N-01", 
            "name": "Tiger_Ingest_Core", 
            "hardware": "64_CORE_CPU_STAGING", 
            "load": 42.5, 
            "status": "Nominal", 
            "link": "N-02"
        },
        {
            "id": "N-02", 
            "name": "Turnkey_Industrial_Bridge", 
            "hardware": "NVIDIA_L4_VRAM_0", 
            "load": 74.1, 
            "status": "Nominal", 
            "link": "N-04"
        },
        {
            "id": "N-03", 
            "name": "Bridges_Turf_Automation", 
            "hardware": "NVIDIA_L4_VRAM_1", 
            "load": 91.8, 
            "status": "Nominal", 
            "link": "N-04"
        },
        # --- NEW INTEGRATED BENNAN NODE RECORD ---
        {
            "id": "N-04", 
            "name": "Bennan_Satellite_Bayes_Core", 
            "hardware": "DUAL_L4_CO_PROCESSING", 
            "load": 82.4, 
            "computed_variance": 5.12,  # Breaches standard deviation limit -> triggers 'Critical' override
            "status": "Nominal", 
            "link": ""
        }
    ]
    
    export_tiger_node_topology_to_visio(mock_system_state)
