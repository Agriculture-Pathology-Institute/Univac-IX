# load_config.py
import os
import yaml
import sys
import pathlib

def load_and_validate_univac_fabric(config_path: str = "config.yaml"):
    """
    Parses config.yaml, maps environment overrides, and validates that 
    concurrent multi-node resource allocation matches physical cluster profiles.
    """
    path = pathlib.Path(config_path)
    if not path.exists():
        print(f"[-] Critical Error: Configuration target '{config_path}' cannot be resolved.")
        return False

    print("=========================================================")
    print(f"⚙️  UNIVAC-IX: Initializing Global Cluster Fabric Orchestrator")
    print("=========================================================")
    
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"[-] Invalid YAML Configuration Format: {exc}")
            return False

    # 1. Enforce Environment Overrides Natively
    cluster_meta = config.get("global_cluster_settings", {})
    overrides = cluster_meta.get("environment_overrides", {})
    for key, value in overrides.items():
        os.environ[key] = str(value)
        print(f" 👉 Setting Environment Variable: {key}={value}")

    # 2. Extract Cluster Capacities
    max_cpus = float(cluster_meta.get("total_system_cpus", 64))
    max_gpus = float(cluster_meta.get("total_system_gpus", 2))
    print(f"\n[SYSTEM PROFILE]: {cluster_meta.get('cluster_identity')}")
    print(f" -> Memory Pool Allocator   : {cluster_meta.get('system_ram_capacity_gb')} GB Available.")
    print(f" -> Maximum Compute Cores   : {max_cpus} Threads.")
    print(f" -> Maximum Graphics Engines: {max_gpus} NVIDIA Devices.")
    print("-" * 57)

    # 3. Aggregate Allocation across All Active Stages
    allocated_cpus = 0.0
    allocated_gpus = 0.0
    stages = config.get("pipeline_stages", [])

    print(f"\n[VALIDATING STAGE MANIFESTS]: Found {len(stages)} pipeline stages.")
    for stage in stages:
        name = stage.get("stage_name")
        resources = stage.get("resource_allocation", {})
        
        cpu_req = float(resources.get("cpus", 0))
        gpu_req = float(resources.get("gpus", 0))
        is_spmd = resources.get("is_spmd", False)
        
        allocated_cpus += cpu_req
        allocated_gpus += gpu_req
        
        print(f" 📦 {name}")
        print(f"    - Target Class: {stage.get('module_class')}")
        print(f"    - Allocation  : {cpu_req} Cores | {gpu_req} GPUs [SPMD Mode: {is_spmd}]")

    # 4. Perform Strict Hardware Limit Boundary Validation Checks
    print("\n[VERIFYING ALLOCATION BOUNDARIES]...")
    print(f" -> CPU Footprint: Allocated {allocated_cpus} / {max_cpus} Cores Total.")
    print(f" -> GPU Footprint: Allocated {allocated_gpus} / {max_gpus} NVIDIA Units Total.")

    if allocated_cpus > max_cpus:
        print("❌ CRITICAL OVER-ALLOCATION EXCEPTION: Total CPU threads request exceeds 64-core system limits!")
        return False
    if allocated_gpus > max_gpus:
        print("❌ CRITICAL OVER-ALLOCATION EXCEPTION: Total GPU resource footprint exceeds physical dual NVIDIA L4 limitations!")
        return False

    print("\n✅ CONFIGURATION COMPLIANT. Infrastructure nodes cleared for multi-threaded streaming execution.")
    return True

if __name__ == "__main__":
    success = load_and_validate_univac_fabric()
    if not success:
        sys.exit(1)
