import time
import random
from typing import Dict, Any, List

class CropState:
    SEED = "Seed"
    VEGETATIVE = "Vegetative"
    MATURE = "Mature"
    WITHERED = "Withered"

class AdvancedFarmingNode:
    """
    Represents an automated, sensory-driven farming node that evaluates environmental 
    metrics to optimize crop yield, irrigation cycles, and harvesting windows.
    """
    def __init__(self, node_id: str, field_capacity: int = 100):
        self.node_id: str = node_id
        self.field_capacity: int = field_capacity
        self.soil_moisture: float = 65.0  # Percentage
        self.nutrient_levels: float = 80.0  # Percentage
        self.crop_health: float = 100.0  # Percentage
        self.growth_progress: float = 0.0  # Out of 100
        self.current_state: str = CropState.SEED
        self.yield_multiplier: float = 1.0

    def read_sensors(self) -> Dict[str, float]:
        """Simulates internal hardware telemetry checks."""
        # Environmental degradation simulation
        self.soil_moisture -= random.uniform(0.5, 2.0)
        self.nutrient_levels -= random.uniform(0.1, 0.8)
        
        # Clamp sensor limits
        self.soil_moisture = max(0.0, min(100.0, self.soil_moisture))
        self.nutrient_levels = max(0.0, min(100.0, self.nutrient_levels))
        return {"moisture": self.soil_moisture, "nutrients": self.nutrient_levels}

    def execute_logic_cycle(self) -> List[str]:
        """
        State machine control block managing node life-cycle execution 
        and adaptive intervention triggers.
        """
        telemetry = self.read_sensors()
        actions_taken = []

        # 1. Resource Intervention Rules
        if telemetry["moisture"] < 40.0:
            actions_taken.append(self._trigger_irrigation())
        if telemetry["nutrients"] < 50.0:
            actions_taken.append(self._apply_nutrients())

        # 2. Environmental Degradation Adjustments
        if self.soil_moisture < 20.0 or self.soil_moisture > 90.0:
            self.crop_health -= 3.0
            self.yield_multiplier *= 0.95
        
        # 3. Growth Phase Management
        if self.crop_health <= 0:
            self.current_state = CropState.WITHERED
            return ["NODE_FAILURE: Crop has withered."]

        if self.current_state == CropState.SEED:
            self.growth_progress += 5.0 * self.yield_multiplier
            if self.growth_progress >= 25.0:
                self.current_state = CropState.VEGETATIVE
                actions_taken.append("PHASE_SHIFT: Transitioned to Vegetative state.")
                
        elif self.current_state == CropState.VEGETATIVE:
            self.growth_progress += 3.5 * self.yield_multiplier
            if self.growth_progress >= 85.0:
                self.current_state = CropState.MATURE
                actions_taken.append("PHASE_SHIFT: Crop is fully Mature and ready for harvest.")

        return [action for action in actions_taken if action]

    def harvest(self) -> Dict[str, Any]:
        """Harvests the node, evaluating final resource outputs."""
        if self.current_state != CropState.MATURE:
            return {"success": False, "yield": 0, "message": "Harvest aborted: Crop not mature."}
        
        final_yield = int(self.field_capacity * (self.crop_health / 100.0) * self.yield_multiplier)
        
        # Reset node state
        self.current_state = CropState.SEED
        self.growth_progress = 0.0
        self.crop_health = 100.0
        self.yield_multiplier = 1.0
        
        return {
            "success": True,
            "yield": final_yield,
            "message": f"Harvest complete. Gathered {final_yield} units from Node {self.node_id}."
        }

    def _trigger_irrigation(self) -> str:
        self.soil_moisture = min(75.0, self.soil_moisture + 30.0)
        return "RESOURCE_DISPATCH: Solenoid open. Irrigation cycle applied."

    def _apply_nutrients(self) -> str:
        self.nutrient_levels = min(90.0, self.nutrient_levels + 25.0)
        return "RESOURCE_DISPATCH: Fertilizer injector open. NPK values restored."


# --- Execution Sandbox ---
if __name__ == "__main__":
    print("Initializing Farm Node Array Alpha...")
    farm_node = AdvancedFarmingNode(node_id="Node-01A", field_capacity=150)
    
    # Run loop to simulate growth over cycles
    for cycle in range(1, 31):
        logs = farm_node.execute_logic_cycle()
        
        if farm_node.current_state == CropState.MATURE:
            print(f"[Cycle {cycle}] Status Check: {farm_node.current_state} (Progress: {farm_node.growth_progress:.1f}%)")
            harvest_report = farm_node.harvest()
            print(f" >> {harvest_report['message']}")
            break
        
        if logs:
            for log in logs:
                print(f"[Cycle {cycle}] {log}")
