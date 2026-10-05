import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def process_36bit_agricultural_stream(raw_data_array, field_capacity):
    """
    Numba-accelerated parallel node for unpacking 36-bit legacy chunks
    into separate 4-bit metrics (e.g., soil, yield, moisture).
    """
    num_elements = len(raw_data_array)
    unpacked_yields = np.zeros(num_elements, dtype=np.float32)
    
    # prange forces execution across all available multi-core CPU threads
    for i in prange(num_elements):
        word = raw_data_array[i]
        
        # Isolate specific 4-bit nibbles from the 36-bit layout
        soil_moisture = (word >> 32) & 0x0F
        nutrient_level = (word >> 28) & 0x0F
        crop_health = (word >> 24) & 0x0F
        
        # Calculate real-time yield multipliers based on legacy telemetry thresholds
        if soil_moisture < 4 or nutrient_level < 5:
            health_penalty = 0.5
        else:
            health_penalty = 1.0
            
        unpacked_yields[i] = field_capacity * (crop_health / 15.0) * health_penalty
        
    return unpacked_yields

if __name__ == "__main__":
    print("Initializing Multi-Core Univac-IX Parallel Node...")
    # Simulate a stream of 100,000 legacy 36-bit data words
    simulated_words = np.random.randint(0, 2**36 - 1, size=100000, dtype=np.int64)
    
    # Process stream using parallel execution blocks
    yield_results = process_36bit_agricultural_stream(simulated_words, field_capacity=150.0)
    print(f"Successfully processed {len(yield_results)} node records via parallel threads.")
    print(f"Mean calculated node yield: {np.mean(yield_results):.2f} units.")
