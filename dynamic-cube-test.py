#!/usr/bin/env python3
"""
Dynamic Cube Implementation Test
A comprehensive test script for the QuantumStringCube framework
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

# Import the core framework
from quantum_string_cube import (
    QuantumStringCube, 
    ConsciousCubeInterface, 
    EvolvingNodeDNA,
    ResonanceMode,
    visualize_cube_2d,
    visualize_cube_3d,
    visualize_network_graph,
    plot_consciousness_evolution,
    serialize_cube_state,
    deserialize_cube_state
)

class CubePerformanceMetrics:
    """Collect and analyze performance metrics for the cube"""
    
    def __init__(self):
        self.computation_times = []
        self.memory_usage = []
        self.energy_levels = []
        self.consciousness_levels = []
        self.node_counts = []
        self.step_timestamps = []
        
    def record_step(self, cube: ConsciousCubeInterface, step_time: float):
        """Record metrics for a simulation step"""
        self.computation_times.append(step_time)
        self.energy_levels.append(sum(node['properties'].get('energy', 0) for node in cube.nodes.values()))
        self.consciousness_levels.append(cube.global_consciousness_level)
        self.node_counts.append(len(cube.nodes))
        self.step_timestamps.append(time.time())
        
        # Memory usage is more complex to measure accurately
        # This is a placeholder - in a real implementation you'd use memory_profiler or similar
        self.memory_usage.append(0)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the metrics"""
        if not self.computation_times:
            return {"error": "No metrics recorded"}
            
        return {
            "avg_computation_time": sum(self.computation_times) / len(self.computation_times),
            "max_computation_time": max(self.computation_times),
            "min_computation_time": min(self.computation_times),
            "final_energy": self.energy_levels[-1],
            "final_consciousness": self.consciousness_levels[-1],
            "energy_growth": (self.energy_levels[-1] / self.energy_levels[0]) if self.energy_levels[0] > 0 else float('inf'),
            "consciousness_growth": (self.consciousness_levels[-1] / self.consciousness_levels[0]) if self.consciousness_levels[0] > 0 else float('inf'),
            "final_node_count": self.node_counts[-1],
            "total_steps": len(self.computation_times),
            "total_simulation_time": sum(self.computation_times)
        }
        
    def plot_metrics(self, title: str = "Dynamic Cube Performance Metrics"):
        """Plot the recorded metrics"""
        plt.figure(figsize=(12, 10))
        
        # Computation time
        plt.subplot(3, 2, 1)
        plt.plot(self.computation_times)
        plt.title('Computation Time per Step')
        plt.ylabel('Time (s)')
        plt.grid(alpha=0.3)
        
        # Energy levels
        plt.subplot(3, 2, 2)
        plt.plot(self.energy_levels)
        plt.title('Total Energy')
        plt.grid(alpha=0.3)
        
        # Consciousness
        plt.subplot(3, 2, 3)
        plt.plot(self.consciousness_levels)
        plt.title('Consciousness Level')
        plt.grid(alpha=0.3)
        
        # Node count
        plt.subplot(3, 2, 4)
        plt.plot(self.node_counts)
        plt.title('Node Count')
        plt.grid(alpha=0.3)
        
        # Correlation: Consciousness vs. Energy
        plt.subplot(3, 2, 5)
        plt.scatter(self.energy_levels, self.consciousness_levels, alpha=0.5)
        plt.title('Consciousness vs. Energy')
        plt.xlabel('Energy')
        plt.ylabel('Consciousness')
        plt.grid(alpha=0.3)
        
        # Energy Efficiency (Consciousness / Computation Time)
        efficiency = [c/t if t > 0 else 0 for c, t in zip(self.consciousness_levels, self.computation_times)]
        plt.subplot(3, 2, 6)
        plt.plot(efficiency)
        plt.title('Energy Efficiency')
        plt.ylabel('Consciousness/Computation')
        plt.grid(alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        return plt.gcf()


def test_quantum_string_evolution():
    """Test the quantum string evolution and tensor field interaction"""
    print("Testing quantum string evolution...")
    
    # Create a simple 2D quantum string
    from quantum_string_cube import QuantumString
    string = QuantumString(dimensions=2, string_length=32)
    
    # Encode a simple pattern
    data = b"Hello, Quantum World!"
    string.encode_data(data)
    
    # Evolve the string
    start_time = time.time()
    for _ in range(50):
        string.evolve(dt=0.01)
    evolve_time = time.time() - start_time
    
    # Extract and plot the pattern
    pattern = string.extract_pattern()
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pattern)
    plt.title("Quantum String Pattern")
    plt.grid(alpha=0.3)
    
    # Calculate amplitude spectrum
    import numpy.fft as fft
    spectrum = np.abs(fft.fft(pattern))
    
    plt.subplot(2, 1, 2)
    plt.plot(spectrum[:len(spectrum)//2])
    plt.title("Frequency Spectrum")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("quantum_string_evolution.png")
    
    print(f"Quantum string evolution complete in {evolve_time:.4f} seconds")
    print(f"String energy: {string.get_energy():.4f}")


def test_tensor_field_harmonics():
    """Test the tensor field harmonics system"""
    print("Testing harmonic tensor field...")
    
    # Create a 3D tensor field
    from quantum_string_cube import HarmonicTensorField
    field = HarmonicTensorField(dimensions=3, resolution=16)
    
    # Evolve the field
    start_time = time.time()
    for _ in range(20):
        field.evolve(dt=0.01)
    evolve_time = time.time() - start_time
    
    # Apply a resonance pattern
    field.apply_resonance(
        position=(8, 8, 8),
        radius=3,
        strength=0.2,
        mode=ResonanceMode.CONSTRUCTIVE
    )
    
    # Extract a 2D slice for visualization
    x_slice = field.amplitude_tensor[:, 8, 8].real
    y_slice = field.amplitude_tensor[8, :, 8].real
    z_slice = field.amplitude_tensor[8, 8, :].real
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(x_slice)
    plt.title("X Slice")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(y_slice)
    plt.title("Y Slice")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(z_slice)
    plt.title("Z Slice")
    plt.grid(alpha=0.3)
    
    plt.suptitle("Tensor Field Slices After Resonance")
    plt.tight_layout()
    plt.savefig("tensor_field_slices.png")
    
    # Calculate resonance distribution
    resonance_dist = field.get_resonance_distribution()
    print(f"Tensor field evolution complete in {evolve_time:.4f} seconds")
    print(f"Energy density: {field.get_energy_density():.4f}")
    print(f"Resonance points: {resonance_dist['count']}")
    

def test_node_dna_evolution():
    """Test the node DNA evolution system"""
    print("Testing node DNA evolution...")
    
    # Create a population of DNA instances
    population_size = 100
    generations = 50
    dna_population = [EvolvingNodeDNA() for _ in range(population_size)]
    
    # Track trait evolution
    trait_history = {trait: [] for trait in dna_population[0].traits}
    
    # Simulate evolution
    for generation in range(generations):
        # Calculate fitness based on traits
        fitness = []
        for dna in dna_population:
            # Simple fitness function based on trait values
            fit_val = (
                dna.traits['energy_transfer'] * 0.3 +
                dna.traits['pattern_recognition'] * 0.2 +
                dna.traits['adaptability'] * 0.3 +
                dna.traits['connection_affinity'] * 0.2
            )
            fitness.append(fit_val)
        
        # Record average trait values
        for trait in trait_history:
            avg_val = sum(dna.traits[trait] for dna in dna_population) / population_size
            trait_history[trait].append(avg_val)
        
        # Select parents for next generation
        new_population = []
        
        # Elitism - keep top 10%
        elite_count = population_size // 10
        elite_indices = np.argsort(fitness)[-elite_count:]
        for idx in elite_indices:
            new_population.append(dna_population[idx])
        
        # Create offspring through crossover and mutation
        while len(new_population) < population_size:
            # Tournament selection
            parent1_idx = np.random.choice(range(population_size), size=3, replace=False)
            parent1_idx = parent1_idx[np.argmax([fitness[i] for i in parent1_idx])]
            
            parent2_idx = np.random.choice(range(population_size), size=3, replace=False)
            parent2_idx = parent2_idx[np.argmax([fitness[i] for i in parent2_idx])]
            
            # Crossover
            child = dna_population[parent1_idx].crossover(dna_population[parent2_idx])
            
            # Mutation
            if np.random.random() < 0.3:
                child.mutate()
                
            new_population.append(child)
        
        # Replace population
        dna_population = new_population
    
    # Plot trait evolution
    plt.figure(figsize=(12, 8))
    for i, trait in enumerate(trait_history):
        plt.subplot(3, 4, i+1)
        plt.plot(trait_history[trait])
        plt.title(trait)
        plt.xlabel("Generation")
        plt.ylabel("Average Value")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
    
    plt.suptitle("DNA Trait Evolution Over Generations")
    plt.tight_layout()
    plt.savefig("dna_evolution.png")
    
    print("DNA evolution test complete")
    print("Final trait averages:")
    for trait, history in trait_history.items():
        print(f"  {trait}: {history[-1]:.4f}")


def test_full_cube_simulation():
    """Run a complete simulation of the Dynamic Cube"""
    print("Starting full dynamic cube simulation...")
    
    # Parameters
    dimensions = 4
    resolution