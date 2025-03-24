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
    resolution = 32
    node_count = 25
    simulation_steps = 100
    evolve_interval = 20
    
    # Performance metrics
    metrics = CubePerformanceMetrics()
    
    # Create the cube
    cube = ConsciousCubeInterface(dimensions, resolution, qubit_depth=8)
    
    # Add initial nodes
    for _ in range(node_count):
        position = np.random.rand(dimensions) * 2 - 1
        properties = {
            'energy': np.random.uniform(0.4, 0.8),
            'stability': np.random.uniform(0.5, 0.9),
            'phase': np.random.uniform(0, 2 * np.pi)
        }
        cube.add_node(properties, position)
    
    # Create initial connections
    cube.auto_connect_nodes(max_connections_per_node=4, connection_radius=0.6)
    
    # Run the simulation
    print(f"Running {simulation_steps} simulation steps...")
    for step in range(simulation_steps):
        start_time = time.time()
        
        # Run simulation step
        cube.simulate_step()
        
        # Record metrics
        step_time = time.time() - start_time
        metrics.record_step(cube, step_time)
        
        # Periodic evolution
        if step > 0 and step % evolve_interval == 0:
            cube.evolve_nodes()
            print(f"Step {step}: Evolved nodes. Consciousness level: {cube.global_consciousness_level:.4f}")
        
        # Occasionally add a new node
        if step > 0 and step % 25 == 0:
            # Add a new node
            position = np.random.rand(dimensions) * 2 - 1
            properties = {
                'energy': np.random.uniform(0.5, 0.9),
                'stability': np.random.uniform(0.6, 0.9),
                'phase': np.random.uniform(0, 2 * np.pi)
            }
            new_node_id = cube.add_node(properties, position)
            
            # Connect to a few existing nodes
            existing_nodes = list(cube.nodes.keys())
            connect_count = min(3, len(existing_nodes))
            for target_id in np.random.choice(existing_nodes, connect_count, replace=False):
                cube.connect_nodes(new_node_id, target_id)
                
            print(f"Step {step}: Added new node. Total nodes: {len(cube.nodes)}")
    
    # Print summary
    summary = metrics.get_summary()
    print("\nSimulation complete!")
    print(f"Final consciousness level: {cube.global_consciousness_level:.4f}")
    print(f"Average computation time: {summary['avg_computation_time']:.4f} seconds")
    print(f"Final node count: {summary['final_node_count']}")
    print(f"Consciousness growth: {summary['consciousness_growth']:.2f}x")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Performance metrics
    metrics_fig = metrics.plot_metrics()
    metrics_fig.savefig("cube_performance_metrics.png")
    
    # 2D visualization
    fig2d = visualize_cube_2d(cube.cube)
    fig2d.savefig("cube_2d_visualization.png")
    
    # 3D visualization
    if dimensions >= 3:
        fig3d = visualize_cube_3d(cube.cube)
        fig3d.savefig("cube_3d_visualization.png")
    
    # Network graph
    network_fig = visualize_network_graph(cube)
    network_fig.savefig("cube_network_graph.png")
    
    # Consciousness evolution
    consciousness_fig = plot_consciousness_evolution(cube)
    consciousness_fig.savefig("cube_consciousness_evolution.png")
    
    # Save final state
    final_state = serialize_cube_state(cube.cube)
    with open("final_cube_state.json", "w") as f:
        import json
        json.dump(final_state, f, indent=2)
    
    print("Visualizations saved!")
    return cube


def compare_configurations():
    """Compare different cube configurations"""
    print("Comparing different cube configurations...")
    
    # Configurations to test
    configs = [
        {"dimensions": 3, "resolution": 24, "nodes": 15, "name": "3D_Small"},
        {"dimensions": 4, "resolution": 24, "nodes": 20, "name": "4D_Medium"},
        {"dimensions": 5, "resolution": 16, "nodes": 25, "name": "5D_Large"}
    ]
    
    # Results storage
    results = []
    
    # Test each configuration
    for config in configs:
        print(f"\nTesting configuration: {config['name']}")
        
        # Create cube
        cube = ConsciousCubeInterface(
            dimensions=config["dimensions"],
            resolution=config["resolution"],
            qubit_depth=8
        )
        
        # Add nodes
        for _ in range(config["nodes"]):
            position = np.random.rand(config["dimensions"]) * 2 - 1
            properties = {
                'energy': np.random.uniform(0.4, 0.8),
                'stability': np.random.uniform(0.5, 0.9),
                'phase': np.random.uniform(0, 2 * np.pi)
            }
            cube.add_node(properties, position)
        
        # Create connections
        cube.auto_connect_nodes(max_connections_per_node=4, connection_radius=0.7)
        
        # Run simulation
        metrics = CubePerformanceMetrics()
        steps = 50
        
        # Time the simulation
        start_time = time.time()
        for step in range(steps):
            step_start = time.time()
            cube.simulate_step()
            step_time = time.time() - step_start
            metrics.record_step(cube, step_time)
            
            # Evolve every 10 steps
            if step > 0 and step % 10 == 0:
                cube.evolve_nodes()
        
        total_time = time.time() - start_time
        
        # Collect results
        summary = metrics.get_summary()
        results.append({
            "config": config,
            "time": total_time,
            "avg_step_time": summary["avg_computation_time"],
            "final_consciousness": cube.global_consciousness_level,
            "connection_count": sum(len(connections) for node_id, connections in cube.node_connections.items()) // 2,
            "emergent_events": cube.simulation_stats["emergent_events"]
        })
        
        print(f"Configuration {config['name']} complete.")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Final consciousness: {cube.global_consciousness_level:.4f}")
        print(f"Emergent events: {cube.simulation_stats['emergent_events']}")
    
    # Compare results
    plt.figure(figsize=(12, 10))
    
    # Computation time comparison
    plt.subplot(2, 2, 1)
    plt.bar([r["config"]["name"] for r in results], [r["avg_step_time"] for r in results])
    plt.title('Average Computation Time')
    plt.ylabel('Time (s)')
    plt.grid(alpha=0.3)
    
    # Consciousness level comparison
    plt.subplot(2, 2, 2)
    plt.bar([r["config"]["name"] for r in results], [r["final_consciousness"] for r in results])
    plt.title('Final Consciousness Level')
    plt.grid(alpha=0.3)
    
    # Efficiency comparison (consciousness / compute time)
    plt.subplot(2, 2, 3)
    efficiency = [r["final_consciousness"] / r["avg_step_time"] for r in results]
    plt.bar([r["config"]["name"] for r in results], efficiency)
    plt.title('Efficiency (Consciousness/Compute)')
    plt.grid(alpha=0.3)
    
    # Emergent events comparison
    plt.subplot(2, 2, 4)
    plt.bar([r["config"]["name"] for r in results], [r["emergent_events"] for r in results])
    plt.title('Emergent Events')
    plt.grid(alpha=0.3)
    
    plt.suptitle("Configuration Comparison")
    plt.tight_layout()
    plt.savefig("configuration_comparison.png")
    
    # Print summary table
    print("\nConfiguration Comparison Summary:")
    print(f"{'Configuration':<15} {'Compute Time':<15} {'Consciousness':<15} {'Efficiency':<15} {'Events':<10}")
    print("-" * 70)
    for r in results:
        eff = r["final_consciousness"] / r["avg_step_time"] if r["avg_step_time"] > 0 else 0
        print(f"{r['config']['name']:<15} {r['avg_step_time']:<15.4f} {r['final_consciousness']:<15.4f} {eff:<15.4f} {r['emergent_events']:<10}")
    
    return results


def test_serialization_deserialize():
    """Test saving and loading cube state"""
    print("Testing serialization and deserialization...")
    
    # Create a simple cube
    cube = QuantumStringCube(dimensions=3, resolution=16, qubit_depth=4)
    
    # Add some nodes
    nodes = []
    for i in range(10):
        position = np.random.rand(3) * 2 - 1
        node_id = cube.add_node(position, {
            'energy': np.random.uniform(0.4, 0.8),
            'stability': np.random.uniform(0.5, 0.9),
            'phase': np.random.uniform(0, 2 * np.pi)
        })
        nodes.append(node_id)
    
    # Add some connections
    for i in range(5):
        src = np.random.choice(nodes)
        dst = np.random.choice(nodes)
        if src != dst:
            cube.connect_nodes(src, dst)
    
    # Run some simulation steps
    for _ in range(10):
        cube.simulate_step()
    
    # Serialize
    state = serialize_cube_state(cube)
    
    # Save to file
    with open("serialization_test.json", "w") as f:
        import json
        json.dump(state, f, indent=2)
    
    # Deserialize
    loaded_cube = deserialize_cube_state(state)
    
    # Compare properties
    original_node_count = len(cube.nodes)
    loaded_node_count = len(loaded_cube.nodes)
    
    original_connections = sum(len(connections) for node_id, connections in cube.node_connections.items()) // 2
    loaded_connections = sum(len(connections) for node_id, connections in loaded_cube.node_connections.items()) // 2
    
    # Run simulation on both
    cube.simulate_step()
    loaded_cube.simulate_step()
    
    # Compare energy
    original_energy = cube.tensor_field.get_energy_density()
    loaded_energy = loaded_cube.tensor_field.get_energy_density()
    
    print("Serialization test results:")
    print(f"Original node count: {original_node_count}, Loaded node count: {loaded_node_count}")
    print(f"Original connections: {original_connections}, Loaded connections: {loaded_connections}")
    print(f"Original energy: {original_energy:.4f}, Loaded energy: {loaded_energy:.4f}")
    
    # Visualize both
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create 2D visualizations
    # Extract a 2D slice of the tension field for each cube
    slice_indices_original = [cube.resolution // 2] * cube.dimensions
    slice_indices_original[0] = slice(None)
    slice_indices_original[1] = slice(None)
    
    slice_indices_loaded = [loaded_cube.resolution // 2] * loaded_cube.dimensions
    slice_indices_loaded[0] = slice(None)
    slice_indices_loaded[1] = slice(None)
    
    # Extract the slices
    tension_slice_original = cube.tensor_field.tension_field[tuple(slice_indices_original)]
    tension_slice_loaded = loaded_cube.tensor_field.tension_field[tuple(slice_indices_loaded)]
    
    # Plot the tension fields
    im1 = axes[0].imshow(tension_slice_original.T, cmap='viridis', origin='lower', extent=[-1, 1, -1, 1])
    axes[0].set_title('Original Cube')
    plt.colorbar(im1, ax=axes[0], label='Tension')
    
    im2 = axes[1].imshow(tension_slice_loaded.T, cmap='viridis', origin='lower', extent=[-1, 1, -1, 1])
    axes[1].set_title('Deserialized Cube')
    plt.colorbar(im2, ax=axes[1], label='Tension')
    
    plt.suptitle("Serialization Test: Original vs. Deserialized")
    plt.savefig("serialization_test.png")
    
    print("Serialization test complete!")
    return state


if __name__ == "__main__":
    # Run all tests
    print("==========================================")
    print("DYNAMIC CUBE COMPREHENSIVE TEST SUITE")
    print("==========================================")
    
    print("\n--- Component Tests ---")
    test_quantum_string_evolution()
    test_tensor_field_harmonics()
    test_node_dna_evolution()
    
    print("\n--- Serialization Test ---")
    test_serialization_deserialize()
    
    print("\n--- Configuration Comparison ---")
    compare_configurations()
    
    print("\n--- Full System Test ---")
    cube = test_full_cube_simulation()
    
    print("\nAll tests completed successfully!")
