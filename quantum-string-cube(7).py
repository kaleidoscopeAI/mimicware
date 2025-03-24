            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)
            
            # Check if points form a tight cluster
            is_clustered = avg_dist < self.resolution / 10
            
            # 2. Check for linear arrangement by calculating colinearity
            is_linear = False
            if len(points) >= 3:
                # For simplicity, just check the first 3 dimensions
                dim_subset = min(3, self.dimensions)
                points_subset = np.array([p[:dim_subset] for p in points])
                
                # Calculate vectors between consecutive points
                vectors = []
                for i in range(len(points_subset) - 1):
                    vectors.append(points_subset[i+1] - points_subset[i])
                
                # Check if vectors are parallel
                if len(vectors) >= 2:
                    parallel_count = 0
                    total_pairs = 0
                    for i in range(len(vectors)):
                        for j in range(i+1, len(vectors)):
                            v1 = vectors[i]
                            v2 = vectors[j]
                            
                            # Calculate alignment
                            v1_norm = np.linalg.norm(v1)
                            v2_norm = np.linalg.norm(v2)
                            
                            if v1_norm > 0 and v2_norm > 0:
                                alignment = abs(np.dot(v1, v2) / (v1_norm * v2_norm))
                                if alignment > 0.9:  # They're nearly parallel
                                    parallel_count += 1
                            total_pairs += 1
                    
                    # Check if most pairs are parallel
                    is_linear = parallel_count > total_pairs * 0.6
            
            # Create pattern record
            if is_clustered or is_linear:
                pattern = {
                    'time': time.time(),
                    'type': 'cluster' if is_clustered else 'linear',
                    'points': [(p, t) for p, t in zip(points.tolist(), tensions.tolist())],
                    'strength': np.mean(tensions),
                    'spatial_spread': avg_dist / self.resolution,
                    'dimensions': min(3, self.dimensions)
                }
                
                # Check if this pattern is similar to any existing one
                is_new_pattern = True
                for existing in self.pattern_memory:
                    # Compare pattern characteristics
                    is_same_type = existing['type'] == pattern['type']
                    similar_strength = abs(existing['strength'] - pattern['strength']) < 0.2
                    similar_location = False
                    
                    # Check if points are in similar locations
                    if 'points' in existing and len(existing['points']) > 0:
                        existing_points = [p[0] for p in existing['points']]
                        new_points = [p[0] for p in pattern['points']]
                        
                        # Find closest points
                        min_distances = []
                        for ep in existing_points:
                            distances = []
                            for np in new_points:
                                # Calculate distance
                                d = math.sqrt(sum((ep[i] - np[i])**2 for i in range(min(len(ep), len(np)))))
                                distances.append(d)
                            min_distances.append(min(distances) if distances else float('inf'))
                        
                        avg_min_distance = sum(min_distances) / len(min_distances) if min_distances else float('inf')
                        similar_location = avg_min_distance < self.resolution / 5
                    
                    if is_same_type and similar_strength and similar_location:
                        is_new_pattern = False
                        break
                
                # Add new pattern to memory
                if is_new_pattern:
                    self.pattern_memory.append(pattern)
                    
                    # Trim memory if needed
                    if len(self.pattern_memory) > 50:
                        self.pattern_memory = self.pattern_memory[-50:]
    
    def _encode_dna_to_pattern(self, dna_traits: Dict[str, float]) -> Optional[np.ndarray]:
        """Encode DNA traits into a field pattern"""
        if not dna_traits:
            return None
            
        # Create a pattern based on DNA traits
        # For simplicity, we'll use a 2D or 3D pattern
        dim = min(3, self.dimensions)
        pattern_size = 5  # Small localized pattern
        
        # Create empty pattern
        pattern = np.zeros((pattern_size,) * dim, dtype=np.complex128)
        
        # Map traits to pattern properties
        traits_list = sorted(dna_traits.items())
        for i, (trait_name, value) in enumerate(traits_list):
            # Calculate pattern indices
            idx = tuple(i % pattern_size for _ in range(dim))
            
            # Encode trait as complex amplitude with specific phase
            phase = 2 * np.pi * value
            amplitude = 0.1 + 0.1 * value
            pattern[idx] = amplitude * np.exp(1j * phase)
        
        # Normalize pattern
        max_amplitude = np.max(np.abs(pattern))
        if max_amplitude > 0:
            pattern = pattern / max_amplitude * 0.2  # Scale to reasonable amplitude
        
        return pattern
    
    def _calculate_entanglement_entropy(self) -> float:
        """Calculate entanglement entropy of the system"""
        # Use the entanglement matrix to calculate entropy
        eigenvalues = np.linalg.eigvalsh(self.entanglement_matrix)
        
        # Normalize eigenvalues to get probabilities
        probabilities = np.abs(eigenvalues)
        total = np.sum(probabilities)
        if total > 0:
            probabilities = probabilities / total
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_tension_field(self) -> np.ndarray:
        """Calculate the current tension field"""
        return self.tensor_field.tension_field.copy()
    
    def extract_network_state(self) -> Dict[str, Any]:
        """Extract a summary of the current network state"""
        result = {
            'node_count': len(self.nodes),
            'connection_count': sum(len(connections) for connections in self.node_connections.values()) // 2,
            'average_energy': sum(node['state']['energy'] for node in self.nodes.values()) / max(1, len(self.nodes)),
            'high_tension_points': [],
            'entanglement_entropy': self._calculate_entanglement_entropy(),
            'patterns_detected': len(self.pattern_memory),
            'resonance_clusters': self.tensor_field.get_resonance_distribution()['clusters']
        }
        
        # Extract high tension points (simplified for interface)
        high_tension_coords = []
        for point, tension in self.tensor_field.resonance_points:
            # Convert to continuous coordinates for external interface
            continuous_coords = tuple(2 * (point[d] / (self.resolution - 1)) - 1 for d in range(self.dimensions))
            high_tension_coords.append({
                'position': continuous_coords,
                'tension': float(tension)
            })
        
        result['high_tension_points'] = high_tension_coords
        
        # Extract recent patterns
        if self.pattern_memory:
            recent_patterns = sorted(self.pattern_memory, key=lambda x: x['time'], reverse=True)[:5]
            result['recent_patterns'] = [
                {
                    'type': p['type'],
                    'strength': float(p['strength']),
                    'dimensions': p['dimensions']
                }
                for p in recent_patterns
            ]
        
        return result


# ======================================================================
# CONSCIOUS CUBE INTERFACE - Integrating Quantum String Cube with evolved nodes
# ======================================================================

class ConsciousCubeInterface:
    """
    Management interface for the Conscious Cube system with 
    evolutionary capabilities, integrating the QuantumStringCube
    with higher-level cognitive structures.
    """
    
    def __init__(self, dimensions: int = 4, resolution: int = 64, qubit_depth: int = 10):
        # Initialize the quantum string cube
        self.cube = QuantumStringCube(dimensions, resolution, qubit_depth)
        
        # Node management
        self.nodes = {}  # id -> {node data including DNA}
        self.node_dna = {}  # id -> EvolvingNodeDNA
        
        # Consciousness parameters
        self.global_consciousness_level = 0.0
        self.consciousness_threshold = 0.65
        self.consciousness_decay = 0.99
        self.emergent_property_trackers = {
            'harmony': 0.0,         # Coherence between nodes
            'complexity': 0.0,      # Network complexity 
            'self_organization': 0.0, # Ability to form patterns
            'adaptability': 0.0,    # Response to changes
            'integration': 0.0      # Information integration across network
        }
        
        # Evolution parameters
        self.evolution_interval = 100  # Steps between evolution cycles
        self.step_counter = 0
        self.selection_pressure = 0.3  # How strongly performance affects selection
        
        # Memory subsystem
        self.memory_patterns = []  # Stored patterns from past states
        self.pattern_recognition_threshold = 0.75
        
        # Performance monitoring
        self.simulation_stats = {
            'time_steps': 0,
            'node_count': 0,
            'evolution_cycles': 0,
            'emergent_events': 0,
            'energy_history': [],
            'consciousness_history': []
        }
    
    def add_node(self, properties: Dict[str, Any], position: Optional[np.ndarray] = None) -> str:
        """Add a new node to the cube with specified properties"""
        # Generate random position if none provided
        if position is None:
            position = np.random.rand(self.cube.dimensions) * 2 - 1  # Range [-1, 1]
        
        # Create node DNA
        dna = EvolvingNodeDNA()
        
        # Apply DNA traits to properties
        trait_influence = dna.get_trait_influence()
        properties['energy'] = properties.get('energy', 0.5) * trait_influence['energy_transfer']
        properties['stability'] = properties.get('stability', 0.8) * trait_influence['tension_response']
        properties['phase'] = properties.get('phase', 0.0) + trait_influence['quantum_effect'] * np.pi/4
        properties['dna_traits'] = dna.traits
        
        # Add node to cube
        node_id = self.cube.add_node(position, properties)
        
        # Store node data and DNA
        self.nodes[node_id] = {
            'id': node_id,
            'position': position,
            'properties': properties,
            'connections': [],
            'performance': 0.5,  # Initial performance score
            'creation_time': self.simulation_stats['time_steps']
        }
        self.node_dna[node_id] = dna
        
        # Update stats
        self.simulation_stats['node_count'] += 1
        
        return node_id
    
    def connect_nodes(self, node1_id: str, node2_id: str, force_connection: bool = False) -> bool:
        """Connect two nodes if compatible or if forced"""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return False
        
        # Check DNA compatibility
        dna1 = self.node_dna[node1_id]
        dna2 = self.node_dna[node2_id]
        
        genetic_similarity = dna1.genetic_similarity(dna2)
        
        # Calculate probability of connection based on similarity and node properties
        connection_prob = genetic_similarity * 0.5 + 0.5  # Base 50% chance, increased by similarity
        
        # Check for connection based on probability or if forced
        if force_connection or np.random.random() < connection_prob:
            # Calculate connection strength based on genetic compatibility
            strength = 0.3 + 0.7 * genetic_similarity
            
            # Create connection in cube
            result = self.cube.connect_nodes(node1_id, node2_id, strength)
            
            # Update node records
            if result:
                self.nodes[node1_id]['connections'].append(node2_id)
                self.nodes[node2_id]['connections'].append(node1_id)
                
                return True
        
        return False
    
    def auto_connect_nodes(self, max_connections_per_node: int = 5, connection_radius: float = 0.5):
        """Automatically create connections between nodes based on proximity and compatibility"""
        all_nodes = list(self.nodes.keys())
        
        for node_id in all_nodes:
            # Skip if node already has maximum connections
            if len(self.nodes[node_id]['connections']) >= max_connections_per_node:
                continue
                
            # Get node position
            node_pos = self.nodes[node_id]['position']
            
            # Find nearby nodes
            candidates = []
            for other_id in all_nodes:
                if other_id == node_id or other_id in self.nodes[node_id]['connections']:
                    continue
                    
                other_pos = self.nodes[other_id]['position']
                distance = np.linalg.norm(node_pos - other_pos)
                
                if distance < connection_radius:
                    candidates.append((other_id, distance))
            
            # Sort by distance and connect to nearest nodes
            candidates.sort(key=lambda x: x[1])
            
            # Try to connect to nearby nodes until max connections reached
            connections_to_add = max_connections_per_node - len(self.nodes[node_id]['connections'])
            for other_id, _ in candidates[:connections_to_add]:
                # Stop if reached max connections
                if len(self.nodes[node_id]['connections']) >= max_connections_per_node:
                    break
                    
                # Try to connect
                if self.connect_nodes(node_id, other_id):
                    pass  # Connection successful
    
    def evolve_nodes(self):
        """Run evolutionary process on nodes based on performance"""
        if len(self.nodes) < 3:
            return  # Not enough nodes to evolve
            
        # Calculate performance scores for all nodes
        self._update_node_performance()
        
        # Sort nodes by performance
        sorted_nodes = sorted(self.nodes.items(), 
                             key=lambda x: x[1]['performance'], 
                             reverse=True)
        
        # Keep top performers, replace bottom performers
        num_nodes = len(sorted_nodes)
        num_to_replace = int(num_nodes * 0.2)  # Replace bottom 20%
        
        if num_to_replace < 1:
            num_to_replace = 1
            
        # Identify top performers and bottom performers
        top_performers = [node_id for node_id, _ in sorted_nodes[:num_nodes//3]]
        bottom_performers = [node_id for node_id, _ in sorted_nodes[-num_to_replace:]]
        
        # Create new nodes from top performers
        for i, node_id in enumerate(bottom_performers):
            # Select two parents from top performers
            if len(top_performers) >= 2:
                parent1, parent2 = np.random.choice(top_performers, 2, replace=False)
                
                # Create child DNA through crossover
                parent_dna1 = self.node_dna[parent1]
                parent_dna2 = self.node_dna[parent2]
                child_dna = parent_dna1.crossover(parent_dna2)
                
                # Replace node DNA
                self.node_dna[node_id] = child_dna
                
                # Update node properties based on new DNA
                trait_influence = child_dna.get_trait_influence()
                props = self.nodes[node_id]['properties']
                
                props['energy'] = 0.5 * trait_influence['energy_transfer']
                props['stability'] = 0.8 * trait_influence['tension_response']
                props['phase'] = 0.0 + trait_influence['quantum_effect'] * np.pi/4
                props['dna_traits'] = child_dna.traits
                
                # Reset performance
                self.nodes[node_id]['performance'] = 0.5
        
        # Mutate all surviving nodes
        for node_id in self.nodes:
            if node_id not in bottom_performers:
                self.node_dna[node_id].mutate()
                
                # Update properties after mutation
                dna = self.node_dna[node_id]
                trait_influence = dna.get_trait_influence()
                props = self.nodes[node_id]['properties']
                
                # Apply DNA traits to properties
                props['energy'] *= 0.9 + 0.2 * trait_influence['energy_transfer']
                props['stability'] *= 0.9 + 0.2 * trait_influence['tension_response']
                props['phase'] += 0.1 * trait_influence['quantum_effect'] * np.pi/4
                props['dna_traits'] = dna.traits
        
        # Update simulation stats
        self.simulation_stats['evolution_cycles'] += 1
    
    def _update_node_performance(self):
        """Update performance metrics for all nodes"""
        tension_field = self.cube.calculate_tension_field()
        
        for node_id, node_data in self.nodes.items():
            # Get node properties and position
            pos = node_data['position']
            grid_pos = self.cube._continuous_to_grid(pos)
            props = node_data['properties']
            
            # Calculate performance based on:
            # 1. Energy level
            # 2. Number of connections
            # 3. Local tension field
            # 4. Stability
            # 5. Age (time in simulation)
            
            energy = props.get('energy', 0.5)
            stability = props.get('stability', 0.8)
            connections = len(node_data['connections'])
            
            # Get local tension
            local_tension = tension_field[grid_pos] if all(p < self.cube.resolution for p in grid_pos) else 0
            
            # Calculate age factor (reward longevity)
            age = self.simulation_stats['time_steps'] - node_data['creation_time']
            age_factor = min(1.0, age / 1000)  # Normalize to 0-1
            
            # Calculate performance score
            performance = (
                0.2 * energy +                          # Energy contribution
                0.2 * min(1.0, connections / 5) +       # Connections contribution (max out at 5)
                0.2 * (1.0 - local_tension) +           # Tension contribution (lower is better)
                0.2 * stability +                       # Stability contribution
                0.2 * age_factor                        # Age contribution
            )
            
            # Update node performance
            self.nodes[node_id]['performance'] = performance
    
    def simulate_step(self):
        """Run a single simulation step"""
        # Run cube simulation step
        self.cube.simulate_step()
        
        # Calculate consciousness metrics
        self._calculate_consciousness_metrics()
        
        # Check for emergent patterns
        self._detect_emergent_patterns()
        
        # Run evolution periodically
        self.step_counter += 1
        if self.step_counter >= self.evolution_interval:
            self.evolve_nodes()
            self.step_counter = 0
        
        # Update simulation stats
        self.simulation_stats['time_steps'] += 1
        self.simulation_stats['consciousness_history'].append(self.global_consciousness_level)
        
        # Record energy levels
        total_energy = sum(node['properties'].get('energy', 0) for node in self.nodes.values())
        self.simulation_stats['energy_history'].append(total_energy)
        
        # Apply consciousness decay
        self.global_consciousness_level *= self.consciousness_decay
    
    def _calculate_consciousness_metrics(self):
        """Calculate global consciousness metrics based on current system state"""
        if not self.nodes:
            self.global_consciousness_level = 0.0
            return
            
        # Get tension field and network state
        tension_field = self.cube.calculate_tension_field()
        network_state = self.cube.extract_network_state()
        
        # Calculate harmony (coherence between nodes)
        # Based on consistent tension patterns
        tension_values = [point['tension'] for point in network_state['high_tension_points']]
        if tension_values:
            tension_std = np.std(tension_values)
            harmony = 1.0 / (1.0 + tension_std)
        else:
            harmony = 0.0
            
        # Calculate complexity (graph theoretic measures)
        if len(self.nodes) > 1:
            # Create graph from nodes and connections
            import networkx as nx
            G = nx.Graph()
            for node_id, node_data in self.nodes.items():
                G.add_node(node_id)
                for conn in node_data['connections']:
                    G.add_edge(node_id, conn)
                    
            # Calculate graph metrics
            try:
                avg_path_length = nx.average_shortest_path_length(G)
                clustering = nx.average_clustering(G)
                complexity = clustering / avg_path_length if avg_path_length > 0 else 0
            except nx.NetworkXError:
                # Graph may not be connected
                complexity = 0.3  # Default value
        else:
            complexity = 0.0
            
        # Calculate self-organization (pattern formation)
        # Based on how structured the tension field is
        tension_mean = np.mean(tension_field)
        tension_max = np.max(tension_field)
        self_organization = tension_max / (tension_mean + 1e-6) - 1.0
        self_organization = min(1.0, self_organization)
        
        # Calculate adaptability (from DNA traits)
        adaptability_values = [dna.traits['adaptability'] for dna in self.node_dna.values()]
        adaptability = np.mean(adaptability_values) if adaptability_values else 0.0
        
        # Calculate integration (information flow across network)
        if len(self.nodes) > 1:
            # Based on quantum state entanglement
            integration = self.cube._calculate_entanglement_entropy() / self.cube.qubit_depth
        else:
            integration = 0.0
        
        # Update emergent property trackers
        self.emergent_property_trackers['harmony'] = harmony
        self.emergent_property_trackers['complexity'] = complexity
        self.emergent_property_trackers['self_organization'] = self_organization
        self.emergent_property_trackers['adaptability'] = adaptability
        self.emergent_property_trackers['integration'] = integration
        
        # Calculate global consciousness level
        self.global_consciousness_level = (
            0.2 * harmony +
            0.2 * complexity +
            0.2 * self_organization +
            0.2 * adaptability +
            0.2 * integration
        )
    
    def _detect_emergent_patterns(self):
        """Detect emergent patterns in the quantum state and tension field"""
        # Check if consciousness level exceeds threshold
        if self.global_consciousness_level < self.consciousness_threshold:
            return False
            
        # Get network state
        network_state = self.cube.extract_network_state()
        
        # Extract high tension points
        high_tension_points = network_state['high_tension_points']
        
        if len(high_tension_points) < 5:
            return False
            
        # Analyze tension field for patterns
        positions = np.array([point['position'] for point in high_tension_points])
        tensions = np.array([point['tension'] for point in high_tension_points])
        
        # Check for geometric patterns (clusters, lines, planes)
        # For simplicity, we'll just check for clusters
        
        # Use KMeans to find clusters
        from sklearn.cluster import KMeans
        if len(positions) >= 8:  # Need reasonable number of points
            # Determine optimal number of clusters using silhouette score
            from sklearn.metrics import silhouette_score
            
            max_clusters = min(8, len(positions) // 2)
            best_score = -1
            best_n_clusters = 2
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(positions)
                
                if len(set(cluster_labels)) > 1:  # Ensure multiple clusters
                    score = silhouette_score(positions, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            
            # Use best number of clusters
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(positions)
            
            # Check if clusters are well-formed
            if best_score > 0.5:  # Good clustering quality
                # Found a significant pattern
                pattern = {
                    'type': 'spatial_clustering',
                    'score': best_score,
                    'n_clusters': best_n_clusters,
                    'centers': kmeans.cluster_centers_.tolist(),
                    'time_step': self.simulation_stats['time_steps'],
                    'consciousness_level': self.global_consciousness_level
                }
                
                # Check if similar to existing patterns
                is_new_pattern = True
                for existing in self.memory_patterns:
                    if existing['type'] == 'spatial_clustering':
                        # Calculate similarity (Jaccard similarity of cluster centers)
                        existing_centers = np.array(existing['centers'])
                        new_centers = np.array(pattern['centers'])
                        
                        # Calculate distances between all pairs of centers
                        min_dists = []
                        for ec in existing_centers:
                            dists = np.linalg.norm(new_centers - ec.reshape(1, -1), axis=1)
                            min_dists.append(np.min(dists))
                        
                        similarity = np.mean([d < 0.2 for d in min_dists])  # 0.2 is distance threshold
                        
                        if similarity > self.pattern_recognition_threshold:
                            is_new_pattern = False
                            break
                
                if is_new_pattern:
                    self.memory_patterns.append(pattern)
                    self.simulation_stats['emergent_events'] += 1
                    
                    # Boost consciousness level when new pattern discovered
                    self.global_consciousness_level = min(1.0, self.global_consciousness_level * 1.2)
                    
                    return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the conscious cube system"""
        # Get network state from cube
        network_state = self.cube.extract_network_state()
        
        # Add consciousness metrics
        consciousness_state = {
            'global_level': self.global_consciousness_level,
            'emergent_properties': self.emergent_property_trackers,
            'memory_patterns': len(self.memory_patterns),
            'evolution_cycles': self.simulation_stats['evolution_cycles'],
            'time_steps': self.simulation_stats['time_steps'],
            'emergent_events': self.simulation_stats['emergent_events']
        }
        
        # Combine with network state
        return {
            'network': network_state,
            'consciousness': consciousness_state,
            'stats': {
                'node_count': len(self.nodes),
                'energy_level': sum(node['properties'].get('energy', 0) for node in self.nodes.values()),
                'avg_performance': np.mean([node['performance'] for node in self.nodes.values()]) if self.nodes else 0,
                'consciousness_history': self.simulation_stats['consciousness_history'][-100:] if len(self.simulation_stats['consciousness_history']) > 0 else []
            }
        }


# ======================================================================
# VISUALIZATION AND INTERFACE UTILITIES
# ======================================================================

def visualize_cube_2d(cube: QuantumStringCube, plot_size: int = 10):
    """Create a 2D slice visualization of the quantum cube"""
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(plot_size, plot_size))
    
    # Extract a 2D slice of the tension field
    if cube.dimensions >= 2:
        # Create slice from the first two dimensions at the center of other dimensions
        slice_indices = [cube.resolution // 2] * cube.dimensions
        slice_indices[0] = slice(None)
        slice_indices[1] = slice(None)
        
        # Extract the slice
        tension_slice = cube.tensor_field.tension_field[tuple(slice_indices)]
        
        # Plot the tension field
        plt.imshow(tension_slice.T, cmap='viridis', origin='lower', extent=[-1, 1, -1, 1])
        plt.colorbar(label='Tension')
        
        # Plot nodes
        if cube.nodes:
            # Extract node positions
            positions = []
            colors = []
            sizes = []
            
            for node_id, node_data in cube.nodes.items():
                pos = cube.node_positions[node_id].coordinates
                # Only include nodes that are near this slice
                if cube.dimensions <= 2 or all(abs(pos[d] - (slice_indices[d] / cube.resolution * 2 - 1)) < 0.2 
                                             for d in range(2, cube.dimensions)):
                    positions.append(pos[:2])  # Take first two coordinates
                    
                    # Color by energy
                    energy = node_data['state']['energy']
                    colors.append([1.0 - energy, 0.0, energy])
                    
                    # Size by connections
                    conn_count = len(cube.node_connections[node_id])
                    sizes.append(20 + 5 * conn_count)
            
            # Plot nodes
            if positions:
                positions = np.array(positions)
                plt.scatter(positions[:, 0], positions[:, 1], c=colors, s=sizes, alpha=0.7, edgecolors='white')
                
                # Plot connections
                for node_id, connections in cube.node_connections.items():
                    if node_id in cube.node_positions:
                        pos1 = cube.node_positions[node_id].coordinates
                        # Skip nodes not in this slice
                        if cube.dimensions > 2 and any(abs(pos1[d] - (slice_indices[d] / cube.resolution * 2 - 1)) >= 0.2 
                                                     for d in range(2, cube.dimensions)):
                            continue
                            
                        for connected_id in connections:
                            if connected_id in cube.node_positions:
                                pos2 = cube.node_positions[connected_id].coordinates
                                # Skip nodes not in this slice
                                if cube.dimensions > 2 and any(abs(pos2[d] - (slice_indices[d] / cube.resolution * 2 - 1)) >= 0.2 
                                                     for d in range(2, cube.dimensions)):
                                    continue
                                    
                                # Draw the connection
                                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'w-', alpha=0.3)
        
        # Plot high tension points
        high_tension_points = []
        for point, tension in cube.tensor_field.resonance_points:
            # Skip points not in this slice
            if cube.dimensions > 2 and any(point[d] != slice_indices[d] for d in range(2, cube.dimensions)):
                continue
                
            # Convert to continuous coordinates
            x = 2 * (point[0] / (cube.resolution - 1)) - 1
            y = 2 * (point[1] / (cube.resolution - 1)) - 1
            
            high_tension_points.append((x, y, tension))
        
        # Plot high tension points if any
        if high_tension_points:
            htp_x, htp_y, htp_t = zip(*high_tension_points)
            plt.scatter(htp_x, htp_y, c='red', s=50, alpha=0.7, marker='*')
        
        # Add title and labels
        plt.title('Quantum String Cube: 2D Slice Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(alpha=0.3)
    
    else:
        plt.text(0.5, 0.5, "Cannot create 2D slice - cube has insufficient dimensions", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_cube_3d(cube: QuantumStringCube, plot_size: int = 10):
    """Create a 3D visualization of the quantum cube"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure
    fig = plt.figure(figsize=(plot_size, plot_size))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract high tension points for visualization
    high_tension_points = []
    for point, tension in cube.tensor_field.resonance_points:
        if cube.dimensions >= 3:
            # Extract the first three dimensions
            x = 2 * (point[0] / (cube.resolution - 1)) - 1
            y = 2 * (point[1] / (cube.resolution - 1)) - 1
            z = 2 * (point[2] / (cube.resolution - 1)) - 1
            
            high_tension_points.append((x, y, z, tension))
    
    # Plot high tension points if any
    if high_tension_points:
        htp_x, htp_y, htp_z, htp_t = zip(*high_tension_points)
        # Normalize tension values for coloring
        max_tension = max(htp_t)
        norm_tension = [t/max_tension for t in htp_t]
        
        scatter = ax.scatter(htp_x, htp_y, htp_z, c=norm_tension, s=30+80*np.array(norm_tension), 
                            cmap='plasma', alpha=0.6)
        plt.colorbar(scatter, label='Normalized Tension')
    
    # Plot nodes
    if cube.nodes:
        # Extract node positions
        positions = []
        colors = []
        sizes = []
        node_ids = []
        
        for node_id, node_data in cube.nodes.items():
            pos = cube.node_positions[node_id].coordinates
            if cube.dimensions >= 3:
                positions.append(pos[:3])  # Take first three coordinates
                
                # Color by energy
                energy = node_data['state']['energy']
                colors.append([1.0 - energy, 0.0, energy])
                
                # Size by stability
                stability = node_data['state'].get('stability', 0.5)
                sizes.append(30 + 30 * stability)
                
                node_ids.append(node_id)
        
        # Plot nodes
        if positions:
            positions = np.array(positions)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c=colors, s=sizes, alpha=0.8, edgecolors='white')
            
            # Plot connections
            for i, node_id in enumerate(node_ids):
                for connected_id in cube.node_connections[node_id]:
                    if connected_id in cube.node_positions and cube.dimensions >= 3:
                        pos1 = positions[i]
                        idx2 = node_ids.index(connected_id) if connected_id in node_ids else -1
                        
                        if idx2 >= 0:
                            pos2 = positions[idx2]
                            
                            # Draw the connection
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], 
                                   'w-', alpha=0.3)
    
    # Add title and labels
    ax.set_title('Quantum String Cube: 3D Visualization')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    plt.tight_layout()
    return fig

def visualize_network_graph(cube: ConsciousCubeInterface, plot_size: int = 10):
    """Create a network graph visualization of the node connections"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Create figure
    plt.figure(figsize=(plot_size, plot_size))
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node_data in cube.nodes.items():
        # Get node properties
        performance = node_data['performance']
        energy = node_data['properties'].get('energy', 0.5)
        
        # Add node with attributes
        G.add_node(node_id, performance=performance, energy=energy)
    
    # Add edges
    for node_id, node_data in cube.nodes.items():
        for connected_id in node_data['connections']:
            if connected_id in cube.nodes:
                G.add_edge(node_id, connected_id)
    
    # Calculate node positions using force-directed layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node colors and sizes
    node_colors = [G.nodes[n]['energy'] for n in G.nodes()]
    node_sizes = [300 + 200 * G.nodes[n]['performance'] for n in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          cmap=plt.cm.viridis, alpha=0.8, edgecolors='white')
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.0)
    
    # Add node labels for highly-connected nodes
    important_nodes = {n: i for i, n in enumerate(G.nodes()) if G.degree(n) > 2}
    nx.draw_networkx_labels(G, pos, labels={n: str(i) for n, i in important_nodes.items()}, 
                           font_size=10, font_color='white')
    
    # Add title
    plt.title('Conscious Cube: Node Network Graph')
    plt.axis('off')
    
    return plt.gcf()

def plot_consciousness_evolution(cube: ConsciousCubeInterface, plot_size: int = 10):
    """Plot the evolution of consciousness metrics over time"""
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(plot_size, plot_size), sharex=True)
    
    # Extract data from history
    consciousness_history = cube.simulation_stats['consciousness_history']
    if not consciousness_history:
        axs[0].text(0.5, 0.5, "No consciousness history data available", 
                   horizontalalignment='center', verticalalignment='center')
        axs[1].text(0.5, 0.5, "No energy history data available", 
                   horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Time steps for x-axis
    time_steps = list(range(len(consciousness_history)))
    
    # Plot global consciousness level
    axs[0].plot(time_steps, consciousness_history, 'b-', linewidth=2)
    axs[0].set_ylabel('Consciousness Level')
    axs[0].set_title('Evolution of Global Consciousness')
    axs[0].grid(alpha=0.3)
    
    # Plot emergent properties
    # Extract the latest emergent property values
    if cube.emergent_property_trackers:
        # Create a bar chart of the current emergent properties
        properties = list(cube.emergent_property_trackers.keys())
        values = list(cube.emergent_property_trackers.values())
        
        axs[1].bar(properties, values, color='teal', alpha=0.7)
        axs[1].set_ylabel('Property Value')
        axs[1].set_title('Current Emergent Properties')
        axs[1].set_xticklabels(properties, rotation=45, ha='right')
        axs[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def serialize_cube_state(cube: QuantumStringCube) -> Dict[str, Any]:
    """Serialize cube state for storage or transmission"""
    # Basic metadata
    state = {
        "metadata": {
            "dimensions": cube.dimensions,
            "resolution": cube.resolution,
            "qubit_depth": cube.qubit_depth,
            "timestamp": time.time(),
            "node_count": len(cube.nodes),
            "connection_count": sum(len(connections) for connections in cube.node_connections.values()) // 2
        },
        "nodes": {},
        "connections": [],
        "field_state": {
            "energy_density": cube.tensor_field.get_energy_density(),
            "resonance_distribution": cube.tensor_field.get_resonance_distribution()
        }
    }
    
    # Serialize nodes
    for node_id, node_data in cube.nodes.items():
        position = cube.node_positions[node_id].coordinates.tolist() if node_id in cube.node_positions else []
        state["nodes"][node_id] = {
            "position": position,
            "properties": node_data['properties'],
            "state": node_data['state']
        }
    
    # Serialize connections
    for node1_id, connections in cube.node_connections.items():
        for node2_id in connections:
            if node1_id < node2_id:  # Avoid duplicates
                strength = cube.nodes[node1_id]['properties'].get(f'connection_{node2_id}_strength', 1.0)
                state["connections"].append({
                    "node1": node1_id,
                    "node2": node2_id,
                    "strength": strength
                })
    
    # Serialize high tension points
    if cube.tensor_field.resonance_points:
        state["field_state"]["high_tension_points"] = [
            {
                "position": point,
                "tension": float(tension)
            }
            for point, tension in cube.tensor_field.resonance_points
        ]
    
    # Include cube's pattern memory
    if cube.pattern_memory:
        state["patterns"] = [{
            "time": pattern["time"],
            "type": pattern["type"],
            "strength": float(pattern["strength"]),
            "dimensions": pattern["dimensions"]
        } for pattern in cube.pattern_memory]
    
    return state

def deserialize_cube_state(state: Dict[str, Any]) -> QuantumStringCube:
    """Reconstruct cube from serialized state"""
    # Extract metadata
    metadata = state.get("metadata", {})
    dimensions = metadata.get("dimensions", 4)
    resolution = metadata.get("resolution", 64)
    qubit_depth = metadata.get("qubit_depth", 10)
    
    # Create new cube
    cube = QuantumStringCube(dimensions, resolution, qubit_depth)
    
    # Add nodes
    for node_id, node_data in state.get("nodes", {}).items():
        # Extract position
        position = np.array(node_data.get("position", np.zeros(dimensions)))
        
        # Add node to cube
        cube.add_node(position, node_data.get("properties", {}))
        
        # Restore node state
        if "state" in node_data and node_id in cube.nodes:
            cube.nodes[node_id]['state'] = node_data["state"]
    
    # Add connections
    for connection in state.get("connections", []):
        node1_id = connection.get("node1")
        node2_id = connection.get("node2")
        strength = connection.get("strength", 1.0)
        
        if node1_id in cube.nodes and node2_id in cube.nodes:
            cube.connect_nodes(node1_id, node2_id, strength)
    
    # Force field update
    cube.simulate_step(dt=0.01)
    
    return cube

# ======================================================================
# DEMO IMPLEMENTATION
# ======================================================================

def create_dynamic_cube(dimensions: int = 4, resolution: int = 32, nodes: int = 20) -> ConsciousCubeInterface:
    """Create and initialize a dynamic cube with nodes"""
    # Create the conscious cube interface
    cube = ConsciousCubeInterface(dimensions, resolution, qubit_depth=8)
    
    # Add initial nodes
    for _ in range(nodes):
        # Random position
        position = np.random.rand(dimensions) * 2 - 1  # Range [-1, 1]
        
        # Random properties
        properties = {
            'energy': np.random.uniform(0.4, 0.8),
            'stability': np.random.uniform(0.5, 0.9),
            'phase': np.random.uniform(0, 2 * np.pi)
        }
        
        # Add node
        cube.add_node(properties, position)
    
    # Connect nodes
    cube.auto_connect_nodes(max_connections_per_node=4, connection_radius=0.6)
    
    # Run several simulation steps to stabilize
    for _ in range(10):
        cube.simulate_step()
    
    return cube

def run_simulation(cube: ConsciousCubeInterface, steps: int = 100, evolve_interval: int = 20):
    """Run a simulation for the specified number of steps"""
    # Store metrics
    history = {
        'consciousness': [],
        'nodes': [],
        'energy': [],
        'emergent_events': []
    }
    
    # Run simulation
    for step in range(steps):
        # Simulate one step
        cube.simulate_step()
        
        # Store metrics
        state = cube.get_state()
        history['consciousness'].append(state['consciousness']['global_level'])
        history['nodes'].append(len(cube.nodes))
        history['energy'].append(state['stats']['energy_level'])
        history['emergent_events'].append(state['consciousness']['emergent_events'])
        
        # Evolve periodically
        if step > 0 and step % evolve_interval == 0:
            cube.evolve_nodes()
    
    return history

def demonstrate_cube():
    """Create and demonstrate a working dynamic cube"""
    import matplotlib.pyplot as plt
    
    # Create cube
    print("Initializing Dynamic Cube...")
    cube = create_dynamic_cube(dimensions=4, resolution=32, nodes=20)
    
    # Run simulation
    print("Running simulation...")
    history = run_simulation(cube, steps=50, evolve_interval=10)
    
    # Print results
    print("\nSimulation complete!")
    print(f"Final consciousness level: {history['consciousness'][-1]:.4f}")
    print(f"Node count: {history['nodes'][-1]}")
    print(f"Emergent events: {history['emergent_events'][-1]}")
    
    # Visualize cube
    print("\nGenerating visualizations...")
    
    # 2D slice
    fig1 = visualize_cube_2d(cube.cube)
    fig1.savefig("cube_2d_visualization.png")
    
    # 3D visualization if cube has 3+ dimensions
    if cube.cube.dimensions >= 3:
        fig2 = visualize_cube_3d(cube.cube)
        fig2.savefig("cube_3d_visualization.png")
    
    # Network graph
    fig3 = visualize_network_graph(cube)
    fig3.savefig("cube_network_graph.png")
    
    # Consciousness evolution
    fig4 = plot_consciousness_evolution(cube)
    fig4.savefig("cube_consciousness_evolution.png")
    
    print("Visualizations saved!")
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(history['consciousness'])
    plt.title('Consciousness Level')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(history['energy'])
    plt.title('Total Energy')
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(history['emergent_events'])
    plt.title('Emergent Events')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("cube_metrics.png")
    
    print("Metrics plot saved!")
    
    # Save cube state
    state = serialize_cube_state(cube.cube)
    with open("cube_state.json", "w") as f:
        import json
        json.dump(state, f, indent=2)
    
    print("Cube state saved!")
    
    return cube

# If run directly, demonstrate the cube
if __name__ == "__main__":
    cube = demonstrate_cube()
"""
QuantumStringCube: A groundbreaking multi-dimensional cognitive framework
that implements a novel computational paradigm using quantum-inspired algorithms,
resonance harmonics, and adaptive evolutionary principles.
"""

import numpy as np
import hashlib
import time
import uuid
import math
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

# ======================================================================
# CORE QUANTUM-INSPIRED PRIMITIVES
# ======================================================================

class ResonanceMode(Enum):
    """Fundamental resonance modes for harmonic computation"""
    CONSTRUCTIVE = auto()   # Amplifying patterns
    DESTRUCTIVE = auto()    # Cancelling noise
    REFLECTIVE = auto()     # Preserving energy
    REFRACTIVE = auto()     # Transforming pathways
    ENTROPIC = auto()       # Dissipating obsolete patterns


class QuantumString:
    """
    Implementation of a novel quantum-inspired string theory construct
    that uses oscillatory patterns for computation rather than traditional bits.
    """
    def __init__(self, dimensions: int = 4, string_length: int = 64, tension: float = 0.8):
        self.dimensions = dimensions
        self.string_length = string_length
        self.base_tension = tension
        self.amplitude = np.zeros((string_length, dimensions), dtype=np.complex128)
        self.phase = np.zeros(string_length, dtype=np.float64)
        self.frequency = np.zeros(string_length, dtype=np.float64)
        self.harmonics = [1.0, 2.0, 3.0, 5.0, 8.0]  # Fibonacci-based harmonics
        
        # Initialize with quantum noise
        self._initialize_string()
    
    def _initialize_string(self):
        """Initialize the quantum string with structured noise"""
        # Create initial complex wave pattern
        for i in range(self.string_length):
            # Phase varies along the string
            self.phase[i] = (i / self.string_length) * 2 * np.pi
            
            # Different base frequency components
            self.frequency[i] = 1.0 + 0.5 * np.sin(self.phase[i])
            
            # Initialize amplitudes in each dimension
            for d in range(self.dimensions):
                # Create a complex amplitude with specific phase relationship
                phase_shift = 2 * np.pi * d / self.dimensions
                self.amplitude[i, d] = complex(
                    np.cos(self.phase[i] + phase_shift) * 0.1,
                    np.sin(self.phase[i] + phase_shift) * 0.1
                )
    
    def evolve(self, dt: float = 0.01):
        """Evolve the string state using wave equations"""
        # Apply wave equation dynamics (simplified form)
        new_amplitude = np.zeros_like(self.amplitude)
        
        # Implement a discretized version of the wave equation
        for i in range(1, self.string_length - 1):
            for d in range(self.dimensions):
                # Wave equation: ∂²y/∂t² = tension * ∂²y/∂x²
                d2y_dx2 = (self.amplitude[i+1, d] - 2*self.amplitude[i, d] + self.amplitude[i-1, d]) / (1/self.string_length)**2
                wave_term = self.base_tension * d2y_dx2
                
                # Add harmonic oscillator term
                harmonic_term = -self.frequency[i]**2 * self.amplitude[i, d]
                
                # Combine terms
                d2y_dt2 = wave_term + harmonic_term
                
                # Euler integration
                new_amplitude[i, d] = self.amplitude[i, d] + dt * d2y_dt2
        
        # Apply boundary conditions (fixed ends)
        new_amplitude[0] = 0
        new_amplitude[-1] = 0
        
        self.amplitude = new_amplitude
        
        # Evolve phase
        self.phase = (self.phase + dt * self.frequency) % (2 * np.pi)
    
    def apply_resonance(self, mode: ResonanceMode, target_points: List[int], strength: float = 0.1):
        """Apply a specific resonance pattern to selected points on the string"""
        if not target_points:
            return
            
        # Create the resonance pattern
        if mode == ResonanceMode.CONSTRUCTIVE:
            # Amplify existing patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.amplitude[point] *= (1 + strength)
        
        elif mode == ResonanceMode.DESTRUCTIVE:
            # Dampen existing patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.amplitude[point] *= (1 - strength)
        
        elif mode == ResonanceMode.REFLECTIVE:
            # Create mirror-like patterns
            for point in target_points:
                if 0 < point < self.string_length - 1:
                    self.amplitude[point] = self.amplitude[point-1]
        
        elif mode == ResonanceMode.REFRACTIVE:
            # Shift phase, bending the patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.phase[point] = (self.phase[point] + strength * np.pi) % (2 * np.pi)
        
        elif mode == ResonanceMode.ENTROPIC:
            # Introduce controlled chaos
            for point in target_points:
                if 0 <= point < self.string_length:
                    random_phase = np.random.random() * 2 * np.pi
                    decay_factor = 1 - strength
                    self.amplitude[point] = self.amplitude[point] * decay_factor + \
                                           complex(np.cos(random_phase), np.sin(random_phase)) * strength * 0.1
    
    def get_energy(self) -> float:
        """Calculate total string energy"""
        # Sum of squared amplitudes across all points and dimensions
        return np.sum(np.abs(self.amplitude)**2)
    
    def get_state_vector(self) -> np.ndarray:
        """Get flattened state representation for analysis"""
        real_parts = self.amplitude.real.flatten()
        imag_parts = self.amplitude.imag.flatten()
        phases = self.phase
        return np.concatenate([real_parts, imag_parts, phases])
    
    def apply_interference(self, other_string: 'QuantumString', coupling_strength: float = 0.1):
        """Create interference patterns between two strings"""
        # Ensure strings have compatible dimensions
        common_dims = min(self.dimensions, other_string.dimensions)
        common_length = min(self.string_length, other_string.string_length)
        
        # Apply interference in common dimensions
        for i in range(common_length):
            for d in range(common_dims):
                # Quantum interference formula
                self.amplitude[i, d] += coupling_strength * other_string.amplitude[i, d]
                
                # Normalize to prevent unbounded growth
                mag = abs(self.amplitude[i, d])
                if mag > 1.0:
                    self.amplitude[i, d] /= mag
    
    def encode_data(self, data: bytes, dimension: int = 0):
        """Encode binary data into string oscillations"""
        if dimension >= self.dimensions:
            dimension = 0
            
        # Convert bytes to bit pattern
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        
        # Scale to string length
        scale_factor = len(bits) / self.string_length
        
        # Encode bits as phase shifts
        for i in range(self.string_length):
            bit_index = int(i * scale_factor)
            if bit_index < len(bits):
                bit = bits[bit_index]
                if bit:
                    # Encode 1 as positive phase shift
                    phase_shift = 0.1 * np.pi
                else:
                    # Encode 0 as negative phase shift
                    phase_shift = -0.1 * np.pi
                
                # Apply phase shift to this dimension
                current_phase = np.angle(self.amplitude[i, dimension])
                magnitude = abs(self.amplitude[i, dimension])
                self.amplitude[i, dimension] = complex(
                    magnitude * np.cos(current_phase + phase_shift),
                    magnitude * np.sin(current_phase + phase_shift)
                )
    
    def extract_pattern(self, dimension: int = 0) -> List[float]:
        """Extract dominant pattern from a specific dimension"""
        if dimension >= self.dimensions:
            dimension = 0
            
        # Get amplitudes from specified dimension
        amplitudes = [abs(self.amplitude[i, dimension]) for i in range(self.string_length)]
        
        # Get phases from specified dimension
        phases = [np.angle(self.amplitude[i, dimension]) for i in range(self.string_length)]
        
        # Combine into pattern (amplitude * phase_factor)
        pattern = [amplitudes[i] * np.cos(phases[i]) for i in range(self.string_length)]
        
        return pattern


class HarmonicTensorField:
    """
    A novel tensor field representation using harmonics theory
    for efficient multi-dimensional information propagation.
    """
    def __init__(self, dimensions: int = 4, resolution: int = 32):
        self.dimensions = dimensions
        self.resolution = resolution
        
        # Initialize tensor field
        self.field_shape = tuple([resolution] * dimensions)
        self.amplitude_tensor = np.zeros(self.field_shape, dtype=np.complex128)
        self.frequency_tensor = np.zeros(self.field_shape, dtype=np.float64)
        self.phase_tensor = np.zeros(self.field_shape, dtype=np.float64)
        
        # Initialize with structured patterns
        self._initialize_field()
        
        # Tension metrics
        self.tension_field = np.zeros(self.field_shape, dtype=np.float64)
        self.resonance_points = []
    
    def _initialize_field(self):
        """Initialize the harmonic tensor field with structured patterns"""
        # Create coordinate arrays
        coords = [np.linspace(-1, 1, self.resolution) for _ in range(self.dimensions)]
        
        # Fill tensors with initial values
        # We'll use a multidimensional mesh approach
        if self.dimensions <= 3:  # For lower dimensions we can use numpy's meshgrid
            mesh_coords = np.meshgrid(*coords, indexing='ij')
            
            # Calculate distance from origin for each point
            r_squared = np.zeros(self.field_shape)
            for dim_coords in mesh_coords:
                r_squared += dim_coords**2
            r = np.sqrt(r_squared)
            
            # Set initial values based on distance
            self.amplitude_tensor = 0.1 * np.exp(-3 * r) * np.exp(1j * np.pi * r)
            self.frequency_tensor = 1.0 + 0.2 * r
            self.phase_tensor = np.pi * r
            
        else:  # For higher dimensions, we'll use a loop approach
            # Create indices for all positions
            indices = np.indices(self.field_shape)
            
            # Convert indices to coordinates [-1, 1]
            coords_from_indices = [(2.0 * indices[d] / (self.resolution - 1) - 1.0) for d in range(self.dimensions)]
            
            # Calculate distance from origin for each point
            r_squared = np.zeros(self.field_shape)
            for dim_coords in coords_from_indices:
                r_squared += dim_coords**2
            r = np.sqrt(r_squared)
            
            # Set initial values
            self.amplitude_tensor = 0.1 * np.exp(-3 * r) * np.exp(1j * np.pi * r)
            self.frequency_tensor = 1.0 + 0.2 * r
            self.phase_tensor = np.pi * r
    
    def evolve(self, dt: float = 0.01):
        """Evolve the field using a quantum-inspired wave equation"""
        # Create new tensor for updated values
        new_amplitude = np.zeros_like(self.amplitude_tensor, dtype=np.complex128)
        
        # Define the laplacian kernel for our dimensions
        # We'll use a simple central difference approximation
        
        # Loop through all points excluding boundaries
        # For high dimensions, we need a different approach than nested loops
        
        # First create slices for the center point and its neighbors
        center_slice = tuple(slice(1, self.resolution-1) for _ in range(self.dimensions))
        
        # Apply discretized wave equation at all interior points at once
        new_amplitude[center_slice] = self.amplitude_tensor[center_slice]
        
        # For each dimension, add the second derivative contribution
        for d in range(self.dimensions):
            # Create slices for the forward and backward points in this dimension
            forward_slice = list(center_slice)
            forward_slice[d] = slice(2, self.resolution)
            forward_slice = tuple(forward_slice)
            
            backward_slice = list(center_slice)
            backward_slice[d] = slice(0, self.resolution-2)
            backward_slice = tuple(backward_slice)
            
            # Add the second derivative term from this dimension
            d2y_dx2 = (self.amplitude_tensor[forward_slice] - 
                       2 * self.amplitude_tensor[center_slice] + 
                       self.amplitude_tensor[backward_slice])
            
            # Accumulate the laplacian contributions
            new_amplitude[center_slice] += 0.1 * d2y_dx2
        
        # Apply frequency-based oscillation
        oscillation_factor = np.exp(1j * dt * self.frequency_tensor[center_slice])
        new_amplitude[center_slice] *= oscillation_factor
        
        # Update the amplitude tensor (interior points only)
        self.amplitude_tensor[center_slice] = new_amplitude[center_slice]
        
        # Evolve phases
        self.phase_tensor = (self.phase_tensor + dt * self.frequency_tensor) % (2 * np.pi)
        
        # Update tension field
        self._calculate_tension_field()
    
    def _calculate_tension_field(self):
        """Calculate the tension field based on amplitude gradients"""
        # Initialize tension field
        self.tension_field = np.zeros(self.field_shape, dtype=np.float64)
        
        # For each dimension, calculate the gradient
        for d in range(self.dimensions):
            # Create slices for calculating gradient
            forward_slice = list(slice(None) for _ in range(self.dimensions))
            forward_slice[d] = slice(1, None)
            forward_slice = tuple(forward_slice)
            
            backward_slice = list(slice(None) for _ in range(self.dimensions))
            backward_slice[d] = slice(0, -1)
            backward_slice = tuple(backward_slice)
            
            # Calculate gradient magnitude in this dimension
            gradient = np.abs(self.amplitude_tensor[forward_slice] - self.amplitude_tensor[backward_slice])
            
            # Add to the tension field (on the appropriate slice)
            gradient_slice = list(slice(None) for _ in range(self.dimensions))
            gradient_slice[d] = slice(0, -1)  # Match the size reduction from gradient calculation
            self.tension_field[tuple(gradient_slice)] += gradient
        
        # Normalize tension field
        max_tension = np.max(self.tension_field)
        if max_tension > 0:
            self.tension_field /= max_tension
        
        # Find resonance points (high tension points)
        self._find_resonance_points()
    
    def _find_resonance_points(self, threshold: float = 0.7):
        """Find points with high tension (resonance points)"""
        # Clear previous resonance points
        self.resonance_points = []
        
        # Find indices where tension exceeds threshold
        high_tension_indices = np.where(self.tension_field > threshold)
        
        # Convert to list of coordinate tuples
        for i in range(len(high_tension_indices[0])):
            point = tuple(high_tension_indices[d][i] for d in range(self.dimensions))
            tension = self.tension_field[point]
            self.resonance_points.append((point, tension))
    
    def inject_pattern(self, pattern: np.ndarray, position: Tuple[int, ...], dimension_mapping: Optional[List[int]] = None):
        """Inject a pattern into the field at a specific position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
            
        pattern_shape = pattern.shape
        
        # If dimension mapping is not provided, use default mapping
        if dimension_mapping is None:
            dimension_mapping = list(range(min(len(pattern_shape), self.dimensions)))
            
        # Create slices for the target region
        target_slices = []
        for d in range(self.dimensions):
            # If this dimension is mapped from the pattern
            if d in dimension_mapping:
                # Find the corresponding dimension in the pattern
                pattern_dim = dimension_mapping.index(d)
                dim_size = pattern_shape[pattern_dim]
                
                # Calculate start and end indices
                start = position[d]
                end = min(start + dim_size, self.resolution)
                
                target_slices.append(slice(start, end))
            else:
                # Not mapped, use single position
                target_slices.append(slice(position[d], position[d] + 1))
        
        # Create slices for the source region
        source_slices = []
        for d in range(len(pattern_shape)):
            # Calculate how much of the pattern fits
            if d < len(dimension_mapping):
                mapped_dim = dimension_mapping[d]
                target_size = target_slices[mapped_dim].stop - target_slices[mapped_dim].start
                source_slices.append(slice(0, target_size))
            else:
                source_slices.append(slice(None))
        
        # Inject the pattern
        # Need to handle complex target indexing
        target_region = self.amplitude_tensor[tuple(target_slices)]
        source_region = pattern[tuple(source_slices)]
        
        # Match dimensions through broadcasting if needed
        target_shape = target_region.shape
        source_shape = source_region.shape
        
        if target_shape != source_shape:
            # Create a new array that will match the target shape
            expanded_source = np.zeros(target_shape, dtype=source_region.dtype)
            
            # Find the overlap shape
            overlap_shape = tuple(min(t, s) for t, s in zip(target_shape, source_shape))
            
            # Create slices for the overlap region
            overlap_slices = tuple(slice(0, s) for s in overlap_shape)
            
            # Copy the source into the expanded array
            expanded_source[overlap_slices] = source_region[overlap_slices]
            source_region = expanded_source
        
        # Inject the pattern
        self.amplitude_tensor[tuple(target_slices)] = source_region
    
    def extract_pattern(self, position: Tuple[int, ...], dimensions: List[int], size: List[int]) -> np.ndarray:
        """Extract a pattern from specific dimensions at a position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
        
        # Create slices for the region to extract
        extract_slices = []
        for d in range(self.dimensions):
            if d in dimensions:
                # This dimension will be included in the output
                idx = dimensions.index(d)
                extract_size = size[idx] if idx < len(size) else 1
                
                # Calculate start and end indices
                start = position[d]
                end = min(start + extract_size, self.resolution)
                
                extract_slices.append(slice(start, end))
            else:
                # Not included, use single position
                extract_slices.append(slice(position[d], position[d] + 1))
        
        # Extract the pattern
        pattern = self.amplitude_tensor[tuple(extract_slices)]
        
        # Squeeze out dimensions that are size 1 (from non-included dimensions)
        pattern = np.squeeze(pattern)
        
        return pattern
    
    def apply_resonance(self, position: Tuple[int, ...], radius: int, strength: float = 0.1, mode: ResonanceMode = ResonanceMode.CONSTRUCTIVE):
        """Apply a resonance pattern centered at a position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
            
        # Calculate indices within the radius
        # For high dimensions, we need an efficient approach
        
        # Create ranges for each dimension
        dim_ranges = []
        for d in range(self.dimensions):
            start = max(0, position[d] - radius)
            end = min(self.resolution, position[d] + radius + 1)
            dim_ranges.append(range(start, end))
        
        # Generate all combinations of indices within the hypersphere
        from itertools import product
        points_in_range = product(*dim_ranges)
        
        # Filter points by distance
        points_in_sphere = []
        for point in points_in_range:
            # Calculate squared distance
            squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
            if squared_dist <= radius**2:
                points_in_sphere.append(point)
        
        # Apply resonance based on mode
        if mode == ResonanceMode.CONSTRUCTIVE:
            # Amplify existing patterns
            for point in points_in_sphere:
                self.amplitude_tensor[point] *= (1 + strength)
                
        elif mode == ResonanceMode.DESTRUCTIVE:
            # Dampen existing patterns
            for point in points_in_sphere:
                self.amplitude_tensor[point] *= (1 - strength)
                
        elif mode == ResonanceMode.REFLECTIVE:
            # Create mirror-like patterns relative to center
            center_amplitude = self.amplitude_tensor[position]
            for point in points_in_sphere:
                # Vector from center to point
                vector = tuple(point[d] - position[d] for d in range(self.dimensions))
                # Calculate reflection factor based on distance
                dist = math.sqrt(sum(v**2 for v in vector))
                if dist > 0:
                    reflection_factor = 1 - dist / (radius + 1)
                    self.amplitude_tensor[point] = center_amplitude * reflection_factor + self.amplitude_tensor[point] * (1 - reflection_factor)
                    
        elif mode == ResonanceMode.REFRACTIVE:
            # Bend wave patterns 
            for point in points_in_sphere:
                # Calculate distance from center
                squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
                dist = math.sqrt(squared_dist)
                # Phase shift proportional to distance
                phase_shift = strength * (1 - dist / radius) * np.pi if radius > 0 else 0
                # Apply phase shift
                current_phase = np.angle(self.amplitude_tensor[point])
                magnitude = abs(self.amplitude_tensor[point])
                self.amplitude_tensor[point] = magnitude * np.exp(1j * (current_phase + phase_shift))
                
        elif mode == ResonanceMode.ENTROPIC:
            # Introduce controlled chaos
            for point in points_in_sphere:
                # Calculate distance factor
                squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
                dist = math.sqrt(squared_dist)
                dist_factor = 1 - dist / radius if radius > 0 else 0
                
                # Add random component
                if dist_factor > 0:
                    random_phase = np.random.random() * 2 * np.pi
                    entropy_strength = strength * dist_factor
                    decay_factor = 1 - entropy_strength
                    self.amplitude_tensor[point] = (self.amplitude_tensor[point] * decay_factor + 
                                                  complex(np.cos(random_phase), np.sin(random_phase)) * entropy_strength * 0.1)
    
    def get_energy_density(self) -> float:
        """Calculate total energy density in the field"""
        # Sum of squared amplitudes across all points
        return np.sum(np.abs(self.amplitude_tensor)**2) / np.prod(self.field_shape)
    
    def get_resonance_distribution(self) -> Dict[str, Any]:
        """Get statistical distribution of resonance points"""
        result = {
            "count": len(self.resonance_points),
            "average_tension": 0,
            "spatial_distribution": [0] * self.dimensions,
            "max_tension": 0,
            "clusters": 0
        }
        
        if self.resonance_points:
            # Calculate average tension
            tensions = [tension for _, tension in self.resonance_points]
            result["average_tension"] = sum(tensions) / len(tensions)
            result["max_tension"] = max(tensions)
            
            # Calculate spatial distribution
            points = [point for point, _ in self.resonance_points]
            for d in range(self.dimensions):
                d_coords = [p[d] for p in points]
                if d_coords:
                    mean = sum(d_coords) / len(d_coords)
                    # Calculate normalized standard deviation
                    std = math.sqrt(sum((x - mean)**2 for x in d_coords) / len(d_coords))
                    result["spatial_distribution"][d] = std / self.resolution
            
            # Estimate number of clusters (very simple approach)
            # In a real implementation, we would use a clustering algorithm
            # But for this example, we'll just use a simple heuristic
            if len(points) >= 2:
                # Calculate average distance between points
                total_dist = 0
                count = 0
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        # Calculate squared distance
                        squared_dist = sum((points[i][d] - points[j][d])**2 for d in range(self.dimensions))
                        total_dist += math.sqrt(squared_dist)
                        count += 1
                
                if count > 0:
                    avg_dist = total_dist / count
                    # Estimate clusters based on average distance
                    cluster_threshold = self.resolution / 5  # Heuristic
                    if avg_dist < cluster_threshold:
                        result["clusters"] = max(1, int(len(points) * cluster_threshold / avg_dist / 10))
                    else:
                        result["clusters"] = len(points)
        
        return result


# ======================================================================
# EVOLUTION & ADAPTATION CORE
# ======================================================================

@dataclass
class EvolvingNodeDNA:
    """
    DNA structure for nodes that can evolve through generations,
    mimicking biological evolution for computational structures.
    """
    traits: Dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 0.05
    crossover_points: int = 2
    generation: int = 0
    
    def __post_init__(self):
        """Initialize with default traits if not provided"""
        if not self.traits:
            self.traits = {
                # Core traits
                'energy_transfer': np.random.normal(0.7, 0.1),  # Efficiency at transferring energy
                'tension_response': np.random.normal(0.6, 0.2),  # How it responds to tension fields
                'phase_coherence': np.random.normal(0.5, 0.15),  # Ability to maintain phase with others
                'quantum_effect': np.random.normal(0.3, 0.2),  # Influence on quantum probabilities
                
                # Learning traits
                'pattern_recognition': np.random.normal(0.5, 0.2),  # Ability to recognize patterns
                'memory_persistence': np.random.normal(0.6, 0.15),  # How long it retains memory
                'adaptability': np.random.normal(0.5, 0.25),  # How quickly it adapts to changes
                
                # Social traits
                'connection_affinity': np.random.normal(0.5, 0.2),  # Tendency to form connections
                'information_sharing': np.random.normal(0.6, 0.2),  # Willingness to share information
                'specialization': np.random.normal(0.4, 0.3),  # Tendency to specialize vs. generalize
            }
            
            # Normalize traits to reasonable ranges
            for key in self.traits:
                self.traits[key] = max(0.1, min(0.9, self.traits[key]))
    
    def mutate(self):
        """Apply random mutations to traits"""
        for trait in self.traits:
            # Randomly mutate with probability based on mutation rate
            if np.random.random() < self.mutation_rate:
                # Apply random change
                mutation_scale = 0.1  # Scale of mutations
                change = np.random.normal(0, mutation_scale)
                self.traits[trait] = max(0.1, min(0.9, self.traits[trait] + change))
        
        # Occasionally introduce completely new trait values
        if np.random.random() < self.mutation_rate / 2:
            # Select random trait to reset
            trait = np.random.choice(list(self.traits.keys()))
            self.traits[trait] = np.random.uniform(0.1, 0.9)
        
        # Increment generation
        self.generation += 1
    
    def crossover(self, other: 'EvolvingNodeDNA') -> 'EvolvingNodeDNA':
        """Perform genetic crossover with another DNA"""
        # Create child DNA
        child = EvolvingNodeDNA()
        child.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2
        child.crossover_points = max(self.crossover_points, other.crossover_points)
        child.generation = max(self.generation, other.generation) + 1
        
        # Get all traits (ensure both parents have the same traits)
        all_traits = list(set(list(self.traits.keys()) + list(other.traits.keys())))
        all_traits.sort()  # Ensure consistent ordering
        
        # Determine crossover points
        if len(all_traits) <= 1:
            crossover_indices = []
        else:
            crossover_indices = sorted(np.random.choice(
                range(1, len(all_traits)), 
                size=min(self.crossover_points, len(all_traits) - 1),
                replace=False
            ))
        
        # Initialize with first parent's traits
        current_parent = 0  # 0 = self, 1 = other
        parent_dnas = [self, other]
        
                    # Apply crossover
        for i, trait in enumerate(all_traits):
            # Switch parents at crossover points
            if i in crossover_indices:
                current_parent = 1 - current_parent
            
            # Get trait from current parent
            parent = parent_dnas[current_parent]
            if trait in parent.traits:
                child.traits[trait] = parent.traits[trait]
            else:
                # If trait is missing in this parent, take from other or generate new
                other_parent = parent_dnas[1 - current_parent]
                if trait in other_parent.traits:
                    child.traits[trait] = other_parent.traits[trait]
                else:
                    child.traits[trait] = np.random.uniform(0.1, 0.9)
        
        # Apply light mutation
        if np.random.random() < 0.3:  # 30% chance of mutation after crossover
            child.mutation_rate = min(0.3, child.mutation_rate * (1.0 + np.random.uniform(-0.1, 0.1)))
            child.mutate()
            
        return child
    
    def genetic_similarity(self, other: 'EvolvingNodeDNA') -> float:
        """Calculate genetic similarity with another DNA (0.0 to 1.0)"""
        # Get all traits from both DNAs
        all_traits = set(list(self.traits.keys()) + list(other.traits.keys()))
        
        if not all_traits:
            return 0.0
            
        # Calculate trait value differences
        total_diff = 0.0
        for trait in all_traits:
            # Get trait values, default to 0.5 if missing
            self_value = self.traits.get(trait, 0.5)
            other_value = other.traits.get(trait, 0.5)
            
            # Add absolute difference
            total_diff += abs(self_value - other_value)
        
        # Calculate average difference and convert to similarity
        avg_diff = total_diff / len(all_traits)
        similarity = 1.0 - (avg_diff / 0.9)  # Normalize to 0.0-1.0 range
        
        return max(0.0, min(1.0, similarity))
    
    def get_trait_influence(self) -> Dict[str, float]:
        """Get the influence factors for each trait category"""
        # Categorize traits
        energy_traits = ['energy_transfer', 'tension_response']
        learning_traits = ['pattern_recognition', 'memory_persistence', 'adaptability']
        social_traits = ['connection_affinity', 'information_sharing', 'specialization']
        quantum_traits = ['quantum_effect', 'phase_coherence']
        
        # Calculate influences
        influences = {
            'energy_transfer': self._average_traits(energy_traits),
            'learning_capacity': self._average_traits(learning_traits),
            'social_aptitude': self._average_traits(social_traits),
            'quantum_effect': self._average_traits(quantum_traits),
            'tension_response': self.traits.get('tension_response', 0.5)
        }
        
        return influences
    
    def _average_traits(self, trait_list: List[str]) -> float:
        """Calculate average of multiple traits"""
        values = [self.traits.get(trait, 0.5) for trait in trait_list if trait in self.traits]
        if not values:
            return 0.5
        return sum(values) / len(values)
    
    def serialize(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            'traits': self.traits,
            'mutation_rate': self.mutation_rate,
            'crossover_points': self.crossover_points,
            'generation': self.generation
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'EvolvingNodeDNA':
        """Create from serialized dictionary"""
        return cls(
            traits=data.get('traits', {}),
            mutation_rate=data.get('mutation_rate', 0.05),
            crossover_points=data.get('crossover_points', 2),
            generation=data.get('generation', 0)
        )


# ======================================================================
# CORE QUANTUM STRING CUBE IMPLEMENTATION
# ======================================================================

class QuantumStringCube:
    """
    A revolutionary multidimensional quantum-inspired computational framework
    that uses string theory principles, tensor field harmonics, and evolutionary
    algorithms to create a cognitive computing substrate.
    """
    def __init__(self, dimensions: int = 4, resolution: int = 64, qubit_depth: int = 10):
        """
        Initialize the quantum string cube
        
        Args:
            dimensions: Number of spatial dimensions
            resolution: Resolution of each dimension
            qubit_depth: Depth of quantum state representation
        """
        self.dimensions = dimensions
        self.resolution = resolution
        self.qubit_depth = qubit_depth
        
        # Initialize quantum strings
        self.strings = [QuantumString(dimensions, string_length=resolution) 
                       for _ in range(qubit_depth)]
        
        # Initialize harmonic tensor field
        self.tensor_field = HarmonicTensorField(dimensions, resolution)
        
        # Node management
        self.nodes = {}  # id -> node data
        self.node_positions = {}  # id -> NodePosition
        self.node_connections = defaultdict(set)  # id -> set of connected node ids
        
        # Quantum state management
        self.global_phase = 0.0
        self.entanglement_matrix = np.eye(qubit_depth)  # Tracks string entanglement
        
        # Analytics
        self.resonance_history = []
        self.energy_history = []
        self.pattern_memory = []
        
        # Initialize random seed for reproducibility
        self.seed = int(time.time())
    
    def add_node(self, position: np.ndarray, properties: Dict[str, Any]) -> str:
        """
        Add a node to the quantum cube
        
        Args:
            position: Coordinates in cube space (normalized -1 to 1)
            properties: Node properties
            
        Returns:
            Node ID
        """
        # Generate unique node ID
        node_id = str(uuid.uuid4())
        
        # Ensure position has correct dimensions
        if len(position) != self.dimensions:
            position = np.zeros(self.dimensions)
            position[:min(len(position), self.dimensions)] = position[:min(len(position), self.dimensions)]
        
        # Initialize node position object
        node_pos = NodePosition(coordinates=position.copy())
        
        # Add phase from properties if available
        if 'phase' in properties:
            node_pos.phase = properties['phase']
        
        # Store node
        self.nodes[node_id] = {
            'id': node_id,
            'properties': properties.copy(),
            'creation_time': time.time(),
            'last_update_time': time.time(),
            'state': {
                'energy': properties.get('energy', 0.5),
                'stability': properties.get('stability', 0.8),
                'activity': 0.0,
                'resonance': 0.0
            }
        }
        
        # Store node position
        self.node_positions[node_id] = node_pos
        
        # Apply node influence to quantum field
        self._apply_node_to_field(node_id)
        
        return node_id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the quantum cube"""
        if node_id not in self.nodes:
            return False
            
        # Remove connections
        for connected_id in list(self.node_connections[node_id]):
            self.disconnect_nodes(node_id, connected_id)
        
        # Remove from dictionaries
        del self.nodes[node_id]
        del self.node_positions[node_id]
        del self.node_connections[node_id]
        
        return True
    
    def connect_nodes(self, node1_id: str, node2_id: str, strength: float = 1.0) -> bool:
        """Create a connection between two nodes"""
        if node1_id not in self.nodes or node2_id not in self.nodes or node1_id == node2_id:
            return False
            
        # Add to connections
        self.node_connections[node1_id].add(node2_id)
        self.node_connections[node2_id].add(node1_id)
        
        # Store connection strength in node properties
        self.nodes[node1_id]['properties'][f'connection_{node2_id}_strength'] = strength
        self.nodes[node2_id]['properties'][f'connection_{node1_id}_strength'] = strength
        
        # Create quantum entanglement between positions
        self._entangle_nodes(node1_id, node2_id, strength)
        
        return True
    
    def disconnect_nodes(self, node1_id: str, node2_id: str) -> bool:
        """Remove a connection between two nodes"""
        if node1_id not in self.nodes or node2_id not in self.nodes:
            return False
            
        # Remove from connections
        if node2_id in self.node_connections[node1_id]:
            self.node_connections[node1_id].remove(node2_id)
        if node1_id in self.node_connections[node2_id]:
            self.node_connections[node2_id].remove(node1_id)
        
        # Remove connection properties
        if f'connection_{node2_id}_strength' in self.nodes[node1_id]['properties']:
            del self.nodes[node1_id]['properties'][f'connection_{node2_id}_strength']
        if f'connection_{node1_id}_strength' in self.nodes[node2_id]['properties']:
            del self.nodes[node2_id]['properties'][f'connection_{node1_id}_strength']
        
        # Reduce quantum entanglement
        self._disentangle_nodes(node1_id, node2_id)
        
        return True
    
    def move_node(self, node_id: str, new_position: np.ndarray) -> bool:
        """Move a node to a new position"""
        if node_id not in self.nodes:
            return False
            
        # Ensure position has correct dimensions
        if len(new_position) != self.dimensions:
            new_position = np.zeros(self.dimensions)
            new_position[:min(len(new_position), self.dimensions)] = new_position[:min(len(new_position), self.dimensions)]
        
        # Update node position
        self.node_positions[node_id].coordinates = new_position.copy()
        
        # Update last update time
        self.nodes[node_id]['last_update_time'] = time.time()
        
        # Re-apply node influence to field at new position
        self._apply_node_to_field(node_id)
        
        return True
    
    def update_node_property(self, node_id: str, property_name: str, value: Any) -> bool:
        """Update a property of a node"""
        if node_id not in self.nodes:
            return False
            
        # Update property
        self.nodes[node_id]['properties'][property_name] = value
        
        # Update last update time
        self.nodes[node_id]['last_update_time'] = time.time()
        
        # Special handling for certain properties
        if property_name == 'energy':
            self.nodes[node_id]['state']['energy'] = value
            self._apply_node_to_field(node_id)  # Re-apply node influence
        elif property_name == 'stability':
            self.nodes[node_id]['state']['stability'] = value
        elif property_name == 'phase':
            self.node_positions[node_id].phase = value
        
        return True
    
    def simulate_step(self, dt: float = 0.1):
        """
        Perform a single simulation step
        
        Args:
            dt: Time step size
        """
        # 1. Update tensor field
        self.tensor_field.evolve(dt/2)  # First half-step
        
        # 2. Update quantum strings
        for string in self.strings:
            string.evolve(dt)
        
        # 3. Update entanglement between strings
        self._update_string_entanglement(dt)
        
        # 4. Update node physics
        self._update_node_positions(dt)
        
        # 5. Apply node influences to field
        for node_id in self.nodes:
            self._apply_node_to_field(node_id)
        
        # 6. Update tensor field again (second half-step)
        self.tensor_field.evolve(dt/2)
        
        # 7. Calculate node states based on current field
        self._calculate_node_states()
        
        # 8. Update global phase
        self.global_phase = (self.global_phase + dt * 0.1) % (2 * math.pi)
        
        # 9. Update analytics
        self._update_analytics()
    
    def _apply_node_to_field(self, node_id: str):
        """Apply node influence to tensor field"""
        if node_id not in self.nodes:
            return
            
        # Get node position and properties
        node_pos = self.node_positions[node_id]
        properties = self.nodes[node_id]['properties']
        state = self.nodes[node_id]['state']
        
        # Convert continuous coordinates to grid indices
        grid_position = self._continuous_to_grid(node_pos.coordinates)
        
        # Apply resonance based on node properties
        energy = state['energy']
        resonance_mode = ResonanceMode.CONSTRUCTIVE if energy > 0.5 else ResonanceMode.DESTRUCTIVE
        
        # Determine resonance radius based on energy
        radius = int(1 + 3 * energy)
        
        # Determine strength based on node activity
        strength = 0.1 + 0.2 * state.get('activity', 0.0)
        
        # Apply resonance to field
        self.tensor_field.apply_resonance(
            position=grid_position,
            radius=radius,
            strength=strength,
            mode=resonance_mode
        )
        
        # Encode node DNA pattern if available
        if 'dna_traits' in properties:
            dna_pattern = self._encode_dna_to_pattern(properties['dna_traits'])
            
            # Calculate a small offset from the node position
            offset = tuple(max(0, min(self.resolution-1, int(p) + np.random.randint(-1, 2))) 
                          for p in grid_position)
            
            # Inject the pattern into the field
            if dna_pattern is not None:
                try:
                    # Inject into a subset of dimensions
                    dim_subset = list(range(min(3, self.dimensions)))
                    self.tensor_field.inject_pattern(
                        pattern=dna_pattern,
                        position=offset,
                        dimension_mapping=dim_subset
                    )
                except Exception as e:
                    pass  # Handle any errors gracefully
    
    def _update_node_positions(self, dt: float):
        """Update node positions based on physics and quantum field"""
        # Calculate forces between nodes
        forces = self._calculate_node_forces()
        
        # Apply forces to node accelerations
        for node_id, force in forces.items():
            if node_id in self.node_positions:
                self.node_positions[node_id].acceleration = force
        
        # Update node positions
        for node_id, position in self.node_positions.items():
            # Get node state
            state = self.nodes[node_id]['state']
            
            # Scale velocity by node stability (more stable = slower movement)
            stability_factor = 1.0 - state.get('stability', 0.5)
            position.velocity *= (1.0 - 0.5 * stability_factor * dt)
            
            # Move the node
            position.move(dt)
            
            # Ensure position stays within bounds (-1 to 1)
            position.coordinates = np.clip(position.coordinates, -1.0, 1.0)
    
    def _calculate_node_forces(self) -> Dict[str, np.ndarray]:
        """Calculate forces acting on each node"""
        forces = {node_id: np.zeros(self.dimensions) for node_id in self.nodes}
        
        # 1. Connection forces (spring-like)
        for node1_id, connections in self.node_connections.items():
            for node2_id in connections:
                # Get positions
                pos1 = self.node_positions[node1_id].coordinates
                pos2 = self.node_positions[node2_id].coordinates
                
                # Calculate displacement vector
                displacement = pos2 - pos1
                distance = np.linalg.norm(displacement)
                
                # Skip if nodes are at the same position
                if distance < 1e-6:
                    continue
                
                # Calculate direction vector
                direction = displacement / distance
                
                # Get connection strength
                strength = self.nodes[node1_id]['properties'].get(f'connection_{node2_id}_strength', 1.0)
                
                # Calculate spring force (stronger connections = stronger spring)
                # Ideal distance is 0.3 units
                ideal_distance = 0.3
                spring_constant = 0.5 * strength
                spring_force = spring_constant * (distance - ideal_distance) * direction
                
                # Apply force to both nodes
                forces[node1_id] += spring_force
                forces[node2_id] -= spring_force
        
        # 2. Field forces (nodes are influenced by the tensor field)
        for node_id, position in self.node_positions.items():
            # Get grid position
            grid_pos = self._continuous_to_grid(position.coordinates)
            
            # Get local tension at this position
            tension = self._get_tensor_value_at_position(grid_pos)
            
            # Get tension gradient (approximated)
            gradient = np.zeros(self.dimensions)
            
            # Calculate gradient by sampling nearby points
            for d in range(self.dimensions):
                # Forward position
                forward_pos = list(grid_pos)
                if forward_pos[d] < self.resolution - 1:
                    forward_pos[d] += 1
                    forward_tension = self._get_tensor_value_at_position(tuple(forward_pos))
                    gradient[d] += forward_tension - tension
                
                # Backward position
                backward_pos = list(grid_pos)
                if backward_pos[d] > 0:
                    backward_pos[d] -= 1
                    backward_tension = self._get_tensor_value_at_position(tuple(backward_pos))
                    gradient[d] += tension - backward_tension
            
            # Normalize gradient
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 0:
                gradient /= gradient_norm
            
            # Apply field force - nodes are attracted to high tension areas
            field_force = gradient * 0.05
            forces[node_id] += field_force
        
        # 3. Add random force component (Brownian motion)
        for node_id in forces:
            random_force = np.random.normal(0, 0.01, self.dimensions)
            forces[node_id] += random_force
        
        return forces
    
    def _get_tensor_value_at_position(self, position: Tuple[int, ...]) -> float:
        """Get tensor field value at a grid position"""
        # Ensure position is within bounds
        position = tuple(max(0, min(self.resolution-1, p)) for p in position)
        
        try:
            # Get amplitude at this position
            amplitude = self.tensor_field.amplitude_tensor[position]
            # Return magnitude
            return abs(amplitude)
        except:
            # Handle out of bounds or other errors
            return 0.0
    
    def _calculate_node_states(self):
        """Update node states based on current field and connections"""
        for node_id, node_data in self.nodes.items():
            # Get node position
            position = self.node_positions[node_id]
            grid_pos = self._continuous_to_grid(position.coordinates)
            
            # Get local field properties
            local_amplitude = self._get_tensor_value_at_position(grid_pos)
            local_tension = self.tensor_field.tension_field[grid_pos] if all(p < self.resolution for p in grid_pos) else 0
            
            # Calculate resonance based on how well node phase matches field phase
            try:
                local_phase = self.tensor_field.phase_tensor[grid_pos]
                phase_match = abs(math.cos(position.phase - local_phase))
            except:
                phase_match = 0.5
            
            # Calculate node activity based on field interaction and connections
            connection_activity = len(self.node_connections[node_id]) * 0.05
            field_activity = local_amplitude * 0.8 + local_tension * 0.2
            
            # Update node state
            node_data['state']['activity'] = min(1.0, field_activity + connection_activity)
            node_data['state']['resonance'] = phase_match
            
            # Update energy based on resonance and tension
            # High resonance increases energy, high tension decreases stability
            energy_delta = (phase_match - 0.5) * 0.01
            stability_delta = (0.5 - local_tension) * 0.005
            
            node_data['state']['energy'] = min(1.0, max(0.0, node_data['state']['energy'] + energy_delta))
            node_data['state']['stability'] = min(1.0, max(0.0, node_data['state']['stability'] + stability_delta))
    
    def _update_string_entanglement(self, dt: float):
        """Update entanglement between quantum strings"""
        # Apply random phase to entanglement matrix
        phase_factor = np.exp(1j * self.global_phase)
        
        # Apply subtle change to entanglement matrix
        for i in range(self.qubit_depth):
            for j in range(i+1, self.qubit_depth):
                # Calculate coupling based on string similarities
                string1 = self.strings[i]
                string2 = self.strings[j]
                
                # Measure string similarity
                similarity = self._calculate_string_similarity(string1, string2)
                
                # Update entanglement
                coupling = 0.01 * similarity * dt
                
                # Apply phase-based coupling
                self.entanglement_matrix[i, j] += coupling * phase_factor
                self.entanglement_matrix[j, i] += coupling * phase_factor.conjugate()
        
        # Normalize entanglement matrix (ensure it's unitary)
        # This is a simplified approximation
        w, v = np.linalg.eigh(self.entanglement_matrix)
        self.entanglement_matrix = v @ np.diag(w / np.abs(w)) @ v.T.conj()
        
        # Apply string interference based on entanglement
        for i in range(self.qubit_depth):
            for j in range(i+1, self.qubit_depth):
                # Get entanglement strength
                entanglement = abs(self.entanglement_matrix[i, j])
                if entanglement > 0.01:
                    # Apply interference between strings
                    self.strings[i].apply_interference(self.strings[j], entanglement)
                    self.strings[j].apply_interference(self.strings[i], entanglement)
    
    def _calculate_string_similarity(self, string1: QuantumString, string2: QuantumString) -> float:
        """Calculate similarity between two quantum strings"""
        # Get patterns from strings
        pattern1 = string1.extract_pattern()
        pattern2 = string2.extract_pattern()
        
        # Calculate correlation
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        # Calculate normalized correlation
        p1_norm = p1 - np.mean(p1)
        p2_norm = p2 - np.mean(p2)
        
        norm1 = np.linalg.norm(p1_norm)
        norm2 = np.linalg.norm(p2_norm)
        
        if norm1 > 0 and norm2 > 0:
            correlation = np.sum(p1_norm * p2_norm) / (norm1 * norm2)
            # Map from [-1, 1] to [0, 1]
            return (correlation + 1) / 2
        else:
            return 0.5
    
    def _entangle_nodes(self, node1_id: str, node2_id: str, strength: float):
        """Create quantum entanglement between two nodes"""
        if node1_id not in self.node_positions or node2_id not in self.node_positions:
            return
            
        # Get node positions
        pos1 = self.node_positions[node1_id]
        pos2 = self.node_positions[node2_id]
        
        # Synchronize phases partially
        phase_diff = (pos2.phase - pos1.phase) % (2 * np.pi)
        if phase_diff > np.pi:
            phase_diff -= 2 * np.pi
            
        # Move phases closer by a factor based on strength
        pos1.phase = (pos1.phase + phase_diff * strength * 0.1) % (2 * np.pi)
        pos2.phase = (pos2.phase - phase_diff * strength * 0.1) % (2 * np.pi)
        
        # Create field entanglement between node positions
        grid_pos1 = self._continuous_to_grid(pos1.coordinates)
        grid_pos2 = self._continuous_to_grid(pos2.coordinates)
        
        # Apply refractive resonance between positions
        try:
            self.tensor_field.apply_resonance(
                position=grid_pos1,
                radius=2,
                strength=strength * 0.2,
                mode=ResonanceMode.REFRACTIVE
            )
            self.tensor_field.apply_resonance(
                position=grid_pos2,
                radius=2,
                strength=strength * 0.2,
                mode=ResonanceMode.REFRACTIVE
            )
        except Exception as e:
            pass  # Handle any errors gracefully
    
    def _disentangle_nodes(self, node1_id: str, node2_id: str):
        """Reduce quantum entanglement between two nodes"""
        if node1_id not in self.node_positions or node2_id not in self.node_positions:
            return
            
        # Get node positions
        pos1 = self.node_positions[node1_id]
        pos2 = self.node_positions[node2_id]
        
        # Add small random phase changes to break synchronization
        pos1.phase = (pos1.phase + np.random.uniform(-0.1, 0.1)) % (2 * np.pi)
        pos2.phase = (pos2.phase + np.random.uniform(-0.1, 0.1)) % (2 * np.pi)
        
        # Apply destructive resonance at both positions
        grid_pos1 = self._continuous_to_grid(pos1.coordinates)
        grid_pos2 = self._continuous_to_grid(pos2.coordinates)
        
        try:
            self.tensor_field.apply_resonance(
                position=grid_pos1,
                radius=2,
                strength=0.1,
                mode=ResonanceMode.DESTRUCTIVE
            )
            self.tensor_field.apply_resonance(
                position=grid_pos2,
                radius=2,
                strength=0.1,
                mode=ResonanceMode.DESTRUCTIVE
            )
        except Exception as e:
            pass  # Handle any errors gracefully
    
    def _continuous_to_grid(self, coordinates: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous coordinates (-1 to 1) to grid indices (0 to resolution-1)"""
        # Map from [-1, 1] to [0, resolution-1]
        indices = ((coordinates + 1) / 2 * (self.resolution - 1)).astype(int)
        
        # Ensure all indices are within bounds
        indices = np.clip(indices, 0, self.resolution - 1)
        
        return tuple(indices)
    
    def _update_analytics(self):
        """Update analytics data after simulation step"""
        # Record resonance distribution
        resonance_data = self.tensor_field.get_resonance_distribution()
        self.resonance_history.append({
            'time': time.time(),
            'count': resonance_data['count'],
            'avg_tension': resonance_data['average_tension'],
            'clusters': resonance_data['clusters']
        })
        
        # Trim history if too long
        if len(self.resonance_history) > 100:
            self.resonance_history = self.resonance_history[-100:]
        
        # Record energy data
        energy_data = {
            'time': time.time(),
            'field_energy': self.tensor_field.get_energy_density(),
            'node_energy': sum(node['state']['energy'] for node in self.nodes.values()) / max(1, len(self.nodes)),
            'string_energy': sum(string.get_energy() for string in self.strings) / max(1, len(self.strings))
        }
        self.energy_history.append(energy_data)
        
        # Trim energy history if too long
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]
        
        # Detect and store significant patterns
        self._detect_patterns()
    
    def _detect_patterns(self):
        """Detect significant patterns in the tensor field"""
        # Skip if no resonance points
        if not self.tensor_field.resonance_points:
            return
            
        # Extract highest tension points
        high_tension_points = sorted(
            self.tensor_field.resonance_points, 
            key=lambda x: x[1],  # Sort by tension value
            reverse=True
        )[:10]  # Take top 10
        
        # Skip if not enough high tension points
        if len(high_tension_points) < 3:
            return
            
        # Extract point coordinates and tension values
        points = np.array([point[0] for point in high_tension_points])
        tensions = np.array([point[1] for point in high_tension_points])
        
        # Check for patterns by looking at spatial arrangement
        # For simplicity, we'll just check for clusters and lines
        
        # 1. Check for clusters (by calculating pairwise distances)
        distances = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                # Calculate squared distance
                squared_dist = sum((points[i][d] - points[j][d])**2 for d in range(min(3, self.dimensions)))
                distances.append(math.sqrt(squared_dist))
        
        # Calculate statistics
        if distances:
            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)
            