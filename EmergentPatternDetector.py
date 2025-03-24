class EmergentPatternDetector:
    """Detects emergent patterns in the quantum simulation"""
    
    def __init__(self, network: EmergentIntelligenceNetwork):
        self.network = network
        self.patterns = []
        self.pattern_history = []
        self.stability_scores = {}
        
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect emergent patterns in the current network state"""
        # Only detect if we have enough evolution steps
        if self.network.evolution_steps < 10:
            return []
            
        # Create graph representation for analysis
        G = self.network.graph
        
        # Detect patterns
        new_patterns = []
        
        # 1. Community detection
        communities = list(nx.algorithms.community.greedy_modularity_communities(G.to_undirected()))
        if communities and len(communities) >= 2:
            # Check if we have distinct communities
            modularity = nx.algorithms.community.modularity(G.to_undirected(), communities)
            
            if modularity > 0.3:  # Good separation
                community_pattern = {
                    'type': 'community_structure',
                    'count': len(communities),
                    'modularity': modularity,
                    'sizes': [len(c) for c in communities],
                    'step': self.network.evolution_steps
                }
                new_patterns.append(community_pattern)
        
        # 2. Motif detection (recurrent network patterns)
        motifs = self._find_network_motifs(G)
        if motifs:
            motif_pattern = {
                'type': 'recurring_motifs',
                'motifs': motifs,
                'step': self.network.evolution_steps
            }
            new_patterns.append(motif_pattern)
        
        # 3. Phase transitions (global coherence jumps)
        if len(self.network.coherence_history) >= 5:
            recent_coherence = self.network.coherence_history[-5:]
            coherence_diff = recent_coherence[-1] - recent_coherence[0]
            
            if abs(coherence_diff) > 0.2:  # Significant jump
                transition_pattern = {
                    'type': 'phase_transition',
                    'direction': 'increasing' if coherence_diff > 0 else 'decreasing',
                    'magnitude': abs(coherence_diff),
                    'step': self.network.evolution_steps
                }
                new_patterns.append(transition_pattern)
        
        # 4. Small-world properties emergence
        if nx.is_connected(G.to_undirected()) and len(G) > 10:
            try:
                avg_path = nx.average_shortest_path_length(G.to_undirected())
                clustering = nx.average_clustering(G.to_undirected())
                
                # Check for small-world property (high clustering, low path length)
                if clustering > 0.4 and avg_path < 3.0:
                    small_world_pattern = {
                        'type': 'small_world_network',
                        'clustering': clustering,
                        'avg_path_length': avg_path,
                        'step': self.network.evolution_steps
                    }
                    new_patterns.append(small_world_pattern)
            except:
                pass
        
        # 5. Attractor states (stability analysis)
        if self.network.evolution_steps % 10 == 0:
            # Every 10 steps, check if we're in a stable state
            state_hash = self._hash_network_state()
            
            if state_hash in self.stability_scores:
                # We've seen this state before, increase stability score
                self.stability_scores[state_hash] += 1
                
                # Check if stability score is high enough
                if self.stability_scores[state_hash] >= 3:
                    attractor_pattern = {
                        'type': 'attractor_state',
                        'stability_score': self.stability_scores[state_hash],
                        'step': self.network.evolution_steps
                    }
                    new_patterns.append(attractor_pattern)
            else:
                # New state
                self.stability_scores[state_hash] = 1
        
        # Add new patterns to history
        for pattern in new_patterns:
            if not self._is_duplicate_pattern(pattern):
                self.patterns.append(pattern)
                self.pattern_history.append({
                    'step': self.network.evolution_steps,
                    'pattern': pattern
                })
        
        return new_patterns
    
    def _find_network_motifs(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """Find recurring motifs (subgraph patterns) in the network"""
        motifs = []
        
        # Look for common network motifs
        
        # 1. Feedforward loops
        ffl_count = 0
        for node in G.nodes():
            # Check successors (outgoing edges)
            successors = list(G.successors(node))
            if len(successors) < 2:
                continue
                
            # Look for feed-forward pattern: A -> B -> C and A -> C
            for i, b in enumerate(successors):
                b_successors = list(G.successors(b))
                for c in b_successors:
                    if c in successors:
                        ffl_count += 1
        
        if ffl_count > 0:
            motifs.append({
                'name': 'feed_forward_loop',
                'count': ffl_count
            })
        
        # 2. Bi-fan motifs (A,B -> C,D)
        bifan_count = 0
        for a in G.nodes():
            a_successors = list(G.successors(a))
            if len(a_successors) < 2:
                continue
                
            # Look for other nodes with similar output pattern
            for b in G.nodes():
                if a == b:
                    continue
                    
                b_successors = list(G.successors(b))
                if len(b_successors) < 2:
                    continue
                
                # Find common successors
                common = set(a_successors) & set(b_successors)
                if len(common) >= 2:
                    bifan_count += 1
        
        if bifan_count > 0:
            motifs.append({
                'name': 'bi_fan',
                'count': bifan_count
            })
        
        # 3. Feedback loops
        feedback_count = 0
        try:
            cycles = list(nx.simple_cycles(G))
            feedback_count = len(cycles)
            
            if feedback_count > 0:
                motifs.append({
                    'name': 'feedback_loop',
                    'count': feedback_count
                })
        except:
            pass
        
        return motifs
    
    def _hash_network_state(self) -> str:
        """Create a hash of the current network state for stability analysis"""
        # Simplified hash - just check node states
        state_str = ""
        
        # Get node states in sorted order
        nodes = sorted(self.network.nodes.keys())
        for node_id in nodes:
            node = self.network.nodes[node_id]
            # Use first two components of state vector for simplicity
            if len(node.state_vector) >= 2:
                state_str += f"{node.state_vector[0]:.2f}_{node.state_vector[1]:.2f}_"
        
        # Use md5 for hash
        import hashlib
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _is_duplicate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Check if a pattern is a duplicate of an existing one"""
        # Only check recent patterns (last 10)
        recent_patterns = self.patterns[-10:] if len(self.patterns) > 10 else self.patterns
        
        for existing in recent_patterns:
            if existing['type'] != pattern['type']:
                continue
                
            # Type-specific comparison
            if pattern['type'] == 'community_structure':
                if abs(existing['modularity'] - pattern['modularity']) < 0.1:
                    return True
            elif pattern['type'] == 'phase_transition':
                if existing['direction'] == pattern['direction'] and \
                   abs(existing['magnitude'] - pattern['magnitude']) < 0.1:
                    return True
            elif pattern['type'] == 'small_world_network':
                if abs(existing['clustering'] - pattern['clustering']) < 0.1 and \
                   abs(existing['avg_path_length'] - pattern['avg_path_length']) < 0.2:
                    return True
            elif pattern['type'] == 'attractor_state':
                # Attractor states are already checked via hash
                return False
        
        return False
    
    def get_emergent_properties(self) -> Dict[str, Any]:
        """Get emergent properties summary"""
        result = {
            'pattern_count': len(self.patterns),
            'pattern_types': {},
            'temporal_evolution': {},
            'emergent_intelligence_score': 0.0
        }
        
        # Count pattern types
        for pattern in self.patterns:
            pattern_type = pattern['type']
            if pattern_type not in result['pattern_types']:
                result['pattern_types'][pattern_type] = 0
            result['pattern_types'][pattern_type] += 1
        
        # Analyze temporal evolution
        if self.pattern_history:
            # Group patterns by time steps
            step_patterns = {}
            for entry in self.pattern_history:
                step = entry['step']
                if step not in step_patterns:
                    step_patterns[step] = 0
                step_patterns[step] += 1
            
            # Find peaks of activity
            sorted_steps = sorted(step_patterns.keys())
            result['temporal_evolution'] = {
                'pattern_timeline': [{'step': k, 'count': step_patterns[k]} for k in sorted_steps],
                'activity_peaks': [k for k in sorted_steps if step_patterns[k] > 2]
            }
        
        # Calculate emergent intelligence score
        # Weight different factors:
        # - Pattern diversity (number of different pattern types)
        # - Pattern complexity (especially small world and attractor states)
        # - Coherence stability
        
        pattern_diversity = len(result['pattern_types'])
        complex_pattern_count = result['pattern_types'].get('small_world_network', 0) + \
                               result['pattern_types'].get('attractor_state', 0)
        
        # Coherence stability (if available)
        coherence_stability = 0.0
        if len(self.network.coherence_history) > 10:
            recent = self.network.coherence_history[-10:]
            coherence_stability = 1.0 - np.std(recent) / max(0.01, np.mean(recent))
        
        # Calculate score (0-1 scale)
        score = (0.4 * min(1.0, pattern_diversity / 4.0) + 
                 0.4 * min(1.0, complex_pattern_count / 5.0) +
                 0.2 * coherence_stability)
        
        result['emergent_intelligence_score'] = score
        
        return result
