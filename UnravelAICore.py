class UnravelAICore:
    """
    Core class that integrates quantum-inspired code analysis, emergent pattern detection,
    and advanced software reconstruction.
    """
    
    def __init__(self, work_dir: str = None):
        # Initialize working directory
        self.work_dir = work_dir or os.path.join(os.getcwd(), "unravel_ai_workdir")
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Initialize quantum network
        self.quantum_network = EmergentIntelligenceNetwork(dimensions=4, resolution=64)
        
        # Initialize pattern detector
        self.pattern_detector = EmergentPatternDetector(self.quantum_network)
        
        # Initialize quantum-aware code analyzer
        self.code_analyzer = QuantumAwareCodeAnalyzer(dimensions=4)
        
        # Initialize local LLM interface (if available)
        self.llm_client = None
        try:
            from src.core.llm import LLMClient
            self.llm_client = LLMClient(
                api_key=os.environ.get("LLM_API_KEY", ""),
                model=os.environ.get("LLM_MODEL", "gpt-4"),
                endpoint=os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1")
            )
        except ImportError:
            print("LLM integration not available - running in standalone mode")
        
        # Create subdirectories
        self.uploads_dir = os.path.join(self.work_dir, "uploads")
        self.analysis_dir = os.path.join(self.work_dir, "analysis")
        self.reconstructed_dir = os.path.join(self.work_dir, "reconstructed")
        
        for d in [self.uploads_dir, self.analysis_dir, self.reconstructed_dir]:
            os.makedirs(d, exist_ok=True)
    
    async def process_codebase(self, input_directory: str, 
                              target_language: Optional[str] = None) -> Dict[str, Any]:
        """Process an entire codebase with quantum-aware analysis"""
        print(f"Processing codebase: {input_directory}")
        
        # Create session directory
        session_id = str(uuid.uuid4())[:8]
        session_dir = os.path.join(self.analysis_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Find all code files
        code_files = []
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.py', '.js', '.ts', '.c', '.cpp', '.h', '.hpp', '.cs', '.java', '.go', '.rs']:
                    code_files.append(os.path.join(root, file))
        
        print(f"Found {len(code_files)} code files")
        
        # Analyze each file and build quantum network
        file_nodes = {}
        for file_path in code_files:
            print(f"Analyzing file: {os.path.basename(file_path)}")
            node_id = self.code_analyzer.analyze_file(file_path)
            file_nodes[file_path] = node_id
        
        # Analyze dependencies between files
        print("Analyzing dependencies between files")
        self.code_analyzer.analyze_dependencies(code_files)
        
        # Evolve the network
        print("Evolving quantum network to find emergent patterns")
        for _ in range(50):
            self.quantum_network.evolve_network(1)
            self.pattern_detector.detect_patterns()
        
        # Get emergent properties
        print("Extracting emergent properties")
        emergent_properties = self.pattern_detector.get_emergent_properties()
        
        # Get detailed analysis
        network_analysis = self.code_analyzer.get_analysis_report()
        
        # Save analysis to file
        analysis_path = os.path.join(session_dir, "analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump({
                'session_id': session_id,
                'file_count': len(code_files),
                'emergent_properties': emergent_properties,
                'network_analysis': network_analysis
            }, f, indent=2)
        
        # Perform reconstruction if target language specified
        reconstructed_files = []
        if target_language:
            print(f"Reconstructing codebase in {target_language}")
            reconstructed_files = await self._reconstruct_codebase(
                code_files, network_analysis, target_language, session_dir
            )
        
        return {
            'session_id': session_id,
            'file_count': len(code_files),
            'emergent_properties': emergent_properties,
            'network_analysis': network_analysis,
            'reconstructed_files': reconstructed_files
        }
    
    async def _reconstruct_codebase(self, 
                                   code_files: List[str],
                                   network_analysis: Dict[str, Any],
                                   target_language: str,
                                   session_dir: str) -> List[str]:
        """Reconstruct codebase in target language using quantum-guided approach"""
        if not self.llm_client:
            print("Cannot reconstruct codebase - LLM integration not available")
            return []
        
        # Create reconstruction directory
        recon_dir = os.path.join(self.reconstructed_dir, os.path.basename(session_dir))
        os.makedirs(recon_dir, exist_ok=True)
        
        # Identify most important files from network analysis
        file_importance = {}
        for file_path, metrics in network_analysis['file_metrics'].items():
            file_importance[file_path] = metrics['centrality']
        
        # Sort files by importance
        important_files = sorted(file_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Prioritize handling of most important files
        reconstructed_files = []
        
        # Setup reconstruction config
        from dataclasses import dataclass, field
        
        @dataclass
        class ReconstructionConfig:
            quality_level: str = "high"
            add_comments: bool = True
            improve_security: bool = True
            optimize_performance: bool = True
            modernize_codebase: bool = True
            add_testing: bool = False
            target_language: str = target_language
            preserve_functionality: bool = True
            rename_variables: bool = False
            code_style: str = "clean"
            custom_patterns: List[Dict[str, str]] = field(default_factory=list)
            
            # Add quantum insights from analysis
            quantum_insights: Dict[str, Any] = field(default_factory=dict)
        
        # Initialize config with quantum insights
        config = ReconstructionConfig()
        config.quantum_insights = {
            'emergent_properties': self.pattern_detector.get_emergent_properties(),
            'coherence': self.quantum_network.global_coherence,
            'important_patterns': [p for p in self.pattern_detector.patterns if 'small_world_network' in p.get('type', '') or 'attractor_state' in p.get('type', '')]
        }
        
        # Process files in order of importance
        for file_path, importance in important_files:
            print(f"Reconstructing {os.path.basename(file_path)} (importance: {importance:.4f})")
            
            # Create output path
            rel_path = os.path.relpath(file_path, os.path.commonpath(code_files))
            output_path = os.path.join(recon_dir, rel_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # If target language is different, adjust file extension
            if target_language:
                ext_map = {
                    "python": ".py",
                    "javascript": ".js",
                    "typescript": ".ts",
                    "c": ".c",
                    "cpp": ".cpp",
                    "csharp": ".cs",
                    "java": ".java",
                    "go": ".go",
                    "rust": ".rs"
                }
                
                if target_language.lower() in ext_map:
                    base_name = os.path.splitext(output_path)[0]
                    output_path = base_name + ext_map[target_language.lower()]
            
            try:
                # Read source file
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                if target_language:
                    # Translate to target language
                    source_lang = os.path.splitext(file_path)[1][1:]  # Remove dot
                    if source_lang == "hpp":
                        source_lang = "cpp"
                    elif source_lang == "h":
                        source_lang = "c"
                    
                    # Use LLM to translate
                    translated = await self.llm_client.translate_to_language(
                        content, source_lang, target_language
                    )
                    
                    # Write translated file
                    with open(output_path, 'w') as f:
                        f.write(translated)
                    
                    reconstructed_files.append(output_path)
                else:
                    # Just copy the file if no translation needed
                    shutil.copy2(file_path, output_path)
                    reconstructed_files.append(output_path)
            
            except Exception as e:
                print(f"Error reconstructing {file_path}: {str(e)}")
        
        return reconstructed_files
    
    def visualize_quantum_network(self, output_path: str) -> None:
        """Generate visualization of the quantum network"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Get graph from quantum network
            G = self.quantum_network.graph
            
            # Get node positions using force-directed layout
            pos = nx.spring_layout(G, seed=42)
            
            # Get node colors based on quantum state
            colors = []
            for node_id in G.nodes():
                node = self.quantum_network.nodes.get(node_id)
                if node:
                    # Use first two components of state vector to determine color
                    if node.state_vector.size >= 2:
                        r = (node.state_vector[0] + 1) / 2  # Map from [-1,1] to [0,1]
                        g = (node.state_vector[1] + 1) / 2
                        b = 0.5  # Fixed blue component
                        colors.append((r, g, b))
                    else:
                        colors.append((0.5, 0.5, 0.5))  # Default gray
                else:
                    colors.append((0.5, 0.5, 0.5))  # Default gray
            
            # Calculate edge weights for width
            edge_weights = [G[u][v].get('weight', 1.0) * 2 for u, v in G.edges()]
            
            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=80, node_color=colors, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, arrows=True)
            
            # Add labels to important nodes
            node_centrality = nx.eigenvector_centrality_numpy(G)
            important_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = {node: node for node, _ in important_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
            
            plt.title("Quantum Network Visualization")
            plt.axis('off')
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Network visualization saved to {output_path}")
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")

# Main entry point for command-line usage
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unravel AI - Quantum Code Analysis")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing code files")
    parser.add_argument("--output", "-o", help="Output directory for analysis results")
    parser.add_argument("--target", "-t", help="Target language for reconstruction")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate network visualization")
    
    args = parser.parse_args()
    
    # Create core instance
    core = UnravelAICore(args.output)
    
    # Process codebase
    result = await core.process_codebase(args.input, args.target)
    
    # Generate visualization if requested
    if args.visualize:
        viz_path = os.path.join(core.analysis_dir, result['session_id'], "network_visualization.png")
        core.visualize_quantum_network(viz_path)
    
    print(f"Analysis complete. Session ID: {result['session_id']}")
    print(f"Emergent intelligence score: {result['emergent_properties']['emergent_intelligence_score']:.4f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
