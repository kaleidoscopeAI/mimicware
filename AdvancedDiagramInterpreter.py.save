class AdvancedDiagramInterpreter:
    """Advanced diagram interpretation system with integrated C/Python components"""
    
    def __init__(self):
        self.analyzer = EnhancedDiagramAnalyzer()
        self.graph_analyzer = GraphTheoreticDiagramAnalyzer()
        self.probabilistic_model = None
    
    def interpret_diagram(self, elements):
        """Main entry point for diagram interpretation"""
        # Convert elements to internal format if needed
        diagram_elements = [self._element_to_diagram_element(e) for e in elements]
        
        # Initialize probabilistic model
        self.probabilistic_model = ProbabilisticDiagramConnectionModel(diagram_elements)
        
        # Analyze diagram structure
        structure = self.analyzer.analyze(diagram_elements)
        
        # Perform graph-theoretic analysis
        graph_analysis = self.graph_analyzer.analyze(
            structure.get("nodes", []), 
            structure.get("connections", structure.get("edges", []))
        )
        
        # Merge analyses
        structure["graph_analysis"] = graph_analysis
        
        # Extract entities and relationships
        entities = self._extract_entities_from_structure(structure)
        structure["entity_model"] = entities
        
        return structure
    
    def _element_to_diagram_element(self, element):
        """Convert a raw element to DiagramElement"""
        if isinstance(element, DiagramElement):
            return element
            
        # Handle dict or other formats
        if isinstance(element, dict):
            return DiagramElement(
                id=element.get('id', f"elem_{id(element)}"),
                type=element.get('type', 'unknown'),
                text=element.get('text', ''),
                x=element.get('x', 0.0),
                y=element.get('y', 0.0),
                width=element.get('width', 50.0),
                height=element.get('height', 50.0),
                x1=element.get('x1', 0.0),
                y1=element.get('y1', 0.0),
                x2=element.get('x2', 0.0),
                y2=element.get('y2', 0.0),
                style=element.get('style', 'solid'),
                arrow_type=element.get('arrow_type', 'none'),
                arrow_fill=element.get('arrow_fill', False)
            )
        
        # Unknown element type
        return None
    
    def _extract_entities_from_structure(self, structure):
        """Extract entities and relationships from diagram structure"""
        entities = []
        
        # Extract from classes
        if "classes" in structure:
            for cls in structure["classes"]:
                entities.append({
                    "name": cls.get("name", ""),
                    "type": "class",
                    "attributes": cls.get("attributes", []),
                    "methods": cls.get("methods", []),
                    "source": "structure"
                })
        
        # Extract from nodes (diagrams)
        if "nodes" in structure:
            for node in structure["nodes"]:
                entities.append({
                    "name": node.get("text", node.get("id", "")),
                    "type": node.get("type", "node"),
                    "source": "structure"
                })
        
        # Extract from entities (ER diagrams)
        if "entities" in structure:
            for entity in structure["entities"]:
                entities.append({
                    "name": entity.get("name", ""),
                    "type": "entity",
                    "attributes": entity.get("attributes", []),
                    "source": "structure"
                })
        
        # Extract connections and relationships
        relationships = []
        for conn_key in ["relationships", "edges", "connections"]:
            if conn_key in structure:
                for conn in structure[conn_key]:
                    relationships.append({
                        "type": "relationship",
                        "name": conn.get("label", conn.get("name", "connection")),
                        "source": conn.get("source", conn.get("source_id", conn.get("entity1", ""))),
                        "target": conn.get("target", conn.get("target_id", conn.get("entity2", ""))),
                        "relationship_type": conn.get("type", "association")
                    })
        
        return {
            "entities": entities,
            "relationships": relationships
        }
