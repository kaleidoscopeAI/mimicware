import numpy as np
import networkx as nx
import re
import scipy.spatial as spatial
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DiagramElement:
    """Rich representation of a diagram element with comprehensive attributes"""
    id: str
    type: str
    text: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    style: str = "solid"
    arrow_type: str = "none"
    arrow_fill: bool = False
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def center(self) -> Tuple[float, float]:
        """Calculate element center point"""
        if self.type in ["line", "arrow"]:
            return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
        return (self.x, self.y)
    
    def bbox(self) -> Tuple[float, float, float, float]:
        """Return element bounding box as (x_min, y_min, x_max, y_max)"""
        if self.type in ["line", "arrow"]:
            return (min(self.x1, self.x2), min(self.y1, self.y2), 
                    max(self.x1, self.x2), max(self.y1, self.y2))
        else:
            half_w = self.width / 2
            half_h = self.height / 2
            return (self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h)
    
    def distance_to_point(self, px: float, py: float) -> float:
        """Calculate minimum distance from element to point"""
        if self.type in ["line", "arrow"]:
            # Calculate distance from point to line segment
            x1, y1, x2, y2 = self.x1, self.y1, self.x2, self.y2
            
            # Line segment length squared
            l2 = (x2 - x1)**2 + (y2 - y1)**2
            
            if l2 == 0:  # Point case
                return np.sqrt((px - x1)**2 + (py - y1)**2)
            
            # Calculate projection parameter
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
            
            # Calculate closest point on line segment
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            
            # Return distance to that point
            return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        else:
            # For shapes, use center point and add an approximation for dimensions
            x_min, y_min, x_max, y_max = self.bbox()
            
            # Check if point is inside bounding box
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return 0
            
            # Calculate distance to nearest edge
            dx = max(x_min - px, 0, px - x_max)
            dy = max(y_min - py, 0, py - y_max)
            return np.sqrt(dx**2 + dy**2)

    def contains_point(self, px: float, py: float, padding: float = 0) -> bool:
        """Check if element contains or is very close to a point"""
        if self.type in ["line", "arrow"]:
            return self.distance_to_point(px, py) <= padding
        else:
            x_min, y_min, x_max, y_max = self.bbox()
            return (x_min - padding <= px <= x_max + padding and 
                    y_min - padding <= py <= y_max + padding)


class SpatialIndex:
    """Efficient spatial index for diagram elements using R-tree-like structure"""
    
    def __init__(self, elements: List[DiagramElement]):
        self.elements = elements
        self.rtree = None
        self._build_index()
    
    def _build_index(self):
        """Build spatial index from elements"""
        if not self.elements:
            return
        
        # Extract bounding boxes for all elements
        boxes = np.array([element.bbox() for element in self.elements])
        # Create spatial index
        self.rtree = spatial.cKDTree(np.column_stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,  # center x
            (boxes[:, 1] + boxes[:, 3]) / 2   # center y
        ]))
    
    def query_point(self, x: float, y: float, k: int = 5, distance_upper_bound: float = np.inf) -> List[int]:
        """Find k nearest elements to a point"""
        if self.rtree is None:
            return []
        
        # Query the R-tree
        distances, indices = self.rtree.query([x, y], k=k, distance_upper_bound=distance_upper_bound)
        
        # Filter out infinite distances
        valid_indices = [idx for idx, dist in zip(indices, distances) if dist != np.inf]
        
        # Further refine results by checking actual element distance
        refined_indices = []
        for idx in valid_indices:
            if self.elements[idx].distance_to_point(x, y) <= distance_upper_bound:
                refined_indices.append(idx)
        
        return refined_indices
    
    def query_box(self, x_min: float, y_min: float, x_max: float, y_max: float) -> List[int]:
        """Find all elements that overlap with the given box"""
        if self.rtree is None:
            return []
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2) / 2
        
        # Initial broad search
        indices = self.rtree.query_ball_point([center_x, center_y], r=radius)
        
        # Refine results by checking actual overlap
        results = []
        for idx in indices:
            element = self.elements[idx]
            e_x_min, e_y_min, e_x_max, e_y_max = element.bbox()
            
            # Check if element overlaps with query box
            if (e_x_min <= x_max and e_x_max >= x_min and 
                e_y_min <= y_max and e_y_max >= y_min):
                results.append(idx)
                
        return results


class DiagramClassifier:
    """ML-based diagram type classifier"""
    
    DIAGRAM_TYPES = [
        "flowchart", "uml_class_diagram", "uml_sequence_diagram", 
        "er_diagram", "uml_state_diagram", "generic_diagram"
    ]
    
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self._initialize()
    
    def _initialize(self):
        """Initialize with some hardcoded training data for common diagrams"""
        # Generate synthetic training data
        X, y = self._generate_synthetic_data()
        
        # Train classifier
        if len(X) > 0:
            self.classifier.fit(X, y)
            self.is_trained = True
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data based on diagram heuristics"""
        X = []
        y = []
        
        # Flowchart features
        for _ in range(20):
            # Randomize feature values around typical flowchart properties
            rectangles = np.random.randint(4, 15)
            diamonds = np.random.randint(1, 8)
            ellipses = np.random.randint(0, 5)
            arrows = np.random.randint(rectangles + diamonds, (rectangles + diamonds) * 2)
            decision_words = np.random.randint(1, 6)
            flow_words = np.random.randint(2, 8)
            
            X.append([rectangles, diamonds, ellipses, arrows, decision_words, flow_words, 0, 0, 0, 0, 0, 0, 0])
            y.append("flowchart")
        
        # UML Class diagram features
        for _ in range(20):
            rectangles = np.random.randint(3, 20)
            diamonds = np.random.randint(0, 2)
            ellipses = np.random.randint(0, 2)
            arrows = np.random.randint(2, rectangles * 2)
            class_words = np.random.randint(3, rectangles + 3)
            method_words = np.random.randint(3, rectangles * 2)
            attribute_words = np.random.randint(3, rectangles * 2)
            inheritance_words = np.random.randint(0, 5)
            compartments = np.random.randint(rectangles, rectangles * 3)
            
            X.append([rectangles, diamonds, ellipses, arrows, 0, 0, class_words, method_words, 
                      attribute_words, inheritance_words, compartments, 0, 0])
            y.append("uml_class_diagram")
        
        # UML Sequence diagram features
        for _ in range(20):
            rectangles = np.random.randint(2, 10)  # For actors
            vertical_lines = np.random.randint(rectangles, rectangles + 5)  # Lifelines
            arrows = np.random.randint(3, 20)  # Messages
            sequence_words = np.random.randint(1, 5)
            message_words = np.random.randint(arrows // 2, arrows * 2)
            
            X.append([rectangles, 0, 0, arrows, 0, 0, 0, 0, 0, 0, 0, 
                      vertical_lines, message_words])
            y.append("uml_sequence_diagram")
        
        # ER diagram features
        for _ in range(20):
            rectangles = np.random.randint(3, 15)  # Entities
            diamonds = np.random.randint(2, 10)  # Relationships
            arrows = np.random.randint(rectangles + diamonds, (rectangles + diamonds) * 2)
            entity_words = np.random.randint(rectangles, rectangles * 2)
            attribute_words = np.random.randint(rectangles, rectangles * 4)
            relationship_words = np.random.randint(diamonds, diamonds * 2)
            
            X.append([rectangles, diamonds, 0, arrows, 0, 0, 0, 0, 
                      attribute_words, 0, 0, 0, relationship_words])
            y.append("er_diagram")
        
        return np.array(X), np.array(y)
    
    def extract_features(self, elements: List[DiagramElement]) -> np.ndarray:
        """Extract classification features from diagram elements"""
        # Count different element types
        rectangles = sum(1 for e in elements if e.type == "rectangle")
        diamonds = sum(1 for e in elements if e.type == "diamond")
        ellipses = sum(1 for e in elements if e.type in ["ellipse", "circle"])
        arrows = sum(1 for e in elements if e.type == "arrow")
        
        # Check for keywords in text
        all_text = " ".join(e.text.lower() for e in elements if e.text)
        
        decision_words = len(re.findall(r'\b(if|then|else|decision|condition|yes|no)\b', all_text))
        flow_words = len(re.findall(r'\b(start|end|process|flow|return)\b', all_text))
        class_words = len(re.findall(r'\b(class|interface|abstract|extends|implements)\b', all_text))
        method_words = len(re.findall(r'\b(method|function|void|return|public|private|protected)\b', all_text))
        attribute_words = len(re.findall(r'\b(attribute|field|property|variable|data|type)\b', all_text))
        inheritance_words = len(re.findall(r'\b(inherit|extend|implement|parent|child|base|derived)\b', all_text))
        
        # Count other structural elements
        compartments = sum(1 for e in elements if e.type == "rectangle" and e.text.count("\n") > 1 and "---" in e.text)
        vertical_lines = sum(1 for e in elements if e.type == "line" and abs(e.x1 - e.x2) < 10)
        relationship_words = len(re.findall(r'\b(one|many|has|belongs|to|entity|relation)\b', all_text))
        
        return np.array([[
            rectangles, diamonds, ellipses, arrows, decision_words, flow_words,
            class_words, method_words, attribute_words, inheritance_words,
            compartments, vertical_lines, relationship_words
        ]])
    
    def predict_diagram_type(self, elements: List[DiagramElement]) -> str:
        """Predict diagram type based on elements"""
        if not self.is_trained:
            return self._fallback_detection(elements)
        
        # Extract features and make prediction
        features = self.extract_features(elements)
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(features)[0]
        probs = self.classifier.predict_proba(features)[0]
        
        # If confidence is too low, use heuristic fallback
        confidence = max(probs)
        if confidence < 0.6:
            fallback = self._fallback_detection(elements)
            if fallback != prediction:
                # Use majority vote
                return prediction if confidence > 0.4 else fallback
        
        return prediction
    
    def _fallback_detection(self, elements: List[DiagramElement]) -> str:
        """Traditional heuristic detection as fallback"""
        # Flowchart detection
        diamonds = sum(1 for e in elements if e.type == "diamond")
        arrows = sum(1 for e in elements if e.type == "arrow")
        
        all_text = " ".join(e.text.lower() for e in elements if e.text)
        flow_indicators = len(re.findall(r'\b(start|end|if|then|else|process|decision)\b', all_text))
        
        if (diamonds >= 1 or flow_indicators >= 2) and arrows >= 2:
            return "flowchart"
        
        # UML Class detection
        rectangles = sum(1 for e in elements if e.type == "rectangle")
        compartments = sum(1 for e in elements if e.type == "rectangle" and e.text.count("\n") > 1 and "---" in e.text)
        class_indicators = len(re.findall(r'\b(class|interface|abstract|attributes|methods)\b', all_text))
        
        if compartments >= 1 or class_indicators >= 2:
            return "uml_class_diagram"
        
        # Sequence diagram detection
        top_rectangles = sum(1 for e in elements if e.type == "rectangle" and e.y < 100)
        vertical_lines = sum(1 for e in elements if e.type == "line" and abs(e.x1 - e.x2) < 10)
        horizontal_arrows = sum(1 for e in elements if e.type == "arrow" and abs(e.y1 - e.y2) < 10)
        
        if top_rectangles >= 2 and vertical_lines >= 2 and horizontal_arrows >= 1:
            return "uml_sequence_diagram"
        
        # ER diagram detection
        entity_indicators = len(re.findall(r'\b(entity|attribute|relationship|key|cardinality)\b', all_text))
        
        if (diamonds >= 1 and rectangles >= 2) or entity_indicators >= 2:
            return "er_diagram"
        
        # Default to generic diagram
        return "generic_diagram"


class DiagramAnalyzer:
    """Enhanced diagram analyzer with advanced spatial and graph-theoretic analysis"""
    
    def __init__(self):
        self.classifier = DiagramClassifier()
        self.elements = []
        self.spatial_index = None
        self.graph = None
    
    def analyze(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main analysis entry point"""
        # Convert to DiagramElement objects
        self.elements = [self._dict_to_diagram_element(e) for e in elements]
        
        # Create spatial index
        self.spatial_index = SpatialIndex(self.elements)
        
        # Determine diagram type
        diagram_type = self.classifier.predict_diagram_type(self.elements)
        
        # Create a graph representation
        self._build_graph()
        
        # Perform diagram-specific analysis
        if diagram_type == "flowchart":
            structure = self._extract_flowchart()
        elif diagram_type == "uml_class_diagram":
            structure = self._extract_uml_class_diagram()
        elif diagram_type == "uml_sequence_diagram":
            structure = self._extract_uml_sequence_diagram()
        elif diagram_type == "er_diagram":
            structure = self._extract_er_diagram()
        else:
            structure = self._interpret_generic_diagram()
        
        # Add graph metrics
        structure["metrics"] = self._calculate_graph_metrics()
        
        return structure
    
    def _dict_to_diagram_element(self, element_dict: Dict[str, Any]) -> DiagramElement:
        """Convert dictionary representation to DiagramElement"""
        # Handle properties not explicitly defined in DiagramElement
        properties = {k: v for k, v in element_dict.items() 
                     if k not in ['id', 'type', 'text', 'x', 'y', 'width', 'height',
                                 'x1', 'y1', 'x2', 'y2', 'style', 'arrow_type', 'arrow_fill']}
        
        # Generate an ID if not present
        if 'id' not in element_dict:
            element_dict['id'] = f"elem_{id(element_dict)}"
            
        return DiagramElement(
            id=element_dict.get('id', f"elem_{id(element_dict)}"),
            type=element_dict.get('type', 'unknown'),
            text=element_dict.get('text', ''),
            x=element_dict.get('x', 0.0),
            y=element_dict.get('y', 0.0),
            width=element_dict.get('width', 50.0),
            height=element_dict.get('height', 50.0),
            x1=element_dict.get('x1', 0.0),
            y1=element_dict.get('y1', 0.0),
            x2=element_dict.get('x2', 0.0),
            y2=element_dict.get('y2', 0.0),
            style=element_dict.get('style', 'solid'),
            arrow_type=element_dict.get('arrow_type', 'none'),
            arrow_fill=element_dict.get('arrow_fill', False),
            properties=properties
        )
    
    def _build_graph(self):
        """Build a graph representation of the diagram"""
        self.graph = nx.DiGraph()
        
        # Add all non-line elements as nodes
        for i, elem in enumerate(self.elements):
            if elem.type not in ["line", "arrow"]:
                self.graph.add_node(elem.id, element=elem, index=i)
        
        # Add arrows and lines as edges
        for i, elem in enumerate(self.elements):
            if elem.type in ["line", "arrow"]:
                # Find the closest elements to the start and end points
                start_nearest = self._find_closest_element(elem.x1, elem.y1, exclude_types=["line", "arrow"])
                end_nearest = self._find_closest_element(elem.x2, elem.y2, exclude_types=["line", "arrow"])
                
                if start_nearest and end_nearest and start_nearest != end_nearest:
                    # Add edge to graph
                    self.graph.add_edge(
                        self.elements[start_nearest].id, 
                        self.elements[end_nearest].id, 
                        element=elem,
                        index=i
                    )
    
    def _find_closest_element(self, x: float, y: float, max_distance: float = 40.0, 
                             exclude_types: List[str] = None) -> Optional[int]:
        """Find index of the closest element to a point"""
        if not self.spatial_index:
            return None
            
        exclude_types = exclude_types or []
        
        # Get candidates from spatial index
        candidates = self.spatial_index.query_point(x, y, k=5, distance_upper_bound=max_distance)
        
        # Filter by type and find closest
        min_dist = float('inf')
        closest_idx = None
        
        for idx in candidates:
            elem = self.elements[idx]
            if elem.type in exclude_types:
                continue
                
            dist = elem.distance_to_point(x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
        
        return closest_idx
    
    def _calculate_graph_metrics(self) -> Dict[str, Any]:
        """Calculate graph-theoretic metrics for the diagram"""
        metrics = {}
        
        if not self.graph:
            return metrics
        
        # Basic metrics
        metrics["node_count"] = self.graph.number_of_nodes()
        metrics["edge_count"] = self.graph.number_of_edges()
        
        # Connected components
        if not nx.is_directed(self.graph):
            metrics["connected_components"] = nx.number_connected_components(self.graph.to_undirected())
        
        # Centrality measures (only if graph is not empty)
        if metrics["node_count"] > 0:
            # Degree centrality
            try:
                deg_centrality = nx.degree_centrality(self.graph)
                metrics["max_degree_centrality"] = max(deg_centrality.values()) if deg_centrality else 0
                
                # Most central node
                if deg_centrality:
                    most_central = max(deg_centrality.items(), key=lambda x: x[1])
                    metrics["most_central_node"] = most_central[0]
                    
                # Betweenness centrality (can be expensive for large graphs)
                if metrics["node_count"] < 100:
                    bet_centrality = nx.betweenness_centrality(self.graph)
                    metrics["max_betweenness_centrality"] = max(bet_centrality.values()) if bet_centrality else 0
            except:
                # Skip metrics that can't be calculated
                pass
                
        # Path metrics
        if metrics["node_count"] > 1:
            try:
                # Convert to undirected for path calculations if needed
                g = self.graph.to_undirected() if nx.is_directed(self.graph) else self.graph
                
                if nx.is_connected(g):
                    metrics["diameter"] = nx.diameter(g)
                    metrics["average_shortest_path"] = nx.average_shortest_path_length(g)
            except:
                # Skip path metrics if graph is not connected
                pass
        
        # Clustering
        if metrics["node_count"] > 2:
            try:
                metrics["average_clustering"] = nx.average_clustering(self.graph.to_undirected())
            except:
                pass
        
        # Cycles
        try:
            metrics["has_cycles"] = not nx.is_directed_acyclic_graph(self.graph) if nx.is_directed(self.graph) else True
        except:
            metrics["has_cycles"] = "unknown"
            
        return metrics
    
    def _extract_flowchart(self) -> Dict[str, Any]:
        """Extract flowchart with sophisticated detection algorithms"""
        # Initial nodes and edges
        nodes = []
        edges = []
        
        # Identify node types based on shape and text
        for i, element in enumerate(self.elements):
            if element.type not in ["line", "arrow"]:
                node_type = self._determine_flowchart_node_type(element)
                
                nodes.append({
                    "id": element.id,
                    "type": node_type,
                    "text": element.text,
                    "x": element.x,
                    "y": element.y,
                    "width": element.width,
                    "height": element.height
                })
        
        # Process edges from the graph
        for u, v, data in self.graph.edges(data=True):
            edge_element = data.get('element')
            if not edge_element:
                continue
                
            # Determine if this is a conditional branch (from decision node)
            source_type = None
            for node in nodes:
                if node["id"] == u:
                    source_type = node["type"]
                    break
            
            # Set label based on source node type and position
            label = edge_element.text if edge_element.text else ""
            
            if source_type == "decision" and not label:
                # Infer yes/no based on direction
                source_node = self.graph.nodes[u]["element"]
                target_node = self.graph.nodes[v]["element"]
                
                # Simple heuristic: right = yes, left = no
                if target_node.x > source_node.x + 10:
                    label = "Yes"
                elif target_node.x < source_node.x - 10:
                    label = "No"
            
            edges.append({
                "source": u,
                "target": v,
                "label": label,
                "id": edge_element.id
            })
        
        # Use graph analysis to identify the flow structure
        structured_nodes = self._structure_flowchart_nodes(nodes, edges)
        
        return {
            "type": "flowchart",
            "nodes": structured_nodes,
            "edges": edges
        }
    
    def _determine_flowchart_node_type(self, element: DiagramElement) -> str:
        """Determine flowchart node type based on shape and text"""
        if element.type == "diamond":
            return "decision"
        elif element.type in ["ellipse", "circle"]:
            text_lower = element.text.lower()
            if "start" in text_lower:
                return "start"
            elif "end" in text_lower:
                return "end"
            else:
                return "terminal"
        elif element.type == "rectangle":
            text_lower = element.text.lower()
            if "process" in text_lower:
                return "process"
            elif "input" in text_lower or "output" in text_lower:
                return "input_output"
            elif "document" in text_lower:
                return "document"
            else:
                return "process"  # Default for rectangles
        elif element.type == "parallelogram":
            return "input_output"
        else:
            return "process"  # Default type
    
    def _structure_flowchart_nodes(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add hierarchical structure to flowchart nodes using graph analysis"""
        # Create a temporary graph
        g = nx.DiGraph()
        for node in nodes:
            g.add_node(node["id"], **node)
        
        for edge in edges:
            g.add_edge(edge["source"], edge["target"], **edge)
        
        # Find start and end nodes
        start_nodes = [n for n in nodes if n["type"] == "start"]
        end_nodes = [n for n in nodes if n["type"] == "end"]
        
        # If no explicit start/end, try to find them by connectivity
        if not start_nodes:
            # Find nodes with no incoming edges
            for node_id in g.nodes():
                if g.in_degree(node_id) == 0 and g.out_degree(node_id) > 0:
                    for node in nodes:
                        if node["id"] == node_id:
                            node["type"] = "start"
                            start_nodes = [node]
                            break
        
        if not end_nodes:
            # Find nodes with no outgoing edges
            for node_id in g.nodes():
                if g.out_degree(node_id) == 0 and g.in_degree(node_id) > 0:
                    for node in nodes:
                        if node["id"] == node_id:
                            node["type"] = "end"
                            end_nodes = [node]
                            break
        
        # Enhanced nodes with additional structure
        enhanced_nodes = []
        for node in nodes:
            node_id = node["id"]
            enhanced_node = dict(node)  # Copy original node
            
            # Calculate levels by shortest paths from start nodes
            if start_nodes:
                start_id = start_nodes[0]["id"]
                try:
                    if nx.has_path(g, start_id, node_id):
                        path_length = nx.shortest_path_length(g, start_id, node_id)
                        enhanced_node["level"] = path_length
                    else:
                        enhanced_node["level"] = -1
                except:
                    enhanced_node["level"] = -1
            else:
                enhanced_node["level"] = -1
            
            # Identify branches and loops
            if node["type"] == "decision":
                # Analyze outgoing edges to detect loops and branches
                out_edges = list(g.out_edges(node_id, data=True))
                branch_structure = {"branches": [], "loop": False}
                
                # Check if this is part of a loop
                if node_id in nx.simple_cycles(g):
                    branch_structure["loop"] = True
                
                # Analyze branches
                for _, target, edge_data in out_edges:
                    branch_info = {
                        "target": target,
                        "label": edge_data.get("label", ""),
                        "is_back_edge": False
                    }
                    
                    # Check if this is a back edge (going to an earlier node)
                    if "level" in enhanced_node and enhanced_node["level"] >= 0:
                        for n in enhanced_nodes:
                            if n["id"] == target and n["level"] <= enhanced_node["level"]:
                                branch_info["is_back_edge"] = True
                                break
                    
                    branch_structure["branches"].append(branch_info)
                
                enhanced_node["branch_structure"] = branch_structure
            
            enhanced_nodes.append(enhanced_node)
        
        return enhanced_nodes
    
    def _extract_uml_class_diagram(self) -> Dict[str, Any]:
        """Extract UML class diagram with advanced parsing"""
        classes = []
        relationships = []
        
        # Extract classes with detailed parsing
        for element in self.elements:
            if element.type == "rectangle":
                class_info = self._parse_uml_class(element)
                if class_info:
                    classes.append(class_info)
        
        # Extract relationships from the graph
        for u, v, data in self.graph.edges(data=True):
            rel_element = data.get('element')
            if not rel_element:
                continue
            
            # Determine relationship type
            rel_type = self._determine_uml_relationship_type(rel_element)
            
            # Find class names
            source_name = None
            target_name = None
            
            for cls in classes:
                if cls["id"] == u:
                    source_name = cls["name"]
                if cls["id"] == v:
                    target_name = cls["name"]
            
            if source_name and target_name:
                relationships.append({
                    "source": source_name,
                    "target": target_name,
                    "type": rel_type,
                    "label": rel_element.text,
                    "id": rel_element.id
                })
        
        # Use graph analysis to identify inheritance hierarchies
        inheritance_hierachy = self._analyze_inheritance_hierarchy(classes, relationships)
        
        return {
            "type": "uml_class_diagram",
            "classes": classes,
            "relationships": relationships,
            "inheritance_hierarchy": inheritance_hierachy
        }
    
    def _parse_uml_class(self, element: DiagramElement) -> Optional[Dict[str, Any]]:
        """Parse a UML class from a rectangle element using regex"""
        text = element.text.strip()
        if not text:
            return None
            
        # Regular expression patterns for class parsing
        class_patterns = [
            # Standard 3-compartment class: Class Name -- Attributes -- Methods
            r'^\s*(?P<name>[^{\n]+)(?:\s*{(?P<stereotype>[^}]+)})?\s*(?P<sections>.+)$',
            
            # Simple class pattern
            r'^\s*(?P<name>[^{\n]+)(?:\s*{(?P<stereotype>[^}]+)})?\s*$',
        ]
        
        # Try different patterns
        class_match = None
        for pattern in class_patterns:
            class_match = re.search(pattern, text, re.DOTALL)
            if class_match:
                break
                
        if not class_match:
            # Can't parse as class
            return None
        
        # Get class name and stereotype
        class_name = class_match.group('name').strip()
        stereotype = class_match.group('stereotype').strip() if 'stereotype' in class_match.groupdict() and class_match.group('stereotype') else ""
        
        # Default for abstract
        is_abstract = 'abstract' in stereotype.lower() or class_name.strip().startswith('<<abstract>>') or 'abstract' in class_name.lower()
        
        # Process sections (attributes and methods)
        attributes = []
        methods = []
        
        if 'sections' in class_match.groupdict() and class_match.group('sections'):
            sections_text = class_match.group('sections')
            
            # Split on separator lines (---, ===, ___, etc.)
            sections = re.split(r'\s*[-=_]{2,}\s*', sections_text)
            
            if len(sections) >= 2:  # At least attributes and methods sections
                # Process attributes (first section)
                for line in sections[0].strip().split('\n'):
                    if line.strip():
                        attributes.append(self._parse_uml_attribute(line))
                
                # Process methods (second section)
                for line in sections[1].strip().split('\n'):
                    if line.strip():
                        methods.append(self._parse_uml_method(line))
        
        return {
            "id": element.id,
            "name": class_name,
            "stereotype": stereotype,
            "is_abstract": is_abstract,
            "attributes": attributes,
            "methods": methods,
            "x": element.x,
            "y": element.y
        }
    
    def _parse_uml_attribute(self, attr_text: str) -> Dict[str, str]:
        """Parse a UML class attribute"""
        # Remove surrounding whitespace
        attr_text = attr_text.strip()
        
        # Look for visibility marker
        visibility = "package"  # default
        if attr_text.startswith('+'):
            visibility = "public"
            attr_text = attr_text[1:].strip()
        elif attr_text.startswith('-'):
            visibility = "private"
            attr_text = attr_text[1:].strip()
        elif attr_text.startswith('#'):
            visibility = "protected"
            attr_text = attr_text[1:].strip()
        elif attr_text.startswith('~'):
            visibility = "package"
            attr_text = attr_text[1:].strip()
        
        # Parse name and type
        type_match = re.search(r'^(.*?)\s*:\s*(.+)$', attr_text)
        if type_match:
            name = type_match.group(1).strip()
            attr_type = type_match.group(2).strip()
        else:
            name = attr_text
            attr_type = ""
        
        # Look for default value
        default_value = ""
        if '=' in name:
            name_parts = name.split('=', 1)
            name = name_parts[0].strip()
            default_value = name_parts[1].strip()
        
        return {
            "name": name,
            "type": attr_type,
            "visibility": visibility,
            "default_value": default_value
        }
    
    def _parse_uml_method(self, method_text: str) -> Dict[str, Any]:
        """Parse a UML class method"""
        # Remove surrounding whitespace
        method_text = method_text.strip()
        
        # Look for visibility marker
        visibility = "package"  # default
        if method_text.startswith('+'):
            visibility = "public"
            method_text = method_text[1:].strip()
        elif method_text.startswith('-'):
            visibility = "private"
            method_text = method_text[1:].strip()
        elif method_text.startswith('#'):
            visibility = "protected"
            method_text = method_text[1:].strip()
        elif method_text.startswith('~'):
            visibility = "package"
            method_text = method_text[1:].strip()
        
        # Check for abstract/static markers
        is_abstract = False
        is_static = False
        
        if '{abstract}' in method_text:
            is_abstract = True
            method_text = method_text.replace('{abstract}', '').strip()
        
        if '{static}' in method_text or 'static' in method_text:
            is_static = True
            method_text = method_text.replace('{static}', '').replace('static', '').strip()
        
        # Parse name, parameters, and return type
        # Method format: name(param1: type, param2: type): return_type
        method_match = re.search(r'^(.*?)\((.*?)\)(?:\s*:\s*(.+))?$', method_text)
        
        if method_match:
            name = method_match.group(1).strip()
            params_text = method_match.group(2).strip()
            return_type = method_match.group(3).strip() if method_match.group(3) else ""
            
            # Parse parameters
            parameters = []
            if params_text:
                # Split on commas but respect parentheses
                params = []
                current_param = ""
                paren_level = 0
                
                for char in params_text:
                    if char == '(':
                        paren_level += 1
                        current_param += char
                    elif char == ')':
                        paren_level -= 1
                        current_param += char
                    elif char == ',' and paren_level == 0:
                        params.append(current_param.strip())
                        current_param = ""
                    else:
                        current_param += char
                
                if current_param:
                    params.append(current_param.strip())
                
                # Parse each parameter
                for param in params:
                    if ':' in param:
                        param_parts = param.split(':', 1)
                        param_name = param_parts[0].strip()
                        param_type = param_parts[1].strip()
                    else:
                        param_name = param
                        param_type = ""
                    
                    parameters.append({
                        "name": param_name,
                        "type": param_type
                    })
        else:
            # Couldn't parse method, treat as name only
            name = method_text
            parameters = []
            return_type = ""
        
        return {
            "name": name,
            "parameters": parameters,
            "return_type": return_type,
            "visibility": visibility,
            "is_abstract": is_abstract,
            "is_static": is_static
        }
    
    def _determine_uml_relationship_type(self, element: DiagramElement) -> str:
        """Determine UML relationship type based on line style and arrow head"""
        # Default to association
        rel_type = "association"
        
        # Check line style
        if element.style == "dashed":
            rel_type = "dependency"
        
        # Check arrow type
        if element.arrow_type == "triangle":
            rel_type = "inheritance"
        elif element.arrow_type == "diamond":
            rel_type = "aggregation" if not element.arrow_fill else "composition"
        
        # Check for specialized relationships in text
        if element.text:
            text_lower = element.text.lower()
            if "use" in text_lower:
                rel_type = "dependency"
            elif "implement" in text_lower:
                rel_type = "implementation"
            elif "extend" in text_lower:
                rel_type = "inheritance"
            elif "has" in text_lower or "contains" in text_lower:
                rel_type = "aggregation" if "aggregation" in text_lower else "composition"
        
        return rel_type
    
    def _analyze_inheritance_hierarchy(self, classes: List[Dict[str, Any]], 
                                      relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze class inheritance hierarchy"""
        # Create a graph for inheritance only
        inheritance_graph = nx.DiGraph()
        
        # Add all classes as nodes
        for cls in classes:
            inheritance_graph.add_node(cls["name"], **cls)
        
        # Add inheritance relationships as edges
        for rel in relationships:
            if rel["type"] in ["inheritance", "implementation"]:
                inheritance_graph.add_edge(rel["source"], rel["target"], **rel)
        
        # Identify root classes (no incoming inheritance edges)
        root_classes = [node for node in inheritance_graph.nodes() 
                       if inheritance_graph.in_degree(node) == 0 and inheritance_graph.out_degree(node) > 0]
        
        # Build hierarchy tree
        hierarchy = {}
        for root in root_classes:
            hierarchy[root] = self._build_hierarchy_tree(root, inheritance_graph)
        
        # Detect multiple inheritance
        multiple_inheritance = []
        for node in inheritance_graph.nodes():
            if inheritance_graph.in_degree(node) > 1:
                parents = list(inheritance_graph.predecessors(node))
                multiple_inheritance.append({"class": node, "parents": parents})
        
        return {
            "root_classes": root_classes,
            "hierarchy": hierarchy,
            "multiple_inheritance": multiple_inheritance
        }
    
    def _build_hierarchy_tree(self, root: str, graph: nx.DiGraph) -> Dict[str, Any]:
        """Recursively build a hierarchy tree from a graph starting at root"""
        children = {}
        for child in graph.successors(root):
            children[child] = self._build_hierarchy_tree(child, graph)
        return children
    
    def _extract_uml_sequence_diagram(self) -> Dict[str, Any]:
        """Extract UML sequence diagram with temporal analysis"""
        actors = []
        lifelines = []
        messages = []
        
        # Find actors (usually rectangles or actor shapes at the top)
        for element in self.elements:
            if element.type in ["rectangle", "actor"] and element.y < 100:
                actor_name = element.text if element.text else f"Actor{len(actors)+1}"
                
                actors.append({
                    "id": element.id,
                    "name": actor_name,
                    "x": element.x,
                    "y": element.y
                })
                
                # Look for associated lifeline (vertical line)
                for line_elem in self.elements:
                    if (line_elem.type == "line" and 
                        abs(line_elem.x1 - element.x) < 15 and 
                        abs(line_elem.x2 - line_elem.x) < 15 and
                        line_elem.y1 > element.y):
                        
                        lifelines.append({
                            "id": line_elem.id,
                            "actor_id": element.id,
                            "x": element.x,
                            "y1": line_elem.y1,
                            "y2": line_elem.y2
                        })
        
        # Extract messages (arrows between lifelines)
        for element in self.elements:
            if element.type == "arrow":
                # Find nearest lifelines to arrow endpoints
                source_actor = None
                target_actor = None
                
                min_src_dist = float('inf')
                min_tgt_dist = float('inf')
                
                for actor in actors:
                    # Distance to source point
                    src_dist = abs(actor["x"] - element.x1)
                    if src_dist < min_src_dist:
                        min_src_dist = src_dist
                        source_actor = actor
                    
                    # Distance to target point
                    tgt_dist = abs(actor["x"] - element.x2)
                    if tgt_dist < min_tgt_dist:
                        min_tgt_dist = tgt_dist
                        target_actor = actor
                
                if source_actor and target_actor and min_src_dist < 30 and min_tgt_dist < 30:
                    messages.append({
                        "id": element.id,
                        "source_id": source_actor["id"],
                        "target_id": target_actor["id"],
                        "message": element.text if element.text else "message",
                        "y": (element.y1 + element.y2) / 2,
                        "is_return": element.style == "dashed",
                        "is_async": element.arrow_type == "open"
                    })
        
        # Sort messages by vertical position (time)
        messages.sort(key=lambda m: m["y"])
        
        # Identify activation boxes
        activations = self._identify_sequence_activations(actors, messages, lifelines)
        
        # Identify fragments/combined fragments (alt, opt, loop, etc.)
        fragments = self._identify_sequence_fragments()
        
        # Build interaction model (full sequence including implied activations)
        interaction_model = self._build_sequence_interaction_model(actors, messages, activations)
        
        return {
            "type": "uml_sequence_diagram",
            "actors": actors,
            "messages": messages,
            "lifelines": lifelines,
            "activations": activations,
            "fragments": fragments,
            "interaction_model": interaction_model
        }
    
    def _identify_sequence_activations(self, actors, messages, lifelines) -> List[Dict[str, Any]]:
        """Identify activation boxes in sequence diagram"""
        activations = []
        
        # Look for actual activation boxes (rectangles on lifelines)
        for element in self.elements:
            if element.type == "rectangle":
                # Find if this rectangle is on a lifeline
                for lifeline in lifelines:
                    if abs(element.x - lifeline["x"]) < 10 and element.height > 20:
                        # This is likely an activation box
                        activations.append({
                            "id": element.id,
                            "actor_id": lifeline["actor_id"],
                            "x": element.x,
                            "y1": element.y - element.height/2,
                            "y2": element.y + element.height/2
                        })
        
        # Infer activations from messages if needed
        if not activations and messages:
            # Create implicit activations
            for i, msg in enumerate(messages):
                # Find next return message or next message from this actor
                end_y = 1000  # Default to a large value
                
                for next_msg in messages[i+1:]:
                    if next_msg["is_return"] and next_msg["source_id"] == msg["target_id"] and next_msg["target_id"] == msg["source_id"]:
                        end_y = next_msg["y"]
                        break
                    elif next_msg["source_id"] == msg["target_id"]:
                        # Another message is sent from this actor
                        end_y = next_msg["y"] - 10
                        break
                
                # Check if we already have an activation for this target at this time
                has_activation = False
                for act in activations:
                    if act["actor_id"] == msg["target_id"] and act["y1"] <= msg["y"] <= act["y2"]:
                        has_activation = True
                        break
                
                if not has_activation:
                    # Find the actor's x position
                    actor_x = next((a["x"] for a in actors if a["id"] == msg["target_id"]), None)
                    if actor_x is not None:
                        activations.append({
                            "id": f"implicit_activation_{len(activations)}",
                            "actor_id": msg["target_id"],
                            "x": actor_x,
                            "y1": msg["y"],
                            "y2": min(end_y, msg["y"] + 100),
                            "is_implicit": True
                        })
        
        return activations
    
    def _identify_sequence_fragments(self) -> List[Dict[str, Any]]:
        """Identify combined fragments in sequence diagram (alt, opt, loop)"""
        fragments = []
        
        # Look for rectangles with fragment-like text that are larger than activations
        for element in self.elements:
            if element.type == "rectangle" and element.width > 100 and element.height > 50:
                text_lower = element.text.lower() if element.text else ""
                
                # Check for fragment type keywords
                fragment_type = "region"
                for keyword, ftype in [
                    ("alt", "alternative"), ("opt", "optional"), 
                    ("loop", "loop"), ("par", "parallel"),
                    ("region", "region"), ("ref", "reference"),
                    ("critical", "critical"), ("neg", "negative"),
                    ("break", "break"), ("strict", "strict")
                ]:
                    if keyword in text_lower:
                        fragment_type = ftype
                        break
                
                # If this looks like a fragment, add it
                if fragment_type != "region" or "[" in text_lower or ":" in text_lower:
                    # Find potential guard condition
                    guard = ""
                    guard_match = re.search(r'\[(.*?)\]', text_lower)
                    if guard_match:
                        guard = guard_match.group(1).strip()
                    
                    fragments.append({
                        "id": element.id,
                        "type": fragment_type,
                        "guard": guard,
                        "text": element.text,
                        "x": element.x,
                        "y": element.y,
                        "width": element.width,
                        "height": element.height
                    })
        
        # Sort fragments by size (smaller fragments might be nested in larger ones)
        fragments.sort(key=lambda f: f["width"] * f["height"], reverse=True)
        
        # Detect nesting structure
        for i, fragment in enumerate(fragments):
            fragment["parent_id"] = None
            fragment["children"] = []
            
            # Check if this fragment is contained by any larger fragment
            for j, parent in enumerate(fragments):
                if i != j:
                    if (parent["x"] - parent["width"]/2 <= fragment["x"] - fragment["width"]/2 and
                        parent["x"] + parent["width"]/2 >= fragment["x"] + fragment["width"]/2 and
                        parent["y"] - parent["height"]/2 <= fragment["y"] - fragment["height"]/2 and
                        parent["y"] + parent["height"]/2 >= fragment["y"] + fragment["height"]/2):
                        
                        fragment["parent_id"] = parent["id"]
                        parent["children"].append(fragment["id"])
                        break
        
        return fragments
    
    def _build_sequence_interaction_model(self, actors, messages, activations) -> Dict[str, Any]:
        """Build a complete interaction model of the sequence"""
        # Create a record of actor activations (active periods)
        actor_timelines = {}
        
        for actor in actors:
            actor_timelines[actor["id"]] = {
                "name": actor["name"],
                "active_periods": []
            }
        
        # Add activation periods
        for activation in activations:
            if activation["actor_id"] in actor_timelines:
                actor_timelines[activation["actor_id"]]["active_periods"].append({
                    "start_y": activation["y1"],
                    "end_y": activation["y2"],
                    "messages_sent": [],
                    "messages_received": []
                })
        
        # Add messages to the activation periods
        for message in messages:
            # Find the source activation
            source_activations = actor_timelines.get(message["source_id"], {}).get("active_periods", [])
            for activation in source_activations:
                if activation["start_y"] <= message["y"] <= activation["end_y"]:
                    activation["messages_sent"].append(message["id"])
                    break
            
            # Find the target activation
            target_activations = actor_timelines.get(message["target_id"], {}).get("active_periods", [])
            for activation in target_activations:
                if activation["start_y"] <= message["y"] <= activation["end_y"]:
                    activation["messages_received"].append(message["id"])
                    break
        
        # Sort messages into a time-ordered sequence
        time_ordered_sequence = []
        for message in sorted(messages, key=lambda m: m["y"]):
            time_ordered_sequence.append({
                "type": "message",
                "id": message["id"],
                "source": message["source_id"],
                "target": message["target_id"],
                "text": message["message"],
                "is_return": message["is_return"],
                "y": message["y"]
            })
        
        # Add activation start/end events interspersed with messages
        for actor_id, timeline in actor_timelines.items():
            for activation in timeline["active_periods"]:
                # Add activation start
                time_ordered_sequence.append({
                    "type": "activation_start",
                    "actor_id": actor_id,
                    "y": activation["start_y"]
                })
                
                # Add activation end
                time_ordered_sequence.append({
                    "type": "activation_end",
                    "actor_id": actor_id,
                    "y": activation["end_y"]
                })
        
        # Re-sort everything by y-position
        time_ordered_sequence.sort(key=lambda e: e["y"])
        
        return {
            "actor_timelines": actor_timelines,
            "sequence": time_ordered_sequence
        }
    
    def _extract_er_diagram(self) -> Dict[str, Any]:
        """Extract ER diagram with semantic relationship analysis"""
        entities = []
        relationships = []
        
        # Extract entities (rectangles)
        for element in self.elements:
            if element.type == "rectangle":
                entity_name = element.text.split('\n')[0].strip() if element.text else f"Entity{len(entities)+1}"
                
                # Parse attributes from remaining text
                attributes = []
                is_weak_entity = False
                primary_key = None
                
                if element.text and '\n' in element.text:
                    lines = element.text.split('\n')[1:]
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Check for PK/FK markers
                        is_pk = "pk" in line.lower() or "primary key" in line.lower()
                        is_fk = "fk" in line.lower() or "foreign key" in line.lower()
                        
                        # Extract attribute name
                        attr_name = line
                        attr_type = ""
                        
                        # Check for type information
                        if ":" in line:
                            parts = line.split(":", 1)
                            attr_name = parts[0].strip()
                            attr_type = parts[1].strip()
                        
                        # Remove markers from attribute name
                        attr_name = re.sub(r'\b(pk|fk|primary key|foreign key)\b', '', attr_name, flags=re.IGNORECASE).strip()
                        
                        # Create attribute entry
                        attr = {
                            "name": attr_name,
                            "type": attr_type,
                            "is_primary_key": is_pk,
                            "is_foreign_key": is_fk
                        }
                        
                        attributes.append(attr)
                        
                        # Track primary key
                        if is_pk:
                            primary_key = attr_name
                
                # Check if this is a weak entity (often has double border or "weak" in text)
                if element.style == "double" or (element.text and "weak" in element.text.lower()):
                    is_weak_entity = True
                
                entities.append({
                    "id": element.id,
                    "name": entity_name,
                    "attributes": attributes,
                    "is_weak_entity": is_weak_entity,
                    "primary_key": primary_key,
                    "x": element.x,
                    "y": element.y
                })
        
        # Extract relationships from diamonds
        for element in self.elements:
            if element.type in ["diamond", "rhombus"]:
                rel_name = element.text if element.text else f"Relationship{len(relationships)+1}"
                
                # Find connected entities via the graph
                connected_entities = []
                
                # Check incident edges in the graph
                for u, v, data in self.graph.edges(data=True):
                    edge_element = data.get('element')
                    
                    # Check if this edge connects to our diamond
                    if edge_element:
                        if u == element.id:
                            # Diamond is source, v is target
                            connected_entities.append({
                                "entity_id": v,
                                "cardinality": edge_element.text,
                                "role": "to"
                            })
                        elif v == element.id:
                            # u is source, diamond is target
                            connected_entities.append({
                                "entity_id": u,
                                "cardinality": edge_element.text,
                                "role": "from"
                            })
                
                # Only add if we found at least two connected entities
                if len(connected_entities) >= 2:
                    # Determine if this is an identifying relationship (for weak entities)
                    is_identifying = False
                    weak_entity_found = False
                    
                    for conn in connected_entities:
                        for entity in entities:
                            if entity["id"] == conn["entity_id"] and entity["is_weak_entity"]:
                                weak_entity_found = True
                                break
                    
                    # For a relationship to be identifying, it must have a weak entity and
                    # often has a double-bordered diamond or "identifying" in its text
                    if weak_entity_found and (element.style == "double" or (element.text and "identifying" in element.text.lower())):
                        is_identifying = True
                    
                    relationships.append({
                        "id": element.id,
                        "name": rel_name,
                        "connected_entities": connected_entities,
                        "is_identifying": is_identifying,
                        "x": element.x,
                        "y": element.y
                    })
        
        # Also check for direct relationships (lines between entities without diamonds)
        if len(relationships) == 0:
            for u, v, data in self.graph.edges(data=True):
                edge_element = data.get('element')
                
                # Check if both endpoints are entities
                u_is_entity = any(e["id"] == u for e in entities)
                v_is_entity = any(e["id"] == v for e in entities)
                
                if u_is_entity and v_is_entity and edge_element:
                    # This is a direct relationship between entities
                    rel_name = edge_element.text if edge_element.text else f"Relates_to_{len(relationships)+1}"
                    
                    relationships.append({
                        "id": edge_element.id,
                        "name": rel_name,
                        "connected_entities": [
                            {"entity_id": u, "cardinality": "", "role": "from"},
                            {"entity_id": v, "cardinality": "", "role": "to"}
                        ],
                        "is_identifying": False,
                        "is_direct": True
                    })
        
        # Parse cardinalities
        for relationship in relationships:
            for entity_conn in relationship["connected_entities"]:
                cardinality = entity_conn.get("cardinality", "")
                
                if not cardinality:
                    continue
                
                # Try to parse cardinality notation
                cardinality_type = "unknown"
                min_value = 0
                max_value = "*"
                
                # Check for N:M, 1:N, etc.
                if re.search(r'\d+\s*:\s*\d+', cardinality) or re.search(r'\d+\s*:\s*[nNmM*]', cardinality):
                    cardinality_type = "ratio"
                # Check for min..max notation
                elif ".." in cardinality:
                    cardinality_type = "range"
                    parts = cardinality.split("..")
                    if len(parts) == 2:
                        try:
                            min_value = int(parts[0].strip()) if parts[0].strip() != "" else 0
                            max_value = parts[1].strip()
                            if max_value.isdigit():
                                max_value = int(max_value)
                        except:
                            pass
                # Check for crow's foot style text descriptions
                elif "one" in cardinality.lower() and "many" in cardinality.lower():
                    cardinality_type = "one_to_many"
                    min_value = 0
                    max_value = "*"
                elif "one" in cardinality.lower() and "one" in cardinality.lower():
                    cardinality_type = "one_to_one"
                    min_value = 1
                    max_value = 1
                elif "many" in cardinality.lower() and "many" in cardinality.lower():
                    cardinality_type = "many_to_many"
                    min_value = 0
                    max_value = "*"
                
                # Update the entity connection with parsed cardinality
                entity_conn["cardinality_type"] = cardinality_type
                entity_conn["min"] = min_value
                entity_conn["max"] = max_value
        
        # Enriched response with additional relationship analysis
        normalized_schema = self._create_normalized_er_schema(entities, relationships)
        
        return {
            "type": "er_diagram",
            "entities": entities,
            "relationships": relationships,
            "normalized_schema": normalized_schema
        }
    
    def _create_normalized_er_schema(self, entities: List[Dict[str, Any]], 
                                    relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a normalized database schema from ER diagram"""
        tables = []
        foreign_keys = []
        
        # First, create tables for each entity
        for entity in entities:
            # Gather all attributes and identify the primary key
            attributes = []
            primary_keys = []
            
            for attr in entity["attributes"]:
                attributes.append({
                    "name": attr["name"],
                    "type": attr["type"],
                    "nullable": not attr["is_primary_key"]
                })
                
                if attr["is_primary_key"]:
                    primary_keys.append(attr["name"])
            
            # Create the table
            tables.append({
                "name": entity["name"],
                "attributes": attributes,
                "primary_keys": primary_keys,
                "source": "entity"
            })
        
        # Process relationships to determine if they need their own tables
        for relationship in relationships:
            connected_entities = relationship["connected_entities"]
            
            if len(connected_entities) < 2:
                continue
                
            # Determine relationship type based on cardinalities
            is_many_to_many = False
            
            # Check if any entity has max cardinality > 1
            many_count = 0
            for conn in connected_entities:
                max_card = conn.get("max", "*")
                if max_card == "*" or (isinstance(max_card, int) and max_card > 1):
                    many_count += 1
            
            is_many_to_many = many_count >= 2
            
            if is_many_to_many:
                # Many-to-many relationships need a junction table
                junction_table_name = relationship["name"]
                
                # Create primary key columns from connected entities
                attributes = []
                primary_keys = []
                
                for conn in connected_entities:
                    # Find the entity
                    entity = next((e for e in entities if e["id"] == conn["entity_id"]), None)
                    if entity:
                        # Find primary key of this entity
                        entity_pk = next((attr["name"] for attr in entity["attributes"] 
                                        if attr.get("is_primary_key")), None)
                        
                        if entity_pk:
                            fk_col_name = f"{entity['name'].lower()}_{entity_pk}"
                            attributes.append({
                                "name": fk_col_name,
                                "type": "FOREIGN KEY",
                                "nullable": False
                            })
                            primary_keys.append(fk_col_name)
                            
                            # Add foreign key constraint
                            foreign_keys.append({
                                "table": junction_table_name,
                                "column": fk_col_name,
                                "references_table": entity["name"],
                                "references_column": entity_pk
                            })
                
                # Add any attributes that might belong to the relationship itself
                # (This is a simplification - would need to extract from diagram if available)
                
                tables.append({
                    "name": junction_table_name,
                    "attributes": attributes,
                    "primary_keys": primary_keys,
                    "source": "relationship"
                })
            else:
                # One-to-many or one-to-one: add foreign key to the "many" side
                many_side = None
                one_side = None
                
                for conn in connected_entities:
                    max_card = conn.get("max", "*")
                    if max_card == "*" or (isinstance(max_card, int) and max_card > 1):
                        many_side = conn
                    else:
                        one_side = conn
                
                if many_side and one_side:
                    # Find the entities
                    many_entity = next((e for e in entities if e["id"] == many_side["entity_id"]), None)
                    one_entity = next((e for e in entities if e["id"] == one_side["entity_id"]), None)
                    
                    if many_entity and one_entity:
                        # Find primary key of the "one" side
                        one_pk = next((attr["name"] for attr in one_entity["attributes"] 
                                      if attr.get("is_primary_key")), None)
                        
                        if one_pk:
                            fk_col_name = f"{one_entity['name'].lower()}_{one_pk}"
                            
                            # Find the corresponding table for the many entity
                            for table in tables:
                                if table["name"] == many_entity["name"]:
                                    # Add the foreign key column
                                    table["attributes"].append({
                                        "name": fk_col_name,
                                        "type": "FOREIGN KEY",
                                        "nullable": True
                                    })
                                    
                                    # Add foreign key constraint
                                    foreign_keys.append({
                                        "table": many_entity["name"],
                                        "column": fk_col_name,
                                        "references_table": one_entity["name"],
                                        "references_column": one_pk
                                    })
                                    
                                    break
        
        return {
            "tables": tables,
            "foreign_keys": foreign_keys
        }
    
    def _interpret_generic_diagram(self) -> Dict[str, Any]:
        """Interpret a general diagram without specific known type"""
        # Extract nodes and connections using graph structure
        nodes = []
        connections = []
        
        # Process nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if 'element' in node_data:
                element = node_data['element']
                
                nodes.append({
                    "id": element.id,
                    "type": element.type,
                    "text": element.text,
                    "x": element.x,
                    "y": element.y,
                    "width": element.width,
                    "height": element.height
                })
        
        # Process connections
        for u, v, data in self.graph.edges(data=True):
            if 'element' in data:
                element = data['element']
                
                connections.append({
                    "id": element.id,
                    "source": u,
                    "target": v,
                    "label": element.text,
                    "type": "directed" if element.type == "arrow" else "undirected",
                    "style": element.style
                })
        
        # Try to identify clusters or groups
        clusters = self._identify_clusters(nodes)
        
        # Analyze connection patterns
        patterns = self._analyze_connection_patterns(nodes, connections)
        
        return {
            "type": "generic_diagram",
            "nodes": nodes,
            "connections": connections,
            "clusters": clusters,
            "patterns": patterns
        }
    
    def _identify_clusters(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify clusters of nodes using spatial proximity"""
        if len(nodes) < 3:
            return []
            
        # Extract coordinates for clustering
        points = np.array([[node["x"], node["y"]] for node in nodes])
        
        try:
            # Use DBSCAN clustering
            from sklearn.cluster import DBSCAN
            
            # Compute a reasonable epsilon based on diagram scale
            coords_range = np.max(points, axis=0) - np.min(points, axis=0)
            eps = np.min(coords_range) * 0.1  # 10% of the smaller dimension
            
            clustering = DBSCAN(eps=eps, min_samples=2).fit(points)
            
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Create cluster objects
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) >= 2:
                    # Compute cluster center
                    cluster_points = points[cluster_indices]
                    center = np.mean(cluster_points, axis=0)
                    
                    # Get node IDs in this cluster
                    cluster_node_ids = [nodes[idx]["id"] for idx in cluster_indices]
                    
                    clusters.append({
                        "id": f"cluster_{i}",
                        "nodes": cluster_node_ids,
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        "size": len(cluster_indices)
                    })
            
            return clusters
        except:
            # Fallback if sklearn is not available or clustering fails
            return []
    
    def _analyze_connection_patterns(self, nodes: List[Dict[str, Any]], 
                                   connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in diagram connections"""
        patterns = {}
        
        # Create a temporary directed graph
        g = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            g.add_node(node["id"], **node)
        
        # Add edges
        for conn in connections:
            g.add_edge(conn["source"], conn["target"], **conn)
        
        # Check for hierarchical structure
        try:
            patterns["is_hierarchical"] = nx.is_directed_acyclic_graph(g)
        except:
            patterns["is_hierarchical"] = False
        
        # Check for cycles
        try:
            patterns["contains_cycles"] = not nx.is_directed_acyclic_graph(g)
            if patterns["contains_cycles"]:
                patterns["cycle_count"] = len(list(nx.simple_cycles(g)))
        except:
            patterns["contains_cycles"] = "unknown"
        
        # Check for star pattern (central hub)
        if len(g) > 3:
            degree_centrality = nx.degree_centrality(g)
            if max(degree_centrality.values()) > 0.5:  # Central node connected to >50% of other nodes
                patterns["has_central_hub"] = True
                # Find the hub node
                hub_node = max(degree_centrality.items(), key=lambda x: x[1])[0]
                patterns["hub_node"] = hub_node
            else:
                patterns["has_central_hub"] = False
        
        # Check for chain pattern
        if len(g) > 2:
            patterns["is_chain"] = all(d <= 2 for d in dict(g.degree()).values())
        
        # Check for mesh pattern (highly connected)
        if len(g) > 3:
            density = nx.density(g)
            patterns["is_dense_mesh"] = density > 0.5
            patterns["graph_density"] = density
        
        # Check for tree structure
        if len(g) > 2:
            try:
                patterns["is_tree"] = nx.is_tree(g.to_undirected())
            except:
                patterns["is_tree"] = False
        
        # Check for bipartite structure (two groups of nodes with connections only between groups)
        if len(g) > 3:
            try:
                patterns["is_bipartite"] = nx.is_bipartite(g.to_undirected())
            except:
                patterns["is_bipartite"] = False
        
        return patterns


# Enhanced Main Diagramming System with Extended Capabilities
class EnhancedDiagramSystem:
    """Advanced diagram interpretation system with computer vision support"""
    
    def __init__(self):
        self.analyzer = DiagramAnalyzer()
        self.cv_processor = None
        try:
            # Initialize computer vision based element extractor if OpenCV is available
            self.cv_processor = CVDiagramProcessor()
        except:
            pass
    
    def extract_flowchart(self, image_data: str) -> Dict[str, Any]:
        """Extract flowchart structure from image data"""
        elements = self._extract_visual_elements(image_data)
        return self.analyzer.analyze(elements)
    
    def interpret_diagram_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interpret generic diagram elements"""
        return self.analyzer.analyze(elements)
    
    def _extract_visual_elements(self, image_data: str) -> List[Dict[str, Any]]:
        """Extract visual elements from image data"""
        if self.cv_processor:
            # Use computer vision to extract elements
            return self.cv_processor.extract_elements(image_data)
        else:
            # Fall back to provided elements
            raise ValueError("Computer vision processing not available and no elements provided")

    def extract_entities_from_structure(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entity details from analyzed diagram structure"""
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
        
        # Extract from actors (sequence diagrams)
        if "actors" in structure:
            for actor in structure["actors"]:
                entities.append({
                    "name": actor["name"] if isinstance(actor, dict) else actor,
                    "type": "actor",
                    "source": "structure"
                })
        
        # Extract relationships/connections
        if "relationships" in structure:
            for rel in structure["relationships"]:
                entities.append({
                    "name": rel.get("name", ""),
                    "type": "relationship",
                    "source_entity": rel.get("source", rel.get("entity1", "")),
                    "target_entity": rel.get("target", rel.get("entity2", "")),
                    "source": "structure"
                })
        elif "edges" in structure or "connections" in structure:
            connections = structure.get("edges", structure.get("connections", []))
            for conn in connections:
                entities.append({
                    "name": conn.get("label", "connection"),
                    "type": "connection",
                    "source_entity": conn.get("source", ""),
                    "target_entity": conn.get("target", ""),
                    "source": "structure"
                })
        
        return entities


# Computer Vision Processor for Image Diagrams
class CVDiagramProcessor:
    """Processes diagram images to extract elements using computer vision"""
    
    def __init__(self):
        """Initialize the CV processor"""
        self.image = None
        self.gray_image = None
        self.binary_image = None
        self.contours = None
    
    def extract_elements(self, image_data: str) -> List[Dict[str, Any]]:
        """Extract diagram elements from image data"""
        # Load image
        self._load_image(image_data)
        
        # Preprocess image
        self._preprocess_image()
        
        # Detect shapes
        self._detect_shapes()
        
        # Detect lines and arrows
        self._detect_lines_and_arrows()
        
        # Extract text using OCR
        self._extract_text()
        
        # Process and return elements
        return self._process_detected_elements()
    
    def _load_image(self, image_data: str):
        """Load image from data"""
        # In real implementation, decode image data and load with OpenCV
        # This is a placeholder
        pass
    
    def _preprocess_image(self):
        """Preprocess image for better element detection"""
        # Apply filters and transformations for shape detection
        pass
    
    def _detect_shapes(self):
        """Detect shapes in the image"""
        # Use contour detection and shape recognition
        pass
    
    def _detect_lines_and_arrows(self):
        """Detect lines and arrows in the image"""
        # Use Hough line detection and specialized arrow detection
        pass
    
    def _extract_text(self):
        """Extract text from the image using OCR"""
        # In a real implementation, use tesseract or other OCR engine
        pass
    
    def _process_detected_elements(self) -> List[Dict[str, Any]]:
        """Process detected elements and return structured list"""
        # Convert raw CV detections to structured elements
        # Placeholder for real implementation
        return []


# Example Usage
def analyze_diagram(elements):
    """Analyze a diagram from provided elements"""
    system = EnhancedDiagramSystem()
    result = system.interpret_diagram_elements(elements)
    return result
