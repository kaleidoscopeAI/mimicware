#!/usr/bin/env python3
"""
Advanced Graph-Based Workflow Engine for System Builder
Handles script dependencies, execution order, and parallelization
"""

import os
import sys
import math
import time
import logging
import threading
import multiprocessing
import queue
import uuid
import json
import heapq
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from enum import Enum, auto
import ctypes
from ctypes import c_int, c_void_p, c_char_p, POINTER, Structure, CDLL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WorkflowEngine")

# ======================================================================
# Data Structures and Types
# ======================================================================

class NodeType(Enum):
    """Types of nodes in the workflow graph"""
    SCRIPT = auto()
    DATA = auto()
    RESOURCE = auto()
    COMPUTATION = auto()
    CONDITION = auto()
    MERGE = auto()

class NodeState(Enum):
    """Execution states for nodes"""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()

class EdgeType(Enum):
    """Types of edges in the workflow graph"""
    CONTROL = auto()
    DATA = auto()
    RESOURCE = auto()

@dataclass
class NodeMetadata:
    """Metadata for workflow nodes"""
    id: str
    name: str
    type: NodeType
    properties: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class ExecutionMetadata:
    """Execution metadata for workflow nodes"""
    state: NodeState = NodeState.PENDING
    start_time: float = 0.0
    end_time: float = 0.0
    attempts: int = 0
    max_attempts: int = 1
    assigned_worker: Optional[int] = None
    result: Any = None
    error: Optional[str] = None

@dataclass
class Node:
    """Node in the workflow graph"""
    metadata: NodeMetadata
    execution: ExecutionMetadata = field(default_factory=ExecutionMetadata)
    dependencies: Dict[str, EdgeType] = field(default_factory=dict)
    dependents: Dict[str, EdgeType] = field(default_factory=dict)
    
    @property
    def id(self) -> str:
        return self.metadata.id
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def type(self) -> NodeType:
        return self.metadata.type
    
    @property
    def state(self) -> NodeState:
        return self.execution.state
    
    @state.setter
    def state(self, value: NodeState):
        self.execution.state = value
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id

class WorkflowGraph:
    """Graph representation of a workflow"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.nx_graph = nx.DiGraph()
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        if node.id in self.nodes:
            raise ValueError(f"Node with ID {node.id} already exists")
        
        self.nodes[node.id] = node
        self.nx_graph.add_node(node.id, node=node)
    
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        """Add an edge between nodes"""
        if source_id not in self.nodes:
            raise ValueError(f"Source node {source_id} not found")
        if target_id not in self.nodes:
            raise ValueError(f"Target node {target_id} not found")
        
        # Add to networkx graph
        self.nx_graph.add_edge(source_id, target_id, type=edge_type)
        
        # Update node dependency tracking
        source_node = self.nodes[source_id]
        target_node = self.nodes[target_id]
        
        source_node.dependents[target_id] = edge_type
        target_node.dependencies[source_id] = edge_type
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        # Get the node
        node = self.nodes[node_id]
        
        # Remove edges from dependent nodes
        for dep_id in list(node.dependents.keys()):
            if dep_id in self.nodes:
                dep_node = self.nodes[dep_id]
                if node_id in dep_node.dependencies:
                    del dep_node.dependencies[node_id]
        
        # Remove edges from dependency nodes
        for dep_id in list(node.dependencies.keys()):
            if dep_id in self.nodes:
                dep_node = self.nodes[dep_id]
                if node_id in dep_node.dependents:
                    del dep_node.dependents[node_id]
        
        # Remove from networkx graph
        self.nx_graph.remove_node(node_id)
        
        # Remove from nodes dict
        del self.nodes[node_id]
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_dependencies(self, node_id: str) -> Dict[str, EdgeType]:
        """Get node dependencies"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        return self.nodes[node_id].dependencies
    
    def get_dependents(self, node_id: str) -> Dict[str, EdgeType]:
        """Get node dependents"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        return self.nodes[node_id].dependents
    
    def is_cyclical(self) -> bool:
        """Check if the graph contains cycles"""
        try:
            nx.find_cycle(self.nx_graph)
            return True
        except nx.NetworkXNoCycle:
            return False
    
    def get_roots(self) -> List[Node]:
        """Get all root nodes (nodes with no dependencies)"""
        return [node for node in self.nodes.values() if not node.dependencies]
    
    def get_leaves(self) -> List[Node]:
        """Get all leaf nodes (nodes with no dependents)"""
        return [node for node in self.nodes.values() if not node.dependents]
    
    def calculate_critical_path(self) -> Tuple[List[str], int]:
        """Calculate the critical path through the graph"""
        if self.is_cyclical():
            raise ValueError("Cannot calculate critical path for cyclical graph")
        
        # Create a weighted graph for critical path calculation
        weighted_graph = nx.DiGraph()
        
        for node_id, node in self.nodes.items():
            # Use execution time as weight, or 1 if unknown
            weight = 1
            if hasattr(node.metadata, 'estimated_time'):
                weight = node.metadata.estimated_time
            
            weighted_graph.add_node(node_id, weight=weight)
        
        for source, target, data in self.nx_graph.edges(data=True):
            weighted_graph.add_edge(source, target, weight=1)
        
        # Find the critical path
        # We use a dynamic programming approach
        
        # Get a topological sort
        topo_order = list(nx.topological_sort(weighted_graph))
        
        # Initialize distances
        dist = {node: 0 for node in weighted_graph.nodes()}
        
        # Map to store the predecessor for each node on the critical path
        pred = {node: None for node in weighted_graph.nodes()}
        
        # Calculate the longest path (critical path)
        for node in topo_order:
            # If we haven't processed any nodes yet, set distance to the node's weight
            if dist[node] == 0:
                dist[node] = weighted_graph.nodes[node]['weight']
            
            # Look at all the nodes this one points to
            for successor in weighted_graph.successors(node):
                # Calculate new distance
                new_dist = dist[node] + weighted_graph.nodes[successor]['weight']
                
                # If this path is longer, update
                if new_dist > dist[successor]:
                    dist[successor] = new_dist
                    pred[successor] = node
        
        # Find the node with the maximum distance
        end_node = max(dist, key=dist.get)
        path_length = dist[end_node]
        
        # Reconstruct the path
        path = []
        while end_node is not None:
            path.append(end_node)
            end_node = pred[end_node]
        
        # Reverse to get path from start to end
        path.reverse()
        
        return path, path_length
    
    def topological_sort(self) -> List[str]:
        """Get a topological sort of the graph"""
        if self.is_cyclical():
            raise ValueError("Cannot perform topological sort on cyclical graph")
        
        return list(nx.topological_sort(self.nx_graph))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to a serializable dictionary"""
        result = {
            "nodes": {},
            "edges": []
        }
        
        for node_id, node in self.nodes.items():
            # Convert node to dict
            node_dict = {
                "id": node.id,
                "name": node.name,
                "type": node.type.name,
                "properties": node.metadata.properties,
                "tags": node.metadata.tags,
                "state": node.state.name
            }
            
            result["nodes"][node_id] = node_dict
        
        for source, target, data in self.nx_graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target,
                "type": data["type"].name
            }
            
            result["edges"].append(edge_dict)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowGraph':
        """Create a graph from a dictionary"""
        graph = cls()
        
        # Create nodes
        for node_id, node_data in data["nodes"].items():
            node_type = NodeType[node_data["type"]]
            
            metadata = NodeMetadata(
                id=node_data["id"],
                name=node_data["name"],
                type=node_type,
                properties=node_data.get("properties", {}),
                tags=node_data.get("tags", [])
            )
            
            execution = ExecutionMetadata(
                state=NodeState[node_data.get("state", "PENDING")]
            )
            
            node = Node(
                metadata=metadata,
                execution=execution
            )
            
            graph.add_node(node)
        
        # Create edges
        for edge_data in data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            edge_type = EdgeType[edge_data["type"]]
            
            graph.add_edge(source, target, edge_type)
        
        return graph
    
    def visualize(self, filepath: str = None) -> None:
        """Visualize the graph using NetworkX and matplotlib"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import rgb2hex
            
            plt.figure(figsize=(12, 8))
            
            # Define colors for different node types
            node_colors = {
                NodeType.SCRIPT: (0.2, 0.4, 0.8),
                NodeType.DATA: (0.8, 0.2, 0.2),
                NodeType.RESOURCE: (0.2, 0.8, 0.2),
                NodeType.COMPUTATION: (0.8, 0.8, 0.2),
                NodeType.CONDITION: (0.8, 0.4, 0.1),
                NodeType.MERGE: (0.4, 0.8, 0.8)
            }
            
            # Define colors for different node states
            state_colors = {
                NodeState.PENDING: (0.8, 0.8, 0.8),
                NodeState.READY: (0.8, 0.8, 0.2),
                NodeState.RUNNING: (0.2, 0.6, 0.8),
                NodeState.COMPLETED: (0.2, 0.8, 0.2),
                NodeState.FAILED: (0.8, 0.2, 0.2),
                NodeState.SKIPPED: (0.6, 0.6, 0.6)
            }
            
            # Create position layout
            pos = nx.spring_layout(self.nx_graph, seed=42)
            
            # Draw nodes with colors based on type and state
            node_colors_list = []
            node_border_colors = []
            
            for node_id in self.nx_graph.nodes():
                node = self.nodes[node_id]
                # Blend type and state colors
                type_color = node_colors.get(node.type, (0.5, 0.5, 0.5))
                state_color = state_colors.get(node.state, (0.5, 0.5, 0.5))
                
                # Node fill color based on type
                node_colors_list.append(rgb2hex(type_color))
                
                # Border color based on state
                node_border_colors.append(rgb2hex(state_color))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.nx_graph, pos,
                node_color=node_colors_list,
                edgecolors=node_border_colors,
                linewidths=2.0,
                node_size=500
            )
            
            # Draw edges
            edge_colors = []
            for _, _, edge_data in self.nx_graph.edges(data=True):
                edge_type = edge_data.get('type', EdgeType.CONTROL)
                if edge_type == EdgeType.CONTROL:
                    edge_colors.append('black')
                elif edge_type == EdgeType.DATA:
                    edge_colors.append('blue')
                else:  # EdgeType.RESOURCE
                    edge_colors.append('green')
            
            nx.draw_networkx_edges(
                self.nx_graph, pos,
                arrows=True,
                arrowsize=20,
                width=1.5,
                edge_color=edge_colors
            )
            
            # Draw node labels
            labels = {node_id: self.nodes[node_id].name for node_id in self.nx_graph.nodes()}
            nx.draw_networkx_labels(self.nx_graph, pos, labels=labels, font_size=10)
            
            # Create legend
            plt.figlegend(
                [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=rgb2hex(color), markersize=10) 
                 for _, color in node_colors.items()],
                [node_type.name for node_type in node_colors.keys()],
                title="Node Types",
                loc="upper left"
            )
            
            plt.figlegend(
                [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                           markeredgecolor=rgb2hex(color), markeredgewidth=2, markersize=10) 
                 for _, color in state_colors.items()],
                [state.name for state in state_colors.keys()],
                title="Node States",
                loc="upper right"
            )
            
            plt.figlegend(
                [plt.Line2D([0], [0], color=c, linewidth=2) for c in ['black', 'blue', 'green']],
                [edge_type.name for edge_type in EdgeType],
                title="Edge Types",
                loc="lower left"
            )
            
            plt.axis('off')
            plt.tight_layout()
            
            if filepath:
                plt.savefig(filepath)
                plt.close()
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Visualization requires matplotlib. Install with: pip install matplotlib")

# ======================================================================
# Workflow Execution Engine
# ======================================================================

@dataclass
class WorkflowTask:
    """Task for the workflow engine to execute"""
    node_id: str
    priority: int = 0
    
    def __lt__(self, other):
        if not isinstance(other, WorkflowTask):
            return NotImplemented
        return self.priority > other.priority  # Higher priority first

class WorkerThread(threading.Thread):
    """Worker thread for executing workflow tasks"""
    
    def __init__(self, worker_id: int, task_queue: queue.Queue, 
                result_queue: queue.Queue, graph: WorkflowGraph, 
                context: Dict[str, Any], stop_event: threading.Event):
        super().__init__()
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.graph = graph
        self.context = context
        self.stop_event = stop_event
        self.daemon = True
    
    def run(self):
        """Main worker thread loop"""
        logger.info(f"Worker {self.worker_id} started")
        
        while not self.stop_event.is_set():
            try:
                # Get a task with timeout to check stop_event periodically
                task = self.task_queue.get(timeout=0.5)
                
                # Process the task
                success, result, error = self.process_task(task)
                
                # Return the result
                self.result_queue.put((task.node_id, success, result, error))
                
                # Mark task as done
                self.task_queue.task_done()
                
            except queue.Empty:
                # No tasks available, just continue
                continue
            except Exception as e:
                logger.error(f"Worker {self.worker_id} encountered an error: {e}")
                # Put a failed result
                self.result_queue.put((task.node_id if 'task' in locals() else None, 
                                     False, None, str(e)))
                
                # Mark task as done if we got one
                if 'task' in locals():
                    self.task_queue.task_done()
        
        logger.info(f"Worker {self.worker_id} stopped")
    
    def process_task(self, task: WorkflowTask) -> Tuple[bool, Any, Optional[str]]:
        """Process a workflow task"""
        # Get the node
        node = self.graph.get_node(task.node_id)
        if not node:
            return False, None, f"Node {task.node_id} not found"
        
        # Mark as running
        node.state = NodeState.RUNNING
        node.execution.start_time = time.time()
        node.execution.assigned_worker = self.worker_id
        
        logger.info(f"Worker {self.worker_id} processing node {node.name} ({node.id})")
        
        try:
            # Execute based on node type
            if node.type == NodeType.SCRIPT:
                result = self.execute_script(node)
            elif node.type == NodeType.COMPUTATION:
                result = self.execute_computation(node)
            elif node.type == NodeType.CONDITION:
                result = self.execute_condition(node)
            elif node.type == NodeType.DATA:
                result = self.process_data(node)
            elif node.type == NodeType.RESOURCE:
                result = self.manage_resource(node)
            elif node.type == NodeType.MERGE:
                result = self.merge_results(node)
            else:
                return False, None, f"Unsupported node type: {node.type}"
            
            # Record execution time
            node.execution.end_time = time.time()
            node.execution.result = result
            
            return True, result, None
            
        except Exception as e:
            # Record failure
            node.execution.end_time = time.time()
            node.execution.error = str(e)
            
            return False, None, str(e)
    
    def execute_script(self, node: Node) -> Any:
        """Execute a script node"""
        # Get script properties
        script_path = node.metadata.properties.get('path')
        if not script_path:
            raise ValueError(f"Script node {node.id} missing path property")
        
        script_type = node.metadata.properties.get('script_type', 'python')
        
        # Prepare command
        cmd = []
        if script_type == 'python':
            cmd = [sys.executable, script_path]
        elif script_type in ('shell', 'bash'):
            cmd = ['bash', script_path]
        elif script_type == 'node':
            cmd = ['node', script_path]
        else:
            # Assume it's an executable
            cmd = [script_path]
        
        # Add arguments if specified
        args = node.metadata.properties.get('args', [])
        cmd.extend(args)
        
        # Get environment variables
        env = os.environ.copy()
        env_vars = node.metadata.properties.get('env', {})
        env.update(env_vars)
        
        # Execute the script
        import subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.path.dirname(script_path)
        )
        
        stdout, stderr = process.communicate()
        
        # Check return code
        if process.returncode != 0:
            raise RuntimeError(f"Script failed with code {process.returncode}: {stderr.decode()}")
        
        # Return stdout as result
        return {
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'return_code': process.returncode
        }
    
    def execute_computation(self, node: Node) -> Any:
        """Execute a computation node"""
        # Get computation function
        function_name = node.metadata.properties.get('function')
        if not function_name:
            raise ValueError(f"Computation node {node.id} missing function property")
        
        # Get input data
        inputs = {}
        for dep_id, edge_type in node.dependencies.items():
            if edge_type == EdgeType.DATA:
                dep_node = self.graph.get_node(dep_id)
                if dep_node and dep_node.execution.result is not None:
                    inputs[dep_node.name] = dep_node.execution.result
        
        # Get function parameters
        params = node.metadata.properties.get('params', {})
        
        # Try to get the function from context
        func = self.context.get('functions', {}).get(function_name)
        if not func:
            # Try to import and find the function
            module_name, func_name = function_name.rsplit('.', 1)
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Function {function_name} not found: {e}")
        
        # Execute the function
        return func(inputs=inputs, params=params, context=self.context)
    
    def execute_condition(self, node: Node) -> bool:
        """Execute a condition node"""
        # Get condition expression
        expression = node.metadata.properties.get('expression')
        if not expression:
            raise ValueError(f"Condition node {node.id} missing expression property")
        
        # Get input data
        inputs = {}
        for dep_id, edge_type in node.dependencies.items():
            if edge_type == EdgeType.DATA:
                dep_node = self.graph.get_node(dep_id)
                if dep_node and dep_node.execution.result is not None:
                    inputs[dep_node.name] = dep_node.execution.result
        
        # Add inputs to evaluation context
        eval_context = {'inputs': inputs}
        eval_context.update(self.context.get('constants', {}))
        
        # Evaluate the expression
        result = eval(expression, {'__builtins__': {}}, eval_context)
        
        return bool(result)
    
    def process_data(self, node: Node) -> Any:
        """Process a data node"""
        # Get data source type
        source_type = node.metadata.properties.get('source_type', 'static')
        
        if source_type == 'static':
            # Return static data
            return node.metadata.properties.get('data')
        
        elif source_type == 'file':
            # Read from file
            file_path = node.metadata.properties.get('file_path')
            if not file_path:
                raise ValueError(f"Data node {node.id} missing file_path property")
            
            format_type = node.metadata.properties.get('format', 'text')
            
            # Read the file
            with open(file_path, 'r' if format_type != 'binary' else 'rb') as f:
                if format_type == 'text':
                    return f.read()
                elif format_type == 'json':
                    import json
                    return json.load(f)
                elif format_type == 'csv':
                    import csv
                    return list(csv.reader(f))
                else:  # binary
                    return f.read()
        
        elif source_type == 'input':
            # Get data from input dependencies
            inputs = {}
            for dep_id, edge_type in node.dependencies.items():
                if edge_type == EdgeType.DATA:
                    dep_node = self.graph.get_node(dep_id)
                    if dep_node and dep_node.execution.result is not None:
                        inputs[dep_node.name] = dep_node.execution.result
            
            # Apply transformation if specified
            transform = node.metadata.properties.get('transform')
            if transform:
                # Get transformation function
                if transform in self.context.get('transforms', {}):
                    func = self.context['transforms'][transform]
                    return func(inputs)
                else:
                    # Try to evaluate as a lambda
                    lambda_func = eval(transform, {'__builtins__': {}})
                    return lambda_func(inputs)
            
            return inputs
        
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    def manage_resource(self, node: Node) -> Any:
        """Manage a resource node"""
        # Get resource type
        resource_type = node.metadata.properties.get('resource_type')
        if not resource_type:
            raise ValueError(f"Resource node {node.id} missing resource_type property")
        
        # Handle different resource types
        if resource_type == 'file':
            # File resource
            file_path = node.metadata.properties.get('path')
            if not file_path:
                raise ValueError(f"File resource node {node.id} missing path property")
            
            operation = node.metadata.properties.get('operation', 'check')
            
            if operation == 'check':
                # Check if file exists
                return os.path.exists(file_path)
            
            elif operation == 'create':
                # Create an empty file or directory
                if node.metadata.properties.get('is_directory', False):
                    os.makedirs(file_path, exist_ok=True)
                else:
                    with open(file_path, 'w') as f:
                        pass
                return True
            
            elif operation == 'delete':
                # Delete file or directory
                if os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                elif os.path.exists(file_path):
                    os.remove(file_path)
                return True
            
            else:
                raise ValueError(f"Unsupported file operation: {operation}")
        
        elif resource_type == 'lock':
            # Lock resource for synchronization
            lock_name = node.metadata.properties.get('name', node.id)
            operation = node.metadata.properties.get('operation', 'acquire')
            
            # Get lock from context
            locks = self.context.setdefault('locks', {})
            if lock_name not in locks:
                locks[lock_name] = threading.Lock()
            
            lock = locks[lock_name]
            
            if operation == 'acquire':
                # Acquire the lock
                acquired = lock.acquire(timeout=node.metadata.properties.get('timeout', 10))
                if not acquired:
                    raise TimeoutError(f"Failed to acquire lock {lock_name}")
                return True
            
            elif operation == 'release':
                # Release the lock
                try:
                    lock.release()
                    return True
                except RuntimeError:
                    # Lock wasn't acquired, ignore
                    return False
            
            else:
                raise ValueError(f"Unsupported lock operation: {operation}")
        
        else:
            raise ValueError(f"Unsupported resource type: {resource_type}")
    
    def merge_results(self, node: Node) -> Any:
        """Merge results from multiple inputs"""
        # Get merge strategy
        strategy = node.metadata.properties.get('strategy', 'dict')
        
        # Collect input data
        inputs = {}
        for dep_id, edge_type in node.dependencies.items():
            if edge_type == EdgeType.DATA:
                dep_node = self.graph.get_node(dep_id)
                if dep_node and dep_node.execution.result is not None:
                    inputs[dep_node.name] = dep_node.execution.result
        
        # Apply merge strategy
        if strategy == 'dict':
            # Merge as dictionary
            result = {}
            for name, value in inputs.items():
                result[name] = value
            return result
        
        elif strategy == 'list':
            # Merge as list
            return list(inputs.values())
        
        elif strategy == 'concat':
            # Concatenate strings
            return ''.join(str(v) for v in inputs.values())
        
        elif strategy == 'sum':
            # Sum values
            return sum(inputs.values())
        
        elif strategy == 'custom':
            # Custom merge function
            merge_func_name = node.metadata.properties.get('merge_function')
            if not merge_func_name:
                raise ValueError(f"Merge node {node.id} with custom strategy missing merge_function property")
            
            # Get merge function
            merge_func = self.context.get('merge_functions', {}).get(merge_func_name)
            if not merge_func:
                raise ValueError(f"Merge function {merge_func_name} not found")
            
            return merge_func(inputs)
        
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")

class WorkflowEngine:
    """Engine for executing workflow graphs"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or min(4, multiprocessing.cpu_count())
        self.graph = None
        self.context = {}
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.stop_event = threading.Event()
        self.result_thread = None
        self.on_node_complete = None
        self.on_node_failed = None
    
    def set_graph(self, graph: WorkflowGraph) -> None:
        """Set the workflow graph to execute"""
        self.graph = graph
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set the execution context"""
        self.context = context
    
    def set_callbacks(self, on_complete: Callable = None, on_failed: Callable = None) -> None:
        """Set callback functions for node completion and failure"""
        self.on_node_complete = on_complete
        self.on_node_failed = on_failed
    
    def _result_processor(self):
        """Thread for processing execution results"""
        while not self.stop_event.is_set():
            try:
                # Get a result with timeout
                node_id, success, result, error = self.result_queue.get(timeout=0.5)
                
                if node_id is None:
                    # Skip invalid results
                    self.result_queue.task_done()
                    continue
                
                # Get the node
                node = self.graph.get_node(node_id)
                if not node:
                    logger.error(f"Result for unknown node {node_id}")
                    self.result_queue.task_done()
                    continue
                
                # Update node state
                if success:
                    node.state = NodeState.COMPLETED
                    node.execution.result = result
                    
                    # Call completion callback
                    if self.on_node_complete:
                        try:
                            self.on_node_complete(node)
                        except Exception as e:
                            logger.error(f"Error in on_node_complete callback: {e}")
                    
                    # Check if any dependent nodes are ready
                    self._update_dependent_nodes(node)
                    
                else:
                    node.state = NodeState.FAILED
                    node.execution.error = error
                    
                    # Call failure callback
                    if self.on_node_failed:
                        try:
                            self.on_node_failed(node)
                        except Exception as e:
                            logger.error(f"Error in on_node_failed callback: {e}")
                    
                    # Handle retries
                    if node.execution.attempts < node.execution.max_attempts:
                        # Retry the node
                        node.execution.attempts += 1
                        node.state = NodeState.READY
                        self.task_queue.put(WorkflowTask(node_id=node.id, priority=10))  # High priority for retries
                        logger.info(f"Retrying node {node.name} ({node.id}), attempt {node.execution.attempts}")
                    else:
                        # No more retries, propagate failure to dependents
                        self._handle_node_failure(node)
                
                # Mark result as processed
                self.result_queue.task_done()
                
            except queue.Empty:
                # No results yet, continue
                continue
            except Exception as e:
                logger.error(f"Error in result processor: {e}")
    
    def _update_dependent_nodes(self, node: Node) -> None:
        """Update state of nodes that depend on the completed node"""
        for dep_id in node.dependents:
            dep_node = self.graph.get_node(dep_id)
            if not dep_node:
                continue
            
            # Skip nodes that are already completed, failed, or running
            if dep_node.state in (NodeState.COMPLETED, NodeState.FAILED, NodeState.RUNNING):
                continue
            
            # Check if all dependencies are completed
            ready = True
            for upstream_id in dep_node.dependencies:
                upstream = self.graph.get_node(upstream_id)
                if not upstream or upstream.state != NodeState.COMPLETED:
                    ready = False
                    break
            
            if ready:
                # Node is ready for execution
                if dep_node.state != NodeState.READY:
                    dep_node.state = NodeState.READY
                    
                    # For condition nodes, adjust priority based on critical path
                    priority = 0
                    if dep_node.type == NodeType.CONDITION:
                        priority = 5  # Higher priority for conditions
                    
                    self.task_queue.put(WorkflowTask(node_id=dep_id, priority=priority))
                    logger.info(f"Node {dep_node.name} ({dep_id}) is ready for execution")
    
    def _handle_node_failure(self, node: Node) -> None:
        """Handle a node failure by updating dependent nodes"""
        logger.warning(f"Node {node.name} ({node.id}) failed: {node.execution.error}")
        
        # Propagate failure to dependent nodes
        for dep_id in node.dependents:
            dep_node = self.graph.get_node(dep_id)
            if not dep_node:
                continue
            
            # Skip nodes that are already completed or failed
            if dep_node.state in (NodeState.COMPLETED, NodeState.FAILED):
                continue
            
            # Mark node as failed due to dependency failure
            dep_node.state = NodeState.FAILED
            dep_node.execution.error = f"Dependency {node.name} failed"
            
            # Recursively propagate failure
            self._handle_node_failure(dep_node)
    
    def start(self) -> None:
        """Start the workflow engine"""
        if not self.graph:
            raise ValueError("No workflow graph set")
        
        if self.is_running():
            raise RuntimeError("Workflow engine is already running")
        
        # Reset stop event
        self.stop_event.clear()
        
        # Check if graph has cycles
        if self.graph.is_cyclical():
            raise ValueError("Workflow graph contains cycles")
        
        # Reset task and result queues
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        
        # Start worker threads
        self.workers = []
        for i in range(self.num_workers):
            worker = WorkerThread(
                worker_id=i,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                graph=self.graph,
                context=self.context,
                stop_event=self.stop_event
            )
            worker.start()
            self.workers.append(worker)
        
        # Start result processor thread
        self.result_thread = threading.Thread(target=self._result_processor)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        # Initialize node states
        for node in self.graph.nodes.values():
            node.execution.state = NodeState.PENDING
            node.execution.start_time = 0
            node.execution.end_time = 0
            node.execution.attempts = 0
            node.execution.assigned_worker = None
            node.execution.result = None
            node.execution.error = None
        
        # Find start nodes (nodes with no dependencies)
        start_nodes = self.graph.get_roots()
        if not start_nodes:
            raise ValueError("Workflow graph has no start nodes")
        
        # Queue start nodes
        for node in start_nodes:
            node.state = NodeState.READY
            self.task_queue.put(WorkflowTask(node_id=node.id))
        
        logger.info(f"Workflow engine started with {len(start_nodes)} start nodes")
    
    def stop(self) -> None:
        """Stop the workflow engine"""
        if not self.is_running():
            return
        
        # Set stop event to signal threads to stop
        self.stop_event.set()
        
        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Wait for result thread to terminate
        if self.result_thread:
            self.result_thread.join(timeout=5.0)
        
        # Clear thread references
        self.workers = []
        self.result_thread = None
        
        logger.info("Workflow engine stopped")
    
    def is_running(self) -> bool:
        """Check if the workflow engine is running"""
        return bool(self.workers) and not self.stop_event.is_set()
    
    def wait(self, timeout: float = None) -> bool:
        """Wait for workflow execution to complete"""
        if not self.is_running():
            return True
        
        start_time = time.time()
        
        # Wait for task queue to be empty
        try:
            self.task_queue.join()
            self.result_queue.join()
            return True
        except KeyboardInterrupt:
            self.stop()
            return False
    
    def get_node_states(self) -> Dict[str, NodeState]:
        """Get the current state of all nodes"""
        if not self.graph:
            return {}
        
        return {node_id: node.state for node_id, node in self.graph.nodes.items()}
    
    def get_results(self) -> Dict[str, Any]:
        """Get the results of completed nodes"""
        if not self.graph:
            return {}
        
        return {
            node_id: node.execution.result 
            for node_id, node in self.graph.nodes.items() 
            if node.state == NodeState.COMPLETED
        }
    
    def get_errors(self) -> Dict[str, str]:
        """Get the errors of failed nodes"""
        if not self.graph:
            return {}
        
        return {
            node_id: node.execution.error 
            for node_id, node in self.graph.nodes.items() 
            if node.state == NodeState.FAILED and node.execution.error
        }
    
    def get_execution_times(self) -> Dict[str, float]:
        """Get the execution time of completed nodes"""
        if not self.graph:
            return {}
        
        return {
            node_id: node.execution.end_time - node.execution.start_time
            for node_id, node in self.graph.nodes.items()
            if node.state == NodeState.COMPLETED and node.execution.end_time > 0
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow execution"""
        if not self.graph:
            return {}
        
        # Count nodes in each state
        state_counts = {state: 0 for state in NodeState}
        for node in self.graph.nodes.values():
            state_counts[node.state] += 1
        
        # Calculate total execution time
        total_time = 0
        max_time = 0
        completed_nodes = 0
        
        for node in self.graph.nodes.values():
            if node.state == NodeState.COMPLETED:
                exec_time = node.execution.end_time - node.execution.start_time
                total_time += exec_time
                max_time = max(max_time, exec_time)
                completed_nodes += 1
        
        avg_time = total_time / completed_nodes if completed_nodes > 0 else 0
        
        # Get critical path
        try:
            critical_path, critical_path_length = self.graph.calculate_critical_path()
        except:
            critical_path, critical_path_length = [], 0
        
        return {
            "status": "running" if self.is_running() else "stopped",
            "states": {state.name: count for state, count in state_counts.items()},
            "total_nodes": len(self.graph.nodes),
            "completed_nodes": state_counts[NodeState.COMPLETED],
            "failed_nodes": state_counts[NodeState.FAILED],
            "pending_nodes": state_counts[NodeState.PENDING] + state_counts[NodeState.READY],
            "running_nodes": state_counts[NodeState.RUNNING],
            "skipped_nodes": state_counts[NodeState.SKIPPED],
            "total_execution_time": total_time,
            "average_execution_time": avg_time,
            "max_execution_time": max_time,
            "critical_path": critical_path,
            "critical_path_length": critical_path_length
        }

# ======================================================================
# Workflow Builder
# ======================================================================

class WorkflowBuilder:
    """Builder for creating workflow graphs"""
    
    def __init__(self):
        self.graph = WorkflowGraph()
        self.current_node_id = None
    
    def add_script(self, name: str, path: str, script_type: str = 'python', 
                 args: List[str] = None, env: Dict[str, str] = None) -> str:
        """Add a script node to the workflow"""
        node_id = str(uuid.uuid4())
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.SCRIPT,
            properties={
                'path': path,
                'script_type': script_type,
                'args': args or [],
                'env': env or {}
            }
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def add_data(self, name: str, data=None, file_path: str = None, 
               format_type: str = None, transform: str = None) -> str:
        """Add a data node to the workflow"""
        node_id = str(uuid.uuid4())
        
        properties = {}
        
        if data is not None:
            properties['source_type'] = 'static'
            properties['data'] = data
        elif file_path:
            properties['source_type'] = 'file'
            properties['file_path'] = file_path
            if format_type:
                properties['format'] = format_type
        else:
            properties['source_type'] = 'input'
            if transform:
                properties['transform'] = transform
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.DATA,
            properties=properties
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def add_computation(self, name: str, function: str, params: Dict[str, Any] = None) -> str:
        """Add a computation node to the workflow"""
        node_id = str(uuid.uuid4())
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.COMPUTATION,
            properties={
                'function': function,
                'params': params or {}
            }
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def add_condition(self, name: str, expression: str) -> str:
        """Add a condition node to the workflow"""
        node_id = str(uuid.uuid4())
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.CONDITION,
            properties={
                'expression': expression
            }
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def add_resource(self, name: str, resource_type: str, 
                   resource_params: Dict[str, Any]) -> str:
        """Add a resource node to the workflow"""
        node_id = str(uuid.uuid4())
        
        properties = {'resource_type': resource_type}
        properties.update(resource_params)
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.RESOURCE,
            properties=properties
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def add_merge(self, name: str, strategy: str = 'dict', 
                merge_function: str = None) -> str:
        """Add a merge node to the workflow"""
        node_id = str(uuid.uuid4())
        
        properties = {'strategy': strategy}
        if strategy == 'custom' and merge_function:
            properties['merge_function'] = merge_function
        
        metadata = NodeMetadata(
            id=node_id,
            name=name,
            type=NodeType.MERGE,
            properties=properties
        )
        
        node = Node(metadata=metadata)
        self.graph.add_node(node)
        
        self.current_node_id = node_id
        return node_id
    
    def connect(self, source_id: str, target_id: str, 
              edge_type: EdgeType = EdgeType.CONTROL) -> None:
        """Connect two nodes"""
        self.graph.add_edge(source_id, target_id, edge_type)
    
    def connect_from_current(self, target_id: str, 
                           edge_type: EdgeType = EdgeType.CONTROL) -> None:
        """Connect from the current node to the target node"""
        if not self.current_node_id:
            raise ValueError("No current node")
        
        self.graph.add_edge(self.current_node_id, target_id, edge_type)
    
    def connect_to_current(self, source_id: str, 
                         edge_type: EdgeType = EdgeType.CONTROL) -> None:
        """Connect from the source node to the current node"""
        if not self.current_node_id:
            raise ValueError("No current node")
        
        self.graph.add_edge(source_id, self.current_node_id, edge_type)
    
    def build(self) -> WorkflowGraph:
        """Build and return the workflow graph"""
        # Validate graph
        if self.graph.is_cyclical():
            raise ValueError("Workflow graph contains cycles")
        
        return self.graph
    
    def from_dict(self, data: Dict[str, Any]) -> WorkflowGraph:
        """Build a graph from a dictionary representation"""
        self.graph = WorkflowGraph.from_dict(data)
        return self.graph

# ======================================================================
# Quick Workflow Building Functions
# ======================================================================

def build_script_workflow(scripts: List[str], dependencies: Dict[str, List[str]] = None) -> WorkflowGraph:
    """Quickly build a workflow from a list of scripts with optional dependencies"""
    builder = WorkflowBuilder()
    
    # Map of script path to node ID
    script_nodes = {}
    
    # Add script nodes
    for script_path in scripts:
        name = os.path.basename(script_path)
        
        # Detect script type
        _, ext = os.path.splitext(script_path)
        script_type = 'python'
        if ext == '.sh' or ext == '.bash':
            script_type = 'shell'
        elif ext == '.js':
            script_type = 'node'
        
        node_id = builder.add_script(name, script_path, script_type)
        script_nodes[script_path] = node_id
    
    # Add dependencies
    if dependencies:
        for script_path, deps in dependencies.items():
            if script_path not in script_nodes:
                continue
                
            target_id = script_nodes[script_path]
            
            for dep in deps:
                if dep not in script_nodes:
                    continue
                    
                source_id = script_nodes[dep]
                builder.connect(source_id, target_id)
    
    return builder.build()

def build_parallel_workflow(tasks: List[Callable], merge_results: bool = False) -> Tuple[WorkflowGraph, WorkflowEngine]:
    """Build a workflow to execute tasks in parallel"""
    builder = WorkflowBuilder()
    
    # Create computation nodes for each task
    task_nodes = []
    for i, task in enumerate(tasks):
        # Get function name
        if hasattr(task, '__name__'):
            name = task.__name__
        else:
            name = f"Task_{i}"
        
        # Add computation node
        node_id = builder.add_computation(name, f"__tasks__.{name}")
        task_nodes.append(node_id)
    
    # Add merge node if requested
    if merge_results and task_nodes:
        merge_id = builder.add_merge("Merge_Results")
        
        # Connect task nodes to merge node
        for task_id in task_nodes:
            builder.connect(task_id, merge_id, EdgeType.DATA)
    
    graph = builder.build()
    
    # Create engine with tasks in context
    engine = WorkflowEngine()
    engine.set_graph(graph)
    
    # Add tasks to context
    task_dict = {}
    for i, task in enumerate(tasks):
        if hasattr(task, '__name__'):
            name = task.__name__
        else:
            name = f"Task_{i}"
        task_dict[name] = task
    
    engine.set_context({
        "functions": {
            "__tasks__.Task_0": tasks[0]  # Example
        },
        "__tasks__": task_dict
    })
    
    return graph, engine

# ======================================================================
# Example Usage
# ======================================================================

def example_workflow():
    """Create an example workflow"""
    # Create a builder
    builder = WorkflowBuilder()
    
    # Add script nodes
    script1 = builder.add_script("Script1", "/path/to/script1.py")
    script2 = builder.add_script("Script2", "/path/to/script2.py")
    script3 = builder.add_script("Script3", "/path/to/script3.py")
    
    # Add a condition node
    condition = builder.add_condition("Check_Result", "inputs['Script1'] == 0")
    
    # Add a computation node
    compute = builder.add_computation("Compute", "math.sqrt", {"value": 16})
    
    # Connect nodes
    builder.connect(script1, condition)
    builder.connect(condition, script2)
    builder.connect(script1, script3)
    builder.connect(script2, compute)
    builder.connect(script3, compute)
    
    # Build the graph
    graph = builder.build()
    
    # Create and start the engine
    engine = WorkflowEngine()
    engine.set_graph(graph)
    engine.start()
    
    # Wait for completion
    engine.wait()
    
    # Get results
    results = engine.get_results()
    print(f"Results: {results}")
    
    # Get errors
    errors = engine.get_errors()
    print(f"Errors: {errors}")
    
    # Stop the engine
    engine.stop()
    
    return graph, engine

if __name__ == "__main__":
    # Run the example
    graph, engine = example_workflow()
