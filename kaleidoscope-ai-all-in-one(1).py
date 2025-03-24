#!/usr/bin/env python3
# kaleidoscope_ai.py - Complete KaleidoscopeAI System with Web Visualization
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import asyncio
import logging
import pickle
import uuid
import json
import base64
import time
from io import BytesIO
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify, render_template, send_from_directory
import pennylane as qml
import ray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('kaleidoscope.log')]
)
logger = logging.getLogger("KaleidoscopeAI")

# Initialize Ray for distributed processing
ray.init(ignore_reinit_error=True)

###########################################
# CORE SYSTEM IMPLEMENTATION - DATA MODELS
###########################################

@dataclass
class Vector4D:
    x: float
    y: float
    z: float
    w: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w])

@dataclass
class Supercluster:
    position: Vector4D
    intensity: float
    connections: Set[Tuple[int, int]] = field(default_factory=set)

@dataclass
class TensorShard:
    data: torch.Tensor
    indices: torch.LongTensor
    dimension: int

@dataclass
class HyperEdge:
    vertices: List[int]
    weight: float
    dimension: int

@dataclass
class SystemState:
    embeddings: torch.Tensor
    quantum_state: torch.Tensor
    topology_state: torch.Tensor
    optimization_history: List[Dict[str, float]] = field(default_factory=list)

@dataclass
class PersistenceFeatures:
    diagrams: List[np.ndarray]
    bottleneck_distances: np.ndarray
    landscape_features: torch.Tensor
    connected_components: List[List[int]]

@dataclass
class Node:
    id: int
    memory_threshold: float
    embedded_data: torch.Tensor
    insights: List[torch.Tensor] = field(default_factory=list)
    perspective: List[torch.Tensor] = field(default_factory=list)

@dataclass
class SuperNode:
    id: int
    nodes: List[Node]
    dna: torch.Tensor
    objective: Optional[str] = None

#################################################
# HYPERCUBE CORE IMPLEMENTATION
#################################################

class HypercubeStringNetwork:
    def __init__(self, dimension: int = 4, resolution: int = 10):
        self.dimension = dimension
        self.resolution = resolution
        self.vertices = self._generate_vertices()
        self.strings = self._generate_strings()
        self.superclusters = self._find_intersections()
        
    def _generate_vertices(self) -> List[Vector4D]:
        vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    for w in [-1, 1]:
                        vertices.append(Vector4D(x, y, z, w))
        return vertices
    
    def _generate_strings(self) -> List[Tuple[Vector4D, Vector4D]]:
        strings = []
        steps = np.linspace(-1, 1, self.resolution)
        
        # Generate strings for each face pair in 4D
        for dim1 in range(4):
            for dim2 in range(dim1 + 1, 4):
                for i in steps:
                    for j in steps:
                        start = [0] * 4
                        end = [0] * 4
                        start[dim1] = i
                        start[dim2] = j
                        start[-1] = -1
                        end[dim1] = i
                        end[dim2] = j
                        end[-1] = 1
                        strings.append((
                            Vector4D(*start),
                            Vector4D(*end)
                        ))
        return strings
    
    def _find_intersections(self) -> List[Supercluster]:
        superclusters = []
        threshold = 0.1  # Distance threshold for intersection detection
        
        # O(n^2) intersection check with spatial optimization
        string_segments = np.array([[
            [s[0].x, s[0].y, s[0].z, s[0].w],
            [s[1].x, s[1].y, s[1].z, s[1].w]
        ] for s in self.strings])
        
        for i in range(len(self.strings)):
            for j in range(i + 1, len(self.strings)):
                intersection = self._compute_intersection(
                    string_segments[i],
                    string_segments[j]
                )
                if intersection is not None:
                    superclusters.append(Supercluster(
                        position=Vector4D(*intersection),
                        intensity=1.0,
                        connections={(i, j)}
                    ))
        
        return self._merge_nearby_clusters(superclusters, threshold)
    
    def _compute_intersection(self, seg1: np.ndarray, seg2: np.ndarray) -> Optional[np.ndarray]:
        # Compute closest point between two 4D line segments using linear algebra
        d1 = seg1[1] - seg1[0]
        d2 = seg2[1] - seg2[0]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        
        if n1 < 1e-10 or n2 < 1e-10:
            return None
            
        d1 /= n1
        d2 /= n2
        
        # Simplified 4D check using the first 3 components for cross product
        normal = np.cross(d1[:3], d2[:3])
        if np.linalg.norm(normal) < 1e-10:
            return None
            
        # Solve system of equations for intersection parameters
        A = np.vstack((d1, -d2)).T
        b = seg2[0] - seg1[0]
        
        try:
            # Use least squares to find parameters
            t, s = np.linalg.lstsq(A, b, rcond=None)[0]
            if 0 <= t <= n1 and 0 <= s <= n2:
                return seg1[0] + t * d1
        except:
            return None
            
        return None
    
    def _merge_nearby_clusters(self, clusters: List[Supercluster], threshold: float) -> List[Supercluster]:
        if not clusters:
            return []
            
        merged = []
        used = set()
        
        for i, c1 in enumerate(clusters):
            if i in used:
                continue
                
            current = c1
            used.add(i)
            
            for j, c2 in enumerate(clusters[i+1:], i+1):
                if j in used:
                    continue
                    
                dist = np.sqrt(
                    (c1.position.x - c2.position.x) ** 2 +
                    (c1.position.y - c2.position.y) ** 2 +
                    (c1.position.z - c2.position.z) ** 2 +
                    (c1.position.w - c2.position.w) ** 2
                )
                
                if dist < threshold:
                    current.intensity += c2.intensity
                    current.connections.update(c2.connections)
                    used.add(j)
            
            merged.append(current)
            
        return merged
    
    def project_to_3d(self, w_slice: float = 0) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        # Project 4D strings and clusters to 3D for visualization
        string_points_3d = []
        for start, end in self.strings:
            if abs(start.w - w_slice) < 0.1 or abs(end.w - w_slice) < 0.1:
                string_points_3d.append([
                    [start.x, start.y, start.z],
                    [end.x, end.y, end.z]
                ])
                
        cluster_points_3d = []
        intensities = []
        for cluster in self.superclusters:
            if abs(cluster.position.w - w_slice) < 0.1:
                cluster_points_3d.append([
                    cluster.position.x,
                    cluster.position.y,
                    cluster.position.z
                ])
                intensities.append(cluster.intensity)
                
        return np.array(string_points_3d), np.array(cluster_points_3d), intensities
        
    def to_plotly_figure(self, w_slices: List[float] = [-0.5, 0, 0.5]) -> dict:
        fig = go.Figure()
        
        for w in w_slices:
            strings_3d, clusters_3d, intensities = self.project_to_3d(w)
            
            # Add strings
            for string in strings_3d:
                fig.add_trace(go.Scatter3d(
                    x=string[:, 0],
                    y=string[:, 1],
                    z=string[:, 2],
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                    showlegend=False
                ))
            
            # Add clusters
            if len(clusters_3d) > 0:
                fig.add_trace(go.Scatter3d(
                    x=clusters_3d[:, 0],
                    y=clusters_3d[:, 1],
                    z=clusters_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5 * np.array(intensities),
                        color=intensities,
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name=f'w = {w}'
                ))
        
        fig.update_layout(
            title='4D Hypercube Projection',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return json.loads(fig.to_json())

#################################################
# NEURAL NETWORK IMPLEMENTATION
#################################################

class KaleidoscopeEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        self.insight_generator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim//2,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=6
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        insights = self.insight_generator(encoded)
        return insights

class MirrorEngine(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.perspective_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        self.predictor = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        perspective = self.perspective_generator(x)
        predictions, _ = self.predictor(x.unsqueeze(0))
        return perspective, predictions.squeeze(0)

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
    def quantum_circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)
            
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qnode = qml.QNode(self.quantum_circuit, self.dev)
        qnode_torch = qml.qnn.TorchLayer(qnode, self.weights)
        return qnode_torch(x)

class HyperdimensionalNetwork:
    def __init__(self, 
                 dimensions: List[int],
                 vector_dimension: int = 10000):
        self.dimensions = dimensions
        self.vector_dimension = vector_dimension
        self.edges: List[HyperEdge] = []
        self.vertex_embeddings = {}
        self.basis_vectors = self._generate_basis()
        
    def _generate_basis(self) -> torch.Tensor:
        basis = torch.randn(len(self.dimensions), self.vector_dimension)
        basis = basis / basis.norm(dim=1, keepdim=True)
        return basis
        
    def add_edge(self, vertices: List[int], weight: float, dimension: int):
        edge = HyperEdge(vertices, weight, dimension)
        self.edges.append(edge)
        
    def embed_vertex(self, vertex_id: int, data: torch.Tensor):
        projected = torch.zeros(self.vector_dimension)
        for dim, basis in enumerate(self.basis_vectors):
            projection = torch.dot(data, basis)
            projected += projection * basis
        self.vertex_embeddings[vertex_id] = projected
        
    def compute_laplacian(self) -> torch.Tensor:
        N = len(self.vertex_embeddings)
        L = torch.zeros((N, N))
        
        for edge in self.edges:
            n = len(edge.vertices)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        v1, v2 = edge.vertices[i], edge.vertices[j]
                        L[v1, v2] -= edge.weight
                        L[v1, v1] += edge.weight
                        
        return L

class HyperdimensionalProcessor(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hdim: int = 10000,
                 n_dimensions: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hdim = hdim
        self.n_dimensions = n_dimensions
        
        self.network = HyperdimensionalNetwork(
            dimensions=list(range(n_dimensions)),
            vector_dimension=hdim
        )
        
        self.dimension_embeddings = nn.Parameter(
            torch.randn(n_dimensions, hdim)
        )
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim)
        )
        
    def project_to_dimension(self,
                           x: torch.Tensor,
                           dimension: int) -> torch.Tensor:
        base_projection = self.projector(x)
        dim_embedding = self.dimension_embeddings[dimension]
        return base_projection * dim_embedding
        
    def compute_spectral_features(self,
                                laplacian: torch.Tensor,
                                k: int = 10) -> torch.Tensor:
        values = torch.ones(k)
        vectors = torch.zeros((laplacian.size(0), k))
        
        # Non-destructive fallback to numpy for eigenvectors
        try:
            np_laplacian = laplacian.detach().numpy()
            eigenvalues, eigenvectors = eigsh(np_laplacian, k=k, which='SM')
            values = torch.from_numpy(eigenvalues).float()
            vectors = torch.from_numpy(eigenvectors).float()
        except:
            logger.warning("Fallback to identity for spectral features")
            
        return vectors
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Project to each dimension
        dimension_projections = []
        for dim in range(self.n_dimensions):
            proj = self.project_to_dimension(x, dim)
            dimension_projections.append(proj)
            
        # Build hypergraph structure
        for i in range(batch_size):
            self.network.embed_vertex(i, x[i])
            
        # Add hyperedges
        for dim in range(self.n_dimensions):
            similarities = torch.cosine_similarity(
                dimension_projections[dim].unsqueeze(1),
                dimension_projections[dim].unsqueeze(0),
                dim=2
            )
            edges = torch.nonzero(similarities > 0.5)
            
            for edge in edges:
                if edge.size(0) > 1:  # Valid edge
                    self.network.add_edge(
                        vertices=edge.tolist(),
                        weight=similarities[edge[0], edge[1]].item(),
                        dimension=dim
                    )
                
        # Compute spectral features
        try:
            laplacian = self.network.compute_laplacian()
            spectral_features = self.compute_spectral_features(laplacian)
        except:
            logger.warning("Error in computing spectral features, using fallback")
            spectral_features = torch.zeros((batch_size, 10))
        
        # Combine features
        combined_projections = torch.stack(dimension_projections, dim=1)
        
        return combined_projections, spectral_features

class SuperNodeProcessor(nn.Module):
    def __init__(self, hdim: int = 10000, n_heads: int = 8):
        super().__init__()
        self.hdim = hdim
        self.n_heads = n_heads
        
        self.node_encoder = nn.Sequential(
            nn.Linear(hdim, hdim * 2),
            nn.ReLU(),
            nn.Linear(hdim * 2, hdim)
        )
        
        self.cluster_encoder = nn.Sequential(
            nn.Linear(hdim, hdim * 2),
            nn.ReLU(),
            nn.Linear(hdim * 2, hdim)
        )
        
        self.attention = nn.MultiheadAttention(hdim, n_heads)
        
    def forward(self, nodes: List[torch.Tensor]) -> torch.Tensor:
        # Encode individual nodes
        node_embeddings = torch.stack([
            self.node_encoder(node) for node in nodes
        ])
        
        # Apply attention across nodes
        attended, _ = self.attention(
            node_embeddings,
            node_embeddings,
            node_embeddings
        )
        
        # Final cluster encoding
        cluster_embedding = self.cluster_encoder(attended.mean(dim=0))
        
        return cluster_embedding

#################################################
# CORE KALEIDOSCOPE AI IMPLEMENTATION
#################################################

class KaleidoscopeAI:
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dim: int = 1024,
                 chatbot_model: str = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kaleidoscope = KaleidoscopeEngine(input_dim, hidden_dim)
        self.mirror = MirrorEngine(input_dim, hidden_dim)
        self.environment = HypercubeStringNetwork()
        self.nodes: List[Node] = []
        self.supernodes: List[SuperNode] = []
        self.tokenizer = None
        self.chatbot = None
        
        # Initialize hyperdimensional processor
        self.hdprocessor = HyperdimensionalProcessor(input_dim)
        
        # Initialize quantum features
        self.quantum_layer = QuantumLayer(n_qubits=8, n_layers=4)
        
        # Initialize processing queues
        self.data_queue = asyncio.Queue()
        self.insight_queue = asyncio.Queue()
        self.perspective_queue = asyncio.Queue()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.kaleidoscope.parameters()) + 
            list(self.mirror.parameters()),
            lr=0.001
        )
        
    def calculate_node_requirements(self, data_size: int) -> Tuple[int, float]:
        total_memory = data_size * 8  # Bytes to bits
        target_insights = int(np.sqrt(data_size))
        num_nodes = max(1, int(np.ceil(total_memory / (target_insights * self.input_dim))))
        memory_per_node = total_memory / num_nodes
        return num_nodes, memory_per_node
        
    def initialize_nodes(self, num_nodes: int, memory_threshold: float):
        self.nodes = [
            Node(
                id=i,
                memory_threshold=memory_threshold,
                embedded_data=torch.zeros(self.input_dim),
                insights=[],
                perspective=[]
            )
            for i in range(num_nodes)
        ]
        
    def process_data_chunk(self, node: Node, data_chunk: torch.Tensor):
        if not isinstance(data_chunk, torch.Tensor):
            data_chunk = torch.tensor(data_chunk, dtype=torch.float32)
            
        if len(data_chunk.shape) == 1:
            data_chunk = data_chunk.unsqueeze(0)
            
        # Ensure proper dimensions
        if data_chunk.shape[-1] != self.input_dim:
            data_chunk = nn.functional.pad(
                data_chunk, 
                (0, self.input_dim - data_chunk.shape[-1])
            )
            
        if node.embedded_data.norm() + data_chunk.norm() <= node.memory_threshold:
            node.embedded_data += data_chunk.sum(dim=0)
            if node.embedded_data.norm() >= 0.8 * node.memory_threshold:
                light_insights = self.kaleidoscope(node.embedded_data.unsqueeze(0))
                self.insight_queue.put_nowait(light_insights)
                node.embedded_data = torch.zeros_like(node.embedded_data)
                
    async def run_engines(self):
        while True:
            if not self.insight_queue.empty():
                insights = await self.insight_queue.get()
                perspective, predictions = self.mirror(insights)
                await self.perspective_queue.put((perspective, predictions))
                
            if not self.perspective_queue.empty():
                perspective, predictions = await self.perspective_queue.get()
                for node in self.nodes:
                    node.insights.append(insights)
                    node.perspective.append(perspective)
                    
            await asyncio.sleep(0.01)  # Prevent CPU blocking
                    
    def merge_nodes_to_supernode(self, nodes: List[Node]) -> SuperNode:
        # Combine insights and perspective from nodes
        combined_insights = []
        combined_perspective = []
        
        for node in nodes:
            if node.insights:
                combined_insights.append(torch.stack(node.insights).mean(0))
            if node.perspective:
                combined_perspective.append(torch.stack(node.perspective).mean(0))
        
        if not combined_insights:
            combined_insights = [torch.zeros((1, self.input_dim))]
        if not combined_perspective:
            combined_perspective = [torch.zeros((1, self.input_dim))]
        
        dna = torch.cat([
            torch.stack(combined_insights).mean(0),
            torch.stack(combined_perspective).mean(0)
        ], dim=-1)
        
        return SuperNode(
            id=len(self.supernodes),
            nodes=nodes,
            dna=dna
        )
        
    def process_data(self, data: torch.Tensor) -> Dict[str, Any]:
        # Calculate node requirements
        num_nodes, memory_threshold = self.calculate_node_requirements(
            data.size(0) * data.size(1) if len(data.shape) > 1 else data.size(0)
        )
        self.initialize_nodes(num_nodes, memory_threshold)
        
        # Process data in nodes
        for i, node in enumerate(self.nodes):
            start_idx = i * (data.size(0) // len(self.nodes))
            end_idx = (i + 1) * (data.size(0) // len(self.nodes)) if i < len(self.nodes) - 1 else data.size(0)
            self.process_data_chunk(node, data[start_idx:end_idx])
        
        # Process with quantum layer
        quantum_features = None
        try:
            if len(data) > 8:
                sample_data = data[:8].mean(dim=0)
            else:
                sample_data = data.mean(dim=0)
                
            # Normalize to appropriate range for quantum processing
            norm_data = torch.nn.functional.normalize(sample_data)
            quantum_features = self.quantum_layer(norm_data)
        except Exception as e:
            logger.error(f"Error in quantum processing: {e}")
            quantum_features = torch.zeros(8)
            
        # Process with hyperdimensional processor
        hdim_projections, spectral = None, None
        try:
            sample_size = min(100, data.size(0))
            hdim_projections, spectral = self.hdprocessor(data[:sample_size])
        except Exception as e:
            logger.error(f"Error in hyperdimensional processing: {e}")
            hdim_projections = torch.zeros((min(100, data.size(0)), 4, self.hdprocessor.hdim))
            spectral = torch.zeros((min(100, data.size(0)), 10))
        
        # Create supernode
        supernode = self.merge_nodes_to_supernode(self.nodes)
        self.supernodes.append(supernode)
        
        # Generate visualization data
        visualization_data = {
            'hypercube': self.environment.to_plotly_figure(),
            'tensor_network': self._generate_tensor_network_viz(supernode.dna),
            'quantum_features': quantum_features.detach().cpu().numpy().tolist(),
            'hdim_projections': hdim_projections.detach().cpu().mean(dim=0).numpy().tolist(),
            'spectral_features': spectral.detach().cpu().mean(dim=0).numpy().tolist()
        }
        
        return {
            'supernode': supernode,
            'visualization_data': visualization_data
        }
        
    def _generate_tensor_network_viz(self, tensor: torch.Tensor) -> dict:
        """Generate tensor network visualization data"""
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (tensors)
        tensor = tensor.reshape(-1)
        nodes = min(20, tensor.size(0))
        
        for i in range(nodes):
            G.add_node(f'Node{i}', type='tensor', value=tensor[i].item())
            
        # Add edges (connections)
        for i in range(nodes):
            for j in range(i + 1, nodes):
                weight = abs(tensor[i].item() * tensor[j].item())
                if weight > 0.01:
                    G.add_edge(f'Node{i}', f'Node{j}', weight=weight)
        
        # Create graph visualization
        pos = nx.spring_layout(G, seed=42)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            text=[node for node in G.nodes()],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[G.nodes[node]['value'] for node in G.nodes()],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Value',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace],
                      layout=go.Layout(
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))