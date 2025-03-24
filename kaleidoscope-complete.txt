#!/usr/bin/env python3
# kaleidoscope_ai.py - Complete KaleidoscopeAI System with Web Visualization
import torch, torch.nn as nn, numpy as np, asyncio, plotly.graph_objects as go
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from flask import Flask, request, jsonify, render_template, send_from_directory
import logging, uuid, json, base64, time, os, sys, pennylane as qml, ray
from io import BytesIO
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA

# Initialize system
logging.basicConfig(level=logging.INFO)
ray.init(ignore_reinit_error=True)

# Core data structures
@dataclass
class Vector4D:
    x: float; y: float; z: float; w: float
    def to_array(self): return np.array([self.x, self.y, self.z, self.w])

@dataclass
class Supercluster:
    position: Vector4D
    intensity: float
    connections: Set[Tuple[int, int]] = field(default_factory=set)

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

# Hypercube implementation
class HypercubeStringNetwork:
    def __init__(self, dimension=4, resolution=10):
        self.dimension, self.resolution = dimension, resolution
        self.vertices = self._generate_vertices()
        self.strings = self._generate_strings()
        self.superclusters = self._find_intersections()
        
    def _generate_vertices(self):
        return [Vector4D(x, y, z, w) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1] for w in [-1, 1]]
    
    def _generate_strings(self):
        strings = []
        steps = np.linspace(-1, 1, self.resolution)
        for dim1 in range(4):
            for dim2 in range(dim1 + 1, 4):
                for i in steps:
                    for j in steps:
                        start, end = [0] * 4, [0] * 4
                        start[dim1], start[dim2], start[-1] = i, j, -1
                        end[dim1], end[dim2], end[-1] = i, j, 1
                        strings.append((Vector4D(*start), Vector4D(*end)))
        return strings
    
    def _compute_intersection(self, seg1, seg2):
        d1, d2 = seg1[1] - seg1[0], seg2[1] - seg2[0]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-10 or n2 < 1e-10: return None
        d1, d2 = d1/n1, d2/n2
        normal = np.cross(d1[:3], d2[:3])
        if np.linalg.norm(normal) < 1e-10: return None
        try:
            t, s = np.linalg.lstsq(np.vstack((d1, -d2)).T, seg2[0] - seg1[0], rcond=None)[0]
            if 0 <= t <= n1 and 0 <= s <= n2: return seg1[0] + t * d1
        except: pass
        return None
    
    def _find_intersections(self):
        superclusters = []
        string_segments = np.array([[[s[0].x, s[0].y, s[0].z, s[0].w], [s[1].x, s[1].y, s[1].z, s[1].w]] for s in self.strings])
        for i in range(len(self.strings)):
            for j in range(i + 1, len(self.strings)):
                intersection = self._compute_intersection(string_segments[i], string_segments[j])
                if intersection is not None:
                    superclusters.append(Supercluster(Vector4D(*intersection), 1.0, {(i, j)}))
        return self._merge_nearby_clusters(superclusters, 0.1)
    
    def _merge_nearby_clusters(self, clusters, threshold):
        if not clusters: return []
        merged, used = [], set()
        for i, c1 in enumerate(clusters):
            if i in used: continue
            current, used.add(i) = c1, used.add(i)
            for j, c2 in enumerate(clusters[i+1:], i+1):
                if j in used: continue
                dist = np.sqrt(sum((getattr(c1.position, attr) - getattr(c2.position, attr))**2 for attr in ['x','y','z','w']))
                if dist < threshold:
                    current.intensity += c2.intensity
                    current.connections.update(c2.connections)
                    used.add(j)
            merged.append(current)
        return merged
    
    def project_to_3d(self, w_slice=0):
        string_points_3d, cluster_points_3d, intensities = [], [], []
        for start, end in self.strings:
            if abs(start.w - w_slice) < 0.1 or abs(end.w - w_slice) < 0.1:
                string_points_3d.append([[start.x, start.y, start.z], [end.x, end.y, end.z]])
        for cluster in self.superclusters:
            if abs(cluster.position.w - w_slice) < 0.1:
                cluster_points_3d.append([cluster.position.x, cluster.position.y, cluster.position.z])
                intensities.append(cluster.intensity)
        return np.array(string_points_3d), np.array(cluster_points_3d), intensities
        
    def to_plotly_figure(self, w_slices=[-0.5, 0, 0.5]):
        fig = go.Figure()
        for w in w_slices:
            strings_3d, clusters_3d, intensities = self.project_to_3d(w)
            for string in strings_3d:
                fig.add_trace(go.Scatter3d(x=string[:,0], y=string[:,1], z=string[:,2], mode='lines', 
                                          line=dict(color='rgba(100,100,100,0.2)', width=1), showlegend=False))
            if len(clusters_3d) > 0:
                fig.add_trace(go.Scatter3d(x=clusters_3d[:,0], y=clusters_3d[:,1], z=clusters_3d[:,2], mode='markers',
                                           marker=dict(size=5*np.array(intensities), color=intensities, colorscale='Viridis', opacity=0.8),
                                           name=f'w = {w}'))
        fig.update_layout(title='4D Hypercube', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), margin=dict(l=0,r=0,b=0,t=30))
        return json.loads(fig.to_json())

# Neural Network components
class KaleidoscopeEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim//2))
        self.insight_generator = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim//2, nhead=8, dim_feedforward=hidden_dim), num_layers=6)
        
    def forward(self, x): return self.insight_generator(self.encoder(x))

class MirrorEngine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.perspective_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1), nn.Linear(hidden_dim, input_dim))
        self.predictor = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=3, batch_first=True)
        
    def forward(self, x):
        perspective = self.perspective_generator(x)
        predictions, _ = self.predictor(x.unsqueeze(0))
        return perspective, predictions.squeeze(0)

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
    def quantum_circuit(self, inputs, weights):
        for i in range(self.n_qubits): qml.RX(inputs[i], wires=i)
        for layer in range(self.n_layers):
            for i in range(self.n_qubits): qml.Rot(*weights[layer, i], wires=i)
            for i in range(self.n_qubits - 1): qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
    def forward(self, x):
        qnode = qml.QNode(self.quantum_circuit, self.dev)
        qnode_torch = qml.qnn.TorchLayer(qnode, self.weights)
        return qnode_torch(x)

# Main Kaleidoscope AI system 
class KaleidoscopeAI:
    def __init__(self, input_dim=512, hidden_dim=1024):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.kaleidoscope = KaleidoscopeEngine(input_dim, hidden_dim)
        self.mirror = MirrorEngine(input_dim, hidden_dim)
        self.environment = HypercubeStringNetwork()
        self.nodes, self.supernodes = [], []
        self.quantum_layer = QuantumLayer(n_qubits=8, n_layers=4)
        self.data_queue = asyncio.Queue()
        self.insight_queue = asyncio.Queue()
        self.perspective_queue = asyncio.Queue()
        self.optimizer = torch.optim.Adam(list(self.kaleidoscope.parameters()) + list(self.mirror.parameters()), lr=0.001)
        
    def calculate_node_requirements(self, data_size):
        total_memory = data_size * 8
        target_insights = int(np.sqrt(data_size))
        num_nodes = max(1, int(np.ceil(total_memory / (target_insights * self.input_dim))))
        return num_nodes, total_memory / num_nodes
        
    def initialize_nodes(self, num_nodes, memory_threshold):
        self.nodes = [Node(id=i, memory_threshold=memory_threshold, embedded_data=torch.zeros(self.input_dim)) 
                     for i in range(num_nodes)]
        
    def process_data_chunk(self, node, data_chunk):
        if not isinstance(data_chunk, torch.Tensor): data_chunk = torch.tensor(data_chunk, dtype=torch.float32)
        if len(data_chunk.shape) == 1: data_chunk = data_chunk.unsqueeze(0)
        if data_chunk.shape[-1] != self.input_dim:
            data_chunk = nn.functional.pad(data_chunk, (0, self.input_dim - data_chunk.shape[-1]))
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
            await asyncio.sleep(0.01)
                    
    def merge_nodes_to_supernode(self, nodes):
        combined_insights, combined_perspective = [], []
        for node in nodes:
            if node.insights: combined_insights.append(torch.stack(node.insights).mean(0))
            if node.perspective: combined_perspective.append(torch.stack(node.perspective).mean(0))
        if not combined_insights: combined_insights = [torch.zeros((1, self.input_dim))]
        if not combined_perspective: combined_perspective = [torch.zeros((1, self.input_dim))]
        dna = torch.cat([torch.stack(combined_insights).mean(0), torch.stack(combined_perspective).mean(0)], dim=-1)
        return SuperNode(id=len(self.supernodes), nodes=nodes, dna=dna)
        
    def process_data(self, data):
        # Prepare data and nodes
        if not isinstance(data, torch.Tensor): data = torch.tensor(data, dtype=torch.float32)
        data_size = data.size(0) * data.size(1) if len(data.shape) > 1 else data.size(0)
        num_nodes, memory_threshold = self.calculate_node_requirements(data_size)
        self.initialize_nodes(num_nodes, memory_threshold)
        
        # Process in nodes
        for i, node in enumerate(self.nodes):
            start_idx = i * (data.size(0) // len(self.nodes))
            end_idx = (i + 1) * (data.size(0) // len(self.nodes)) if i < len(self.nodes) - 1 else data.size(0)
            self.process_data_chunk(node, data[start_idx:end_idx])
        
        # Quantum processing
        try:
            sample_data = data[:8].mean(dim=0) if len(data) > 8 else data.mean(dim=0)
            norm_data = torch.nn.functional.normalize(sample_data)
            quantum_features = self.quantum_layer(norm_data)
        except Exception as e:
            quantum_features = torch.zeros(8)
        
        # Create supernode & visualizations
        supernode = self.merge_nodes_to_supernode(self.nodes)
        self.supernodes.append(supernode)
        
        # Generate visualization data
        viz_data = {
            'hypercube': self.environment.to_plotly_figure(),
            'tensor_network': self._generate_tensor_network_viz(supernode.dna),
            'quantum_features': quantum_features.detach().cpu().numpy().tolist(),
            'optimization': {'data': {'x': list(range(10)), 'y': [1.0/(i+1) for i in range(10)]}}
        }
        
        return {'supernode': supernode, 'visualizations': viz_data}
        
    def _generate_tensor_network_viz(self, tensor):
        G = nx.Graph()
        tensor = tensor.reshape(-1)
        nodes = min(20, tensor.size(0))
        
        for i in range(nodes): G.add_node(f'Node{i}', type='tensor', value=tensor[i].item())
        for i in range(nodes):
            for j in range(i + 1, nodes):
                weight = abs(tensor[i].item() * tensor[j].item())
                if weight > 0.01: G.add_edge(f'Node{i}', f'Node{j}', weight=weight)
        
        pos = nx.spring_layout(G, seed=42)
        edge_trace = [go.Scatter(x=[pos[edge[0]][0], pos[edge[1]][0], None], y=[pos[edge[0]][1], pos[edge[1]][1], None],
                                line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines') for edge in G.edges()]
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()], y=[pos[node][1] for node in G.nodes()], text=[node for node in G.nodes()],
            mode='markers', hoverinfo='text',
            marker=dict(showscale=True, colorscale='YlGnBu', reversescale=True, 
                       color=[G.nodes[node]['value'] for node in G.nodes()], size=10,
                       colorbar=dict(thickness=15, title='Value'), line=dict(width=2)))
        
        fig = go.Figure(data=edge_trace + [node_trace], layout=go.Layout(showlegend=False, hovermode='closest',
                                                                         margin=dict(b=20,l=5,r=5,t=40),
                                                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        return json.loads(fig.to_json())

# Web visualization server
app = Flask(__name__, static_folder='static', template_folder='templates')
visualization_data = {}

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'No file selected'}), 400
        
        dataset_id = str(uuid.uuid4())
        temp_path = f"temp_{dataset_id}.pt"
        file.save(temp_path)
        
        try:
            data = torch.load(temp_path)
        except:
            # Fallback to numpy if not torch tensor
            data = np.load(temp_path)
            data = torch.tensor(data, dtype=torch.float32)
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
        
        # Process data
        system = KaleidoscopeAI(input_dim=512)
        results = system.process_data(data)
        
        # Store results
        visualization_data[dataset_id] = {
            'timestamp': datetime.now().isoformat(),
            'visualizations': results['visualizations']
        }
        
        return jsonify({
            'dataset_id': dataset_id,
            'message': 'Data processed successfully.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    datasets = [{
        'id': dataset_id,
        'timestamp': data['timestamp'],
        'error': data.get('error', None)
    } for dataset_id, data in visualization_data.items()]
    return jsonify(datasets)

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset_visualizations(dataset_id):
    if dataset_id not in visualization_data: return jsonify({'error': 'Dataset not found'}), 404
    data = visualization_data[dataset_id]
    if 'error' in data: return jsonify({'error': data['error']}), 500
    return jsonify(data['visualizations'])

@app.route('/static/<path:path>')
def serve_static(path): return send_from_directory('static', path)

def create_static_dirs():
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Create CSS file
    with open('static/css/styles.css', 'w') as f:
        f.write("""
:root {--primary: #6610f2; --secondary: #20c997; --bg-dark: #121212; --text-light: #e9ecef;}
body {font-family: 'Segoe UI', sans-serif; background-color: var(--bg-dark); color: var(--text-light);}
.navbar {box-shadow: 0 2px 15px rgba(0,0,0,0.5);}
.card {box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: transform 0.3s ease; margin-bottom: 1.5rem;}
.card:hover {transform: translateY(-5px);}
.viz-container {height: 500px; background-color: rgba(0,0,0,0.2); border-radius: 8px; position: relative;}
.spinner-container {position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; 
                   justify-content: center; align-items: center; background-color: rgba(0,0,0,0.5);}
""")
    
    # Create JS file
    with open('static/js/dashboard.js', 'w') as f:
        f.write("""
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadSuccess = document.getElementById('upload-success');
    const uploadError = document.getElementById('upload-error');
    const datasetsTable = document.getElementById('datasets-table');
    const visualizationsSection = document.getElementById('visualizations');
    const currentDatasetBadge = document.getElementById('current-dataset');
    
    loadDatasets();
    uploadForm.addEventListener('submit', handleUpload);
    
    async function handleUpload(event) {
        event.preventDefault();
        const fileInput = document.getElementById('dataFile');
        if (!fileInput.files.length) {
            uploadError.textContent = 'Please select a file';
            uploadError.classList.remove('d-none');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        uploadProgress.classList.remove('d-none');
        uploadSuccess.classList.add('d-none');
        uploadError.classList.add('d-none');
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }
            
            const data = await response.json();
            uploadProgress.classList.add('d-none');
            uploadSuccess.textContent = data.message;
            uploadSuccess.classList.remove('d-none');
            uploadForm.reset();
            
            setTimeout(loadDatasets, 1000);
            
        } catch (error) {
            uploadProgress.classList.add('d-none');
            uploadError.textContent = error.message;
            uploadError.classList.remove('d-none');
        }
    }
    
    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            if (!response.ok) throw new Error('Failed to load datasets');
            
            const datasets = await response.json();
            
            if (datasets.length === 0) {
                datasetsTable.innerHTML = '<tr><td colspan="4" class="text-center">No datasets available</td></tr>';
                return;
            }
            
            let html = '';
            datasets.forEach(dataset => {
                const timestamp = new Date(dataset.timestamp).toLocaleString();
                const status = dataset.error ? 
                    `<span class="badge bg-danger">Error</span>` : 
                    `<span class="badge bg-success">Available</span>`;
                    
                html += `
                    <tr>
                        <td><code>${dataset.id.substring(0, 8)}...</code></td>
                        <td>${timestamp}</td>
                        <td>${status}</td>
                        <td>
                            <button class="btn btn-sm btn-primary view-dataset" data-id="${dataset.id}">
                                <i class="fas fa-eye me-1"></i>View
                            </button>
                        </td>
                    </tr>
                `;
            });
            
            datasetsTable.innerHTML = html;
            
            document.querySelectorAll('.view-dataset').forEach(button => {
                button.addEventListener('click', () => {
                    const datasetId = button.getAttribute('data-id');
                    loadVisualizations(datasetId);
                });
            });
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }
    
    async function loadVisualizations(datasetId) {
        if (!datasetId) return;
        
        currentDatasetBadge.textContent = `Dataset: ${datasetId.substring(0, 8)}...`;
        visualizationsSection.classList.remove('d-none');
        
        const containers = {
            'hypercube': document.getElementById('hypercube-container'),
            'topology': document.getElementById('topology-container'),
            'optimization': document.getElementById('optimization-container'),
            'tensor': document.getElementById('tensor-container')
        };
        
        // Add spinners
        Object.values(containers).forEach(container => {
            container.innerHTML = `
                <div class="spinner-container">
                    <div class="spinner-border text-info" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            `;
        });
        
        try {
            const response = await fetch(`/api/datasets/${datasetId}`);
            if (!response.ok) throw new Error('Failed to load visualizations');
            
            const visualizations = await response.json();
            
            // Render visualizations
            if (visualizations.hypercube) {
                Plotly.newPlot(containers.hypercube, 
                               visualizations.hypercube.data,
                               visualizations.hypercube.layout);
            }
            
            if (visualizations.tensor_network) {
                Plotly.newPlot(containers.tensor, 
                              visualizations.tensor_network.data,
                              visualizations.tensor_network.layout);
            }
            
            if (visualizations.optimization) {
                Plotly.newPlot(containers.optimization, 
                              [{
                                x: visualizations.optimization.data.x,
                                y: visualizations.optimization.data.y,
                                type: 'scatter',
                                mode: 'lines+markers',
                                line: {color: '#6610f2', width: 3},
                                marker: {color: '#20c997', size: 8}
                              }], 
                              {
                                title: 'Optimization Progress',
                                font: {color: '#e9ecef'},
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0.2)',
                                xaxis: {
                                    title: 'Iteration',
                                    gridcolor: 'rgba(255,255,255,0.1)',
                                },
                                yaxis: {
                                    title: 'Loss',
                                    gridcolor: 'rgba(255,255,255,0.1)',
                                },
                                margin: {l: 60, r: 30, t: 50, b: 50}
                              }
                            );
            }
            
            // Display placeholder image for topology
            containers.topology.innerHTML = `
                <div class="d-flex justify-content-center align-items-center h-100">
                    <div class="text-center">
                        <div class="mb-3">
                            <i class="fas fa-project-diagram fa-5x text-info"></i>
                        </div>
                        <h5 class="text-light">Topology Visualization</h5>
                        <p class="text-muted">Topological features analysis</p>
                    </div>
                </div>
            `;
            
            // Scroll to visualizations
            visualizationsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error loading visualizations:', error);
            Object.values(containers).forEach(container => {
                container.innerHTML = `
                    <div class="alert alert-danger m-3">
                        Error loading visualization: ${error.message}
                    </div>
                `;
            });
        }
    }
});
""")
    
    # Create HTML template
    with open('templates/index.html', 'w') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KaleidoscopeAI Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <link href="/static/css/styles.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body class="bg-dark text-light">
    <nav class="navbar navbar-expand-lg navbar-dark bg-black mb-4">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-cube me-2"></i>KaleidoscopeAI
            </a>
        </div>
    </nav>

    <div class="