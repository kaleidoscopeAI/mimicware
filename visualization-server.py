import flask
from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import plotly
import plotly.graph_objs as go
import json
import os
import asyncio
import logging
from datetime import datetime
import uuid
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage for visualization data
visualization_data = {}

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Handle data upload for visualization"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Generate unique ID for this dataset
        dataset_id = str(uuid.uuid4())
        
        # Save file temporarily
        temp_path = f"temp_{dataset_id}.pt"
        file.save(temp_path)
        
        # Load data
        data = torch.load(temp_path)
        os.remove(temp_path)  # Clean up
        
        # Process data asynchronously
        asyncio.create_task(process_data(dataset_id, data))
        
        return jsonify({
            'dataset_id': dataset_id,
            'message': 'Data uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

async def process_data(dataset_id, data):
    """Process data for visualization"""
    try:
        # Initialize system
        manager = SystemManager({
            'world_size': 4,
            'hdim': data.size(-1)
        })
        
        # Process data
        results = await manager.run_workflow(data)
        
        # Generate visualizations
        visualizations = generate_visualizations(results)
        
        # Store results
        visualization_data[dataset_id] = {
            'timestamp': datetime.now().isoformat(),
            'visualizations': visualizations,
            'raw_results': results
        }
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        visualization_data[dataset_id] = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
        
def generate_visualizations(results):
    """Generate visualization data from processing results"""
    visualizations = {}
    
    # 1. Hypercube visualization
    visualizations['hypercube'] = generate_hypercube_viz(
        results['quantum_features']
    )
    
    # 2. Topology visualization
    visualizations['topology'] = generate_topology_viz(
        results['topology_features']
    )
    
    # 3. Optimization curve
    visualizations['optimization'] = {
        'type': 'line',
        'data': {
            'x': list(range(len(results['optimization_curve']))),
            'y': results['optimization_curve']
        },
        'layout': {
            'title': 'Optimization Progress',
            'xaxis': {'title': 'Iteration'},
            'yaxis': {'title': 'Loss'}
        }
    }
    
    # 4. Tensor network visualization
    visualizations['tensor_network'] = generate_tensor_network_viz(
        results['processed_tensor']
    )
    
    return visualizations
    
def generate_hypercube_viz(quantum_features):
    """Generate hypercube visualization"""
    # Create 3D projection of 4D hypercube
    features = quantum_features.cpu().numpy()
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=features_3d[:, 0],
            y=features_3d[:, 1],
            z=features_3d[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=np.linalg.norm(features, axis=1),
                colorscale='Viridis',
                opacity=0.8
            )
        )
    ])
    
    # Add edges between nearby points
    edges = compute_hypercube_edges(features_3d)
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[features_3d[edge[0], 0], features_3d[edge[1], 0]],
            y=[features_3d[edge[0], 1], features_3d[edge[1], 1]],
            z=[features_3d[edge[0], 2], features_3d[edge[1], 2]],
            mode='lines',
            line=dict(color='rgba(100, 100, 100, 0.4)', width=1)
        ))
    
    fig.update_layout(
        title='4D Hypercube Projection',
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    return json.loads(fig.to_json())
    
def compute_hypercube_edges(points, threshold=0.1):
    """Compute edges for hypercube visualization"""
    n_points = len(points)
    edges = []
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < threshold:
                edges.append((i, j))
                
    return edges
    
def generate_topology_viz(topology_features):
    """Generate topology visualization"""
    # Create persistence diagram
    fig = plt.figure(figsize=(10, 8))
    
    # Create 3D axes for persistence diagram
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot persistence points
    for dim, color in enumerate(['blue', 'red', 'green', 'purple']):
        if dim < len(topology_features['cohomology']):
            points = topology_features['cohomology'][dim]
            if len(points) > 0:
                birth = points[:, 0]
                death = points[:, 1]
                ax.scatter(birth, death, [dim] * len(birth), c=color, label=f'Dimension {dim}')
    
    # Add diagonal
    lims = ax.get_xlim()
    ax.plot([lims[0], lims[1]], [lims[0], lims[1]], [0, 0], 'k--')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_zlabel('Dimension')
    ax.set_title('Persistence Diagram')
    ax.legend()
    
    # Convert to base64 image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return {
        'type': 'image',
        'data': f'data:image/png;base64,{img_str}'
    }
    
def generate_tensor_network_viz(tensor):
    """Generate tensor network visualization"""
    # Compute tensor decomposition (SVD)
    U, S, V = torch.svd(tensor)
    
    # Create network graph
    G = nx.Graph()
    
    # Add nodes (tensors)
    for i in range(min(20, tensor.size(0))):
        G.add_node(f'U{i}', type='U', value=S[i].item())
        
    for i in range(min(20, tensor.size(1))):
        G.add_node(f'V{i}', type='V', value=1.0)
        
    # Add edges (connections)
    for i in range(min(20, tensor.size(0))):
        for j in range(min(20, tensor.size(1))):
            weight = float(U[i, j].item() * S[j].item() * V[j, i].item())
            if abs(weight) > 0.01:
                G.add_edge(f'U{i}', f'V{j}', weight=weight)
    
    # Create graph visualization
    pos = nx.spring_layout(G, seed=42)
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
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
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['color'] += (G.nodes[node]['value'],)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title='Tensor Network',
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                  ))
    
    return json.loads(fig.to_json())

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of available datasets"""
    datasets = [
        {
            'id': dataset_id,
            'timestamp': data['timestamp'],
            'error': data.get('error', None)
        }
        for dataset_id, data in visualization_data.items()
    ]
    return jsonify(datasets)

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset_visualizations(dataset_id):
    """Get visualizations for a specific dataset"""
    if dataset_id not in visualization_data:
        return jsonify({'error': 'Dataset not found'}), 404
        
    data = visualization_data[dataset_id]
    
    if 'error' in data:
        return jsonify({'error': data['error']}), 500
        
    return jsonify(data['visualizations'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
