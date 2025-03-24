// dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadForm = document.getElementById('upload-form');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadSuccess = document.getElementById('upload-success');
    const uploadError = document.getElementById('upload-error');
    const datasetsTable = document.getElementById('datasets-table');
    const visualizationsSection = document.getElementById('visualizations');
    const currentDatasetBadge = document.getElementById('current-dataset');
    const refreshVizButton = document.getElementById('refresh-viz');
    const systemStatusButton = document.getElementById('system-status');
    
    // Visualization containers
    const hypercubeContainer = document.getElementById('hypercube-container');
    const topologyContainer = document.getElementById('topology-container');
    const optimizationContainer = document.getElementById('optimization-container');
    const tensorContainer = document.getElementById('tensor-container');
    
    // Current state
    let currentDatasetId = null;
    let visualizations = {};
    let hypercubeRenderer = null;
    
    // Load datasets on page load
    loadDatasets();
    
    // Setup event listeners
    uploadForm.addEventListener('submit', handleUpload);
    refreshVizButton.addEventListener('click', () => loadVisualizations(currentDatasetId));
    systemStatusButton.addEventListener('click', checkSystemStatus);
    
    // Set up auto-refresh for datasets
    setInterval(loadDatasets, 30000); // Every 30 seconds
    
    /**
     * Handle file upload and processing
     */
    async function handleUpload(event) {
        event.preventDefault();
        
        const fileInput = document.getElementById('dataFile');
        const processingType = document.getElementById('processingType').value;
        
        if (!fileInput.files || fileInput.files.length === 0) {
            showUploadError('Please select a file');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('processingType', processingType);
        
        showUploadProgress(0);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
                // Use XMLHttpRequest for progress tracking
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }
            
            const data = await response.json();
            showUploadSuccess(data.message);
            
            // Auto-load datasets after successful upload
            setTimeout(loadDatasets, 1000);
            
        } catch (error) {
            showUploadError(error.message);
        }
    }
    
    /**
     * Load available datasets
     */
    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            if (!response.ok) throw new Error('Failed to load datasets');
            
            const datasets = await response.json();
            renderDatasetsTable(datasets);
            
        } catch (error) {
            console.error('Error loading datasets:', error);
        }
    }
    
    /**
     * Render datasets table
     */
    function renderDatasetsTable(datasets) {
        if (!datasets || datasets.length === 0) {
            datasetsTable.innerHTML = '<tr><td colspan="4" class="text-center">No datasets available</td></tr>';
            return;
        }
        
        // Sort by timestamp (newest first)
        datasets.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        
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
        
        // Add event listeners to view buttons
        document.querySelectorAll('.view-dataset').forEach(button => {
            button.addEventListener('click', () => {
                const datasetId = button.getAttribute('data-id');
                loadVisualizations(datasetId);
            });
        });
    }
    
    /**
     * Load visualizations for a dataset
     */
    async function loadVisualizations(datasetId) {
        if (!datasetId) return;
        
        currentDatasetId = datasetId;
        currentDatasetBadge.textContent = `Dataset: ${datasetId.substring(0, 8)}...`;
        
        visualizationsSection.classList.remove('d-none');
        showLoadingSpinners();
        
        try {
            const response = await fetch(`/api/datasets/${datasetId}`);
            if (!response.ok) throw new Error('Failed to load visualizations');
            
            visualizations = await response.json();
            renderVisualizations();
            
            // Scroll to visualizations
            visualizationsSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (error) {
            console.error('Error loading visualizations:', error);
        }
    }
    
    /**
     * Render all visualizations
     */
    function renderVisualizations() {
        if (!visualizations) return;
        
        renderHypercubeVisualization();
        renderTopologyVisualization();
        renderOptimizationVisualization();
        renderTensorVisualization();
    }
    
    /**
     * Render 4D hypercube visualization using Three.js
     */
    function renderHypercubeVisualization() {
        const vizData = visualizations.hypercube;
        if (!vizData || !vizData.data || vizData.data.length === 0) {
            hypercubeContainer.innerHTML = '<div class="alert alert-warning m-3">No hypercube data available</div>';
            return;
        }
        
        // Clear previous visualization
        while (hypercubeContainer.firstChild) {
            hypercubeContainer.removeChild(hypercubeContainer.firstChild);
        }
        
        if (hypercubeRenderer) {
            hypercubeRenderer.dispose();
        }
        
        // Create new three.js scene
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, hypercubeContainer.clientWidth / hypercubeContainer.clientHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        
        renderer.setSize(hypercubeContainer.clientWidth, hypercubeContainer.clientHeight);
        hypercubeContainer.appendChild(renderer.domElement);
        hypercubeRenderer = renderer;
        
        // Extract data
        const data = vizData.data[0];
        const positions = [];
        const colors = [];
        
        // Create points (handle missing data case)
        if (data && data.x && data.y && data.z) {
            const pointGeometry = new THREE.BufferGeometry();
            const pointMaterial = new THREE.PointsMaterial({
                size: 0.05,
                vertexColors: true,
                transparent: true,
                opacity: 0.8
            });
            
            // Add points to scene
            for (let i = 0; i < data.x.length; i++) {
                if (data.x[i] !== null && data.y[i] !== null && data.z[i] !== null) {
                    positions.push(data.x[i], data.y[i], data.z[i]);
                    
                    // Color based on value (handle missing color data)
                    const colorScale = data.marker && data.marker.color ? 
                        data.marker.color[i] / Math.max(...data.marker.color) : 0.5;
                    colors.push(0.5 - colorScale * 0.5, 0.2 + colorScale * 0.8, 0.8);
                }
            }
            
            pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            pointGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            
            const points = new THREE.Points(pointGeometry, pointMaterial);
            scene.add(points);
            
            // Add edges
            if (vizData.data.length > 1) {
                // Process line data
                for (let j = 1; j < vizData.data.length; j++) {
                    const lineData = vizData.data[j];
                    if (lineData && lineData.x && lineData.y && lineData.z) {
                        const linePositions = [];
                        
                        for (let i = 0; i < lineData.x.length; i++) {
                            if (lineData.x[i] !== null && lineData.y[i] !== null && lineData.z[i] !== null) {
                                linePositions.push(lineData.x[i], lineData.y[i], lineData.z[i]);
                            }
                        }
                        
                        if (linePositions.length > 0) {
                            const lineGeometry = new THREE.BufferGeometry();
                            lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
                            
                            const lineMaterial = new THREE.LineBasicMaterial({
                                color: 0x555555,
                                transparent: true,
                                opacity: 0.3
                            });
                            
                            const line = new THREE.LineSegments(lineGeometry, lineMaterial);
                            scene.add(line);
                        }
                    }
                }
            }
            
            // Set camera position
            camera.position.z = 2;
            
            // Add ambient light
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);
            
            // Add directional light
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(0, 1, 1);
            scene.add(directionalLight);
            
            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                
                // Rotate scene
                scene.children.forEach(child => {
                    if (child instanceof THREE.Points || child instanceof THREE.LineSegments) {
                        child.rotation.x += 0.003;
                        child.rotation.y += 0.002;
                    }
                });
                
                renderer.render(scene, camera);
            }
            
            animate();
            
            // Handle window resize
            window.addEventListener('resize', () => {
                camera.aspect = hypercubeContainer.clientWidth / hypercubeContainer.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(hypercubeContainer.clientWidth, hypercubeContainer.clientHeight);
            });
        } else {
            hypercubeContainer.innerHTML = '<div class="alert alert-warning m-3">Invalid hypercube data format</div>';
        }
    }
    
    /**
     * Render topology visualization
     */
    function renderTopologyVisualization() {
        const vizData = visualizations.topology;
        if (!vizData) return;
        
        // Clear previous visualization
        while (topologyContainer.firstChild) {
            topologyContainer.removeChild(topologyContainer.firstChild);
        }
        
        if (vizData.type === 'image') {
            // Create image element
            const img = document.createElement('img');
            img.src = vizData.data;
            img.style.maxWidth = '100%';
            img.style.maxHeight = '100%';
            img.style.display = 'block';
            img.style.margin = '0 auto';
            
            topologyContainer.appendChild(img);
        }
    }
    
    /**
     * Render optimization visualization
     */
    function renderOptimizationVisualization() {
        const vizData = visualizations.optimization;
        if (!vizData) return;
        
        // Clear previous visualization
        while (optimizationContainer.firstChild) {
            optimizationContainer.removeChild(optimizationContainer.firstChild);
        }
        
        Plotly.newPlot(optimizationContainer, 
            [{
                x: vizData.data.x,
                y: vizData.data.y,
                type: 'scatter',
                mode: 'lines+markers',
                line: {
                    color: '#6610f2',
                    width: 3
                },
                marker: {
                    color: '#20c997',
                    size: 8
                }
            }], 
            {
                title: vizData.layout.title,
                font: {
                    color: '#e9ecef'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0.2)',
                xaxis: {
                    title: vizData.layout.xaxis.title,
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.3)'
                },
                yaxis: {
                    title: vizData.layout.yaxis.title,
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.3)'
                },
                margin: {
                    l: 60,
                    r: 30,
                    t: 50,
                    b: 50
                }
            }
        );
    }
    
    /**
     * Render tensor network visualization
     */
    function renderTensorVisualization() {
        const vizData = visualizations.tensor_network;
        if (!vizData) return;
        
        // Clear previous visualization
        while (tensorContainer.firstChild) {
            tensorContainer.removeChild(tensorContainer.firstChild);
        }
        
        Plotly.newPlot(tensorContainer, vizData.data, {
            ...vizData.layout,
            font: {
                color: '#e9ecef'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.2)',
            margin: {
                l: 20,
                r: 20,
                t: 50,
                b: 20
            }
        });
    }
    
    /**
     * Show loading spinners for visualizations
     */
    function showLoadingSpinners() {
        [hypercubeContainer, topologyContainer, optimizationContainer, tensorContainer].forEach(container => {
            const spinner = document.createElement('div');
            spinner.classList.add('spinner-container');
            spinner.innerHTML = `
                <div class="spinner-border text-info" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            `;
            
            // Clear container and add spinner
            while (container.firstChild) {
                container.removeChild(container.firstChild);
            }
            container.appendChild(spinner);
        });
    }
    
    /**
     * Show upload progress
     */
    function showUploadProgress(progress) {
        uploadProgress.classList.remove('d-none');
        uploadSuccess.classList.add('d-none');
        uploadError.classList.add('d-none');
        
        const progressBar = uploadProgress.querySelector('.progress-bar');
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }
    
    /**
     * Show upload success message
     */
    function showUploadSuccess(message) {
        uploadProgress.classList.add('d-none');
        uploadSuccess.classList.remove('d-none');
        uploadError.classList.add('d-none');
        uploadSuccess.textContent = message;
        
        // Reset form
        uploadForm.reset();
    }
    
    /**
     * Show upload error message
     */
    function showUploadError(message) {
        uploadProgress.classList.add('d-none');
        uploadSuccess.classList.add('d-none');
        uploadError.classList.remove('d-none');
        uploadError.textContent = message;
    }
    
    /**
     * Check system status
     */
    async function checkSystemStatus() {
        systemStatusButton.disabled = true;
        
        try {
            const response = await fetch('/api/system/status');
            if (!response.ok) throw new Error('Failed to check system status');
            
            const statusData = await response.json();
            
            const statusString = `
                System Status: ${statusData.status}
                Workers: ${statusData.workers_active}/${statusData.workers_total}
                Memory: ${statusData.memory_usage}%
                Processing: ${statusData.current_processing}
            `;
            
            alert(statusString);
            
        } catch (error) {
            console.error('Error checking system status:', error);
            alert('Error checking system status: ' + error.message);
        } finally {
            systemStatusButton.disabled = false;
        }
    }
});
