#!/bin/bash

# Script to fix and start the System Builder application from ntegratnytem

# Set working directory
WORK_DIR="/home/jg/Music/claud/ntegratnytem"
cd "$WORK_DIR" || { echo "Error: Cannot access $WORK_DIR"; exit 1; }

# Define paths
VENV_DIR="$WORK_DIR/venv"
PYTHON="$VENV_DIR/bin/python"
SYSTEM_BUILDER_PY="$WORK_DIR/system-builder.py"
SYSTEMBUILDER="$WORK_DIR/systembuilder"

# Step 1: Ensure virtual environment exists and is set up
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || { echo "Error: Failed to create virtual environment"; exit 1; }
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate" || { echo "Error: Failed to activate virtual environment"; exit 1; }

# Install required packages (including tkinterdnd2)
echo "Installing required packages..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install networkx numpy Pillow matplotlib tkinterdnd2 || { echo "Error: Failed to install packages"; exit 1; }

# Step 2: Fix system-builder.py with tkinterdnd2 integration
echo "Updating $SYSTEM_BUILDER_PY with tkinterdnd2 fixes..."
cat > "$SYSTEM_BUILDER_PY" << 'EOF'
#!/usr/bin/env python3
"""
SystemBuilder: Advanced Drag-and-Drop System Integration Platform
Core implementation with GUI, dependency resolution, and execution engine
"""

import os
import sys
import subprocess
import tempfile
import hashlib
import json
import importlib
import inspect
import re
import shutil
import logging
import threading
import queue
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Union, Callable, Any
from enum import Enum, auto
from dataclasses import dataclass, field
from tkinterdnd2 import *  # Use tkinterdnd2 for drag-and-drop support
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import ctypes
from ctypes import c_void_p, c_int, c_char_p, POINTER, Structure, CDLL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("SystemBuilder")

# C optimizer code (unchanged)
C_OPTIMIZER_CODE = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int node_id;
    int weight;
} NodeWeight;

typedef struct {
    int num_nodes;
    int* adjacency_matrix;
    NodeWeight* node_weights;
} Graph;

Graph* create_graph(int num_nodes) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    graph->adjacency_matrix = (int*)calloc(num_nodes * num_nodes, sizeof(int));
    graph->node_weights = (NodeWeight*)malloc(num_nodes * sizeof(NodeWeight));
    for (int i = 0; i < num_nodes; i++) {
        graph->node_weights[i].node_id = i;
        graph->node_weights[i].weight = 0;
    }
    return graph;
}

void add_edge(Graph* graph, int source, int target, int weight) {
    if (source >= 0 && source < graph->num_nodes && 
        target >= 0 && target < graph->num_nodes) {
        graph->adjacency_matrix[source * graph->num_nodes + target] = weight;
    }
}

void set_node_weight(Graph* graph, int node, int weight) {
    if (node >= 0 && node < graph->num_nodes) {
        graph->node_weights[node].weight = weight;
    }
}

int* topological_sort(Graph* graph, int* result_size) {
    int num_nodes = graph->num_nodes;
    int* in_degree = (int*)calloc(num_nodes, sizeof(int));
    int* result = (int*)malloc(num_nodes * sizeof(int));
    *result_size = 0;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (graph->adjacency_matrix[i * num_nodes + j] > 0) {
                in_degree[j]++;
            }
        }
    }
    int queue_size = 0;
    int queue_capacity = num_nodes;
    int* queue = (int*)malloc(queue_capacity * sizeof(int));
    int front = 0, rear = 0;
    for (int i = 0; i < num_nodes; i++) {
        if (in_degree[i] == 0) {
            queue[rear++] = i;
            queue_size++;
        }
    }
    while (front != rear) {
        int current = queue[front++];
        front %= queue_capacity;
        queue_size--;
        result[(*result_size)++] = current;
        for (int i = 0; i < num_nodes; i++) {
            if (graph->adjacency_matrix[current * num_nodes + i] > 0) {
                in_degree[i]--;
                if (in_degree[i] == 0) {
                    queue[rear++] = i;
                    rear %= queue_capacity;
                    queue_size++;
                }
            }
        }
    }
    if (*result_size != num_nodes) {
        free(result);
        result = NULL;
        *result_size = 0;
    }
    free(in_degree);
    free(queue);
    return result;
}

bool resolve_dependencies(Graph* graph) {
    int result_size;
    int* sorted = topological_sort(graph, &result_size);
    bool success = (sorted != NULL);
    if (sorted) {
        free(sorted);
    }
    return success;
}

void free_graph(Graph* graph) {
    free(graph->adjacency_matrix);
    free(graph->node_weights);
    free(graph);
}
"""

def compile_c_optimizer():
    """Compile the C optimizer code into a shared library"""
    with tempfile.NamedTemporaryFile(suffix='.c', delete=False) as f:
        f.write(C_OPTIMIZER_CODE.encode('utf-8'))
        c_file = f.name
    
    output_file = os.path.join(tempfile.gettempdir(), f"optimizer{os.getpid()}.so")
    
    try:
        if sys.platform.startswith('win'):
            subprocess.check_call(['gcc', c_file, '-shared', '-o', output_file])
        else:
            subprocess.check_call(['gcc', c_file, '-shared', '-fPIC', '-o', output_file])
        return output_file
    except Exception as e:
        logger.error(f"Failed to compile C optimizer: {e}")
        return None
    finally:
        if os.path.exists(c_file):
            os.unlink(c_file)

# Load the compiled C library
try:
    c_lib_path = compile_c_optimizer()
    if c_lib_path and os.path.exists(c_lib_path):
        c_optimizer = CDLL(c_lib_path)
        c_optimizer.create_graph.argtypes = [c_int]
        c_optimizer.create_graph.restype = c_void_p
        c_optimizer.add_edge.argtypes = [c_void_p, c_int, c_int, c_int]
        c_optimizer.add_edge.restype = None
        c_optimizer.set_node_weight.argtypes = [c_void_p, c_int, c_int]
        c_optimizer.set_node_weight.restype = None
        c_optimizer.resolve_dependencies.argtypes = [c_void_p]
        c_optimizer.resolve_dependencies.restype = c_int
        c_optimizer.free_graph.argtypes = [c_void_p]
        c_optimizer.free_graph.restype = None
        logger.info("C optimizer successfully compiled and loaded")
    else:
        logger.warning("C optimizer compilation failed, falling back to Python implementation")
        c_optimizer = None
except Exception as e:
    logger.warning(f"Failed to load C optimizer: {e}. Using Python fallback.")
    c_optimizer = None

# ... (ScriptType, DependencyType, ScriptNode, ScriptAnalyzer classes unchanged)

class ScriptAnalyzer:
    """Analyzes scripts to determine their dependencies and requirements"""
    
    EXTENSION_MAP = {
        '.py': ScriptType.PYTHON,
        '.sh': ScriptType.SHELL,
        '.bash': ScriptType.SHELL,
        '.c': ScriptType.C,
        '.cpp': ScriptType.CPP,
        '.js': ScriptType.JAVASCRIPT,
        '.asm': ScriptType.ASSEMBLY,
        '.s': ScriptType.ASSEMBLY,
    }
    
    PYTHON_IMPORT_PATTERN = re.compile(r'(?:import|from)\s+([a-zA-Z0-9_.]+)')
    CPP_INCLUDE_PATTERN = re.compile(r'#include\s+[<"]([^>"]+)[>"]')
    PYTHON_MAGIC_COMMENT_PATTERN = re.compile(r'#\s*depends-on:\s*(.+)$', re.MULTILINE)
    
    def __init__(self):
        self.scripts: Dict[str, ScriptNode] = {}
        self.dependency_graph = nx.DiGraph()
    
    def detect_script_type(self, filepath: str) -> ScriptType:
        """Detect script type based on file extension and content"""
        _, ext = os.path.splitext(filepath)
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]
        
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read(1024)  # Read first 1KB
            
            if content.startswith('#!/usr/bin/env python') or '#!/usr/bin/python' in content:
                return ScriptType.PYTHON
            elif content.startswith('#!/bin/bash') or '#!/bin/sh' in content:
                return ScriptType.SHELL
            elif '#include <stdio.h>' in content:
                return ScriptType.C
            elif re.search(r'#include\s+<iostream>', content):
                return ScriptType.CPP
                
        return ScriptType.UNKNOWN
    
    def analyze_script(self, filepath: str) -> ScriptNode:
        """Analyze a script to determine its dependencies"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Script not found: {filepath}")
        
        script_id = hashlib.md5(filepath.encode()).hexdigest()
        
        if script_id in self.scripts:
            return self.scripts[script_id]
        
        with open(filepath, 'r', errors='ignore') as f:
            content = f.read()
        
        script_type = self.detect_script_type(filepath)
        node = ScriptNode(
            id=script_id,
            path=filepath,
            script_type=script_type,
            content=content
        )
        
        if script_type == ScriptType.PYTHON:
            self._analyze_python_script(node)
        elif script_type == ScriptType.C or script_type == ScriptType.CPP:
            self._analyze_c_cpp_script(node)
        elif script_type == ScriptType.SHELL:
            self._analyze_shell_script(node)
        
        self._check_magic_comments(node)
        
        self.scripts[script_id] = node
        return node
    
    def _analyze_python_script(self, node: ScriptNode):
        """Analyze Python script for imports and other dependencies"""
        imports = self.PYTHON_IMPORT_PATTERN.findall(node.content)
        for imp in imports:
            if not (imp.startswith('_') or '.' in imp):
                continue
            node.metadata.setdefault('imports', []).append(imp)
        
        if re.search(r'open\([\'"]([^\'"]+)[\'"]', node.content):
            node.metadata['uses_files'] = True
    
    def _analyze_c_cpp_script(self, node: ScriptNode):
        """Analyze C/C++ script for includes and dependencies"""
        includes = self.CPP_INCLUDE_PATTERN.findall(node.content)
        node.metadata['includes'] = includes
        
        if re.search(r'(fopen|system|exec)', node.content):
            node.metadata['uses_system_calls'] = True
    
    def _analyze_shell_script(self, node: ScriptNode):
        """Analyze shell script for dependencies"""
        script_calls = re.findall(r'(?:bash|sh|source|\.)\s+([^\s;|&<>]+)', node.content)
        node.metadata['script_calls'] = script_calls
    
    def _check_magic_comments(self, node: ScriptNode):
        """Check for magic comments that specify dependencies"""
        for line in node.content.split('\n'):
            match = self.PYTHON_MAGIC_COMMENT_PATTERN.search(line)
            if match:
                deps = [dep.strip() for dep in match.group(1).split(',')]
                node.metadata.setdefault('declared_dependencies', []).extend(deps)
    
    def build_dependency_graph(self, scripts: List[str]) -> nx.DiGraph:
        """Build a dependency graph from a list of scripts"""
        self.dependency_graph = nx.DiGraph()
        
        for script_path in scripts:
            node = self.analyze_script(script_path)
            self.dependency_graph.add_node(node.id, node=node)
        
        for node_id, node_data in self.dependency_graph.nodes(data=True):
            node = node_data['node']
            if 'declared_dependencies' in node.metadata:
                for dep_path in node.metadata['declared_dependencies']:
                    if not os.path.isabs(dep_path):
                        base_dir = os.path.dirname(node.path)
                        dep_path = os.path.abspath(os.path.join(base_dir, dep_path))
                    
                    for other_id, other_data in self.dependency_graph.nodes(data=True):
                        other = other_data['node']
                        if os.path.samefile(other.path, dep_path):
                            self.dependency_graph.add_edge(other.id, node.id, 
                                                         type=DependencyType.EXECUTION)
                            node.dependencies[other.id] = DependencyType.EXECUTION
                            other.reverse_dependencies.add(node.id)
                            break
        
        return self.dependency_graph
    
    def resolve_execution_order(self) -> List[ScriptNode]:
        """Resolve the execution order of scripts"""
        if not self.dependency_graph:
            return []
        
        try:
            if c_optimizer:
                return self._resolve_execution_order_c()
            else:
                return self._resolve_execution_order_python()
        except Exception as e:
            logger.error(f"Error resolving execution order: {e}")
            return []
    
    def _resolve_execution_order_c(self) -> List[ScriptNode]:
        """Use C implementation to resolve execution order"""
        num_nodes = len(self.dependency_graph)
        node_ids = list(self.dependency_graph.nodes())
        id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        graph_ptr = c_optimizer.create_graph(c_int(num_nodes))
        
        for source, target, data in self.dependency_graph.edges(data=True):
            weight = 1
            if 'weight' in data:
                weight = data['weight']
            
            c_optimizer.add_edge(
                graph_ptr, 
                c_int(id_to_idx[source]), 
                c_int(id_to_idx[target]), 
                c_int(weight)
            )
        
        for node_id in self.dependency_graph.nodes():
            node = self.dependency_graph.nodes[node_id]['node']
            weight = os.path.getsize(node.path) // 1024  # KB
            c_optimizer.set_node_weight(graph_ptr, c_int(id_to_idx[node_id]), c_int(weight))
        
        success = c_optimizer.resolve_dependencies(graph_ptr)
        
        c_optimizer.free_graph(graph_ptr)
        
        if not success:
            raise ValueError("Circular dependency detected in scripts")
        
        try:
            order = list(nx.topological_sort(self.dependency_graph))
            return [self.dependency_graph.nodes[node_id]['node'] for node_id in order]
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependency detected in scripts")
    
    def _resolve_execution_order_python(self) -> List[ScriptNode]:
        """Use Python implementation to resolve execution order"""
        try:
            order = list(nx.topological_sort(self.dependency_graph))
            return [self.dependency_graph.nodes[node_id]['node'] for node_id in order]
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependency detected in scripts")

# ... (RuntimeEnvironment, ExecutionResult, ExecutionEngine classes unchanged)

class RuntimeEnvironment:
    """Manages the runtime environment for script execution"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="sysbuilder_")
        self.env_vars = os.environ.copy()
        self.installed_packages = set()
        
        os.makedirs(os.path.join(self.base_dir, 'bin'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'lib'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'include'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'scripts'), exist_ok=True)
        
        self.env_vars['PATH'] = os.path.join(self.base_dir, 'bin') + os.pathsep + self.env_vars.get('PATH', '')
        self.env_vars['SYSBUILDER_ENV'] = self.base_dir
    
    def install_requirements(self, scripts: List[ScriptNode]):
        """Install required packages for scripts"""
        requirements = set()
        
        for script in scripts:
            if script.script_type == ScriptType.PYTHON:
                if 'imports' in script.metadata:
                    for imp in script.metadata['imports']:
                        if imp and not imp.startswith('_') and '.' not in imp:
                            requirements.add(imp)
        
        if requirements:
            logger.info(f"Installing Python packages: {', '.join(requirements)}")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    '--target', os.path.join(self.base_dir, 'lib', 'python'),
                    *requirements
                ])
                self.installed_packages.update(requirements)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install Python packages: {e}")
        
        python_lib_path = os.path.join(self.base_dir, 'lib', 'python')
        self.env_vars['PYTHONPATH'] = python_lib_path + os.pathsep + self.env_vars.get('PYTHONPATH', '')
    
    def prepare_script(self, script: ScriptNode) -> str:
        """Prepare a script for execution in this environment"""
        target_path = os.path.join(self.base_dir, 'scripts', script.name)
        
        shutil.copy2(script.path, target_path)
        
        if script.script_type in (ScriptType.PYTHON, ScriptType.SHELL):
            os.chmod(target_path, 0o755)
        
        return target_path
    
    def get_environment(self) -> Dict[str, str]:
        """Get the environment variables for this runtime"""
        return self.env_vars.copy()
    
    def cleanup(self):
        """Clean up the environment"""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)

@dataclass
class ExecutionResult:
    """Result of script execution"""
    script: ScriptNode
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration: float

class ExecutionEngine:
    """Executes scripts in the appropriate environment"""
    
    def __init__(self, runtime: RuntimeEnvironment):
        self.runtime = runtime
        self.results: Dict[str, ExecutionResult] = {}
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.workers: List[threading.Thread] = []
        self.max_workers = os.cpu_count() or 4
    
    def execute_script(self, script: ScriptNode) -> ExecutionResult:
        """Execute a single script and return the result"""
        import time
        
        prepared_path = self.runtime.prepare_script(script)
        env = self.runtime.get_environment()
        command = []
        
        if script.script_type == ScriptType.PYTHON:
            command = [sys.executable, prepared_path]
        elif script.script_type == ScriptType.SHELL:
            command = ['bash', prepared_path]
        elif script.script_type == ScriptType.C or script.script_type == ScriptType.CPP:
            output_file = os.path.join(self.runtime.base_dir, 'bin', os.path.splitext(script.name)[0])
            compiler = 'gcc' if script.script_type == ScriptType.C else 'g++'
            
            compile_cmd = [
                compiler, prepared_path, '-o', output_file,
                '-I', os.path.join(self.runtime.base_dir, 'include'),
                '-L', os.path.join(self.runtime.base_dir, 'lib')
            ]
            
            compile_result = subprocess.run(
                compile_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if compile_result.returncode != 0:
                return ExecutionResult(
                    script=script,
                    success=False,
                    return_code=compile_result.returncode,
                    stdout=compile_result.stdout,
                    stderr=f"Compilation failed: {compile_result.stderr}",
                    duration=0
                )
            
            command = [output_file]
        elif script.script_type == ScriptType.ASSEMBLY:
            obj_file = os.path.join(self.runtime.base_dir, 'bin', f"{os.path.splitext(script.name)[0]}.o")
            output_file = os.path.join(self.runtime.base_dir, 'bin', os.path.splitext(script.name)[0])
            
            assemble_cmd = ['as', prepared_path, '-o', obj_file]
            assemble_result = subprocess.run(
                assemble_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if assemble_result.returncode != 0:
                return ExecutionResult(
                    script=script,
                    success=False,
                    return_code=assemble_result.returncode,
                    stdout=assemble_result.stdout,
                    stderr=f"Assembly failed: {assemble_result.stderr}",
                    duration=0
                )
            
            link_cmd = ['ld', obj_file, '-o', output_file]
            link_result = subprocess.run(
                link_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            
            if link_result.returncode != 0:
                return ExecutionResult(
                    script=script,
                    success=False,
                    return_code=link_result.returncode,
                    stdout=link_result.stdout,
                    stderr=f"Linking failed: {link_result.stderr}",
                    duration=0
                )
            
            command = [output_file]
        else:
            return ExecutionResult(
                script=script,
                success=False,
                return_code=-1,
                stdout="",
                stderr=f"Unsupported script type: {script.script_type}",
                duration=0
            )
        
        start_time = time.time()
        
        try:
            process = subprocess.run(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                cwd=os.path.dirname(prepared_path)
            )
            
            duration = time.time() - start_time
            
            return ExecutionResult(
                script=script,
                success=process.returncode == 0,
                return_code=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                script=script,
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                duration=duration
            )
    
    def execute_all(self, scripts: List[ScriptNode], callback: Callable = None) -> Dict[str, ExecutionResult]:
        """Execute all scripts in the correct order"""
        self.results = {}
        
        for script in scripts:
            self.queue.put(script)
        
        self.workers = []
        for _ in range(min(len(scripts), self.max_workers)):
            worker = threading.Thread(target=self._worker, args=(callback,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        self.queue.join()
        
        return self.results
    
    def _worker(self, callback: Callable = None):
        """Worker thread to process scripts from the queue"""
        while True:
            try:
                script = self.queue.get(block=False)
            except queue.Empty:
                break
            
            result = self.execute_script(script)
            
            with self.lock:
                self.results[script.id] = result
                if callback:
                    callback(script, result)
            
            self.queue.task_done()

# GUI Components
class ScriptPanel(ttk.Frame):
    """Panel for displaying and managing scripts"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.scripts = {}
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        self.scripts_frame = ttk.LabelFrame(self, text="Scripts")
        self.scripts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scrollbar = ttk.Scrollbar(self.scripts_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Use TkinterDnD.Listbox for drag-and-drop support
        self.scripts_list = TkinterDnD.Listbox(self.scripts_frame, selectmode=tk.EXTENDED,
                                               yscrollcommand=self.scrollbar.set)
        self.scripts_list.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.scripts_list.yview)
        
        self.scripts_list.bind('<Double-1>', self.on_script_double_click)
        self.scripts_list.bind('<Delete>', self.on_delete_script)
        
        self.scripts_list.drop_target_register('DND_Files')
        self.scripts_list.dnd_bind('<<Drop>>', self.on_drop)
        
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_button = ttk.Button(self.buttons_frame, text="Add Script", 
                                    command=self.on_add_script)
        self.add_button.pack(side=tk.LEFT, padx=5)
        
        self.remove_button = ttk.Button(self.buttons_frame, text="Remove Selected", 
                                       command=self.on_remove_selected)
        self.remove_button.pack(side=tk.LEFT, padx=5)
        
        self.analyze_button = ttk.Button(self.buttons_frame, text="Analyze Scripts", 
                                        command=self.on_analyze_scripts)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
    
    def on_drop(self, event):
        """Handle drag and drop of files"""
        files = event.data.split()
        for file in files:
            file = file.strip()
            if file.startswith('{'):
                file = file[1:]
            if file.endswith('}'):
                file = file[:-1]
            if file.startswith('file://'):
                file = file[7:]
            self.add_script(file)
    
    def add_script(self, filepath):
        """Add a script to the list"""
        if not os.path.exists(filepath):
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
        
        name = os.path.basename(filepath)
        if name in self.scripts:
            messagebox.showinfo("Info", f"Script {name} already added")
            return
        
        self.scripts[name] = filepath
        self.scripts_list.insert(tk.END, name)
        self.controller.on_scripts_changed()
    
    def on_add_script(self):
        """Handle add script button click"""
        filetypes = [
            ("Script files", "*.py;*.sh;*.bash;*.c;*.cpp;*.js;*.asm;*.s"),
            ("Python files", "*.py"),
            ("Shell scripts", "*.sh;*.bash"),
            ("C/C++ files", "*.c;*.cpp;*.h;*.hpp"),
            ("JavaScript files", "*.js"),
            ("Assembly files", "*.asm;*.s"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(title="Select Scripts", filetypes=filetypes)
        for file in files:
            self.add_script(file)
    
    def on_remove_selected(self):
        """Remove selected scripts from the list"""
        selected = self.scripts_list.curselection()
        if not selected:
            return
        
        for index in sorted(selected, reverse=True):
            name = self.scripts_list.get(index)
            del self.scripts[name]
            self.scripts_list.delete(index)
        self.controller.on_scripts_changed()
    
    def on_delete_script(self, event):
        """Handle delete key on script list"""
        self.on_remove_selected()
    
    def on_script_double_click(self, event):
        """Handle double click on a script"""
        selected = self.scripts_list.curselection()
        if not selected:
            return
        
        index = selected[0]
        name = self.scripts_list.get(index)
        filepath = self.scripts[name]
        
        try:
            if sys.platform.startswith('win'):
                os.startfile(filepath)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', filepath])
            else:
                subprocess.run(['xdg-open', filepath])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open script: {e}")
    
    def on_analyze_scripts(self):
        """Trigger script analysis"""
        if not self.scripts:
            messagebox.showinfo("Info", "No scripts to analyze")
            return
        self.controller.analyze_scripts()
    
    def get_script_paths(self):
        """Get list of script file paths"""
        return list(self.scripts.values())

class DependencyPanel(ttk.Frame):
    """Panel for displaying script dependencies"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        self.frame = ttk.LabelFrame(self, text="Dependency Graph")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.controls_frame = ttk.Frame(self)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.run_button = ttk.Button(self.controls_frame, text="Run Scripts", 
                                    command=self.on_run_scripts)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(self.controls_frame, text="Export Configuration", 
                                       command=self.on_export_config)
        self.export_button.pack(side=tk.LEFT, padx=5)
    
    def on_run_scripts(self):
        """Handle run scripts button click"""
        self.controller.run_scripts()
    
    def on_export_config(self):
        """Handle export configuration button click"""
        self.controller.export_configuration()
    
    def update_graph(self, graph: nx.DiGraph):
        """Update the graph visualization"""
        import random
        
        self.canvas.delete("all")
        
        if not graph or not graph.nodes:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="No dependency graph available",
                fill="gray"
            )
            return
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            width = 600
            height = 400
        
        nodes = list(graph.nodes())
        node_positions = {}
        
        for node in nodes:
            node_positions[node] = (
                random.uniform(50, width - 50),
                random.uniform(50, height - 50)
            )
        
        iterations = 50
        k = math.sqrt(width * height / len(nodes))
        
        for _ in range(iterations):
            displacement = {node: [0, 0] for node in nodes}
            
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    dx = node_positions[node1][0] - node_positions[node2][0]
                    dy = node_positions[node1][1] - node_positions[node2][1]
                    
                    distance = max(0.01, math.sqrt(dx*dx + dy*dy))
                    force = k*k / distance
                    
                    displacement[node1][0] += dx / distance * force
                    displacement[node1][1] += dy / distance * force
                    displacement[node2][0] -= dx / distance * force
                    displacement[node2][1] -= dy / distance * force
            
            for edge in graph.edges():
                node1, node2 = edge
                dx = node_positions[node1][0] - node_positions[node2][0]
                dy = node_positions[node1][1] - node_positions[node2][1]
                
                distance = max(0.01, math.sqrt(dx*dx + dy*dy))
                force = distance*distance / k
                
                displacement[node1][0] -= dx / distance * force
                displacement[node1][1] -= dy / distance * force
                displacement[node2][0] += dx / distance * force
                displacement[node2][1] += dy / distance * force
            
            for node in nodes:
                disp = displacement[node]
                dist = max(0.01, math.sqrt(disp[0]*disp[0] + disp[1]*disp[1]))
                
                node_positions[node] = (
                    node_positions[node][0] + min(dist, 5) * disp[0] / dist,
                    node_positions[node][1] + min(dist, 5) * disp[1] / dist
                )
                
                node_positions[node] = (
                    max(50, min(width - 50, node_positions[node][0])),
                    max(50, min(height - 50, node_positions[node][1]))
                )
        
        for edge in graph.edges():
            source, target = edge
            x1, y1 = node_positions[source]
            x2, y2 = node_positions[target]
            
            self.canvas.create_line(
                x1, y1, x2, y2,
                arrow=tk.LAST,
                fill="gray",
                width=1
            )
        
        node_radius = 20
        for node_id in graph.nodes():
            x, y = node_positions[node_id]
            node_data = graph.nodes[node_id]
            
            node_type = node_data['node'].script_type
            
            if node_type == ScriptType.PYTHON:
                color = "#3498db"
            elif node_type == ScriptType.SHELL:
                color = "#e74c3c"
            elif node_type == ScriptType.C or node_type == ScriptType.CPP:
                color = "#2ecc71"
            elif node_type == ScriptType.ASSEMBLY:
                color = "#f39c12"
            else:
                color = "#95a5a6"
            
            self.canvas.create_oval(
                x - node_radius, y - node_radius,
                x + node_radius, y + node_radius,
                fill=color,
                outline="black",
                width=2
            )
            
            name = os.path.basename(node_data['node'].path)
            self.canvas.create_text(
                x, y,
                text=name[:10] + "..." if len(name) > 10 else name,
                fill="white",
                font=("Arial", 8, "bold")
            )

class LogPanel(ttk.Frame):
    """Panel for displaying execution logs"""
    
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        self.frame = ttk.LabelFrame(self, text="Execution Log")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.scrollbar = ttk.Scrollbar(self.frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(self.frame, wrap=tk.WORD, yscrollcommand=self.scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.log_text.yview)
        
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("info", foreground="blue")
        
        self.log_text.config(state=tk.DISABLED)
    
    def log(self, message, tag=None):
        """Add a message to the log"""
        self.log_text.config(state=tk.NORMAL)
        
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] ", "info")
        
        self.log_text.insert(tk.END, f"{message}\n", tag)
        
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def clear(self):
        """Clear the log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

class SystemBuilderApp(TkinterDnD.Tk):
    """Main System Builder Application"""
    
    def __init__(self):
        super().__init__()
        
        self.title("System Builder")
        self.geometry("1024x768")
        
        self.script_analyzer = ScriptAnalyzer()
        self.runtime_env = None
        self.execution_engine = None
        
        self.create_widgets()
        
        try:
            self.drop_target_register('DND_Files')
            self.dnd_bind('<<Drop>>', self.on_drop)
        except:
            pass
    
    def create_widgets(self):
        """Create application widgets"""
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.script_panel = ScriptPanel(self.left_panel, self)
        self.script_panel.pack(fill=tk.BOTH, expand=True)
        
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.dependency_panel = DependencyPanel(self.notebook, self)
        self.notebook.add(self.dependency_panel, text="Dependencies")
        
        self.log_panel = LogPanel(self.notebook, self)
        self.notebook.add(self.log_panel, text="Logs")
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        self.statusbar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.create_menu()
    
    def create_menu(self):
        """Create application menu"""
        self.menubar = tk.Menu(self)
        
        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(label="Add Scripts", command=self.script_panel.on_add_script)
        file_menu.add_command(label="Analyze Scripts", command=self.analyze_scripts)
        file_menu.add_command(label="Run Scripts", command=self.run_scripts)
        file_menu.add_separator()
        file_menu.add_command(label="Export Configuration", command=self.export_configuration)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        self.menubar.add_cascade(label="File", menu=file_menu)
        
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        tools_menu.add_command(label="Clear Logs", command=self.log_panel.clear)
        tools_menu.add_command(label="Reset Environment", command=self.reset_environment)
        tools_menu.add_command(label="Preferences", command=self.show_preferences)
        
        self.menubar.add_cascade(label="Tools", menu=tools_menu)
        
        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        
        self.menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=self.menubar)
    
    def on_drop(self, event):
        """Handle drag and drop on the application"""
        self.script_panel.on_drop(event)
    
    def on_scripts_changed(self):
        """Handle script list changes"""
        count = len(self.script_panel.scripts)
        self.status_var.set(f"{count} scripts loaded")
    
    def analyze_scripts(self):
        """Analyze loaded scripts"""
        script_paths = self.script_panel.get_script_paths()
        
        if not script_paths:
            messagebox.showinfo("Info", "No scripts to analyze")
            return
        
        try:
            self.status_var.set("Analyzing scripts...")
            self.log_panel.log("Starting script analysis...", "info")
            
            graph = self.script_analyzer.build_dependency_graph(script_paths)
            
            self.dependency_panel.update_graph(graph)
            
            node_count = len(graph.nodes())
            edge_count = len(graph.edges())
            self.log_panel.log(f"Analysis complete: {node_count} nodes, {edge_count} edges", "success")
            
            self.notebook.select(0)
            
            self.status_var.set(f"Analysis complete: {node_count} scripts analyzed")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.log_panel.log(f"Analysis failed: {e}", "error")
            self.status_var.set("Analysis failed")
    
    def run_scripts(self):
        """Run the analyzed scripts"""
        if not self.script_analyzer.dependency_graph or not self.script_analyzer.dependency_graph.nodes():
            messagebox.showinfo("Info", "No scripts to run. Analyze scripts first.")
            return
        
        try:
            self.status_var.set("Resolving execution order...")
            execution_order = self.script_analyzer.resolve_execution_order()
            
            if not execution_order:
                messagebox.showinfo("Info", "No scripts to execute")
                return
            
            self.log_panel.log("Creating runtime environment...", "info")
            self.runtime_env = RuntimeEnvironment()
            
            self.log_panel.log("Installing requirements...", "info")
            self.runtime_env.install_requirements(execution_order)
            
            self.execution_engine = ExecutionEngine(self.runtime_env)
            
            self.notebook.select(1)
            
            self.log_panel.log(f"Executing {len(execution_order)} scripts...", "info")
            self.status_var.set(f"Executing {len(execution_order)} scripts...")
            
            def execution_callback(script, result):
                if result.success:
                    self.log_panel.log(
                        f"Script '{script.name}' executed successfully in {result.duration:.2f}s",
                        "success"
                    )
                else:
                    self.log_panel.log(
                        f"Script '{script.name}' failed with code {result.return_code}",
                        "error"
                    )
                    self.log_panel.log(f"Error: {result.stderr}", "error")
            
            results = self.execution_engine.execute_all(execution_order, execution_callback)
            
            success_count = sum(1 for r in results.values() if r.success)
            fail_count = len(results) - success_count
            
            summary = f"Execution complete: {success_count} succeeded, {fail_count} failed"
            
            if fail_count > 0:
                self.log_panel.log(summary, "error")
                self.status_var.set(summary)
            else:
                self.log_panel.log(summary, "success")
                self.status_var.set(summary)
            
            if fail_count > 0:
                messagebox.showwarning("Execution Complete", summary)
            else:
                messagebox.showinfo("Execution Complete", summary)
            
        except Exception as e:
            messagebox.showerror("Error", f"Execution failed: {e}")
            self.log_panel.log(f"Execution failed: {e}", "error")
            self.status_var.set("Execution failed")
    
    def export_configuration(self):
        """Export system configuration"""
        if not self.script_analyzer.dependency_graph or not self.script_analyzer.dependency_graph.nodes():
            messagebox.showinfo("Info", "No scripts analyzed. Analyze scripts first.")
            return
        
        try:
            filepath = filedialog.asksaveasfilename(
                title="Export Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not filepath:
                return
            
            config = {
                "scripts": [],
                "dependencies": [],
                "metadata": {
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "version": "1.0"
                }
            }
            
            for node_id, node_data in self.script_analyzer.dependency_graph.nodes(data=True):
                node = node_data['node']
                script_config = {
                    "id": node.id,
                    "name": node.name,
                    "path": node.path,
                    "type": node.script_type.name,
                    "metadata": node.metadata
                }
                config["scripts"].append(script_config)
            
            for source, target, data in self.script_analyzer.dependency_graph.edges(data=True):
                dep_type = data.get('type', DependencyType.EXECUTION).name
                config["dependencies"].append({
                    "source": source,
                    "target": target,
                    "type": dep_type
                })
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.log_panel.log(f"Configuration exported to {filepath}", "success")
            self.status_var.set(f"Configuration exported")
            
            messagebox.showinfo("Export Complete", f"Configuration exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
            self.log_panel.log(f"Export failed: {e}", "error")
    
    def reset_environment(self):
        """Reset the runtime environment"""
        if self.runtime_env:
            self.runtime_env.cleanup()
            self.runtime_env = None
            
        if self.execution_engine:
            self.execution_engine = None
        
        self.log_panel.log("Runtime environment reset", "info")
        self.status_var.set("Environment reset")
    
    def show_preferences(self):
        """Show preferences dialog"""
        dialog = tk.Toplevel(self)
        dialog.title("Preferences")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Preferences", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(dialog, text="No configurable preferences yet").pack(pady=20)
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def show_documentation(self):
        """Show documentation"""
        dialog = tk.Toplevel(self)
        dialog.title("Documentation")
        dialog.geometry("600x400")
        dialog.transient(self)
        
        text = tk.Text(dialog, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        docs = """
# System Builder Documentation

## Overview
This application allows you to build systems by dragging and dropping scripts.

## Features
- Drag and drop interface
- Dependency analysis
- Automatic environment configuration
- Script execution in correct order

## Usage
1. Add scripts by dragging them onto the application
2. Click "Analyze Scripts" to build the dependency graph
3. Click "Run Scripts" to execute them in the correct order
4. Export your configuration for later use

## Supported Script Types
- Python (.py)
- Shell (.sh, .bash)
- C/C++ (.c, .cpp)
- Assembly (.asm, .s)
- JavaScript (.js)

## Advanced Features
- Script dependency detection
- Environment configuration
- Parallel execution
"""
        text.insert(tk.END, docs)
        text.config(state=tk.DISABLED)
        
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About System Builder",
            "System Builder v1.0\n\n"
            "A drag and drop system builder with automatic script configuration.\n\n"
            "Copyright © 2025\n"
            "All rights reserved."
        )
    
    def quit(self):
        """Clean up and quit the application"""
        if self.runtime_env:
            self.runtime_env.cleanup()
        super().quit()

def main():
    """Main entry point for the application"""
    app = SystemBuilderApp()
    app.mainloop()

if __name__ == "__main__":
    main()
EOF

# Step 3: Create or update the systembuilder launcher
echo "Creating $SYSTEMBUILDER launcher..."
cat > "$SYSTEMBUILDER" << EOF
#!/bin/bash
"$PYTHON" "$SYSTEM_BUILDER_PY" "\$@"
EOF
chmod +x "$SYSTEMBUILDER"

# Step 4: Start the System Builder
echo "Starting System Builder..."
"$SYSTEMBUILDER"

# Deactivate virtual environment when done
deactivate
