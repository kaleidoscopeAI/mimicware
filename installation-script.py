#!/usr/bin/env python3
"""
System Builder Installation Script
Sets up the environment and dependencies for the System Builder application
"""

import os
import sys
import subprocess
import shutil
import platform
import tempfile
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemBuilder-Install")

# Default installation directories
DEFAULT_INSTALL_DIR = {
    'Windows': os.path.join(os.path.expanduser('~'), 'SystemBuilder'),
    'Linux': os.path.join(os.path.expanduser('~'), 'systembuilder'),
    'Darwin': os.path.join(os.path.expanduser('~'), 'Applications', 'SystemBuilder')
}

# Required packages
REQUIRED_PACKAGES = [
    'networkx',
    'numpy',
    'Pillow',
    'matplotlib',
    'tkinterdnd2'  # Added for drag-and-drop support
]

# Source files to copy
SOURCE_FILES = [
    'system-builder.py',
    'script-analyzer.c',
    'graph-engine.py',
    'advanced-gui.py'
]

class Installer:
    """Installer for the System Builder application"""
    
    def __init__(self, install_dir: str = None, venv: bool = True, 
                 build_c_modules: bool = True):
        self.system = platform.system()
        self.install_dir = install_dir or DEFAULT_INSTALL_DIR.get(self.system)
        self.venv = venv
        self.build_c_modules = build_c_modules
        self.venv_dir = os.path.join(self.install_dir, 'venv')
        self.temp_dir = tempfile.mkdtemp(prefix='systembuilder_install_')
        
        # Determine executable paths
        if self.system == 'Windows':
            self.python_exe = 'python.exe'
            self.pip_exe = 'pip.exe'
            self.venv_python = os.path.join(self.venv_dir, 'Scripts', 'python.exe')
            self.venv_pip = os.path.join(self.venv_dir, 'Scripts', 'pip.exe')
        else:
            self.python_exe = 'python3'
            self.pip_exe = 'pip3'
            self.venv_python = os.path.join(self.venv_dir, 'bin', 'python')
            self.venv_pip = os.path.join(self.venv_dir, 'bin', 'pip')
    
    def run(self) -> bool:
        """Run the installation process"""
        try:
            logger.info(f"Starting System Builder installation to {self.install_dir}")
            
            # Create installation directory structure
            self.create_directories()
            
            # Create virtual environment if requested
            if self.venv:
                self.create_virtual_environment()
            
            # Get pip and python executables
            pip = self.venv_pip if self.venv else self.pip_exe
            python = self.venv_python if self.venv else self.python_exe
            
            # Install required packages
            self.install_packages(pip, REQUIRED_PACKAGES)
            
            # Build C modules if requested
            if self.build_c_modules:
                self.build_c_optimizer()
            
            # Copy source files
            self.copy_source_files()
            
            # Create launch scripts
            self.create_launch_scripts(python)
            
            # Clean up
            self.cleanup()
            
            logger.info(f"Installation completed successfully. System Builder installed to {self.install_dir}")
            logger.info(f"Start System Builder by running: {os.path.join(self.install_dir, 'systembuilder')}")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            self.cleanup()
            return False
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        subdirs = ['bin', 'lib', 'scripts', 'templates', 'workflows', 'logs']
        for subdir in subdirs:
            dir_path = os.path.join(self.install_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def create_virtual_environment(self) -> None:
        """Create a virtual environment"""
        if os.path.exists(self.venv_dir):
            logger.info(f"Virtual environment already exists at {self.venv_dir}")
            return
        
        logger.info(f"Creating virtual environment in {self.venv_dir}")
        subprocess.check_call([self.python_exe, '-m', 'venv', self.venv_dir])
        logger.info("Virtual environment created successfully")
    
    def install_packages(self, pip: str, packages: List[str]) -> None:
        """Install Python packages"""
        if not packages:
            return
        
        logger.info(f"Installing packages: {', '.join(packages)}")
        cmd = [pip, 'install'] + packages
        subprocess.check_call(cmd)
        logger.info("Packages installed successfully")
    
    def build_c_optimizer(self) -> None:
        """Build the C optimizer module"""
        logger.info("Building C optimizer module")
        
        # Check if C source exists
        script_analyzer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'script-analyzer.c')
        if not os.path.exists(script_analyzer_path):
            logger.warning("C source file not found, generating from template")
            c_code = self.generate_c_optimizer_code()
            script_analyzer_path = os.path.join(self.temp_dir, 'script-analyzer.c')
            with open(script_analyzer_path, 'w') as f:
                f.write(c_code)
        
        # Compile the C code
        output_path = os.path.join(self.install_dir, 'lib', 'optimizer')
        try:
            if self.system == 'Windows':
                output_path += '.dll'
                subprocess.check_call(['gcc', script_analyzer_path, '-shared', '-o', output_path])
            elif self.system == 'Darwin':
                output_path += '.dylib'
                subprocess.check_call(['gcc', script_analyzer_path, '-shared', '-fPIC', '-o', output_path])
            else:
                output_path += '.so'
                subprocess.check_call(['gcc', script_analyzer_path, '-shared', '-fPIC', '-o', output_path])
            logger.info(f"C optimizer module built successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build C optimizer module: {e}")
            logger.warning("Falling back to Python implementation")
    
    def generate_c_optimizer_code(self) -> str:
        """Generate C optimizer code if not available"""
        return """#include <stdio.h>
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
    
    def copy_source_files(self) -> None:
        """Copy source files to the installation directory"""
        logger.info("Copying source files")
        
        for file in SOURCE_FILES:
            source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
            if not os.path.exists(source_path):
                logger.warning(f"Source file not found: {source_path}")
                continue
            
            if file.endswith('.py'):
                dest_path = os.path.join(self.install_dir, 'bin', file)
            elif file.endswith('.c'):
                dest_path = os.path.join(self.install_dir, 'lib', file)
            else:
                dest_path = os.path.join(self.install_dir, file)
            
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied {file} to {dest_path}")
            if file.endswith('.py') and self.system != 'Windows':
                os.chmod(dest_path, 0o755)
    
    def create_launch_scripts(self, python: str) -> None:
        """Create launch scripts for the application"""
        logger.info("Creating launch scripts")
        
        bin_dir = os.path.join(self.install_dir, 'bin')
        
        if self.system == 'Windows':
            launcher_path = os.path.join(bin_dir, 'systembuilder.bat')
            with open(launcher_path, 'w') as f:
                f.write('@echo off\n')
                f.write(f'"{python}" "{os.path.join(bin_dir, "system-builder.py")}" %*\n')
        else:
            launcher_path = os.path.join(bin_dir, 'systembuilder')
            with open(launcher_path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'"{python}" "{os.path.join(bin_dir, "system-builder.py")}" "$@"\n')
            os.chmod(launcher_path, 0o755)
        
        # Create configuration file
        config_path = os.path.join(self.install_dir, 'config.json')
        config = {
            'version': '1.0',
            'python_path': python,
            'install_dir': self.install_dir,
            'venv': self.venv,
            'bin_dir': bin_dir,
            'lib_dir': os.path.join(self.install_dir, 'lib'),
            'scripts_dir': os.path.join(self.install_dir, 'scripts'),
            'templates_dir': os.path.join(self.install_dir, 'templates'),
            'workflows_dir': os.path.join(self.install_dir, 'workflows'),
            'logs_dir': os.path.join(self.install_dir, 'logs')
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created configuration file: {config_path}")
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="System Builder Installer")
    parser.add_argument('--install-dir', help='Installation directory')
    parser.add_argument('--no-venv', action='store_true', help='Do not create a virtual environment')
    parser.add_argument('--no-c-modules', action='store_true', help='Do not build C modules')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    return parser.parse_args()

def main() -> int:
    """Main entry point"""
    args = parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create installer
    installer = Installer(
        install_dir=args.install_dir,
        venv=not args.no_venv,
        build_c_modules=not args.no_c_modules
    )
    
    # Run installation
    success = installer.run()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
