#!/usr/bin/env python3
"""
System Builder Initialization Fix
A script to properly initialize the System Builder environment and fix import errors
"""

import os
import sys
import shutil
import importlib.util
import subprocess
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('init.log')
    ]
)
logger = logging.getLogger("SystemInitializer")

class SystemInitializer:
    """Initialize and fix System Builder environment"""
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.module_paths = self._get_module_paths()
        self.lib_dir = os.path.join(self.base_dir, 'lib')
        self.bin_dir = os.path.join(self.base_dir, 'bin')
        self.env_vars = {}
    
    def _get_module_paths(self):
        """Get paths to all module files"""
        modules = {}
        for file in os.listdir(self.base_dir):
            if file.endswith('.py'):
                module_name = file[:-3].replace('-', '_')
                modules[module_name] = os.path.join(self.base_dir, file)
        return modules
    
    def create_directory_structure(self):
        """Create directory structure required by System Builder"""
        dirs = [
            os.path.join(self.base_dir, 'bin'),
            os.path.join(self.base_dir, 'lib'),
            os.path.join(self.base_dir, 'lib', 'python'),
            os.path.join(self.base_dir, 'scripts'),
            os.path.join(self.base_dir, 'templates'),
            os.path.join(self.base_dir, 'workflows'),
            os.path.join(self.base_dir, 'logs')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def install_dependencies(self):
        """Install required Python packages"""
        required_packages = [
            'numpy',
            'networkx',
            'matplotlib',
            'pillow',
            'tkinter',
        ]
        
        optional_packages = [
            'tkinterdnd2'  # For drag and drop functionality
        ]
        
        # Check which packages are already installed
        installed = self._get_installed_packages()
        to_install = [pkg for pkg in required_packages if pkg.lower() not in installed]
        
        # Also try to install optional packages
        for pkg in optional_packages:
            if pkg.lower() not in