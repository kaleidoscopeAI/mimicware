#!/usr/bin/env python3
"""
Dynamic Cube Setup Script
Installs dependencies, sets up directory structure, and launches the Dynamic Cube GUI
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import time
import argparse
import venv
from pathlib import Path
import pkg_resources
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import socket

# Configuration
VENV_DIR = "venv"
PROJECT_DIR = "dynamic_cube"
DATA_DIR = "data"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
DEPENDENCIES = [
    "numpy>=1.20.0",
    "matplotlib>=3.5.0",
    "networkx>=2.7.0",
    "scipy>=1.8.0",
    "scikit-learn>=1.0.0",
    "flask>=2.0.0",
    "flask-socketio>=5.2.0",
    "flask-cors>=3.0.0",
    "eventlet>=0.33.0",
    "dataclasses-json>=0.5.7",
    "pillow>=9.0.0",
    "pandas>=1.4.0",
    "psutil>=5.9.0"
]
GUI_PORT = 8080
API_PORT = 5000

class SetupLogger:
    """Simple logger for the setup process"""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            self.file = open(log_file, 'w')
        else:
            self.file = None
    
    def log(self, message, level="INFO"):
        """Log a message to console and file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}"
        print(formatted_message)
        if self.file:
            self.file.write(formatted_message + "\n")
            self.file.flush()
    
    def close(self):
        """Close the log file"""
        if self.file:
            self.file.close()
            self.file = None

def check_python_version():
    """Check if Python version is compatible"""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        logger.log(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required", "ERROR")
        logger.log(f"Current Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}", "ERROR")
        sys.exit(1)
    
    logger.log(f"Python version check passed: {current_version[0]}.{current_version[1]}.{current_version[2]}")

def create_virtual_environment():
    """Create a virtual environment for the project"""
    if os.path.exists(VENV_DIR):
        logger.log(f"Virtual environment already exists at {VENV_DIR}")
        return
    
    logger.log("Creating virtual environment...")
    try:
        venv.create(VENV_DIR, with_pip=True)
        logger.log("Virtual environment created successfully")
    except Exception as e:
        logger.log(f"Error creating virtual environment: {str(e)}", "ERROR")
        sys.exit(1)

def get_venv_python():
    """Get the path to the Python executable in the virtual environment"""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "python")

def get_venv_pip():
    """Get the path to pip in the virtual environment"""
    if platform.system() == "Windows":
        return os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        return os.path.join(VENV_DIR, "bin", "pip")

def install_dependencies():
    """Install required dependencies in the virtual environment"""
    logger.log("Installing dependencies...")
    
    pip_path = get_venv_pip()
    
    # Upgrade pip first
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        logger.log("Upgraded pip to latest version")
    except subprocess.CalledProcessError as e:
        logger.log(f"Warning: Failed to upgrade pip: {str(e)}", "WARNING")
    
    # Install dependencies
    for dep in DEPENDENCIES:
        logger.log(f"Installing {dep}...")
        try:
            subprocess.run([pip_path, "install", dep], check=True)
            logger.log(f"Successfully installed {dep}")
        except subprocess.CalledProcessError as e:
            logger.log(f"Error installing {dep}: {str(e)}", "ERROR")
            return False
    
    logger.log("All dependencies installed successfully")
    return True

def create_directory_structure():
    """Create the project directory structure"""
    dirs = [
        PROJECT_DIR,
        os.path.join(PROJECT_DIR, DATA_DIR),
        os.path.join(PROJECT_DIR, OUTPUT_DIR),
        os.path.join(PROJECT_DIR, LOGS_DIR)
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.log(f"Created directory: {directory}")
        else:
            logger.log(f"Directory already exists: {directory}")

def create_configuration():
    """Create default configuration files"""
    config_file = os.path.join(PROJECT_DIR, "config.json")
    
    default_config = {
        "dimensions": 4,
        "resolution": 32,
        "qubit_depth": 8,
        "initial_nodes": 25,
        "max_connections": 4,
        "connection_radius": 0.6,
        "evolution_interval": 20,
        "output_dir": OUTPUT_DIR,
        "data_dir": DATA_DIR,
        "logs_dir": LOGS_DIR,
        "auto_save_interval": 100,
        "visualization_interval": 50,
        "max_steps": 1000,
        "gui_port": GUI_PORT,
        "api_port": API_PORT
    }
    
    # Create configuration file if it doesn't exist
    if not os.path.exists(config_file):
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.log(f"Created default configuration at {config_file}")
    else:
        logger.log(f"Configuration already exists at {config_file}")

def copy_project_files():
    """Copy core Python files to the project directory"""
    source_files = {
        "quantum_string_cube.py": """#!/usr/bin/env python3
\"""
QuantumStringCube: A groundbreaking multi-dimensional cognitive framework
that implements a novel computational paradigm using quantum-inspired algorithms,
resonance harmonics, and adaptive evolutionary principles.
\"""

import numpy as np
import hashlib
import time
import uuid
import math
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque

# ======================================================================
# CORE QUANTUM-INSPIRED PRIMITIVES
# ======================================================================

class ResonanceMode(Enum):
    \"""Fundamental resonance modes for harmonic computation\"""
    CONSTRUCTIVE = auto()   # Amplifying patterns
    DESTRUCTIVE = auto()    # Cancelling noise
    REFLECTIVE = auto()     # Preserving energy
    REFRACTIVE = auto()     # Transforming pathways
    ENTROPIC = auto()       # Dissipating obsolete patterns


class QuantumString:
    \"""
    Implementation of a novel quantum-inspired string theory construct
    that uses oscillatory patterns for computation rather than traditional bits.
    \"""
    def __init__(self, dimensions: int = 4, string_length: int = 64, tension: float = 0.8):
        self.dimensions = dimensions
        self.string_length = string_length
        self.base_tension = tension
        self.amplitude = np.zeros((string_length, dimensions), dtype=np.complex128)
        self.phase = np.zeros(string_length, dtype=np.float64)
        self.frequency = np.zeros(string_length, dtype=np.float64)
        self.harmonics =# Fibonacci-based harmonics
        
        # Initialize with quantum noise
        self._initialize_string()
    
    def _initialize_string(self):
        \"""Initialize the quantum string with structured noise\"""
        # Create initial complex wave pattern
        for i in range(self.string_length):
            # Phase varies along the string
            self.phase[i] = (i / self.string_length) * 2 * np.pi
            
            # Different base frequency components
            self.frequency[i] = 1.0 + 0.5 * np.sin(self.phase[i])
            
            # Initialize amplitudes in each dimension
            for d in range(self.dimensions):
                # Create a complex amplitude with specific phase relationship
                phase_shift = 2 * np.pi * d / self.dimensions
                self.amplitude[i, d] = complex(
                    np.cos(self.phase[i] + phase_shift) * 0.1,
                    np.sin(self.phase[i] + phase_shift) * 0.1
                )
    
    def evolve(self, dt: float = 0.01):
        \"""Evolve the string state using wave equations\"""
        # Apply wave equation dynamics (simplified form)
        new_amplitude = np.zeros_like(self.amplitude)
        
        # Implement a discretized version of the wave equation
        for i in range(1, self.string_length - 1):
            for d in range(self.dimensions):
                # Wave equation: ∂²y/∂t² = tension * ∂²y/∂x²
                d2y_dx2 = (self.amplitude[i+1, d] - 2*self.amplitude[i, d] + self.amplitude[i-1, d]) / (1/self.string_length)**2
                wave_term = self.base_tension * d2y_dx2
                
                # Add harmonic oscillator term
                harmonic_term = -self.frequency[i]**2 * self.amplitude[i, d]
                
                # Combine terms
                d2y_dt2 = wave_term + harmonic_term
                
                # Euler integration
                new_amplitude[i, d] = self.amplitude[i, d] + dt * d2y_dt2
        
        # Apply boundary conditions (fixed ends)
        new_amplitude[0] = 0
        new_amplitude[-1] = 0
        
        self.amplitude = new_amplitude
        
        # Evolve phase
        self.phase = (self.phase + dt * self.frequency) % (2 * np.pi)
    
    def apply_resonance(self, mode: ResonanceMode, target_points: List[int], strength: float = 0.1):
        \"""Apply a specific resonance pattern to selected points on the string\"""
        if not target_points:
            return
            
        # Create the resonance pattern
        if mode == ResonanceMode.CONSTRUCTIVE:
            # Amplify existing patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.amplitude[point] *= (1 + strength)
        
        elif mode == ResonanceMode.DESTRUCTIVE:
            # Dampen existing patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.amplitude[point] *= (1 - strength)
        
        elif mode == ResonanceMode.REFLECTIVE:
            # Create mirror-like patterns
            for point in target_points:
                if 0 < point < self.string_length - 1:
                    self.amplitude[point] = self.amplitude[point-1]
        
        elif mode == ResonanceMode.REFRACTIVE:
            # Shift phase, bending the patterns
            for point in target_points:
                if 0 <= point < self.string_length:
                    self.phase[point] = (self.phase[point] + strength * np.pi) % (2 * np.pi)
        
        elif mode == ResonanceMode.ENTROPIC:
            # Introduce controlled chaos
            for point in target_points:
                if 0 <= point < self.string_length:
                    random_phase = np.random.random() * 2 * np.pi
                    decay_factor = 1 - strength
                    self.amplitude[point] = self.amplitude[point] * decay_factor + \
                                           complex(np.cos(random_phase), np.sin(random_phase)) * strength * 0.1
    
    def get_energy(self) -> float:
        \"""Calculate total string energy\"""
        # Sum of squared amplitudes across all points and dimensions
        return np.sum(np.abs(self.amplitude)**2)
    
    def get_state_vector(self) -> np.ndarray:
        \"""Get flattened state representation for analysis\"""
        real_parts = self.amplitude.real.flatten()
        imag_parts = self.amplitude.imag.flatten()
        phases = self.phase
        return np.concatenate([real_parts, imag_parts, phases])
    
    def apply_interference(self, other_string: 'QuantumString', coupling_strength: float = 0.1):
        \"""Create interference patterns between two strings\"""
        # Ensure strings have compatible dimensions
        common_dims = min(self.dimensions, other_string.dimensions)
        common_length = min(self.string_length, other_string.string_length)
        
        # Apply interference in common dimensions
        for i in range(common_length):
            for d in range(common_dims):
                # Quantum interference formula
                self.amplitude[i, d] += coupling_strength * other_string.amplitude[i, d]
                
                # Normalize to prevent unbounded growth
                mag = abs(self.amplitude[i, d])
                if mag > 1.0:
                    self.amplitude[i, d] /= mag
    
    def encode_data(self, data: bytes, dimension: int = 0):
        """Encode binary data into string oscillations"""
        if dimension >= self.dimensions:
            dimension = 0
            
        # Convert bytes to bit pattern
        bits =
        for byte in data:
            for i in range(8):
                bits.append((byte >> i) & 1)
        
        # Scale to string length
        scale_factor = len(bits) / self.string_length
        
        # Encode bits as phase shifts
        for i in range(self.string_length):
            bit_index = int(i * scale_factor)
            if bit_index < len(bits):
                bit = bits[bit_index]
                if bit:
                    # Encode 1 as positive phase shift
                    phase_shift = 0.1 * np.pi
                else:
                    # Encode 0 as negative phase shift
                    phase_shift = -0.1 * np.pi
                
                # Apply phase shift to this dimension
                current_phase = np.angle(self.amplitude[i, dimension])
                magnitude = abs(self.amplitude[i, dimension])
                self.amplitude[i, dimension] = complex(
                    magnitude * np.cos(current_phase + phase_shift),
                    magnitude * np.sin(current_phase + phase_shift)
                )
    
    def extract_pattern(self, dimension: int = 0) -> List[float]:
        """Extract dominant pattern from a specific dimension"""
        if dimension >= self.dimensions:
            dimension = 0
            
        # Get amplitudes from specified dimension
        amplitudes = [abs(self.amplitude[i, dimension]) for i in range(self.string_length)]
        
        # Get phases from specified dimension
        phases = [np.angle(self.amplitude[i, dimension]) for i in range(self.string_length)]
        
        # Combine into pattern (amplitude * phase_factor)
        pattern = [amplitudes[i] * np.cos(phases[i]) for i in range(self.string_length)]
        
        return pattern


class HarmonicTensorField:
    """
    A novel tensor field representation using harmonics theory
    for efficient multi-dimensional information propagation.
    """
    def __init__(self, dimensions: int = 4, resolution: int = 32):
        self.dimensions = dimensions
        self.resolution = resolution
        
        # Initialize tensor field
        self.field_shape = tuple([resolution] * dimensions)
        self.amplitude_tensor = np.zeros(self.field_shape, dtype=np.complex128)
        self.frequency_tensor = np.zeros(self.field_shape, dtype=np.float64)
        self.phase_tensor = np.zeros(self.field_shape, dtype=np.float64)
        
        # Initialize with structured patterns
        self._initialize_field()
        
        # Tension metrics
        self.tension_field = np.zeros(self.field_shape, dtype=np.float64)
        self.resonance_points =
    
    def _initialize_field(self):
        """Initialize the harmonic tensor field with structured patterns"""
        # Create coordinate arrays
        coords = [np.linspace(-1, 1, self.resolution) for _ in range(self.dimensions)]
        
        # Fill tensors with initial values
        # We'll use a multidimensional mesh approach
        if self.dimensions <= 3:  # For lower dimensions we can use numpy's meshgrid
            mesh_coords = np.meshgrid(*coords, indexing='ij')
            
            # Calculate distance from origin for each point
            r_squared = np.zeros(self.field_shape)
            for dim_coords in mesh_coords:
                r_squared += dim_coords**2
            r = np.sqrt(r_squared)
            
            # Set initial values based on distance
            self.amplitude_tensor = 0.1 * np.exp(-3 * r) * np.exp(1j * np.pi * r)
            self.frequency_tensor = 1.0 + 0.2 * r
            self.phase_tensor = np.pi * r
            
        else:  # For higher dimensions, we'll use a loop approach
            # Create indices for all positions
            indices = np.indices(self.field_shape)
            
            # Convert indices to coordinates [-1, 1]
            coords_from_indices = [(2.0 * indices[d] / (self.resolution - 1) - 1.0) for d in range(self.dimensions)]
            
            # Calculate distance from origin for each point
            r_squared = np.zeros(self.field_shape)
            for dim_coords in coords_from_indices:
                r_squared += dim_coords**2
            r = np.sqrt(r_squared)
            
            # Set initial values
            self.amplitude_tensor = 0.1 * np.exp(-3 * r) * np.exp(1j * np.pi * r)
            self.frequency_tensor = 1.0 + 0.2 * r
            self.phase_tensor = np.pi * r
    
    def evolve(self, dt: float = 0.01):
        """Evolve the field using a quantum-inspired wave equation"""
        # Create new tensor for updated values
        new_amplitude = np.zeros_like(self.amplitude_tensor, dtype=np.complex128)
        
        # Define the laplacian kernel for our dimensions
        # We'll use a simple central difference approximation
        
        # Loop through all points excluding boundaries
        # For high dimensions, we need a different approach than nested loops
        
        # First create slices for the center point and its neighbors
        center_slice = tuple(slice(1, self.resolution-1) for _ in range(self.dimensions))
        
        # Apply discretized wave equation at all interior points at once
        new_amplitude[center_slice] = self.amplitude_tensor[center_slice]
        
        # For each dimension, add the second derivative contribution
        for d in range(self.dimensions):
            # Create slices for the forward and backward points in this dimension
            forward_slice = list(center_slice)
            forward_slice[d] = slice(2, self.resolution)
            forward_slice = tuple(forward_slice)
            
            backward_slice = list(center_slice)
            backward_slice[d] = slice(0, self.resolution-2)
            backward_slice = tuple(backward_slice)
            
            # Add the second derivative term from this dimension
            d2y_dx2 = (self.amplitude_tensor[forward_slice] - 
                       2 * self.amplitude_tensor[center_slice] + 
                       self.amplitude_tensor[backward_slice])
            
            # Accumulate the laplacian contributions
            new_amplitude[center_slice] += 0.1 * d2y_dx2
        
        # Apply frequency-based oscillation
        oscillation_factor = np.exp(1j * dt * self.frequency_tensor[center_slice])
        new_amplitude[center_slice] *= oscillation_factor
        
        # Update the amplitude tensor (interior points only)
        self.amplitude_tensor[center_slice] = new_amplitude[center_slice]
        
        # Evolve phases
        self.phase_tensor = (self.phase_tensor + dt * self.frequency_tensor) % (2 * np.pi)
        
        # Update tension field
        self._calculate_tension_field()
    
    def _calculate_tension_field(self):
        """Calculate the tension field based on amplitude gradients"""
        # Initialize tension field
        self.tension_field = np.zeros(self.field_shape, dtype=np.float64)
        
        # For each dimension, calculate the gradient
        for d in range(self.dimensions):
            # Create slices for calculating gradient
            forward_slice = list(slice(None) for _ in range(self.dimensions))
            forward_slice[d] = slice(1, None)
            forward_slice = tuple(forward_slice)
            
            backward_slice = list(slice(None) for _ in range(self.dimensions))
            backward_slice[d] = slice(0, -1)
            backward_slice = tuple(backward_slice)
            
            # Calculate gradient magnitude in this dimension
            gradient = np.abs(self.amplitude_tensor[forward_slice] - self.amplitude_tensor[backward_slice])
            
            # Add to the tension field (on the appropriate slice)
            gradient_slice = list(slice(None) for _ in range(self.dimensions))
            gradient_slice[d] = slice(0, -1)  # Match the size reduction from gradient calculation
            self.tension_field[tuple(gradient_slice)] += gradient
        
        # Normalize tension field
        max_tension = np.max(self.tension_field)
        if max_tension > 0:
            self.tension_field /= max_tension
        
        # Find resonance points (high tension points)
        self._find_resonance_points()
    
    def _find_resonance_points(self, threshold: float = 0.7):
        """Find points with high tension (resonance points)"""
        # Clear previous resonance points
        self.resonance_points =
        
        # Find indices where tension exceeds threshold
        high_tension_indices = np.where(self.tension_field > threshold)
        
        # Convert to list of coordinate tuples
        for i in range(len(high_tension_indices[0])):
            point = tuple(high_tension_indices[d][i] for d in range(self.dimensions))
            tension = self.tension_field[point]
            self.resonance_points.append((point, tension))
    
    def inject_pattern(self, pattern: np.ndarray, position: Tuple[int, ...], dimension_mapping: Optional[List[int]] = None):
        """Inject a pattern into the field at a specific position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
            
        pattern_shape = pattern.shape
        
        # If dimension mapping is not provided, use default mapping
        if dimension_mapping is None:
            dimension_mapping = list(range(min(len(pattern_shape), self.dimensions)))
            
        # Create slices for the target region
        target_slices =
        for d in range(self.dimensions):
            # If this dimension is mapped from the pattern
            if d in dimension_mapping:
                # Find the corresponding dimension in the pattern
                pattern_dim = dimension_mapping.index(d)
                dim_size = pattern_shape[pattern_dim]
                
                # Calculate start and end indices
                start = position[d]
                end = min(start + dim_size, self.resolution)
                
                target_slices.append(slice(start, end))
            else:
                # Not mapped, use single position
                target_slices.append(slice(position[d], position[d] + 1))
        
        # Create slices for the source region
        source_slices =
        for d in range(len(pattern_shape)):
            # Calculate how much of the pattern fits
            if d < len(dimension_mapping):
                mapped_dim = dimension_mapping[d]
                target_size = target_slices[mapped_dim].stop - target_slices[mapped_dim].start
                source_slices.append(slice(0, target_size))
            else:
                source_slices.append(slice(None))
        
        # Inject the pattern
        # Need to handle complex target indexing
        target_region = self.amplitude_tensor[tuple(target_slices)]
        source_region = pattern[tuple(source_slices)]
        
        # Match dimensions through broadcasting if needed
        target_shape = target_region.shape
        source_shape = source_region.shape
        
        if target_shape != source_shape:
            # Create a new array that will match the target shape
            expanded_source = np.zeros(target_shape, dtype=source_region.dtype)
            
            # Find the overlap shape
            overlap_shape = tuple(min(t, s) for t, s in zip(target_shape, source_shape))
            
            # Create slices for the overlap region
            overlap_slices = tuple(slice(0, s) for s in overlap_shape)
            
            # Copy the source into the expanded array
            expanded_source[overlap_slices] = source_region[overlap_slices]
            source_region = expanded_source
        
        # Inject the pattern
        self.amplitude_tensor[tuple(target_slices)] = source_region
    
    def extract_pattern(self, position: Tuple[int, ...], dimensions: List[int], size: List[int]) -> np.ndarray:
        """Extract a pattern from specific dimensions at a position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
        
        # Create slices for the region to extract
        extract_slices =
        for d in range(self.dimensions):
            if d in dimensions:
                # This dimension will be included in the output
                idx = dimensions.index(d)
                extract_size = size[idx] if idx < len(size) else 1
                
                # Calculate start and end indices
                start = position[d]
                end = min(start + extract_size, self.resolution)
                
                extract_slices.append(slice(start, end))
            else:
                # Not included, use single position
                extract_slices.append(slice(position[d], position[d] + 1))
        
        # Extract the pattern
        pattern = self.amplitude_tensor[tuple(extract_slices)]
        
        # Squeeze out dimensions that are size 1 (from non-included dimensions)
        pattern = np.squeeze(pattern)
        
        return pattern
    
    def apply_resonance(self, position: Tuple[int, ...], radius: int, strength: float = 0.1, mode: ResonanceMode = ResonanceMode.CONSTRUCTIVE):
        """Apply a resonance pattern centered at a position"""
        if len(position) != self.dimensions:
            raise ValueError(f"Position must have {self.dimensions} coordinates")
            
        # Calculate indices within the radius
        # For high dimensions, we need an efficient approach
        
        # Create ranges for each dimension
        dim_ranges =
        for d in range(self.dimensions):
            start = max(0, position[d] - radius)
            end = min(self.resolution, position[d] + radius + 1)
            dim_ranges.append(range(start, end))
        
        # Generate all combinations of indices within the hypersphere
        from itertools import product
        points_in_range = product(*dim_ranges)
        
        # Filter points by distance
        points_in_sphere =
        for point in points_in_range:
            # Calculate squared distance
            squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
            if squared_dist <= radius**2:
                points_in_sphere.append(point)
        
        # Apply resonance based on mode
        if mode == ResonanceMode.CONSTRUCTIVE:
            # Amplify existing patterns
            for point in points_in_sphere:
                self.amplitude_tensor[point] *= (1 + strength)
                
        elif mode == ResonanceMode.DESTRUCTIVE:
            # Dampen existing patterns
            for point in points_in_sphere:
                self.amplitude_tensor[point] *= (1 - strength)
                
        elif mode == ResonanceMode.REFLECTIVE:
            # Create mirror-like patterns relative to center
            center_amplitude = self.amplitude_tensor[position]
            for point in points_in_sphere:
                # Vector from center to point
                vector = tuple(point[d] - position[d] for d in range(self.dimensions))
                # Calculate reflection factor based on distance
                dist = math.sqrt(sum(v**2 for v in vector))
                if dist > 0:
                    reflection_factor = 1 - dist / (radius + 1)
                    self.amplitude_tensor[point] = center_amplitude * reflection_factor + self.amplitude_tensor[point] * (1 - reflection_factor)
                    
        elif mode == ResonanceMode.REFRACTIVE:
            # Bend wave patterns 
            for point in points_in_sphere:
                # Calculate distance from center
                squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
                dist = math.sqrt(squared_dist)
                # Phase shift proportional to distance
                phase_shift = strength * (1 - dist / radius) * np.pi if radius > 0 else 0
                # Apply phase shift
                current_phase = np.angle(self.amplitude_tensor[point])
                magnitude = abs(self.amplitude_tensor[point])
                self.amplitude_tensor[point] = magnitude * np.exp(1j * (current_phase + phase_shift))
                
        elif mode == ResonanceMode.ENTROPIC:
            # Introduce controlled chaos
            for point in points_in_sphere:
                # Calculate distance factor
                squared_dist = sum((point[d] - position[d])**2 for d in range(self.dimensions))
                dist = math.sqrt(squared_dist)
                dist_factor = 1 - dist / radius if radius > 0 else 0
                
                # Add random component
                if dist_factor > 0:
                    random_phase = np.random.random() * 2 * np.pi
                    entropy_strength = strength * dist_factor
                    decay_factor = 1 - entropy_strength
                    self.amplitude_tensor[point] = (self.amplitude_tensor[point] * decay_factor + 
                                                  complex(np.cos(random_phase), np.sin(random_phase)) * entropy_strength * 0.1)
    
    def get_energy_density(self) -> float:
        """Calculate total energy density in the field"""
        # Sum of squared amplitudes across all points
        return np.sum(np.abs(self.amplitude_tensor)**2) / np.prod(self.field_shape)
    
    def get_resonance_distribution(self) -> Dict[str, Any]:
        """Get statistical distribution of resonance points"""
        result = {
            "count": len(self.resonance_points),
            "average_tension": 0,
            "spatial_distribution": [0] * self.dimensions,
            "max_tension": 0,
            "clusters": 0
        }
        
        if self.resonance_points:
            # Calculate average tension
            tensions = [tension for _, tension in self.resonance_points]
            result["average_tension"] = sum(tensions) / len(tensions)
            result["max_tension"] = max(tensions)
            
            # Calculate spatial distribution
            points = [point for point, _ in self.resonance_points]
            for d in range(self.dimensions):
                d_coords = [p[d] for p in points]
                if d_coords:
                    mean = sum(d_coords) /
