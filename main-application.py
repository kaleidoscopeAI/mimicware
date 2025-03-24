#!/usr/bin/env python3
"""
Main entry point for the System Builder application.
Integrates all components and starts the application.
"""

import os
import sys
import argparse
import logging
import json
import importlib.util
import time
import traceback
from pathlib import Path
import tkinter as tk
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'systembuilder.log'))
    ]
)
logger = logging.getLogger("SystemBuilder")

# Determine application directories
def get_app_directories() -> Dict[str, str]:
    """Get application directories"""
    # Base directory is the parent of this script's directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs = {
        'base': base_dir,
        'bin': os.path.join(base_dir, 'bin'),
        'lib': os.path.join(base_dir, 'lib'),
        'scripts': os.path.join(base_dir, 'scripts'),
        'templates': os.path.join(base_dir, 'templates'),
        'workflows': os.path.join(base_dir, 'workflows'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Ensure all directories exist
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    dirs = get_app_directories()
    config_path = os.path.join(dirs['base'], 'config.json')
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Add directories to config
            config.update(dirs)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    # Default configuration
    return {
        'version': '1.0',
        'python_path': sys.executable,
        **dirs
    }

# Add lib directory to Python path
def setup_python_path() -> None:
    """Add library directory to Python path"""
    dirs = get_app_directories()
    lib_dir = dirs['lib']
    
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    
    # Also add bin directory for components
    bin_dir = dirs['bin']
    if bin_dir not in sys.path:
        sys.path.insert(0, bin_dir)

# Load C optimizer module
def load_c_optimizer() -> Optional[Any]:
    """Load C optimizer module if available"""
    dirs = get_app_directories()
    
    # Check different possible names and paths
    possible_paths = [
        os.path.join(dirs['lib'], 'optimizer.so'),
        os.path.join(dirs['lib'], 'optimizer.dll'),
        os.path.join(dirs['lib'], 'optimizer.dylib'),
        os.path.join(dirs['lib'], 'optimizer')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                import ctypes
                return ctypes.CDLL(path)
            except Exception as e:
                logger.warning(f"Failed to load C optimizer module {path}: {e}")
    
    return None

# Import components
def import_components() -> Dict[str, Any]:
    """Import all required components"""
    components = {}
    
    # System builder core
    try:
        spec = importlib.util.spec_from_file_location(
            "system_builder",
            os.path.join(get_app_directories()['bin'], 'system-builder.py')
        )
        system_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(system_builder)
        components['system_builder'] = system_builder
        logger.info("Imported system-builder module")
    except Exception as e:
        logger.error(f"Failed to import system-builder module: {e}")
        traceback.print_exc()
    
    # Graph engine
    try:
        spec = importlib.util.spec_from_file_location(
            "graph_engine",
            os.path.join(get_app_directories()['bin'], 'graph-engine.py')
        )
        graph_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_engine)
        components['graph_engine'] = graph_engine
        logger.info("Imported graph-engine module")
    except Exception as e:
        logger.error(f"Failed to import graph-engine module: {e}")
        traceback.print_exc()
    
    # Advanced GUI
    try:
        spec = importlib.util.spec_from_file_location(
            "advanced_gui",
            os.path.join(get_app_directories()['bin'], 'advanced-gui.py')
        )
        advanced_gui = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(advanced_gui)
        components['advanced_gui'] = advanced_gui
        logger.info("Imported advanced-gui module")
    except Exception as e:
        logger.error(f"Failed to import advanced-gui module: {e}")
        traceback.print_exc()
    
    return components

# Parse command line arguments
def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="System Builder")
    
    parser.add_argument(
        '--workflow', '-w',
        help='Path to workflow file to open'
    )
    
    parser.add_argument(
        '--execute', '-e',
        action='store_true',
        help='Execute the workflow automatically after opening'
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run in headless mode (no GUI)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory for workflow results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version information and exit'
    )
    
    return parser.parse_args()

# Show version information
def show_version():
    """Show version information"""
    config = load_config()
    version = config.get('version', '1.0')
    
    print(f"System Builder v{version}")
    print("A drag-and-drop system for building and executing script-based workflows")
    print("")
    print(f"Python: {sys.version}")
    print(f"Installation Directory: {config['base']}")

# Run the application in headless mode
def run_headless(args: argparse.Namespace, components: Dict[str, Any]) -> int:
    """Run the application in headless mode"""
    if not args.workflow:
        logger.error("No workflow file specified for headless mode")
        return 1
    
    if not os.path.exists(args.workflow):
        logger.error(f"Workflow file not found: {args.workflow}")
        return 1
    
    try:
        # Load workflow from file
        with open(args.workflow, 'r') as f:
            workflow_data = json.load(f)
        
        # Create workflow graph
        system_builder = components.get('system_builder')
        graph_engine = components.get('graph_engine')
        
        if not system_builder or not graph_engine:
            logger.error("Required components not available")
            return 1
        
        # Create a workflow graph
        analyzer = system_builder.ScriptAnalyzer()
        
        # Extract script paths
        script_paths = []
        for node_id, node_data in workflow_data.get('nodes', {}).items():
            if node_data.get('type') == 'script':
                script_path = node_data.get('properties', {}).get('path', '')
                if script_path:
                    script_paths.append(script_path)
        
        # Build dependency graph
        dependency_graph = analyzer.build_dependency_graph(script_paths)
        
        # Get execution order
        try:
            execution_order = analyzer.resolve_execution_order()
            logger.info(f"Resolved execution order with {len(execution_order)} scripts")
        except Exception as e:
            logger.error(f"Failed to resolve execution order: {e}")
            return 1
        
        # Create runtime environment
        runtime_env = system_builder.RuntimeEnvironment(args.output)
        
        # Install requirements
        runtime_env.install_requirements(execution_order)
        
        # Create execution engine
        execution_engine = system_builder.ExecutionEngine(runtime_env)
        
        # Execute scripts
        logger.info(f"Executing {len(execution_order)} scripts...")
        
        # Define callback for script execution
        def execution_callback(script, result):
            if result.success:
                logger.info(f"Script '{script.name}' executed successfully in {result.duration:.2f}s")
            else:
                logger.error(f"Script '{script.name}' failed with code {result.return_code}")
                logger.error(f"Error: {result.stderr}")
        
        # Execute all scripts
        results = execution_engine.execute_all(execution_order, execution_callback)
        
        # Display summary
        success_count = sum(1 for r in results.values() if r.success)
        fail_count = len(results) - success_count
        
        summary = f"Execution complete: {success_count} succeeded, {fail_count} failed"
        
        if fail_count > 0:
            logger.error(summary)
        else:
            logger.info(summary)
        
        return 0 if fail_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Error in headless execution: {e}")
        traceback.print_exc()
        return 1

# Main entry point
def main() -> int:
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Show version and exit if requested
    if args.version:
        show_version()
        return 0
    
    # Setup Python path
    setup_python_path()
    
    # Import components
    components = import_components()
    
    # Check if all required components are available
    required_components = ['system_builder', 'graph_engine', 'advanced_gui']
    missing_components = [comp for comp in required_components if comp not in components]
    
    if missing_components:
        logger.error(f"Missing required components: {', '.join(missing_components)}")
        return 1
    
    # Run in headless mode if requested
    if args.headless:
        return run_headless(args, components)
    
    # Otherwise, start the GUI
    try:
        # Get the SystemBuilderApp class
        app_class = components['advanced_gui'].SystemBuilderApp
        
        # Create and run the application
        app = app_class()
        
        # Open workflow if specified
        if args.workflow and os.path.exists(args.workflow):
            # Schedule opening the workflow after the app starts
            app.after(500, lambda: app.workflow.import_workflow(args.workflow))
            
            # Execute if requested
            if args.execute:
                app.after(1000, app.workflow.execute_workflow)
        
        # Start the main loop
        app.mainloop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
