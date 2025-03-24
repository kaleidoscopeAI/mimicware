#!/usr/bin/env python3
# SetupAndConfiguration.py
# System initialization and configuration for AI Consciousness

import os
import sys
import json
import asyncio
import argparse
import logging
import subprocess
import importlib.util
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AIConsciousSetup")

# Import our task manager
from AIConsciousnessTaskManager import AIConsciousnessTaskManager, ProcessingPhase, create_task_manager

class SystemConfiguration:
    """Configuration and setup for AI Consciousness system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize system configuration"""
        self.config_path = config_path or os.path.join(os.getcwd(), "config.json")
        self.config = self._load_config()
        self.work_dir = self.config.get("work_dir", os.path.join(os.getcwd(), "ai_consciousness_workdir"))
        
        # Create work directory
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Components tracking
        self.components_status = {}
        self.task_manager = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading config file: {str(e)}. Using defaults.")
        
        # Default configuration
        default_config = {
            "work_dir": os.path.join(os.getcwd(), "ai_consciousness_workdir"),
            "components": {
                "task_manager": {"enabled": True},
                "diagram_interpreter": {"enabled": True},
                "neural_quantum_bridge": {"enabled": True},
                "runtime_optimizer": {"enabled": True},
                "pattern_detector": {"enabled": True}
            },
            "processing": {
                "parallel_tasks": 4,
                "consciousness_simulation_enabled": True,
                "consciousness_threshold": 0.8,
                "default_target_language": None
            },
            "logging": {
                "level": "INFO",
                "file": "ai_consciousness.log"
            }
        }
        
        # Save default config
        try:
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.warning(f"Error saving default config: {str(e)}")
        
        return default_config
    
    async def setup_system(self) -> bool:
        """Set up the AI Consciousness system components"""
        logger.info("Setting up AI Consciousness system...")
        
        # Set up task manager first
        try:
            self.task_manager = create_task_manager(self.work_dir)
            self.components_status["task_manager"] = "initialized"
            logger.info("Task manager initialized")
            
            # Start task manager process loop in background
            asyncio.create_task(self.task_manager.run())
            
            # Check for dependencies
            missing_deps = self._check_dependencies()
            if missing_deps:
                logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                self._install_dependencies(missing_deps)
            
            # Verify core components
            await self._verify_components()
            
            # Setup is complete
            logger.info("System setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up system: {str(e)}")
            return False
    
    def _check_dependencies(self) -> List[str]:
        """Check for required Python dependencies"""
        required_packages = [
            "numpy", "networkx", "matplotlib", "scipy", 
            "asyncio", "aiohttp", "torch"
        ]
        
        missing = []
        for package in required_packages:
            if not self._is_package_installed(package):
                missing.append(package)
        
        return missing
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed"""
        return importlib.util.find_spec(package_name) is not None
    
    def _install_dependencies(self, packages: List[str]) -> None:
        """Install missing dependencies"""
        if not packages:
            return
        
        logger.info(f"Installing dependencies: {', '.join(packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {str(e)}")
    
    async def _verify_components(self) -> None:
        """Verify that all required components are available"""
        components_config = self.config.get("components", {})
        
        # Check each component
        for component_name, config in components_config.items():
            if not config.get("enabled", True):
                logger.info(f"Component {component_name} is disabled in config")
                self.components_status[component_name] = "disabled"
                continue
            
            # Verify component availability
            if component_name == "task_manager":
                # Already initialized
                continue
                
            elif component_name == "diagram_interpreter":
                # Check for diagram interpreter
                try:
                    from AdvancedDiagramInterpreter import AdvancedDiagramInterpreter
                    self.components_status[component_name] = "available"
                except ImportError:
                    logger.warning(f"Component {component_name} not available")
                    self.components_status[component_name] = "unavailable"
                    
            elif component_name == "neural_quantum_bridge":
                # Check for neural quantum bridge
                try:
                    from neural_quantum_bridge import NeuralQuantumBridge
                    self.components_status[component_name] = "available"
                except ImportError:
                    logger.warning(f"Component {component_name} not available")
                    self.components_status[component_name] = "unavailable"
                    
            elif component_name == "runtime_optimizer":
                # Check for runtime optimizer
                try:
                    from RuntimeOptimizationCircuit import RuntimeOptimizationCircuit
                    self.components_status[component_name] = "available"
                except ImportError:
                    logger.warning(f"Component {component_name} not available")
                    self.components_status[component_name] = "unavailable"
                    
            elif component_name == "pattern_detector":
                # Check for pattern detector
                try:
                    from EmergentPatternDetector import EmergentPatternDetector
                    self.components_status[component_name] = "available"
                except ImportError:
                    logger.warning(f"Component {component_name} not available")
                    self.components_status[component_name] = "unavailable"
            
            else:
                logger.warning(f"Unknown component {component_name}")
                self.components_status[component_name] = "unknown"
        
        # Log component status
        for component, status in self.components_status.items():
            logger.info(f"Component {component}: {status}")
    
    async def initialize_project(self, 
                               input_dir: str, 
                               target_language: Optional[str] = None) -> bool:
        """Initialize a new AI Consciousness project"""
        if not self.task_manager:
            logger.error("Task manager not initialized")
            return False
        
        logger.info(f"Initializing project with input directory: {input_dir}")
        
        # Ensure input directory exists
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        # Validate target language if specified
        if target_language:
            valid_languages = ["python", "javascript", "typescript", "java", "c", "cpp", "csharp", "go", "rust"]
            if target_language.lower() not in valid_languages:
                logger.warning(f"Invalid target language: {target_language}. Using default.")
                target_language = self.config.get("processing", {}).get("default_target_language")
        
        # Find code files in input directory
        code_files = self._find_code_files(input_dir)
        if not code_files:
            logger.error(f"No code files found in input directory: {input_dir}")
            return False
        
        logger.info(f"Found {len(code_files)} code files")
        
        # Initialize the system
        project_dir = os.path.join(self.work_dir, "projects", Path(input_dir).name)
        await self.task_manager.schedule_task(
            task_id="init",
            phase=ProcessingPhase.INITIALIZATION,
            task_func=self.task_manager._handle_initialization,
            project_dir=project_dir,
            target_language=target_language
        )
        
        # Schedule decomposition task
        await self.task_manager.schedule_task(
            task_id="decomposition",
            phase=ProcessingPhase.DECOMPOSITION,
            task_func=self.task_manager._handle_decomposition,
            file_paths=code_files,
            dependencies=["init"]
        )
        
        return True
    
    def _find_code_files(self, directory: str) -> List[str]:
        """Find all code files in the given directory"""
        code_extensions = [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', 
            '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs'
        ]
        
        code_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    code_files.append(os.path.join(root, file))
        
        return code_files
    
    async def monitor_progress(self) -> None:
        """Monitor the progress of the current project"""
        if not self.task_manager:
            logger.error("Task manager not initialized")
            return
        
        while True:
            # Get current phase
            current_phase = self.task_manager.state.current_phase
            
            # Count tasks in each state
            task_counts = {
                "pending": 0,
                "running": 0,
                "completed": 0,
                "failed": 0
            }
            
            for task in self.task_manager.state.tasks.values():
                task_counts[task.status] += 1
            
            # Calculate overall progress
            total_tasks = sum(task_counts.values())
            if total_tasks > 0:
                progress = (task_counts["completed"] + task_counts["failed"]) / total_tasks
                
                # Print progress
                logger.info(f"Phase: {current_phase.name}, Progress: {progress:.1%}, "
                          f"Tasks: {task_counts['completed']}/{total_tasks} completed, "
                          f"{task_counts['running']} running, "
                          f"{task_counts['failed']} failed")
                
                # Check if all tasks are done
                if task_counts["pending"] == 0 and task_counts["running"] == 0:
                    if current_phase == ProcessingPhase.CONSCIOUSNESS:
                        logger.info("Project processing completed")
                        break
            
            # Wait before checking again
            await asyncio.sleep(5)
    
    def get_component_status(self) -> Dict[str, str]:
        """Get the status of all components"""
        return self.components_status
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the overall system status"""
        if not self.task_manager:
            return {"status": "not_initialized"}
        
        return {
            "current_phase": self.task_manager.state.current_phase.name,
            "phases_completed": [p.name for p in self.task_manager.state.phases_completed],
            "quantum_coherence": self.task_manager.state.quantum_coherence,
            "components": self.components_status,
            "tasks": {
                "total": len(self.task_manager.state.tasks),
                "pending": sum(1 for t in self.task_manager.state.tasks.values() if t.status == "pending"),
                "running": sum(1 for t in self.task_manager.state.tasks.values() if t.status == "running"),
                "completed": sum(1 for t in self.task_manager.state.tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self.task_manager.state.tasks.values() if t.status == "failed")
            }
        }

async def run_gui_interface(system_config: SystemConfiguration):
    """Run graphical user interface for the system"""
    # This would launch a GUI interface, but for now we'll just use console
    try:
        import PySimpleGUI as sg
        logger.info("GUI interface is not implemented yet")
        # GUI implementation would go here
    except ImportError:
        logger.warning("PySimpleGUI not installed, using console interface")
    
    # Fall back to console interface
    await run_console_interface(system_config)

async def run_console_interface(system_config: SystemConfiguration):
    """Run command-line interface for the system"""
    print("=" * 50)
    print("AI Consciousness System")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Initialize project")
        print("2. Show system status")
        print("3. Monitor progress")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == "1":
            input_dir = input("Enter input directory: ")
            target_lang = input("Enter target language (or leave empty): ")
            target_lang = target_lang if target_lang else None
            
            success = await system_config.initialize_project(input_dir, target_lang)
            if success:
                print(f"Project initialized. Files will be processed from {input_dir}")
                
                # Start progress monitoring
                monitor_task = asyncio.create_task(system_config.monitor_progress())
                await monitor_task
            
        elif choice == "2":
            status = system_config.get_system_status()
            print("\nSystem Status:")
            print(f"Current Phase: {status.get('current_phase', 'N/A')}")
            print(f"Phases Completed: {', '.join(status.get('phases_completed', []))}")
            print(f"Quantum Coherence: {status.get('quantum_coherence', 0):.2f}")
            
            print("\nComponents Status:")
            for component, status in system_config.get_component_status().items():
                print(f"- {component}: {status}")
            
            print("\nTasks Status:")
            tasks = status.get("tasks", {})
            print(f"Total: {tasks.get('total', 0)}")
            print(f"Pending: {tasks.get('pending', 0)}")
            print(f"Running: {tasks.get('running', 0)}")
            print(f"Completed: {tasks.get('completed', 0)}")
            print(f"Failed: {tasks.get('failed', 0)}")
            
        elif choice == "3":
            print("Monitoring progress... (Press Ctrl+C to stop)")
            try:
                await system_config.monitor_progress()
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
                
        elif choice == "4":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AI Consciousness System Setup")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--input", "-i", help="Input directory containing code files")
    parser.add_argument("--target", "-t", help="Target language for code transformation")
    parser.add_argument("--gui", "-g", action="store_true", help="Launch graphical interface")
    
    args = parser.parse_args()
    
    # Create system configuration
    system_config = SystemConfiguration(args.config)
    
    # Set up system
    setup_success = await system_config.setup_system()
    if not setup_success:
        logger.error("System setup failed")
        return
    
    # If input directory specified, initialize project
    if args.input:
        await system_config.initialize_project(args.input, args.target)
        
        # Monitor progress
        await system_config.monitor_progress()
    else:
        # Run interface
        if args.gui:
            await run_gui_interface(system_config)
        else:
            await run_console_interface(system_config)

if __name__ == "__main__":
    asyncio.run(main())
