# System Builder

A comprehensive drag-and-drop system for building and executing script-based workflows.

## Overview

System Builder provides a visual interface for creating, configuring, and executing workflows composed of scripts and other computational components. It allows users to:

- Drag and drop scripts into a workflow canvas
- Connect scripts to define execution dependencies
- Configure script parameters and execution environment
- Analyze script dependencies automatically
- Execute workflows with proper dependency resolution
- Monitor execution progress and view results

The system is designed to be extensible, supporting various script types including Python, Bash, JavaScript, C/C++, and Assembly.

## Features

- **Visual Workflow Editor**: Intuitive drag-and-drop interface for building workflows
- **Script Integration**: Seamlessly integrate scripts written in Python, Bash, C/C++, and more
- **Automatic Dependency Resolution**: Analyzes scripts to determine execution order
- **Parallel Execution**: Execute independent tasks in parallel
- **Template System**: Create scripts from templates with customizable variables
- **Advanced Graph Engine**: Optimized graph-based workflow execution using directed graph algorithms
- **C Language Optimization**: Critical components implemented in C for maximum performance
- **Real-time Monitoring**: Monitor workflow execution progress

## Installation

### Quick Install

For a quick installation with default options:

```bash
# Clone the repository
git clone https://github.com/yourusername/system-builder.git
cd system-builder

# Install using the installation script
python install.py
```

### Advanced Installation

For more control over the installation process:

```bash
python install.py --install-dir /path/to/install --feature-sets graph_visualization data_processing
```

Available options:
- `--install-dir`: Specify installation directory (default is platform-dependent)
- `--no-venv`: Skip creating a virtual environment
- `--feature-sets`: Additional feature sets to install (space-separated list)
  - `graph_visualization`: Tools for visualizing workflows
  - `execution_optimization`: Performance optimization libraries
  - `data_processing`: Data processing capabilities
  - `web_integration`: Web API integration
  - `machine_learning`: Machine learning components
- `--no-c-modules`: Skip building C modules
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)

### Manual Installation

If you prefer a manual installation:

1. Ensure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install networkx numpy Pillow matplotlib
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/system-builder.git
   ```
4. Build the C optimizer (optional but recommended):
   ```bash
   gcc script-analyzer.c -shared -fPIC -o optimizer.so
   ```
5. Run the application:
   ```bash
   python system-builder.py
   ```

## Usage

### Starting the Application

After installation, you can start System Builder using:

```bash
# On Linux/macOS
/path/to/install/bin/systembuilder

# On Windows
C:\path\to\install\bin\systembuilder.bat
# or
C:\path\to\install\bin\SystemBuilder.exe
```

### Creating a Workflow

1. Open System Builder
2. Use the toolbar to add script nodes to the canvas
3. Connect nodes by clicking and dragging between ports
4. Configure node properties using the properties panel
5. Save your workflow using File > Save

### Adding Scripts

There are several ways to add scripts to your workflow:

- Drag and drop script files directly onto the canvas
- Use the "Add Script" button to select existing scripts
- Create new scripts from templates using the "Create Script" button

### Executing a Workflow

1. Click the "Execute" button in the toolbar
2. Monitor execution progress in the execution dialog
3. View execution results in the log panel

## Architecture

System Builder is composed of several core components:

- **System Builder Core**: Main application logic and user interface
- **Script Analyzer**: Analyzes scripts to determine dependencies
- **Graph Engine**: Manages workflow execution using directed graph algorithms
- **Advanced GUI**: Provides the drag-and-drop interface
- **Runtime Environment**: Configures the execution environment for scripts

## Examples

The installation includes example workflows to help you get started:

- `simple_workflow.json`: A basic data processing workflow
- More examples in the `workflows` directory

## Extending System Builder

You can extend System Builder with additional functionality:

- **Custom Node Types**: Implement new node types by extending the base Node class
- **Script Templates**: Create new script templates in the templates directory
- **Feature Modules**: Develop feature modules that integrate with the core system

## Requirements

- Python 3.7 or higher
- Operating Systems: Windows, macOS, Linux
- Dependencies:
  - networkx
  - numpy
  - Pillow
  - matplotlib
  - C compiler (for optional C module optimization)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, feature requests, or questions:
- Open an issue on GitHub
- Contact the maintainers at support@systembuilder.example.com

## Acknowledgments

- NetworkX library for graph algorithms
- Tkinter for the graphical user interface
- The open-source community for inspiration and tools

---

# Quick Start Guide

## Creating Your First Workflow

This guide will walk you through creating a simple data processing workflow.

### Step 1: Install System Builder

Follow the installation instructions above to install System Builder.

### Step 2: Launch System Builder

Start the application using the appropriate command for your operating system.

### Step 3: Create a Workflow

1. Click "New Workflow" from the File menu or toolbar
2. Add a script node by right-clicking on the canvas and selecting "Add Script Node"
3. Name the node "Read Data" and select a Python script that reads data
4. Add a second script node and name it "Process Data"
5. Add a third script node and name it "Generate Report"

### Step 4: Connect the Nodes

1. Click on the output port of the "Read Data" node
2. Drag to the input port of the "Process Data" node to create a connection
3. Repeat to connect "Process Data" to "Generate Report"

### Step 5: Configure Nodes

1. Double-click on each node to edit its properties
2. Set the script path, arguments, and environment variables as needed

### Step 6: Save the Workflow

1. Click "Save" from the File menu
2. Choose a location and name for your workflow file

### Step 7: Execute the Workflow

1. Click the "Execute" button in the toolbar
2. Monitor the execution progress
3. Review the results when execution completes

Congratulations! You've created and executed your first workflow with System Builder.
