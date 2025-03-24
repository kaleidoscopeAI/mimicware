#!/bin/bash
# AI Consciousness System Runner

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to check Python installation
check_python() {
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
    elif command -v python &>/dev/null; then
        PYTHON="python"
    else
        echo "Error: Python is not installed."
        exit 1
    fi
    
    # Check Python version
    VERSION=$($PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if (( $(echo "$VERSION < 3.7" | bc -l) )); then
        echo "Error: Python 3.7 or higher is required (found: $VERSION)"
        exit 1
    fi
}

# Function to set up virtual environment
setup_venv() {
    if [ ! -d "$SCRIPT_DIR/venv" ]; then
        echo "Setting up virtual environment..."
        $PYTHON -m venv "$SCRIPT_DIR/venv"
    fi
    
    # Activate virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Install required packages
    echo "Installing required packages..."
    pip install -q numpy networkx matplotlib scipy asyncio aiohttp torch pysimplegui
}

# Function to run the system
run_system() {
    echo "Starting AI Consciousness System..."
    
    # Run with provided arguments
    $PYTHON "$SCRIPT_DIR/SetupAndConfiguration.py" "$@"
    
    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: System execution failed."
        return 1
    fi
    
    return 0
}

# Main execution
check_python
setup_venv

# Check if the required scripts exist
if [ ! -f "$SCRIPT_DIR/SetupAndConfiguration.py" ]; then
    echo "Error: SetupAndConfiguration.py not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/AIConsciousnessTaskManager.py" ]; then
    echo "Error: AIConsciousnessTaskManager.py not found in $SCRIPT_DIR"
    exit 1
fi

# Run the system with all arguments passed to this script
run_system "$@"

# Deactivate virtual environment
deactivate
