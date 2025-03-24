#!/usr/bin/env python3
"""
Comprehensive test framework for System Builder
Includes unit tests, integration tests, and performance benchmarks
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import time
import logging
import multiprocessing
import subprocess
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin')))

# Import system modules
from system_builder import ScriptAnalyzer, RuntimeEnvironment, ExecutionEngine, ScriptNode, ScriptType
from graph_engine import WorkflowGraph, WorkflowEngine, WorkflowBuilder, Node, NodeType, EdgeType
import advanced_gui

# Configure logging
logging.basicConfig(level=logging.ERROR)

@contextmanager
def temp_directory():
    """Context manager for creating and cleaning up a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

@contextmanager
def create_test_script(content, extension='.py'):
    """Create a temporary script file for testing"""
    with tempfile.NamedTemporaryFile(suffix=extension, mode='w', delete=False) as f:
        f.write(content)
        script_path = f.name
    
    try:
        yield script_path
    finally:
        if os.path.exists(script_path):
            os.unlink(script_path)

class ScriptAnalyzerTests(unittest.TestCase):
    """Tests for the ScriptAnalyzer class"""
    
    def setUp(self):
        self.analyzer = ScriptAnalyzer()
    
    def test_detect_python_script(self):
        python_content = """#!/usr/bin/env python3
import os
import sys

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"""
        with create_test_script(python_content, '.py') as script_path:
            script_type = self.analyzer.detect_script_type(script_path)
            self.assertEqual(script_type, ScriptType.PYTHON)
    
    def test_detect_shell_script(self):
        shell_content = """#!/bin/bash
echo "Hello, world!"
"""
        with create_test_script(shell_content, '.sh') as script_path:
            script_type = self.analyzer.detect_script_type(script_path)
            self.assertEqual(script_type, ScriptType.SHELL)
    
    def test_analyze_python_script(self):
        python_content = """#!/usr/bin/env python3
import os
import sys
import numpy as np
from my_module import custom_function

def main():
    input_file = "data.txt"
    with open(input_file, 'r') as f:
        data = f.read()
    
    print("Processing data...")
    
if __name__ == "__main__":
    main()
"""
        with create_test_script(python_content, '.py') as script_path:
            node = self.analyzer.analyze_script(script_path)
            
            self.assertEqual(node.script_type, ScriptType.PYTHON)
            self.assertIn('my_module', node.metadata.get('imports', []))
            self.assertTrue(node.metadata.get('uses_files', False))
    
    def test_analyze_shell_script(self):
        shell_content = """#!/bin/bash
# This script processes data

echo "Processing data..."
bash ./other_script.sh
source ./env_setup.sh
"""
        with create_test_script(shell_content, '.sh') as script_path:
            node = self.analyzer.analyze_script(script_path)
            
            self.assertEqual(node.script_type, ScriptType.SHELL)
            script_calls = node.metadata.get('script_calls', [])
            self.assertIn('./other_script.sh', script_calls)
            self.assertIn('./env_setup.sh', script_calls)
    
    def test_build_dependency_graph(self):
        script1_content = """#!/usr/bin/env python3
# Main script
# depends-on: script2.py, script3.py

import os
import sys

def main():
    print("Main script")

if __name__ == "__main__":
    main()
"""

        script2_content = """#!/usr/bin/env python3
# Script 2
import os
import sys

def process_data():
    print("Processing data")

if __name__ == "__main__":
    process_data()
"""

        script3_content = """#!/usr/bin/env python3
# Script 3
# depends-on: script2.py

import os
import sys

def generate_report():
    print("Generating report")

if __name__ == "__main__":
    generate_report()
"""
        with temp_directory() as temp_dir:
            # Create scripts
            script1_path = os.path.join(temp_dir, 'script1.py')
            script2_path = os.path.join(temp_dir, 'script2.py')
            script3_path = os.path.join(temp_dir, 'script3.py')
            
            with open(script1_path, 'w') as f:
                f.write(script1_content)
            with open(script2_path, 'w') as f:
                f.write(script2_content)
            with open(script3_path, 'w') as f:
                f.write(script3_content)
            
            # Build dependency graph
            graph = self.analyzer.build_dependency_graph([script1_path, script2_path, script3_path])
            
            # Check if the graph has the correct number of nodes
            self.assertEqual(len(graph.nodes), 3)
            
            # Verify execution order
            execution_order = self.analyzer.resolve_execution_order()
            
            # script2 should be before script3 and script1
            # script3 should be before script1
            script_paths = [node.path for node in execution_order]
            
            script2_idx = script_paths.index(script2_path)
            script3_idx = script_paths.index(script3_path)
            script1_idx = script_paths.index(script1_path)
            
            self.assertLess(script2_idx, script3_idx)
            self.assertLess(script2_idx, script1_idx)
            self.assertLess(script3_idx, script1_idx)
    
    def test_circular_dependency_detection(self):
        script1_content = """#!/usr/bin/env python3
# Script 1
# depends-on: script2.py

import os
import sys

def main():
    print("Script 1")

if __name__ == "__main__":
    main()
"""

        script2_content = """#!/usr/bin/env python3
# Script 2
# depends-on: script3.py

import os
import sys

def process_data():
    print("Script 2")

if __name__ == "__main__":
    process_data()
"""

        script3_content = """#!/usr/bin/env python3
# Script 3
# depends-on: script1.py

import os
import sys

def generate_report():
    print("Script 3")

if __name__ == "__main__":
    generate_report()
"""
        with temp_directory() as temp_dir:
            # Create scripts
            script1_path = os.path.join(temp_dir, 'script1.py')
            script2_path = os.path.join(temp_dir, 'script2.py')
            script3_path = os.path.join(temp_dir, 'script3.py')
            
            with open(script1_path, 'w') as f:
                f.write(script1_content)
            with open(script2_path, 'w') as f:
                f.write(script2_content)
            with open(script3_path, 'w') as f:
                f.write(script3_content)
            
            # Build dependency graph
            graph = self.analyzer.build_dependency_graph([script1_path, script2_path, script3_path])
            
            # Check for circular dependency
            with self.assertRaises(ValueError):
                execution_order = self.analyzer.resolve_execution_order()

class RuntimeEnvironmentTests(unittest.TestCase):
    """Tests for the RuntimeEnvironment class"""
    
    def test_create_environment(self):
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            
            # Check that directories were created
            dirs = ['bin', 'lib', 'include', 'scripts']
            for dirname in dirs:
                dir_path = os.path.join(temp_dir, dirname)
                self.assertTrue(os.path.exists(dir_path))
            
            # Check environment variables
            env_vars = runtime.get_environment()
            self.assertIn('PATH', env_vars)
            self.assertIn('SYSBUILDER_ENV', env_vars)
            self.assertEqual(env_vars['SYSBUILDER_ENV'], temp_dir)
    
    def test_prepare_script(self):
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            
            script_content = "#!/usr/bin/env python3\nprint('Hello, world!')\n"
            
            with create_test_script(script_content, '.py') as script_path:
                script_name = os.path.basename(script_path)
                
                # Create script node
                node = ScriptNode(
                    id="test_script",
                    path=script_path,
                    script_type=ScriptType.PYTHON,
                    content=script_content
                )
                
                # Prepare script
                prepared_path = runtime.prepare_script(node)
                
                # Check that script was copied to the correct location
                expected_path = os.path.join(temp_dir, 'scripts', script_name)
                self.assertEqual(prepared_path, expected_path)
                self.assertTrue(os.path.exists(prepared_path))
                
                # Check script content
                with open(prepared_path, 'r') as f:
                    content = f.read()
                    self.assertEqual(content, script_content)
    
    @unittest.skipIf(sys.platform == "win32", "Skip pip test on Windows")
    @patch('subprocess.check_call')
    def test_install_requirements(self, mock_check_call):
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            
            # Create script nodes with dependencies
            node1 = ScriptNode(
                id="script1",
                path="script1.py",
                script_type=ScriptType.PYTHON,
                metadata={'imports': ['numpy', 'pandas']}
            )
            
            node2 = ScriptNode(
                id="script2",
                path="script2.py",
                script_type=ScriptType.PYTHON,
                metadata={'imports': ['matplotlib', 'scipy']}
            )
            
            # Install requirements
            runtime.install_requirements([node1, node2])
            
            # Check that pip was called with the correct arguments
            mock_check_call.assert_called_once()
            args = mock_check_call.call_args[0][0]
            
            self.assertEqual(args[0:3], [sys.executable, '-m', 'pip'])
            self.assertEqual(args[3], 'install')
            self.assertEqual(args[5], os.path.join(temp_dir, 'lib', 'python'))
            
            # Check that all packages were included
            for pkg in ['numpy', 'pandas', 'matplotlib', 'scipy']:
                self.assertIn(pkg, args)

class ExecutionEngineTests(unittest.TestCase):
    """Tests for the ExecutionEngine class"""
    
    def test_execute_python_script(self):
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            engine = ExecutionEngine(runtime)
            
            script_content = """#!/usr/bin/env python3
import sys
import os

def main():
    print("Hello from Python!")
    print(f"Arguments: {sys.argv[1:]}")
    with open(os.path.join(os.environ['SYSBUILDER_ENV'], 'output.txt'), 'w') as f:
        f.write("Script executed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
            with create_test_script(script_content, '.py') as script_path:
                # Create script node
                node = ScriptNode(
                    id="test_script",
                    path=script_path,
                    script_type=ScriptType.PYTHON,
                    content=script_content
                )
                
                # Execute script
                result = engine.execute_script(node)
                
                # Check execution result
                self.assertTrue(result.success)
                self.assertEqual(result.return_code, 0)
                self.assertIn("Hello from Python!", result.stdout)
                
                # Check output file
                output_path = os.path.join(temp_dir, 'output.txt')
                self.assertTrue(os.path.exists(output_path))
                
                with open(output_path, 'r') as f:
                    content = f.read()
                    self.assertEqual(content, "Script executed successfully")
    
    def test_execute_shell_script(self):
        if sys.platform == "win32":
            self.skipTest("Skipping shell script test on Windows")
        
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            engine = ExecutionEngine(runtime)
            
            script_content = """#!/bin/bash
echo "Hello from Bash!"
echo "Environment: $SYSBUILDER_ENV"
echo "Script executed successfully" > "$SYSBUILDER_ENV/output.txt"
exit 0
"""
            with create_test_script(script_content, '.sh') as script_path:
                # Make script executable
                os.chmod(script_path, 0o755)
                
                # Create script node
                node = ScriptNode(
                    id="test_script",
                    path=script_path,
                    script_type=ScriptType.SHELL,
                    content=script_content
                )
                
                # Execute script
                result = engine.execute_script(node)
                
                # Check execution result
                self.assertTrue(result.success)
                self.assertEqual(result.return_code, 0)
                self.assertIn("Hello from Bash!", result.stdout)
                
                # Check output file
                output_path = os.path.join(temp_dir, 'output.txt')
                self.assertTrue(os.path.exists(output_path))
                
                with open(output_path, 'r') as f:
                    content = f.read()
                    self.assertEqual(content, "Script executed successfully")
    
    def test_execute_all(self):
        with temp_directory() as temp_dir:
            runtime = RuntimeEnvironment(temp_dir)
            engine = ExecutionEngine(runtime)
            
            # Create script nodes
            nodes = []
            results = []
            
            script1_content = """#!/usr/bin/env python3
import os
print("Script 1 executed")
with open(os.path.join(os.environ['SYSBUILDER_ENV'], 'script1.txt'), 'w') as f:
    f.write("Script 1")
"""
            
            script2_content = """#!/usr/bin/env python3
import os
print("Script 2 executed")
with open(os.path.join(os.environ['SYSBUILDER_ENV'], 'script2.txt'), 'w') as f:
    f.write("Script 2")
"""
            
            script3_content = """#!/usr/bin/env python3
import os
print("Script 3 executed")
with open(os.path.join(os.environ['SYSBUILDER_ENV'], 'script3.txt'), 'w') as f:
    f.write("Script 3")
"""
            
            with create_test_script(script1_content, '.py') as script1_path, \
                 create_test_script(script2_content, '.py') as script2_path, \
                 create_test_script(script3_content, '.py') as script3_path:
                
                nodes.append(ScriptNode(
                    id="script1",
                    path=script1_path,
                    script_type=ScriptType.PYTHON,
                    content=script1_content
                ))
                
                nodes.append(ScriptNode(
                    id="script2",
                    path=script2_path,
                    script_type=ScriptType.PYTHON,
                    content=script2_content
                ))
                
                nodes.append(ScriptNode(
                    id="script3",
                    path=script3_path,
                    script_type=ScriptType.PYTHON,
                    content=script3_content
                ))
                
                # Execute all scripts
                def callback(script, result):
                    results.append((script.id, result.success))
                
                exec_results = engine.execute_all(nodes, callback)
                
                # Check execution results
                self.assertEqual(len(exec_results), 3)
                self.assertTrue(all(r.success for r in exec_results.values()))
                
                # Check callback results
                self.assertEqual(len(results), 3)
                self.assertTrue(all(success for _, success in results))
                
                # Check output files
                for i in range(1, 4):
                    output_path = os.path.join(temp_dir, f'script{i}.txt')
                    self.assertTrue(os.path.exists(output_path))
                    
                    with open(output_path, 'r') as f:
                        content = f.read()
                        self.assertEqual(content, f"Script {i}")

class GraphEngineTests(unittest.TestCase):
    """Tests for the Workflow Graph Engine"""
    
    def test_workflow_graph(self):
        # Create a workflow graph
        graph = WorkflowGraph()
        
        # Create nodes
        node1 = Node(
            metadata=graph_engine.NodeMetadata(
                id="1",
                name="Node 1",
                type=NodeType.SCRIPT,
                properties={'path': 'script1.py'}
            )
        )
        
        node2 = Node(
            metadata=graph_engine.NodeMetadata(
                id="2",
                name="Node 2",
                type=NodeType.SCRIPT,
                properties={'path': 'script2.py'}
            )
        )
        
        node3 = Node(
            metadata=graph_engine.NodeMetadata(
                id="3",
                name="Node 3",
                type=NodeType.SCRIPT,
                properties={'path': 'script3.py'}
            )
        )
        
        # Add nodes to graph
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Add edges
        graph.add_edge("1", "2", EdgeType.CONTROL)
        graph.add_edge("2", "3", EdgeType.CONTROL)
        
        # Check graph properties
        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.get_dependencies("3")), 1)
        self.assertEqual(len(graph.get_dependents("1")), 1)
        
        # Check topological sort
        topo_order = graph.topological_sort()
        self.assertEqual(topo_order, ["1", "2", "3"])
        
        # Calculate critical path
        path, length = graph.calculate_critical_path()
        self.assertEqual(path, ["1", "2", "3"])
        
        # Convert to dict and back
        graph_dict = graph.to_dict()
        new_graph = WorkflowGraph.from_dict(graph_dict)
        
        self.assertEqual(len(new_graph.nodes), 3)
        self.assertEqual(new_graph.topological_sort(), ["1", "2", "3"])
    
    def test_workflow_builder(self):
        # Create a workflow builder
        builder = WorkflowBuilder()
        
        # Add script nodes
        script1 = builder.add_script("Script 1", "/path/to/script1.py")
        script2 = builder.add_script("Script 2", "/path/to/script2.py")
        script3 = builder.add_script("Script 3", "/path/to/script3.py")
        
        # Add a condition node
        condition = builder.add_condition("Check Result", "inputs['Script 1'] == 0")
        
        # Connect nodes
        builder.connect(script1, condition)
        builder.connect(condition, script2)
        builder.connect(script1, script3)
        
        # Build the graph
        graph = builder.build()
        
        # Check graph properties
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.get_dependencies(script2)), 1)
        self.assertEqual(len(graph.get_dependents(script1)), 2)
        
        # Check for cycles
        self.assertFalse(graph.is_cyclical())
    
    @patch('threading.Thread')
    def test_workflow_engine(self, mock_thread):
        # Create a workflow graph
        graph = WorkflowGraph()
        
        # Create nodes
        node1 = Node(
            metadata=graph_engine.NodeMetadata(
                id="1",
                name="Node 1",
                type=NodeType.SCRIPT,
                properties={'path': 'script1.py'}
            )
        )
        
        node2 = Node(
            metadata=graph_engine.NodeMetadata(
                id="2",
                name="Node 2",
                type=NodeType.SCRIPT,
                properties={'path': 'script2.py'}
            )
        )
        
        # Add nodes to graph
        graph.add_node(node1)
        graph.add_node(node2)
        
        # Add edges
        graph.add_edge("1", "2", EdgeType.CONTROL)
        
        # Create engine
        engine = WorkflowEngine()
        engine.set_graph(graph)
        
        # Set context
        context = {'data': {'value': 42}}
        engine.set_context(context)
        
        # Mock the thread behavior
        mock_thread.return_value.daemon = False
        mock_thread.return_value.start.return_value = None
        
        # Start engine
        engine.start()
        
        # Check that worker threads were created
        self.assertEqual(mock_thread.call_count, engine.num_workers + 1)  # Workers + result thread
        
        # Check engine state
        self.assertTrue(engine.is_running())
        
        # Stop engine
        engine.stop()
        
        # Check engine state
        self.assertFalse(engine.is_running())

class IntegrationTests(unittest.TestCase):
    """Integration tests for System Builder components"""
    
    def test_script_analysis_and_execution(self):
        with temp_directory() as temp_dir:
            # Create test scripts
            script1_path = os.path.join(temp_dir, 'script1.py')
            script2_path = os.path.join(temp_dir, 'script2.py')
            output_path = os.path.join(temp_dir, 'output.txt')
            
            with open(script1_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
# Script 1
# depends-on: script2.py

import os
import sys

def main():
    print("Running Script 1")
    
    # Read intermediate data from script2
    with open(os.path.join(os.path.dirname(__file__), 'intermediate.txt'), 'r') as f:
        data = f.read()
    
    # Write final output
    with open(os.path.join(os.path.dirname(__file__), 'output.txt'), 'w') as f:
        f.write(f"Processed: {data}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
            
            with open(script2_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
# Script 2

import os
import sys

def main():
    print("Running Script 2")
    
    # Generate intermediate data
    with open(os.path.join(os.path.dirname(__file__), 'intermediate.txt'), 'w') as f:
        f.write("Data generated by Script 2")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
""")
            
            # Create analyzer and runtime
            analyzer = ScriptAnalyzer()
            runtime = RuntimeEnvironment(temp_dir)
            engine = ExecutionEngine(runtime)
            
            # Analyze scripts
            graph = analyzer.build_dependency_graph([script1_path, script2_path])
            
            # Resolve execution order
            execution_order = analyzer.resolve_execution_order()
            
            # Check execution order
            self.assertEqual(len(execution_order), 2)
            self.assertEqual(execution_order[0].path, script2_path)
            self.assertEqual(execution_order[1].path, script1_path)
            
            # Execute scripts
            results = engine.execute_all(execution_order)
            
            # Check results
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.success for r in results.values()))
            
            # Check output file
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertEqual(content, "Processed: Data generated by Script 2")
    
    def test_workflow_graph_execution(self):
        # Create a workflow builder
        builder = WorkflowBuilder()
        
        with temp_directory() as temp_dir:
            # Create test scripts
            script1_path = os.path.join(temp_dir, 'script1.py')
            script2_path = os.path.join(temp_dir, 'script2.py')
            script3_path = os.path.join(temp_dir, 'script3.py')
            
            with open(script1_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
# Generate data
import os
import json

data = {"value": 42, "text": "Hello, world!"}

with open(os.path.join(os.path.dirname(__file__), 'data.json'), 'w') as f:
    json.dump(data, f)

print("Data generated")
""")
            
            with open(script2_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
# Process data
import os
import json

with open(os.path.join(os.path.dirname(__file__), 'data.json'), 'r') as f:
    data = json.load(f)

data["processed"] = True
data["value"] *= 2

with open(os.path.join(os.path.dirname(__file__), 'processed_data.json'), 'w') as f:
    json.dump(data, f)

print("Data processed")
""")
            
            with open(script3_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
# Generate report
import os
import json

with open(os.path.join(os.path.dirname(__file__), 'processed_data.json'), 'r') as f:
    data = json.load(f)

report = f"Report: Value={data['value']}, Text='{data['text']}', Processed={data['processed']}"

with open(os.path.join(os.path.dirname(__file__), 'report.txt'), 'w') as f:
    f.write(report)

print("Report generated")
""")
            
            # Add script nodes
            node1 = builder.add_script("Generate Data", script1_path)
            node2 = builder.add_script("Process Data", script2_path)
            node3 = builder.add_script("Generate Report", script3_path)
            
            # Connect nodes
            builder.connect(node1, node2)
            builder.connect(node2, node3)
            
            # Build graph
            graph = builder.build()
            
            # Create workflow engine
            engine = WorkflowEngine()
            engine.set_graph(graph)
            engine.set_context({'working_dir': temp_dir})
            
            # Execute workflow
            engine.start()
            engine.wait(timeout=10.0)
            
            # Stop engine
            engine.stop()
            
            # Check results
            self.assertEqual(len(engine.get_results()), 3)
            
            # Check output files
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'data.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'processed_data.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'report.txt')))
            
            # Check report content
            with open(os.path.join(temp_dir, 'report.txt'), 'r') as f:
                content = f.read()
                self.assertEqual(content, "Report: Value=84, Text='Hello, world!', Processed=True")

class PerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for System Builder"""
    
    def test_graph_algorithm_performance(self):
        """Benchmark graph algorithm performance"""
        # Generate a large random DAG
        num_nodes = 1000
        edge_probability = 0.01
        
        # Create random adjacency matrix (upper triangular to ensure DAG)
        np.random.seed(42)  # For reproducibility
        adj_matrix = np.random.random((num_nodes, num_nodes)) < edge_probability
        adj_matrix = np.triu(adj_matrix, k=1)  # Upper triangular
        
        # Create networkx graph
        G = nx.DiGraph()
        for i in range(num_nodes):
            G.add_node(i)
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj_matrix[i, j]:
                    G.add_edge(i, j)
        
        # Benchmark topological sort
        start_time = time.time()
        topo_order = list(nx.topological_sort(G))
        nx_time = time.time() - start_time
        
        # Create graph using our implementation
        graph = WorkflowGraph()
        
        for i in range(num_nodes):
            node = Node(
                metadata=graph_engine.NodeMetadata(
                    id=str(i),
                    name=f"Node {i}",
                    type=NodeType.COMPUTATION,
                    properties={}
                )
            )
            graph.add_node(node)
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if adj_matrix[i, j]:
                    graph.add_edge(str(i), str(j), EdgeType.CONTROL)
        
        # Benchmark our topological sort
        start_time = time.time()
        our_topo_order = graph.topological_sort()
        our_time = time.time() - start_time
        
        # Print performance comparison
        print(f"\nTopological sort performance:")
        print(f"NetworkX: {nx_time:.6f} seconds")
        print(f"Our implementation: {our_time:.6f} seconds")
        print(f"Speedup: {nx_time / our_time:.2f}x")
        
        # Calculate critical path performance
        start_time = time.time()
        path, length = graph.calculate_critical_path()
        critical_path_time = time.time() - start_time
        
        print(f"Critical path calculation: {critical_path_time:.6f} seconds")
        print(f"Critical path length: {length}")
        
        # Verify correctness
        self.assertEqual(len(topo_order), num_nodes)
        self.assertEqual(len(our_topo_order), num_nodes)
    
    @unittest.skipIf(True, "Skip long-running test by default")
    def test_script_analysis_scaling(self):
        """Test performance scaling of script analysis"""
        with temp_directory() as temp_dir:
            # Generate a large number of scripts with dependencies
            num_scripts = 100
            scripts = []
            
            for i in range(num_scripts):
                script_path = os.path.join(temp_dir, f'script{i}.py')
                
                # Determine dependencies
                dependencies = []
                if i > 0:
                    # Add 1-3 dependencies to previous scripts
                    for _ in range(min(3, i)):
                        dep_idx = np.random.randint(0, i)
                        dependencies.append(f'script{dep_idx}.py')
                
                # Create script content
                content = f"""#!/usr/bin/env python3
# Script {i}
"""
                if dependencies:
                    content += f"# depends-on: {', '.join(dependencies)}\n"
                
                content += f"""
import os
import sys

def main():
    print("Running Script {i}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
                
                with open(script_path, 'w') as f:
                    f.write(content)
                
                scripts.append(script_path)
            
            # Measure analysis time
            analyzer = ScriptAnalyzer()
            
            start_time = time.time()
            graph = analyzer.build_dependency_graph(scripts)
            analysis_time = time.time() - start_time
            
            start_time = time.time()
            execution_order = analyzer.resolve_execution_order()
            resolution_time = time.time() - start_time
            
            print(f"\nScript analysis scaling ({num_scripts} scripts):")
            print(f"Build dependency graph: {analysis_time:.6f} seconds")
            print(f"Resolve execution order: {resolution_time:.6f} seconds")
            print(f"Total analysis time: {analysis_time + resolution_time:.6f} seconds")
            print(f"Average time per script: {(analysis_time + resolution_time) / num_scripts:.6f} seconds")

def run_tests():
    """Run all tests"""
    unittest.main()

if __name__ == "__main__":
    run_tests()
