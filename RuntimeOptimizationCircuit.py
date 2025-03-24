import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
import math
import time
import ast
import re
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RuntimeOptimizationCircuit")

class OptimizationType(Enum):
    """Types of optimizations to perform"""
    MEMORY = auto()
    CPU = auto()
    IO = auto()
    NETWORK = auto()
    DATABASE = auto()
    MULTITHREADING = auto()
    CACHING = auto()
    ALGORITHM = auto()

@dataclass
class OptimizationContext:
    """Runtime optimization context"""
    language: str
    # Target environment (e.g., embedded, server, mobile)
    target_environment: str = "server"
    # Available resources
    available_memory_mb: int = 4096
    available_cpu_cores: int = 4
    # Optimization priority (0.0-1.0)
    memory_priority: float = 0.5
    cpu_priority: float = 0.5
    io_priority: float = 0.5
    network_priority: float = 0.5
    # Allowed optimizations
    allowed_optimizations: Set[OptimizationType] = field(default_factory=set)
    # Detected bottlenecks
    detected_bottlenecks: Dict[str, float] = field(default_factory=dict)
    # Specific library dependencies
    dependencies: Dict[str, str] = field(default_factory=dict)

@dataclass
class OptimizationSuggestion:
    """Suggested optimization for performance improvement"""
    # Type of optimization
    optimization_type: OptimizationType
    # Location in code
    location: Dict[str, Any]
    # Description of optimization
    description: str
    # Code changes suggested
    suggested_changes: str
    # Expected performance improvement
    expected_improvement: str
    # Implementation complexity (0.0-1.0)
    complexity: float
    # Confidence in optimization effectiveness (0.0-1.0)
    confidence: float
    # Original code snippet
    original_code: str
    # Optimized code
    optimized_code: str
    # References to explain the optimization
    references: List[str] = field(default_factory=list)

class RuntimeOptimizationCircuit:
    """
    Quantum-inspired circuit for runtime optimization detection and application
    """
    
    def __init__(self):
        """Initialize the runtime optimization circuit"""
        # Initialize optimization analyzers
        self._init_optimization_analyzers()
        
        # Initialize optimization patterns
        self._init_optimization_patterns()
        
        # Initialize algorithm complexity database
        self._init_algorithm_database()
        
        # Default context
        self.default_context = OptimizationContext(
            language="python",
            allowed_optimizations=set(OptimizationType)
        )
    
    def _init_optimization_analyzers(self):
        """Initialize analyzers for different optimization types"""
        self.optimization_analyzers = {
            OptimizationType.MEMORY: self._analyze_memory_optimizations,
            OptimizationType.CPU: self._analyze_cpu_optimizations,
            OptimizationType.IO: self._analyze_io_optimizations,
            OptimizationType.NETWORK: self._analyze_network_optimizations,
            OptimizationType.DATABASE: self._analyze_database_optimizations,
            OptimizationType.MULTITHREADING: self._analyze_threading_optimizations,
            OptimizationType.CACHING: self._analyze_caching_optimizations,
            OptimizationType.ALGORITHM: self._analyze_algorithm_optimizations
        }
    
    def _init_optimization_patterns(self):
        """Initialize patterns for detecting optimization opportunities"""
        self.optimization_patterns = {
            "python": {
                "memory_leak": {
                    "pattern": r'(?:append|extend|add)\s*\(.*\)\s*(?:in|inside)\s*(?:while|for)',
                    "description": "Potential memory leak in loop",
                    "suggestion": "Consider using a fixed-size data structure or clearing the collection periodically"
                },
                "inefficient_string_concat": {
                    "pattern": r'(?:\s*\+\s*=\s*|\s*\+\s*)(?:[\'"]\w+[\'"])',
                    "description": "Inefficient string concatenation in loop",
                    "suggestion": "Use join() method or list comprehension for string building"
                },
                "expensive_operation_in_loop": {
                    "pattern": r'for\s+.*\s+in\s+.*:(?:.*(?:sort|sorted|deepcopy|json\.loads|json\.dumps))',
                    "description": "Expensive operation inside a loop",
                    "suggestion": "Move the expensive operation outside the loop if possible"
                },
                "redundant_computation": {
                    "pattern": r'(?:len|max|min|sum)\s*\([^)]*\).*(?:for|while)',
                    "description": "Redundant computation in each loop iteration",
                    "suggestion": "Compute the value once before the loop"
                }
            },
            "javascript": {
                "dom_in_loop": {
                    "pattern": r'for\s*\(.*\)\s*{[^}]*document\.(?:getElementById|querySelector)',
                    "description": "DOM access inside loop",
                    "suggestion": "Cache DOM elements outside the loop"
                },
                "array_concat_in_loop": {
                    "pattern": r'for\s*\(.*\)\s*{[^}]*(?:concat|push)',
                    "description": "Array manipulation in loop",
                    "suggestion": "Use map/filter/reduce instead of imperative loops"
                }
            }
        }
    
    def _init_algorithm_database(self):
        """Initialize database of algorithm complexities"""
        self.algorithm_database = {
            "sorting": {
                "bubble_sort": {"time": "O(n²)", "space": "O(1)"},
                "selection_sort": {"time": "O(n²)", "space": "O(1)"},
                "insertion_sort": {"time": "O(n²)", "space": "O(1)"},
                "merge_sort": {"time": "O(n log n)", "space": "O(n)"},
                "quick_sort": {"time": "O(n log n)", "space": "O(log n)"},
                "heap_sort": {"time": "O(n log n)", "space": "O(1)"},
                "tim_sort": {"time": "O(n log n)", "space": "O(n)"}
            },
            "search": {
                "linear_search": {"time": "O(n)", "space": "O(1)"},
                "binary_search": {"time": "O(log n)", "space": "O(1)"},
                "depth_first_search": {"time": "O(V+E)", "space": "O(V)"},
                "breadth_first_search": {"time": "O(V+E)", "space": "O(V)"}
            },
            "data_structures": {
                "array_access": {"time": "O(1)", "space": "O(n)"},
                "linked_list_access": {"time": "O(n)", "space": "O(n)"},
                "hash_table_access": {"time": "O(1)", "space": "O(n)"},
                "binary_tree_access": {"time": "O(log n)", "space": "O(n)"},
                "graph_adjacency_list": {"time": "O(V+E)", "space": "O(V+E)"},
                "graph_adjacency_matrix": {"time": "O(V²)", "space": "O(V²)"}
            }
        }
    
    def analyze_code(self, code: str, filename: str, language: str = "", 
                    context: Optional[OptimizationContext] = None) -> List[OptimizationSuggestion]:
        """
        Analyze code for runtime optimization opportunities
        
        Args:
            code: Source code to analyze
            filename: Name of the file
            language: Programming language
            context: Optimization context
            
        Returns:
            List of optimization suggestions
        """
        # Determine language if not provided
        if not language:
            language = self._detect_language(code, filename)
        
        # Create context if not provided
        if not context:
            context = copy.deepcopy(self.default_context)
            context.language = language
        
        # Detect resource bottlenecks
        detected_bottlenecks = self._detect_bottlenecks(code, language)
        context.detected_bottlenecks = detected_bottlenecks
        
        # Apply each analyzer based on bottlenecks
        suggestions = []
        
        # Sort optimization types by bottleneck severity
        optimization_priorities = sorted(
            [(opt_type, detected_bottlenecks.get(opt_type.name.lower(), 0))
             for opt_type in OptimizationType],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Apply analyzers in priority order
        for opt_type, priority in optimization_priorities:
            if priority > 0.3 and opt_type in context.allowed_optimizations:
                analyzer = self.optimization_analyzers[opt_type]
                opt_suggestions = analyzer(code, filename, context)
                suggestions.extend(opt_suggestions)
        
        # Apply pattern-based detection
        pattern_suggestions = self._detect_with_patterns(code, filename, language)
        suggestions.extend(pattern_suggestions)
        
        # Remove duplicates and sort by confidence * expected improvement
        unique_suggestions = []
        seen_locations = set()
        
        for suggestion in suggestions:
            loc_key = f"{suggestion.location.get('file')}:{suggestion.location.get('line_start')}"
            if loc_key not in seen_locations:
                unique_suggestions.append(suggestion)
                seen_locations.add(loc_key)
        
        # Sort by effectiveness
        unique_suggestions.sort(
            key=lambda x: x.confidence * (1.0 - x.complexity),
            reverse=True
        )
        
        return unique_suggestions
    
    def optimize_code(self, code: str, suggestions: List[OptimizationSuggestion], 
                     language: str) -> str:
        """
        Apply optimizations to the code
        
        Args:
            code: Original code
            suggestions: List of optimization suggestions
            language: Programming language
            
        Returns:
            Optimized code
        """
        # Sort suggestions by their location in the code (from bottom to top)
        sorted_suggestions = sorted(
            suggestions, 
            key=lambda x: (x.location.get('line_end', 0), x.location.get('line_start', 0)),
            reverse=True
        )
        
        # Apply optimizations
        optimized_code = code
        for suggestion in sorted_suggestions:
            optimizer = self._get_optimizer(suggestion.optimization_type, language)
            if optimizer:
                optimized_code = optimizer(optimized_code, suggestion)
        
        return optimized_code
    
    def _detect_language(self, code: str, filename: str) -> str:
        """Detect programming language from code and filename"""
        # Check file extension first
        if filename:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.py':
                return 'python'
            elif ext in ['.js', '.jsx']:
                return 'javascript'
            elif ext in ['.java']:
                return 'java'
            elif ext in ['.c', '.cpp', '.cc', '.h', '.hpp']:
                return 'cpp'
        
        # Check code patterns
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code or 'var ' in code or 'const ' in code:
            return 'javascript'
        elif 'public class ' in code or 'private class ' in code:
            return 'java'
        elif '#include ' in code:
            return 'cpp'
        
        return 'unknown'
    
    def _detect_bottlenecks(self, code: str, language: str) -> Dict[str, float]:
        """Detect potential resource bottlenecks in code"""
        bottlenecks = {
            "memory": 0.0,
            "cpu": 0.0,
            "io": 0.0,
            "network": 0.0,
            "database": 0.0,
            "threading": 0.0
        }
        
        # Memory bottleneck detection
        memory_patterns = [
            (r'(?:list|dict|set|array)\s*\(.*\)', 0.3),  # Large data structures
            (r'(?:while|for).*(?:append|add|push|extend)', 0.5),  # Growing collections in loops
            (r'(?:deepcopy|copy\.deepcopy)', 0.4),  # Deep copying large structures
        ]
        
        for pattern, weight in memory_patterns:
            if re.search(pattern, code, re.DOTALL):
                bottlenecks["memory"] += weight
        
        # CPU bottleneck detection
        cpu_patterns = [
            (r'(?:for|while).*(?:for|while)', 0.6),  # Nested loops
            (r'(?:sort\(|sorted\()', 0.4),  # Sorting operations
            (r'(?:factorial|math\.factorial)', 0.7),  # Factorial calculations
            (r'(?:recursion|recursive)', 0.5),  # Recursive functions
        ]
        
        for pattern, weight in cpu_patterns:
            if re.search(pattern, code, re.DOTALL):
                bottlenecks["cpu"] += weight
        
        # I/O bottleneck detection
        io_patterns = [
            (r'(?:open\(|file\(|with\s+open)', 0.5),  # File operations
            (r'(?:read|write)\s*\(', 0.4),  # File reading/writing
            (r'(?:for|while).*(?:open\(|file\()', 0.7),  # File operations in loops
        ]
        
        for pattern, weight in io_patterns:
            if re.search(pattern, code, re.DOTALL):
                bottlenecks["io"] += weight
        
        # Network bottleneck detection
        network_patterns = [
            (r'(?:requests\.|http\.|fetch\(|urllib|curl)', 0.6),  # Network calls
            (r'(?:socket\.|connect\()', 0.5),  # Socket operations
            (r'(?:for|while).*(?:requests\.|http\.|fetch\()', 0.8),  # Network calls in loops
        ]
        
        for pattern, weight in network_patterns:
            if re.search(pattern, code, re.DOTALL):
                bottlenecks["network"] += weight
        
        # Database bottleneck detection
        db_patterns = [
            (r'(?:select|insert|update|delete).*(?:from|into|where)', 0.5),  # SQL operations
            (r'(?:cursor\.|connection\.|query\()', 0.4),  # Database connections
            (r'(?:for|while).*(?:cursor\.|connection\.|query\()', 0.8),  # DB operations in loops
        ]
        
        for pattern, weight in db_patterns:
            if re.search(pattern, code, re.DOTALL):
                bottlenecks["database"] += weight
        
        # Limit values to 1.0 max
        for key in bottlenecks:
            bottlenecks[key] = min(1.0, bottlenecks[key])
        
        return bottlenecks
    
    def _detect_with_patterns(self, code: str, filename: str, language: str) -> List[OptimizationSuggestion]:
        """Detect optimization opportunities using regex patterns"""
        suggestions = []
        
        # Get language-specific patterns
        lang_patterns = self.optimization_patterns.get(language, {})
        
        lines = code.splitlines()
        for pattern_name, pattern_info in lang_patterns.items():
            regex = pattern_info["pattern"]
            
            # Find matches in code
            for i, line in enumerate(lines):
                if re.search(regex, line):
                    # Create optimization suggestion
                    suggestion = OptimizationSuggestion(
                        optimization_type=self._get_optimization_type(pattern_name),
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=pattern_info["description"],
                        suggested_changes=pattern_info["suggestion"],
                        expected_improvement="Moderate performance improvement",
                        complexity=0.3,
                        confidence=0.7,
                        original_code=line,
                        optimized_code=self._get_optimized_code(line, pattern_name, language)
                    )
                    
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _get_optimization_type(self, pattern_name: str) -> OptimizationType:
        """Map pattern name to optimization type"""
        if "memory" in pattern_name:
            return OptimizationType.MEMORY
        elif "string" in pattern_name or "array" in pattern_name:
            return OptimizationType.CPU
        elif "file" in pattern_name:
            return OptimizationType.IO
        elif "network" in pattern_name:
            return OptimizationType.NETWORK
        elif "query" in pattern_name or "database" in pattern_name:
            return OptimizationType.DATABASE
        elif "thread" in pattern_name or "concurrent" in pattern_name:
            return OptimizationType.MULTITHREADING
        elif "cache" in pattern_name:
            return OptimizationType.CACHING
        else:
            return OptimizationType.ALGORITHM
    
    def _get_optimized_code(self, line: str, pattern_name: str, language: str) -> str:
        """Generate optimized code based on pattern"""
        optimized_line = line
        
        if language == "python":
            if pattern_name == "inefficient_string_concat":
                # Replace string concatenation with join
                if "+=" in line and any(q in line for q in ["'", '"']):
                    # Simple case: building a string with +=
                    var_match = re.search(r'(\w+)\s*\+=', line)
                    if var_match:
                        var_name = var_match.group(1)
                        # Replace with a comment and join suggestion
                        optimized_line = f"# Use join instead: {var_name} = ''.join([...parts...])"
            
            elif pattern_name == "redundant_computation":
                # Move computation outside loop
                func_match = re.search(r'((?:len|max|min|sum)\s*\([^)]*\))', line)
                if func_match:
                    computation = func_match.group(1)
                    # Replace with a pre-computed variable
                    var_name = f"{func_match.group(1).split('(')[0]}_value"
                    optimized_line = f"# Pre-compute: {var_name} = {computation}\n" + line.replace(computation, var_name)
            
            elif pattern_name == "expensive_operation_in_loop":
                # Move expensive operation outside loop
                if "sort" in line or "deepcopy" in line:
                    # Add comment suggesting optimization
                    optimized_line = "# Move expensive operation outside the loop\n" + line
        
        elif language == "javascript":
            if pattern_name == "dom_in_loop":
                # Cache DOM elements
                dom_match = re.search(r'(document\.(?:getElementById|querySelector)\([^)]*\))', line)
                if dom_match:
                    dom_operation = dom_match.group(1)
                    # Create variable name for cached element
                    var_name = f"cachedElement"
                    optimized_line = f"// Cache DOM element: const {var_name} = {dom_operation};\n" + line.replace(dom_operation, var_name)
        
        return optimized_line
    
    def _analyze_memory_optimizations(self, code: str, filename: str, 
                                    context: OptimizationContext) -> List[OptimizationSuggestion]:
        """Analyze code for memory optimization opportunities"""
        suggestions = []
        
        # Memory leak detection in loops
        lines = code.splitlines()
        growing_collections = {}
        
        for i, line in enumerate(lines):
            # Check for growing collections in loops
            if "for " in line or "while " in line:
                in_loop = True
                loop_start = i
                loop_indent = len(line) - len(line.lstrip())
                
                # Find collections that grow inside this loop
                for j in range(i+1, min(i+100, len(lines))):
                    if j >= len(lines):
                        break
                        
                    # Check if still in loop (by indentation)
                    current_indent = len(lines[j]) - len(lines[j].lstrip())
                    if current_indent <= loop_indent and len(lines[j].strip()) > 0:
                        # Loop ended
                        in_loop = False
                        break
                    
                    # Check for collection growth operations
                    growth_ops = {
                        'python': ['.append(', '.extend(', '.add(', '.update('],
                        'javascript': ['.push(', '.concat(', '.unshift('],
                        'java': ['.add(', '.put(', '.addAll('],
                        'cpp': ['.push_back(', '.insert(', '.emplace(']
                    }
                    
                    lang_ops = growth_ops.get(context.language, growth_ops['python'])
                    
                    for op in lang_ops:
                        if op in lines[j]:
                            # Extract variable being grown
                            var_match = re.search(r'(\w+)\s*' + re.escape(op), lines[j])
                            if var_match:
                                var_name = var_match.group(1)
                                if var_name not in growing_collections:
                                    growing_collections[var_name] = []
                                
                                # Add location of growth
                                growing_collections[var_name].append(j)
        
        # Generate suggestions for memory leaks
        for var_name, locations in growing_collections.items():
            if len(locations) > 2:  # Multiple growth operations for the same collection
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.MEMORY,
                    location={"file": filename, "line_start": locations[0]+1, "line_end": locations[-1]+1},
                    description=f"Potential memory leak: Collection '{var_name}' grows inside loop",
                    suggested_changes="Consider using a fixed-size data structure or clearing the collection periodically",
                    expected_improvement="Reduced memory usage and potential performance improvement",
                    complexity=0.4,
                    confidence=0.7,
                    original_code=lines[locations[0]],
                    optimized_code=f"# Consider pre-allocating or clearing periodically\n{lines[locations[0]]}"
                )
                
                suggestions.append(suggestion)
        
        # Large object allocation detection
        large_allocations = []
        
        for i, line in enumerate(lines):
            # Check for large array/list/dictionary allocations
            allocation_patterns = {
                'python': [
                    r'(\w+)\s*=\s*\[\s
def _analyze_memory_optimizations(self, code: str, filename: str, 
                                context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for memory optimization opportunities"""
    suggestions = []
    
    # Memory leak detection in loops
    lines = code.splitlines()
    growing_collections = {}
    
    for i, line in enumerate(lines):
        # Check for growing collections in loops
        if "for " in line or "while " in line:
            in_loop = True
            loop_start = i
            loop_indent = len(line) - len(line.lstrip())
            
            # Find collections that grow inside this loop
            for j in range(i+1, min(i+100, len(lines))):
                if j >= len(lines):
                    break
                    
                # Check if still in loop (by indentation)
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent and len(lines[j].strip()) > 0:
                    # Loop ended
                    in_loop = False
                    break
                
                # Check for collection growth operations
                growth_ops = {
                    'python': ['.append(', '.extend(', '.add(', '.update('],
                    'javascript': ['.push(', '.concat(', '.unshift('],
                    'java': ['.add(', '.put(', '.addAll('],
                    'cpp': ['.push_back(', '.insert(', '.emplace(']
                }
                
                lang_ops = growth_ops.get(context.language, growth_ops['python'])
                
                for op in lang_ops:
                    if op in lines[j]:
                        # Extract variable being grown
                        var_match = re.search(r'(\w+)\s*' + re.escape(op), lines[j])
                        if var_match:
                            var_name = var_match.group(1)
                            if var_name not in growing_collections:
                                growing_collections[var_name] = []
                            
                            # Add location of growth
                            growing_collections[var_name].append(j)
    
    # Generate suggestions for memory leaks
    for var_name, locations in growing_collections.items():
        if len(locations) > 2:  # Multiple growth operations for the same collection
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.MEMORY,
                location={"file": filename, "line_start": locations[0]+1, "line_end": locations[-1]+1},
                description=f"Potential memory leak: Collection '{var_name}' grows inside loop",
                suggested_changes="Consider using a fixed-size data structure or clearing the collection periodically",
                expected_improvement="Reduced memory usage and potential performance improvement",
                complexity=0.4,
                confidence=0.7,
                original_code=lines[locations[0]],
                optimized_code=f"# Consider pre-allocating or clearing periodically\n{lines[locations[0]]}"
            )
            
            suggestions.append(suggestion)
    
    # Large object allocation detection
    large_allocations = []
    
    for i, line in enumerate(lines):
        # Check for large array/list/dictionary allocations
        allocation_patterns = {
            'python': [
                r'(\w+)\s*=\s*\[.*\]',
                r'(\w+)\s*=\s*\{.*\}',
                r'(\w+)\s*=\s*dict\(',
                r'(\w+)\s*=\s*list\('
            ],
            'javascript': [
                r'(?:let|const|var)\s+(\w+)\s*=\s*\[.*\]',
                r'(?:let|const|var)\s+(\w+)\s*=\s*\{.*\}'
            ],
            'java': [
                r'(?:List|ArrayList|Map|HashMap|Set|HashSet)<.*>\s+(\w+)\s*=\s*new',
                r'(?:int|double|float|char)\[\]\s+(\w+)\s*='
            ],
            'cpp': [
                r'(?:vector|map|set|unordered_map|array)<.*>\s+(\w+)',
                r'(?:int|double|float|char)\s+(\w+)\s*\[.*\]'
            ]
        }
        
        lang_patterns = allocation_patterns.get(context.language, allocation_patterns['python'])
        
        for pattern in lang_patterns:
            match = re.search(pattern, line)
            if match:
                var_name = match.group(1)
                # Check if the allocation seems large
                if ('*' in line or 
                    any(str(n) in line for n in range(1000, 10000)) or
                    'new' in line and ('(' in line and ')' in line)):
                    large_allocations.append((i, var_name, line))
    
    # Generate suggestions for large allocations
    for i, var_name, line in large_allocations:
        suggestion = OptimizationSuggestion(
            optimization_type=OptimizationType.MEMORY,
            location={"file": filename, "line_start": i+1, "line_end": i+1},
            description=f"Large object allocation: '{var_name}' may consume significant memory",
            suggested_changes="Consider lazy loading, pagination, or streaming if appropriate",
            expected_improvement="Reduced memory usage and potential for better scalability",
            complexity=0.5,
            confidence=0.6,
            original_code=line,
            optimized_code=f"# Consider lazy loading or streaming for large data\n{line}"
        )
        
        suggestions.append(suggestion)
    
    return suggestions

def _analyze_cpu_optimizations(self, code: str, filename: str, 
                            context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for CPU optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Nested loops detection
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for nested loops
            for j in range(i+1, min(i+50, len(lines))):
                if j >= len(lines):
                    break
                
                # Check indentation to see if we're still in the same loop
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent:
                    break
                
                # Check for nested loop
                if ("for " in lines[j] or "while " in lines[j]) and current_indent > loop_indent:
                    nested_indent = current_indent
                    
                    # Look for a third level of nesting
                    for k in range(j+1, min(j+50, len(lines))):
                        if k >= len(lines):
                            break
                        
                        third_indent = len(lines[k]) - len(lines[k].lstrip())
                        if third_indent <= nested_indent:
                            break
                        
                        # Triple nested loop detected
                        if ("for " in lines[k] or "while " in lines[k]) and third_indent > nested_indent:
                            suggestion = OptimizationSuggestion(
                                optimization_type=OptimizationType.CPU,
                                location={"file": filename, "line_start": i+1, "line_end": k+1},
                                description="Triple nested loop detected - potential O(n³) complexity",
                                suggested_changes="Consider optimization strategies: memoization, better algorithms, or data preprocessing",
                                expected_improvement="Potentially significant performance improvement for large datasets",
                                complexity=0.7,
                                confidence=0.9,
                                original_code=f"{lines[i]}\n  {lines[j]}\n    {lines[k]}",
                                optimized_code="# Consider algorithm optimization to reduce complexity\n# Options: memoization, better data structures, or precomputation"
                            )
                            
                            suggestions.append(suggestion)
                            # Skip to after the third loop
                            break
                    
                    # Double nested loop (if no triple was found)
                    if not any(s.location["line_start"] == i+1 for s in suggestions):
                        suggestion = OptimizationSuggestion(
                            optimization_type=OptimizationType.CPU,
                            location={"file": filename, "line_start": i+1, "line_end": j+1},
                            description="Nested loop detected - potential O(n²) complexity",
                            suggested_changes="Consider whether this operation can be optimized",
                            expected_improvement="Potential performance improvement for large datasets",
                            complexity=0.5,
                            confidence=0.8,
                            original_code=f"{lines[i]}\n  {lines[j]}",
                            optimized_code="# Consider optimization: can this nested loop be avoided?"
                        )
                        
                        suggestions.append(suggestion)
                    
                    break
    
    # Expensive operations detection
    expensive_ops = {
        'python': [
            ('deepcopy', 'Consider using shallow copy if possible or optimizing what needs to be copied'),
            ('sort', 'If sorting is called multiple times, consider sorting once or using a data structure that maintains order'),
            ('pickle.loads', 'Deserialization can be expensive, consider caching results'),
            ('json.loads', 'JSON parsing can be expensive, consider caching or using a more efficient format'),
            ('re.compile', 'Move regex compilation outside of loops if used repeatedly')
        ],
        'javascript': [
            ('JSON.parse', 'JSON parsing can be expensive, consider caching results'),
            ('sort', 'Sorting is expensive, consider alternative approaches if called frequently'),
            ('map', 'Large array operations can be costly')
        ],
        'java': [
            ('new ', 'Object creation inside loops can be expensive'),
            ('Collections.sort', 'Sorting is expensive, consider alternatives'),
            ('Pattern.compile', 'Move regex compilation outside of loops')
        ]
    }
    
    lang_ops = expensive_ops.get(context.language, expensive_ops['python'])
    
    for i, line in enumerate(lines):
        for op, suggestion_text in lang_ops:
            if op in line:
                # Check if inside a loop
                in_loop = False
                for j in range(i-1, max(0, i-20), -1):
                    if "for " in lines[j] or "while " in lines[j]:
                        loop_indent = len(lines[j]) - len(lines[j].lstrip())
                        line_indent = len(line) - len(line.lstrip())
                        if line_indent > loop_indent:
                            in_loop = True
                            loop_line = j
                            break
                
                if in_loop:
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.CPU,
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=f"Expensive operation '{op}' inside loop",
                        suggested_changes=suggestion_text,
                        expected_improvement="May significantly improve performance for large datasets",
                        complexity=0.4,
                        confidence=0.8,
                        original_code=line,
                        optimized_code=f"# Move outside loop if possible:\n# {op}_result = {line.strip()}\n{line}"
                    )
                    
                    suggestions.append(suggestion)
    
    return suggestions

def _analyze_io_optimizations(self, code: str, filename: str, 
                           context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for I/O optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # File operations in loops
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for file operations in the loop
            for j in range(i+1, min(i+50, len(lines))):
                if j >= len(lines):
                    break
                
                # Check if still in loop by indentation
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent:
                    break
                
                # File operation patterns by language
                file_ops = {
                    'python': ['open(', '.read(', '.write(', '.close('],
                    'javascript': ['fs.', '.readFile', '.writeFile', 'createReadStream'],
                    'java': ['FileInputStream', 'FileOutputStream', 'BufferedReader', '.read(', '.write('],
                    'cpp': ['ifstream', 'ofstream', 'fstream', '.read(', '.write(']
                }
                
                lang_ops = file_ops.get(context.language, file_ops['python'])
                
                if any(op in lines[j] for op in lang_ops):
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.IO,
                        location={"file": filename, "line_start": j+1, "line_end": j+1},
                        description="File operation inside loop",
                        suggested_changes="Move file operations outside loops where possible, use buffering, or consider batch operations",
                        expected_improvement="Significant I/O performance improvement",
                        complexity=0.4,
                        confidence=0.9,
                        original_code=lines[j],
                        optimized_code="# Move file operation outside loop\n# Use buffering or batch processing"
                    )
                    
                    suggestions.append(suggestion)
    
    # Multiple file open/close operations
    file_opens = []
    
    for i, line in enumerate(lines):
        # File opening patterns
        open_patterns = {
            'python': r'(?:(\w+)\s*=\s*open\(|with\s+open\(.*\)\s+as\s+(\w+))',
            'javascript': r'(?:(\w+)\s*=\s*fs\.(?:open|readFile|writeFile)|require\([\'"]fs[\'"]\))',
            'java': r'(?:new\s+(?:FileInputStream|FileOutputStream|FileReader|FileWriter)\(|(\w+)\s*=\s*new\s+File\()',
            'cpp': r'(?:(\w+)\.open\(|(\w+)\s*=\s*(?:ifstream|ofstream|fstream))'
        }
        
        pattern = open_patterns.get(context.language, open_patterns['python'])
        match = re.search(pattern, line)
        
        if match:
            file_var = match.group(1) if match.group(1) else match.group(2) if len(match.groups()) > 1 else "file_handle"
            file_opens.append((i, file_var, line))
    
    # Check for repeated file operations
    if len(file_opens) > 3:
        suggestion = OptimizationSuggestion(
            optimization_type=OptimizationType.IO,
            location={"file": filename, "line_start": file_opens[0][0]+1, "line_end": file_opens[-1][0]+1},
            description="Multiple file operations detected",
            suggested_changes="Consider using a connection pool or persistent file handles",
            expected_improvement="Reduced I/O overhead and better resource management",
            complexity=0.5,
            confidence=0.7,
            original_code=file_opens[0][2],
            optimized_code="# Consider file handle reuse or connection pooling\n# Open files once and reuse handles"
        )
        
        suggestions.append(suggestion)
    
    return suggestions

def _analyze_network_optimizations(self, code: str, filename: str, 
                               context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for network optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Network calls in loops
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for network operations in the loop
            for j in range(i+1, min(i+50, len(lines))):
                if j >= len(lines):
                    break
                
                # Check if still in loop by indentation
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent:
                    break
                
                # Network operation patterns by language
                net_ops = {
                    'python': ['requests.', 'urllib.', 'http.client', 'socket.', '.recv(', '.send('],
                    'javascript': ['fetch(', 'axios.', '.ajax(', 'http.', 'https.', 'XMLHttpRequest'],
                    'java': ['HttpClient', 'URL(', 'Socket(', '.connect(', '.openConnection()'],
                    'cpp': ['socket(', 'connect(', 'send(', 'recv(']
                }
                
                lang_ops = net_ops.get(context.language, net_ops['python'])
                
                if any(op in lines[j] for op in lang_ops):
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.NETWORK,
                        location={"file": filename, "line_start": j+1, "line_end": j+1},
                        description="Network operation inside loop",
                        suggested_changes="Batch network requests, use pagination, or implement parallel requests",
                        expected_improvement="Significant network performance improvement",
                        complexity=0.5,
                        confidence=0.9,
                        original_code=lines[j],
                        optimized_code="# Consider batch requests or connection pooling\n# Use async/parallel requests if appropriate"
                    )
                    
                    suggestions.append(suggestion)
    
    # Connection reuse opportunities
    connection_create = []
    
    for i, line in enumerate(lines):
        # Connection creation patterns
        conn_patterns = {
            'python': r'(?:requests\.(?:get|post|put|delete)|urllib\.(?:request|urlopen)|http\.client\.HTTPConnection)',
            'javascript': r'(?:new\s+XMLHttpRequest\(\)|fetch\(|axios\.(?:get|post|put|delete))',
            'java': r'(?:new\s+URL\(|\.openConnection\(\)|HttpClient\.newBuilder\(\))',
            'cpp': r'(?:socket\(|connect\()'
        }
        
        pattern = conn_patterns.get(context.language, conn_patterns['python'])
        if re.search(pattern, line):
            connection_create.append((i, line))
    
    # Check for multiple connections without reuse
    if len(connection_create) > 3:
        close_pattern = r'(?:\.close\(\)|\.disconnect\(\))'
        close_count = sum(1 for i, line in enumerate(lines) if re.search(close_pattern, line))
        
        if close_count < len(connection_create) / 2:  # Less than half of connections are explicitly closed
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.NETWORK,
                location={"file": filename, "line_start": connection_create[0][0]+1, "line_end": connection_create[-1][0]+1},
                description="Multiple network connections without explicit cleanup",
                suggested_changes="Implement connection pooling or session reuse",
                expected_improvement="Better resource management and network performance",
                complexity=0.6,
                confidence=0.7,
                original_code=connection_create[0][1],
                optimized_code="# Use connection pooling:\n# session = create_connection_pool()\n# reuse session for multiple requests"
            )
            
            suggestions.append(suggestion)
    
    return suggestions

def _analyze_database_optimizations(self, code: str, filename: str, 
                                context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for database optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Database queries in loops
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for DB operations in the loop
            for j in range(i+1, min(i+50, len(lines))):
                if j >= len(lines):
                    break
                
                # Check if still in loop by indentation
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent:
                    break
                
                # DB operation patterns by language
                db_ops = {
                    'python': ['.execute(', '.query(', '.commit(', 'cursor.', 'connection.'],
                    'javascript': ['.query(', '.findOne(', '.findAll(', '.update(', '.delete('],
                    'java': ['.executeQuery(', '.executeUpdate(', '.prepareStatement(', 'ResultSet'],
                    'cpp': ['.query(', '.execute(', 'sqlite3_']
                }
                
                lang_ops = db_ops.get(context.language, db_ops['python'])
                
                if any(op in lines[j] for op in lang_ops):
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.DATABASE,
                        location={"file": filename, "line_start": j+1, "line_end": j+1},
                        description="Database operation inside loop",
                        suggested_changes="Use batch queries, JOINs, or optimize with WHERE clauses",
                        expected_improvement="Significant database performance improvement",
                        complexity=0.5,
                        confidence=0.9,
                        original_code=lines[j],
                        optimized_code="# Use batch operations or optimize queries\n# Consider: SELECT batch_data WHERE condition\n# Instead of: for item in items: SELECT data WHERE id=item"
                    )
                    
                    suggestions.append(suggestion)
    
    # Query efficiency opportunities
    for i, line in enumerate(lines):
        # Check for SELECT * queries
        if "SELECT *" in line or "select *" in line:
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.DATABASE,
                location={"file": filename, "line_start": i+1, "line_end": i+1},
                description="Inefficient SELECT * query",
                suggested_changes="Specify only needed columns instead of selecting all",
                expected_improvement="Reduced network traffic and query processing time",
                complexity=0.2,
                confidence=0.8,
                original_code=line,
                optimized_code=line.replace("SELECT *", "SELECT specific_columns").replace("select *", "select specific_columns")
            )
            
            suggestions.append(suggestion)
        
        # Check for missing indices hint
        if ("WHERE" in line or "where" in line) and not ("INDEX" in line or "index" in line):
            suggestion = OptimizationSuggestion(
                optimization_type=OptimizationType.DATABASE,
                location={"file": filename, "line_start": i+1, "line_end": i+1},
                description="Query may benefit from indexing",
                suggested_changes="Ensure proper indices exist for query conditions",
                expected_improvement="Potentially significant query performance improvement",
                complexity=0.4,
                confidence=0.6,
                original_code=line,
                optimized_code="# Ensure indices exist for query conditions\n" + line
            )
            
            suggestions.append(suggestion)
    
    return suggestions

def _analyze_threading_optimizations(self, code: str, filename: str, 
                                  context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for multithreading optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Look for CPU-intensive operations that could be parallelized
    cpu_intensive = []
    
    # Detect CPU-intensive code sections
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for computation in the loop
            computation = False
            loop_end = i
            
            for j in range(i+1, min(i+50, len(lines))):
                if j >= len(lines):
                    break
                
                # Check if still in loop by indentation
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent and len(lines[j].strip()) > 0:
                    loop_end = j
                    break
                
                # Look for computation patterns
                comp_patterns = [
                    r'[\+\-\*\/\%]', # Math operations
                    r'(?:math\.|np\.|calculate|compute)',  # Math libraries
                    r'(?:sort|map|reduce|filter)',  # Data operations
                ]
                
                if any(re.search(pattern, lines[j]) for pattern in comp_patterns):
                    computation = True
            
            # Check if loop has computation but no I/O, DB, or network operations
            if computation:
                io_patterns = [
                    r'(?:open\(|read\(|write\(|print\()',  # I/O operations
                    r'(?:requests\.|http\.|fetch\()',  # Network operations
                    r'(?:execute\(|query\(|cursor\.|connection\.)'  # DB operations
                ]
                
                # If no I/O operations in the loop, it's CPU-bound and can be parallelized
                if not any(any(re.search(pattern, lines[j]) for pattern in io_patterns) 
                          for j in range(i+1, loop_end)):
                    cpu_intensive.append((i, loop_end, line))
    
    # Generate parallelization suggestions
    for start, end, line in cpu_intensive:
        parallelization_code = ""
        
        # Generate language-specific parallelization code
        if context.language == 'python':
            parallelization_code = """# Parallelize using ThreadPoolExecutor or ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def process_chunk(chunk):
    results = []
    for item in chunk:
        # Process the item
        result = compute_something(item)
        results.append(result)
    return results

# Split data into chunks
chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

# Process in parallel
with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(process_chunk, chunks))

# Combine results
final_results = [item for sublist in results for item in sublist]"""
        
        elif context.language == 'javascript':
            parallelization_code = """// Use Worker threads for parallelization
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');

if (isMainThread) {
  // Split data into chunks
  const chunks = [];
  for (let i = 0; i < data.length; i += chunkSize) {
    chunks.push(data.slice(i, i + chunkSize));
  }
  
  // Create workers for each chunk
  const workers = chunks.map(chunk => {
    return new Promise((resolve, reject) => {
      const worker = new Worker(__filename, { workerData: chunk });
      worker.on('message', resolve);
      worker.on('error', reject);
    });
  });
  
  // Wait for all workers to complete
  Promise.all(workers).then(results => {
    // Combine results
    const finalResults = [].concat(...results);
    console.log('Done processing all data');
  });
} else {
  // Worker process
  const results = [];
  for (const item of workerData) {
    // Process the item
    const result = computeSomething(item);
    results.push(result);
  }
  parentPort.postMessage(results);
}"""
        
        suggestion = OptimizationSuggestion(
            optimization_type=OptimizationType.MULTITHREADING,
            location={"file": filename, "line_start": start+1, "line_end": end+1},
            description="CPU-intensive loop that could be parallelized",
            suggested_changes="Implement parallel processing to utilize multiple CPU cores",
            expected_improvement=f"Up to {context.available_cpu_cores}x speedup for CPU-bound operations",
            complexity=0.7,
            confidence=0.7,
            original_code=line,
            optimized_code=parallelization_code
        )
        
        suggestions.append(suggestion)
    
    return suggestions

def _analyze_caching_optimizations(self, code: str, filename: str, 
                               context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for caching optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Identify expensive function calls that could be cached
    for i, line in enumerate(lines):
        # Function definition patterns
        if context.language == 'python' and re.search(r'def\s+(\w+)\s*\(', line):
            func_name = re.search(r'def\s+(\w+)\s*\(', line).group(1)
            func_start = i
            func_indent = len(line) - len(line.lstrip())
            func_end = i
            
            # Find function boundaries
            for j in range(i+1, len(lines)):
                if j >= len(lines):
                    break
                
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= func_indent and len(lines[j].strip()) > 0:
                    func_end = j - 1
                    break
            
            # Check for expensive operations inside function
            expensive_patterns = [
                r'(?:requests\.|urllib\.|http\.)',  # Network calls
                r'(?:open\(|read\(|write\()',  # File I/O
                r'(?:execute\(|query\()',  # Database operations
                r'(?:math\.factorial|itertools\.permutations)',  # Expensive computations
                r'(?:json\.loads|json\.dumps|pickle)',  # Serialization
                r'(?:re\.match|re\.search|re\.compile)'  # Regular expressions
            ]
            
            has_expensive_ops = False
            for j in range(i+1, func_end+1):
                if any(re.search(pattern, lines[j]) for pattern in expensive_patterns):
                    has_expensive_ops = True
                    break
            
            # Check for potential idempotent function (same inputs -> same outputs)
            if has_expensive_ops:
                params = re.search(r'def\s+\w+\s*\((.*?)\)', line).group(1)
                has_side_effects = any(re.search(r'global\s+', lines[j]) for j in range(i+1, func_end+1))
                
                if not has_side_effects and params:
                    caching_code = ""
                    
                    if context.language == 'python':
                        caching_code = f"""# Add memoization/caching:
from functools import lru_cache

@lru_cache(maxsize=128)
{line}"""
                    
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.CACHING,
                        location={"file": filename, "line_start": i+1, "line_end": func_end+1},
                        description=f"Function '{func_name}' performs expensive operations and could benefit from caching",
                        suggested_changes="Implement memoization or caching for repeated function calls with the same inputs",
                        expected_improvement="Significant performance improvement for repeated calls",
                        complexity=0.3,
                        confidence=0.8,
                        original_code=line,
                        optimized_code=caching_code
                    )
                    
                    suggestions.append(suggestion)
    
    # Function calls that are repeated and could be cached
    function_calls = {}
    
    # Detect repeated function calls
    for i, line in enumerate(lines):
        # Function call patterns
        call_match = re.search(r'(\w+)\s*\([^)]*\)', line)
        if call_match:
            func_name = call_match.group(1)
            if func_name not in ['if', 'for', 'while']:  # Skip control structures
                if func_name not in function_calls:
                    function_calls[func_name] = []
                function_calls[func_name].append(i)
    
    # Check for functions called multiple times
    for func_name, call_lines in function_calls.items():
        if len(call_lines) > 3:  # Called more than 3 times
            call_intervals = [call_lines[i+1] - call_lines[i] for i in range(len(call_lines)-1)]
            avg_interval = sum(call_intervals) / len(call_intervals)
            
            if avg_interval < 10:  # Close calls suggest potential for caching
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.CACHING,
                    location={"file": filename, "line_start": call_lines[0]+1, "line_end": call_lines[-1]+1},
                    description=f"Function '{func_name}' is called frequently and may benefit from caching",
                    suggested_changes="Cache results for frequently called functions",
                    expected_improvement="Reduced redundant computation",
                    complexity=0.4,
                    confidence=0.7,
                    original_code=lines[call_lines[0]],
                    optimized_code=f"# Implement caching for frequently called function\n# result = cached_call('{func_name}', params...)"
                )
                
                suggestions.append(suggestion)
    
    return suggestions

def _analyze_algorithm_optimizations(self, code: str, filename: str, 
                                  context: OptimizationContext) -> List[OptimizationSuggestion]:
    """Analyze code for algorithm optimization opportunities"""
    suggestions = []
    
    lines = code.splitlines()
    
    # Check for inefficient sorting algorithms
    for i, line in enumerate(lines):
        if context.language == 'python':
            # Check for bubble sort implementation
            bubble_sort_pattern = r'for\s+.*\s+in\s+.*:.*\s+for\s+.*\s+in\s+.*:.*\s+if\s+.*\[\s*.*\s*\]\s*>\s*.*\[\s*.*\s*\].*\s+.*\[\s*.*\s*\],\s*.*\[\s*.*\s*\]\s*='
            if re.search(bubble_sort_pattern, '\n'.join(lines[i:i+10]), re.DOTALL):
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.ALGORITHM,
                    location={"file": filename, "line_start": i+1, "line_end": i+10},
                    description="Inefficient bubble sort implementation detected",
                    suggested_changes="Use language-provided sorting functions (sorted() or list.sort())",
                    expected_improvement="O(n log n) instead of O(n²) complexity",
                    complexity=0.2,
                    confidence=0.8,
                    original_code='\n'.join(lines[i:i+10]),
                    optimized_code="# Replace with built-in sorting\ndata.sort()  # or sorted(data)"
                )
                
                suggestions.append(suggestion)
        
        # Check for linear search on sorted data
        linear_search_pattern = r'for\s+.*\s+in\s+.*:.*\s+if\s+.*\s*==\s*.*:'
        if re.search(linear_search_pattern, '\n'.join(lines[i:i+5]), re.DOTALL):
            sorted_data_indicators = ['sorted', 'ordered', 'ascending', 'descending']
            context_lines = '\n'.join(lines[max(0, i-5):i+5])
            
            if any(indicator in context_lines for indicator in sorted_data_indicators):
                suggestion = OptimizationSuggestion(
                    optimization_type=OptimizationType.ALGORITHM,
                    location={"file": filename, "line_start": i+1, "line_end": i+5},
                    description="Linear search on potentially sorted data",
                    suggested_changes="Use binary search for sorted data",
                    expected_improvement="O(log n) instead of O(n) complexity",
                    complexity=0.4,
                    confidence=0.6,
                    original_code='\n'.join(lines[i:i+5]),
                    optimized_code="# Use binary search for sorted data\n# index = bisect.bisect_left(sorted_data, target)"
                )
                
                suggestions.append(suggestion)
    
    # Check for O(n²) string concatenation in loops
    for i, line in enumerate(lines):
        if "for " in line or "while " in line:
            loop_indent = len(line) - len(line.lstrip())
            
            # Look for string concatenation in loop
            for j in range(i+1, min(i+20, len(lines))):
                if j >= len(lines):
                    break
                
                # Check if still in loop by indentation
                current_indent = len(lines[j]) - len(lines[j].lstrip())
                if current_indent <= loop_indent:
                    break
                
                if context.language == 'python' and "+=" in lines[j] and any(q in lines[j] for q in ["'", '"']):
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.ALGORITHM,
                        location={"file": filename, "line_start": j+1, "line_end": j+1},
                        description="Inefficient string concatenation in loop",
                        suggested_changes="Use list comprehension or join() for string building",
                        expected_improvement="O(n) instead of O(n²) complexity",
                        complexity=0.2,
                        confidence=0.9,
                        original_code=lines[j],
                        optimized_code="# Instead of string += part\n# parts = []\n# for item in items: parts.append(str(item))\n# result = ''.join(parts)"
                    )
                    
                    suggestions.append(suggestion)
                
                elif context.language == 'javascript' and "+=" in lines[j] and any(q in lines[j] for q in ["'", '"']):
                    suggestion = OptimizationSuggestion(
                        optimization_type=OptimizationType.ALGORITHM,
                        location={"file": filename, "line_start": j+1, "line_end": j+1},
                        description="Inefficient string concatenation in loop",
                        suggested_changes="Use array join() for string building",
                        expected_improvement="O(n) instead of O(n²) complexity",
                        complexity=0.2,
                        confidence=0.9,
                        original_code=lines[j],
                        optimized_code="// Instead of string += part\n// const parts = [];\n// for (const item of items) { parts.push(item); }\n// const result = parts.join('');"
                    )
                    
                    suggestions.append(suggestion)
    
    # Check for potential recursive functions without memoization
    if context.language == 'python':
        for i, line in enumerate(lines):
            if "def " in line:
                func_name = re.search(r'def\s+(\w+)', line).group(1)
                func_indent = len(line) - len(line.lstrip())
                
                # Check if function is recursive
                is_recursive = False
                for j in range(i+1, min(i+50, len(lines))):
                    if j >= len(lines):
                        break
                    
                    current_indent = len(lines[j]) - len(lines[j].lstrip())
                    if current_indent <= func_indent and len(lines[j].strip()) > 0:
                        break
                    
                    if func_name in lines[j] and "def " not in lines[j]:
                        is_recursive = True
                        break
                
                if is_recursive:
                    # Check for fibonacci-like patterns (exponential complexity)
                    fib_pattern = r'return\s+.*\(\s*.*\s*-\s*1\s*\)\s*\+\s*.*\(\s*.*\s*-\s*2\s*\)'
                    if re.search(fib_pattern, '\n'.join(lines[i:j+1]), re.DOTALL):
                        suggestion = OptimizationSuggestion(
                            optimization_type=OptimizationType.ALGORITHM,
                            location={"file": filename, "line_start": i+1, "line_end": j+1},
                            description="Recursive function with potential exponential complexity",
                            suggested_changes="Implement memoization or dynamic programming",
                            expected_improvement="O(n) instead of O(2^n) complexity",
                            complexity=0.5,
                            confidence=0.8,
                            original_code=lines[i],
                            optimized_code="# Add memoization to recursive function\nfrom functools import lru_cache\n\n@lru_cache(maxsize=None)\n" + lines[i]
                        )
                        
                        suggestions.append(suggestion)
    
    return suggestions

def _get_optimizer(self, optimization_type: OptimizationType, language: str):
    """Get appropriate optimizer function for an optimization type"""
    optimizers = {
        OptimizationType.MEMORY: {
            "python": self._optimize_memory_python,
            "javascript": self._optimize_memory_js
        },
        OptimizationType.CPU: {
            "python": self._optimize_cpu_python,
            "javascript": self._optimize_cpu_js
        },
        OptimizationType.IO: {
            "python": self._optimize_io_python,
            "javascript": self._optimize_io_js
        },
        OptimizationType.ALGORITHM: {
            "python": self._optimize_algorithm_python,
            "javascript": self._optimize_algorithm_js
        }
    }
    
    type_optimizers = optimizers.get(optimization_type, {})
    return type_optimizers.get(language)

def _optimize_memory_python(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply memory optimizations to Python code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "memory leak" in suggestion.description.lower():
        # Look for collection growth in loops
        if "append" in target_code or "extend" in target_code or "add" in target_code:
            # Add comment about potential memory leak
            indent = len(target_lines[0]) - len(target_lines[0].lstrip())
            indent_str = " " * indent
            
            # Add a fixed-size solution or clear collection periodically
            fixed_size_comment = f"{indent_str}# OPTIMIZATION: Avoid unbounded growth\n"
            fixed_size_comment += f"{indent_str}# Consider using collections.deque with maxlen or periodically clearing\n"
            fixed_size_comment += f"{indent_str}# Example: collection = collections.deque(maxlen=1000)\n"
            
            # Insert comment before the loop
            lines.insert(line_start, fixed_size_comment)
            
            # If imports are at the top, add collections import
            if not any("import collections" in line for line in lines[:20]):
                import_line = "import collections  # Added for deque optimization"
                # Find the last import line
                for i, line in enumerate(lines[:20]):
                    if "import " in line:
                        last_import = i
                
                # Insert after the last import or at the top
                lines.insert(last_import + 1 if 'last_import' in locals() else 0, import_line)
    
    elif "large object allocation" in suggestion.description.lower():
        # Add comment about lazy loading or pagination
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        lazy_load_comment = f"{indent_str}# OPTIMIZATION: Consider lazy loading/pagination\n"
        lazy_load_comment += f"{indent_str}# For large data, process in chunks or implement an iterator\n"
        
        # Insert comment before the allocation
        lines.insert(line_start, lazy_load_comment)
    
    return '\n'.join(lines)

def _optimize_cpu_python(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply CPU optimizations to Python code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "nested loop" in suggestion.description.lower():
        # Add comment about loop optimization
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        loop_opt_comment = f"{indent_str}# OPTIMIZATION: Nested loop optimization\n"
        loop_opt_comment += f"{indent_str}# Consider using vectorized operations or more efficient algorithms\n"
        
        if "numpy" in target_code:
            loop_opt_comment += f"{indent_str}# Use NumPy vectorization: result = np.outer(array1, array2)\n"
        else:
            loop_opt_comment += f"{indent_str}# Example: Use comprehension instead of nested loops\n"
            loop_opt_comment += f"{indent_str}# [func(x, y) for x in xs for y in ys]\n"
        
        # Insert comment before the loop
        lines.insert(line_start, loop_opt_comment)
    
    elif "expensive operation" in suggestion.description.lower():
        # Add comment about moving operation outside loop
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        # Extract operation to move
        for op in ["sort", "sorted", "deepcopy", "json.loads", "json.dumps"]:
            if op in target_code:
                operation = op
                break
        else:
            operation = "expensive_operation"
        
        opt_comment = f"{indent_str}# OPTIMIZATION: Move {operation} outside loop\n"
        opt_comment += f"{indent_str}# Calculate once before the loop if possible\n"
        
        # Insert comment before the operation
        for i in range(line_start, line_end + 1):
            if operation in lines[i]:
                lines.insert(i, opt_comment)
                break
    
    return '\n'.join(lines)

def _optimize_io_python(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply I/O optimizations to Python code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "file operation inside loop" in suggestion.description.lower():
        # Add comment about moving file operations outside loop
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        # Decide on specific optimization based on operation
        if "open(" in target_code:
            file_opt_comment = f"{indent_str}# OPTIMIZATION: Move file open/close outside loop\n"
            file_opt_comment += f"{indent_str}# Open the file once before the loop\n"
            file_opt_comment += f"{indent_str}# with open(filename) as f:\n"
            file_opt_comment += f"{indent_str}#     data = f.readlines()  # Read all at once\n"
            file_opt_comment += f"{indent_str}#     for line in data:  # Process in memory\n"
        elif "read(" in target_code or "write(" in target_code:
            file_opt_comment = f"{indent_str}# OPTIMIZATION: Batch read/write operations\n"
            file_opt_comment += f"{indent_str}# Collect data in memory, then read/write in batches\n"
            file_opt_comment += f"{indent_str}# buffer = []\n"
            file_opt_comment += f"{indent_str}# for item in items: buffer.append(process(item))\n"
            file_opt_comment += f"{indent_str}# f.write(''.join(buffer))  # Single write operation\n"
        else:
            file_opt_comment = f"{indent_str}# OPTIMIZATION: Optimize file operations\n"
            file_opt_comment += f"{indent_str}# Consider buffering or batch processing\n"
        
        # Insert comment before the loop
        for i in range(line_start, line_end + 1):
            if "open(" in lines[i] or "read(" in lines[i] or "write(" in lines[i]:
                lines.insert(i, file_opt_comment)
                break
    
    elif "multiple file operations" in suggestion.description.lower():
        # Add comment about connection pooling
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        pool_comment = f"{indent_str}# OPTIMIZATION: Reuse file handles\n"
        pool_comment += f"{indent_str}# Open file once and reuse the handle\n"
        pool_comment += f"{indent_str}# with open(filename, 'r') as f:\n"
        pool_comment += f"{indent_str}#     for operation in operations:\n"
        pool_comment += f"{indent_str}#         process_using_file_handle(f, operation)\n"
        
        # Insert comment before the first file operation
        lines.insert(line_start, pool_comment)
    
    return '\n'.join(lines)

def _optimize_algorithm_python(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply algorithm optimizations to Python code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "bubble sort" in suggestion.description.lower():
        # Replace bubble sort with built-in sort
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        sort_comment = f"{indent_str}# OPTIMIZATION: Replace bubble sort with built-in sort\n"
        sort_comment += f"{indent_str}# array.sort()  # or\n"
        sort_comment += f"{indent_str}# sorted_array = sorted(array)\n"
        
        # Keep the original code but comment it out
        commented_code = '\n'.join([f"{indent_str}# {line.lstrip()}" for line in target_lines])
        
        # Replace the bubble sort with the optimized version
        optimized_block = sort_comment + '\n' + commented_code
        
        # Replace the original lines
        for i in range(line_end, line_start - 1, -1):
            lines.pop(i)
        
        lines.insert(line_start, optimized_block)
    
    elif "string concatenation" in suggestion.description.lower():
        # Replace string concatenation with join
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        # Extract the variable being built
        var_match = re.search(r'(\w+)\s*\+=', target_code)
        if var_match:
            var_name = var_match.group(1)
            
            join_comment = f"{indent_str}# OPTIMIZATION: Use join() for string building\n"
            join_comment += f"{indent_str}# string_parts = []\n"
            join_comment += f"{indent_str}# for item in items:\n"
            join_comment += f"{indent_str}#     string_parts.append(str(item))\n"
            join_comment += f"{indent_str}# {var_name} = ''.join(string_parts)\n"
            
            # Insert comment before the string concatenation
            lines.insert(line_start, join_comment)
    
    elif "recursive function" in suggestion.description.lower():
        # Add memoization to recursive function
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        # Check if functools is already imported
        has_functools = any("import functools" in line for line in lines[:30])
        
        memo_comment = f"{indent_str}# OPTIMIZATION: Add memoization to recursive function\n"
        if not has_functools:
            memo_comment += f"import functools  # Added for memoization\n\n"
        
        memo_comment += f"{indent_str}@functools.lru_cache(maxsize=None)\n"
        
        # Insert before the function definition
        for i in range(line_start, line_end + 1):
            if "def " in lines[i]:
                lines.insert(i, memo_comment)
                break
    
    return '\n'.join(lines)

def _optimize_memory_js(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply memory optimizations to JavaScript code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "memory leak" in suggestion.description.lower():
        # Look for collection growth in loops
        if "push" in target_code or "concat" in target_code or "unshift" in target_code:
            # Add comment about potential memory leak
            indent = len(target_lines[0]) - len(target_lines[0].lstrip())
            indent_str = " " * indent
            
            # Add a fixed-size solution or periodically clearing
            fixed_size_comment = f"{indent_str}// OPTIMIZATION: Avoid unbounded growth\n"
            fixed_size_comment += f"{indent_str}// Consider using a fixed-size array or periodically clearing\n"
            fixed_size_comment += f"{indent_str}// Example: if (array.length > MAX_SIZE) array.length = 0;\n"
            
            # Insert comment before the loop
            lines.insert(line_start, fixed_size_comment)
    
    elif "large object allocation" in suggestion.description.lower():
        # Add comment about lazy loading or pagination
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        lazy_load_comment = f"{indent_str}// OPTIMIZATION: Consider lazy loading/pagination\n"
        lazy_load_comment += f"{indent_str}// For large data, process in chunks or implement a generator\n"
        
        # Insert comment before the allocation
        lines.insert(line_start, lazy_load_comment)
    
    return '\n'.join(lines)

def _optimize_cpu_js(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply CPU optimizations to JavaScript code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "dom access" in suggestion.description.lower():
        # Add comment about caching DOM elements
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        dom_opt_comment = f"{indent_str}// OPTIMIZATION: Cache DOM elements\n"
        dom_opt_comment += f"{indent_str}// Get elements once before the loop\n"
        dom_opt_comment += f"{indent_str}// const element = document.getElementById('id');\n"
        dom_opt_comment += f"{indent_str}// for (let i = 0; i < items.length; i++) {{ /* use element */ }}\n"
        
        # Insert comment before the loop
        lines.insert(line_start, dom_opt_comment)
    
    return '\n'.join(lines)

def _optimize_io_js(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply I/O optimizations to JavaScript code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "file operation inside loop" in suggestion.description.lower():
        # Add comment about batch operations
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        file_opt_comment = f"{indent_str}// OPTIMIZATION: Batch file operations\n"
        file_opt_comment += f"{indent_str}// Instead of multiple I/O calls, collect data and perform one operation\n"
        file_opt_comment += f"{indent_str}// const buffer = [];\n"
        file_opt_comment += f"{indent_str}// for (const item of items) {{ buffer.push(process(item)); }}\n"
        file_opt_comment += f"{indent_str}// fs.writeFile(filename, buffer.join('\\n'), callback);\n"
        
        # Insert comment before the loop
        lines.insert(line_start, file_opt_comment)
    
    return '\n'.join(lines)

def _optimize_algorithm_js(self, code: str, suggestion: OptimizationSuggestion) -> str:
    """Apply algorithm optimizations to JavaScript code"""
    lines = code.splitlines()
    line_start = suggestion.location.get("line_start", 1) - 1
    line_end = suggestion.location.get("line_end", line_start + 1) - 1
    
    if line_start < 0 or line_start >= len(lines) or line_end >= len(lines):
        return code
    
    # Get the target lines
    target_lines = lines[line_start:line_end+1]
    target_code = '\n'.join(target_lines)
    
    # Apply optimization based on description
    if "array manipulation" in suggestion.description.lower():
        # Replace imperative loops with functional methods
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        func_comment = f"{indent_str}// OPTIMIZATION: Use functional methods instead of loops\n"
        func_comment += f"{indent_str}// const result = array.map(item => transform(item));\n"
        func_comment += f"{indent_str}// const filtered = array.filter(item => condition(item));\n"
        func_comment += f"{indent_str}// const sum = array.reduce((acc, item) => acc + item, 0);\n"
        
        # Insert comment before the loop
        lines.insert(line_start, func_comment)
    
    elif "string concatenation" in suggestion.description.lower():
        # Replace string concatenation with join
        indent = len(target_lines[0]) - len(target_lines[0].lstrip())
        indent_str = " " * indent
        
        join_comment = f"{indent_str}// OPTIMIZATION: Use array join for string building\n"
        join_comment += f"{indent_str}// const parts = [];\n"
        join_comment += f"{indent_str}// for (const item of items) {{ parts.push(item); }}\n"
        join_comment += f"{indent_str}// const result = parts.join('');\n"
        
        # Insert comment before the string concatenation
        lines.insert(line_start, join_comment)
    
    return '\n'.join(lines)
