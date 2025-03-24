import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from enum import Enum, auto
import math
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ParadigmQuantumCircuit")

class ProgrammingParadigm(Enum):
    """Represents different programming paradigms"""
    OBJECT_ORIENTED = auto()
    FUNCTIONAL = auto()
    PROCEDURAL = auto()
    LOGIC = auto()
    DATAFLOW = auto()
    EVENT_DRIVEN = auto()
    CONCURRENT = auto()

@dataclass
class QuantumParadigmState:
    """Quantum state representation for programming paradigms"""
    # Core state vector (complex amplitudes)
    amplitudes: Dict[int, complex] = field(default_factory=dict)
    # Dimension of the quantum circuit
    dimension: int = 12
    # Entanglement map tracking qubit relationships
    entanglement_map: Dict[int, Set[int]] = field(default_factory=dict)
    # Phase shifts for each qubit
    phase_shifts: np.ndarray = None
    # Tracks paradigm-specific features
    paradigm_features: Dict[ProgrammingParadigm, float] = field(default_factory=dict)
    # Basis state meanings in the computational basis
    basis_meanings: Dict[int, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.phase_shifts is None:
            self.phase_shifts = np.zeros(self.dimension)
            
        # Initialize entanglement map if empty
        if not self.entanglement_map:
            self.entanglement_map = {i: set() for i in range(self.dimension)}
            
        # Setup default paradigm features
        if not self.paradigm_features:
            for paradigm in ProgrammingParadigm:
                self.paradigm_features[paradigm] = 0.0
                
        # Setup basis state meanings
        if not self.basis_meanings:
            # OOP-related states
            self.basis_meanings[0] = "class_definition"
            self.basis_meanings[1] = "method_call"
            self.basis_meanings[2] = "inheritance"
            self.basis_meanings[3] = "polymorphism"
            
            # Functional-related states
            self.basis_meanings[4] = "pure_function"
            self.basis_meanings[5] = "higher_order_function"
            self.basis_meanings[6] = "recursion"
            self.basis_meanings[7] = "immutability"
            
            # Other paradigm states
            self.basis_meanings[8] = "sequential_execution"
            self.basis_meanings[9] = "concurrent_execution"
            self.basis_meanings[10] = "event_handler"
            self.basis_meanings[11] = "data_flow"

class ParadigmQuantumCircuit:
    """Quantum circuit specialized for processing programming paradigms"""
    
    def __init__(self, dimension: int = 12):
        """Initialize the paradigm quantum circuit"""
        self.dimension = dimension
        # Initialize empty quantum state
        self.state = QuantumParadigmState(dimension=dimension)
        # Initialize specific gates for programming constructs
        self._init_paradigm_gates()
        # Track detected paradigms
        self.detected_paradigms = {}
        # Default gate list to apply
        self.gate_sequence = []
        
    def _init_paradigm_gates(self):
        """Initialize specialized gates for programming constructs"""
        # Create specialized gates dictionary
        self.paradigm_gates = {
            # Object-oriented gates
            "class_gate": np.array([
                [0.9, 0.1j],
                [0.1j, 0.9]
            ]),
            "inheritance_gate": np.array([
                [0.8, 0.2j],
                [0.2j, 0.8]
            ]),
            "polymorphism_gate": np.array([
                [0.7, 0.3j],
                [0.3j, 0.7]
            ]),
            
            # Functional gates
            "pure_function_gate": np.array([
                [0.7, -0.3j],
                [0.3j, 0.7]
            ]),
            "higher_order_gate": np.array([
                [0.6, -0.4j],
                [0.4j, 0.6]
            ]),
            "immutability_gate": np.array([
                [1.0, 0.0],
                [0.0, -1.0]
            ]),
            
            # Control flow gates
            "conditional_gate": np.array([
                [0.8, 0.2],
                [0.2, -0.8]
            ]),
            "loop_gate": np.array([
                [0.5, 0.5],
                [0.5, -0.5]
            ]),
            
            # Entanglement gates (for relationships between concepts)
            "CNOT": np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
        }
    
    def analyze_code_structure(self, code_elements: Dict[str, Any]) -> Dict[ProgrammingParadigm, float]:
        """
        Analyze code structure to determine programming paradigms
        
        Args:
            code_elements: Dictionary of code elements extracted from AST analysis
            
        Returns:
            Dictionary mapping paradigms to their probability scores
        """
        # Reset state
        self.state = QuantumParadigmState(dimension=self.dimension)
        
        # Prepare initial state based on code elements
        self._prepare_initial_state(code_elements)
        
        # Apply paradigm-specific gate sequences
        self._apply_paradigm_gates(code_elements)
        
        # Measure final state to determine paradigm probabilities
        paradigm_probs = self._measure_paradigms()
        
        # Update detected paradigms
        self.detected_paradigms = paradigm_probs
        
        return paradigm_probs
    
    def _prepare_initial_state(self, code_elements: Dict[str, Any]) -> None:
        """Prepare initial quantum state based on code elements"""
        # Reset amplitudes to zero state
        self.state.amplitudes = {0: complex(1.0, 0.0)}
        
        # Calculate counts of different elements
        class_count = len(code_elements.get('classes', []))
        function_count = len(code_elements.get('functions', []))
        method_count = sum(len(cls.get('methods', [])) for cls in code_elements.get('classes', []))
        inheritance_count = sum(1 for cls in code_elements.get('classes', []) if cls.get('inherits_from'))
        
        # Prepare superposition based on code elements
        total_elements = class_count + function_count + method_count + inheritance_count
        if total_elements > 0:
            # Initialize with equal distribution
            init_amplitude = 1.0 / math.sqrt(4)
            
            # Set amplitudes for key basis states
            if class_count > 0:
                self.state.amplitudes[0] = complex(init_amplitude, 0.0)  # class_definition
                
            if method_count > 0:
                self.state.amplitudes[1] = complex(init_amplitude, 0.0)  # method_call
                
            if inheritance_count > 0:
                self.state.amplitudes[2] = complex(init_amplitude, 0.0)  # inheritance
                
            if function_count > 0:
                self.state.amplitudes[4] = complex(init_amplitude, 0.0)  # pure_function
            
            # Normalize the state
            self._normalize_state()
    
    def _normalize_state(self) -> None:
        """Normalize the quantum state"""
        # Calculate total probability
        total_prob = sum(abs(amp)**2 for amp in self.state.amplitudes.values())
        
        # Normalize if non-zero
        if total_prob > 0:
            norm_factor = 1.0 / math.sqrt(total_prob)
            for basis_state in list(self.state.amplitudes.keys()):
                self.state.amplitudes[basis_state] *= norm_factor
    
    def _apply_paradigm_gates(self, code_elements: Dict[str, Any]) -> None:
        """Apply paradigm-specific gate sequences"""
        # Apply gates based on code elements
        
        # Object-oriented paradigm gates
        if code_elements.get('classes'):
            self._apply_oop_gates(code_elements)
            
        # Functional paradigm gates
        if self._has_functional_elements(code_elements):
            self._apply_functional_gates(code_elements)
            
        # Procedural paradigm gates
        if self._has_procedural_elements(code_elements):
            self._apply_procedural_gates(code_elements)
            
        # Check for concurrent elements
        if self._has_concurrent_elements(code_elements):
            self._apply_concurrent_gates(code_elements)
    
    def _apply_oop_gates(self, code_elements: Dict[str, Any]) -> None:
        """Apply object-oriented programming gates"""
        for class_info in code_elements.get('classes', []):
            # Apply class gate
            self._apply_single_qubit_gate(0, self.paradigm_gates["class_gate"])
            
            # Check for inheritance
            if class_info.get('inherits_from'):
                self._apply_single_qubit_gate(2, self.paradigm_gates["inheritance_gate"])
                # Entangle class and inheritance concepts
                self._apply_entanglement(0, 2)
            
            # Check for polymorphic methods
            method_names = set()
            for method in class_info.get('methods', []):
                if method['name'] in method_names:
                    # Potential method override (polymorphism)
                    self._apply_single_qubit_gate(3, self.paradigm_gates["polymorphism_gate"])
                    # Entangle class and polymorphism
                    self._apply_entanglement(0, 3)
                    break
                method_names.add(method['name'])
    
    def _apply_functional_gates(self, code_elements: Dict[str, Any]) -> None:
        """Apply functional programming gates"""
        functions = code_elements.get('functions', [])
        
        # Look for higher-order functions (functions that take functions as parameters)
        for func in functions:
            # Check for pure functions (no side effects)
            if self._is_pure_function(func):
                self._apply_single_qubit_gate(4, self.paradigm_gates["pure_function_gate"])
            
            # Check for higher-order functions
            if self._is_higher_order_function(func):
                self._apply_single_qubit_gate(5, self.paradigm_gates["higher_order_gate"])
                # Entangle pure functions and higher-order functions
                self._apply_entanglement(4, 5)
            
            # Check for recursion
            if self._is_recursive(func):
                # Apply recursion gate (Hadamard gate)
                self._apply_hadamard(6)
            
            # Check for immutability
            if self._uses_immutability(func):
                self._apply_single_qubit_gate(7, self.paradigm_gates["immutability_gate"])
    
    def _apply_procedural_gates(self, code_elements: Dict[str, Any]) -> None:
        """Apply procedural programming gates"""
        # Sequential execution
        self._apply_hadamard(8)
        
        # Apply conditional gate for if statements
        if_count = code_elements.get('if_statements', 0)
        if if_count > 0:
            self._apply_single_qubit_gate(8, self.paradigm_gates["conditional_gate"])
        
        # Apply loop gate for loops
        loop_count = code_elements.get('loops', 0)
        if loop_count > 0:
            self._apply_single_qubit_gate(8, self.paradigm_gates["loop_gate"])
    
    def _apply_concurrent_gates(self, code_elements: Dict[str, Any]) -> None:
        """Apply gates for concurrent programming patterns"""
        # Check for thread/async patterns
        if self._has_threading(code_elements) or self._has_async(code_elements):
            # Apply Hadamard to concurrent execution qubit
            self._apply_hadamard(9)
            
            # Entangle with sequential execution in superposition
            self._apply_entanglement(8, 9)
    
    def _apply_single_qubit_gate(self, target: int, gate: np.ndarray) -> None:
        """Apply a single-qubit gate to the target qubit"""
        new_amplitudes = {}
        
        # For each basis state in current superposition
        for basis_state, amplitude in list(self.state.amplitudes.items()):
            # Check if target bit is 0 or 1 in this basis state
            bit_val = (basis_state >> target) & 1
            
            # Calculate new basis states after gate application
            if bit_val == 0:
                # Apply gate[0,0] to |0⟩ and gate[0,1] to |1⟩
                basis_state_0 = basis_state  # Keep the same basis state (target bit is 0)
                basis_state_1 = basis_state | (1 << target)  # Flip target bit to 1
                
                # Update amplitudes
                new_amplitudes[basis_state_0] = new_amplitudes.get(basis_state_0, 0) + amplitude * gate[0, 0]
                new_amplitudes[basis_state_1] = new_amplitudes.get(basis_state_1, 0) + amplitude * gate[0, 1]
            else:
                # Apply gate[1,0] to |0⟩ and gate[1,1] to |1⟩
                basis_state_0 = basis_state & ~(1 << target)  # Flip target bit to 0
                basis_state_1 = basis_state  # Keep the same basis state (target bit is 1)
                
                # Update amplitudes
                new_amplitudes[basis_state_0] = new_amplitudes.get(basis_state_0, 0) + amplitude * gate[1, 0]
                new_amplitudes[basis_state_1] = new_amplitudes.get(basis_state_1, 0) + amplitude * gate[1, 1]
        
        # Filter out very small amplitudes to maintain sparsity
        self.state.amplitudes = {k: v for k, v in new_amplitudes.items() if abs(v) > 1e-10}
        
        # Update entanglement map
        self.state.entanglement_map[target] = set()  # Reset entanglements for this qubit
    
    def _apply_hadamard(self, target: int) -> None:
        """Apply Hadamard gate to create superposition"""
        new_amplitudes = {}
        norm_factor = 1.0 / math.sqrt(2.0)
        
        for basis_state, amplitude in list(self.state.amplitudes.items()):
            # Get the value of the target qubit
            bit_val = (basis_state >> target) & 1
            
            # Calculate the corresponding basis state with flipped target bit
            flipped_state = basis_state ^ (1 << target)
            
            if bit_val == 0:
                # |0⟩ -> (|0⟩ + |1⟩)/√2
                new_amplitudes[basis_state] = new_amplitudes.get(basis_state, 0) + amplitude * norm_factor
                new_amplitudes[flipped_state] = new_amplitudes.get(flipped_state, 0) + amplitude * norm_factor
            else:
                # |1⟩ -> (|0⟩ - |1⟩)/√2
                new_amplitudes[basis_state] = new_amplitudes.get(basis_state, 0) + amplitude * (-norm_factor)
                new_amplitudes[flipped_state] = new_amplitudes.get(flipped_state, 0) + amplitude * norm_factor
        
        # Filter out very small amplitudes
        self.state.amplitudes = {k: v for k, v in new_amplitudes.items() if abs(v) > 1e-10}
    
    def _apply_entanglement(self, control: int, target: int) -> None:
        """Apply entanglement operation between qubits (CNOT gate)"""
        new_amplitudes = {}
        
        for basis_state, amplitude in list(self.state.amplitudes.items()):
            # Get control bit value
            control_val = (basis_state >> control) & 1
            
            if control_val == 1:
                # Flip the target bit
                new_state = basis_state ^ (1 << target)
                new_amplitudes[new_state] = amplitude
            else:
                # Leave the state unchanged
                new_amplitudes[basis_state] = amplitude
        
        self.state.amplitudes = new_amplitudes
        
        # Update entanglement map
        self.state.entanglement_map[control].add(target)
        self.state.entanglement_map[target].add(control)
    
    def _measure_paradigms(self) -> Dict[ProgrammingParadigm, float]:
        """Measure the quantum state to determine paradigm probabilities"""
        paradigm_probs = {}
        
        # Object-oriented measurement
        oop_prob = self._measure_paradigm_probability([0, 1, 2, 3])
        paradigm_probs[ProgrammingParadigm.OBJECT_ORIENTED] = oop_prob
        
        # Functional measurement
        func_prob = self._measure_paradigm_probability([4, 5, 6, 7])
        paradigm_probs[ProgrammingParadigm.FUNCTIONAL] = func_prob
        
        # Procedural measurement
        proc_prob = self._measure_paradigm_probability([8])
        paradigm_probs[ProgrammingParadigm.PROCEDURAL] = proc_prob
        
        # Concurrent measurement
        conc_prob = self._measure_paradigm_probability([9])
        paradigm_probs[ProgrammingParadigm.CONCURRENT] = conc_prob
        
        # Event-driven measurement
        event_prob = self._measure_paradigm_probability([10])
        paradigm_probs[ProgrammingParadigm.EVENT_DRIVEN] = event_prob
        
        # Dataflow measurement
        data_prob = self._measure_paradigm_probability([11])
        paradigm_probs[ProgrammingParadigm.DATAFLOW] = data_prob
        
        return paradigm_probs
    
    def _measure_paradigm_probability(self, qubits: List[int]) -> float:
        """
        Measure the probability of a specific paradigm based on its qubits
        
        Args:
            qubits: List of qubit indices associated with this paradigm
            
        Returns:
            Probability value between 0 and 1
        """
        total_prob = 0.0
        
        # Sum probabilities for all basis states where any of the paradigm qubits are 1
        for basis_state, amplitude in self.state.amplitudes.items():
            # Check if any of the paradigm qubits are 1 in this basis state
            for qubit in qubits:
                if (basis_state >> qubit) & 1:
                    total_prob += abs(amplitude) ** 2
                    break
        
        return total_prob
    
    def get_specialized_circuit(self, paradigm: ProgrammingParadigm) -> "ParadigmQuantumCircuit":
        """
        Create a specialized circuit optimized for a specific paradigm
        
        Args:
            paradigm: The programming paradigm to specialize for
            
        Returns:
            Specialized quantum circuit
        """
        # Create a new circuit
        specialized = ParadigmQuantumCircuit(self.dimension)
        
        # Customize based on paradigm
        if paradigm == ProgrammingParadigm.OBJECT_ORIENTED:
            # Enhance OOP-related gates
            specialized.paradigm_gates["class_gate"] = np.array([
                [0.95, 0.05j],
                [0.05j, 0.95]
            ])
            specialized.paradigm_gates["inheritance_gate"] = np.array([
                [0.9, 0.1j],
                [0.1j, 0.9]
            ])
            
        elif paradigm == ProgrammingParadigm.FUNCTIONAL:
            # Enhance functional-related gates
            specialized.paradigm_gates["pure_function_gate"] = np.array([
                [0.9, -0.1j],
                [0.1j, 0.9]
            ])
            specialized.paradigm_gates["higher_order_gate"] = np.array([
                [0.8, -0.2j],
                [0.2j, 0.8]
            ])
            
        elif paradigm == ProgrammingParadigm.CONCURRENT:
            # Enhance concurrent-related operations
            specialized._apply_hadamard(9)
            specialized._apply_entanglement(8, 9)
        
        return specialized
    
    # Helper methods for detecting code patterns
    def _has_functional_elements(self, code_elements: Dict[str, Any]) -> bool:
        """Check if code has functional programming elements"""
        functions = code_elements.get('functions', [])
        
        # Check for higher-order functions, pure functions, etc.
        for func in functions:
            if (self._is_higher_order_function(func) or 
                self._is_pure_function(func) or 
                self._uses_immutability(func)):
                return True
                
        return False
    
    def _is_higher_order_function(self, func: Dict[str, Any]) -> bool:
        """Check if a function is higher-order (takes functions as args or returns them)"""
        # Check parameters for function types
        params = func.get('parameters', [])
        for param in params:
            param_type = param.get('type', '').lower()
            if ('function' in param_type or 'callable' in param_type or 
                'lambda' in param_type or 'callback' in param_type):
                return True
        
        # Check return type
        return_type = func.get('return_type', '').lower()
        return ('function' in return_type or 'callable' in return_type or 
                'lambda' in return_type)
    
    def _is_pure_function(self, func: Dict[str, Any]) -> bool:
        """Check if a function appears to be pure (no side effects)"""
        # This is a simplified heuristic
        # Real detection would need deeper code analysis
        side_effects = func.get('side_effects', False)
        return not side_effects
    
    def _is_recursive(self, func: Dict[str, Any]) -> bool:
        """Check if a function is recursive"""
        # Check if function calls itself
        function_name = func.get('name', '')
        calls = func.get('calls', [])
        return function_name in calls
    
    def _uses_immutability(self, func: Dict[str, Any]) -> bool:
        """Check if a function uses immutable data patterns"""
        # Check for immutable data usage
        return func.get('uses_immutable_data', False)
    
    def _has_procedural_elements(self, code_elements: Dict[str, Any]) -> bool:
        """Check if code has procedural elements"""
        # Check for sequential control flow statements
        if_count = code_elements.get('if_statements', 0)
        loop_count = code_elements.get('loops', 0)
        return if_count > 0 or loop_count > 0
    
    def _has_concurrent_elements(self, code_elements: Dict[str, Any]) -> bool:
        """Check if code has concurrent elements"""
        return (self._has_threading(code_elements) or 
                self._has_async(code_elements))
    
    def _has_threading(self, code_elements: Dict[str, Any]) -> bool:
        """Check if code uses threading"""
        # Look for threading imports or thread creation
        imports = code_elements.get('imports', [])
        for imp in imports:
            if 'thread' in imp.lower():
                return True
                
        # Check for thread creation statements
        return code_elements.get('creates_threads', False)
    
    def _has_async(self, code_elements: Dict[str, Any]) -> bool:
        """Check if code uses async/await patterns"""
        # Look for async keywords
        return (code_elements.get('has_async', False) or 
                code_elements.get('has_await', False))

class ProgrammingParadigmTranslator:
    """
    Translates code between different programming paradigms using 
    quantum-inspired transformation circuits
    """
    
    def __init__(self):
        # Create paradigm-specific circuits
        self.paradigm_circuits = {
            paradigm: ParadigmQuantumCircuit().get_specialized_circuit(paradigm)
            for paradigm in ProgrammingParadigm
        }
        
        # Initialize translation mappings
        self._init_translation_mappings()
    
    def _init_translation_mappings(self):
        """Initialize mappings between paradigm-specific constructs"""
        # Maps programming constructs between paradigms
        self.translation_mappings = {
            # OOP to Functional translation
            (ProgrammingParadigm.OBJECT_ORIENTED, ProgrammingParadigm.FUNCTIONAL): {
                "class": "module with pure functions",
                "method": "function with explicit self parameter",
                "inheritance": "composition and higher-order functions",
                "polymorphism": "function dispatch based on type",
                "encapsulation": "closure and module patterns",
                "this/self": "explicit parameter to functions"
            },
            
            # Functional to OOP translation
            (ProgrammingParadigm.FUNCTIONAL, ProgrammingParadigm.OBJECT_ORIENTED): {
                "pure function": "method with no side effects",
                "higher-order function": "strategy pattern or decorator",
                "closure": "private class members",
                "recursion": "method calling itself",
                "immutable data": "private fields with getters only"
            },
            
            # OOP to Procedural translation
            (ProgrammingParadigm.OBJECT_ORIENTED, ProgrammingParadigm.PROCEDURAL): {
                "class": "struct/record + related functions",
                "method": "function taking struct as first parameter",
                "inheritance": "nested structures",
                "polymorphism": "function pointers or switch/case dispatch"
            },
            
            # Procedural to OOP translation
            (ProgrammingParadigm.PROCEDURAL, ProgrammingParadigm.OBJECT_ORIENTED): {
                "global variable": "class static field",
                "struct/record": "class with properties",
                "function": "method on appropriate class",
                "function pointer": "interface implementation"
            }
        }
    
    def translate(self, 
                 code_elements: Dict[str, Any],
                 source_paradigm: ProgrammingParadigm,
                 target_paradigm: ProgrammingParadigm) -> Dict[str, Any]:
        """
        Translate code elements from source paradigm to target paradigm
        
        Args:
            code_elements: Dictionary of code elements to translate
            source_paradigm: Source programming paradigm
            target_paradigm: Target programming paradigm
            
        Returns:
            Translated code elements
        """
        # First analyze with source paradigm circuit
        source_circuit = self.paradigm_circuits[source_paradigm]
        source_circuit.analyze_code_structure(code_elements)
        
        # Translate code elements using quantum-inspired transformation
        translated_elements = self._quantum_transform(
            code_elements, 
            source_paradigm, 
            target_paradigm
        )
        
        # Analyze with target paradigm circuit for validation
        target_circuit = self.paradigm_circuits[target_paradigm]
        paradigm_probs = target_circuit.analyze_code_structure(translated_elements)
        
        # Add transformation confidence score
        target_paradigm_prob = paradigm_probs.get(target_paradigm, 0.0)
        translated_elements['transformation_confidence'] = target_paradigm_prob
        
        return translated_elements
    
    def _quantum_transform(self,
                          code_elements: Dict[str, Any],
                          source_paradigm: ProgrammingParadigm,
                          target_paradigm: ProgrammingParadigm) -> Dict[str, Any]:
        """
        Apply quantum-inspired transformation between paradigms
        
        Args:
            code_elements: Original code elements
            source_paradigm: Source paradigm
            target_paradigm: Target paradigm
            
        Returns:
            Transformed code elements
        """
        # Create a deep copy of code elements to transform
        transformed = dict(code_elements)
        
        # Look up mapping between these paradigms
        paradigm_key = (source_paradigm, target_paradigm)
        mapping = self.translation_mappings.get(paradigm_key, {})
        
        # Transform based on specific paradigm types
        if source_paradigm == ProgrammingParadigm.OBJECT_ORIENTED and target_paradigm == ProgrammingParadigm.FUNCTIONAL:
            transformed = self._transform_oop_to_functional(transformed, mapping)
        elif source_paradigm == ProgrammingParadigm.FUNCTIONAL and target_paradigm == ProgrammingParadigm.OBJECT_ORIENTED:
            transformed = self._transform_functional_to_oop(transformed, mapping)
        elif source_paradigm == ProgrammingParadigm.OBJECT_ORIENTED and target_paradigm == ProgrammingParadigm.PROCEDURAL:
            transformed = self._transform_oop_to_procedural(transformed, mapping)
        elif source_paradigm == ProgrammingParadigm.PROCEDURAL and target_paradigm == ProgrammingParadigm.OBJECT_ORIENTED:
            transformed = self._transform_procedural_to_oop(transformed, mapping)
        
        # Add translation mapping used
        transformed['paradigm_mapping'] = {
            k: v for k, v in mapping.items()
        }
        
        return transformed
    
    def _transform_oop_to_functional(self, 
                                     code_elements: Dict[str, Any],
                                     mapping: Dict[str, str]) -> Dict[str, Any]:
        """Transform OOP code to functional style"""
        transformed = dict(code_elements)
        
        # Transform classes to modules with pure functions
        transformed['modules'] = []
        transformed['functions'] = transformed.get('functions', [])
        
        for cls in code_elements.get('classes', []):
            # Create a module for each class
            module_name = cls['name']
            module = {
                'name': module_name,
                'functions': []
            }
            
            # Convert methods to pure functions with explicit self parameter
            for method in cls.get('methods', []):
                function_name = method['name']
                
                # Skip constructor (will handle specially)
                if function_name == '__init__' or function_name == 'constructor':
                    # Create factory function instead
                    factory_func = {
                        'name': f"create_{module_name.lower()}",
def _transform_oop_to_functional(self, 
                                 code_elements: Dict[str, Any],
                                 mapping: Dict[str, str]) -> Dict[str, Any]:
    """Transform OOP code to functional style"""
    transformed = dict(code_elements)
    
    # Transform classes to modules with pure functions
    transformed['modules'] = []
    transformed['functions'] = transformed.get('functions', [])
    
    for cls in code_elements.get('classes', []):
        # Create a module for each class
        module_name = cls['name']
        module = {
            'name': module_name,
            'functions': []
        }
        
        # Convert methods to pure functions with explicit self parameter
        for method in cls.get('methods', []):
            function_name = method['name']
            
            # Skip constructor (will handle specially)
            if function_name == '__init__' or function_name == 'constructor':
                # Create factory function instead
                factory_func = {
                    'name': f"create_{module_name.lower()}",
                    'parameters': method.get('parameters', [])[1:],  # Skip self
                    'return_type': module_name,
                    'is_pure': True,
                    'body': f"Creates a new {module_name} data structure"
                }
                module['functions'].append(factory_func)
            else:
                # Convert method to function with explicit self
                func = dict(method)
                # Ensure first parameter is data structure
                if not func.get('parameters'):
                    func['parameters'] = []
                
                if func['parameters'] and func['parameters'][0].get('name') == 'self':
                    # Rename self to something more explicit
                    func['parameters'][0]['name'] = module_name.lower()
                    func['parameters'][0]['type'] = module_name
                else:
                    # Add data structure parameter if not present
                    func['parameters'].insert(0, {
                        'name': module_name.lower(),
                        'type': module_name
                    })
                
                # Add to module functions
                module['functions'].append(func)
        
        # Handle inheritance through composition
        if cls.get('inherits_from'):
            parent_class = cls['inherits_from']
            # Add delegation functions for parent methods
            module['functions'].append({
                'name': f"get_{parent_class.lower()}",
                'parameters': [{
                    'name': module_name.lower(),
                    'type': module_name
                }],
                'return_type': parent_class,
                'is_pure': True,
                'body': f"Gets the {parent_class} component from {module_name}"
            })
        
        transformed['modules'].append(module)
    
    # We're transforming to functional, so remove classes
    transformed['classes'] = []
    
    return transformed

def _transform_functional_to_oop(self, 
                               code_elements: Dict[str, Any],
                               mapping: Dict[str, str]) -> Dict[str, Any]:
    """Transform functional code to OOP style"""
    transformed = dict(code_elements)
    
    # Initialize classes list
    transformed['classes'] = []
    
    # Group functions by their first parameter type
    function_groups = {}
    for func in code_elements.get('functions', []):
        # Skip functions with no parameters
        if not func.get('parameters'):
            continue
            
        # Get type of first parameter
        param_type = func['parameters'][0].get('type', '').strip()
        if param_type:
            if param_type not in function_groups:
                function_groups[param_type] = []
            function_groups[param_type].append(func)
    
    # Convert grouped functions to classes
    for param_type, funcs in function_groups.items():
        # Create a class
        cls = {
            'name': param_type,
            'methods': [],
            'properties': []
        }
        
        # Create constructor
        constructor = {
            'name': '__init__',
            'parameters': [{'name': 'self', 'type': 'self'}],
            'body': f"Initialize {param_type} instance"
        }
        cls['methods'].append(constructor)
        
        # Convert functions to methods
        for func in funcs:
            # Skip first parameter (becomes 'self')
            method = dict(func)
            method['parameters'] = [{'name': 'self', 'type': 'self'}] + func['parameters'][1:]
            cls['methods'].append(method)
        
        transformed['classes'].append(cls)
    
    # Handle modules as classes
    for module in code_elements.get('modules', []):
        # Create a class from module
        cls = {
            'name': module['name'],
            'methods': [],
            'properties': []
        }
        
        # Convert module functions to class methods
        for func in module.get('functions', []):
            method = dict(func)
            method['parameters'] = [{'name': 'self', 'type': 'self'}] + func.get('parameters', [])
            cls['methods'].append(method)
        
        transformed['classes'].append(cls)
    
    # Keep remaining functions that don't fit into classes
    remaining_funcs = []
    for func in code_elements.get('functions', []):
        if not func.get('parameters') or func not in function_groups.get(func['parameters'][0].get('type', ''), []):
            remaining_funcs.append(func)
    
    transformed['functions'] = remaining_funcs
    transformed['modules'] = []  # Remove modules as they're now classes
    
    return transformed

def _transform_oop_to_procedural(self, 
                               code_elements: Dict[str, Any],
                               mapping: Dict[str, str]) -> Dict[str, Any]:
    """Transform OOP code to procedural style"""
    transformed = dict(code_elements)
    
    # Create structures for classes
    transformed['structs'] = []
    transformed['functions'] = transformed.get('functions', [])
    
    for cls in code_elements.get('classes', []):
        # Create a struct for each class
        struct_name = cls['name']
        struct = {
            'name': struct_name,
            'fields': []
        }
        
        # Extract fields from constructor
        for method in cls.get('methods', []):
            if method['name'] == '__init__' or method['name'] == 'constructor':
                # Extract field assignments from constructor
                assignments = method.get('assignments', [])
                for assignment in assignments:
                    if assignment.get('target', '').startswith('self.'):
                        field_name = assignment['target'].split('.')[1]
                        struct['fields'].append({
                            'name': field_name,
                            'type': assignment.get('type', 'any')
                        })
        
        # Convert methods to functions
        for method in cls.get('methods', []):
            # Skip constructor
            if method['name'] == '__init__' or method['name'] == 'constructor':
                continue
                
            # Create function from method
            func_name = f"{struct_name}_{method['name']}"
            func = {
                'name': func_name,
                'parameters': [{'name': 'obj', 'type': struct_name}] + method.get('parameters', [])[1:],
                'return_type': method.get('return_type'),
                'body': method.get('body', '')
            }
            
            transformed['functions'].append(func)
        
        # Handle inheritance by nesting structures
        if cls.get('inherits_from'):
            parent_class = cls['inherits_from']
            struct['fields'].insert(0, {
                'name': parent_class.lower(),
                'type': parent_class
            })
        
        transformed['structs'].append(struct)
    
    # Remove classes
    transformed['classes'] = []
    
    return transformed

def _transform_procedural_to_oop(self, 
                               code_elements: Dict[str, Any],
                               mapping: Dict[str, str]) -> Dict[str, Any]:
    """Transform procedural code to OOP style"""
    transformed = dict(code_elements)
    
    # Initialize classes
    transformed['classes'] = []
    
    # Convert structs to classes
    for struct in code_elements.get('structs', []):
        cls = {
            'name': struct['name'],
            'methods': [],
            'properties': struct.get('fields', [])
        }
        
        # Create constructor
        constructor = {
            'name': '__init__',
            'parameters': [{'name': 'self', 'type': 'self'}],
            'body': f"Initialize {struct['name']} instance"
        }
        cls['methods'].append(constructor)
        
        transformed['classes'].append(cls)
    
    # Group functions by their first parameter type
    for func in code_elements.get('functions', []):
        # Skip functions with no parameters
        if not func.get('parameters'):
            continue
            
        # Get type of first parameter
        param_type = func['parameters'][0].get('type', '').strip()
        
        # Find matching class
        for cls in transformed['classes']:
            if cls['name'] == param_type:
                # Convert function to method
                method = dict(func)
                method['name'] = func['name'].replace(f"{param_type}_", "")
                method['parameters'] = [{'name': 'self', 'type': 'self'}] + func['parameters'][1:]
                cls['methods'].append(method)
                break
    
    # Keep remaining functions that don't belong to classes
    remaining_funcs = []
    for func in code_elements.get('functions', []):
        # Skip functions with no parameters
        if not func.get('parameters'):
            remaining_funcs.append(func)
            continue
            
        # Get type of first parameter
        param_type = func['parameters'][0].get('type', '').strip()
        
        # Keep function if no matching class exists
        if not any(cls['name'] == param_type for cls in transformed['classes']):
            remaining_funcs.append(func)
    
    transformed['functions'] = remaining_funcs
    transformed['structs'] = []  # Remove structs as they're now classes
    
    return transformed

def optimize_translation(self, 
                       code_elements: Dict[str, Any], 
                       source_paradigm: ProgrammingParadigm,
                       target_paradigm: ProgrammingParadigm) -> Dict[str, Any]:
    """
    Apply additional paradigm-specific optimizations to the translated code
    
    Args:
        code_elements: Translated code elements
        source_paradigm: Original programming paradigm
        target_paradigm: Target programming paradigm
        
    Returns:
        Optimized code elements
    """
    # Get specialized circuit for target paradigm
    target_circuit = self.paradigm_circuits[target_paradigm]
    
    # Analyze current paradigm distribution
    paradigm_probs = target_circuit.analyze_code_structure(code_elements)
    
    # Apply paradigm-specific optimizations
    if target_paradigm == ProgrammingParadigm.FUNCTIONAL:
        return self._optimize_for_functional(code_elements, paradigm_probs)
    elif target_paradigm == ProgrammingParadigm.OBJECT_ORIENTED:
        return self._optimize_for_oop(code_elements, paradigm_probs)
    elif target_paradigm == ProgrammingParadigm.PROCEDURAL:
        return self._optimize_for_procedural(code_elements, paradigm_probs)
    
    return code_elements

def _optimize_for_functional(self, 
                           code_elements: Dict[str, Any],
                           paradigm_probs: Dict[ProgrammingParadigm, float]) -> Dict[str, Any]:
    """Apply functional programming optimizations"""
    # Make functions more pure
    for func in code_elements.get('functions', []):
        func['is_pure'] = True
        func['side_effects'] = False
    
    # Add immutability flags
    code_elements['uses_immutable_data'] = True
    
    # Add higher-order function patterns
    if len(code_elements.get('functions', [])) > 1:
        # Create a higher-order function
        higher_order = {
            'name': 'compose',
            'parameters': [
                {'name': 'f', 'type': 'callable'},
                {'name': 'g', 'type': 'callable'}
            ],
            'return_type': 'callable',
            'is_pure': True,
            'body': 'Returns a new function that is the composition of f and g'
        }
        code_elements['functions'].append(higher_order)
    
    return code_elements

def _optimize_for_oop(self, 
                    code_elements: Dict[str, Any],
                    paradigm_probs: Dict[ProgrammingParadigm, float]) -> Dict[str, Any]:
    """Apply object-oriented programming optimizations"""
    # Ensure all classes have proper encapsulation
    for cls in code_elements.get('classes', []):
        # Add getter/setter methods for properties
        properties = cls.get('properties', [])
        methods = cls.get('methods', [])
        method_names = {m['name'] for m in methods}
        
        for prop in properties:
            # Add getter if not exists
            getter_name = f"get_{prop['name']}"
            if getter_name not in method_names:
                getter = {
                    'name': getter_name,
                    'parameters': [{'name': 'self', 'type': 'self'}],
                    'return_type': prop.get('type', 'any'),
                    'body': f"Get the {prop['name']} property"
                }
                methods.append(getter)
            
            # Add setter if not exists
            setter_name = f"set_{prop['name']}"
            if setter_name not in method_names:
                setter = {
                    'name': setter_name,
                    'parameters': [
                        {'name': 'self', 'type': 'self'},
                        {'name': 'value', 'type': prop.get('type', 'any')}
                    ],
                    'return_type': 'None',
                    'body': f"Set the {prop['name']} property"
                }
                methods.append(setter)
    
    return code_elements

def _optimize_for_procedural(self, 
                           code_elements: Dict[str, Any],
                           paradigm_probs: Dict[ProgrammingParadigm, float]) -> Dict[str, Any]:
    """Apply procedural programming optimizations"""
    # Add more control flow constructs
    code_elements['if_statements'] = code_elements.get('if_statements', 0) + 1
    code_elements['loops'] = code_elements.get('loops', 0) + 1
    
    # Ensure functions follow procedural naming conventions
    for func in code_elements.get('functions', []):
        # Add verb prefix if not present
        if not any(func['name'].startswith(verb) for verb in 
                  ['get', 'set', 'create', 'update', 'delete', 'process', 'calculate']):
            func['name'] = f"process_{func['name']}"
    
    return code_elements
