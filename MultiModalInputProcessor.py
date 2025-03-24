import numpy as np
import json
import re
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import hashlib
from enum import Enum, auto
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiModalInputProcessor")

class InputType(Enum):
    """Types of input that can be processed"""
    TEXT = auto()
    SKETCH = auto()
    CODE = auto()
    DIAGRAM = auto()
    AUDIO = auto()
    MIXED = auto()

@dataclass
class ProcessedInput:
    """Represents processed input in a unified format"""
    # Original input type
    input_type: InputType
    # Extracted intents from the input
    intents: List[str] = field(default_factory=list)
    # Structured representation of the input
    structure: Dict[str, Any] = field(default_factory=dict)
    # Confidence score of processing
    confidence: float = 0.0
    # Structured requirements extracted from input
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    # Abstract syntax tree (for code inputs)
    ast: Optional[Dict[str, Any]] = None
    # Vectorized representation for embedding
    embedding: Optional[np.ndarray] = None
    # Extracted entities (objects, concepts)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiModalInputProcessor:
    """
    Processes multi-modal inputs (text, sketches, code, diagrams) 
    and converts them to a unified format for software generation
    """
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize the multi-modal input processor"""
        self.embedding_dim = embedding_dim
        self._init_processors()
        
        # Initialize vectorizers
        self.text_vectorizer = self._create_text_vectorizer()
        self.code_vectorizer = self._create_code_vectorizer()
        self.diagram_vectorizer = self._create_diagram_vectorizer()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _init_processors(self):
        """Initialize specialized processors for each input type"""
        self.processors = {
            InputType.TEXT: self._process_text,
            InputType.CODE: self._process_code,
            InputType.SKETCH: self._process_sketch,
            InputType.DIAGRAM: self._process_diagram,
            InputType.AUDIO: self._process_audio,
            InputType.MIXED: self._process_mixed
        }
    
    def detect_input_type(self, input_data: Dict[str, Any]) -> InputType:
        """
        Detect the type of input
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Detected input type
        """
        # Check if explicitly specified
        if 'type' in input_data:
            type_str = input_data['type'].upper()
            if hasattr(InputType, type_str):
                return getattr(InputType, type_str)
        
        # Check content for text
        if 'text' in input_data and isinstance(input_data['text'], str) and len(input_data['text']) > 0:
            # Check if it contains code markers
            if self._contains_code_markers(input_data['text']):
                return InputType.CODE
            return InputType.TEXT
        
        # Check for base64 encoded image
        if 'image' in input_data and isinstance(input_data['image'], str) and input_data['image'].startswith('data:image'):
            # If it has structured lines, it's likely a diagram
            if 'structured' in input_data and input_data['structured']:
                return InputType.DIAGRAM
            return InputType.SKETCH
        
        # Check for audio
        if 'audio' in input_data:
            return InputType.AUDIO
        
        # If multiple input modes are present
        if len(input_data) > 1:
            return InputType.MIXED
        
        # Default to text
        return InputType.TEXT
    
    def process_input(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """
        Process input data and convert to unified format
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Processed input in unified format
        """
        # Detect input type
        input_type = self.detect_input_type(input_data)
        logger.info(f"Detected input type: {input_type.name}")
        
        # Process using appropriate processor
        processor = self.processors[input_type]
        processed = processor(input_data)
        
        # Ensure processed input has embeddings
        if processed.embedding is None:
            processed.embedding = self._create_embedding(processed, input_type)
        
        # Extract requirements if not already present
        if not processed.requirements:
            processed.requirements = self._extract_requirements(processed)
        
        return processed
    
    def _process_text(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process natural language text input"""
        text = input_data.get('text', '')
        
        # Create initial processed input
        processed = ProcessedInput(
            input_type=InputType.TEXT,
            confidence=1.0
        )
        
        # Extract intents using rule-based and keyword approach
        processed.intents = self._extract_intents(text)
        
        # Extract entities (objects, actions, etc.)
        processed.entities = self._extract_entities_from_text(text)
        
        # Create structure based on extracted entities
        processed.structure = self._create_structure_from_entities(processed.entities)
        
        # Create embedding
        processed.embedding = self.text_vectorizer.transform([text])[0]
        
        return processed
    
    def _process_code(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process code input"""
        # Extract code from input data
        if 'code' in input_data:
            code = input_data['code']
        elif 'text' in input_data:
            # Extract code from text
            code = self._extract_code_from_text(input_data['text'])
        else:
            code = ""
        
        # Detect language
        language = input_data.get('language', self._detect_language(code))
        
        # Create processed input
        processed = ProcessedInput(
            input_type=InputType.CODE,
            confidence=0.95 if code else 0.5
        )
        
        # Parse code to AST if non-empty
        if code:
            processed.ast = self._parse_code_to_ast(code, language)
            
            # Extract structure from AST
            processed.structure = self._extract_structure_from_ast(processed.ast)
            
            # Create embedding using code vectorizer
            processed.embedding = self.code_vectorizer.transform([code])[0]
        
        # Extract intents
        processed.intents = ["code_analysis"] if code else ["code_generation"]
        
        # Set metadata
        processed.metadata = {
            "language": language,
            "code_length": len(code) if code else 0,
            "has_comments": "/*" in code or "//" in code or "#" in code if code else False
        }
        
        return processed
    
    def _process_sketch(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process sketch/drawing input"""
        # Get image data
        image_data = input_data.get('image', '')
        
        # Create processed input
        processed = ProcessedInput(
            input_type=InputType.SKETCH,
            confidence=0.85
        )
        
        # Extract visual elements (shapes, connections, text)
        visual_elements = self._extract_visual_elements(image_data)
        
        # Convert visual elements to entities
        processed.entities = self._convert_visual_to_entities(visual_elements)
        
        # Create structure from entities
        processed.structure = self._create_structure_from_entities(processed.entities)
        
        # Determine intents based on sketch type
        if self._is_ui_sketch(visual_elements):
            processed.intents = ["ui_generation"]
        elif self._is_flowchart(visual_elements):
            processed.intents = ["workflow_generation"]
        elif self._is_architecture_diagram(visual_elements):
            processed.intents = ["architecture_generation"]
        else:
            processed.intents = ["visual_to_code"]
        
        # Create embedding for the image (using simplified features)
        processed.embedding = self._extract_sketch_features(visual_elements)
        
        return processed
    
    def _process_diagram(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process structured diagram input"""
        # Get diagram data
        image_data = input_data.get('image', '')
        diagram_type = input_data.get('diagram_type', 'unknown')
        
        # Create processed input
        processed = ProcessedInput(
            input_type=InputType.DIAGRAM,
            confidence=0.9
        )
        
        # Extract structural elements based on diagram type
        if diagram_type.lower() == 'uml_class':
            processed.structure = self._extract_uml_class_diagram(image_data)
            processed.intents = ["class_generation"]
        elif diagram_type.lower() == 'uml_sequence':
            processed.structure = self._extract_uml_sequence_diagram(image_data)
            processed.intents = ["interaction_generation"]
        elif diagram_type.lower() == 'er_diagram':
            processed.structure = self._extract_er_diagram(image_data)
            processed.intents = ["database_generation"]
        elif diagram_type.lower() == 'flowchart':
            processed.structure = self._extract_flowchart(image_data)
            processed.intents = ["algorithm_generation"]
        else:
            # Generic diagram processing
            elements = self._extract_visual_elements(image_data)
            processed.structure = self._interpret_diagram_elements(elements)
            processed.intents = ["structural_generation"]
        
        # Extract entities from structure
        processed.entities = self._extract_entities_from_structure(processed.structure)
        
        # Create embedding
        processed.embedding = self.diagram_vectorizer.transform([json.dumps(processed.structure)])[0]
        
        return processed
    
    def _process_audio(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process audio input (converting to text first)"""
        # This would typically use a speech-to-text system
        # For CPU-only implementation, we'll use a simplified approach
        
        # Get audio data or transcript
        transcript = input_data.get('transcript', '')
        
        if not transcript and 'audio' in input_data:
            # In a real implementation, this would process audio data
            # Here we'll just use placeholder text
            transcript = "Generated transcript from audio input"
        
        # Create text input from transcript
        text_input = {'text': transcript}
        
        # Process as text
        processed = self._process_text(text_input)
        
        # Change input type to audio
        processed.input_type = InputType.AUDIO
        processed.metadata['original_type'] = 'audio'
        
        return processed
    
    def _process_mixed(self, input_data: Dict[str, Any]) -> ProcessedInput:
        """Process mixed input types by processing each separately and combining"""
        processed_inputs = []
        
        # Process each input type separately
        if 'text' in input_data:
            text_processed = self._process_text({'text': input_data['text']})
            processed_inputs.append(text_processed)
        
        if 'code' in input_data:
            code_processed = self._process_code({'code': input_data['code']})
            processed_inputs.append(code_processed)
        
        if 'image' in input_data:
            if input_data.get('is_diagram', False):
                diagram_processed = self._process_diagram({'image': input_data['image']})
                processed_inputs.append(diagram_processed)
            else:
                sketch_processed = self._process_sketch({'image': input_data['image']})
                processed_inputs.append(sketch_processed)
        
        if 'audio' in input_data or 'transcript' in input_data:
            audio_processed = self._process_audio({
                'audio': input_data.get('audio', ''),
                'transcript': input_data.get('transcript', '')
            })
            processed_inputs.append(audio_processed)
        
        # Combine processed inputs
        combined = self._combine_processed_inputs(processed_inputs)
        combined.input_type = InputType.MIXED
        
        return combined
    
    def _combine_processed_inputs(self, processed_inputs: List[ProcessedInput]) -> ProcessedInput:
        """Combine multiple processed inputs into a single unified representation"""
        if not processed_inputs:
            return ProcessedInput(input_type=InputType.MIXED)
        
        # Start with the first processed input
        combined = ProcessedInput(
            input_type=InputType.MIXED,
            confidence=0.0,
            intents=[],
            entities=[],
            requirements=[]
        )
        
        # Combine all processed inputs
        for processed in processed_inputs:
            # Combine intents
            for intent in processed.intents:
                if intent not in combined.intents:
                    combined.intents.append(intent)
            
            # Combine entities (avoiding duplicates)
            entity_names = {e.get('name', '') for e in combined.entities}
            for entity in processed.entities:
                if entity.get('name', '') not in entity_names:
                    combined.entities.append(entity)
                    entity_names.add(entity.get('name', ''))
            
            # Combine requirements
            req_texts = {r.get('description', '') for r in combined.requirements}
            for req in processed.requirements:
                if req.get('description', '') not in req_texts:
                    combined.requirements.append(req)
                    req_texts.add(req.get('description', ''))
            
            # Update confidence
            combined.confidence = max(combined.confidence, processed.confidence)
        
        # Create combined structure
        combined.structure = self._create_structure_from_entities(combined.entities)
        
        # Create combined embedding
        embeddings = [p.embedding for p in processed_inputs if p.embedding is not None]
        if embeddings:
            combined.embedding = np.mean(embeddings, axis=0)
        
        return combined
    
    def _extract_intents(self, text: str) -> List[str]:
        """Extract intents from text using keyword matching"""
        intents = []
        
        # Define intent patterns
        intent_patterns = {
            'application_generation': [
                r'create\s+(?:an?|the)\s+app',
                r'build\s+(?:an?|the)\s+application',
                r'generate\s+(?:an?|the)\s+program',
                r'develop\s+(?:an?|the)\s+software'
            ],
            'code_generation': [
                r'write\s+(?:some|the)?\s*code',
                r'generate\s+code',
                r'implement\s+(?:an?|the)?'
            ],
            'ui_generation': [
                r'create\s+(?:an?|the)\s+(?:user\s+)?interface',
                r'design\s+(?:an?|the)\s+ui',
                r'build\s+(?:an?|the)\s+front(?:\s*|-)end'
            ],
            'database_generation': [
                r'create\s+(?:an?|the)\s+database',
                r'design\s+(?:an?|the)\s+data\s*model',
                r'set\s*up\s+(?:an?|the)\s+db'
            ],
            'api_generation': [
                r'create\s+(?:an?|the)\s+api',
                r'build\s+(?:an?|the)\s+(?:rest|graphql|web)\s+api',
                r'implement\s+(?:an?|the)\s+endpoint'
            ]
        }
        
        # Check each intent pattern
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    intents.append(intent)
                    break
        
        # If no specific intent is found, use generic app
def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
    """Extract entities (objects, concepts) from text"""
    entities = []
    
    # Simple entity extraction using regex patterns for various object types
    patterns = {
        "class": r"(?:a|an|the)\s+(\w+)\s+(?:class|object)",
        "function": r"(?:a|an|the)\s+(\w+)\s+(?:function|method)",
        "database": r"(?:a|an|the)\s+(\w+)\s+(?:table|database)",
        "ui_component": r"(?:a|an|the)\s+(\w+)\s+(?:button|field|form|page|screen)",
        "data_type": r"(?:a|an|the)\s+(\w+)\s+(?:string|integer|float|boolean|array)"
    }
    
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entity_name = match.group(1)
            entities.append({
                "name": entity_name,
                "type": entity_type,
                "source": "text"
            })
    
    return entities

def _create_structure_from_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert entities to a structural representation"""
    structure = {
        "classes": [],
        "functions": [],
        "data_models": [],
        "ui_components": []
    }
    
    for entity in entities:
        entity_type = entity.get("type", "")
        
        if entity_type == "class":
            structure["classes"].append({
                "name": entity.get("name", ""),
                "methods": [],
                "attributes": []
            })
        elif entity_type == "function":
            structure["functions"].append({
                "name": entity.get("name", ""),
                "parameters": []
            })
        elif entity_type == "database":
            structure["data_models"].append({
                "name": entity.get("name", ""),
                "fields": []
            })
        elif entity_type == "ui_component":
            structure["ui_components"].append({
                "name": entity.get("name", ""),
                "type": "component"
            })
    
    return structure

def _extract_requirements(self, processed: ProcessedInput) -> List[Dict[str, Any]]:
    """Extract functional and non-functional requirements"""
    requirements = []
    
    # Extract from text or structure based on input type
    if processed.input_type == InputType.TEXT:
        # Extract requirements from text using patterns
        if hasattr(processed, 'structure') and isinstance(processed.structure, dict):
            text = processed.metadata.get('original_text', '')
            
            # Functional requirements patterns
            func_patterns = [
                r"system\s+(?:shall|should|must|will)\s+([^\.]+)",
                r"user\s+(?:shall|should|must|will)\s+be\s+able\s+to\s+([^\.]+)",
                r"application\s+(?:shall|should|must|will)\s+([^\.]+)"
            ]
            
            # Non-functional requirements patterns
            nonfunc_patterns = [
                r"(?:performance|security|reliability|usability|scalability)\s+(?:requirement|needs):\s*([^\.]+)",
                r"system\s+(?:shall|should|must|will)\s+be\s+(fast|secure|reliable|user-friendly|scalable)"
            ]
            
            # Extract functional requirements
            for pattern in func_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    requirements.append({
                        "type": "functional",
                        "description": match.group(1).strip(),
                        "priority": "medium",
                        "source": "text"
                    })
            
            # Extract non-functional requirements
            for pattern in nonfunc_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    requirements.append({
                        "type": "non-functional",
                        "description": match.group(1).strip(),
                        "priority": "high",
                        "source": "text"
                    })
    
    # Add default requirements based on intent
    if processed.intents:
        if "application_generation" in processed.intents:
            requirements.append({
                "type": "functional",
                "description": "The system shall provide a complete application",
                "priority": "high",
                "source": "intent"
            })
        
        if "ui_generation" in processed.intents:
            requirements.append({
                "type": "non-functional",
                "description": "The user interface shall be intuitive and responsive",
                "priority": "high",
                "source": "intent"
            })
    
    return requirements

def _create_embedding(self, processed: ProcessedInput, input_type: InputType) -> np.ndarray:
    """Create vector embedding for any processed input"""
    # Create a feature dictionary 
    features = {}
    
    # Add intents as features
    for intent in processed.intents:
        features[f"intent_{intent}"] = 1.0
    
    # Add entity types as features
    entity_types = {}
    for entity in processed.entities:
        entity_type = entity.get("type", "unknown")
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    
    for entity_type, count in entity_types.items():
        features[f"entity_type_{entity_type}"] = count
    
    # Add requirement types as features
    req_types = {}
    for req in processed.requirements:
        req_type = req.get("type", "unknown")
        req_types[req_type] = req_types.get(req_type, 0) + 1
    
    for req_type, count in req_types.items():
        features[f"requirement_type_{req_type}"] = count
    
    # Create simple vector from features
    vector = np.zeros(self.embedding_dim)
    
    # Hash features to indices
    for feature, value in features.items():
        idx = int(hashlib.md5(feature.encode()).hexdigest(), 16) % self.embedding_dim
        vector[idx] = value
    
    # Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector

def _contains_code_markers(self, text: str) -> bool:
    """Check if text contains code markers"""
    code_markers = [
        '```', 'def ', 'class ', 'function ', 'import ', 
        'var ', 'const ', 'let ', 'public class', '#include'
    ]
    
    for marker in code_markers:
        if marker in text:
            return True
    
    return False

def _create_text_vectorizer(self):
    """Create a simple vectorizer for text"""
    class SimpleVectorizer:
        def __init__(self, dim=128):
            self.dim = dim
        
        def transform(self, texts):
            vectors = []
            for text in texts:
                # Create a simple bag-of-words vector
                vector = np.zeros(self.dim)
                words = re.findall(r'\w+', text.lower())
                for word in words:
                    # Hash the word to an index
                    idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                    vector[idx] += 1
                
                # Normalize
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                vectors.append(vector)
            
            return np.array(vectors)
    
    return SimpleVectorizer(dim=self.embedding_dim)

def _create_code_vectorizer(self):
    """Create a simple vectorizer for code"""
    class CodeVectorizer:
        def __init__(self, dim=128):
            self.dim = dim
        
        def transform(self, code_samples):
            vectors = []
            for code in code_samples:
                vector = np.zeros(self.dim)
                
                # Extract patterns that indicate code structure
                patterns = {
                    'function': len(re.findall(r'(def|function)\s+\w+\s*\(', code)),
                    'class': len(re.findall(r'class\s+\w+', code)),
                    'loop': len(re.findall(r'(for|while)\s*\(', code)),
                    'condition': len(re.findall(r'if\s*\(', code)),
                    'import': len(re.findall(r'(import|require|include)', code)),
                    'assignment': len(re.findall(r'=', code)),
                    'comment': len(re.findall(r'(#|//|/\*)', code))
                }
                
                # Map features to vector indices
                for feature, count in patterns.items():
                    idx = int(hashlib.md5(feature.encode()).hexdigest(), 16) % self.dim
                    vector[idx] = min(count / 10.0, 1.0)  # Normalize to 0-1
                
                # Add language features if detectable
                if 'def ' in code or 'import ' in code:
                    # Python indicators
                    idx = int(hashlib.md5(b'lang_python').hexdigest(), 16) % self.dim
                    vector[idx] = 1.0
                elif 'function ' in code or 'var ' in code or 'const ' in code or 'let ' in code:
                    # JavaScript indicators
                    idx = int(hashlib.md5(b'lang_javascript').hexdigest(), 16) % self.dim
                    vector[idx] = 1.0
                
                # Normalize
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                vectors.append(vector)
            
            return np.array(vectors)
    
    return CodeVectorizer(dim=self.embedding_dim)

def _create_diagram_vectorizer(self):
    """Create a simple vectorizer for diagrams"""
    class DiagramVectorizer:
        def __init__(self, dim=128):
            self.dim = dim
        
        def transform(self, diagrams):
            vectors = []
            for diagram_json in diagrams:
                vector = np.zeros(self.dim)
                
                try:
                    # Parse diagram JSON
                    diagram = json.loads(diagram_json) if isinstance(diagram_json, str) else diagram_json
                    
                    # Count elements by type
                    element_counts = {}
                    def count_elements(obj):
                        if isinstance(obj, dict):
                            element_type = obj.get('type', 'unknown')
                            element_counts[element_type] = element_counts.get(element_type, 0) + 1
                            for value in obj.values():
                                count_elements(value)
                        elif isinstance(obj, list):
                            for item in obj:
                                count_elements(item)
                    
                    count_elements(diagram)
                    
                    # Set vector values based on element counts
                    for element_type, count in element_counts.items():
                        idx = int(hashlib.md5(element_type.encode()).hexdigest(), 16) % self.dim
                        vector[idx] = min(count / 5.0, 1.0)  # Normalize to 0-1
                    
                    # Add diagram type features if available
                    if 'class' in element_counts:
                        idx = int(hashlib.md5(b'diagram_uml_class').hexdigest(), 16) % self.dim
                        vector[idx] = 1.0
                    elif 'entity' in element_counts:
                        idx = int(hashlib.md5(b'diagram_er').hexdigest(), 16) % self.dim
                        vector[idx] = 1.0
                    elif 'sequence' in element_counts:
                        idx = int(hashlib.md5(b'diagram_sequence').hexdigest(), 16) % self.dim
                        vector[idx] = 1.0
                
                except (json.JSONDecodeError, TypeError):
                    pass
                
                # Normalize
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                vectors.append(vector)
            
            return np.array(vectors)
    
    return DiagramVectorizer(dim=self.embedding_dim)
def _extract_code_from_text(self, text: str) -> str:
    """Extract code blocks from text"""
    # Look for code blocks with backticks
    code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        return '\n\n'.join(code_blocks)
    
    # Look for indented code blocks
    lines = text.split('\n')
    indented_blocks = []
    current_block = []
    in_block = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            in_block = True
            current_block.append(line.lstrip())
        elif in_block and line.strip() == '':
            # Empty line within a block
            current_block.append('')
        elif in_block:
            # End of block
            indented_blocks.append('\n'.join(current_block))
            current_block = []
            in_block = False
    
    # Add last block if exists
    if current_block:
        indented_blocks.append('\n'.join(current_block))
    
    if indented_blocks:
        return '\n\n'.join(indented_blocks)
    
    # If no code blocks found, look for lines that might be code
    code_indicators = ['def ', 'class ', 'import ', 'function ', 'var ', 'const ', 'let ', 'if ', 'for ']
    potential_code_lines = []
    
    for line in lines:
        if any(line.strip().startswith(indicator) for indicator in code_indicators):
            potential_code_lines.append(line)
    
    if potential_code_lines:
        return '\n'.join(potential_code_lines)
    
    return ""

def _detect_language(self, code: str) -> str:
    """Detect the programming language of the code"""
    language_markers = {
        'python': ['def ', 'import ', 'from ', 'class ', '    ', '#'],
        'javascript': ['function ', 'const ', 'let ', 'var ', 'export ', 'import ', '() => {'],
        'java': ['public class', 'private ', 'protected ', 'import java', 'void ', '@Override'],
        'c': ['#include', 'int main', 'void ', 'struct ', 'char *', 'printf'],
        'cpp': ['#include', 'namespace', 'template<', 'std::', 'void ', 'class '],
        'csharp': ['namespace ', 'using ', 'public class', 'private ', 'protected ', 'void '],
        'go': ['package ', 'import (', 'func ', 'type ', 'struct {', 'interface {'],
        'ruby': ['def ', 'require ', 'class ', 'module ', 'attr_', 'end'],
        'php': ['<?php', 'function ', 'public function', '$', '->', '::'],
        'swift': ['import ', 'class ', 'struct ', 'let ', 'var ', 'func ']
    }
    
    language_scores = {lang: 0 for lang in language_markers}
    
    for language, markers in language_markers.items():
        for marker in markers:
            if marker in code:
                language_scores[language] += 1
    
    # Get the language with the highest score
    max_score = 0
    detected_language = 'unknown'
    
    for language, score in language_scores.items():
        if score > max_score:
            max_score = score
            detected_language = language
    
    return detected_language

def _parse_code_to_ast(self, code: str, language: str) -> Dict[str, Any]:
    """Parse code into an abstract syntax tree representation"""
    # This would typically use language-specific parsers
    # For this CPU-only implementation, we'll use regex-based parsing
    
    ast = {
        "type": "Program",
        "body": [],
        "language": language
    }
    
    if language == "python":
        # Extract classes
        class_matches = re.finditer(r'class\s+(\w+)(?:\(([\w,\s]+)\))?:(.*?)(?=\n\S|\Z)', code, re.DOTALL)
        for match in class_matches:
            class_name = match.group(1)
            inheritance = match.group(2) or ""
            class_body = match.group(3)
            
            # Extract methods
            method_matches = re.finditer(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([\w\[\],\s]+))?\s*:(.*?)(?=\n\s+def|\n\S|\Z)', class_body, re.DOTALL)
            methods = []
            
            for method_match in method_matches:
                method_name = method_match.group(1)
                params = method_match.group(2)
                return_type = method_match.group(3) or ""
                method_body = method_match.group(4)
                
                methods.append({
                    "type": "Method",
                    "name": method_name,
                    "params": self._parse_parameters(params),
                    "return_type": return_type.strip(),
                    "body": method_body.strip()
                })
            
            ast["body"].append({
                "type": "ClassDeclaration",
                "name": class_name,
                "superClass": inheritance.split(",") if inheritance else [],
                "methods": methods
            })
        
        # Extract functions
        func_matches = re.finditer(r'def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([\w\[\],\s]+))?\s*:(.*?)(?=\n(?:def|class)|\Z)', code, re.DOTALL)
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            return_type = match.group(3) or ""
            func_body = match.group(4)
            
            ast["body"].append({
                "type": "FunctionDeclaration",
                "name": func_name,
                "params": self._parse_parameters(params),
                "return_type": return_type.strip(),
                "body": func_body.strip()
            })
            
        # Extract imports
        import_matches = re.finditer(r'(?:from\s+([\w.]+)\s+)?import\s+([\w*,\s]+)(?:\s+as\s+(\w+))?', code)
        for match in import_matches:
            module = match.group(1) or ""
            imports = match.group(2).split(",")
            alias = match.group(3) or ""
            
            ast["body"].append({
                "type": "ImportStatement",
                "module": module.strip(),
                "imports": [imp.strip() for imp in imports],
                "alias": alias.strip()
            })
            
    elif language in ["javascript", "typescript"]:
        # Extract functions
        func_matches = re.finditer(r'function\s+(\w+)\s*\((.*?)\)\s*{(.*?)}', code, re.DOTALL)
        for match in func_matches:
            func_name = match.group(1)
            params = match.group(2)
            func_body = match.group(3)
            
            ast["body"].append({
                "type": "FunctionDeclaration",
                "name": func_name,
                "params": params.split(","),
                "body": func_body.strip()
            })
        
        # Extract arrow functions
        arrow_matches = re.finditer(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\((.*?)\)|(\w+))\s*=>\s*(?:{(.*?)}|(.+?)(?:;|\n))', code, re.DOTALL)
        for match in arrow_matches:
            func_name = match.group(1)
            params = match.group(2) or match.group(3) or ""
            func_body = match.group(4) or match.group(5) or ""
            
            ast["body"].append({
                "type": "ArrowFunction",
                "name": func_name,
                "params": params.split(",") if params else [],
                "body": func_body.strip()
            })
        
        # Extract classes
        class_matches = re.finditer(r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{(.*?)}', code, re.DOTALL)
        for match in class_matches:
            class_name = match.group(1)
            inheritance = match.group(2) or ""
            class_body = match.group(3)
            
            # Extract methods
            method_matches = re.finditer(r'(?:async\s+)?(?:static\s+)?(\w+)\s*\((.*?)\)\s*{(.*?)}', class_body, re.DOTALL)
            methods = []
            
            for method_match in method_matches:
                method_name = method_match.group(1)
                params = method_match.group(2)
                method_body = method_match.group(3)
                
                methods.append({
                    "type": "Method",
                    "name": method_name,
                    "params": params.split(",") if params else [],
                    "body": method_body.strip()
                })
            
            ast["body"].append({
                "type": "ClassDeclaration",
                "name": class_name,
                "superClass": inheritance if inheritance else "",
                "methods": methods
            })
            
        # Extract imports
        import_matches = re.finditer(r'import\s+(?:{(.*?)}|(\w+))\s+from\s+[\'"](.+?)[\'"]', code, re.DOTALL)
        for match in import_matches:
            named_imports = match.group(1) or ""
            default_import = match.group(2) or ""
            module = match.group(3)
            
            ast["body"].append({
                "type": "ImportStatement",
                "named_imports": [imp.strip() for imp in named_imports.split(",")] if named_imports else [],
                "default_import": default_import,
                "module": module
            })
    
    return ast

def _parse_parameters(self, params_str: str) -> List[Dict[str, str]]:
    """Parse function parameters with optional type annotations"""
    if not params_str.strip():
        return []
        
    # Split by comma but handle nested commas in type hints
    params = []
    current_param = ""
    bracket_depth = 0
    
    for char in params_str:
        if char == ',' and bracket_depth == 0:
            if current_param.strip():
                params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char
            if char == '[' or char == '(' or char == '{':
                bracket_depth += 1
            elif char == ']' or char == ')' or char == '}':
                bracket_depth -= 1
    
    if current_param.strip():
        params.append(current_param.strip())
    
    # Parse each parameter
    result = []
    for param in params:
        # Check for type annotations
        if ':' in param:
            name, type_ann = param.split(':', 1)
            # Handle default values
            if '=' in name:
                name, default = name.split('=', 1)
                result.append({
                    "name": name.strip(),
                    "type": type_ann.strip(),
                    "default": default.strip()
                })
            else:
                result.append({
                    "name": name.strip(),
                    "type": type_ann.strip()
                })
        else:
            # Handle default values without type annotation
            if '=' in param:
                name, default = param.split('=', 1)
                result.append({
                    "name": name.strip(),
                    "default": default.strip()
                })
            else:
                result.append({
                    "name": param.strip()
                })
    
    return result

def _extract_structure_from_ast(self, ast: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structural information from AST"""
    structure = {
        "classes": [],
        "functions": [],
        "imports": [],
        "variables": []
    }
    
    for node in ast.get("body", []):
        node_type = node.get("type", "")
        
        if node_type == "ClassDeclaration":
            class_info = {
                "name": node.get("name", ""),
                "superClass": node.get("superClass", []),
                "methods": []
            }
            
            for method in node.get("methods", []):
                class_info["methods"].append({
                    "name": method.get("name", ""),
                    "params": method.get("params", []),
                    "return_type": method.get("return_type", "")
                })
            
            structure["classes"].append(class_info)
            
        elif node_type == "FunctionDeclaration" or node_type == "ArrowFunction":
            structure["functions"].append({
                "name": node.get("name", ""),
                "params": node.get("params", []),
                "return_type": node.get("return_type", "")
            })
            
        elif node_type == "ImportStatement":
            if "module" in node:
                structure["imports"].append({
                    "module": node.get("module", ""),
                    "imports": node.get("imports", []),
                    "alias": node.get("alias", "")
                })
            else:
                structure["imports"].append({
                    "module": node.get("module", ""),
                    "named_imports": node.get("named_imports", []),
                    "default_import": node.get("default_import", "")
                })
    
    return structure

def _extract_visual_elements(self, image_data: str) -> List[Dict[str, Any]]:
    """Extract visual elements from image data"""
    # In a real implementation, this would use computer vision
    # Here we'll simulate by extracting from metadata if available
    
    # Check if this is base64 encoded image with metadata
    metadata_match = re.search(r'data:image/\w+;base64,[^;]+;metadata=(\{.*\})', image_data)
    if metadata_match:
        try:
            metadata = json.loads(metadata_match.group(1))
            if "elements" in metadata:
                return metadata["elements"]
        except json.JSONDecodeError:
            pass
    
    # Simplified placeholder implementation
    # In practice, this would use image recognition to identify shapes, text, etc.
    return [
        {
            "type": "rectangle",
            "x": 100,
            "y": 100,
            "width": 200,
            "height": 100,
            "text": "Placeholder Element"
        },
        {
            "type": "line",
            "x1": 300,
            "y1": 150,
            "x2": 400,
            "y2": 150
        },
        {
            "type": "text",
            "x": 150,
            "y": 150,
            "text": "Sample Text"
        }
    ]

def _convert_visual_to_entities(self, visual_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert visual elements to semantic entities"""
    entities = []
    
    for element in visual_elements:
        element_type = element.get("type", "")
        
        if element_type == "rectangle":
            # Rectangles with text are often classes, components, or data tables
            text = element.get("text", "")
            if text:
                if self._looks_like_class(text):
                    entities.append({
                        "name": text,
                        "type": "class",
                        "source": "visual"
                    })
                elif self._looks_like_database(text):
                    entities.append({
                        "name": text,
                        "type": "database",
                        "source": "visual"
                    })
                else:
                    entities.append({
                        "name": text,
                        "type": "ui_component",
                        "source": "visual"
                    })
        
        elif element_type == "circle" or element_type == "ellipse":
            # Circles often represent actors, states, or endpoints
            text = element.get("text", "")
            if text:
                entities.append({
                    "name": text,
                    "type": "actor",
                    "source": "visual"
                })
        
        elif element_type == "line" or element_type == "arrow":
            # Lines/arrows represent relationships or flows
            label = element.get("label", "")
            source = element.get("source", "")
            target = element.get("target", "")
            
            if source and target:
                entities.append({
                    "name": label or f"{source} to {target}",
                    "type": "relationship",
                    "source_entity": source,
                    "target_entity": target,
                    "source": "visual"
                })
        
        elif element_type == "text":
            # Standalone text might be labels or annotations
            text = element.get("text", "")
            if text:
                entities.append({
                    "name": text,
                    "type": "annotation",
                    "source": "visual"
                })
    
    return entities

def _looks_like_class(self, text: str) -> bool:
    """Check if text appears to be a class name"""
    # Classes often have PascalCase names and might contain fields/methods
    if re.match(r'^[A-Z][a-zA-Z0-9]*$', text.strip()):
        return True
    if ":" in text and any(marker in text.lower() for marker in ["class", "attributes", "methods", "properties"]):
        return True
    return False

def _looks_like_database(self, text: str) -> bool:
    """Check if text appears to be a database table"""
    # Database elements often contain "table", "entity", or have field listings
    lower_text = text.lower()
    if any(marker in lower_text for marker in ["table", "entity", "database", "db"]):
        return True
    if re.search(r'id|key|field|column', lower_text):
        return True
    return False

def _is_ui_sketch(self, visual_elements: List[Dict[str, Any]]) -> bool:
    """Determine if the visual elements represent a UI sketch"""
    ui_indicators = ["button", "input", "form", "label", "menu", "page", "screen"]
    
    for element in visual_elements:
        text = element.get("text", "").lower()
        if any(indicator in text for indicator in ui_indicators):
            return True
    
    # Check for UI-like structure (header, content, footer)
    has_header = False
    has_content = False
    has_footer = False
    
    for element in visual_elements:
        if element.get("y", 0) < 100:  # Top of image
            has_header = True
        elif element.get("y", 0) > 400:  # Bottom of image
            has_footer = True
        else:
            has_content = True
    
    return has_header and has_content

def _is_flowchart(self, visual_elements: List[Dict[str, Any]]) -> bool:
    """Determine if the visual elements represent a flowchart"""
    # Flowcharts typically have connected shapes with directional arrows
    has_decision = False
    has_process = False
    has_arrows = False
    
    for element in visual_elements:
        element_type = element.get("type", "")
        text = element.get("text", "").lower()
        
        if element_type == "diamond" or "decision" in text or "if" in text:
            has_decision = True
        elif element_type == "rectangle" and ("process" in text or "function" in text or "step" in text):
            has_process = True
        elif element_type in ["arrow", "line"] and element.get("direction"):
            has_arrows = True
    
    return has_arrows and (has_decision or has_process)

def _is_architecture_diagram(self, visual_elements: List[Dict[str, Any]]) -> bool:
    """Determine if the visual elements represent an architecture diagram"""
    # Architecture diagrams often have components, servers, databases, services
    architecture_indicators = [
        "server", "database", "service", "api", "cloud", "component", 
        "container", "microservice", "system", "application"
    ]
    
    indicator_count = 0
    for element in visual_elements:
        text = element.get("text", "").lower()
        for indicator in architecture_indicators:
            if indicator in text:
                indicator_count += 1
    
    return indicator_count >= 2

def _extract_sketch_features(self, visual_elements: List[Dict[str, Any]]) -> np.ndarray:
    """Extract feature vector from sketch elements"""
    # Create a feature vector based on visual elements
    features = np.zeros(self.embedding_dim)
    
    # Count element types
    element_types = {}
    for element in visual_elements:
        element_type = element.get("type", "unknown")
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    # Map element types to feature indices
    for element_type, count in element_types.items():
        idx = int(hashlib.md5(f"element_{element_type}".encode()).hexdigest(), 16) % self.embedding_dim
        features[idx] = min(count / 10.0, 1.0)  # Normalize to 0-1
    
    # Extract text features
    text_content = []
    for element in visual_elements:
        if "text" in element and element["text"]:
            text_content.append(element["text"])
    
    if text_content:
        combined_text = " ".join(text_content)
        words = re.findall(r'\w+', combined_text.lower())
        for word in words:
            idx = int(hashlib.md5(f"word_{word}".encode()).hexdigest(), 16) % self.embedding_dim
            features[idx] = 1.0
    
    # Spatial features - check if elements are aligned
    if len(visual_elements) > 1:
        x_positions = [element.get("x", element.get("x1", 0)) for element in visual_elements if "x" in element or "x1" in element]
        y_positions = [element.get("y", element.get("y1", 0)) for element in visual_elements if "y" in element or "y1" in element]
        
        # Check for horizontal alignment
        if x_positions and len(set(x_positions)) < len(x_positions) / 2:
            idx = int(hashlib.md5(b"horizontal_alignment").hexdigest(), 16) % self.embedding_dim
            features[idx] = 1.0
        
        # Check for vertical alignment
        if y_positions and len(set(y_positions)) < len(y_positions) / 2:
            idx = int(hashlib.md5(b"vertical_alignment").hexdigest(), 16) % self.embedding_dim
            features[idx] = 1.0
    
    # Normalize
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features

def _extract_uml_class_diagram(self, image_data: str) -> Dict[str, Any]:
    """Extract UML class diagram structure"""
    # Extract visual elements
    elements = self._extract_visual_elements(image_data)
    
    # Process class diagram
    classes = []
    relationships = []
    
    for element in elements:
        element_type = element.get("type", "")
        
        if element_type == "rectangle" and "text" in element:
            # Process as a class
            text = element["text"]
            
            # Try to parse class structure (name, attributes, methods)
            class_parts = text.split("\n", 2)
            class_name = class_parts[0].strip() if class_parts else ""
            
            class_info = {
                "name": class_name,
                "attributes": [],
                "methods": []
            }
            
            # Extract attributes and methods if available
            if len(class_parts) > 1:
                lines = class_parts[1].split("\n")
                in_attributes = True
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if we've moved from attributes to methods section
                    if line == "---" or line == "__":
                        in_attributes = False
                        continue
                    
                    # Process attribute or method
                    if in_attributes:
                        class_info["attributes"].append(line)
                    else:
                        class_info["methods"].append(line)
            
            classes.append(class_info)
            
        elif element_type in ["line", "arrow"]:
            # Process as a relationship
            source = element.get("source", "")
            target = element.get("target", "")
            relationship_type = element.get("relationship_type", "association")
            label = element.get("label", "")
            
            if source and target:
                relationships.append({
                    "source": source,
                    "target": target,
                    "type": relationship_type,
                    "label": label
                })
    
    return {
        "type": "uml_class_diagram",
        "classes": classes,
        "relationships": relationships
    }

def _extract_uml_sequence_diagram(self, image_data: str) -> Dict[str, Any]:
    """Extract UML sequence diagram structure"""
    elements = self._extract_visual_elements(image_data)
    
    # Process sequence diagram
    actors = []
    messages = []
    
    # Identify actors/lifelines (usually at the top)
    for element in elements:
        if element.get("type") == "rectangle" or element.get("type") == "actor":
            if element.get("y", 0) < 100:  # Top of diagram
                actors.append({
                    "name": element.get("text", "Actor"),
                    "x": element.get("x", 0)
                })
    
    # Process messages (arrows between lifelines)
    for element in elements:
        if element.get("type") in ["arrow", "line"]:
            # Find source and target actors
            source_actor = None
            target_actor = None
            
            source_x = element.get("x1", 0)
            target_x = element.get("x2", 0)
            
            for actor in actors:
                if abs(actor["x"] - source_x) < 30:
                    source_actor = actor["name"]
                if abs(actor["x"] - target_x) < 30:
                    target_actor = actor["name"]
            
            if source_actor and target_actor:
                messages.append({
                    "source": source_actor,
                    "target": target_actor,
                    "message": element.get("text", "message"),
                    "y": element.get("y1", 0)  # For ordering
                })
    
    # Sort messages by vertical position
    messages.sort(key=lambda m: m.get("y", 0))
    
    return {
        "type": "uml_sequence_diagram",
        "actors": [actor["name"] for actor in actors],
        "messages": messages
    }

def _extract_er_diagram(self, image_data: str) -> Dict[str, Any]:
    """Extract entity-relationship diagram structure"""
    elements = self._extract_visual_elements(image_data)
    
    # Process ER diagram
    entities = []
    relationships = []
    
    for element in elements:
        element_type = element.get("type", "")
        
        if element_type == "rectangle":
            # Likely an entity
            text = element.get("text", "")
            
            # Try to parse entity structure (name, attributes)
            entity_parts = text.split("\n", 1)
            entity_name = entity_parts[0].strip() if entity_parts else ""
            
            entity = {
                "name": entity_name,
                "attributes": []
            }
            
            # Extract attributes if available
            if len(entity_parts) > 1:
                lines = entity_parts[1].split("\n")
                for line in lines:
                    line = line.strip()
                    if line:
                        entity["attributes"].append(line)
            
            entities.append(entity)
            
        elif element_type in ["diamond", "rhombus"]:
            # Likely a relationship
            relationships.append({
                "name": element.get("text", "relationship"),
                "x": element.get("x", 0),
                "y": element.get("y", 0)
            })
            
        elif element_type in ["line", "arrow"]:
            # Relationship connection
            source = element.get("source", "")
            target = element.get("target", "")
            
            if source and target:
                # Check if source or target is a relationship
                rel_source = next((r for r in relationships if r["name"] == source), None)
                rel_target = next((r for r in relationships if r["name"] == target), None)
                
                if rel_source:
                    # Entity to relationship
                    rel_source["entity2"] = target
                    rel_source["cardinality2"] = element.get("label", "")
                elif rel_target:
                    # Entity to relationship
                    rel_target["entity1"] = source
                    rel_target["cardinality1"] = element.get("label", "")
                else:
                    # Direct entity to entity
                    relationships.append({
                        "name": element.get("label", "relates_to"),
                        "entity1": source,
                        "entity2": target,
                        "cardinality1": "",
                        "cardinality2": ""
                    })
    
    # Clean up relationship objects
    clean_relationships = []
    for rel in relationships:
        if "name" in rel and "entity1" in rel and "entity2" in rel:
            clean_relationships.append({
                "name": rel["name"],
                "entity1": rel["entity1"],
                "entity2": rel["entity2"],
                "cardinality1": rel.get("cardinality1", ""),
                "cardinality2": rel.get("cardinality2", "")
            })
    
    return {
        "type": "er_diagram",
        "entities": entities,
        "relationships": clean_relationships
    }

def _extract_flowchart(self, image_data: str) -> Dict[str, Any]:
    """Extract flowchart structure"""
    elements = self._extract_visual_elements(image_data)
    
    # Process flowchart
    nodes = []
    edges = []
    
    # First pass: identify nodes
    for element in elements:
        element_type = element.get("type", "")
        
        if element_type in ["rectangle", "diamond", "ellipse", "circle"]:
            node_type = "process"
            if element_type == "diamond":
                node_type = "decision"
            elif element_type in ["ellipse", "circle"]:
                if element.get("y", 0) < 100:  # Near top
                    node_type =
def _extract_flowchart(self, image_data: str) -> Dict[str, Any]:
    """Extract flowchart structure"""
    elements = self._extract_visual_elements(image_data)
    
    # Process flowchart
    nodes = []
    edges = []
    
    # First pass: identify nodes
    for element in elements:
        element_type = element.get("type", "")
        
        if element_type in ["rectangle", "diamond", "ellipse", "circle"]:
            node_type = "process"
            if element_type == "diamond":
                node_type = "decision"
            elif element_type in ["ellipse", "circle"]:
                if element.get("y", 0) < 100:  # Near top
                    node_type = "start"
                else:
                    node_type = "end"
            
            nodes.append({
                "id": f"node_{len(nodes)}",
                "type": node_type,
                "text": element.get("text", ""),
                "x": element.get("x", 0),
                "y": element.get("y", 0)
            })
    
    # Second pass: identify connections
    for element in elements:
        if element.get("type") in ["line", "arrow"]:
            source_node = None
            target_node = None
            
            # Find nodes near the line endpoints
            source_x = element.get("x1", 0)
            source_y = element.get("y1", 0)
            target_x = element.get("x2", 0)
            target_y = element.get("y2", 0)
            
            for node in nodes:
                # Check if node is near source point
                if abs(node["x"] - source_x) < 30 and abs(node["y"] - source_y) < 30:
                    source_node = node["id"]
                
                # Check if node is near target point
                if abs(node["x"] - target_x) < 30 and abs(node["y"] - target_y) < 30:
                    target_node = node["id"]
            
            if source_node and target_node:
                edges.append({
                    "source": source_node,
                    "target": target_node,
                    "label": element.get("text", ""),
                    "condition": element.get("text", "") if source_node.startswith("decision") else ""
                })
    
    return {
        "type": "flowchart",
        "nodes": nodes,
        "edges": edges
    }

def _interpret_diagram_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Interpret generic diagram elements when diagram type is unknown"""
    # Try to determine diagram type from elements
    if self._looks_like_uml_class(elements):
        return self._extract_uml_class_diagram_from_elements(elements)
    elif self._looks_like_uml_sequence(elements):
        return self._extract_uml_sequence_diagram_from_elements(elements)
    elif self._looks_like_er_diagram(elements):
        return self._extract_er_diagram_from_elements(elements)
    elif self._looks_like_flowchart(elements):
        return self._extract_flowchart_from_elements(elements)
    
    # Generic diagram interpretation
    nodes = []
    connections = []
    
    # Extract nodes (rectangles, circles, etc.)
    for element in elements:
        element_type = element.get("type", "")
        
        if element_type in ["rectangle", "circle", "ellipse", "diamond", "triangle"]:
            nodes.append({
                "id": f"node_{len(nodes)}",
                "type": element_type,
                "text": element.get("text", ""),
                "x": element.get("x", 0),
                "y": element.get("y", 0)
            })
    
    # Extract connections (lines, arrows)
    for element in elements:
        if element.get("type") in ["line", "arrow"]:
            # Try to find connecting nodes
            source_node = None
            target_node = None
            
            # Find nodes near the line endpoints
            source_x = element.get("x1", 0)
            source_y = element.get("y1", 0)
            target_x = element.get("x2", 0)
            target_y = element.get("y2", 0)
            
            for node in nodes:
                # Check if node is near source point
                if abs(node["x"] - source_x) < 30 and abs(node["y"] - source_y) < 30:
                    source_node = node["id"]
                
                # Check if node is near target point
                if abs(node["x"] - target_x) < 30 and abs(node["y"] - target_y) < 30:
                    target_node = node["id"]
            
            if source_node and target_node:
                connections.append({
                    "source": source_node,
                    "target": target_node,
                    "label": element.get("text", ""),
                    "type": "directed" if element.get("type") == "arrow" else "undirected"
                })
    
    return {
        "type": "generic_diagram",
        "nodes": nodes,
        "connections": connections
    }

def _looks_like_uml_class(self, elements: List[Dict[str, Any]]) -> bool:
    """Check if elements represent a UML class diagram"""
    # Look for rectangles with class-like text
    rectangle_count = 0
    class_indicators = 0
    
    for element in elements:
        if element.get("type") == "rectangle":
            rectangle_count += 1
            text = element.get("text", "").lower()
            
            if any(indicator in text for indicator in ["class", "interface", "abstract", "attributes", "methods"]):
                class_indicators += 1
            
            # Check for compartmented rectangles (typical in class diagrams)
            if text.count("\n") > 1 and "---" in text:
                class_indicators += 1
    
    # Check for inheritance/association arrows
    arrow_count = sum(1 for e in elements if e.get("type") in ["arrow", "line"])
    
    return rectangle_count >= 2 and class_indicators >= 1 and arrow_count >= 1

def _looks_like_uml_sequence(self, elements: List[Dict[str, Any]]) -> bool:
    """Check if elements represent a UML sequence diagram"""
    # Sequence diagrams have rectangles at top (lifelines) and arrows between vertical lines
    top_rectangles = sum(1 for e in elements if e.get("type") == "rectangle" and e.get("y", 0) < 100)
    vertical_lines = sum(1 for e in elements if e.get("type") == "line" and abs(e.get("x1", 0) - e.get("x2", 0)) < 10)
    horizontal_arrows = sum(1 for e in elements if e.get("type") == "arrow" and abs(e.get("y1", 0) - e.get("y2", 0)) < 10)
    
    return top_rectangles >= 2 and vertical_lines >= 2 and horizontal_arrows >= 1

def _looks_like_er_diagram(self, elements: List[Dict[str, Any]]) -> bool:
    """Check if elements represent an ER diagram"""
    # ER diagrams have rectangles (entities) and diamonds (relationships)
    rectangles = sum(1 for e in elements if e.get("type") == "rectangle")
    diamonds = sum(1 for e in elements if e.get("type") in ["diamond", "rhombus"])
    connecting_lines = sum(1 for e in elements if e.get("type") in ["line", "arrow"])
    
    # Check for entity-like text content
    entity_indicators = 0
    for element in elements:
        if element.get("type") == "rectangle":
            text = element.get("text", "").lower()
            if any(indicator in text for indicator in ["id", "key", "entity", "attribute"]):
                entity_indicators += 1
    
    return rectangles >= 1 and (diamonds >= 1 or entity_indicators >= 1) and connecting_lines >= 1

def _looks_like_flowchart(self, elements: List[Dict[str, Any]]) -> bool:
    """Check if elements represent a flowchart"""
    # Flowcharts have decision diamonds and process rectangles
    decision_diamonds = sum(1 for e in elements if e.get("type") == "diamond")
    process_rectangles = sum(1 for e in elements if e.get("type") == "rectangle")
    arrows = sum(1 for e in elements if e.get("type") == "arrow")
    
    # Check for flowchart-specific text
    flow_indicators = 0
    for element in elements:
        text = element.get("text", "").lower()
        if any(indicator in text for indicator in ["start", "end", "if", "then", "else", "process", "decision"]):
            flow_indicators += 1
    
    return (decision_diamonds >= 1 or flow_indicators >= 1) and process_rectangles >= 1 and arrows >= 1

def _extract_uml_class_diagram_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract UML class diagram from generic elements"""
    # Similar to _extract_uml_class_diagram but takes elements directly
    classes = []
    relationships = []
    
    # Extract classes
    for element in elements:
        if element.get("type") == "rectangle":
            text = element.get("text", "")
            
            # Parse class structure (name, attributes, methods)
            parts = text.split("\n")
            class_name = parts[0].strip() if parts else ""
            
            class_info = {
                "name": class_name,
                "attributes": [],
                "methods": []
            }
            
            # Extract attributes and methods
            current_section = "attributes"
            for i, line in enumerate(parts[1:], 1):
                line = line.strip()
                if not line:
                    continue
                
                if line == "---" or line.startswith("__"):
                    current_section = "methods"
                    continue
                
                if current_section == "attributes":
                    class_info["attributes"].append(line)
                else:
                    class_info["methods"].append(line)
            
            classes.append(class_info)
    
    # Extract relationships
    for element in elements:
        if element.get("type") in ["line", "arrow"]:
            # Determine relationship type
            rel_type = "association"
            if element.get("style") == "dashed":
                rel_type = "dependency"
            elif element.get("arrow_type") == "triangle":
                rel_type = "inheritance"
            elif element.get("arrow_type") == "diamond":
                rel_type = "aggregation" if element.get("arrow_fill", False) else "composition"
            
            # Find source and target classes
            source_class = None
            target_class = None
            
            source_x = element.get("x1", 0)
            source_y = element.get("y1", 0)
            target_x = element.get("x2", 0)
            target_y = element.get("y2", 0)
            
            # Find nearest classes to endpoints
            for i, element in enumerate(elements):
                if element.get("type") == "rectangle":
                    element_x = element.get("x", 0)
                    element_y = element.get("y", 0)
                    element_width = element.get("width", 50)
                    element_height = element.get("height", 50)
                    
                    # Check if source point is near this element
                    if (element_x - element_width/2 <= source_x <= element_x + element_width/2 and
                        element_y - element_height/2 <= source_y <= element_y + element_height/2):
                        if elements[i].get("text"):
                            source_class = elements[i].get("text").split("\n")[0].strip()
                    
                    # Check if target point is near this element
                    if (element_x - element_width/2 <= target_x <= element_x + element_width/2 and
                        element_y - element_height/2 <= target_y <= element_y + element_height/2):
                        if elements[i].get("text"):
                            target_class = elements[i].get("text").split("\n")[0].strip()
            
            if source_class and target_class:
                relationships.append({
                    "source": source_class,
                    "target": target_class,
                    "type": rel_type,
                    "label": element.get("text", "")
                })
    
    return {
        "type": "uml_class_diagram",
        "classes": classes,
        "relationships": relationships
    }

def _extract_uml_sequence_diagram_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract UML sequence diagram from generic elements"""
    actors = []
    lifelines = []
    messages = []
    
    # Find actors/objects (typically rectangles at top)
    for element in elements:
        if element.get("type") in ["rectangle", "actor"] and element.get("y", 0) < 100:
            actor_name = element.get("text", "Actor")
            actor_x = element.get("x", 0)
            
            actors.append({
                "name": actor_name,
                "x": actor_x
            })
            
            # Look for associated lifeline (vertical line below actor)
            for line_elem in elements:
                if (line_elem.get("type") == "line" and
                    abs(line_elem.get("x1", 0) - actor_x) < 10 and
                    abs(line_elem.get("x2", 0) - actor_x) < 10 and
                    line_elem.get("y1", 0) > element.get("y", 0)):
                    
                    lifelines.append({
                        "actor": actor_name,
                        "x": actor_x,
                        "y1": line_elem.get("y1", 0),
                        "y2": line_elem.get("y2", 0)
                    })
    
    # Extract messages (arrows between lifelines)
    for element in elements:
        if element.get("type") == "arrow":
            source_x = element.get("x1", 0)
            target_x = element.get("x2", 0)
            y_position = element.get("y1", 0)
            
            # Find source and target actors by x position
            source_actor = None
            target_actor = None
            
            for actor in actors:
                if abs(actor["x"] - source_x) < 20:
                    source_actor = actor["name"]
                if abs(actor["x"] - target_x) < 20:
                    target_actor = actor["name"]
            
            if source_actor and target_actor:
                messages.append({
                    "source": source_actor,
                    "target": target_actor,
                    "message": element.get("text", "message"),
                    "y": y_position,
                    "is_return": element.get("style") == "dashed"
                })
    
    # Sort messages by vertical position
    messages.sort(key=lambda m: m["y"])
    
    return {
        "type": "uml_sequence_diagram",
        "actors": [actor["name"] for actor in actors],
        "messages": messages
    }

def _extract_er_diagram_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract ER diagram from generic elements"""
    entities = []
    relationships = []
    
    # Extract entities (rectangles)
    for element in elements:
        if element.get("type") == "rectangle":
            entity_text = element.get("text", "")
            entity_parts = entity_text.split("\n", 1)
            
            entity_name = entity_parts[0].strip()
            attributes = []
            
            if len(entity_parts) > 1:
                for line in entity_parts[1].split("\n"):
                    line = line.strip()
                    if line:
                        attributes.append(line)
            
            entities.append({
                "name": entity_name,
                "attributes": attributes,
                "x": element.get("x", 0),
                "y": element.get("y", 0)
            })
    
    # Extract relationships (diamonds)
    for element in elements:
        if element.get("type") in ["diamond", "rhombus"]:
            rel_name = element.get("text", "relationship")
            rel_x = element.get("x", 0)
            rel_y = element.get("y", 0)
            
            # Find connected entities
            connected_entities = []
            
            for line_elem in elements:
                if line_elem.get("type") in ["line", "arrow"]:
                    # Check if line connects to this relationship
                    if ((abs(line_elem.get("x1", 0) - rel_x) < 20 and abs(line_elem.get("y1", 0) - rel_y) < 20) or
                        (abs(line_elem.get("x2", 0) - rel_x) < 20 and abs(line_elem.get("y2", 0) - rel_y) < 20)):
                        
                        # Find entity at the other end
                        other_x = line_elem.get("x2", 0) if abs(line_elem.get("x1", 0) - rel_x) < 20 else line_elem.get("x1", 0)
                        other_y = line_elem.get("y2", 0) if abs(line_elem.get("y1", 0) - rel_y) < 20 else line_elem.get("y1", 0)
                        
                        # Find entity at this position
                        for entity in entities:
                            if abs(entity["x"] - other_x) < 20 and abs(entity["y"] - other_y) < 20:
                                connected_entities.append({
                                    "entity": entity["name"],
                                    "cardinality": line_elem.get("text", "")
                                })
            
            if len(connected_entities) >= 2:
                relationships.append({
                    "name": rel_name,
                    "entity1": connected_entities[0]["entity"],
                    "cardinality1": connected_entities[0]["cardinality"],
                    "entity2": connected_entities[1]["entity"],
                    "cardinality2": connected_entities[1]["cardinality"]
                })
    
    # Also look for direct relationships (lines between entities without diamonds)
    for line_elem in elements:
        if line_elem.get("type") in ["line", "arrow"]:
            source_entity = None
            target_entity = None
            
            source_x = line_elem.get("x1", 0)
            source_y = line_elem.get("y1", 0)
            target_x = line_elem.get("x2", 0)
            target_y = line_elem.get("y2", 0)
            
            # Find entities at endpoints
            for entity in entities:
                if abs(entity["x"] - source_x) < 20 and abs(entity["y"] - source_y) < 20:
                    source_entity = entity["name"]
                if abs(entity["x"] - target_x) < 20 and abs(entity["y"] - target_y) < 20:
                    target_entity = entity["name"]
            
            if source_entity and target_entity:
                # Check if this relationship is already captured via a diamond
                is_new = True
                for rel in relationships:
                    if ((rel["entity1"] == source_entity and rel["entity2"] == target_entity) or
                        (rel["entity1"] == target_entity and rel["entity2"] == source_entity)):
                        is_new = False
                        break
                
                if is_new:
                    relationships.append({
                        "name": line_elem.get("text", "relates_to"),
                        "entity1": source_entity,
                        "cardinality1": "",
                        "entity2": target_entity,
                        "cardinality2": ""
                    })
    
    return {
        "type": "er_diagram",
        "entities": [{"name": e["name"], "attributes": e["attributes"]} for e in entities],
        "relationships": relationships
    }

def _extract_flowchart_from_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract flowchart from generic elements"""
    # Similar to _extract_flowchart but with different input format
    nodes = []
    edges = []
    
    # Extract nodes
    for element in elements:
        if element.get("type") in ["rectangle", "diamond", "ellipse", "circle"]:
            node_type = "process"
            if element.get("type") == "diamond":
                node_type = "decision"
            elif element.get("type") in ["ellipse", "circle"]:
                if "start" in element.get("text", "").lower():
                    node_type = "start"
                elif "end" in element.get("text", "").lower():
                    node_type = "end"
                else:
                    node_type = "terminal"
            
            nodes.append({
                "id": f"node_{len(nodes)}",
                "type": node_type,
                "text": element.get("text", ""),
                "x": element.get("x", 0),
                "y": element.get("y", 0)
            })
    
    # Extract edges
    for element in elements:
        if element.get("type") in ["line", "arrow"]:
            source_x = element.get("x1", 0)
            source_y = element.get("y1", 0)
            target_x = element.get("x2", 0)
            target_y = element.get("y2", 0)
            
            # Find nodes at endpoints
            source_node = None
            target_node = None
            
            for node in nodes:
                if abs(node["x"] - source_x) < 20 and abs(node["y"] - source_y) < 20:
                    source_node = node["id"]
                if abs(node["x"] - target_x) < 20 and abs(node["y"] - target_y) < 20:
                    target_node = node["id"]
            
            if source_node and target_node:
                # Set label based on source node type
                label = element.get("text", "")
                if not label:
                    source_node_type = next((n["type"] for n in nodes if n["id"] == source_node), None)
                    if source_node_type == "decision":
                        # Infer yes/no based on direction
                        if target_x > source_x + 10:
                            label = "Yes"
                        elif target_x < source_x - 10:
                            label = "No"
                
                edges.append({
                    "source": source_node,
                    "target": target_node,
                    "label": label
                })
    
    return {
        "type": "flowchart",
        "nodes": nodes,
        "edges": edges
    }

def _extract_entities_from_structure(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract entities from a structural representation"""
    entities = []
    
    # Extract from classes
    if "classes" in structure:
        for cls in structure["classes"]:
            entities.append({
                "name": cls.get("name", ""),
                "type": "class",
                "attributes": cls.get("attributes", []),
                "methods": cls.get("methods", []),
                "source": "structure"
            })
    
    # Extract from nodes (diagrams)
    if "nodes" in structure:
        for node in structure["nodes"]:
            entities.append({
                "name": node.get("text", node.get("id", "")),
                "type": node.get("type", "node"),
                "source": "structure"
            })
    
    # Extract from entities (ER diagrams)
    if "entities" in structure:
        for entity in structure["entities"]:
            entities.append({
                "name": entity.get("name", ""),
                "type": "entity",
                "attributes": entity.get("attributes", []),
                "source": "structure"
            })
    
    # Extract from actors (sequence diagrams)
    if "actors" in structure:
        for actor in structure["actors"]:
            entities.append({
                "name": actor if isinstance(actor, str) else actor.get("name", ""),
                "type": "actor",
                "source": "structure"
            })
    
    # Extract relationships/connections
    if "relationships" in structure:
        for rel in structure["relationships"]:
            entities.append({
                "name": rel.get("name", ""),
                "type": "relationship",
                "source_entity": rel.get("source", rel.get("entity1", "")),
                "target_entity": rel.get("target", rel.get("entity2", "")),
                "source": "structure"
            })
    elif "edges" in structure or "connections" in structure:
        connections = structure.get("edges", structure.get("connections", []))
        for conn in connections:
            entities.append({
                "name": conn.get("label", "connection"),
                "type": "connection",
                "source_entity": conn.get("source", ""),
                "target_entity": conn.get("target", ""),
                "source": "structure"
            })
    
    return entities
