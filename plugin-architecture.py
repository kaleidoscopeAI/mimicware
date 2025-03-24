#!/usr/bin/env python3
"""
Plugin Architecture for System Builder
Enables dynamic loading and integration of plugins to extend functionality
"""

import os
import sys
import importlib.util
import inspect
import json
import pkgutil
import logging
import traceback
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Set, Type, Callable, Union
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PluginSystem")

# Plugin types
class PluginType(Enum):
    """Types of plugins supported by the system"""
    NODE = auto()  # Node type plugins
    EXECUTOR = auto()  # Script/task execution plugins
    TRANSFORM = auto()  # Data transformation plugins
    EXPORTER = auto()  # Export workflow to different formats
    INTEGRATOR = auto()  # External system integrations
    UI = auto()  # UI extensions and widgets
    VALIDATOR = auto()  # Custom validation rules
    SECURITY = auto()  # Security features
    UTILITY = auto()  # General utility plugins
    CUSTOM = auto()  # Custom plugin types

# Plugin metadata
@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    id: str  # Unique identifier
    name: str  # Display name
    version: str  # Version string
    description: str  # Plugin description
    author: str  # Author name
    plugin_type: PluginType  # Type of plugin
    dependencies: List[str] = field(default_factory=list)  # Plugin dependencies
    requires: Dict[str, str] = field(default_factory=dict)  # Required packages
    compatible_versions: List[str] = field(default_factory=list)  # Compatible System Builder versions
    settings: Dict[str, Any] = field(default_factory=dict)  # Plugin settings schema
    entry_points: Dict[str, str] = field(default_factory=dict)  # Entry points for different features
    tags: List[str] = field(default_factory=list)  # Tags for categorization

# Plugin base class
class PluginBase(ABC):
    """Base class for all plugins"""
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin"""
        pass

# Node plugin interface
class NodePlugin(PluginBase):
    """Interface for node type plugins"""
    
    @abstractmethod
    def get_node_types(self) -> List[Tuple[str, str, Any]]:
        """Get list of node types provided by this plugin
        
        Returns:
            List of tuples: (node_type_id, display_name, node_class)
        """
        pass
    
    @abstractmethod
    def create_node(self, node_type_id: str, **kwargs) -> Any:
        """Create a node instance of specified type"""
        pass

# Executor plugin interface
class ExecutorPlugin(PluginBase):
    """Interface for executor plugins"""
    
    @abstractmethod
    def can_execute(self, script_type: str, **kwargs) -> bool:
        """Check if this executor can execute the given script type"""
        pass
    
    @abstractmethod
    def execute(self, script_path: str, script_type: str, args: List[str] = None, 
               env: Dict[str, str] = None, **kwargs) -> Tuple[int, str, str]:
        """Execute a script
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        pass

# Transform plugin interface
class TransformPlugin(PluginBase):
    """Interface for data transformation plugins"""
    
    @abstractmethod
    def get_transformations(self) -> List[Tuple[str, str, str]]:
        """Get list of transformations provided by this plugin
        
        Returns:
            List of tuples: (transform_id, display_name, description)
        """
        pass
    
    @abstractmethod
    def transform(self, transform_id: str, data: Any, **kwargs) -> Any:
        """Apply a transformation to data"""
        pass

# Exporter plugin interface
class ExporterPlugin(PluginBase):
    """Interface for workflow export plugins"""
    
    @abstractmethod
    def get_export_formats(self) -> List[Tuple[str, str, str]]:
        """Get list of export formats provided by this plugin
        
        Returns:
            List of tuples: (format_id, display_name, file_extension)
        """
        pass
    
    @abstractmethod
    def export_workflow(self, format_id: str, workflow: Any, filepath: str, **kwargs) -> bool:
        """Export a workflow to the specified format"""
        pass

# UI plugin interface
class UIPlugin(PluginBase):
    """Interface for UI extension plugins"""
    
    @abstractmethod
    def get_widgets(self) -> List[Tuple[str, str, Any]]:
        """Get list of widgets provided by this plugin
        
        Returns:
            List of tuples: (widget_id, display_name, widget_class)
        """
        pass
    
    @abstractmethod
    def create_widget(self, widget_id: str, parent: Any, **kwargs) -> Any:
        """Create a widget instance"""
        pass
    
    @abstractmethod
    def get_menu_items(self) -> List[Tuple[str, str, Callable]]:
        """Get list of menu items provided by this plugin
        
        Returns:
            List of tuples: (menu_path, display_name, callback)
        """
        pass

# Plugin discovery and management
class PluginManager:
    """Manages discovery, loading and lifecycle of plugins"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or []
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_modules: Dict[str, Any] = {}
        self.disabled_plugins: Set[str] = set()
        self.dependency_graph = {}
        self.plugin_types: Dict[PluginType, Dict[str, PluginBase]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self.failed_plugins: Dict[str, str] = {}
    
    def add_plugin_directory(self, directory: str) -> None:
        """Add a directory to search for plugins"""
        if os.path.exists(directory) and os.path.isdir(directory):
            if directory not in self.plugin_dirs:
                self.plugin_dirs.append(directory)
        else:
            logger.warning(f"Plugin directory not found: {directory}")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover plugins in the registered directories"""
        discovered_metadata = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
            
            # Look for plugin modules and packages
            for item in os.listdir(plugin_dir):
                item_path = os.path.join(plugin_dir, item)
                
                # Skip hidden files and directories
                if item.startswith('.'):
                    continue
                
                # Check if it's a Python module or package
                if os.path.isfile(item_path) and item.endswith('.py'):
                    # Python module
                    module_name = os.path.splitext(item)[0]
                    metadata = self._get_plugin_metadata(item_path, module_name)
                    if metadata:
                        discovered_metadata.append(metadata)
                
                elif os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, '__init__.py')):
                    # Python package
                    metadata = self._get_plugin_metadata(os.path.join(item_path, '__init__.py'), item)
                    if metadata:
                        discovered_metadata.append(metadata)
        
        return discovered_metadata
    
    def _get_plugin_metadata(self, module_path: str, module_name: str) -> Optional[PluginMetadata]:
        """Extract plugin metadata from a module"""
        try:
            # Look for plugin.json first
            json_path = os.path.join(os.path.dirname(module_path), 'plugin.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Convert plugin_type string to enum
                plugin_type_str = data.get('plugin_type', 'CUSTOM')
                try:
                    plugin_type = PluginType[plugin_type_str]
                except KeyError:
                    plugin_type = PluginType.CUSTOM
                
                return PluginMetadata(
                    id=data.get('id', module_name),
                    name=data.get('name', module_name),
                    version=data.get('version', '0.1.0'),
                    description=data.get('description', ''),
                    author=data.get('author', 'Unknown'),
                    plugin_type=plugin_type,
                    dependencies=data.get('dependencies', []),
                    requires=data.get('requires', {}),
                    compatible_versions=data.get('compatible_versions', []),
                    settings=data.get('settings', {}),
                    entry_points=data.get('entry_points', {}),
                    tags=data.get('tags', [])
                )
            
            # Try to load the module and extract metadata
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if module has PLUGIN_METADATA
            if hasattr(module, 'PLUGIN_METADATA'):
                data = module.PLUGIN_METADATA
                
                # Convert plugin_type string to enum if necessary
                plugin_type = data.get('plugin_type', PluginType.CUSTOM)
                if isinstance(plugin_type, str):
                    try:
                        plugin_type = PluginType[plugin_type]
                    except KeyError:
                        plugin_type = PluginType.CUSTOM
                
                return PluginMetadata(
                    id=data.get('id', module_name),
                    name=data.get('name', module_name),
                    version=data.get('version', '0.1.0'),
                    description=data.get('description', ''),
                    author=data.get('author', 'Unknown'),
                    plugin_type=plugin_type,
                    dependencies=data.get('dependencies', []),
                    requires=data.get('requires', {}),
                    compatible_versions=data.get('compatible_versions', []),
                    settings=data.get('settings', {}),
                    entry_points=data.get('entry_points', {}),
                    tags=data.get('tags', [])
                )
            
            # If no explicit metadata, check if the module has a plugin class
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginBase) and obj is not PluginBase:
                    # Create temporary instance to get metadata
                    try:
                        instance = obj()
                        return instance.metadata
                    except Exception as e:
                        logger.debug(f"Failed to instantiate plugin class in {module_name}: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract metadata from {module_path}: {e}")
            return None
    
    def load_plugin(self, metadata: PluginMetadata) -> Optional[PluginBase]:
        """Load a plugin based on its metadata"""
        if metadata.id in self.plugins:
            return self.plugins[metadata.id]
        
        if metadata.id in self.disabled_plugins:
            logger.debug(f"Plugin {metadata.id} is disabled")
            return None
        
        try:
            # Check dependencies
            for dep_id in metadata.dependencies:
                if dep_id not in self.plugins:
                    logger.warning(f"Plugin {metadata.id} depends on {dep_id} which is not loaded")
                    return None
            
            # Find module path
            module_path = None
            for plugin_dir in self.plugin_dirs:
                # Check if it's a standalone module
                py_path = os.path.join(plugin_dir, f"{metadata.id}.py")
                if os.path.exists(py_path):
                    module_path = py_path
                    break
                
                # Check if it's a package
                pkg_path = os.path.join(plugin_dir, metadata.id, '__init__.py')
                if os.path.exists(pkg_path):
                    module_path = pkg_path
                    break
            
            if not module_path:
                logger.warning(f"Module not found for plugin {metadata.id}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(metadata.id, module_path)
            if not spec or not spec.loader:
                logger.warning(f"Failed to create spec for plugin {metadata.id}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for _, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and issubclass(obj, PluginBase) and obj is not PluginBase
                    and obj is not NodePlugin and obj is not ExecutorPlugin
                    and obj is not TransformPlugin and obj is not ExporterPlugin
                    and obj is not UIPlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.warning(f"No plugin class found in {metadata.id}")
                return None
            
            # Create plugin instance
            plugin = plugin_class()
            
            # Verify metadata
            if plugin.metadata.id != metadata.id:
                logger.warning(f"Plugin ID mismatch: {plugin.metadata.id} != {metadata.id}")
                return None
            
            # Initialize plugin
            if not plugin.initialize():
                logger.warning(f"Failed to initialize plugin {metadata.id}")
                return None
            
            # Store plugin
            self.plugins[metadata.id] = plugin
            self.plugin_modules[metadata.id] = module
            self.plugin_types[plugin.metadata.plugin_type][metadata.id] = plugin
            
            logger.info(f"Loaded plugin: {metadata.id} ({plugin.metadata.name}) version {plugin.metadata.version}")
            
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin {metadata.id}: {e}")
            self.failed_plugins[metadata.id] = str(e)
            traceback.print_exc()
            return None
    
    def load_plugins(self) -> Dict[str, PluginBase]:
        """Discover and load all plugins"""
        # Discover available plugins
        discovered = self.discover_plugins()
        
        # Build dependency graph
        dependency_graph = {}
        for metadata in discovered:
            dependency_graph[metadata.id] = metadata.dependencies
        
        # Topologically sort plugins to load them in the correct order
        load_order = self._topological_sort(dependency_graph)
        
        # Load plugins in order
        for plugin_id in load_order:
            # Find metadata for this plugin
            metadata = next((m for m in discovered if m.id == plugin_id), None)
            if metadata:
                self.load_plugin(metadata)
        
        return self.plugins
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Topologically sort a dependency graph"""
        result = []
        visited = set()
        temp_marks = set()
        
        def visit(node):
            if node in temp_marks:
                raise ValueError(f"Circular dependency detected involving {node}")
            if node not in visited:
                temp_marks.add(node)
                for dep in graph.get(node, []):
                    if dep in graph:  # Only visit dependencies that exist in the graph
                        visit(dep)
                temp_marks.remove(node)
                visited.add(node)
                result.append(node)
        
        # Visit each node
        for node in graph:
            if node not in visited:
                visit(node)
        
        # Reverse to get correct order
        result.reverse()
        return result
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        if plugin_id not in self.plugins:
            return False
        
        # Check if other plugins depend on this one
        for pid, plugin in self.plugins.items():
            if plugin_id in plugin.metadata.dependencies:
                logger.warning(f"Cannot unload {plugin_id} because {pid} depends on it")
                return False
        
        try:
            # Shutdown plugin
            plugin = self.plugins[plugin_id]
            if not plugin.shutdown():
                logger.warning(f"Plugin {plugin_id} did not shutdown cleanly")
            
            # Remove plugin from collections
            del self.plugins[plugin_id]
            if plugin_id in self.plugin_modules:
                del self.plugin_modules[plugin_id]
            
            plugin_type = plugin.metadata.plugin_type
            if plugin_id in self.plugin_types[plugin_type]:
                del self.plugin_types[plugin_type][plugin_id]
            
            logger.info(f"Unloaded plugin: {plugin_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_id}: {e}")
            return False
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable a disabled plugin"""
        if plugin_id in self.disabled_plugins:
            self.disabled_plugins.remove(plugin_id)
            return True
        return False
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """Disable a plugin"""
        if plugin_id in self.plugins:
            # Unload plugin
            if self.unload_plugin(plugin_id):
                self.disabled_plugins.add(plugin_id)
                return True
        elif plugin_id not in self.disabled_plugins:
            self.disabled_plugins.add(plugin_id)
            return True
        return False
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> Dict[str, PluginBase]:
        """Get all plugins of a specific type"""
        return self.plugin_types.get(plugin_type, {})
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """Get a plugin by ID"""
        return self.plugins.get(plugin_id)
    
    def shutdown_all(self) -> bool:
        """Shutdown all plugins"""
        success = True
        
        # Shutdown in reverse dependency order
        plugin_ids = list(self.plugins.keys())
        for plugin_id in reversed(plugin_ids):
            try:
                plugin = self.plugins[plugin_id]
                if not plugin.shutdown():
                    logger.warning(f"Plugin {plugin_id} did not shutdown cleanly")
                    success = False
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin_id}: {e}")
                success = False
        
        self.plugins.clear()
        self.plugin_modules.clear()
        for plugin_type in self.plugin_types:
            self.plugin_types[plugin_type].clear()
        
        return success

# Node type plugin implementation example
class CustomNodePlugin(NodePlugin):
    """Example implementation of a node type plugin"""
    
    def __init__(self):
        self._metadata = PluginMetadata(
            id="custom_node_plugin",
            name="Custom Node Types",
            version="1.0.0",
            description="Adds custom node types to System Builder",
            author="System Builder Team",
            plugin_type=PluginType.NODE,
            tags=["nodes", "custom"]
        )
        self.initialized = False
        self.node_types = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata
    
    def initialize(self) -> bool:
        # Register node types
        self.node_types = {
            "http_request": ("HTTP Request", self.HttpRequestNode),
            "database_query": ("Database Query", self.DatabaseQueryNode),
            "data_filter": ("Data Filter", self.DataFilterNode)
        }
        self.initialized = True
        return True
    
    def shutdown(self) -> bool:
        self.initialized = False
        return True
    
    def get_node_types(self) -> List[Tuple[str, str, Any]]:
        return [(id, name, cls) for id, (name, cls) in self.node_types.items()]
    
    def create_node(self, node_type_id: str, **kwargs) -> Any:
        if node_type_id not in self.node_types:
            raise ValueError(f"Unknown node type: {node_type_id}")
        
        _, node_class = self.node_types[node_type_id]
        return node_class(**kwargs)
    
    # Node type classes
    class HttpRequestNode:
        """HTTP Request node type"""
        def __init__(self, url="", method="GET", headers=None, body="", **kwargs):
            self.url = url
            self.method = method
            self.headers = headers or {}
            self.body = body
            self.kwargs = kwargs
        
        def execute(self):
            """Execute the HTTP request"""
            import requests
            
            response = requests.request(
                method=self.method,
                url=self.url,
                headers=self.headers,
                data=self.body
            )
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
                "elapsed": response.elapsed.total_seconds()
            }
    
    class DatabaseQueryNode:
        """Database Query node type"""
        def __init__(self, connection_string="", query="", parameters=None, **kwargs):
            self.connection_string = connection_string
            self.query = query
            self.parameters = parameters or {}
            self.kwargs = kwargs
        
        def execute(self):
            """Execute the database query"""
            # This is a simplified example - real implementation would depend on the database
            import sqlite3
            
            if not self.connection_string.startswith("sqlite:///"):
                raise ValueError("Only SQLite connections are supported in this example")
            
            db_path = self.connection_string[10:]
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(self.query, self.parameters)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                results = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "row_count": len(results),
                    "columns": columns,
                    "results": results
                }
    
    class DataFilterNode:
        """Data Filter node type"""
        def __init__(self, filter_expression="", **kwargs):
            self.filter_expression = filter_expression
            self.kwargs = kwargs
        
        def execute(self, data):
            """Filter data based on expression"""
            if not data or not isinstance(data, list):
                return []
            
            if not self.filter_expression:
                return data
            
            # Compile the filter expression
            code = compile(f"lambda item: {self.filter_expression}", "<filter>", "eval")
            
            # Apply the filter
            filter_func = eval(code)
            filtered_data = [item for item in data if filter_func(item)]
            
            return filtered_data

# Transform plugin implementation example
class DataTransformPlugin(TransformPlugin):
    """Example implementation of a data transformation plugin"""
    
    def __init__(self):
        self._metadata = PluginMetadata(
            id="data_transform_plugin",
            name="Data Transformations",
            version="1.0.0",
            description="Provides data transformation functions",
            author="System Builder Team",
            plugin_type=PluginType.TRANSFORM,
            tags=["transform", "data"]
        )
        self.initialized = False
        self.transformations = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata
    
    def initialize(self) -> bool:
        # Register transformations
        self.transformations = {
            "json_to_csv": ("JSON to CSV", "Convert JSON array to CSV", self._json_to_csv),
            "csv_to_json": ("CSV to JSON", "Convert CSV to JSON array", self._csv_to_json),
            "filter_data": ("Filter Data", "Filter data by expression", self._filter_data),
            "sort_data": ("Sort Data", "Sort data by field", self._sort_data),
            "group_by": ("Group By", "Group data by field", self._group_by),
            "aggregate": ("Aggregate", "Aggregate data by function", self._aggregate)
        }
        self.initialized = True
        return True
    
    def shutdown(self) -> bool:
        self.initialized = False
        return True
    
    def get_transformations(self) -> List[Tuple[str, str, str]]:
        return [(id, name, desc) for id, (name, desc, _) in self.transformations.items()]
    
    def transform(self, transform_id: str, data: Any, **kwargs) -> Any:
        if transform_id not in self.transformations:
            raise ValueError(f"Unknown transformation: {transform_id}")
        
        _, _, transform_func = self.transformations[transform_id]
        return transform_func(data, **kwargs)
    
    # Transformation functions
    def _json_to_csv(self, data: List[Dict], **kwargs):
        """Convert JSON array to CSV"""
        if not data or not isinstance(data, list):
            return ""
        
        import csv
        import io
        
        # Get all keys from all items
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        # Sort keys for consistent output
        headers = sorted(all_keys)
        
        # Create CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        
        for item in data:
            if isinstance(item, dict):
                writer.writerow(item)
        
        return output.getvalue()
    
    def _csv_to_json(self, data: str, **kwargs):
        """Convert CSV to JSON array"""
        import csv
        import io
        
        input_file = io.StringIO(data)
        reader = csv.DictReader(input_file)
        return list(reader)
    
    def _filter_data(self, data: List[Dict], expression: str = "", **kwargs):
        """Filter data by expression"""
        if not data or not isinstance(data, list) or not expression:
            return data
        
        # Compile the filter expression
        code = compile(f"lambda item: {expression}", "<filter>", "eval")
        
        # Apply the filter
        filter_func = eval(code)
        return [item for item in data if filter_func(item)]
    
    def _sort_data(self, data: List[Dict], field: str = "", reverse: bool = False, **kwargs):
        """Sort data by field"""
        if not data or not isinstance(data, list) or not field:
            return data
        
        # Create key function
        def get_key(item):
            if not isinstance(item, dict):
                return None
            return item.get(field)
        
        # Sort the data
        return sorted(data, key=get_key, reverse=reverse)
    
    def _group_by(self, data: List[Dict], field: str = "", **kwargs):
        """Group data by field"""
        if not data or not isinstance(data, list) or not field:
            return {}
        
        result = {}
        for item in data:
            if not isinstance(item, dict):
                continue
                
            key = item.get(field)
            if key is None:
                continue
                
            # Convert key to string for JSON compatibility
            key_str = str(key)
            
            if key_str not in result:
                result[key_str] = []
            
            result[key_str].append(item)
        
        return result
    
    def _aggregate(self, data: List[Dict], field: str = "", function: str = "sum", **kwargs):
        """Aggregate data by function"""
        if not data or not isinstance(data, list) or not field:
            return None
        
        values = []
        for item in data:
            if not isinstance(item, dict):
                continue
                
            value = item.get(field)
            if value is not None:
                try:
                    values.append(float(value))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return None
        
        if function == "sum":
            return sum(values)
        elif function == "avg" or function == "mean":
            return sum(values) / len(values)
        elif function == "min":
            return min(values)
        elif function == "max":
            return max(values)
        elif function == "count":
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation function: {function}")

# Exporter plugin implementation example
class WorkflowExporterPlugin(ExporterPlugin):
    """Example implementation of a workflow exporter plugin"""
    
    def __init__(self):
        self._metadata = PluginMetadata(
            id="workflow_exporter_plugin",
            name="Workflow Exporters",
            version="1.0.0",
            description="Export workflows to various formats",
            author="System Builder Team",
            plugin_type=PluginType.EXPORTER,
            tags=["export", "workflow"]
        )
        self.initialized = False
        self.formats = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata
    
    def initialize(self) -> bool:
        # Register export formats
        self.formats = {
            "json": ("JSON", "json", self._export_json),
            "yaml": ("YAML", "yaml", self._export_yaml),
            "xml": ("XML", "xml", self._export_xml),
            "python": ("Python Script", "py", self._export_python),
            "dot": ("Graphviz DOT", "dot", self._export_dot),
            "html": ("HTML", "html", self._export_html)
        }
        self.initialized = True
        return True
    
    def shutdown(self) -> bool:
        self.initialized = False
        return True
    
    def get_export_formats(self) -> List[Tuple[str, str, str]]:
        return [(id, name, ext) for id, (name, ext, _) in self.formats.items()]
    
    def export_workflow(self, format_id: str, workflow: Any, filepath: str, **kwargs) -> bool:
        if format_id not in self.formats:
            raise ValueError(f"Unknown export format: {format_id}")
        
        _, _, export_func = self.formats[format_id]
        
        try:
            result = export_func(workflow, **kwargs)
            
            with open(filepath, 'w') as f:
                f.write(result)
            
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    # Export functions
    def _export_json(self, workflow: Any, pretty: bool = True, **kwargs) -> str:
        """Export workflow to JSON"""
        import json
        
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Export to JSON
        if pretty:
            return json.dumps(data, indent=2)
        else:
            return json.dumps(data)
    
    def _export_yaml(self, workflow: Any, **kwargs) -> str:
        """Export workflow to YAML"""
        import yaml
        
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Export to YAML
        return yaml.dump(data, default_flow_style=False)
    
    def _export_xml(self, workflow: Any, **kwargs) -> str:
        """Export workflow to XML"""
        import xml.dom.minidom as md
        import xml.etree.ElementTree as ET
        
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Helper function to convert dict to XML
        def dict_to_xml(tag, d):
            elem = ET.Element(tag)
            for key, val in d.items():
                if isinstance(val, dict):
                    elem.append(dict_to_xml(key, val))
                elif isinstance(val, list):
                    list_elem = ET.SubElement(elem, key)
                    for i, item in enumerate(val):
                        if isinstance(item, dict):
                            list_elem.append(dict_to_xml(f"item_{i}", item))
                        else:
                            item_elem = ET.SubElement(list_elem, f"item_{i}")
                            item_elem.text = str(item)
                else:
                    child = ET.SubElement(elem, key)
                    child.text = str(val)
            return elem
        
        # Create XML
        root = dict_to_xml("workflow", data)
        xml_str = ET.tostring(root, encoding='utf-8')
        
        # Pretty-print XML
        dom = md.parseString(xml_str)
        return dom.toprettyxml(indent="  ")
    
    def _export_python(self, workflow: Any, **kwargs) -> str:
        """Export workflow as Python script"""
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Generate Python script
        script = """#!/usr/bin/env python3
# Generated Workflow Script
# This script recreates the workflow using the System Builder API

import os
import sys
import json

# Add system builder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from graph_engine import WorkflowBuilder, WorkflowEngine
except ImportError:
    print("System Builder modules not found. Please ensure the script is run from the correct directory.")
    sys.exit(1)

def create_workflow():
    \"\"\"Create and execute the workflow\"\"\"
    # Create workflow builder
    builder = WorkflowBuilder()
    
"""
        
        # Add nodes
        nodes = data.get('nodes', {})
        for node_id, node_data in nodes.items():
            node_type = node_data.get('type', 'script')
            node_name = node_data.get('name', f"Node {node_id}")
            properties = node_data.get('properties', {})
            
            script += f"    # Add node: {node_name}\n"
            
            if node_type == 'script':
                script_path = properties.get('path', '')
                script_type = properties.get('script_type', 'python')
                args = properties.get('args', [])
                
                script += f"    {node_id} = builder.add_script(\n"
                script += f"        name=\"{node_name}\",\n"
                script += f"        path=\"{script_path}\",\n"
                script += f"        script_type=\"{script_type}\",\n"
                if args:
                    script += f"        args={json.dumps(args)}\n"
                script += f"    )\n"
                
            elif node_type == 'data':
                data_type = properties.get('data_type', 'static')
                
                script += f"    {node_id} = builder.add_data(\n"
                script += f"        name=\"{node_name}\",\n"
                
                if data_type == 'static':
                    script += f"        data={json.dumps(properties.get('value', ''))}\n"
                elif data_type == 'file':
                    script += f"        file_path=\"{properties.get('file_path', '')}\",\n"
                    script += f"        format_type=\"{properties.get('format', 'text')}\"\n"
                else:
                    script += f"        transform=\"{properties.get('transform', '')}\"\n"
                
                script += f"    )\n"
                
            elif node_type == 'computation':
                script += f"    {node_id} = builder.add_computation(\n"
                script += f"        name=\"{node_name}\",\n"
                script += f"        function=\"{properties.get('function', '')}\",\n"
                script += f"        params={json.dumps(properties.get('params', {}))}\n"
                script += f"    )\n"
                
            elif node_type == 'condition':
                script += f"    {node_id} = builder.add_condition(\n"
                script += f"        name=\"{node_name}\",\n"
                script += f"        expression=\"{properties.get('expression', 'True')}\"\n"
                script += f"    )\n"
                
            else:
                script += f"    {node_id} = builder.add_node(\n"
                script += f"        \"{node_type}\",\n"
                script += f"        name=\"{node_name}\",\n"
                script += f"        properties={json.dumps(properties)}\n"
                script += f"    )\n"
            
            script += "\n"
        
        # Add connections
        connections = data.get('connections', [])
        if connections:
            script += "    # Add connections\n"
            
            for conn in connections:
                source = conn.get('source_node', '')
                target = conn.get('target_node', '')
                conn_type = conn.get('type', 'control')
                
                script += f"    builder.connect(\n"
                script += f"        source_id=\"{source}\",\n"
                script += f"        target_id=\"{target}\",\n"
                script += f"        edge_type=\"{conn_type}\"\n"
                script += f"    )\n"
            
            script += "\n"
        
        # Build and execute workflow
        script += """    # Build the workflow graph
    graph = builder.build()
    
    return graph

def execute_workflow(graph):
    \"\"\"Execute the workflow\"\"\"
    # Create workflow engine
    engine = WorkflowEngine()
    engine.set_graph(graph)
    
    # Set context (customize as needed)
    engine.set_context({
        'working_dir': os.path.dirname(os.path.abspath(__file__))
    })
    
    # Start execution
    print("Starting workflow execution...")
    engine.start()
    
    # Wait for completion
    engine.wait()
    
    # Print results
    results = engine.get_results()
    print(f"Execution complete: {len(results)} nodes executed")
    
    # Check for errors
    errors = engine.get_errors()
    if errors:
        print(f"Errors occurred: {len(errors)} nodes failed")
        for node_id, error in errors.items():
            print(f"  Node {node_id}: {error}")
    
    # Stop engine
    engine.stop()
    
    return len(errors) == 0

if __name__ == "__main__":
    # Create workflow
    workflow = create_workflow()
    
    # Execute workflow
    success = execute_workflow(workflow)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
"""
        
        return script
    
    def _export_dot(self, workflow: Any, **kwargs) -> str:
        """Export workflow as Graphviz DOT file"""
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Generate DOT graph
        dot = "digraph Workflow {\n"
        dot += "    // Graph settings\n"
        dot += "    graph [rankdir=LR, fontname=\"Arial\", bgcolor=\"#FFFFFF\"];\n"
        dot += "    node [shape=box, style=filled, fontname=\"Arial\", color=\"#333333\"];\n"
        dot += "    edge [fontname=\"Arial\", color=\"#666666\"];\n\n"
        
        # Add nodes
        dot += "    // Nodes\n"
        nodes = data.get('nodes', {})
        for node_id, node_data in nodes.items():
            node_type = node_data.get('type', 'script')
            node_name = node_data.get('name', f"Node {node_id}")
            
            # Set color based on node type
            color = "#FFFFFF"
            if node_type == 'script':
                color = "#ADD8E6"  # Light blue
            elif node_type == 'data':
                color = "#90EE90"  # Light green
            elif node_type == 'computation':
                color = "#FFB6C1"  # Light red
            elif node_type == 'condition':
                color = "#FFFFE0"  # Light yellow
            
            dot += f"    n{node_id} [label=\"{node_name}\", fillcolor=\"{color}\"];\n"
        
        dot += "\n"
        
        # Add connections
        dot += "    // Connections\n"
        connections = data.get('connections', [])
        for i, conn in enumerate(connections):
            source = conn.get('source_node', '')
            target = conn.get('target_node', '')
            conn_type = conn.get('type', 'control')
            
            # Set edge style based on connection type
            style = "solid"
            if conn_type == 'data':
                style = "dashed"
            
            dot += f"    n{source} -> n{target} [style={style}, label=\"{conn_type}\"];\n"
        
        dot += "}\n"
        
        return dot
    
    def _export_html(self, workflow: Any, **kwargs) -> str:
        """Export workflow as interactive HTML file"""
        # Convert workflow to dict
        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
        else:
            data = workflow
        
        # Generate HTML with embedded workflow data
        workflow_json = json.dumps(data)
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        #container {
            display: flex;
            height: 100vh;
        }
        #sidebar {
            width: 300px;
            padding: 15px;
            background-color: #fff;
            border-right: 1px solid #ddd;
            overflow-y: auto;
        }
        #graph {
            flex: 1;
            background-color: #fff;
            overflow: hidden;
        }
        h1, h2 {
            margin-top: 0;
            color: #333;
        }
        .node-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .node-item {
            padding: 8px 10px;
            margin-bottom: 5px;
            background-color: #f0f0f0;
            border-radius: 4px;
            cursor: pointer;
        }
        .node-item:hover {
            background-color: #e0e0e0;
        }
        .node-item.selected {
            background-color: #d0e0ff;
        }
        .properties {
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f5f5f5;
        }
        .node {
            cursor: pointer;
            rx: 6;
            ry: 6;
        }
        .node:hover {
            stroke-width: 2px;
            stroke: #666;
        }
        .node-text {
            font-size: 12px;
            pointer-events: none;
        }
        .connection {
            stroke-width: 2px;
        }
        .connection-text {
            font-size: 10px;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>Workflow Viewer</h1>
            <h2>Nodes</h2>
            <ul class="node-list" id="nodeList"></ul>
            <div class="properties" id="properties">
                <h2>Properties</h2>
                <div id="propertyContent">Select a node to view its properties</div>
            </div>
        </div>
        <div id="graph">
            <svg id="graphSvg" width="100%" height="100%"></svg>
        </div>
    </div>

    <script>
        // Workflow data
        const workflowData = """ + workflow_json + """;
        
        // Graph rendering code
        const svg = document.getElementById('graphSvg');
        const nodeList = document.getElementById('nodeList');
        const propertiesDiv = document.getElementById('propertyContent');
        
        let selectedNode = null;
        
        // Convert workflow data to graph format
        const nodes = Object.entries(workflowData.nodes || {}).map(([id, data]) => ({
            id,
            name: data.name || `Node ${id}`,
            type: data.type || 'unknown',
            properties: data.properties || {}
        }));
        
        const connections = (workflowData.connections || []).map(conn => ({
            source: conn.source_node,
            target: conn.target_node,
            type: conn.type || 'control'
        }));
        
        // Render the graph
        function renderGraph() {
            // Clear the SVG
            svg.innerHTML = '';
            
            // Get SVG dimensions
            const width = svg.clientWidth;
            const height = svg.clientHeight;
            
            // Create a simple force-directed layout
            const nodeRadius = 30;
            const nodeSpacing = 150;
            
            // Position nodes in a grid layout
            const nodesPerRow = Math.ceil(Math.sqrt(nodes.length));
            nodes.forEach((node, index) => {
                const row = Math.floor(index / nodesPerRow);
                const col = index % nodesPerRow;
                node.x = (col + 1) * nodeSpacing;
                node.y = (row + 1) * nodeSpacing;
                node.width = nodeRadius * 2;
                node.height = nodeRadius * 2;
            });
            
            // Create a group for the graph
            const graph = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            svg.appendChild(graph);
            
            // Add connections
            connections.forEach(conn => {
                const source = nodes.find(n => n.id === conn.source);
                const target = nodes.find(n => n.id === conn.target);
                
                if (source && target) {
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', source.x + nodeRadius);
                    line.setAttribute('y1', source.y + nodeRadius);
                    line.setAttribute('x2', target.x + nodeRadius);
                    line.setAttribute('y2', target.y + nodeRadius);
                    line.setAttribute('class', 'connection');
                    line.setAttribute('stroke', conn.type === 'data' ? '#4CAF50' : '#2196F3');
                    
                    // Add arrow marker
                    const markerId = `arrow-${conn.source}-${conn.target}`;
                    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                    marker.setAttribute('id', markerId);
                    marker.setAttribute('viewBox', '0 0 10 10');
                    marker.setAttribute('refX', '5');
                    marker.setAttribute('refY', '5');
                    marker.setAttribute('markerWidth', '6');
                    marker.setAttribute('markerHeight', '6');
                    marker.setAttribute('orient', 'auto');
                    
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    path.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
                    path.setAttribute('fill', conn.type === 'data' ? '#4CAF50' : '#2196F3');
                    
                    marker.appendChild(path);
                    defs.appendChild(marker);
                    svg.appendChild(defs);
                    
                    line.setAttribute('marker-end', `url(#${markerId})`);
                    
                    graph.appendChild(line);
                    
                    // Add connection label
                    const labelX = (source.x + target.x) / 2 + nodeRadius;
                    const labelY = (source.y + target.y) / 2 + nodeRadius;
                    
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.setAttribute('x', labelX);
                    label.setAttribute('y', labelY);
                    label.setAttribute('class', 'connection-text');
                    label.setAttribute('text-anchor', 'middle');
                    label.setAttribute('dy', '-5');
                    label.textContent = conn.type;
                    
                    graph.appendChild(label);
                }
            });
            
            // Add nodes
            nodes.forEach(node => {
                const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                group.setAttribute('transform', `translate(${node.x}, ${node.y})`);
                
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('width', node.width);
                rect.setAttribute('height', node.height);
                rect.setAttribute('class', 'node');
                
                // Set color based on node type
                let color = '#BBBBBB';
                switch (node.type) {
                    case 'script': color = '#ADD8E6'; break;
                    case 'data': color = '#90EE90'; break;
                    case 'computation': color = '#FFB6C1'; break;
                    case 'condition': color = '#FFFFE0'; break;
                }
                
                rect.setAttribute('fill', color);
                rect.setAttribute('stroke', '#333');
                rect.setAttribute('stroke-width', selectedNode === node.id ? '2' : '1');
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', node.width / 2);
                text.setAttribute('y', node.height / 2);
                text.setAttribute('class', 'node-text');
                text.setAttribute('text-anchor', 'middle');
                text.setAttribute('dominant-baseline', 'middle');
                text.textContent = node.name;
                
                group.appendChild(rect);
                group.appendChild(text);
                
                group.addEventListener('click', () => selectNode(node.id));
                
                graph.appendChild(group);
            });
            
            // Add pan and zoom functionality
            let isPanning = false;
            let startPoint = { x: 0, y: 0 };
            let endPoint = { x: 0, y: 0 };
            let scale = 1;
            
            svg.addEventListener('mousedown', (event) => {
                if (event.button === 0) {
                    isPanning = true;
                    startPoint = { x: event.clientX, y: event.clientY };
                }
            });
            
            svg.addEventListener('mousemove', (event) => {
                if (isPanning) {
                    endPoint = { x: event.clientX, y: event.clientY };
                    const dx = endPoint.x - startPoint.x;
                    const dy = endPoint.y - startPoint.y;
                    
                    const viewBox = svg.viewBox.baseVal;
                    viewBox.x -= dx / scale;
                    viewBox.y -= dy / scale;
                    
                    startPoint = endPoint;
                }
            });
            
            svg.addEventListener('mouseup', () => {
                isPanning = false;
            });
            
            svg.addEventListener('wheel', (event) => {
                event.preventDefault();
                const delta = event.deltaY < 0 ? 1.1 : 0.9;
                scale *= delta;
                
                const viewBox = svg.viewBox.baseVal;
                const mouseX = event.clientX;
                const mouseY = event.clientY;
                
                // Calculate the point to zoom to
                const point = svg.createSVGPoint();
                point.x = mouseX;
                point.y = mouseY;
                const svgPoint = point.matrixTransform(svg.getScreenCTM().inverse());
                
                viewBox.x = svgPoint.x - (svgPoint.x - viewBox.x) * delta;
                viewBox.y = svgPoint.y - (svgPoint.y - viewBox.y) * delta;
                viewBox.width *= delta;
                viewBox.height *= delta;
            });
            
            // Set initial viewBox
            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        }
        
        // Render the node list
        function renderNodeList() {
            nodeList.innerHTML = '';
            
            nodes.forEach(node => {
                const li = document.createElement('li');
                li.textContent = node.name;
                li.className = 'node-item';
                if (selectedNode === node.id) {
                    li.classList.add('selected');
                }
                li.addEventListener('click', () => selectNode(node.id));
                nodeList.appendChild(li);
            });
        }
        
        // Render properties for a selected node
        function renderProperties(nodeId) {
            const node = nodes.find(n => n.id === nodeId);
            if (!node) {
                propertiesDiv.textContent = 'No node selected';
                return;
            }
            
            propertiesDiv.innerHTML = '';
            
            // Add node info
            const table = document.createElement('table');
            
            // Add basic properties
            addTableRow(table, 'ID', node.id);
            addTableRow(table, 'Name', node.name);
            addTableRow(table, 'Type', node.type);
            
            // Add custom properties
            const properties = node.properties;
            if (properties && Object.keys(properties).length > 0) {
                Object.entries(properties).forEach(([key, value]) => {
                    addTableRow(table, key, JSON.stringify(value));
                });
            }
            
            propertiesDiv.appendChild(table);
        }
        
        function addTableRow(table, key, value) {
            const row = document.createElement('tr');
            
            const keyCell = document.createElement('th');
            keyCell.textContent = key;
            row.appendChild(keyCell);
            
            const valueCell = document.createElement('td');
            valueCell.textContent = value;
            row.appendChild(valueCell);
            
            table.appendChild(row);
        }
        
        function selectNode(nodeId) {
            selectedNode = nodeId;
            renderNodeList();
            renderProperties(nodeId);
            renderGraph();
        }
        
        // Initial render
        renderGraph();
        renderNodeList();
        
        // Handle window resize
        window.addEventListener('resize', renderGraph);
    </script>
</body>
</html>
"""
        
        return html

# Plugin system initialization
def init_plugin_system(base_dir: str) -> PluginManager:
    """Initialize the plugin system"""
    # Create plugin manager
    plugin_manager = PluginManager()
    
    # Add plugin directories
    plugin_dirs = [
        os.path.join(base_dir, 'plugins'),
        os.path.join(os.path.expanduser('~'), '.systembuilder', 'plugins')
    ]
    
    for plugin_dir in plugin_dirs:
        plugin_manager.add_plugin_directory(plugin_dir)
    
    # Register built-in plugins
    register_builtin_plugins(plugin_manager)
    
    return plugin_manager

def register_builtin_plugins(plugin_manager: PluginManager) -> None:
    """Register built-in plugins"""
    # Create a temporary directory for storing plugin modules
    import tempfile
    import atexit
    
    temp_dir = tempfile.mkdtemp(prefix='systembuilder_plugins_')
    plugin_manager.add_plugin_directory(temp_dir)
    
    # Register cleanup function
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    
    # Create built-in plugins
    plugins = [
        # Create the node plugin module
        {
            'filename': 'custom_node_plugin.py',
            'content': inspect.getsource(CustomNodePlugin)
        },
        # Create the transform plugin module
        {
            'filename': 'data_transform_plugin.py',
            'content': inspect.getsource(DataTransformPlugin)
        },
        # Create the exporter plugin module
        {
            'filename': 'workflow_exporter_plugin.py',
            'content': inspect.getsource(WorkflowExporterPlugin)
        }
    ]
    
    # Write plugin files
    for plugin in plugins:
        plugin_path = os.path.join(temp_dir, plugin['filename'])
        with open(plugin_path, 'w') as f:
            f.write(plugin['content'])
    
    # Discover and load plugins
    plugin_manager.load_plugins()

# Plugin usage example
def example_plugin_usage() -> None:
    """Example of using the plugin system"""
    # Initialize plugin system
    base_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_manager = init_plugin_system(base_dir)
    
    # Get all loaded plugins
    plugins = plugin_manager.plugins
    print(f"Loaded {len(plugins)} plugins:")
    for plugin_id, plugin in plugins.items():
        print(f"  - {plugin.metadata.name} (ID: {plugin_id}, Type: {plugin.metadata.plugin_type.name})")
    
    # Use node plugin
    node_plugins = plugin_manager.get_plugins_by_type(PluginType.NODE)
    if node_plugins:
        for plugin_id, plugin in node_plugins.items():
            print(f"\nNode types from {plugin.metadata.name}:")
            node_types = plugin.get_node_types()
            for type_id, name, _ in node_types:
                print(f"  - {name} (ID: {type_id})")
            
            # Create a node
            if node_types:
                type_id, _, _ = node_types[0]
                node = plugin.create_node(type_id, url="https://example.com")
                print(f"\nCreated node of type {type_id}: {node}")
    
    # Use transform plugin
    transform_plugins = plugin_manager.get_plugins_by_type(PluginType.TRANSFORM)
    if transform_plugins:
        for plugin_id, plugin in transform_plugins.items():
            print(f"\nTransformations from {plugin.metadata.name}:")
            transformations = plugin.get_transformations()
            for transform_id, name, desc in transformations:
                print(f"  - {name} (ID: {transform_id}): {desc}")
            
            # Apply a transformation
            if transformations:
                transform_id, _, _ = transformations[0]
                data = [{"name": "Item 1", "value": 10}, {"name": "Item 2", "value": 20}]
                result = plugin.transform(transform_id, data)
                print(f"\nTransformed data with {transform_id}: {result}")
    
    # Use exporter plugin
    exporter_plugins = plugin_manager.get_plugins_by_type(PluginType.EXPORTER)
    if exporter_plugins:
        for plugin_id, plugin in exporter_plugins.items():
            print(f"\nExport formats from {plugin.metadata.name}:")
            formats = plugin.get_export_formats()
            for format_id, name, ext in formats:
                print(f"  - {name} (ID: {format_id}, Extension: {ext})")
    
    # Shutdown plugins
    plugin_manager.shutdown_all()

if __name__ == "__main__":
    example_plugin_usage()
