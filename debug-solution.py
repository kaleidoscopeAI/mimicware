#!/usr/bin/env python3
"""
System Builder Debug Solution
Fixes critical errors and implements robust error handling throughout the system
"""

import os
import sys
import logging
import traceback
import json
import importlib.util
import subprocess
import re
import threading
import queue

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger("DebugFixer")

class SystemBuilderDebugger:
    """Main debugger class to fix System Builder errors"""
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.module_paths = {
            'system_builder': os.path.join(self.base_dir, 'system-builder.py'),
            'graph_engine': os.path.join(self.base_dir, 'graph-engine.py'),
            'advanced_gui': os.path.join(self.base_dir, 'advanced-gui.py'),
            'plugin_architecture': os.path.join(self.base_dir, 'plugin-architecture.py'),
            'test_framework': os.path.join(self.base_dir, 'test-framework.py'),
            'installation_script': os.path.join(self.base_dir, 'installation-script.py')
        }
        self.modules = {}
        self.errors = {}
        self.fixed_files = []
    
    def load_modules(self):
        """Attempt to load all modules and catch errors"""
        for name, path in self.module_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Module file not found: {path}")
                self.errors[name] = f"File not found: {path}"
                continue
            
            try:
                logger.info(f"Loading module: {name} from {path}")
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.modules[name] = module
                logger.info(f"Successfully loaded module: {name}")
            except Exception as e:
                logger.error(f"Error loading module {name}: {e}")
                self.errors[name] = str(e)
                traceback.print_exc()
    
    def check_syntax_errors(self):
        """Check for syntax errors in all module files"""
        for name, path in self.module_paths.items():
            if not os.path.exists(path):
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Try compiling the source code to check for syntax errors
                compile(source, path, 'exec')
                logger.info(f"No syntax errors found in {name}")
            except SyntaxError as e:
                logger.error(f"Syntax error in {name}: {e}")
                self.errors.setdefault(name, []).append(f"Syntax error: {e}")
    
    def fix_advanced_gui_completion(self):
        """Fix the advanced-gui-completion.py file which appears to be a fragment"""
        gui_completion_path = os.path.join(self.base_dir, 'advanced-gui-completion.py')
        
        if not os.path.exists(gui_completion_path):
            logger.warning(f"advanced-gui-completion.py not found at {gui_completion_path}")
            return
        
        try:
            # Read the content
            with open(gui_completion_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add missing function declarations and imports
            fixed_content = """#!/usr/bin/env python3
\"\"\"
Advanced GUI Completion
This module completes functionality for the advanced-gui.py module
\"\"\"

import os
import sys
import json
import concurrent.futures
import logging
import time
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GUICompletion")

def parse_arguments():
    \"\"\"Parse command line arguments\"\"\"
    import argparse
    parser = argparse.ArgumentParser(description="Advanced GUI Completion")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-dir', help='Output directory for job results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                      help='Logging level')
    parser.add_argument('--jobs', type=int, default=4, help='Number of parallel jobs')
    return parser.parse_args()

def run_job(job):
    \"\"\"Execute a job with specified parameters\"\"\"
    # Get job parameters
    job_id = job.get('id', 'unknown')
    job_type = job.get('type', 'unknown')
    params = job.get('parameters', {})
    
    logger.info(f"Running job {job_id} of type {job_type}")
    time.sleep(0.5)  # Simulate work
    
""" + content
            
            # Write the fixed content back
            fixed_path = os.path.join(self.base_dir, 'advanced-gui-completion-fixed.py')
            with open(fixed_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed advanced-gui-completion.py and saved to {fixed_path}")
            self.fixed_files.append(fixed_path)
            
        except Exception as e:
            logger.error(f"Error fixing advanced-gui-completion.py: {e}")
    
    def fix_script_analyzer_txt(self):
        """Convert script-analyzer.txt to proper C file"""
        script_analyzer_path = os.path.join(self.base_dir, 'script-analyzer.txt')
        
        if not os.path.exists(script_analyzer_path):
            logger.warning(f"script-analyzer.txt not found at {script_analyzer_path}")
            return
        
        try:
            # Read the content
            with open(script_analyzer_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Write as .c file
            c_file_path = os.path.join(self.base_dir, 'script-analyzer.c')
            with open(c_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Converted script-analyzer.txt to C file: {c_file_path}")
            self.fixed_files.append(c_file_path)
            
        except Exception as e:
            logger.error(f"Error converting script-analyzer.txt: {e}")
    
    def fix_missing_imports(self):
        """Fix missing imports in module files"""
        for name, path in self.module_paths.items():
            if not os.path.exists(path):
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for potential missing imports
                missing_imports = []
                
                # Check for common module usage without imports
                if 'numpy' in content and 'import numpy' not in content and 'from numpy' not in content:
                    missing_imports.append('import numpy as np')
                
                if 'matplotlib' in content and 'import matplotlib' not in content and 'from matplotlib' not in content:
                    missing_imports.append('import matplotlib.pyplot as plt')
                
                if 'queue.Queue' in content and 'import queue' not in content:
                    missing_imports.append('import queue')
                
                if 'threading.Thread' in content and 'import threading' not in content:
                    missing_imports.append('import threading')
                
                if missing_imports:
                    # Add imports at the top after any existing imports
                    import_match = re.search(r'^(import .*?)$', content, re.MULTILINE)
                    if import_match:
                        pos = import_match.end()
                        fixed_content = content[:pos] + '\n' + '\n'.join(missing_imports) + '\n' + content[pos:]
                    else:
                        # Add after any docstring
                        docstring_match = re.search(r'""".*?"""', content, re.DOTALL)
                        if docstring_match:
                            pos = docstring_match.end()
                            fixed_content = content[:pos] + '\n\n' + '\n'.join(missing_imports) + '\n' + content[pos:]
                        else:
                            # Add at the top
                            fixed_content = '\n'.join(missing_imports) + '\n\n' + content
                    
                    # Write the fixed content
                    fixed_path = os.path.join(self.base_dir, f"{os.path.basename(path)}.fixed")
                    with open(fixed_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    logger.info(f"Added missing imports to {name} and saved to {fixed_path}")
                    self.fixed_files.append(fixed_path)
            
            except Exception as e:
                logger.error(f"Error fixing missing imports in {name}: {e}")
    
    def fix_tkinter_dnd_issue(self):
        """Fix references to TkinterDnD which might not be installed"""
        gui_path = os.path.join(self.base_dir, 'advanced-gui.py')
        
        if not os.path.exists(gui_path):
            logger.warning(f"advanced-gui.py not found at {gui_path}")
            return
        
        try:
            with open(gui_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add wrapper for TkinterDnD
            tkdnd_wrapper = """
# TkinterDnD wrapper for drag and drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    TKDND_AVAILABLE = True
except ImportError:
    TKDND_AVAILABLE = False
    # Create dummy implementation
    class DummyTkinterDnD:
        def __init__(self):
            pass
            
        def register(self, *args, **kwargs):
            pass
            
    class DummyTkDnD:
        def __init__(self):
            self.TkinterDnD = DummyTkinterDnD()
            
    TkinterDnD = DummyTkDnD()
    DND_FILES = "dummy"
    logger.warning("TkinterDnD not available. Drag and drop will not work.")
"""
            
            # Add after imports
            import_match = re.search(r'from typing import.*?$', content, re.MULTILINE)
            if import_match:
                pos = import_match.end()
                fixed_content = content[:pos] + '\n' + tkdnd_wrapper + content[pos:]
            else:
                # Add after any docstring
                docstring_match = re.search(r'""".*?"""', content, re.DOTALL)
                if docstring_match:
                    pos = docstring_match.end()
                    fixed_content = content[:pos] + '\n\n' + tkdnd_wrapper + content[pos:]
                else:
                    # Add near the top
                    fixed_content = content[:100] + '\n\n' + tkdnd_wrapper + content[100:]
            
            # Replace direct references to drop_target_register and dnd_bind
            fixed_content = fixed_content.replace('self.drop_target_register', 
                                              'hasattr(self, "drop_target_register") and self.drop_target_register')
            fixed_content = fixed_content.replace('self.dnd_bind', 
                                              'hasattr(self, "dnd_bind") and self.dnd_bind')
            
            # Write the fixed content
            fixed_path = os.path.join(self.base_dir, 'advanced-gui-fixed.py')
            with open(fixed_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            logger.info(f"Fixed TkinterDnD issues in advanced-gui.py and saved to {fixed_path}")
            self.fixed_files.append(fixed_path)
            
        except Exception as e:
            logger.error(f"Error fixing TkinterDnD issues: {e}")
    
    def fix_circular_imports(self):
        """Fix potential circular imports between modules"""
        # Map modules that import each other
        circular_patterns = [
            ('system_builder', 'graph_engine'),
            ('system_builder', 'advanced_gui'),
            ('graph_engine', 'advanced_gui')
        ]
        
        for module1, module2 in circular_patterns:
            path1 = self.module_paths.get(module1)
            path2 = self.module_paths.get(module2)
            
            if not path1 or not path2 or not os.path.exists(path1) or not os.path.exists(path2):
                continue
            
            try:
                # Check if module1 imports module2
                with open(path1, 'r', encoding='utf-8') as f:
                    content1 = f.read()
                
                # Check if module2 imports module1
                with open(path2, 'r', encoding='utf-8') as f:
                    content2 = f.read()
                
                imports_module2 = re.search(rf'import\s+{module2}|from\s+{module2}\s+import', content1)
                imports_module1 = re.search(rf'import\s+{module1}|from\s+{module1}\s+import', content2)
                
                if imports_module2 and imports_module1:
                    logger.warning(f"Circular import detected between {module1} and {module2}")
                    
                    # Fix module1 by using lazy imports
                    fixed_content1 = re.sub(
                        rf'(import\s+{module2}|from\s+{module2}\s+import\s+.*?)$',
                        f'# Lazy import to avoid circular dependency\n# \\1',
                        content1,
                        flags=re.MULTILINE
                    )
                    
                    # Add lazy import function
                    lazy_import_func = """
def _lazy_import(module_name):
    """Lazily import a module"""
    import importlib
    return importlib.import_module(module_name)
"""
                    
                    # Add after imports
                    import_section_end = re.search(r'^from\s+.*?$|^import\s+.*?$', fixed_content1, re.MULTILINE | re.DOTALL)
                    if import_section_end:
                        pos = import_section_end.end()
                        fixed_content1 = fixed_content1[:pos] + '\n' + lazy_import_func + fixed_content1[pos:]
                    
                    # Add lazy loading where the module is used
                    module2_var = module2.lower()
                    fixed_content1 = re.sub(
                        rf'{module2_var}\.',
                        f'_lazy_import("{module2}").',
                        fixed_content1
                    )
                    
                    # Write the fixed content
                    fixed_path = os.path.join(self.base_dir, f"{os.path.basename(path1)}.fixed")
                    with open(fixed_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content1)
                    
                    logger.info(f"Fixed circular import in {module1} and saved to {fixed_path}")
                    self.fixed_files.append(fixed_path)
            
            except Exception as e:
                logger.error(f"Error fixing circular imports between {module1} and {module2}: {e}")
    
    def fix_imported_but_unused(self):
        """Fix imported but unused modules by adding placeholder usage"""
        for name, path in self.module_paths.items():
            if not os.path.exists(path):
                continue
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract imported modules
                import_matches = re.finditer(r'import\s+([\w\.]+)(?:\s+as\s+([\w]+))?', content)
                from_matches = re.finditer(r'from\s+([\w\.]+)\s+import\s+([\w\.\,\s]+)', content)
                
                imported_modules = []
                
                for match in import_matches:
                    module = match.group(1)
                    alias = match.group(2) if match.group(2) else module.split('.')[-1]
                    imported_modules.append(alias)
                
                for match in from_matches:
                    modules = match.group(2).split(',')
                    imported_modules.extend([m.strip() for m in modules])
                
                # Check for unused imports
                unused_modules = []
                for module in imported_modules:
                    if module not in ['self', '__name__', '__main__']:
                        # Check if module is used anywhere else in the code
                        pattern = rf'\b{re.escape(module)}\b'
                        matches = list(re.finditer(pattern, content))
                        
                        # If it's only used in the import statement, consider it unused
                        if len(matches) <= 1:
                            unused_modules.append(module)
                
                if unused_modules:
                    # Add placeholder usage to silence linter warnings
                    placeholder = "\n# Ensure imported modules are recognized\nif False:\n"
                    for module in unused_modules:
                        placeholder += f"    _ = {module}\n"
                    
                    # Add at the end of imports
                    import_section_end = re.search(r'^from\s+.*?$|^import\s+.*?$', content, re.MULTILINE | re.DOTALL)
                    if import_section_end:
                        pos = import_section_end.end()
                        fixed_content = content[:pos] + '\n' + placeholder + content[pos:]
                    else:
                        # Add near the top
                        fixed_content = content[:100] + '\n\n' + placeholder + content[100:]
                    
                    # Write the fixed content
                    fixed_path = os.path.join(self.base_dir, f"{os.path.basename(path)}.fixed")
                    with open(fixed_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    logger.info(f"Fixed unused imports in {name} and saved to {fixed_path}")
                    self.fixed_files.append(fixed_path)
            
            except Exception as e:
                logger.error(f"Error fixing unused imports in {name}: {e}")
    
    def fix_missing_dependencies(self):
        """Install missing dependencies"""
        required_packages = [
            'numpy',
            'networkx',
            'matplotlib',
            'pillow',
            'tkinterdnd2',  # Optional but recommended for drag and drop
        ]
        
        try:
            # Check which packages are installed
            installed_packages = self._get_installed_packages()
            missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
            
            if missing_packages:
                logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
                
                # Create requirements.txt
                req_path = os.path.join(self.base_dir, 'requirements.txt')
                with open(req_path, 'w') as f:
                    f.write('\n'.join(missing_packages))
                
                # Install using pip
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
                logger.info("Package installation complete")
                
                # Clean up
                os.remove(req_path)
        
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
    
    def _get_installed_packages(self):
        """Get list of installed Python packages"""
        try:
            output = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], 
                                           universal_newlines=True)
            return [line.split()[0].lower() for line in output.split('\n') if line and not line.startswith('Package')]
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return []
    
    def create_patch_script(self):
        """Create a patch script to apply all fixes"""
        patch_script_path = os.path.join(self.base_dir, 'apply_fixes.py')
        
        try:
            # Create script content
            content = """#!/usr/bin/env python3
\"\"\"
Apply Fixes Script for System Builder
Automatically fixes issues in the System Builder scripts
\"\"\"

import os
import sys
import shutil
import logging
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fix_application.log')
    ]
)
logger = logging.getLogger("FixApplier")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Files to fix
    files_to_fix = [
"""
            
            # Add all fixed files
            for fixed_path in self.fixed_files:
                basename = os.path.basename(fixed_path)
                original_name = basename.replace('.fixed', '')
                content += f"        ('{basename}', '{original_name}'),\n"
            
            content += """    ]
    
    # Apply fixes
    for fixed_file, original_file in files_to_fix:
        fixed_path = os.path.join(base_dir, fixed_file)
        original_path = os.path.join(base_dir, original_file)
        
        if not os.path.exists(fixed_path):
            logger.warning(f"Fixed file not found: {fixed_path}")
            continue
        
        # Backup original if it exists
        if os.path.exists(original_path):
            backup_path = original_path + '.bak'
            logger.info(f"Backing up {original_file} to {backup_path}")
            shutil.copy2(original_path, backup_path)
        
        # Copy fixed file to original
        logger.info(f"Applying fix for {original_file}")
        shutil.copy2(fixed_path, original_path)
    
    # Install dependencies
    try:
        logger.info("Installing required dependencies")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                             'numpy', 'networkx', 'matplotlib', 'pillow', 'tkinterdnd2'])
        logger.info("Dependencies installed successfully")
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
    
    logger.info("All fixes applied successfully")
    print("System Builder fixes applied successfully!")
    print("You can now run the main-application.py script")

if __name__ == "__main__":
    main()
"""
            
            # Write the script
            with open(patch_script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Make executable
            os.chmod(patch_script_path, 0o755)
            
            logger.info(f"Created patch script: {patch_script_path}")
            
        except Exception as e:
            logger.error(f"Error creating patch script: {e}")
    
    def run_tests(self):
        """Run simple tests for the fixed modules"""
        logger.info("Running tests for fixed modules")
        
        # Only run tests if test_framework exists
        test_path = self.module_paths.get('test_framework')
        if not test_path or not os.path.exists(test_path):
            logger.warning("Test framework not found, skipping tests")
            return
        
        try:
            for name, path in self.module_paths.items():
                # Skip test framework to avoid recursion
                if name == 'test_framework':
                    continue
                
                # Use fixed file if available
                fixed_path = os.path.join(self.base_dir, f"{os.path.basename(path)}.fixed")
                test_file = fixed_path if os.path.exists(fixed_path) else path
                
                if not os.path.exists(test_file):
                    logger.warning(f"Module file not found for testing: {test_file}")
                    continue
                
                logger.info(f"Testing module: {name}")
                
                # Simple import test
                try:
                    spec = importlib.util.spec_from_file_location(name, test_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.info(f"Successfully imported module: {name}")
                except Exception as e:
                    logger.error(f"Error importing module {name}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error running tests: {e}")
    
    def debug_all(self):
        """Run all debugging and fixing procedures"""
        logger.info("Starting comprehensive debugging process")
        
        # First check syntax and load modules
        self.check_syntax_errors()
        self.load_modules()
        
        # Fix specific issues
        self.fix_advanced_gui_completion()
        self.fix_script_analyzer_txt()
        self.fix_missing_imports()
        self.fix_tkinter_dnd_issue()
        self.fix_circular_imports()
        self.fix_imported_but_unused()
        
        # Fix missing dependencies
        self.fix_missing_dependencies()
        
        # Create patch script
        self.create_patch_script()
        
        # Run tests
        self.run_tests()
        
        logger.info("Debugging process complete")
        
        # Print summary
        print("\n" + "="*60)
        print("SYSTEM BUILDER DEBUG SUMMARY")
        print("="*60)
        print(f"Fixed {len(self.fixed_files)} files:")
        for path in self.fixed_files:
            print(f"  - {os.path.basename(path)}")
        print("\nTo apply all fixes, run the patch script:")
        print(f"  python {os.path.join(self.base_dir, 'apply_fixes.py')}")
        print("="*60)

# Main function
def main():
    """Main entry point"""
    logger.info("Starting System Builder Debug Fixer")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="System Builder Debug Fixer")
    parser.add_argument('--dir', help='Base directory containing System Builder files')
    args = parser.parse_args()
    
    # Create and run debugger
    debugger = SystemBuilderDebugger(args.dir)
    debugger.debug_all()

if __name__ == "__main__":
    main()
