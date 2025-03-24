import os
import ast
import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdaptiveSecurityAnalyzer")

class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    INJECTION = auto()
    XSS = auto()
    BROKEN_AUTH = auto()
    SENSITIVE_DATA_EXPOSURE = auto()
    XXE = auto()
    BROKEN_ACCESS_CONTROL = auto()
    SECURITY_MISCONFIGURATION = auto()
    INSECURE_DESERIALIZATION = auto()
    USING_COMPONENTS_WITH_VULNERABILITIES = auto()
    INSUFFICIENT_LOGGING_MONITORING = auto()
    BUFFER_OVERFLOW = auto()
    MEMORY_LEAK = auto()
    RACE_CONDITION = auto()
    CRYPTOGRAPHIC_FAILURE = auto()
    UNVALIDATED_REDIRECT = auto()

@dataclass
class SecurityIssue:
    """Represents a detected security issue"""
    # Type of vulnerability
    vulnerability_type: VulnerabilityType
    # Severity level (0.0-1.0)
    severity: float
    # Location in code (file, line numbers)
    location: Dict[str, Any]
    # Description of the issue
    description: str
    # How to remediate
    remediation: str
    # Code snippet
    code_snippet: str = ""
    # CWE identifier if applicable
    cwe_id: Optional[str] = None
    # CVSS score if applicable
    cvss_score: Optional[float] = None
    # Reference to related dependencies/components
    related_components: List[str] = field(default_factory=list)
    # Confidence level (0.0-1.0)
    confidence: float = 0.8

@dataclass
class SecurityContext:
    """Security context for a software project"""
    # Programming language
    language: str
    # Framework(s) in use
    frameworks: List[str] = field(default_factory=list)
    # Dependencies and their versions
    dependencies: Dict[str, str] = field(default_factory=dict)
    # Input sources
    input_sources: List[str] = field(default_factory=list)
    # Output sinks
    output_sinks: List[str] = field(default_factory=list)
    # Authentication mechanisms
    auth_mechanisms: List[str] = field(default_factory=list)
    # Authorization mechanisms
    authorization_mechanisms: List[str] = field(default_factory=list)
    # Sensitive data types found in code
    sensitive_data_types: List[str] = field(default_factory=list)
    # Security controls already implemented
    security_controls: Dict[str, bool] = field(default_factory=dict)
    # External services in use
    external_services: List[str] = field(default_factory=list)

class AdaptiveSecurityAnalyzer:
    """
    Adaptive security analyzer that identifies potential vulnerabilities
    during software reconstruction and suggests mitigations
    """
    
    def __init__(self):
        """Initialize the adaptive security analyzer"""
        # Initialize vulnerability detectors
        self._init_vulnerability_detectors()
        
        # Initialize security rules for each language
        self._init_security_rules()
        
        # Initialize security patterns 
        self._init_security_patterns()
        
        # Thread pool for parallel analysis
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Pattern cache for performance
        self.pattern_cache = {}
        
        # Vulnerability database
        self.vuln_db = self._load_vulnerability_database()
        
        # Default security context
        self.default_context = SecurityContext(language="unknown")
    
    def _init_vulnerability_detectors(self):
        """Initialize specialized vulnerability detectors for each type"""
        self.detectors = {
            VulnerabilityType.INJECTION: self._detect_injection,
            VulnerabilityType.XSS: self._detect_xss,
            VulnerabilityType.BROKEN_AUTH: self._detect_broken_auth,
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: self._detect_sensitive_data_exposure,
            VulnerabilityType.BROKEN_ACCESS_CONTROL: self._detect_broken_access_control,
            VulnerabilityType.SECURITY_MISCONFIGURATION: self._detect_security_misconfiguration,
            VulnerabilityType.BUFFER_OVERFLOW: self._detect_buffer_overflow,
            VulnerabilityType.CRYPTOGRAPHIC_FAILURE: self._detect_crypto_failure
        }
    
    def _init_security_rules(self):
        """Initialize security rules for different languages"""
        self.security_rules = {
            "python": {
                "input_functions": ["input", "request.form", "request.args", "request.get_json"],
                "sql_functions": ["execute", "cursor.execute", "session.execute"],
                "output_functions": ["print", "render_template", "jsonify", "response"],
                "auth_functions": ["login_user", "authenticate", "check_password"],
                "crypto_functions": ["encrypt", "decrypt", "hash", "md5", "sha1"]
            },
            "javascript": {
                "input_functions": ["getElementById", "querySelector", "event.target.value", "req.params", "req.body"],
                "sql_functions": ["query", "execute", "findOne", "findAll"],
                "output_functions": ["innerHTML", "document.write", "res.send", "res.json"],
                "auth_functions": ["login", "authenticate", "verify"],
                "crypto_functions": ["encrypt", "decrypt", "createHash", "md5", "sha1"]
            },
            "java": {
                "input_functions": ["getParameter", "getInputStream", "readLine"],
                "sql_functions": ["executeQuery", "executeUpdate", "prepareStatement"],
                "output_functions": ["println", "write", "print", "out.write"],
                "auth_functions": ["login", "authenticate", "checkCredentials"],
                "crypto_functions": ["encrypt", "decrypt", "digest", "MD5", "SHA1"]
            },
            "cpp": {
                "input_functions": ["cin", "scanf", "gets", "read"],
                "sql_functions": ["execute", "query", "prepare"],
                "output_functions": ["cout", "printf", "puts", "write"],
                "auth_functions": ["login", "authenticate", "verify"],
                "crypto_functions": ["encrypt", "decrypt", "hash", "MD5", "SHA1"]
            }
        }
    
    def _init_security_patterns(self):
        """Initialize regex patterns for detecting security issues"""
        self.security_patterns = {
            "hardcoded_credentials": {
                "pattern": r'(?:password|passwd|pwd|secret|key|token|auth)[\s=:"\']+([\w\-\.@!#$%^&*()+<>?]{4,})',
                "description": "Hardcoded credentials",
                "severity": 0.9,
                "remediation": "Use environment variables or secure storage for credentials"
            },
            "sql_injection": {
                "pattern": r'(?:execute|query)\s*\(\s*[\'"](?:[^\'"]|[\'"]\s*\+|\s*\+\s*[\'"]\s*\+)\s*[\'"]\s*\)',
                "description": "Potential SQL injection",
                "severity": 0.9,
                "remediation": "Use parameterized queries or prepared statements"
            },
            "xss": {
                "pattern": r'(?:innerHTML|outerHTML|document\.write)\s*\(\s*(?:[^;]|[\'"]\s*\+|\s*\+\s*[\'"]\s*\+)\s*\)',
                "description": "Potential Cross-Site Scripting (XSS)",
                "severity": 0.8,
                "remediation": "Use safe DOM methods or content sanitization"
            },
            "insecure_cookies": {
                "pattern": r'(?:cookie|setCookie)(?![^;{]*secure(?:;|$|}))',
                "description": "Cookies without secure flag",
                "severity": 0.6,
                "remediation": "Add the 'secure' and 'httpOnly' flags to cookies"
            },
            "weak_crypto": {
                "pattern": r'(?:md5|sha1|DES|RC4)\s*\(',
                "description": "Weak cryptographic algorithm",
                "severity": 0.7,
                "remediation": "Use strong cryptographic algorithms like AES-256, SHA-256, or bcrypt"
            },
            "buffer_overflow": {
                "pattern": r'(?:strcpy|strcat|gets|sprintf)\s*\(',
                "description": "Potential buffer overflow",
                "severity": 0.9,
                "remediation": "Use safe alternatives like strncpy, strncat, fgets, snprintf"
            }
        }
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database"""
        # In a real implementation, this would load from a file or service
        # Here we'll create a simplified in-memory database
        return {
            "components": {
                "jquery": {
                    "versions": {
                        "<1.9.0": ["CVE-2012-6708", "XSS vulnerability"]
                    }
                },
                "log4j": {
                    "versions": {
                        "<2.15.0": ["CVE-2021-44228", "Log4Shell RCE vulnerability"]
                    }
                },
                "spring": {
                    "versions": {
                        "<5.3.20": ["CVE-2022-22965", "Spring4Shell RCE vulnerability"]
                    }
                }
            },
            "cwe": {
                "CWE-79": {
                    "name": "Cross-site Scripting (XSS)",
                    "severity": 0.7
                },
                "CWE-89": {
                    "name": "SQL Injection",
                    "severity": 0.9
                },
                "CWE-119": {
                    "name": "Buffer Overflow",
                    "severity": 0.8
                },
                "CWE-798": {
                    "name": "Use of Hard-coded Credentials",
                    "severity": 0.8
                }
            }
        }
    
    def analyze_code(self, code: str, filename: str, language: str) -> List[SecurityIssue]:
        """
        Analyze code for security vulnerabilities
        
        Args:
            code: Source code to analyze
            filename: Name of the file
            language: Programming language
            
        Returns:
            List of detected security issues
        """
        # Determine language if not provided
        if not language:
            language = self._detect_language(code, filename)
        
        # Create security context
        context = self._create_security_context(code, language)
        
        # Detect vulnerabilities
        issues = []
        
        # Apply each detector
        for vuln_type, detector in self.detectors.items():
            vuln_issues = detector(code, filename, context)
            issues.extend(vuln_issues)
        
        # Apply pattern-based detection
        pattern_issues = self._detect_with_patterns(code, filename, language)
        issues.extend(pattern_issues)
        
        # Detect vulnerable dependencies
        dep_issues = self._detect_vulnerable_dependencies(context)
        issues.extend(dep_issues)
        
        # Sort by severity
        issues.sort(key=lambda x: x.severity, reverse=True)
        
        return issues
    
    def analyze_files(self, files: List[Dict[str, str]]) -> Dict[str, List[SecurityIssue]]:
        """
        Analyze multiple files in parallel
        
        Args:
            files: List of file dictionaries with code, filename, and language
            
        Returns:
            Dictionary mapping filenames to detected issues
        """
        results = {}
        
        # Process files in parallel
        future_to_file = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            for file_dict in files:
                code = file_dict.get('code', '')
                filename = file_dict.get('filename', f"file_{len(future_to_file)}")
                language = file_dict.get('language', '')
                
                future = executor.submit(self.analyze_code, code, filename, language)
                future_to_file[future] = filename
        
        # Collect results
        for future in future_to_file:
            filename = future_to_file[future]
            try:
                issues = future.result()
                results[filename] = issues
            except Exception as exc:
                logger.error(f"Error analyzing {filename}: {exc}")
                results[filename] = []
        
        return results
    
    def mitigate_vulnerabilities(self, code: str, issues: List[SecurityIssue], language: str) -> str:
        """
        Apply mitigations to the code
        
        Args:
            code: Original code
            issues: Detected security issues
            language: Programming language
            
        Returns:
            Mitigated code
        """
        # Sort issues by their location in the code (to process from bottom to top)
        sorted_issues = sorted(
            issues, 
            key=lambda x: (x.location.get('line_end', 0), x.location.get('line_start', 0)),
            reverse=True
        )
        
        # Apply mitigations
        mitigated_code = code
        for issue in sorted_issues:
            mitigator = self._get_mitigator(issue.vulnerability_type, language)
            if mitigator:
                mitigated_code = mitigator(mitigated_code, issue)
        
        return mitigated_code
    
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
    
    def _create_security_context(self, code: str, language: str) -> SecurityContext:
        """Create security context from code"""
        context = SecurityContext(language=language)
        
        # Extract frameworks based on imports/includes
        if language == 'python':
            # Extract Python imports
            imports = re.findall(r'(?:from|import)\s+([\w\.]+)', code)
            for imp in imports:
                if imp in ['flask', 'django', 'fastapi', 'tornado', 'bottle']:
                    context.frameworks.append(imp)
                if imp in ['sqlalchemy', 'pymysql', 'psycopg2', 'sqlite3']:
                    context.dependencies[imp] = '0.0.0'  # Unknown version
        
        elif language == 'javascript':
            # Extract JS imports/requires
            imports = re.findall(r'(?:import.*from\s+[\'"]([^\'"]+)[\'"]|require\s*\(\s*[\'"]([^\'"]+)[\'"])', code)
            for imp in imports:
                module = imp[0] if imp[0] else imp[1]
                if module in ['express', 'react', 'angular', 'vue']:
                    context.frameworks.append(module)
                if module in ['mysql', 'mongodb', 'sequelize', 'mongoose']:
                    context.dependencies[module] = '0.0.0'  # Unknown version
        
        # Extract input and output sources
        rules = self.security_rules.get(language, {})
        
        # Find input functions
        for func in rules.get('input_functions', []):
            if func in code:
                context.input_sources.append(func)
        
        # Find output functions
        for func in rules.get('output_functions', []):
            if func in code:
                context.output_sinks.append(func)
        
        # Find auth functions
        for func in rules.get('auth_functions', []):
            if func in code:
                context.auth_mechanisms.append(func)
        
        # Find sensitive data
        sensitive_patterns = [
            'password', 'secret', 'token', 'api_key', 'credit_card',
            'ssn', 'social_security', 'passport', 'license'
        ]
        
        for pattern in sensitive_patterns:
            if re.search(fr'\b{pattern}\b', code, re.IGNORECASE):
                context.sensitive_data_types.append(pattern)
        
        return context
    
    def _detect_with_patterns(self, code: str, filename: str, language: str) -> List[SecurityIssue]:
        """Detect security issues using regex patterns"""
        issues = []
        
        # Get language-specific patterns
        all_patterns = self.security_patterns.copy()
        
        # Process each pattern
        for pattern_name, pattern_info in all_patterns.items():
            # Get or compile regex
            pattern_regex = self.pattern_cache.get(pattern_name)
            if not pattern_regex:
                pattern_regex = re.compile(pattern_info['pattern'], re.IGNORECASE)
                self.pattern_cache[pattern_name] = pattern_regex
            
            # Find matches
            for match in pattern_regex.finditer(code):
                # Find line numbers
                line_start = code[:match.start()].count('\n') + 1
                line_end = line_start + code[match.start():match.end()].count('\n')
                
                # Get code snippet
                snippet_lines = code.splitlines()[max(0, line_start-2):min(len(code.splitlines()), line_end+1)]
                snippet = '\n'.join(snippet_lines)
                
                # Get CWE ID
                cwe_id = None
                if pattern_name == 'sql_injection':
                    cwe_id = "CWE-89"
                elif pattern_name == 'xss':
                    cwe_id = "CWE-79"
                elif pattern_name == 'buffer_overflow':
                    cwe_id = "CWE-119"
                elif pattern_name == 'hardcoded_credentials':
                    cwe_id = "C

def _detect_injection(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect injection vulnerabilities (SQL, Command, etc.)"""
    issues = []
    
    # Get language-specific input and SQL functions
    language = context.language
    rules = self.security_rules.get(language, {})
    input_funcs = rules.get('input_functions', [])
    sql_funcs = rules.get('sql_functions', [])
    
    # Check for input to SQL flow
    lines = code.splitlines()
    variables_from_input = set()
    
    # Track variables that come from input
    for i, line in enumerate(lines):
        # Check if line contains input function
        for func in input_funcs:
            if func in line:
                # Extract variable being assigned
                var_match = re.search(r'(\w+)\s*=', line)
                if var_match:
                    variables_from_input.add(var_match.group(1))
        
        # Check if line uses input variables in SQL
        for func in sql_funcs:
            if func in line:
                # Check if any input variables are used in this SQL line
                for var in variables_from_input:
                    if var in line and not (f"parameterized" in line or f"prepare" in line):
                        # String concatenation pattern
                        if "+" in line or "%" in line or "format" in line or "f'" in line or 'f"' in line:
                            issues.append(SecurityIssue(
                                vulnerability_type=VulnerabilityType.INJECTION,
                                severity=0.9,
                                location={"file": filename, "line_start": i+1, "line_end": i+1},
                                description=f"Potential SQL injection using user input variable '{var}'",
                                remediation="Use parameterized queries or prepared statements",
                                code_snippet=line,
                                cwe_id="CWE-89",
                                confidence=0.85
                            ))
    
    # Command injection
    cmd_funcs = ["system", "exec", "popen", "subprocess", "shell_exec", "eval", "os.system"]
    for i, line in enumerate(lines):
        for func in cmd_funcs:
            if func in line:
                # Check if any input variables are used
                for var in variables_from_input:
                    if var in line:
                        issues.append(SecurityIssue(
                            vulnerability_type=VulnerabilityType.INJECTION,
                            severity=0.9,
                            location={"file": filename, "line_start": i+1, "line_end": i+1},
                            description=f"Potential command injection using user input variable '{var}'",
                            remediation="Validate input and use safe APIs instead of shell commands",
                            code_snippet=line,
                            cwe_id="CWE-78",
                            confidence=0.9
                        ))
    
    return issues

def _detect_xss(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect Cross-Site Scripting (XSS) vulnerabilities"""
    issues = []
    
    # Get language-specific input and output functions
    language = context.language
    rules = self.security_rules.get(language, {})
    input_funcs = rules.get('input_functions', [])
    output_funcs = rules.get('output_functions', [])
    
    # Track variables from user input
    lines = code.splitlines()
    user_input_vars = set()
    
    for i, line in enumerate(lines):
        # Track variables from input
        for func in input_funcs:
            if func in line:
                var_match = re.search(r'(\w+)\s*=', line)
                if var_match:
                    user_input_vars.add(var_match.group(1))
        
        # Check for output of user input without sanitization
        for func in output_funcs:
            if func in line:
                for var in user_input_vars:
                    if var in line:
                        # Check if no sanitization function is used
                        sanitization_funcs = ["escape", "sanitize", "encode", "purify", "htmlspecialchars"]
                        if not any(san_func in line for san_func in sanitization_funcs):
                            issues.append(SecurityIssue(
                                vulnerability_type=VulnerabilityType.XSS,
                                severity=0.8,
                                location={"file": filename, "line_start": i+1, "line_end": i+1},
                                description=f"Potential XSS vulnerability: unsanitized user input '{var}' in output",
                                remediation="Sanitize user input before rendering in HTML/output",
                                code_snippet=line,
                                cwe_id="CWE-79",
                                confidence=0.8
                            ))
    
    return issues

def _detect_broken_auth(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect broken authentication issues"""
    issues = []
    
    # Check for common auth implementation issues
    auth_issues = [
        {
            "pattern": r"password\s*==\s*",
            "description": "Unsafe password comparison (vulnerable to timing attacks)",
            "remediation": "Use constant-time comparison functions for password verification",
            "severity": 0.7,
            "cwe_id": "CWE-208"
        },
        {
            "pattern": r"(?:md5|sha1)\s*\([^)]*password",
            "description": "Weak password hashing algorithm",
            "remediation": "Use strong hashing algorithms with salt (bcrypt, Argon2, PBKDF2)",
            "severity": 0.8,
            "cwe_id": "CWE-327"
        },
        {
            "pattern": r"remember\s+me.*cookie",
            "description": "Persistent login (remember me) cookie without proper protection",
            "remediation": "Implement secure remember-me cookies with proper expiration and server validation",
            "severity": 0.6,
            "cwe_id": "CWE-539"
        }
    ]
    
    lines = code.splitlines()
    for i, line in enumerate(lines):
        for issue in auth_issues:
            if re.search(issue["pattern"], line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.BROKEN_AUTH,
                    severity=issue["severity"],
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description=issue["description"],
                    remediation=issue["remediation"],
                    code_snippet=line,
                    cwe_id=issue["cwe_id"],
                    confidence=0.75
                ))
    
    # Check for missing authentication
    if not context.auth_mechanisms and "login" in filename.lower():
        issues.append(SecurityIssue(
            vulnerability_type=VulnerabilityType.BROKEN_AUTH,
            severity=0.7,
            location={"file": filename, "line_start": 1, "line_end": len(lines)},
            description="Potential missing authentication mechanism",
            remediation="Implement proper authentication using secure frameworks or libraries",
            cwe_id="CWE-306",
            confidence=0.6
        ))
    
    return issues

def _detect_sensitive_data_exposure(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect sensitive data exposure issues"""
    issues = []
    
    # Check for sensitive data without proper protection
    sensitive_types = context.sensitive_data_types
    lines = code.splitlines()
    
    # Check for plaintext storage of sensitive data
    for i, line in enumerate(lines):
        for data_type in sensitive_types:
            if data_type in line.lower():
                # Check if encryption is used
                crypto_terms = ["encrypt", "cipher", "hash", "bcrypt", "scrypt", "argon"]
                if not any(term in line.lower() for term in crypto_terms):
                    # Check if it's variable assignment
                    if "=" in line and not line.strip().startswith("#"):
                        issues.append(SecurityIssue(
                            vulnerability_type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
                            severity=0.7,
                            location={"file": filename, "line_start": i+1, "line_end": i+1},
                            description=f"Potential plaintext storage of sensitive data ({data_type})",
                            remediation="Encrypt sensitive data or use secure storage mechanisms",
                            code_snippet=line,
                            cwe_id="CWE-312",
                            confidence=0.7
                        ))
    
    # Check for insecure transmission
    http_funcs = ["http.", "requests.", "fetch", "axios", "HttpClient"]
    for i, line in enumerate(lines):
        if any(func in line for func in http_funcs) and "https://" not in line:
            for data_type in sensitive_types:
                if data_type in line.lower():
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
                        severity=0.8,
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=f"Potential insecure transmission of sensitive data ({data_type})",
                        remediation="Use HTTPS for transmitting sensitive information",
                        code_snippet=line,
                        cwe_id="CWE-319",
                        confidence=0.8
                    ))
    
    return issues

def _detect_broken_access_control(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect broken access control issues"""
    issues = []
    
    # Check for missing authorization checks
    if not context.authorization_mechanisms:
        # Look for sensitive operations
        sensitive_ops = ["admin", "delete", "remove", "update", "modify", "create"]
        if any(op in filename.lower() for op in sensitive_ops):
            # Check for lack of auth checks
            auth_checks = ["authorize", "permission", "isAdmin", "can_", "allowed_to", "role"]
            if not any(check in code for check in auth_checks):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.BROKEN_ACCESS_CONTROL,
                    severity=0.7,
                    location={"file": filename, "line_start": 1, "line_end": 1},
                    description="Potential missing access control for sensitive operation",
                    remediation="Implement proper authorization checks before performing sensitive operations",
                    cwe_id="CWE-285",
                    confidence=0.6
                ))
    
    # Check for direct object reference vulnerabilities
    lines = code.splitlines()
    id_params = ["id", "user_id", "account_id", "record_id", "file_id"]
    
    for i, line in enumerate(lines):
        # Check for ID parameters in requests/queries
        for param in id_params:
            # Look for patterns where IDs are used without validation
            if re.search(rf'\b{param}\b.*=', line) and "where" in line.lower():
                # Check if there's no validation or authorization check
                if not re.search(r'(authorize|permission|check|validate|belong|owner)', line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.BROKEN_ACCESS_CONTROL,
                        severity=0.8,
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=f"Potential Insecure Direct Object Reference (IDOR) using {param}",
                        remediation="Validate user's permission to access the requested object",
                        code_snippet=line,
                        cwe_id="CWE-639",
                        confidence=0.7
                    ))
    
    return issues

def _detect_security_misconfiguration(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect security misconfiguration issues"""
    issues = []
    
    # Check for debug mode enabled in production frameworks
    debug_patterns = [
        r'debug\s*=\s*True',
        r'DEBUG\s*=\s*True',
        r'[\'"]debug[\'"]\s*:\s*true',
        r'app\.debug\s*=\s*true'
    ]
    
    lines = code.splitlines()
    for i, line in enumerate(lines):
        for pattern in debug_patterns:
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=0.6,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description="Debug mode potentially enabled in production",
                    remediation="Disable debug mode in production environments",
                    code_snippet=line,
                    cwe_id="CWE-215",
                    confidence=0.7
                ))
    
    # Check for overly permissive CORS
    cors_patterns = [
        r'(?:Access-Control-Allow-Origin:|cors\()\s*[\'"]\*[\'"]',
        r'cors\.enable.*origin:\s*[\'"]\*[\'"]'
    ]
    
    for i, line in enumerate(lines):
        for pattern in cors_patterns:
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=0.6,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description="Overly permissive CORS configuration (allows all origins)",
                    remediation="Restrict CORS to specific trusted origins",
                    code_snippet=line,
                    cwe_id="CWE-346",
                    confidence=0.8
                ))
    
    # Check for insecure cookie configuration
    cookie_patterns = [
        r'(?:cookie|set_cookie).*secure\s*=\s*false',
        r'(?:cookie|set_cookie).*httpOnly\s*=\s*false'
    ]
    
    for i, line in enumerate(lines):
        for pattern in cookie_patterns:
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                    severity=0.5,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description="Insecure cookie configuration (missing secure or httpOnly flags)",
                    remediation="Set secure and httpOnly flags for sensitive cookies",
                    code_snippet=line,
                    cwe_id="CWE-614",
                    confidence=0.8
                ))
    
    return issues

def _detect_buffer_overflow(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect buffer overflow vulnerabilities"""
    issues = []
    
    # Only relevant for C/C++ code
    if context.language.lower() not in ['c', 'cpp', 'c++']:
        return issues
    
    # Check for unsafe functions
    unsafe_funcs = {
        'strcpy': 'strncpy',
        'strcat': 'strncat',
        'sprintf': 'snprintf',
        'gets': 'fgets',
        'scanf': 'scanf (with field width limiters)',
        'vsprintf': 'vsnprintf',
        'realpath': 'realpath (with size argument)'
    }
    
    # Also check for buffer declarations followed by unsafe operations
    buffer_patterns = [
        r'char\s+(\w+)\s*\[\s*\d+\s*\]',
        r'wchar_t\s+(\w+)\s*\[\s*\d+\s*\]'
    ]
    
    lines = code.splitlines()
    buffer_vars = {}
    
    # First find buffer declarations
    for i, line in enumerate(lines):
        for pattern in buffer_patterns:
            match = re.search(pattern, line)
            if match:
                buffer_name = match.group(1)
                # Extract buffer size
                size_match = re.search(r'\[\s*(\d+)\s*\]', line)
                if size_match:
                    buffer_size = int(size_match.group(1))
                    buffer_vars[buffer_name] = {
                        'size': buffer_size,
                        'line': i+1
                    }
    
    # Then check for unsafe functions
    for i, line in enumerate(lines):
        for func, alternative in unsafe_funcs.items():
            func_pattern = fr'{func}\s*\('
            if re.search(func_pattern, line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.BUFFER_OVERFLOW,
                    severity=0.8,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description=f"Use of unsafe function {func} that may cause buffer overflow",
                    remediation=f"Replace with safer alternative: {alternative}",
                    code_snippet=line,
                    cwe_id="CWE-120",
                    confidence=0.9
                ))
        
        # Check for unsafe buffer operations
        for buffer_name, buffer_info in buffer_vars.items():
            # Look for buffer manipulation without bounds checking
            if buffer_name in line and ('=' in line or 'copy' in line or 'memcpy' in line):
                if not ('sizeof' in line and buffer_name in line):
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.BUFFER_OVERFLOW,
                        severity=0.7,
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=f"Potential buffer overflow in manipulation of buffer '{buffer_name}'",
                        remediation="Add bounds checking before buffer operations or use safer alternatives",
                        code_snippet=line,
                        cwe_id="CWE-120",
                        confidence=0.7
                    ))
    
    return issues

def _detect_crypto_failure(self, code: str, filename: str, context: SecurityContext) -> List[SecurityIssue]:
    """Detect cryptographic failures"""
    issues = []
    
    # Check for weak cryptographic algorithms
    weak_algos = {
        'md5': 'SHA-256 or SHA-3',
        'sha1': 'SHA-256 or SHA-3',
        'des': 'AES-256',
        'rc4': 'AES-256',
        'blowfish': 'AES-256',
        'ecb': 'GCM or CBC mode',
        'electronic codebook': 'GCM or CBC mode'
    }
    
    lines = code.splitlines()
    for i, line in enumerate(lines):
        for algo, alternative in weak_algos.items():
            if re.search(fr'\b{algo}\b', line, re.IGNORECASE):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_FAILURE,
                    severity=0.7,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description=f"Use of weak cryptographic algorithm ({algo})",
                    remediation=f"Replace with stronger algorithm: {alternative}",
                    code_snippet=line,
                    cwe_id="CWE-327",
                    confidence=0.8
                ))
    
    # Check for hardcoded crypto keys
    key_patterns = [
        r'(?:key|iv|salt)\s*=\s*[\'"]([\w\+\-\/=]{8,})[\'"]',
        r'(?:key|iv|salt)\s*=\s*bytes\([\'"][\w\+\-\/=]{8,}[\'"]\)'
    ]
    
    for i, line in enumerate(lines):
        for pattern in key_patterns:
            if re.search(pattern, line):
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_FAILURE,
                    severity=0.9,
                    location={"file": filename, "line_start": i+1, "line_end": i+1},
                    description="Hardcoded cryptographic key or initialization vector",
                    remediation="Store keys in secure key management systems or environment variables",
                    code_snippet=line,
                    cwe_id="CWE-321",
                    confidence=0.9
                ))
    
    # Check for insufficient key lengths
    key_length_patterns = [
        r'(?:key_length|key_size|keylength|keysize)\s*=\s*(\d+)',
        r'generateKey\(\s*(\d+)\s*\)'
    ]
    
    for i, line in enumerate(lines):
        for pattern in key_length_patterns:
            match = re.search(pattern, line)
            if match:
                key_length = int(match.group(1))
                if key_length < 128:
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.CRYPTOGRAPHIC_FAILURE,
                        severity=0.6,
                        location={"file": filename, "line_start": i+1, "line_end": i+1},
                        description=f"Insufficient key length ({key_length} bits)",
                        remediation="Use key length of at least 128 bits for symmetric encryption and 2048 bits for RSA",
                        code_snippet=line,
                        cwe_id="CWE-326",
                        confidence=0.7
                    ))
    
    return issues

def _detect_vulnerable_dependencies(self, context: SecurityContext) -> List[SecurityIssue]:
    """Detect vulnerable dependencies"""
    issues = []
    
    # Check each dependency against the vulnerability database
    for dep_name, dep_version in context.dependencies.items():
        if dep_name in self.vuln_db.get("components", {}):
            dep_info = self.vuln_db["components"][dep_name]
            
            # Check each vulnerable version pattern
            for version_pattern, vulnerabilities in dep_info.get("versions", {}).items():
                # Simple version check (in real implementation would use semver)
                if version_pattern.startswith("<") and dep_version != "0.0.0":
                    # Extract version number
                    pattern_version = version_pattern[1:]
                    # Simple string comparison (not accurate for all version formats)
                    if dep_version < pattern_version:
                        # Create an issue for each vulnerability
                        for i, vuln_info in enumerate(vulnerabilities):
                            if i % 2 == 0:  # CVE ID is at even indices
                                cve_id = vuln_info
                                description = vulnerabilities[i+1] if i+1 < len(vulnerabilities) else "Unknown vulnerability"
                                
                                issues.append(SecurityIssue(
                                    vulnerability_type=VulnerabilityType.USING_COMPONENTS_WITH_VULNERABILITIES,
                                    severity=0.8,
                                    location={"file": "dependencies", "line_start": 1, "line_end": 1},
                                    description=f"Vulnerable dependency: {dep_name} version {dep_version} has {description}",
                                    remediation=f"Update {dep_name} to a version newer than {pattern_version}",
                                    related_components=[dep_name],
                                    cwe_id="CWE-1104",
                                    cvss_score=7.5,  # Example score
                                    confidence=0.9
                                ))
    
    return issues

def _get_mitigator(self, vulnerability_type: VulnerabilityType, language: str):
    """Get appropriate mitigation function for a vulnerability type"""
    mitigators = {
        VulnerabilityType.INJECTION: {
            "python": self._mitigate_injection_python,
            "javascript": self._mitigate_injection_js,
            "java": self._mitigate_injection_java,
            "cpp": self._mitigate_injection_cpp
        },
        VulnerabilityType.XSS: {
            "python": self._mitigate_xss_python,
            "javascript": self._mitigate_xss_js,
            "java": self._mitigate_xss_java
        },
        VulnerabilityType.BUFFER_OVERFLOW: {
            "cpp": self._mitigate_buffer_overflow_cpp,
            "c": self._mitigate_buffer_overflow_cpp
        },
        VulnerabilityType.CRYPTOGRAPHIC_FAILURE: {
            "python": self._mitigate_crypto_python,
            "javascript": self._mitigate_crypto_js,
            "java": self._mitigate_crypto_java,
            "cpp": self._mitigate_crypto_cpp
        }
    }
    
    vuln_mitigators = mitigators.get(vulnerability_type, {})
    return vuln_mitigators.get(language)

def _mitigate_injection_python(self, code: str, issue: SecurityIssue) -> str:
    """Mitigate SQL injection in Python code"""
    lines = code.splitlines()
    line_idx = issue.location.get('line_start', 1) - 1
    
    if line_idx < 0 or line_idx >= len(lines):
        return code
    
    line = lines[line_idx]
    
    # Detect the type of injection
    if "sql" in issue.description.lower() or any(func in line for func in ["execute", "cursor.execute"]):
        # SQL injection mitigation
        if "+" in line or "%" in line or "format" in line:
            # Replace string concatenation with parameter binding
            # Extract the SQL statement and variables
            sql_match = re.search(r'(["\'])(.*?)(\1)\s*(\+|\%|\.format)', line)
            if sql_match:
                sql_part = sql_match.group(2)
                # Replace string concatenation with parameter placeholder
                if "SELECT" in sql_part or "select" in sql_part:
                    # Replace with parameterized query
                    new_line = line.replace(sql_match.group(0), f"'{sql_part} WHERE id = %s' ")
                    new_line = new_line.replace(" + ", ", ")
                    new_line = new_line.replace(" % ", ", ")
                    new_line = new_line.replace(".format(", ", ")
                else:
                    # Generic replacement
                    new_line = line.replace(" + ", ", ")
                    new_line = new_line.replace(" % ", ", ")
                    new_line = new_line.replace(".format(", ", ")
                
                lines[line_idx] = new_line
    
    elif "command" in issue.description.lower() or any(func in line for func in ["system", "exec", "subprocess"]):
        # Command injection mitigation
        if "subprocess" in line:
            # Replace shell=True with shell=False and args list
            if "shell=True" in line:
                # Convert to list of arguments
                new_line = line.replace("shell=True", "shell=False")
                # Try to extract the command
                cmd_match = re.search(r'(["\'])(.*?)(\1)', line)
                if cmd_match:
                    cmd = cmd_match.group(2)
                    # Replace string with list of args
                    cmd_parts = cmd.split()
                    cmd_list = "['" + "', '".join(cmd_parts) + "']"
                    new_line = new_line.replace(cmd_match.group(0), cmd_list)
                
                lines[line_idx] = new_line
        elif "os.system" in line:
            # Replace os.system with subprocess.run
            new_line = line.replace("os.system", "subprocess.run")
            # Try to extract the command
            cmd_match = re.search(r'(["\'])(.*?)(\1)', line)
            if cmd_match:
                cmd = cmd_match.group(2)
                # Replace string with list of args
                cmd_parts = cmd.split()
                cmd_list = "['" + "', '".join(cmd_parts) + "']"
                new_line = new_line.replace(cmd_match.group(0), cmd_list)
                # Add shell=False
                if "(" in new_line:
                    insertion_point = new_line.index("(") + 1
                    new_line = new_line[:insertion_point] + cmd_list + ", shell=False" + new_line[insertion_point + len(cmd_match.group(0)):]
                
                lines[line_idx] = new_line
    
    return "\n".join(lines)

def _mitigate_xss_python(self, code: str, issue: SecurityIssue) -> str:
    """Mitigate XSS in Python code"""
    lines = code.splitlines()
    line_idx = issue.location.get('line_start', 1) - 1
    
    if line_idx < 0 or line_idx >= len(lines):
        return code
    
    line = lines[line_idx]
    
    # Detect the framework
    if "flask" in code.lower():
        # Flask application
        if "render_template" in line or "{{ " in line:
            # Add autoescaping or escape filter
            if "render_template" in line and "|safe" in line:
                # Remove the |safe filter
                new_line = line.replace("|safe", "")
                lines[line_idx] = new_line
            elif "render_template" in line and "|" not in line:
                # Add import for escape if needed
                if "from markupsafe import escape" not in code:
                    # Add import at the top
                    lines.insert(0, "from markupsafe import escape")
                
                # Extract variable
                var_match = re.search(r'(\w+)=(\w+)', line)
                if var_match:
                    var_name = var_match.group(1)
                    var_value = var_match.group(2)
                    # Replace with escaped version
                    new_line = line.replace(f"{var_name}={var_value}", f"{var_name}=escape({var_value})")
                    lines[line_idx] = new_line
    
    elif "django" in code.lower():
        # Django application
        if "render" in line:
            # Ensure autoescape is on
            if "autoescape off" in code:
                # Find autoescape tag
                for i, l in enumerate(lines):
                    if "{% autoescape off %}" in l:
                        lines[i] = "{% autoescape on %}"
    
    elif "innerHTML" in line or "document.write" in line:
        # Client-side JavaScript (in Python template)
        if "innerHTML" in line:
            # Replace innerHTML with textContent
            new_line = line.replace("innerHTML", "textContent")
            lines[line_idx] = new_line
        elif "document.write" in line:
            # Replace with safer alternative
            new_line = line.replace("document.write", "/* Use DOM manipulation instead: const el = document.createElement('div'); el.textContent = ")
            new_line += ";
def _mitigate_xss_python(self, code: str, issue: SecurityIssue) -> str:
    """Mitigate XSS in Python code"""
    lines = code.splitlines()
    line_idx = issue.location.get('line_start', 1) - 1
    
    if line_idx < 0 or line_idx >= len(lines):
        return code
    
    line = lines[line_idx]
    
    # Detect the framework
    if "flask" in code.lower():
        # Flask application
        if "render_template" in line or "{{ " in line:
            # Add autoescaping or escape filter
            if "render_template" in line and "|safe" in line:
                # Remove the |safe filter
                new_line = line.replace("|safe", "")
                lines[line_idx] = new_line
            elif "render_template" in line and "|" not in line:
                # Add import for escape if needed
                if "from markupsafe import escape" not in code:
                    # Add import at the top
                    lines.insert(0, "from markupsafe import escape")
                
                # Extract variable
                var_match = re.search(r'(\w+)=(\w+)', line)
                if var_match:
                    var_name = var_match.group(1)
                    var_value = var_match.group(2)
                    # Replace with escaped version
                    new_line = line.replace(f"{var_name}={var_value}", f"{var_name}=escape({var_value})")
                    lines[line_idx] = new_line
    
    elif "django" in code.lower():
        # Django application
        if "render" in line:
            # Ensure autoescape is on
            if "autoescape off" in code:
                # Find autoescape tag
                for i, l in enumerate(lines):
                    if "{% autoescape off %}" in l:
                        lines[i] = "{% autoescape on %}"
    
    elif "innerHTML" in line or "document.write" in line:
        # Client-side JavaScript (in Python template)
        if "innerHTML" in line:
            # Replace innerHTML with textContent
            new_line = line.replace("innerHTML", "textContent")
            lines[line_idx] = new_line
        elif "document.write" in line:
            # Replace with safer alternative
            new_line = line.replace("document.write", "/* Use DOM manipulation instead: const el = document.createElement('div'); el.textContent = ")
            new_line += "; document.body.appendChild(el); */"
            lines[line_idx] = new_line
    
    return "\n".join(lines)

def _mitigate_buffer_overflow_cpp(self, code: str, issue: SecurityIssue) -> str:
    """Mitigate buffer overflow in C/C++ code"""
    lines = code.splitlines()
    line_idx = issue.location.get('line_start', 1) - 1
    
    if line_idx < 0 or line_idx >= len(lines):
        return code
    
    line = lines[line_idx]
    
    # Replace unsafe functions with safe alternatives
    replacements = {
        'strcpy(': 'strncpy(',
        'strcat(': 'strncat(',
        'gets(': 'fgets(',
        'sprintf(': 'snprintf('
    }
    
    for unsafe, safe in replacements.items():
        if unsafe in line:
            # Basic replacement
            new_line = line.replace(unsafe, safe)
            
            # For functions that need a size parameter
            if safe in ['strncpy(', 'strncat(']:
                # Try to extract the buffer name
                buffer_match = re.search(r'(?:strn?(?:cpy|cat))\s*\(\s*(\w+)', new_line)
                if buffer_match:
                    buffer_name = buffer_match.group(1)
                    # Add size parameter if not present
                    if "sizeof" not in new_line:
                        # Add sizeof
                        param_end = new_line.find(')')
                        if param_end != -1:
                            new_line = new_line[:param_end] + f", sizeof({buffer_name})" + new_line[param_end:]
            
            elif safe == 'fgets(':
                # Add buffer size and stdin for fgets
                buffer_match = re.search(r'fgets\s*\(\s*(\w+)', new_line)
                if buffer_match:
                    buffer_name = buffer_match.group(1)
                    # Add size and stdin parameters
                    param_end = new_line.find(')')
                    if param_end != -1:
                        new_line = new_line[:param_end] + f", sizeof({buffer_name}), stdin" + new_line[param_end:]
            
            elif safe == 'snprintf(':
                # Add buffer size for snprintf
                buffer_match = re.search(r'snprintf\s*\(\s*(\w+)', new_line)
                if buffer_match:
                    buffer_name = buffer_match.group(1)
                    # Add size parameter
                    after_buffer = new_line.find(',', new_line.find(buffer_name))
                    if after_buffer != -1:
                        new_line = new_line[:after_buffer] + f", sizeof({buffer_name})" + new_line[after_buffer:]
            
            lines[line_idx] = new_line
    
    # Add bounds checking
    if "memcpy" in line and "sizeof" not in line:
        # Extract buffer name
        buffer_match = re.search(r'memcpy\s*\(\s*(\w+)', line)
        if buffer_match:
            buffer_name = buffer_match.group(1)
            
            # Add check before the memcpy
            check_line = f"if (src_size <= sizeof({buffer_name})) {{ /* bounds check */ "
            post_check = " } else { /* handle error */ }"
            
            lines[line_idx] = check_line + line + post_check
    
    return "\n".join(lines)

def _mitigate_crypto_python(self, code: str, issue: SecurityIssue) -> str:
    """Mitigate cryptographic issues in Python code"""
    lines = code.splitlines()
    line_idx = issue.location.get('line_start', 1) - 1
    
    if line_idx < 0 or line_idx >= len(lines):
        return code
    
    line = lines[line_idx]
    
    # Replace weak algorithms
    if "md5" in line.lower():
        # Replace MD5 with SHA-256
        new_line = line.replace("md5", "sha256")
        new_line = new_line.replace("MD5", "SHA256")
        lines[line_idx] = new_line
        
        # Add import if needed
        if "import hashlib" not in code:
            lines.insert(0, "import hashlib")
    
    elif "sha1" in line.lower():
        # Replace SHA1 with SHA-256
        new_line = line.replace("sha1", "sha256")
        new_line = new_line.replace("SHA1", "SHA256")
        lines[line_idx] = new_line
        
        # Add import if needed
        if "import hashlib" not in code:
            lines.insert(0, "import hashlib")
    
    # Replace hardcoded keys
    key_match = re.search(r'([\'"])((?:key|iv|salt)[\'"]\s*=\s*[\'"]([\w\+\-\/=]{8,})[\'"]);', line)
    if key_match:
        # Replace with environment variable
        var_name = key_match.group(2).strip().replace("'", "").replace('"', "")
        var_upper = var_name.upper()
        new_line = line.replace(key_match.group(0), f"{var_name} = os.environ.get('{var_upper}')")
        lines[line_idx] = new_line
        
        # Add import if needed
        if "import os" not in code:
            lines.insert(0, "import os")
    
    return "\n".join(lines)
