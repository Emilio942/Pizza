#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Validation Tool for the Pizza Detection CI/CD Pipeline

This script validates all Python scripts in the specified directory:
- Checks for syntax errors
- Verifies required dependencies are installed
- Tests basic execution
- Validates script metadata (docstrings, etc.)

Usage:
    python validate_scripts.py --output REPORT_FILE --scripts-dir SCRIPTS_DIR [--log-dir LOG_DIR]
"""

import os
import sys
import ast
import json
import argparse
import importlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pkg_resources

# Set up logging
def setup_logging(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("script_validator")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        log_file = os.path.join(log_dir, "validate_scripts.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Function to check syntax of a Python file
def check_syntax(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error parsing file: {str(e)}"

# Function to extract dependencies from a Python file
def extract_dependencies(file_path: str) -> Set[str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # Filter out standard library modules
        std_lib = set(sys.stdlib_module_names)
        return {pkg for pkg in imports if pkg not in std_lib and pkg != 'src'}
    
    except Exception as e:
        return set()

# Function to check if dependencies are installed
def check_dependencies(dependencies: Set[str]) -> Tuple[bool, List[str]]:
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            if dep.lower() not in installed_packages:
                missing.append(dep)
    
    return len(missing) == 0, missing

# Function to test script execution (dry run)
def test_execution(file_path: str) -> Tuple[bool, Optional[str]]:
    try:
        # Run with --help to test basic execution without side effects
        # Redirect output to /dev/null
        result = subprocess.run(
            [sys.executable, file_path, "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5,  # 5 second timeout
            text=True
        )
        
        # Check if execution was successful or if it just printed help text
        if result.returncode != 0 and not result.stderr.strip().lower().startswith("usage:"):
            return False, f"Execution failed with error: {result.stderr}"
        
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (> 5 seconds)"
    except Exception as e:
        return False, f"Execution error: {str(e)}"

# Function to validate script metadata
def validate_metadata(file_path: str) -> Tuple[bool, Dict[str, Optional[str]]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Extract module docstring
        docstring = ast.get_docstring(tree)
        
        # Check for valid docstring
        has_docstring = docstring is not None and len(docstring.strip()) > 0
        
        # Try to extract author and date from docstring
        author = None
        date = None
        
        if has_docstring:
            lines = docstring.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith(("author:", "autor:")):
                    author = line.split(':', 1)[1].strip()
                elif line.lower().startswith(("date:", "datum:")):
                    date = line.split(':', 1)[1].strip()
        
        metadata = {
            "has_docstring": has_docstring,
            "docstring": docstring,
            "author": author,
            "date": date
        }
        
        # Consider validation successful if it has a docstring
        return has_docstring, metadata
    
    except Exception as e:
        return False, {"error": str(e)}

# Main validation function
def validate_script(file_path: str, logger: logging.Logger) -> Dict:
    logger.info(f"Validating script: {os.path.basename(file_path)}")
    
    results = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "status": "passed",
        "errors": [],
        "warnings": [],
        "metadata": {}
    }
    
    # Check syntax
    syntax_valid, syntax_error = check_syntax(file_path)
    if not syntax_valid:
        results["status"] = "failed"
        results["errors"].append(f"Syntax error: {syntax_error}")
        logger.error(f"Syntax error in {file_path}: {syntax_error}")
    
    # Extract and check dependencies
    dependencies = extract_dependencies(file_path)
    deps_installed, missing_deps = check_dependencies(dependencies)
    
    results["dependencies"] = list(dependencies)
    
    if not deps_installed:
        results["status"] = "warning"
        results["warnings"].append(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning(f"Missing dependencies in {file_path}: {', '.join(missing_deps)}")
    
    # Only test execution if syntax is valid
    if syntax_valid:
        exec_valid, exec_error = test_execution(file_path)
        if not exec_valid:
            results["status"] = "warning"
            results["warnings"].append(f"Execution test issue: {exec_error}")
            logger.warning(f"Execution test issue in {file_path}: {exec_error}")
    
    # Validate metadata
    meta_valid, metadata = validate_metadata(file_path)
    results["metadata"] = metadata
    
    if not meta_valid:
        results["status"] = "warning"
        results["warnings"].append("Missing or incomplete docstring")
        logger.warning(f"Missing or incomplete docstring in {file_path}")
    
    # Final logging
    if results["status"] == "passed":
        logger.info(f"Validation passed for {file_path}")
    elif results["status"] == "warning":
        logger.warning(f"Validation passed with warnings for {file_path}")
    else:
        logger.error(f"Validation failed for {file_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate Python scripts for the Pizza Detection CI/CD Pipeline")
    parser.add_argument("--output", required=True, help="Output JSON file for validation report")
    parser.add_argument("--scripts-dir", required=True, help="Directory containing scripts to validate")
    parser.add_argument("--log-dir", default=None, help="Directory for log files")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    
    # Find all Python scripts
    script_dir = Path(args.scripts_dir)
    scripts = list(script_dir.glob("**/*.py"))
    
    # Add scripts from specific directories
    project_root = script_dir.parent
    model_scripts = list(project_root.glob("models/**/*.py"))
    src_scripts = list(project_root.glob("src/**/*.py"))
    
    all_scripts = scripts + model_scripts + src_scripts
    
    logger.info(f"Found {len(all_scripts)} Python scripts to validate")
    
    # Validate each script
    results = []
    passed = 0
    warnings = 0
    failed = 0
    
    for script in all_scripts:
        script_path = str(script.absolute())
        result = validate_script(script_path, logger)
        results.append(result)
        
        if result["status"] == "passed":
            passed += 1
        elif result["status"] == "warning":
            warnings += 1
        else:
            failed += 1
    
    # Create summary
    summary = {
        "total_scripts": len(all_scripts),
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "pass_rate": round(passed / len(all_scripts) * 100, 2) if all_scripts else 0,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Create final report
    report = {
        "summary": summary,
        "results": results
    }
    
    # Write report to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation complete. Summary: {passed} passed, {warnings} with warnings, {failed} failed")
    logger.info(f"Report written to {args.output}")
    
    # Return non-zero exit code if any scripts failed
    return 1 if failed > 0 else 0

if __name__ == "__main__":
    # Import datetime here to avoid potential circular imports
    import datetime
    sys.exit(main())
