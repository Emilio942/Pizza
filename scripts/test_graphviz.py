#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to verify that Graphviz is installed and working correctly.
"""

import subprocess
import sys
import os

def test_graphviz():
    """Verify Graphviz installation."""
    # Check system-level graphviz installation
    try:
        # Try to run the dot command with version flag
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
        print(f"Graphviz version: {result.stderr.strip()}")
        assert "graphviz" in result.stderr.lower(), "Graphviz not found in system"
        print("✓ System-level Graphviz is installed")
    except FileNotFoundError:
        print("✗ System-level Graphviz is not installed")
        return False
    
    # Check Python graphviz package
    try:
        import graphviz
        print(f"Python graphviz package version: {graphviz.__version__}")
        print("✓ Python graphviz package is installed")
    except ImportError:
        print("✗ Python graphviz package is not installed")
        return False
    
    # Check torchviz package
    try:
        import torchviz
        print(f"Python torchviz package is available")
        print("✓ Python torchviz package is installed")
    except ImportError:
        print("✗ Python torchviz package is not installed")
        return False
    
    # Create a simple test graph and render it
    try:
        g = graphviz.Digraph('G', filename='test_graph')
        g.node('A', 'Node A')
        g.node('B', 'Node B')
        g.edge('A', 'B')
        
        # Render the graph
        g.render(format='png', cleanup=True)
        
        # Check if the file was created
        if os.path.exists('test_graph.png'):
            print("✓ Successfully rendered a test graph with Graphviz")
            # Clean up
            os.remove('test_graph.png')
            return True
        else:
            print("✗ Failed to render a test graph with Graphviz")
            return False
    except Exception as e:
        print(f"✗ Error when testing Graphviz: {e}")
        return False

if __name__ == "__main__":
    print("Testing Graphviz installation...")
    success = test_graphviz()
    if success:
        print("\nGraphviz is successfully installed and working!")
        sys.exit(0)
    else:
        print("\nGraphviz installation appears to have issues.")
        sys.exit(1)
