"""
Shared pytest fixtures for the PGR Vesting Decision Support test suite.
"""

import os
import sys

# Ensure the project root is on sys.path so src.* imports resolve correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
