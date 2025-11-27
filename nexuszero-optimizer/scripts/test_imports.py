#!/usr/bin/env python3
import sys
import importlib
import traceback
import os

print('Starting import checks for supervisor and GNN integration')

# Add the optimizer 'src' directory to sys.path so we can import the package modules by name
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

try:
    # Attempt to import by package path if package is available
    m = importlib.import_module('agents.supervisor')
    print('Imported supervisor module OK')
except Exception:
    traceback.print_exc()
    print('Failed to import supervisor')
    sys.exit(1)

try:
    m2 = importlib.import_module('agents.gnn_integration')
    print('Imported gnn_integration module OK')
except Exception:
    traceback.print_exc()
    print('Failed to import gnn_integration')
    sys.exit(1)

try:
    mg = importlib.import_module('graph.models')
    md = importlib.import_module('graph.database')
    mq = importlib.import_module('graph.query_engine')
    print('Imported graph modules OK')
except Exception:
    traceback.print_exc()
    print('Failed to import graph modules')
    sys.exit(1)

print('Import checks complete')
