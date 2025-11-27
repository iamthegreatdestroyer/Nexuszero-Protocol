#!/usr/bin/env python3
import os
import sys
import pprint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from graph import Neo4jDatabaseManager, create_query_engine
from graph.models import CORE_PRIMITIVES, ProofSystem


def run_demo():
    db = Neo4jDatabaseManager()
    db.connect()
    db.seed_core_primitives()

    engine = create_query_engine(db)

    # Query example: find primitives compatible with GROTH16
    from graph.query_engine import ProofRequirements
    req = ProofRequirements(proof_system=ProofSystem.GROTH16)
    q = engine.query_primitives(req)
    return q


if __name__ == '__main__':
    # Add a small demonstration using fallback storage
    db = Neo4jDatabaseManager()
    db.connect()
    db.seed_core_primitives()
    engine = create_query_engine(db)

    # Demonstrate query primitives -- build requirements to get top candidates
    from graph.query_engine import ProofRequirements
    req = ProofRequirements(proof_system=ProofSystem.GROTH16)
    q_res = engine.query_primitives(req)
    # Pretty print results
    print('\n=== Graph Query Demo Result ===')
    for p, s in zip(q_res.primitives, q_res.relevance_scores):
        print(f"- {p.name} ({p.primitive_id}) - score={s:.2f}")
    print('\nReasoning:')
    print(q_res.reasoning)
    print('\nSuggested Patterns:')
    for pat in q_res.suggested_patterns:
        print(f"- {pat}")
