import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from graph import Neo4jDatabaseManager, create_query_engine
from graph.query_engine import ProofRequirements
from graph.models import ProofSystem


def test_graph_query_basic():
    db = Neo4jDatabaseManager()
    db.connect()
    db.seed_core_primitives()

    engine = create_query_engine(db)

    req = ProofRequirements(proof_system=ProofSystem.GROTH16)
    res = engine.query_primitives(req)

    assert res.primitives, "Expected at least one primitive"
    assert isinstance(res.relevance_scores, list)
