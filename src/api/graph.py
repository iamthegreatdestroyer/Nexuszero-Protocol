"""Graph-Based Primitive Retrieval API Endpoints"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger("GraphAPI")
router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


class ProofSystemEnum(str, Enum):
    GROTH16 = "groth16"
    PLONK = "plonk"
    BULLETPROOFS = "bulletproofs"
    STARK = "stark"
    HALO2 = "halo2"
    NOVA = "nova"


class CategoryEnum(str, Enum):
    COMMITMENT = "commitment"
    HASH = "hash"
    ZK_PROOF = "zk_proof"
    POLYNOMIAL = "polynomial"
    ELLIPTIC_CURVE = "elliptic_curve"
    LATTICE = "lattice"


class QueryRequest(BaseModel):
    proof_system: Optional[ProofSystemEnum] = None
    requires_transparency: bool = False
    requires_post_quantum: bool = False
    use_case: Optional[str] = None
    max_constraints: Optional[int] = None


class PrimitiveResponse(BaseModel):
    primitive_id: str
    name: str
    category: str
    description: str
    relevance_score: float


class QueryResponse(BaseModel):
    primitives: List[PrimitiveResponse]
    reasoning: str
    suggested_patterns: List[str]


@router.post("/query", response_model=QueryResponse)
async def query_primitives(request: QueryRequest):
    from src.main import app_state

    if not app_state.query_engine:
        raise HTTPException(status_code=503, detail="Graph query engine not available")

    try:
        # Import local types inside function to avoid boot-time import errors
        from graph.query_engine import ProofRequirements
        from graph.models import ProofSystem

        proof_system = None
        if request.proof_system:
            proof_system = ProofSystem(request.proof_system.value)

        requirements = ProofRequirements(
            proof_system=proof_system,
            requires_transparency=request.requires_transparency,
            requires_post_quantum=request.requires_post_quantum,
            use_case=request.use_case,
            max_constraints=request.max_constraints,
        )

        result = app_state.query_engine.query_primitives(requirements)

        primitives = [
            PrimitiveResponse(
                primitive_id=p.primitive_id,
                name=p.name,
                category=p.category.value,
                description=p.description,
                relevance_score=score,
            )
            for p, score in zip(result.primitives, result.relevance_scores)
        ]

        return QueryResponse(primitives=primitives, reasoning=result.reasoning, suggested_patterns=result.suggested_patterns)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/primitives")
async def list_primitives(category: Optional[CategoryEnum] = None):
    from src.main import app_state

    if not app_state.graph_db:
        raise HTTPException(status_code=503, detail="Graph database not available")

    try:
        if category:
            from graph.models import PrimitiveCategory
            primitives = app_state.graph_db.find_by_category(PrimitiveCategory(category.value))
        else:
            # fallback listing
            primitives = list(app_state.graph_db._fallback_nodes.values())

        return {"count": len(primitives), "primitives": [p.to_dict() for p in primitives]}
    except Exception as e:
        logger.error(f"List primitives failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/primitives/{primitive_id}")
async def get_primitive(primitive_id: str):
    from src.main import app_state

    if not app_state.graph_db:
        raise HTTPException(status_code=503, detail="Graph database not available")

    primitive = app_state.graph_db.get_primitive(primitive_id)
    if not primitive:
        raise HTTPException(status_code=404, detail="Primitive not found")

    return primitive.to_dict()


@router.get("/primitives/{primitive_id}/dependencies")
async def get_dependencies(primitive_id: str):
    from src.main import app_state

    if not app_state.graph_db:
        raise HTTPException(status_code=503, detail="Graph database not available")

    deps = app_state.graph_db.find_dependencies(primitive_id)
    return {"primitive_id": primitive_id, "dependencies": [{"primitive": d[0].to_dict(), "relationship": d[1]} for d in deps]}


@router.get("/path")
async def find_path(from_id: str = Query(...), to_id: str = Query(...)):
    from src.main import app_state

    if not app_state.graph_db:
        raise HTTPException(status_code=503, detail="Graph database not available")

    path = app_state.graph_db.find_path(from_id, to_id)
    return {"from": from_id, "to": to_id, "path": path}


@router.get("/recommend")
async def recommend_for_circuit(
    constraint_count: int = Query(default=10000),
    has_range_proofs: bool = Query(default=False),
    has_merkle_trees: bool = Query(default=False),
):
    from src.main import app_state

    if not app_state.query_engine:
        raise HTTPException(status_code=503, detail="Query engine not available")

    result = app_state.query_engine.recommend_for_circuit({
        "constraint_count": constraint_count,
        "has_range_proofs": has_range_proofs,
        "has_merkle_trees": has_merkle_trees,
    })

    return {
        "recommendations": [p.to_dict() for p in result.primitives[:5]],
        "reasoning": result.reasoning,
        "patterns": result.suggested_patterns,
    }
