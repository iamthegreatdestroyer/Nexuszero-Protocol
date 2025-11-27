"""
Intelligent Query Engine for Cryptographic Primitive Retrieval
Supports semantic search, path finding, and AI-assisted recommendations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np

from .models import CryptographicPrimitive, ProofSystem, PrimitiveCategory, SecurityAssumption
from .database import Neo4jDatabaseManager

logger = logging.getLogger("GraphQueryEngine")


@dataclass
class QueryResult:
    primitives: List[CryptographicPrimitive]
    relevance_scores: List[float]
    reasoning: str
    suggested_patterns: List[str]


@dataclass
class ProofRequirements:
    proof_system: Optional[ProofSystem] = None
    max_constraints: Optional[int] = None
    requires_transparency: bool = False
    requires_post_quantum: bool = False
    optimization_target: str = "proving_time"
    use_case: Optional[str] = None


class GraphQueryEngine:
    def __init__(self, db_manager: Neo4jDatabaseManager):
        self.db = db_manager
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def query_primitives(self, requirements: ProofRequirements) -> QueryResult:
        """Query primitives based on proof requirements."""
        candidates: List[CryptographicPrimitive] = []
        scores: List[float] = []

        # Filter by proof system
        if requirements.proof_system:
            candidates = self.db.find_by_proof_system(requirements.proof_system)
        else:
            candidates = list(self.db._fallback_nodes.values()) if self.db._use_fallback else []

        # Apply filters and scoring
        filtered: List[CryptographicPrimitive] = []
        for p in candidates:
            score = 1.0

            if requirements.requires_transparency and p.requires_trusted_setup:
                continue
            if requirements.requires_post_quantum and not p.quantum_resistant:
                score *= 0.5
            if requirements.use_case and requirements.use_case in p.common_use_cases:
                score *= 1.5

            score *= (0.5 + 0.5 * p.success_rate)
            filtered.append(p)
            scores.append(score)

        # Sort by score
        sorted_pairs = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
        sorted_primitives = [p for p, _ in sorted_pairs]
        sorted_scores = [s for _, s in sorted_pairs]

        reasoning = self._generate_reasoning(requirements, sorted_primitives[:5])
        patterns = self._suggest_patterns(requirements, sorted_primitives[:5])

        return QueryResult(primitives=sorted_primitives[:10], relevance_scores=sorted_scores[:10], reasoning=reasoning, suggested_patterns=patterns)

    def find_optimal_path(self, start_primitive: str, goal: str) -> List[CryptographicPrimitive]:
        """Find optimal path between primitives for proof construction."""
        path_ids = self.db.find_path(start_primitive, goal)
        return [self.db.get_primitive(pid) for pid in path_ids if self.db.get_primitive(pid)]

    def get_dependencies_tree(self, primitive_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get full dependency tree for a primitive."""
        visited = set()

        def _build_tree(pid: str, depth: int) -> Dict[str, Any]:
            if depth > max_depth or pid in visited:
                return {"id": pid, "dependencies": []}
            visited.add(pid)

            primitive = self.db.get_primitive(pid)
            deps = self.db.find_dependencies(pid)

            return {
                "id": pid,
                "name": primitive.name if primitive else pid,
                "category": primitive.category.value if primitive else "unknown",
                "dependencies": [_build_tree(d[0].primitive_id, depth + 1) for d in deps],
            }

        return _build_tree(primitive_id, 0)

    def recommend_for_circuit(self, circuit_info: Dict[str, Any]) -> QueryResult:
        """Recommend primitives based on circuit characteristics."""
        constraint_count = circuit_info.get("constraint_count", 10000)
        has_range_proofs = circuit_info.get("has_range_proofs", False)
        has_merkle_trees = circuit_info.get("has_merkle_trees", False)

        requirements = ProofRequirements(
            max_constraints=constraint_count,
            use_case="range proofs" if has_range_proofs else ("merkle trees" if has_merkle_trees else None),
        )

        return self.query_primitives(requirements)

    def _generate_reasoning(self, req: ProofRequirements, primitives: List[CryptographicPrimitive]) -> str:
        parts = ["Recommended primitives based on:"]
        if req.proof_system:
            parts.append(f"- Compatibility with {req.proof_system.value}")
        if req.requires_transparency:
            parts.append("- No trusted setup requirement")
        if req.requires_post_quantum:
            parts.append("- Post-quantum security")
        if req.use_case:
            parts.append(f"- Use case: {req.use_case}")
        if primitives:
            parts.append(f"\nTop recommendation: {primitives[0].name} - {primitives[0].description}")
        return "\n".join(parts)

    def _suggest_patterns(self, req: ProofRequirements, primitives: List[CryptographicPrimitive]) -> List[str]:
        patterns: List[str] = []
        categories = {p.category for p in primitives}

        if PrimitiveCategory.COMMITMENT in categories and PrimitiveCategory.HASH in categories:
            patterns.append("Commitment-Hash Pattern: Use for hiding values with efficient verification")
        if PrimitiveCategory.POLYNOMIAL in categories:
            patterns.append("Polynomial Pattern: Efficient for batch operations and lookups")
        if any(p.quantum_resistant for p in primitives):
            patterns.append("Post-Quantum Pattern: Future-proof cryptographic construction")

        return patterns


def create_query_engine(db_manager: Neo4jDatabaseManager) -> GraphQueryEngine:
    return GraphQueryEngine(db_manager)
