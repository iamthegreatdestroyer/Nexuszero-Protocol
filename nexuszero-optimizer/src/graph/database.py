"""
Neo4j Database Manager for Cryptographic Primitive Graph
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager

from .models import (
    CryptographicPrimitive,
    ProofConstructionPattern,
    PrimitiveRelationship,
    RelationshipType,
    PrimitiveCategory,
    ProofSystem,
    CORE_PRIMITIVES,
)

logger = logging.getLogger("Neo4jManager")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False


class Neo4jDatabaseManager:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = None
        self._use_fallback = not NEO4J_AVAILABLE
        self._fallback_nodes: Dict[str, CryptographicPrimitive] = {}
        self._fallback_relationships: List[PrimitiveRelationship] = []

    def connect(self):
        if self._use_fallback:
            logger.info("Using in-memory fallback storage")
            return
        try:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._username, self._password))
            with self._driver.session(database=self._database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self._uri}")
        except Exception as e:
            logger.warning(f"Neo4j unavailable: {e}. Using fallback.")
            self._use_fallback = True

    def disconnect(self):
        if self._driver:
            self._driver.close()

    @contextmanager
    def session(self):
        if self._use_fallback:
            yield None
        else:
            session = self._driver.session(database=self._database)
            try:
                yield session
            finally:
                session.close()

    def initialize_schema(self) -> None:
        if self._use_fallback:
            return
        with self.session() as session:
            # Use CREATE INDEX IF NOT EXISTS (Neo4j 4.x+ uses different syntax); keep defensive
            session.run("CREATE INDEX primitive_id IF NOT EXISTS FOR (p:Primitive) ON (p.primitive_id)")
            session.run("CREATE INDEX category IF NOT EXISTS FOR (p:Primitive) ON (p.category)")
        logger.info("Schema initialized")

    def add_primitive(self, primitive: CryptographicPrimitive) -> None:
        if self._use_fallback:
            self._fallback_nodes[primitive.primitive_id] = primitive
            return
        with self.session() as session:
            session.run(
                """
                MERGE (p:Primitive {primitive_id: $id})
                SET p += $props
                """,
                id=primitive.primitive_id,
                props=primitive.to_dict(),
            )

    def get_primitive(self, primitive_id: str) -> Optional[CryptographicPrimitive]:
        if self._use_fallback:
            return self._fallback_nodes.get(primitive_id)
        with self.session() as session:
            result = session.run(
                """
                MATCH (p:Primitive {primitive_id: $id}) RETURN p
                """,
                id=primitive_id,
            )
            record = result.single()
            if record:
                props = dict(record["p"])
                # Convert fields back to enums and types
                props["category"] = PrimitiveCategory(props["category"])
                props["compatible_proof_systems"] = [ProofSystem(ps) for ps in props.get("compatible_proof_systems", [])]
                return CryptographicPrimitive(**{k: v for k, v in props.items() if k in CryptographicPrimitive.__dataclass_fields__})
        return None

    def add_relationship(self, rel: PrimitiveRelationship) -> None:
        if self._use_fallback:
            self._fallback_relationships.append(rel)
            return
        with self.session() as session:
            session.run(
                f"""
                MATCH (a:Primitive {{primitive_id: $source}})
                MATCH (b:Primitive {{primitive_id: $target}})
                MERGE (a)-[r:{rel.relationship_type.value}]->(b)
                SET r.weight = $weight, r.confidence = $confidence
                """,
                source=rel.source_id,
                target=rel.target_id,
                weight=rel.weight,
                confidence=rel.confidence,
            )

    def find_by_category(self, category: PrimitiveCategory) -> List[CryptographicPrimitive]:
        if self._use_fallback:
            return [p for p in self._fallback_nodes.values() if p.category == category]
        with self.session() as session:
            result = session.run(
                """
                MATCH (p:Primitive {category: $cat}) RETURN p
                """,
                cat=category.value,
            )
            return [self._record_to_primitive(r["p"]) for r in result]

    def find_by_proof_system(self, proof_system: ProofSystem) -> List[CryptographicPrimitive]:
        if self._use_fallback:
            return [p for p in self._fallback_nodes.values() if proof_system in p.compatible_proof_systems]
        with self.session() as session:
            result = session.run(
                """
                MATCH (p:Primitive) WHERE $ps IN p.compatible_proof_systems RETURN p
                """,
                ps=proof_system.value,
            )
            return [self._record_to_primitive(r["p"]) for r in result]

    def find_dependencies(self, primitive_id: str) -> List[Tuple[CryptographicPrimitive, str]]:
        if self._use_fallback:
            deps: List[Tuple[CryptographicPrimitive, str]] = []
            for rel in self._fallback_relationships:
                if rel.source_id == primitive_id and rel.relationship_type == RelationshipType.DEPENDS_ON:
                    if rel.target_id in self._fallback_nodes:
                        deps.append((self._fallback_nodes[rel.target_id], rel.relationship_type.value))
            return deps
        with self.session() as session:
            result = session.run(
                """
                MATCH (p:Primitive {primitive_id: $id})-[r]->(dep:Primitive)
                RETURN dep, type(r) as rel_type
                """,
                id=primitive_id,
            )
            return [(self._record_to_primitive(r["dep"]), r["rel_type"]) for r in result]

    def find_path(self, from_id: str, to_id: str) -> List[str]:
        if self._use_fallback:
            return [from_id, to_id] if from_id in self._fallback_nodes and to_id in self._fallback_nodes else []
        with self.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath((a:Primitive {primitive_id: $from})-[*]-(b:Primitive {primitive_id: $to}))
                RETURN [n IN nodes(path) | n.primitive_id] as node_ids
                """,
                **{"from": from_id, "to": to_id},
            )
            record = result.single()
            return record["node_ids"] if record else []

    def _record_to_primitive(self, record) -> CryptographicPrimitive:
        props = dict(record)
        props["category"] = PrimitiveCategory(props["category"])
        props["security_assumptions"] = [SecurityAssumption(sa) for sa in props.get("security_assumptions", [])]
        props["compatible_proof_systems"] = [ProofSystem(ps) for ps in props.get("compatible_proof_systems", [])]
        return CryptographicPrimitive(**{k: v for k, v in props.items() if k in CryptographicPrimitive.__dataclass_fields__})

    def seed_core_primitives(self) -> None:
        for primitive in CORE_PRIMITIVES:
            self.add_primitive(primitive)
        core_relationships: List[PrimitiveRelationship] = [
            PrimitiveRelationship("kzg_commitment", "msm", RelationshipType.DEPENDS_ON),
            PrimitiveRelationship("bulletproofs_ipa", "pedersen_commitment", RelationshipType.COMPOSED_OF),
            PrimitiveRelationship("fri", "ntt", RelationshipType.DEPENDS_ON),
            PrimitiveRelationship("poseidon_hash", "ntt", RelationshipType.OPTIMIZES),
            PrimitiveRelationship("ring_lwe", "ntt", RelationshipType.DEPENDS_ON),
        ]
        for rel in core_relationships:
            self.add_relationship(rel)
        logger.info(f"Seeded {len(CORE_PRIMITIVES)} primitives and {len(core_relationships)} relationships")
