# Graph-Based Cryptographic Primitive Retrieval

This module implements a lightweight knowledge graph with Neo4j integration for cryptographic primitives used by the Nexuszero Protocol.

Features:

- Graph models representing cryptographic primitives and relationships
- Neo4j database manager with an in-memory fallback for environments without Neo4j
- Query engine for searching primitives, generating reasoning and suggesting composition patterns
- Demo scripts to seed the graph and query the knowledge base

Quick start (Windows PowerShell):

```pwsh
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r nexuszero-optimizer/requirements-graph.txt
python .\nexuszero-optimizer\scripts\demo_graph.py
```

Notes:

- The `Neo4jDatabaseManager` will automatically use an in-memory fallback if `neo4j` python driver is not installed or Neo4j is unavailable.
- The `GraphQueryEngine` contains heuristics for recommending primitives and constructing dependency views; these can be replaced by a semantic search / embedding-based approach for improved matching.

Development tasks to extend:

- Add real circuit/constraint graph conversion and search ranking using embeddings.
- Add integration tests & unit tests for query engine and seeding.
