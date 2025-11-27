"""
Nexuszero Protocol - Main Application Entry Point
Integrates: Agent Supervisor, Graph DB, JWT Auth
"""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

try:
    import uvicorn
except Exception:
    uvicorn = None  # uvicorn optional during import-time checks
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NexuszeroMain")


class AppState:
    def __init__(self):
        self.supervisor = None
        self.graph_db = None
        self.query_engine = None
        self.auth_manager = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Nexuszero Protocol...")

    # Initialize Graph Database
    try:
        from graph.database import Neo4jDatabaseManager
        from graph.query_engine import create_query_engine

        app_state.graph_db = Neo4jDatabaseManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )
        # If Neo4j driver not available, Neo4jDatabaseManager should fallback to in-memory
        app_state.graph_db.connect()
        app_state.graph_db.seed_core_primitives()
        app_state.query_engine = create_query_engine(app_state.graph_db)
        logger.info("Graph database initialized")
    except Exception as e:
        logger.warning(f"Graph DB init failed: {e}")

    # Initialize Agent Supervisor
    try:
        # 'agents' package lives in nexuszero-optimizer/src/agents which is added to PYTHONPATH
        from agents.supervisor import create_proof_optimization_supervisor

        app_state.supervisor = create_proof_optimization_supervisor()
        logger.info("Agent supervisor initialized")
    except Exception as e:
        logger.warning(f"Agent supervisor init failed: {e}")

    # Initialize Auth Manager
    try:
        from src.auth import create_auth_manager

        app_state.auth_manager = create_auth_manager(private_key_path=os.getenv("JWT_PRIVATE_KEY_PATH"))
        logger.info("Auth manager initialized")
    except Exception as e:
        logger.warning(f"Auth manager init failed: {e}")

    yield

    logger.info("Shutting down Nexuszero Protocol...")
    if app_state.graph_db:
        try:
            app_state.graph_db.disconnect()
        except Exception:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nexuszero Protocol API",
        description="Zero-Knowledge Proof System with AI Optimization",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from src.api import health_router, optimization_router, graph_router

    app.include_router(health_router)
    app.include_router(optimization_router)
    app.include_router(graph_router)

    try:
        from src.auth import auth_router

        app.include_router(auth_router, prefix="/auth")
    except ImportError:
        logger.warning("Auth routes not available")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Nexuszero Protocol")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("src.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
