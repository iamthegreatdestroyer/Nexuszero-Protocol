"""Proof Optimization API Endpoints"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any
from enum import Enum
import uuid
import logging

logger = logging.getLogger("OptimizationAPI")
router = APIRouter(prefix="/api/v1/optimization", tags=["Optimization"])

_jobs: Dict[str, Dict[str, Any]] = {}


class ProofType(str, Enum):
    GROTH16 = "groth16"
    PLONK = "plonk"
    BULLETPROOFS = "bulletproofs"
    STARK = "stark"


class OptimizationTarget(str, Enum):
    PROVING_TIME = "proving_time"
    VERIFICATION_TIME = "verification_time"
    PROOF_SIZE = "proof_size"
    CONSTRAINT_COUNT = "constraint_count"


class OptimizationRequest(BaseModel):
    proof_type: ProofType
    circuit_hash: str
    constraints: Dict[str, Any] = Field(default_factory=lambda: {"count": 10000})
    optimization_target: OptimizationTarget = OptimizationTarget.PROVING_TIME
    max_iterations: int = Field(default=10, ge=1, le=100)


class OptimizationResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/submit", response_model=OptimizationResponse)
async def submit_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "request": request.dict(), "result": None, "error": None}

    async def run_optimization():
        try:
            from src.main import app_state

            if app_state.supervisor:
                # Import when needed to avoid hard import-time dependency
                from agents.supervisor import ProofOptimizationTask

                task = ProofOptimizationTask(
                    task_id=job_id,
                    proof_type=request.proof_type.value,
                    circuit_hash=request.circuit_hash,
                    constraints=request.constraints,
                    optimization_target=request.optimization_target.value,
                    max_iterations=request.max_iterations,
                )

                _jobs[job_id]["status"] = "running"
                result = await app_state.supervisor.optimize_proof(task)
                _jobs[job_id]["status"] = "completed" if result.success else "failed"
                _jobs[job_id]["result"] = result.to_dict()
            else:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = "Supervisor not available"
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            logger.error(f"Optimization job {job_id} failed: {e}")

    background_tasks.add_task(run_optimization)
    return OptimizationResponse(job_id=job_id, status="queued", message="Optimization job submitted")


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get("/agents")
async def list_agents():
    from src.main import app_state
    if not app_state.supervisor:
        raise HTTPException(status_code=503, detail="Supervisor not available")
    return {"agents": app_state.supervisor.get_agent_statistics()}
