"""Proof Optimization API Endpoints"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
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


# ---------------------------------------------------------------------------
# Sprint 3 — /optimize endpoint (Phase 6)
# ---------------------------------------------------------------------------

class CircuitOptimizeRequest(BaseModel):
    """Proof parameters for circuit optimization."""
    security_level: int = Field(128, description="Security level in bits (128, 192, or 256)")
    dimension: Optional[int] = Field(None, description="Ring-LWE dimension n (overrides security_level)")
    modulus: Optional[int] = Field(None, description="NTT modulus q (overrides security_level)")
    proof_type: str = Field("ring_lwe", description="Proof type: ring_lwe, bulletproofs, schnorr")
    target: str = Field("proving_time", description="Optimization target")


class CircuitOptimizeResponse(BaseModel):
    """Optimized circuit configuration returned to the Rust caller."""
    dimension: int
    modulus: int
    primitive_root: int
    batch_size: int
    use_ntt_cache: bool
    optimization_note: str


_SECURITY_PRESETS: Dict[int, Dict[str, Any]] = {
    128: {"n": 512,  "q": 12289, "root": 49},
    192: {"n": 1024, "q": 40961, "root": 3},
    256: {"n": 2048, "q": 65537, "root": 3},
}


@router.post("/optimize", response_model=CircuitOptimizeResponse)
async def optimize_circuit(request: CircuitOptimizeRequest) -> CircuitOptimizeResponse:
    """Return an optimized circuit configuration for the given proof parameters.

    This endpoint is designed to be called by the Rust ``nexuszero-integration``
    crate before proof generation.  When the optimizer is unavailable, the Rust
    side falls back to built-in defaults — so this endpoint must never return a
    500 for valid inputs.
    """
    preset = _SECURITY_PRESETS.get(request.security_level)
    if preset is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported security level {request.security_level}. Use 128, 192, or 256.",
        )

    n = request.dimension if request.dimension is not None else preset["n"]
    q = request.modulus if request.modulus is not None else preset["q"]
    root = preset["root"]

    # Heuristic: for small dimensions, skip NTT cache (schoolbook is faster)
    use_ntt_cache = n >= 512

    # Batch size: small for high-security, larger for 128-bit
    batch_size = {128: 16, 192: 8, 256: 4}.get(request.security_level, 8)

    note = (
        f"NTT twiddle cache enabled for n={n}; "
        f"batch_size={batch_size}; target={request.target}"
    )

    logger.info("Circuit optimized: n=%d q=%d batch=%d", n, q, batch_size)
    return CircuitOptimizeResponse(
        dimension=n,
        modulus=q,
        primitive_root=root,
        batch_size=batch_size,
        use_ntt_cache=use_ntt_cache,
        optimization_note=note,
    )
