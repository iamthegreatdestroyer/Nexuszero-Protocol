# Agent Utilization Guide

## Overview

Eight core agents provide specialized expertise. Invoke by supplying the relevant prompt template and replacing `[SPECIFIC TASK]` with your actionable objective.

## How To Use

1. Choose agent based on task domain.
2. Copy its `prompt_template.md` content.
3. Replace `[SPECIFIC TASK]` with a clear, testable deliverable.
4. Provide any code paths or constraints.
5. Run generation (e.g., via GitHub Copilot chat or automation script) with the filled template.

## Agents

- Dr. Alex Cipher (`dr_alex_cipher`): Cryptography design & review.
- Morgan Rustico (`morgan_rustico`): High-performance Rust implementation.
- Taylor Frontend (`taylor_frontend`): TypeScript SDK & React demos.
- Dr. Asha Neural (`dr_asha_neural`): ML/GNN proof optimization.
- Jordan Ops (`jordan_ops`): DevOps, CI/CD, infra reliability.
- Sam Sentinel (`sam_sentinel`): Security audits & threat modeling.
- Dana Docs (`dana_docs`): Documentation & IP strategy.
- Quinn Quality (`quinn_quality`): Testing & QA automation.

## Recommended Workflow Integration

- Planning: Quinn Quality + Dr. Alex Cipher to validate scope & cryptographic risk.
- Implementation: Morgan Rustico for Rust core, Taylor Frontend for SDK, Dr. Asha Neural for optimizer.
- Security & Reliability: Sam Sentinel + Jordan Ops integrate scans and monitors.
- Documentation & IP: Dana Docs captures novel techniques early for patent drafts.

## Prompt Enhancement Tips

- Add explicit performance/security targets.
- Include file paths to focus generation.
- Provide existing interfaces for extension vs. rewrite.

## Example Combined Session

"Use Morgan Rustico to implement a new range proof API in `nexuszero-crypto/src/proof/bulletproofs.rs` targeting <5ms for 8-bit proofs; then have Quinn Quality generate property-based tests and Sam Sentinel assess timing side-channel risks."

## Next Improvements

- Add automation script to rotate through agents for multi-step tasks.
- Introduce evaluation metrics per agent output.
