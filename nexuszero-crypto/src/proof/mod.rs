//! Zero-knowledge proof system components
//!
//! This module provides Statement, Witness, and Proof structures
//! for constructing zero-knowledge proofs.

pub mod bulletproofs;
pub mod proof;
pub mod statement;
pub mod witness;

// Re-export main types
pub use proof::{Proof, ProofMetadata};
pub use statement::{Statement, StatementBuilder, StatementType};
pub use witness::{Witness, WitnessType};
pub use bulletproofs::{BulletproofRangeProof, prove_range, verify_range};
