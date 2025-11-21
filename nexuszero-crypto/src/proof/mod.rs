//! Zero-knowledge proof system components
//!
//! This module provides Statement, Witness, and Proof structures
//! for constructing zero-knowledge proofs.

pub mod proof;
pub mod statement;
pub mod witness;

// Re-export main types
pub use proof::{Proof, ProofMetadata};
pub use statement::{Statement, StatementBuilder, StatementType};
pub use witness::{Witness, WitnessType};
