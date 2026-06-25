//! Real Groth16 ZK-SNARK implementation using arkworks.
//!
//! This module provides actual pairing-based zero-knowledge proofs
//! on the BN254 curve, replacing the stub that delegated to the
//! generic lattice proof system.

use ark_bn254::{Bn254, Fr};
use ark_groth16::{Groth16, PreparedVerifyingKey, ProvingKey};
use ark_r1cs_std::prelude::*;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_snark::SNARK;
use ark_std::rand::thread_rng;
use ark_ff::Field;

/// A simple hash preimage circuit: proves knowledge of x such that x^3 + x + 5 = y
/// This demonstrates real R1CS constraint generation on BN254.
#[derive(Clone)]
pub struct CubePreimageCircuit {
    pub x: Option<Fr>,
}

impl ConstraintSynthesizer<Fr> for CubePreimageCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let x = cs.new_witness_variable(|| self.x.ok_or(SynthesisError::AssignmentMissing))?;
        let x_sq = cs.new_witness_variable(|| {
            let x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            Ok(x_val * x_val)
        })?;
        let x_cu = cs.new_witness_variable(|| {
            let x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            Ok(x_val * x_val * x_val)
        })?;

        // x * x = x_sq
        cs.enforce_constraint(
            ark_relations::r1cs::LinearCombination::from(x),
            ark_relations::r1cs::LinearCombination::from(x),
            ark_relations::r1cs::LinearCombination::from(x_sq),
        )?;

        // x_sq * x = x_cu
        cs.enforce_constraint(
            ark_relations::r1cs::LinearCombination::from(x_sq),
            ark_relations::r1cs::LinearCombination::from(x),
            ark_relations::r1cs::LinearCombination::from(x_cu),
        )?;

        // y = x_cu + x + 5 (public output)
        let five = Fr::from(5u64);
        let y = cs.new_input_variable(|| {
            let x_val = self.x.ok_or(SynthesisError::AssignmentMissing)?;
            Ok(x_val * x_val * x_val + x_val + five)
        })?;

        // x_cu + x + 5 = y
        cs.enforce_constraint(
            ark_relations::r1cs::LinearCombination::from(x_cu)
                + ark_relations::r1cs::LinearCombination::from(x)
                + (five, ark_relations::r1cs::Variable::One),
            ark_relations::r1cs::LinearCombination::from(ark_relations::r1cs::Variable::One),
            ark_relations::r1cs::LinearCombination::from(y),
        )?;

        Ok(())
    }
}

/// Generate a trusted setup (proving key + verifying key) for the circuit.
pub fn setup() -> Result<(ProvingKey<Bn254>, PreparedVerifyingKey<Bn254>), String> {
    let circuit = CubePreimageCircuit { x: None };
    let mut rng = thread_rng();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng)
        .map_err(|e| format!("setup failed: {}", e))?;
    let pvk = Groth16::<Bn254>::process_vk(&vk)
        .map_err(|e| format!("process_vk failed: {}", e))?;
    Ok((pk, pvk))
}

/// Generate a proof that we know x such that x^3 + x + 5 = y.
pub fn prove(pk: &ProvingKey<Bn254>, secret: u64) -> Result<(ark_groth16::Proof<Bn254>, Vec<Fr>), String> {
    let x = Fr::from(secret);
    let five = Fr::from(5u64);
    let y = x * x * x + x + five;

    let circuit = CubePreimageCircuit { x: Some(x) };
    let mut rng = thread_rng();
    let proof = Groth16::<Bn254>::prove(pk, circuit, &mut rng)
        .map_err(|e| format!("prove failed: {}", e))?;
    Ok((proof, vec![y]))
}

/// Verify a proof against the public output.
pub fn verify(
    pvk: &PreparedVerifyingKey<Bn254>,
    proof: &ark_groth16::Proof<Bn254>,
    public_inputs: &[Fr],
) -> Result<bool, String> {
    Groth16::<Bn254>::verify_with_processed_vk(pvk, public_inputs, proof)
        .map_err(|e| format!("verify failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groth16_prove_and_verify() {
        let (pk, pvk) = setup().expect("setup should succeed");

        // Prove knowledge of x=3: 3^3 + 3 + 5 = 35
        let (proof, public_inputs) = prove(&pk, 3).expect("prove should succeed");
        assert!(verify(&pvk, &proof, &public_inputs).expect("verify should succeed"));
    }

    #[test]
    fn test_groth16_wrong_proof_fails() {
        let (pk, pvk) = setup().expect("setup should succeed");

        // Prove for x=3
        let (proof, _) = prove(&pk, 3).expect("prove should succeed");

        // Try to verify with wrong public input (x=4: 4^3+4+5=73)
        let wrong_inputs = vec![Fr::from(73u64)];
        let result = verify(&pvk, &proof, &wrong_inputs).expect("verify should not error");
        assert!(!result, "proof for x=3 should NOT verify against y=73");
    }

    #[test]
    fn test_groth16_different_secrets() {
        let (pk, pvk) = setup().expect("setup should succeed");

        for secret in [1u64, 7, 42, 100] {
            let (proof, public_inputs) = prove(&pk, secret).expect("prove should succeed");
            assert!(
                verify(&pvk, &proof, &public_inputs).expect("verify should succeed"),
                "proof for secret={} should verify",
                secret
            );
        }
    }
}
