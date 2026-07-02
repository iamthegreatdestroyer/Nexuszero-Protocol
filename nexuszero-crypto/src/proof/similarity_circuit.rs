//! ZK-RAG Tier-B: zero-knowledge similarity-threshold proof (Phase 3, Dir 2).
//!
//! Proves, in zero knowledge, that a committed chunk embedding `qa` is at least
//! `tau`-cosine-similar to a (publicly committed) query embedding `q`, WITHOUT
//! revealing `qa`. Combined with Tier-A membership this lets an answer prove a
//! citation is BOTH a committed vault member AND semantically relevant, while
//! hiding the embedding and the rest of the corpus.
//!
//! ## Method (sound arithmetic, one honest-indexing trust boundary)
//! Embeddings are L2-normalized then fixed-point quantized (scale `SCALE`), so
//! for unit vectors `cos(a,b) == <a,b>` and the integer dot product
//! `D = <qa_q, q_q> ≈ cos * SCALE^2`. The circuit proves `D >= T` where
//! `T = round(tau * SCALE^2)` via a range proof on `D - T`.
//!
//! Vectors are committed with the SAME proven 2-to-1 Poseidon CRH used by the
//! Merkle circuit, folded as a small binary tree over the (power-of-two) vector
//! — `leaf = fold(qa)`. So no new hash primitive is introduced.
//!
//! Public inputs: `[root, leaf, query_commit, threshold]`.
//! Private witness: `qa`, `query`, and the Merkle path (siblings + directions).
//! Constraints:
//!   1. `leaf == fold(qa)`          — binds the public leaf to the private qa.
//!   2. `leaf ∈ tree(root)`         — qa is a committed vault member.
//!   3. `query_commit == fold(query)` — binds the public query commitment.
//!   4. `D = Σ qa_i * query_i`.
//!   5. `D - threshold ∈ [0, 2^RANGE_BITS)` — i.e. `D >= threshold`.
//!
//! **Trust boundary (documented, unavoidable without the embedder in-circuit):**
//! the circuit does NOT prove `qa` is the genuine embedding of the cited text —
//! that would require running the embedding model inside the circuit. It proves
//! relevance of the COMMITTED vector; the residual trust is honest indexing at
//! commit time (the same actor already signs the vault root).

use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::crh::{
    poseidon::{
        constraints::{CRHParametersVar, TwoToOneCRHGadget},
        TwoToOneCRH,
    },
    TwoToOneCRHScheme, TwoToOneCRHSchemeGadget,
};
use ark_ff::{PrimeField, Zero};
use ark_groth16::{Groth16, PreparedVerifyingKey, ProvingKey};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::convert::ToBitsGadget;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_r1cs_std::fields::FieldVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_snark::SNARK;
use ark_std::rand::thread_rng;

use super::merkle_circuit::{hash_two, poseidon_config, MerklePath, MerkleTree};
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;

/// Vector dimension the circuit is compiled for. MUST be a power of two. Real
/// 768-d embeddings are L2-normalized then zero-padded to this length (zeros
/// change neither the norm nor the dot product). Kept small here to keep the
/// constraint count and setup/prove time modest; raise (e.g. 1024) for prod.
/// Validated correct + sound at SIM_DIM=256 (real-embedding scale); the
/// dimension-generic tests below run at any SIM_DIM. Cost is linear in
/// SIM_DIM (two vector folds dominate); use release builds for prod dims.
pub const SIM_DIM: usize = 8;

/// Fixed-point scale for quantizing unit-vector components. `SCALE = 2^12`.
pub const SCALE: i64 = 4096;

/// Range-proof width for `D - T`. Must satisfy `2^RANGE_BITS > 2 * SIM_DIM *
/// SCALE^2` and stay well under the field size. For SIM_DIM<=1024, 2*1024*2^24 =
/// 2^35, so 40 bits is ample.
pub const RANGE_BITS: usize = 40;

// ---------------------------------------------------------------------------
// Quantization helpers (native)
// ---------------------------------------------------------------------------

/// Map a signed integer to a field element (negatives wrap to `p - |x|`).
pub fn fr_from_i64(x: i64) -> Fr {
    if x >= 0 {
        Fr::from(x as u64)
    } else {
        -Fr::from((-x) as u64)
    }
}

/// L2-normalize `v` and fixed-point quantize to `Fr` (length must be SIM_DIM).
pub fn quantize_unit(v: &[f64]) -> Vec<Fr> {
    assert_eq!(v.len(), SIM_DIM, "vector must be length SIM_DIM (pad with zeros)");
    let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    let n = if norm == 0.0 { 1.0 } else { norm };
    v.iter()
        .map(|a| fr_from_i64(((a / n) * SCALE as f64).round() as i64))
        .collect()
}

/// Threshold field element for a cosine cutoff `tau`: `round(tau * SCALE^2)`.
pub fn threshold_fr(tau: f64) -> Fr {
    fr_from_i64((tau * (SCALE as f64) * (SCALE as f64)).round() as i64)
}

/// Commit a (power-of-two length) vector as the root of a binary 2-to-1 Poseidon
/// fold — the leaf value stored in the embedding Merkle tree. Native.
pub fn commit_vector(params: &PoseidonConfig<Fr>, elems: &[Fr]) -> Fr {
    assert!(elems.len().is_power_of_two() && !elems.is_empty());
    let mut layer = elems.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks(2) {
            next.push(hash_two(params, pair[0], pair[1]));
        }
        layer = next;
    }
    layer[0]
}

// ---------------------------------------------------------------------------
// The circuit
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SimilarityCircuit {
    pub params: PoseidonConfig<Fr>,
    // PUBLIC
    pub root: Option<Fr>,
    pub leaf: Option<Fr>,
    pub query_commit: Option<Fr>,
    pub threshold: Option<Fr>,
    // PRIVATE
    pub qa: Option<Vec<Fr>>,      // committed chunk embedding (quantized), len SIM_DIM
    pub query: Option<Vec<Fr>>,   // query embedding (quantized), len SIM_DIM
    pub siblings: Option<Vec<Fr>>,
    pub index_bits: Option<Vec<bool>>,
    pub depth: usize,
}

impl SimilarityCircuit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: PoseidonConfig<Fr>,
        root: Fr,
        leaf: Fr,
        query_commit: Fr,
        threshold: Fr,
        qa: Vec<Fr>,
        query: Vec<Fr>,
        path: &MerklePath,
    ) -> Self {
        Self {
            params,
            root: Some(root),
            leaf: Some(leaf),
            query_commit: Some(query_commit),
            threshold: Some(threshold),
            qa: Some(qa),
            query: Some(query),
            siblings: Some(path.siblings.clone()),
            index_bits: Some(path.index_bits.clone()),
            depth: path.siblings.len(),
        }
    }

    pub fn empty(params: PoseidonConfig<Fr>, depth: usize) -> Self {
        Self {
            params,
            root: None,
            leaf: None,
            query_commit: None,
            threshold: None,
            qa: None,
            query: None,
            siblings: None,
            index_bits: None,
            depth,
        }
    }
}

/// In-circuit binary 2-to-1 Poseidon fold of a power-of-two vector of FpVars.
fn commit_vector_gadget(
    params_var: &CRHParametersVar<Fr>,
    elems: &[FpVar<Fr>],
) -> Result<FpVar<Fr>, SynthesisError> {
    let mut layer = elems.to_vec();
    while layer.len() > 1 {
        let mut next = Vec::with_capacity(layer.len() / 2);
        for pair in layer.chunks(2) {
            next.push(
                <TwoToOneCRHGadget<Fr> as TwoToOneCRHSchemeGadget<TwoToOneCRH<Fr>, Fr>>::compress(
                    params_var, &pair[0], &pair[1],
                )?,
            );
        }
        layer = next;
    }
    Ok(layer.pop().unwrap())
}

impl ConstraintSynthesizer<Fr> for SimilarityCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let params_var = CRHParametersVar::<Fr>::new_constant(cs.clone(), self.params.clone())?;

        // Public inputs, order = [root, leaf, query_commit, threshold].
        let root_var = FpVar::<Fr>::new_input(cs.clone(), || {
            self.root.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let leaf_var = FpVar::<Fr>::new_input(cs.clone(), || {
            self.leaf.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let qc_var = FpVar::<Fr>::new_input(cs.clone(), || {
            self.query_commit.ok_or(SynthesisError::AssignmentMissing)
        })?;
        let thr_var = FpVar::<Fr>::new_input(cs.clone(), || {
            self.threshold.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Private vectors.
        let mut qa_vars = Vec::with_capacity(SIM_DIM);
        let mut q_vars = Vec::with_capacity(SIM_DIM);
        for i in 0..SIM_DIM {
            qa_vars.push(FpVar::<Fr>::new_witness(cs.clone(), || {
                let v = self.qa.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(v[i])
            })?);
            q_vars.push(FpVar::<Fr>::new_witness(cs.clone(), || {
                let v = self.query.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(v[i])
            })?);
        }

        // (1) leaf == fold(qa) — binds the public leaf to the private embedding.
        let computed_leaf = commit_vector_gadget(&params_var, &qa_vars)?;
        computed_leaf.enforce_equal(&leaf_var)?;

        // (2) leaf ∈ tree(root) — Merkle membership (fold up the private path).
        let mut cur = leaf_var.clone();
        for level in 0..self.depth {
            let sibling = FpVar::<Fr>::new_witness(cs.clone(), || {
                let s = self.siblings.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(s[level])
            })?;
            let is_right = Boolean::new_witness(cs.clone(), || {
                let b = self.index_bits.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(b[level])
            })?;
            let left = is_right.select(&sibling, &cur)?;
            let right = is_right.select(&cur, &sibling)?;
            cur = <TwoToOneCRHGadget<Fr> as TwoToOneCRHSchemeGadget<TwoToOneCRH<Fr>, Fr>>::compress(
                &params_var, &left, &right,
            )?;
        }
        cur.enforce_equal(&root_var)?;

        // (3) query_commit == fold(query) — binds the public query commitment.
        let computed_qc = commit_vector_gadget(&params_var, &q_vars)?;
        computed_qc.enforce_equal(&qc_var)?;

        // (4) D = Σ qa_i * query_i.
        let mut acc = FpVar::<Fr>::zero();
        for i in 0..SIM_DIM {
            acc += &qa_vars[i] * &q_vars[i];
        }

        // (5) range proof: D - threshold ∈ [0, 2^RANGE_BITS) ⇒ D >= threshold.
        // A negative signed value maps to a ~254-bit field element, whose high
        // bits cannot all be zero — so this rejects D < threshold.
        let diff = &acc - &thr_var;
        let bits = diff.to_bits_le()?;
        for b in bits.iter().skip(RANGE_BITS) {
            b.enforce_equal(&Boolean::constant(false))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// setup / prove / verify
// ---------------------------------------------------------------------------

pub fn setup(depth: usize) -> Result<(ProvingKey<Bn254>, PreparedVerifyingKey<Bn254>), String> {
    let params = poseidon_config();
    let circuit = SimilarityCircuit::empty(params, depth);
    let mut rng = thread_rng();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng)
        .map_err(|e| format!("similarity setup failed: {}", e))?;
    let pvk = Groth16::<Bn254>::process_vk(&vk).map_err(|e| format!("process_vk failed: {}", e))?;
    Ok((pk, pvk))
}

/// Public inputs for a similarity proof, in circuit order.
pub fn public_inputs(root: Fr, leaf: Fr, query_commit: Fr, threshold: Fr) -> Vec<Fr> {
    vec![root, leaf, query_commit, threshold]
}

#[allow(clippy::too_many_arguments)]
pub fn prove(
    pk: &ProvingKey<Bn254>,
    root: Fr,
    leaf: Fr,
    query_commit: Fr,
    threshold: Fr,
    qa: Vec<Fr>,
    query: Vec<Fr>,
    path: &MerklePath,
) -> Result<(ark_groth16::Proof<Bn254>, Vec<Fr>), String> {
    use ark_relations::r1cs::{ConstraintSystem, OptimizationGoal};
    let params = poseidon_config();
    let circuit = SimilarityCircuit::new(
        params, root, leaf, query_commit, threshold, qa, query, path,
    );

    // Honesty guard: if the witness doesn't satisfy the circuit (e.g. the true
    // similarity is below threshold, or a forged/non-member leaf), return Err
    // instead of letting Groth16::prove hit its debug_assert. A prover who does
    // not actually meet the threshold simply cannot produce a proof.
    {
        let cs = ConstraintSystem::<Fr>::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Constraints);
        circuit
            .clone()
            .generate_constraints(cs.clone())
            .map_err(|e| format!("constraint synthesis failed: {}", e))?;
        if !cs.is_satisfied().map_err(|e| format!("is_satisfied failed: {}", e))? {
            return Err("witness does not satisfy the similarity circuit \
                        (below threshold, or leaf/path not committed)"
                .to_string());
        }
    }

    let mut rng = thread_rng();
    let proof = Groth16::<Bn254>::prove(pk, circuit, &mut rng)
        .map_err(|e| format!("similarity prove failed: {}", e))?;
    Ok((proof, public_inputs(root, leaf, query_commit, threshold)))
}

pub fn verify(
    pvk: &PreparedVerifyingKey<Bn254>,
    proof: &ark_groth16::Proof<Bn254>,
    public_inputs: &[Fr],
) -> Result<bool, String> {
    Groth16::<Bn254>::verify_with_processed_vk(pvk, public_inputs, proof)
        .map_err(|e| format!("similarity verify failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit vectors (length SIM_DIM=8). qa and q1 overlap strongly (cos≈0.87);
    // q2 is orthogonal to qa (cos=0).
    fn vecs() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Dimension-generic so the same tests exercise any SIM_DIM: qa = first
        // half ones; q1 overlaps qa strongly (cos high); q2 = orthogonal second
        // half (cos 0).
        let half = SIM_DIM / 2;
        let qa = (0..SIM_DIM).map(|i| if i < half { 1.0 } else { 0.0 }).collect();
        let q1 = (0..SIM_DIM).map(|i| if i < half - 1 { 1.0 } else { 0.0 }).collect();
        let q2 = (0..SIM_DIM).map(|i| if i >= half { 1.0 } else { 0.0 }).collect();
        (qa, q1, q2)
    }

    fn build_tree(params: &PoseidonConfig<Fr>, embeddings: &[Vec<f64>]) -> MerkleTree {
        let leaves: Vec<Fr> = embeddings
            .iter()
            .map(|e| commit_vector(params, &quantize_unit(e)))
            .collect();
        MerkleTree::new(params.clone(), leaves)
    }

    #[test]
    fn test_quantized_dot_tracks_cosine() {
        let params = poseidon_config();
        let _ = &params;
        let (qa, q1, q2) = vecs();
        let a = quantize_unit(&qa);
        let b1 = quantize_unit(&q1);
        let b2 = quantize_unit(&q2);
        // integer dot products
        let dot = |x: &[Fr], y: &[Fr]| -> Fr {
            let mut s = Fr::zero();
            for i in 0..SIM_DIM {
                s += x[i] * y[i];
            }
            s
        };
        let d1 = dot(&a, &b1);
        let d2 = dot(&a, &b2);
        // cos(qa,q1)=0.866 -> ~0.866*SCALE^2 ; cos(qa,q2)=0 -> ~0
        let t_hi = threshold_fr(0.7);
        let t_lo = threshold_fr(0.1);
        // d1 - t_hi >= 0 (represented small), d2 - t_lo is negative (huge field elt)
        assert!(fr_is_small_nonneg(d1 - t_hi), "d1 should clear tau=0.7");
        assert!(!fr_is_small_nonneg(d2 - t_lo), "d2 (orthogonal) should miss tau=0.1");
    }

    // True iff x, as a signed value, is in [0, 2^RANGE_BITS) — mirrors the circuit.
    fn fr_is_small_nonneg(x: Fr) -> bool {
        let bits = x.into_bigint();
        // check no bit at index >= RANGE_BITS is set
        let bytes = {
            use ark_ff::BigInteger;
            bits.to_bytes_le()
        };
        for (i, byte) in bytes.iter().enumerate() {
            for bit in 0..8 {
                let idx = i * 8 + bit;
                if idx >= RANGE_BITS && (byte >> bit) & 1 == 1 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_similarity_prove_and_verify() {
        let params = poseidon_config();
        let (qa, q1, q2) = vecs();
        // corpus of 4 chunk embeddings; qa is index 2.
        let mut corpus: Vec<Vec<f64>> = (0..4)
            .map(|j| (0..SIM_DIM).map(|i| ((i + j) % 3) as f64).collect())
            .collect();
        corpus[2] = qa.clone(); // cited chunk at index 2
        let tree = build_tree(&params, &corpus);
        let root = tree.root();
        let idx = 2usize;
        let path = tree.path(idx);
        let leaf = tree.path(idx).leaf; // = commit_vector(quantize(qa))
        let depth = path.siblings.len();

        let (pk, pvk) = setup(depth).expect("setup");

        let qa_q = quantize_unit(&qa);
        let q1_q = quantize_unit(&q1);
        let qc1 = commit_vector(&params, &q1_q);
        let t_hi = threshold_fr(0.7);

        // Valid: cos(qa,q1)=0.866 >= 0.7 -> proof verifies TRUE.
        let (proof, pubs) =
            prove(&pk, root, leaf, qc1, t_hi, qa_q.clone(), q1_q.clone(), &path).expect("prove");
        assert!(verify(&pvk, &proof, &pubs).expect("verify"), "similar pair must verify");

        // Tamper: wrong query_commit in the public inputs -> reject.
        let bad_pubs = public_inputs(root, leaf, qc1 + Fr::from(1u64), t_hi);
        assert!(!verify(&pvk, &proof, &bad_pubs).unwrap(), "tampered query_commit must reject");

        // Tamper: wrong threshold in the public inputs -> reject.
        let bad_pubs2 = public_inputs(root, leaf, qc1, threshold_fr(0.9));
        assert!(!verify(&pvk, &proof, &bad_pubs2).unwrap(), "changed threshold must reject");

        // Below threshold: cos(qa,q2)=0 < 0.7 -> prover CANNOT produce a proof.
        let q2_q = quantize_unit(&q2);
        let qc2 = commit_vector(&params, &q2_q);
        let res = prove(&pk, root, leaf, qc2, t_hi, qa_q.clone(), q2_q, &path);
        assert!(res.is_err(), "below-threshold similarity must fail to prove");

        // Non-member: a leaf not in the tree -> cannot prove membership.
        let (fa, _, _) = vecs();
        let forged_leaf = commit_vector(&params, &quantize_unit(&fa)) + Fr::from(7u64);
        let res2 = prove(&pk, root, forged_leaf, qc1, t_hi, qa_q, q1_q, &path);
        assert!(res2.is_err(), "non-member leaf must fail to prove");
    }
}
