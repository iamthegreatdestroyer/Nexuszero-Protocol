//! Groth16 zero-knowledge Merkle-membership circuit (ZK-RAG Phase 1).
//!
//! This module proves, in zero knowledge, that a leaf value is a committed
//! member of a Merkle tree whose root is public, WITHOUT revealing the leaf,
//! its position, or the sibling path. This is the core primitive that lets the
//! "In My Head" second brain prove a citation is a committed member of a
//! private vault while keeping the vault contents secret.
//!
//! ## Design
//! - PUBLIC input:  the Merkle root (one `Fr`).
//! - PRIVATE witness: the leaf value (`Fr`), the sibling path (`Vec<Fr>`),
//!   and the path directions (`Vec<bool>`, `true` = current node is the right
//!   child, so sibling is on the left).
//! - Constraints: fold the leaf up the tree using a 2-to-1 Poseidon hash,
//!   selecting left/right ordering by the direction bit at each level, then
//!   enforce the computed root equals the public root.
//!
//! ## Hash soundness
//! The 2-to-1 compression is the **real arkworks Poseidon** CRH
//! (`ark_crypto_primitives::crh::poseidon`) over BN254 `Fr`. Round constants
//! (`ark`) and the MDS matrix are derived with the standard Poseidon Grain LFSR
//! (`find_poseidon_ark_and_mds`) using the reference parameter set for a
//! width-3 (rate-2, capacity-1) sponge: alpha = 5, 8 full rounds, 57 partial
//! rounds. These are the widely deployed BN254 Poseidon parameters and are
//! cryptographically sound (NOT a placeholder). The SAME `PoseidonConfig` is
//! used both in-circuit (`TwoToOneCRHGadget`) and out-of-circuit
//! (`TwoToOneCRH`), so the cleartext tree and the proven tree agree by
//! construction.
//!
//! The Groth16 SNARK itself is the real pairing-based `ark_groth16::Groth16`
//! over BN254, mirroring `ark_groth16.rs` in this crate.

use ark_bn254::{Bn254, Fr};
use ark_crypto_primitives::crh::{
    poseidon::{
        constraints::{CRHParametersVar, TwoToOneCRHGadget},
        TwoToOneCRH,
    },
    TwoToOneCRHScheme, TwoToOneCRHSchemeGadget,
};
use ark_crypto_primitives::sponge::poseidon::{find_poseidon_ark_and_mds, PoseidonConfig};
use ark_groth16::{Groth16, PreparedVerifyingKey, ProvingKey};
use ark_r1cs_std::alloc::AllocVar;
use ark_r1cs_std::boolean::Boolean;
use ark_r1cs_std::eq::EqGadget;
use ark_r1cs_std::fields::fp::FpVar;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_snark::SNARK;
use ark_std::rand::thread_rng;

/// Fixed depth of the Merkle tree used by the default circuit.
/// Depth 8 => 2^8 = 256 leaves. The circuit is generic over depth via
/// the length of the provided path, but setup/prove/verify helpers below
/// use this constant so the proving/verifying keys have a fixed shape.
pub const DEFAULT_DEPTH: usize = 8;

/// Build the standard, cryptographically sound BN254 Poseidon parameters for a
/// width-3 sponge (rate = 2, capacity = 1), which is exactly what a 2-to-1
/// compression needs. Parameters are derived deterministically via the Poseidon
/// Grain LFSR reference construction, so every caller gets the identical config.
///
/// Reference parameters for a 255-bit prime field, width 3:
///   alpha = 5, full_rounds = 8, partial_rounds = 57.
pub fn poseidon_config() -> PoseidonConfig<Fr> {
    let full_rounds: usize = 8;
    let partial_rounds: usize = 57;
    let alpha: u64 = 5;
    let rate: usize = 2;
    let capacity: usize = 1;
    // prime_bits for BN254 scalar field Fr.
    let prime_bits: u64 = 254;
    // skip_matrices = 0: take the first qualifying MDS the LFSR produces,
    // matching the arkworks default-parameter convention.
    let (ark, mds) = find_poseidon_ark_and_mds::<Fr>(
        prime_bits,
        rate,
        full_rounds as u64,
        partial_rounds as u64,
        0,
    );
    PoseidonConfig {
        full_rounds,
        partial_rounds,
        alpha,
        ark,
        mds,
        rate,
        capacity,
    }
}

/// Compute a 2-to-1 Poseidon hash of two field elements OUT of circuit.
/// Used to build the cleartext Merkle tree and to derive the expected root.
pub fn hash_two(params: &PoseidonConfig<Fr>, left: Fr, right: Fr) -> Fr {
    <TwoToOneCRH<Fr> as TwoToOneCRHScheme>::compress(params, left, right)
        .expect("poseidon compress should not fail")
}

/// A cleartext Merkle authentication path for one leaf.
#[derive(Clone, Debug)]
pub struct MerklePath {
    /// The leaf value being proven.
    pub leaf: Fr,
    /// Sibling hashes from the leaf level up to (but not including) the root.
    /// `siblings[0]` is the leaf's sibling; `siblings[depth-1]` is the sibling
    /// just below the root.
    pub siblings: Vec<Fr>,
    /// Direction bits. `index_bits[i] == true` means the current node is the
    /// RIGHT child at level `i` (so the sibling sits on the left).
    pub index_bits: Vec<bool>,
}

/// A minimal in-the-clear Merkle tree over `Fr` leaves using the same Poseidon
/// 2-to-1 hash as the circuit. Depth is `log2(num_leaves)`; `num_leaves` must
/// be a power of two.
#[derive(Clone)]
pub struct MerkleTree {
    params: PoseidonConfig<Fr>,
    /// `layers[0]` = leaves, `layers[depth]` = [root].
    layers: Vec<Vec<Fr>>,
}

impl MerkleTree {
    /// Build a tree from `leaves` (length must be a power of two).
    pub fn new(params: PoseidonConfig<Fr>, leaves: Vec<Fr>) -> Self {
        assert!(
            leaves.len().is_power_of_two() && !leaves.is_empty(),
            "leaf count must be a non-zero power of two"
        );
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut next = Vec::with_capacity(prev.len() / 2);
            for pair in prev.chunks(2) {
                next.push(hash_two(&params, pair[0], pair[1]));
            }
            layers.push(next);
        }
        Self { params, layers }
    }

    /// Depth (number of hash levels from leaf to root).
    pub fn depth(&self) -> usize {
        self.layers.len() - 1
    }

    /// The Merkle root.
    pub fn root(&self) -> Fr {
        self.layers.last().unwrap()[0]
    }

    /// Produce the authentication path for the leaf at `index`.
    pub fn path(&self, mut index: usize) -> MerklePath {
        let leaf = self.layers[0][index];
        let mut siblings = Vec::with_capacity(self.depth());
        let mut index_bits = Vec::with_capacity(self.depth());
        for level in 0..self.depth() {
            let is_right = index & 1 == 1;
            let sibling_index = if is_right { index - 1 } else { index + 1 };
            siblings.push(self.layers[level][sibling_index]);
            index_bits.push(is_right);
            index >>= 1;
        }
        MerklePath {
            leaf,
            siblings,
            index_bits,
        }
    }
}

/// The Merkle-membership circuit.
///
/// The public input is the root; everything else is a private witness.
#[derive(Clone)]
pub struct MerkleMembershipCircuit {
    /// Poseidon parameters (constant, baked into the constraints).
    pub params: PoseidonConfig<Fr>,
    /// PUBLIC: the Merkle root the prover claims membership under.
    pub root: Option<Fr>,
    /// PRIVATE: the leaf value.
    pub leaf: Option<Fr>,
    /// PRIVATE: sibling hashes, leaf level first.
    pub siblings: Option<Vec<Fr>>,
    /// PRIVATE: direction bits (`true` = current node is right child).
    pub index_bits: Option<Vec<bool>>,
    /// Fixed tree depth this circuit is compiled for.
    pub depth: usize,
}

impl MerkleMembershipCircuit {
    /// Build a fully-populated circuit instance (for proving) from a path.
    pub fn from_path(params: PoseidonConfig<Fr>, root: Fr, path: &MerklePath) -> Self {
        Self {
            depth: path.siblings.len(),
            params,
            root: Some(root),
            leaf: Some(path.leaf),
            siblings: Some(path.siblings.clone()),
            index_bits: Some(path.index_bits.clone()),
        }
    }

    /// Build an empty circuit of the given depth (for trusted setup).
    pub fn empty(params: PoseidonConfig<Fr>, depth: usize) -> Self {
        Self {
            depth,
            params,
            root: None,
            leaf: None,
            siblings: None,
            index_bits: None,
        }
    }
}

impl ConstraintSynthesizer<Fr> for MerkleMembershipCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        // Poseidon params as an in-circuit constant.
        let params_var = CRHParametersVar::<Fr>::new_constant(cs.clone(), self.params.clone())?;

        // PUBLIC input: the claimed root.
        let root_var = FpVar::<Fr>::new_input(cs.clone(), || {
            self.root.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // PRIVATE witness: leaf.
        let mut cur = FpVar::<Fr>::new_witness(cs.clone(), || {
            self.leaf.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // Fold up the path, one level per sibling.
        for level in 0..self.depth {
            // PRIVATE witness: sibling hash at this level.
            let sibling = FpVar::<Fr>::new_witness(cs.clone(), || {
                let sibs = self.siblings.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(sibs[level])
            })?;

            // PRIVATE witness: direction bit (true => cur is the right child).
            let is_right = Boolean::new_witness(cs.clone(), || {
                let bits = self.index_bits.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
                Ok(bits[level])
            })?;

            // Order the pair by the direction bit:
            //   if is_right: (left, right) = (sibling, cur)
            //   else:        (left, right) = (cur, sibling)
            // `conditionally_select(cond, a, b)` returns a when cond is true.
            let left = is_right.select(&sibling, &cur)?;
            let right = is_right.select(&cur, &sibling)?;

            // In-circuit 2-to-1 Poseidon compression.
            cur = <TwoToOneCRHGadget<Fr> as TwoToOneCRHSchemeGadget<TwoToOneCRH<Fr>, Fr>>::compress(
                &params_var,
                &left,
                &right,
            )?;
        }

        // Enforce the computed root equals the public root.
        cur.enforce_equal(&root_var)?;
        Ok(())
    }
}

/// Trusted setup: produce proving + prepared verifying keys for a fixed depth.
pub fn setup(depth: usize) -> Result<(ProvingKey<Bn254>, PreparedVerifyingKey<Bn254>), String> {
    let params = poseidon_config();
    let circuit = MerkleMembershipCircuit::empty(params, depth);
    let mut rng = thread_rng();
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng)
        .map_err(|e| format!("merkle setup failed: {}", e))?;
    let pvk = Groth16::<Bn254>::process_vk(&vk).map_err(|e| format!("process_vk failed: {}", e))?;
    Ok((pk, pvk))
}

/// Prove membership of `path.leaf` under `root`. Returns the proof and the
/// public inputs (just the root) needed for verification.
pub fn prove(
    pk: &ProvingKey<Bn254>,
    root: Fr,
    path: &MerklePath,
) -> Result<(ark_groth16::Proof<Bn254>, Vec<Fr>), String> {
    use ark_relations::r1cs::{ConstraintSystem, OptimizationGoal};
    let params = poseidon_config();
    let circuit = MerkleMembershipCircuit::from_path(params, root, path);

    // Honesty guard: if the witness does not actually satisfy the circuit
    // (e.g. a forged leaf or corrupted path that does not hash to `root`),
    // return an error instead of letting `Groth16::prove` hit its internal
    // `debug_assert!(cs.is_satisfied())` and panic. A prover that does not know
    // a valid membership witness simply cannot produce a proof — this is the
    // soundness property, surfaced as a clean `Err`.
    {
        let cs = ConstraintSystem::<Fr>::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Constraints);
        circuit
            .clone()
            .generate_constraints(cs.clone())
            .map_err(|e| format!("constraint synthesis failed: {}", e))?;
        if !cs.is_satisfied().map_err(|e| format!("is_satisfied failed: {}", e))? {
            return Err("witness does not satisfy Merkle-membership constraints \
                        (leaf/path does not hash to the claimed root)"
                .to_string());
        }
    }

    let mut rng = thread_rng();
    let proof = Groth16::<Bn254>::prove(pk, circuit, &mut rng)
        .map_err(|e| format!("merkle prove failed: {}", e))?;
    Ok((proof, vec![root]))
}

/// Verify a membership proof against the public root.
pub fn verify(
    pvk: &PreparedVerifyingKey<Bn254>,
    proof: &ark_groth16::Proof<Bn254>,
    public_inputs: &[Fr],
) -> Result<bool, String> {
    Groth16::<Bn254>::verify_with_processed_vk(pvk, public_inputs, proof)
        .map_err(|e| format!("merkle verify failed: {}", e))
}

// ---------------------------------------------------------------------------
// Serialization helpers (used by the FFI layer and by any Python caller).
// ---------------------------------------------------------------------------

/// Serialize an `Fr` (e.g. the Merkle root or a leaf) to compressed bytes.
pub fn fr_to_bytes(x: &Fr) -> Vec<u8> {
    use ark_serialize::CanonicalSerialize;
    let mut buf = Vec::new();
    x.serialize_compressed(&mut buf).expect("Fr serialize");
    buf
}

/// Parse an `Fr` from compressed bytes.
pub fn fr_from_bytes(b: &[u8]) -> Result<Fr, String> {
    use ark_serialize::CanonicalDeserialize;
    Fr::deserialize_compressed(b).map_err(|e| format!("Fr deserialize: {}", e))
}

/// Serialize a Groth16 proof to compressed bytes.
pub fn proof_to_bytes(p: &ark_groth16::Proof<Bn254>) -> Vec<u8> {
    use ark_serialize::CanonicalSerialize;
    let mut buf = Vec::new();
    p.serialize_compressed(&mut buf).expect("proof serialize");
    buf
}

// ===========================================================================
// FFI SKETCH (ZK-RAG Phase 1, step 4) — how "In My Head" (Python) calls this.
//
// STATUS: documented skeleton. `merkle_ffi_verify` below is fully implemented
// and testable; the prove/setup entry points are intentionally left as design
// notes because they require a decision on key storage/marshalling (see NOTE).
//
// C ABI shape (mirrors src/ffi.rs `#[no_mangle] extern "C"` + error-code style):
//
//   // Python (ctypes) — pseudocode of the intended integration:
//   //   lib = ctypes.CDLL("libnexuszero_crypto.so")
//   //   ok = lib.nexuszero_merkle_verify(
//   //            vk_ptr, vk_len,        # prepared verifying key bytes
//   //            root_ptr, root_len,    # 32-byte compressed Fr (public input)
//   //            proof_ptr, proof_len)  # compressed Groth16 proof bytes
//   //   # ok == 1 -> membership proven; 0 -> rejected; <0 -> error code
//
// The In My Head flow:
//   1. Rust side (this crate) runs `setup(depth)` ONCE per vault, persists the
//      proving key (private, kept by the vault owner) and the prepared verifying
//      key (publishable).
//   2. To prove a citation is in the vault: the vault owner (who holds the
//      leaves) builds the `MerklePath` for the cited chunk and calls `prove()`.
//      Only the root + proof cross the trust boundary — the vault stays secret.
//   3. Any verifier (a peer, an auditor, the Python app) calls
//      `nexuszero_merkle_verify` with (vk, root, proof). No vault data needed.
//
// NOTE (Phase-2 work): a production FFI must decide how the proving key and the
// witness path are marshalled across the boundary. Two clean options:
//   (a) keep prove() entirely Rust-side and expose only verify over FFI (proofs
//       are generated by a Rust binary/service the vault owner runs), or
//   (b) pass the leaf + siblings + direction bits as a packed byte buffer and
//       the proving key as bytes, and reconstruct `MerklePath` in Rust.
// Option (a) is recommended: it keeps private leaves out of the FFI surface,
// consistent with the crate's security rule "private material stays in Rust".
// ===========================================================================

/// Error/So return codes for the Merkle FFI, matching src/ffi.rs conventions.
pub const MERKLE_FFI_REJECTED: i32 = 0;
pub const MERKLE_FFI_ACCEPTED: i32 = 1;
pub const MERKLE_FFI_ERR_NULL: i32 = -3;
pub const MERKLE_FFI_ERR_DESERIALIZE: i32 = -4;
pub const MERKLE_FFI_ERR_INTERNAL: i32 = -2;

/// Verify a Merkle-membership proof from raw bytes (C ABI).
///
/// Returns `MERKLE_FFI_ACCEPTED` (1) if the proof verifies, `MERKLE_FFI_REJECTED`
/// (0) if it does not, or a negative error code on bad input. This is the one
/// FFI entry point safe to expose broadly: it touches no private vault data.
///
/// # Safety
/// All pointers must be valid for the given lengths for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn nexuszero_merkle_verify(
    vk_ptr: *const u8,
    vk_len: usize,
    root_ptr: *const u8,
    root_len: usize,
    proof_ptr: *const u8,
    proof_len: usize,
) -> i32 {
    use ark_serialize::CanonicalDeserialize;

    if vk_ptr.is_null() || root_ptr.is_null() || proof_ptr.is_null() {
        return MERKLE_FFI_ERR_NULL;
    }
    let vk_bytes = std::slice::from_raw_parts(vk_ptr, vk_len);
    let root_bytes = std::slice::from_raw_parts(root_ptr, root_len);
    let proof_bytes = std::slice::from_raw_parts(proof_ptr, proof_len);

    let pvk = match PreparedVerifyingKey::<Bn254>::deserialize_compressed(vk_bytes) {
        Ok(v) => v,
        Err(_) => return MERKLE_FFI_ERR_DESERIALIZE,
    };
    let root = match fr_from_bytes(root_bytes) {
        Ok(r) => r,
        Err(_) => return MERKLE_FFI_ERR_DESERIALIZE,
    };
    let proof = match ark_groth16::Proof::<Bn254>::deserialize_compressed(proof_bytes) {
        Ok(p) => p,
        Err(_) => return MERKLE_FFI_ERR_DESERIALIZE,
    };

    match verify(&pvk, &proof, &[root]) {
        Ok(true) => MERKLE_FFI_ACCEPTED,
        Ok(false) => MERKLE_FFI_REJECTED,
        Err(_) => MERKLE_FFI_ERR_INTERNAL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_r1cs_std::R1CSVar;
    use ark_relations::r1cs::ConstraintSystem;

    /// The native (out-of-circuit) 2-to-1 Poseidon used to build the cleartext
    /// tree MUST equal the in-circuit gadget hash, or the circuit could never be
    /// satisfied by a real path. This guards that invariant.
    #[test]
    fn test_native_matches_gadget_hash() {
        let params = poseidon_config();
        let mut rng = thread_rng();
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);
        let native = hash_two(&params, a, b);

        let cs = ConstraintSystem::<Fr>::new_ref();
        let params_var = CRHParametersVar::<Fr>::new_constant(cs.clone(), params.clone()).unwrap();
        let a_var = FpVar::<Fr>::new_witness(cs.clone(), || Ok(a)).unwrap();
        let b_var = FpVar::<Fr>::new_witness(cs.clone(), || Ok(b)).unwrap();
        let out = <TwoToOneCRHGadget<Fr> as TwoToOneCRHSchemeGadget<TwoToOneCRH<Fr>, Fr>>::compress(
            &params_var, &a_var, &b_var,
        )
        .unwrap();
        println!("native  = {}", native);
        println!("gadget  = {}", out.value().unwrap());
        assert_eq!(native, out.value().unwrap(), "native and gadget hash disagree");
        assert!(cs.is_satisfied().unwrap());
    }

    /// The in-circuit constraint system must be SATISFIED by a valid path and
    /// UNSATISFIED by a forged one. This is the soundness property at the R1CS
    /// level, independent of the SNARK.
    #[test]
    fn test_circuit_accepts_valid_rejects_forged() {
        use ark_relations::r1cs::OptimizationGoal;
        let params = poseidon_config();
        let depth = DEFAULT_DEPTH;
        let num_leaves = 1usize << depth;
        let mut rng = thread_rng();
        let leaves: Vec<Fr> = (0..num_leaves).map(|_| Fr::rand(&mut rng)).collect();
        let tree = MerkleTree::new(params.clone(), leaves);
        let root = tree.root();

        // Valid path -> satisfied.
        let good = tree.path(42);
        let cs = ConstraintSystem::<Fr>::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Constraints);
        MerkleMembershipCircuit::from_path(params.clone(), root, &good)
            .generate_constraints(cs.clone())
            .unwrap();
        assert!(cs.is_satisfied().unwrap(), "valid path must satisfy the circuit");
        let nc = cs.num_constraints();

        // Forged leaf -> unsatisfied against the real root.
        let mut forged = tree.path(42);
        forged.leaf += Fr::from(999u64);
        let cs2 = ConstraintSystem::<Fr>::new_ref();
        cs2.set_optimization_goal(OptimizationGoal::Constraints);
        MerkleMembershipCircuit::from_path(params, root, &forged)
            .generate_constraints(cs2.clone())
            .unwrap();
        assert!(
            !cs2.is_satisfied().unwrap(),
            "forged leaf must NOT satisfy the circuit against the real root"
        );
        println!(
            "depth-{} circuit: {} constraints; valid=satisfied, forged=unsatisfied",
            depth, nc
        );
    }

    /// Build a 256-leaf (depth-8) tree, prove membership of a real leaf,
    /// and check that a valid path verifies TRUE while tampered
    /// leaf / path / root all verify FALSE.
    #[test]
    fn test_merkle_membership_prove_and_verify() {
        let params = poseidon_config();
        let depth = DEFAULT_DEPTH; // 8 => 256 leaves
        let num_leaves = 1usize << depth;

        // Deterministic-but-varied leaves.
        let mut rng = thread_rng();
        let leaves: Vec<Fr> = (0..num_leaves).map(|_| Fr::rand(&mut rng)).collect();
        let tree = MerkleTree::new(params.clone(), leaves.clone());
        let root = tree.root();

        // One trusted setup reused for all proofs at this depth.
        let (pk, pvk) = setup(depth).expect("setup should succeed");

        // ---- Valid membership: leaf at index 42 ----
        let index = 42usize;
        let path = tree.path(index);
        assert_eq!(path.leaf, leaves[index]);
        // Sanity: the cleartext path re-hashes to the root.
        {
            let mut cur = path.leaf;
            for (lvl, sib) in path.siblings.iter().enumerate() {
                cur = if path.index_bits[lvl] {
                    hash_two(&params, *sib, cur)
                } else {
                    hash_two(&params, cur, *sib)
                };
            }
            assert_eq!(cur, root, "cleartext path must reproduce the root");
        }

        let (proof, public_inputs) = prove(&pk, root, &path).expect("prove should succeed");
        assert!(
            verify(&pvk, &proof, &public_inputs).expect("verify should not error"),
            "valid membership proof must verify TRUE"
        );

        // ---- Tamper case 1: wrong public root ----
        let bad_root = root + Fr::from(1u64);
        assert!(
            !verify(&pvk, &proof, &[bad_root]).expect("verify should not error"),
            "proof must NOT verify against a tampered root"
        );

        // ---- Tamper case 2: forged leaf not in the tree ----
        // A leaf value that is not the committed one cannot hash to the real
        // root, so no valid witness exists: prove() must return Err. This IS the
        // soundness property — a prover without a real membership witness cannot
        // produce ANY proof that verifies.
        let mut forged = path.clone();
        forged.leaf += Fr::from(12345u64);
        let forged_res = prove(&pk, root, &forged);
        assert!(
            forged_res.is_err(),
            "proving a forged leaf against the real root must FAIL (no valid witness)"
        );

        // ---- Tamper case 3: corrupted sibling path ----
        let mut bad_path = tree.path(index);
        bad_path.siblings[0] += Fr::from(7u64);
        let bad_path_res = prove(&pk, root, &bad_path);
        assert!(
            bad_path_res.is_err(),
            "proving a corrupted sibling path against the real root must FAIL"
        );

        // ---- Tamper case 4: flipped direction bit ----
        // Corrupting the path index (direction) also breaks the fold.
        let mut bad_dir = tree.path(index);
        bad_dir.index_bits[0] = !bad_dir.index_bits[0];
        assert!(
            prove(&pk, root, &bad_dir).is_err(),
            "proving with a flipped direction bit must FAIL"
        );

        // ---- Tamper case 5: valid proof, but tampered proof bytes ----
        // Mutate the serialized proof and confirm verification is FALSE.
        {
            use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
            let mut bytes = Vec::new();
            proof.serialize_compressed(&mut bytes).unwrap();
            bytes[0] ^= 0x01; // flip a bit
            if let Ok(tampered) =
                ark_groth16::Proof::<Bn254>::deserialize_compressed(&bytes[..])
            {
                assert!(
                    !verify(&pvk, &tampered, &public_inputs).unwrap_or(false),
                    "a tampered proof must NOT verify"
                );
            }
            // If deserialization fails (invalid point), that also means the
            // tampered proof is unusable — equally acceptable.
        }

        // ---- Positive control at a different index (left-child leaf) ----
        let index2 = 7usize;
        let path2 = tree.path(index2);
        let (proof2, pub2) = prove(&pk, root, &path2).expect("prove should succeed");
        assert!(
            verify(&pvk, &proof2, &pub2).expect("verify should not error"),
            "second valid membership proof must verify TRUE"
        );
    }

    /// Exercise the C-ABI verify entry point through raw byte buffers, exactly
    /// as the Python "In My Head" side would via ctypes. Confirms the byte
    /// marshalling round-trips and that accept/reject codes are correct.
    #[test]
    fn test_ffi_verify_roundtrip() {
        use ark_serialize::CanonicalSerialize;

        let params = poseidon_config();
        let depth = 4usize; // small tree keeps this test fast
        let num_leaves = 1usize << depth;
        let mut rng = thread_rng();
        let leaves: Vec<Fr> = (0..num_leaves).map(|_| Fr::rand(&mut rng)).collect();
        let tree = MerkleTree::new(params, leaves);
        let root = tree.root();
        let (pk, pvk) = setup(depth).expect("setup");
        let path = tree.path(3);
        let (proof, _pub) = prove(&pk, root, &path).expect("prove");

        // Marshal to bytes (as the FFI caller would receive/pass).
        let mut vk_bytes = Vec::new();
        pvk.serialize_compressed(&mut vk_bytes).unwrap();
        let root_bytes = fr_to_bytes(&root);
        let proof_bytes = proof_to_bytes(&proof);

        // Valid -> ACCEPTED.
        let rc = unsafe {
            nexuszero_merkle_verify(
                vk_bytes.as_ptr(),
                vk_bytes.len(),
                root_bytes.as_ptr(),
                root_bytes.len(),
                proof_bytes.as_ptr(),
                proof_bytes.len(),
            )
        };
        assert_eq!(rc, MERKLE_FFI_ACCEPTED, "FFI must accept a valid proof");

        // Wrong root -> REJECTED.
        let bad_root_bytes = fr_to_bytes(&(root + Fr::from(1u64)));
        let rc_bad = unsafe {
            nexuszero_merkle_verify(
                vk_bytes.as_ptr(),
                vk_bytes.len(),
                bad_root_bytes.as_ptr(),
                bad_root_bytes.len(),
                proof_bytes.as_ptr(),
                proof_bytes.len(),
            )
        };
        assert_eq!(rc_bad, MERKLE_FFI_REJECTED, "FFI must reject a wrong root");

        // Null pointer -> error code.
        let rc_null = unsafe {
            nexuszero_merkle_verify(
                std::ptr::null(),
                0,
                root_bytes.as_ptr(),
                root_bytes.len(),
                proof_bytes.as_ptr(),
                proof_bytes.len(),
            )
        };
        assert_eq!(rc_null, MERKLE_FFI_ERR_NULL, "FFI must flag null pointers");
    }
}
