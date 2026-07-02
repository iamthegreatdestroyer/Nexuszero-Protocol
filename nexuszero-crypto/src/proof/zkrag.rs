//! ZK-RAG Phase 2: on-disk Groth16 key persistence, deterministic leaf
//! derivation, and a fixed-depth leaves→tree builder.
//!
//! Phase 1 (`merkle_circuit.rs`) delivered the circuit + setup/prove/verify + a
//! tested C-ABI *verify* FFI, but `setup()` regenerates keys on every call and
//! there was no prove-side entry point a caller could actually run. This module
//! closes that gap: keys are persisted once and reloaded, leaves are derived
//! deterministically from arbitrary byte strings, and (together with
//! `src/bin/zkrag.rs`) the Python "In My Head" app can produce real proofs
//! WITHOUT any private vault data crossing the FFI boundary — proofs are made by
//! the local `zkrag` binary and only the public root + proof are emitted. This
//! matches the crate's security rule "private material stays in Rust".

use std::path::Path;

use ark_bn254::{Bn254, Fr};
use ark_ff::PrimeField;
use ark_groth16::{PreparedVerifyingKey, ProvingKey};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::merkle_circuit::{fr_from_bytes, fr_to_bytes, poseidon_config, MerkleTree};

/// Deterministically map arbitrary bytes (a citation chunk id / content hash) to
/// a BN254 field element used as a Merkle leaf: blake3 → reduce mod field order.
/// Rust owns tree construction, so this is the ONLY leaf derivation used, which
/// is why the cleartext tree and the in-circuit tree agree by construction.
pub fn leaf_from_bytes(data: &[u8]) -> Fr {
    let h = blake3::hash(data);
    Fr::from_le_bytes_mod_order(h.as_bytes())
}

/// Domain-separated padding leaf used to fill a tree up to `2^depth`. Distinct
/// from any realistic chunk id, so padding can never be mistaken for membership.
fn pad_leaf() -> Fr {
    leaf_from_bytes(b"\x00ZKRAG_PADDING_LEAF_v1\x00")
}

/// Build a Merkle tree over `items`, padded with `pad_leaf()` to exactly
/// `2^depth` leaves so the tree shape matches the depth the keys were made for.
/// Errors on zero items or more than `2^depth` of them.
pub fn build_tree_fixed(items: &[Vec<u8>], depth: usize) -> Result<(MerkleTree, Fr), String> {
    if depth == 0 {
        return Err("depth must be >= 1".into());
    }
    let cap = 1usize << depth;
    if items.is_empty() {
        return Err("no leaves provided".into());
    }
    if items.len() > cap {
        return Err(format!(
            "{} items exceed depth-{} capacity ({} leaves)",
            items.len(),
            depth,
            cap
        ));
    }
    let params = poseidon_config();
    let mut leaves: Vec<Fr> = items.iter().map(|b| leaf_from_bytes(b)).collect();
    let pad = pad_leaf();
    while leaves.len() < cap {
        leaves.push(pad);
    }
    let tree = MerkleTree::new(params, leaves);
    let root = tree.root();
    Ok((tree, root))
}

// ---- Groth16 key persistence (CanonicalSerialize, compressed) --------------

pub fn save_proving_key(pk: &ProvingKey<Bn254>, path: &Path) -> Result<(), String> {
    let mut buf = Vec::new();
    pk.serialize_compressed(&mut buf)
        .map_err(|e| format!("serialize pk: {e}"))?;
    std::fs::write(path, &buf).map_err(|e| format!("write pk {}: {e}", path.display()))
}

pub fn load_proving_key(path: &Path) -> Result<ProvingKey<Bn254>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read pk {}: {e}", path.display()))?;
    ProvingKey::<Bn254>::deserialize_compressed(&b[..]).map_err(|e| format!("deserialize pk: {e}"))
}

pub fn save_prepared_vk(pvk: &PreparedVerifyingKey<Bn254>, path: &Path) -> Result<(), String> {
    let mut buf = Vec::new();
    pvk.serialize_compressed(&mut buf)
        .map_err(|e| format!("serialize vk: {e}"))?;
    std::fs::write(path, &buf).map_err(|e| format!("write vk {}: {e}", path.display()))
}

pub fn load_prepared_vk(path: &Path) -> Result<PreparedVerifyingKey<Bn254>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read vk {}: {e}", path.display()))?;
    PreparedVerifyingKey::<Bn254>::deserialize_compressed(&b[..])
        .map_err(|e| format!("deserialize vk: {e}"))
}

// ---- Fr <-> 32-byte file helpers (root marshalling for the CLI) ------------

pub fn write_fr(path: &Path, x: &Fr) -> Result<(), String> {
    std::fs::write(path, fr_to_bytes(x)).map_err(|e| format!("write fr {}: {e}", path.display()))
}

pub fn read_fr(path: &Path) -> Result<Fr, String> {
    let b = std::fs::read(path).map_err(|e| format!("read fr {}: {e}", path.display()))?;
    fr_from_bytes(&b)
}

#[cfg(test)]
mod tests {
    use super::super::merkle_circuit::{prove, setup, verify};
    use super::*;

    /// Persist keys, reload them, and prove+verify a real membership with the
    /// RELOADED keys — this is exactly the on-disk lifecycle the CLI uses.
    #[test]
    fn test_zkrag_persisted_keys_roundtrip() {
        let depth = 4usize;
        let items: Vec<Vec<u8>> = (0..10u32)
            .map(|i| format!("chunk-hash-{i}").into_bytes())
            .collect();
        let (tree, root) = build_tree_fixed(&items, depth).expect("build");
        let (pk, pvk) = setup(depth).expect("setup");

        let dir = std::env::temp_dir();
        let pkp = dir.join("zkrag_ut_pk.bin");
        let vkp = dir.join("zkrag_ut_vk.bin");
        save_proving_key(&pk, &pkp).unwrap();
        save_prepared_vk(&pvk, &vkp).unwrap();
        let pk2 = load_proving_key(&pkp).unwrap();
        let pvk2 = load_prepared_vk(&vkp).unwrap();

        let path = tree.path(3);
        let (proof, pub_in) = prove(&pk2, root, &path).expect("prove");
        assert!(
            verify(&pvk2, &proof, &pub_in).unwrap(),
            "proof made with reloaded keys must verify"
        );

        // Determinism: rebuilding the same corpus yields the same root, so a
        // proof is bound to the exact committed set.
        let (_t2, root2) = build_tree_fixed(&items, depth).unwrap();
        assert_eq!(root, root2, "tree build must be deterministic");

        // A different corpus (one chunk changed) yields a different root, so the
        // proof will not verify against it — bound to the committed set.
        let mut items2 = items.clone();
        items2[3] = b"tampered-chunk".to_vec();
        let (_t3, root3) = build_tree_fixed(&items2, depth).unwrap();
        assert_ne!(root, root3, "changing a committed chunk must change the root");
        assert!(
            !verify(&pvk2, &proof, &[root3]).unwrap(),
            "proof must NOT verify against a tampered corpus root"
        );
    }

    #[test]
    fn test_leaf_derivation_is_deterministic_and_distinct() {
        assert_eq!(leaf_from_bytes(b"abc"), leaf_from_bytes(b"abc"));
        assert_ne!(leaf_from_bytes(b"abc"), leaf_from_bytes(b"abd"));
        assert_ne!(leaf_from_bytes(b"abc"), pad_leaf());
    }
}
