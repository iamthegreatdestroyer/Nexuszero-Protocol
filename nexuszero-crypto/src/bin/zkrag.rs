//! `zkrag` — the local ZK-RAG prover/verifier CLI for "In My Head".
//!
//! The vault owner runs this on their own machine; private leaves never leave
//! the process — only the public Merkle root and the Groth16 proof are written
//! out. This is the prove-side "Rust binary the vault owner runs" recommended in
//! `merkle_circuit.rs` (option a): keep private material in Rust, expose proofs.
//!
//! Subcommands:
//!   zkrag setup  --depth D --pk FILE --vk FILE
//!   zkrag prove  --depth D --pk FILE --leaves FILE --index I --root-out FILE --proof-out FILE
//!   zkrag verify --vk FILE --root FILE --proof FILE
//!
//! `--leaves FILE`: one leaf per line; each line's raw bytes are the leaf
//! preimage (Python passes chunk content-id strings, one per committed chunk).
//! `verify` prints ACCEPTED (exit 0) / REJECTED (exit 3); other errors exit 1/2.

use std::path::PathBuf;
use std::process::exit;

use ark_bn254::Bn254;
use ark_serialize::CanonicalDeserialize;

use nexuszero_crypto::proof::merkle_circuit::{proof_to_bytes, prove, setup, verify};
use nexuszero_crypto::proof::zkrag::{
    build_tree_fixed, leaf_from_bytes, load_prepared_vk, load_proving_key, read_fr,
    save_prepared_vk, save_proving_key, write_fr,
};

/// Return the value following any of `flags` on the command line, if present.
fn arg(flags: &[&str]) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    for i in 0..a.len() {
        if flags.contains(&a[i].as_str()) && i + 1 < a.len() {
            return Some(a[i + 1].clone());
        }
    }
    None
}

fn need(flags: &[&str]) -> String {
    arg(flags).unwrap_or_else(|| {
        eprintln!("missing required argument {flags:?}");
        exit(2);
    })
}

fn die(msg: String) -> ! {
    eprintln!("{msg}");
    exit(1);
}

fn read_leaves(path: &str) -> Vec<Vec<u8>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(format!("read leaves {path}: {e}")));
    text.lines()
        .map(|l| l.trim_end_matches(['\r', '\n']))
        .filter(|l| !l.is_empty())
        .map(|l| l.as_bytes().to_vec())
        .collect()
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "setup" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth must be an integer".into()));
            let pk_path = PathBuf::from(need(&["--pk"]));
            let vk_path = PathBuf::from(need(&["--vk"]));
            let (pk, pvk) = setup(depth).unwrap_or_else(|e| die(format!("setup: {e}")));
            save_proving_key(&pk, &pk_path).unwrap_or_else(|e| die(e));
            save_prepared_vk(&pvk, &vk_path).unwrap_or_else(|e| die(e));
            println!("OK setup depth={depth} pk={} vk={}", pk_path.display(), vk_path.display());
        }
        "prove" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth must be an integer".into()));
            let pk_path = PathBuf::from(need(&["--pk"]));
            let leaves = read_leaves(&need(&["--leaves"]));
            let index: usize = need(&["--index"]).parse().unwrap_or_else(|_| die("--index must be an integer".into()));
            let root_out = PathBuf::from(need(&["--root-out"]));
            let proof_out = PathBuf::from(need(&["--proof-out"]));
            if index >= leaves.len() {
                die(format!("index {index} out of range ({} leaves)", leaves.len()));
            }
            let (tree, root) = build_tree_fixed(&leaves, depth).unwrap_or_else(|e| die(e));
            let pk = load_proving_key(&pk_path).unwrap_or_else(|e| die(e));
            let path = tree.path(index);
            let (proof, _pub) = prove(&pk, root, &path).unwrap_or_else(|e| die(format!("prove: {e}")));
            write_fr(&root_out, &root).unwrap_or_else(|e| die(e));
            if let Some(leaf_out) = arg(&["--leaf-out"]) {
                write_fr(&PathBuf::from(leaf_out), &path.leaf).unwrap_or_else(|e| die(e));
            }
            std::fs::write(&proof_out, proof_to_bytes(&proof))
                .unwrap_or_else(|e| die(format!("write proof: {e}")));
            println!("OK prove index={index} depth={depth} root={} proof={}", root_out.display(), proof_out.display());
        }
        "verify" => {
            let vk_path = PathBuf::from(need(&["--vk"]));
            let root = read_fr(&PathBuf::from(need(&["--root"]))).unwrap_or_else(|e| die(e));
            // The public leaf binds the proof to a specific disclosed citation.
            // Accept the raw chunk-id string (--leaf-input, derived the same way
            // the tree was built) or a precomputed Fr file (--leaf).
            let leaf = if let Some(s) = arg(&["--leaf-input"]) {
                leaf_from_bytes(s.as_bytes())
            } else {
                read_fr(&PathBuf::from(need(&["--leaf"]))).unwrap_or_else(|e| die(e))
            };
            let proof_bytes = std::fs::read(need(&["--proof"])).unwrap_or_else(|e| die(format!("read proof: {e}")));
            let pvk = load_prepared_vk(&vk_path).unwrap_or_else(|e| die(e));
            let proof = ark_groth16::Proof::<Bn254>::deserialize_compressed(&proof_bytes[..])
                .unwrap_or_else(|e| die(format!("deserialize proof: {e}")));
            match verify(&pvk, &proof, &[root, leaf]) {
                Ok(true) => {
                    println!("ACCEPTED");
                    exit(0);
                }
                Ok(false) => {
                    println!("REJECTED");
                    exit(3);
                }
                Err(e) => die(format!("verify: {e}")),
            }
        }
        "root" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth must be an integer".into()));
            let leaves = read_leaves(&need(&["--leaves"]));
            let root_out = PathBuf::from(need(&["--root-out"]));
            let (_tree, root) = build_tree_fixed(&leaves, depth).unwrap_or_else(|e| die(e));
            write_fr(&root_out, &root).unwrap_or_else(|e| die(e));
            println!("OK root depth={depth} root={}", root_out.display());
        }
        "leaf" => {
            let input = need(&["--input"]);
            let out = PathBuf::from(need(&["--out"]));
            write_fr(&out, &leaf_from_bytes(input.as_bytes())).unwrap_or_else(|e| die(e));
            println!("OK leaf out={}", out.display());
        }
        other => {
            eprintln!("usage: zkrag <setup|prove|verify|root|leaf> ...  (got {other:?})");
            exit(2);
        }
    }
}
