//! `simrag` — local ZK similarity-threshold prover/verifier (ZK-RAG Tier-B).
//!
//! The vault owner runs this to prove a cited chunk's embedding is >= tau cosine
//! similar to a query embedding, in zero knowledge. Private embeddings never
//! leave the process; only the public root/leaf/query-commit/threshold + the
//! proof are emitted. Rust owns normalization + quantization so the numbers are
//! canonical on both sides.
//!
//!   simrag setup  --depth D --pk F --vk F
//!   simrag prove  --depth D --pk F --vectors F --index I --query F --tau T \
//!                 --root-out F --leaf-out F --qc-out F --thr-out F --proof-out F
//!   simrag verify --vk F --root F --leaf F --query F --tau T --proof F
//!
//! `--vectors F`: corpus, one embedding per line (comma/space-separated floats,
//! each length SIM_DIM). `--query F`: one embedding (first non-empty line).
//! verify RE-DERIVES the query commitment + threshold from the public query and
//! tau, so a proof is bound to that exact query and cutoff.

use std::path::PathBuf;
use std::process::exit;

use ark_bn254::{Bn254, Fr};
use ark_serialize::CanonicalDeserialize;

use nexuszero_crypto::proof::merkle_circuit::{poseidon_config, proof_to_bytes, MerkleTree};
use nexuszero_crypto::proof::similarity_circuit::{
    commit_vector, fr_from_i64, prove, public_inputs, quantize_unit, setup, threshold_fr, verify,
    SIM_DIM,
};
use ark_serialize::CanonicalSerialize;
use nexuszero_crypto::proof::zkrag::{
    load_prepared_vk, load_proving_key, read_fr, save_prepared_vk, save_proving_key, write_fr,
};

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

fn parse_vec(line: &str) -> Vec<f64> {
    line.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f64>().unwrap_or_else(|_| die(format!("bad float {s:?}"))))
        .collect()
}

fn read_vectors(path: &str) -> Vec<Vec<f64>> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(format!("read {path}: {e}")));
    let mut out = vec![];
    for line in text.lines() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        let v = parse_vec(l);
        if v.len() != SIM_DIM {
            die(format!("vector length {} != SIM_DIM {}", v.len(), SIM_DIM));
        }
        out.push(v);
    }
    out
}

fn read_one_vector(path: &str) -> Vec<f64> {
    read_vectors(path)
        .into_iter()
        .next()
        .unwrap_or_else(|| die("query file has no vector".into()))
}

/// Pad the embedding leaves to exactly 2^depth with a deterministic zero-vector
/// leaf, matching the tree the keys were made for.
fn build_tree_fixed(embeddings: &[Vec<f64>], depth: usize) -> (MerkleTree, Fr) {
    let params = poseidon_config();
    let cap = 1usize << depth;
    if embeddings.is_empty() {
        die("no vectors".into());
    }
    if embeddings.len() > cap {
        die(format!("{} vectors exceed depth-{depth} capacity {cap}", embeddings.len()));
    }
    let mut leaves: Vec<Fr> = embeddings
        .iter()
        .map(|e| commit_vector(&params, &quantize_unit(e)))
        .collect();
    let pad = commit_vector(&params, &quantize_unit(&vec![0.0; SIM_DIM]));
    while leaves.len() < cap {
        leaves.push(pad);
    }
    let tree = MerkleTree::new(params, leaves);
    let root = tree.root();
    (tree, root)
}

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "setup" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth int".into()));
            let (pk, pvk) = setup(depth).unwrap_or_else(|e| die(format!("setup: {e}")));
            save_proving_key(&pk, &PathBuf::from(need(&["--pk"]))).unwrap_or_else(|e| die(e));
            save_prepared_vk(&pvk, &PathBuf::from(need(&["--vk"]))).unwrap_or_else(|e| die(e));
            println!("OK simrag setup depth={depth} dim={SIM_DIM}");
        }
        "prove" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth int".into()));
            let pk = load_proving_key(&PathBuf::from(need(&["--pk"]))).unwrap_or_else(|e| die(e));
            let embeddings = read_vectors(&need(&["--vectors"]));
            let index: usize = need(&["--index"]).parse().unwrap_or_else(|_| die("--index int".into()));
            let tau: f64 = need(&["--tau"]).parse().unwrap_or_else(|_| die("--tau float".into()));
            let query = read_one_vector(&need(&["--query"]));
            if query.len() != SIM_DIM {
                die(format!("query length {} != SIM_DIM {}", query.len(), SIM_DIM));
            }
            if index >= embeddings.len() {
                die(format!("index {index} out of range ({} vectors)", embeddings.len()));
            }
            let params = poseidon_config();
            let (tree, root) = build_tree_fixed(&embeddings, depth);
            let path = tree.path(index);
            let leaf = path.leaf;
            let qa = quantize_unit(&embeddings[index]);
            let query_q = quantize_unit(&query);
            let qc = commit_vector(&params, &query_q);
            let thr = threshold_fr(tau);
            let (proof, _pubs) = prove(&pk, root, leaf, qc, thr, qa, query_q, &path)
                .unwrap_or_else(|e| die(format!("prove: {e}")));
            write_fr(&PathBuf::from(need(&["--root-out"])), &root).unwrap_or_else(|e| die(e));
            write_fr(&PathBuf::from(need(&["--leaf-out"])), &leaf).unwrap_or_else(|e| die(e));
            write_fr(&PathBuf::from(need(&["--qc-out"])), &qc).unwrap_or_else(|e| die(e));
            write_fr(&PathBuf::from(need(&["--thr-out"])), &thr).unwrap_or_else(|e| die(e));
            std::fs::write(need(&["--proof-out"]), proof_to_bytes(&proof))
                .unwrap_or_else(|e| die(format!("write proof: {e}")));
            println!("OK simrag prove index={index} depth={depth} tau={tau}");
        }
        "verify" => {
            let pvk = load_prepared_vk(&PathBuf::from(need(&["--vk"]))).unwrap_or_else(|e| die(e));
            let root = read_fr(&PathBuf::from(need(&["--root"]))).unwrap_or_else(|e| die(e));
            let leaf = read_fr(&PathBuf::from(need(&["--leaf"]))).unwrap_or_else(|e| die(e));
            let tau: f64 = need(&["--tau"]).parse().unwrap_or_else(|_| die("--tau float".into()));
            let query = read_one_vector(&need(&["--query"]));
            if query.len() != SIM_DIM {
                die(format!("query length {} != SIM_DIM {}", query.len(), SIM_DIM));
            }
            // Re-derive the query commitment + threshold from the PUBLIC query and
            // tau, so the proof is bound to exactly this query and cutoff.
            let params = poseidon_config();
            let qc = commit_vector(&params, &quantize_unit(&query));
            let thr = threshold_fr(tau);
            let proof_bytes = std::fs::read(need(&["--proof"])).unwrap_or_else(|e| die(format!("read proof: {e}")));
            let proof = ark_groth16::Proof::<Bn254>::deserialize_compressed(&proof_bytes[..])
                .unwrap_or_else(|e| die(format!("deserialize proof: {e}")));
            match verify(&pvk, &proof, &public_inputs(root, leaf, qc, thr)) {
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
        "leaves" => {
            let depth: usize = need(&["--depth"]).parse().unwrap_or_else(|_| die("--depth int".into()));
            let embeddings = read_vectors(&need(&["--vectors"]));
            let (_tree, root) = build_tree_fixed(&embeddings, depth);
            write_fr(&PathBuf::from(need(&["--root-out"])), &root).unwrap_or_else(|e| die(e));
            let params = nexuszero_crypto::proof::merkle_circuit::poseidon_config();
            let leaves_out = need(&["--leaves-out"]);
            let mut buf = String::new();
            for e in &embeddings {
                let leaf = commit_vector(&params, &quantize_unit(e));
                let mut b = Vec::new();
                leaf.serialize_compressed(&mut b).expect("serialize leaf");
                for byte in &b {
                    buf.push_str(&format!("{:02x}", byte));
                }
                buf.push('\n');
            }
            std::fs::write(&leaves_out, buf).unwrap_or_else(|e| die(format!("write leaves: {e}")));
            println!("OK simrag leaves depth={depth} n={} dim={SIM_DIM}", embeddings.len());
            let _ = fr_from_i64; // silence unused-import if not exercised elsewhere
        }
        other => {
            eprintln!("usage: simrag <setup|prove|verify|leaves> ...  (got {other:?})");
            exit(2);
        }
    }
}
