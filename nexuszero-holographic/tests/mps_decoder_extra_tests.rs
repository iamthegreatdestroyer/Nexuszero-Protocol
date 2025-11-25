use nexuszero_holographic::MPS;
use nexuszero_holographic::compression::{encoder, decoder};
use std::fs::{File, remove_file};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(suffix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nexuszero_test_{}{}", nanos, suffix));
    p
}

#[test]
fn test_encode_proof_lossless_empty_error() {
    let data: Vec<u8> = Vec::new();
    let res = encoder::encode_proof_lossless(&data, 8);
    assert!(res.is_err());
}

#[test]
fn test_set_site_onehot_sets_tensor_correctly() {
    // Create small MPS and set a single site to a one-hot vector
    let mut mps = MPS::new(3, 256, 2);
    let idx = 1usize;
    let p_idx = 42usize;
    mps.set_site_onehot(idx, p_idx);
    let t = mps.site_tensor(idx).expect("site tensor present");
    let shape = t.shape();
    for l in 0..shape[0] {
        for r in 0..shape[2] {
            for p in 0..mps.physical_dim() {
                let val = t[[l,p,r]];
                if p == p_idx {
                    assert!((val - 1.0).abs() < 1e-8, "slot should be 1.0");
                } else {
                    assert!((val - 0.0).abs() < 1e-8, "other slots should be 0.0");
                }
            }
        }
    }
}

#[test]
fn test_decode_fallback_argmax_random_mps() {
    // Create a randomly initialized MPS with pd=256 and ensure decode returns some value
    let mps = MPS::new(1, 256, 1);
    let decoded = decoder::decode_proof(&mps);
    assert_eq!(decoded.len(), 1);
}

#[test]
fn test_cli_encode_path_non_lossless_writes_file() {
    use nexuszero_holographic::cli::encode_path;
    // Setup input and output
    let input_path = unique_temp_path("_mps_cli_in.bin");
    let out_path = unique_temp_path("_mps_cli_out.mps");
    let mut f = File::create(&input_path).unwrap();
    f.write_all(&[10u8; 32]).unwrap();
    f.flush().unwrap();
    let res = encode_path(input_path.to_str().unwrap(), 8, Some(out_path.to_str().unwrap()), false, false);
    assert!(res.is_ok());
    assert!(out_path.exists());
    let _ = remove_file(input_path);
    let _ = remove_file(out_path);
}
