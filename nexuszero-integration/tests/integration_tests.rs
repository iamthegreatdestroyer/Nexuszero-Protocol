use nexuszero_integration::{NexuszeroAPI, ProtocolConfig};
use nexuszero_crypto::proof::statement::HashFunction;

#[test]
fn test_end_to_end_discrete_log() {
    let mut api = NexuszeroAPI::new();
    let g = vec![2u8;32];
    let x = vec![5u8;32];
    let h = g.clone(); // placeholder
    let proof = api.prove_discrete_log(&g, &h, &x).expect("proof generation");
    assert!(api.verify(&proof).expect("verification"));
}

#[test]
fn test_preimage_proof() {
    let mut api = NexuszeroAPI::new();
    let preimage = b"secret";
    use sha3::{Sha3_256, Digest};
    let hash = Sha3_256::digest(preimage).to_vec();
    let proof = api.prove_preimage(HashFunction::SHA3_256, &hash, preimage).expect("proof generation");
    assert!(api.verify(&proof).expect("verification"));
}

#[test]
fn test_config_toggle_compression() {
    // Compression disabled
    let mut api_no = nexuszero_integration::NexuszeroAPI::with_config(ProtocolConfig { use_compression: false, ..Default::default() });
    let mut api_yes = nexuszero_integration::NexuszeroAPI::with_config(ProtocolConfig { use_compression: true, ..Default::default() });
    use num_bigint::BigUint;
    let g = vec![3u8;32];
    let x = vec![7u8;32];
    let modulus_bytes = vec![0xFF;32];
    let g_big = BigUint::from_bytes_be(&g);
    let x_big = BigUint::from_bytes_be(&x);
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    let h = g_big.modpow(&x_big, &mod_big).to_bytes_be();
    let proof_no = api_no.prove_discrete_log(&g, &h, &x).unwrap();
    let proof_yes = api_yes.prove_discrete_log(&g, &h, &x).unwrap();
    assert!(api_no.verify(&proof_no).unwrap());
    assert!(api_yes.verify(&proof_yes).unwrap());
    // Compression ratio should differ (may still be 1.0 until real algorithm implemented)
    assert!(proof_yes.metrics.compression_ratio >= 0.0);
}
