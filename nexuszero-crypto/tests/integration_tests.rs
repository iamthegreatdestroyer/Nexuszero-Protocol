//! Integration tests for nexuszero-crypto

use nexuszero_crypto::lattice::lwe::{LWEParameters, keygen, encrypt, decrypt};
use nexuszero_crypto::{CryptoParameters, SecurityLevel};

#[test]
fn test_lwe_integration() {
    let params = LWEParameters::new(32, 64, 97, 2.0);
    // Use deterministic RNG for integration tests to avoid flakiness under
    // instrumentation and tarpaulin runs
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;
    let mut rng = ChaCha8Rng::from_seed([0u8; 32]);

    let (sk, pk) = keygen(&params, &mut rng).unwrap();

    let message = true;
    let ciphertext = encrypt(&pk, message, &params, &mut rng).unwrap();
    let decrypted = decrypt(&sk, &ciphertext, &params).unwrap();

    assert_eq!(message, decrypted);
}

#[test]
fn test_security_parameters() {
    let params_128 = CryptoParameters::new_128bit_security();
    assert_eq!(params_128.security_level, SecurityLevel::Bit128);

    let params_192 = CryptoParameters::new_192bit_security();
    assert_eq!(params_192.security_level, SecurityLevel::Bit192);

    let params_256 = CryptoParameters::new_256bit_security();
    assert_eq!(params_256.security_level, SecurityLevel::Bit256);
}
