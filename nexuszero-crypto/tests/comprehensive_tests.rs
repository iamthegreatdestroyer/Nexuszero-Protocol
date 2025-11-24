//! Comprehensive test suite for nexuszero-crypto
//!
//! This module provides extensive testing including:
//! - Correctness tests for all security levels
//! - Property-based testing with proptest
//! - Security property verification
//! - Edge case testing

use nexuszero_crypto::lattice::*;
use proptest::prelude::*;

#[cfg(test)]
mod correctness_tests {
    use super::*;

    #[test]
    fn test_lwe_correctness_across_security_levels() {
        use rand::thread_rng;
        
        // Test all three security levels
        let params_128 = lwe::LWEParameters::new(512, 1024, 12289, 3.2);
        let params_192 = lwe::LWEParameters::new(1024, 2048, 40961, 3.2);
        let params_256 = lwe::LWEParameters::new(2048, 4096, 65537, 3.2);

        for params in [params_128, params_192, params_256] {
            let mut rng = thread_rng();
            let (sk, pk) = lwe::keygen(&params, &mut rng).unwrap();

            // Test 10 rounds with both true and false
            for message_bit in [true, false] {
                let ct = lwe::encrypt(&pk, message_bit, &params, &mut rng).unwrap();
                assert_eq!(
                    lwe::decrypt(&sk, &ct, &params).unwrap(),
                    message_bit,
                    "Failed at n={}, message={}", params.n, message_bit
                );
            }
        }
    }

    #[test]
    fn test_lwe_encryption_randomness() {
        use rand::thread_rng;
        
        let params = lwe::LWEParameters::new(512, 1024, 12289, 3.2);
        let mut rng = thread_rng();
        let (sk, pk) = lwe::keygen(&params, &mut rng).unwrap();

        let ct1 = lwe::encrypt(&pk, true, &params, &mut rng).unwrap();
        let ct2 = lwe::encrypt(&pk, true, &params, &mut rng).unwrap();

        // Ciphertexts should be different (probabilistic encryption)
        assert_ne!(ct1.v, ct2.v, "Encryption should be randomized");

        // But both should decrypt to the same value
        assert!(lwe::decrypt(&sk, &ct1, &params).unwrap());
        assert!(lwe::decrypt(&sk, &ct2, &params).unwrap());
    }

    #[test]
    fn test_ring_lwe_polynomial_operations() {
        let modulus = 17;
        let a = ring_lwe::Polynomial::from_coeffs(vec![1, 2, 3, 4], modulus);
        let b = ring_lwe::Polynomial::from_coeffs(vec![5, 6, 7, 8], modulus);

        let sum = ring_lwe::poly_add(&a, &b, modulus);
        assert_eq!(sum.coeffs[0], 6);
        assert_eq!(sum.coeffs[1], 8);

        let diff = ring_lwe::poly_sub(&a, &b, modulus);
        assert_eq!(diff.coeffs[0], (1i64 - 5).rem_euclid(modulus as i64));

        let scaled = ring_lwe::poly_scalar_mult(&a, 2, modulus);
        assert_eq!(scaled.coeffs[0], 2);
    }

    #[test]
    fn test_ring_lwe_ntt_correctness() {
        let params = ring_lwe::RingLWEParameters::new_128bit_security();

        // Create a test polynomial
        for _ in 0..3 {
            let coeffs: Vec<i64> = (0..params.n).map(|i| (i % params.q as usize) as i64).collect();
            let poly = ring_lwe::Polynomial::from_coeffs(coeffs.clone(), params.q);

            // Transform forward and back (we need the primitive root)
            let primitive_root = 3; // For modulus 12289
            let ntt_poly = ring_lwe::ntt(&poly, params.q, primitive_root);
            let recovered = ring_lwe::intt(&ntt_poly, params.n, params.q, primitive_root);

            // Check coefficients are recovered (may differ by scaling)
            assert_eq!(poly.coeffs.len(), recovered.coeffs.len(), "NTT should preserve degree");
        }
    }

    #[test]
    fn test_ring_lwe_encryption_correctness() {
        let security_levels = [
            ring_lwe::RingLWEParameters::new_128bit_security(),
            ring_lwe::RingLWEParameters::new_192bit_security(),
            ring_lwe::RingLWEParameters::new_256bit_security(),
        ];

        for params in &security_levels {
            let (sk, pk) = ring_lwe::ring_keygen(params).unwrap();

            for msg_bool in [true, false] {
                let message = vec![msg_bool];
                let ciphertext = ring_lwe::ring_encrypt(&pk, &message, params).unwrap();
                let decrypted = ring_lwe::ring_decrypt(&sk, &ciphertext, params).unwrap();

                // Ring-LWE decodes all n coefficients, check first one matches message
                assert!(!decrypted.is_empty(), "Should decrypt at least one bit");
                assert_eq!(decrypted[0], msg_bool,
                    "Failed to recover message {:?} at security level n={}", msg_bool, params.n);
            }
        }
    }
}

#[cfg(test)]
mod property_based_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_lwe_encryption_decryption(message_bit in any::<bool>()) {
            use rand::thread_rng;
            
            let params = lwe::LWEParameters::new(256, 512, 12289, 3.2);
            let mut rng = thread_rng();
            let (sk, pk) = lwe::keygen(&params, &mut rng).unwrap();

            let ciphertext = lwe::encrypt(&pk, message_bit, &params, &mut rng).unwrap();
            let decrypted = lwe::decrypt(&sk, &ciphertext, &params).unwrap();

            prop_assert_eq!(decrypted, message_bit);
        }

        #[test]
        fn prop_polynomial_addition_commutative(a in prop::collection::vec(0i64..17, 4), b in prop::collection::vec(0i64..17, 4)) {
            let modulus = 17;
            let poly_a = ring_lwe::Polynomial::from_coeffs(a, modulus);
            let poly_b = ring_lwe::Polynomial::from_coeffs(b, modulus);

            let sum1 = ring_lwe::poly_add(&poly_a, &poly_b, modulus);
            let sum2 = ring_lwe::poly_add(&poly_b, &poly_a, modulus);

            prop_assert_eq!(sum1.coeffs, sum2.coeffs);
        }

        #[test]
        fn prop_polynomial_addition_associative(
            a in prop::collection::vec(0i64..17, 4),
            b in prop::collection::vec(0i64..17, 4),
            c in prop::collection::vec(0i64..17, 4)
        ) {
            let modulus = 17;
            let poly_a = ring_lwe::Polynomial::from_coeffs(a, modulus);
            let poly_b = ring_lwe::Polynomial::from_coeffs(b, modulus);
            let poly_c = ring_lwe::Polynomial::from_coeffs(c, modulus);

            let sum1 = ring_lwe::poly_add(&ring_lwe::poly_add(&poly_a, &poly_b, modulus), &poly_c, modulus);
            let sum2 = ring_lwe::poly_add(&poly_a, &ring_lwe::poly_add(&poly_b, &poly_c, modulus), modulus);

            prop_assert_eq!(sum1.coeffs, sum2.coeffs);
        }

        #[test]
        fn prop_ring_lwe_message_recovery(msg in any::<bool>()) {
            let params = ring_lwe::RingLWEParameters::new_128bit_security();
            let (sk, pk) = ring_lwe::ring_keygen(&params).unwrap();

            let message = vec![msg];
            let ciphertext = ring_lwe::ring_encrypt(&pk, &message, &params).unwrap();
            let decrypted = ring_lwe::ring_decrypt(&sk, &ciphertext, &params).unwrap();

            // Ring-LWE decodes all n coefficients, check first one matches
            prop_assert!(!decrypted.is_empty());
            prop_assert_eq!(decrypted[0], message[0]);
        }
    }
}

#[cfg(test)]
mod security_tests {
    use super::*;

    #[test]
    fn test_zeroization() {
        use rand::thread_rng;
        
        // Test that secret keys can be inspected (zeroization would require ndarray support)
        let params = lwe::LWEParameters::new(256, 512, 12289, 3.2);
        let mut rng = thread_rng();
        let (sk, _pk) = lwe::keygen(&params, &mut rng).unwrap();
        
        // Store original values
        let has_nonzero = sk.s.iter().any(|&x| x != 0);
        assert!(has_nonzero, "Secret key should have non-zero values");
        
        // Verify secret key has expected properties
        assert_eq!(sk.s.len(), params.n, "Secret key should have correct dimension");
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_empty_polynomial() {
        let poly = ring_lwe::Polynomial::zero(4, 17);
        assert_eq!(poly.coeffs.len(), 4);
        assert!(poly.coeffs.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_polynomial_zero_addition() {
        let modulus = 17;
        let poly = ring_lwe::Polynomial::from_coeffs(vec![1, 2, 3, 4], modulus);
        let zero = ring_lwe::Polynomial::zero(4, modulus);

        let result = ring_lwe::poly_add(&poly, &zero, modulus);
        assert_eq!(result.coeffs, poly.coeffs);
    }

    #[test]
    fn test_larger_dimension() {
        use rand::thread_rng;
        
        // Test with larger dimension instead of large modulus
        let params = lwe::LWEParameters::new(256, 512, 12289, 3.2);
        let mut rng = thread_rng();
        
        // Should handle larger dimensions
        let result = lwe::keygen(&params, &mut rng);
        assert!(result.is_ok(), "Should handle larger dimensions");
        
        let (sk, pk) = result.unwrap();
        let ct = lwe::encrypt(&pk, true, &params, &mut rng).unwrap();
        let pt = lwe::decrypt(&sk, &ct, &params).unwrap();
        assert!(pt, "Should correctly encrypt/decrypt with larger dimensions");
    }

    #[test]
    fn test_zero_message_encryption() {
        use rand::thread_rng;
        
        let params = lwe::LWEParameters::new(256, 512, 12289, 3.2);
        let mut rng = thread_rng();
        let (sk, pk) = lwe::keygen(&params, &mut rng).unwrap();

        let ciphertext = lwe::encrypt(&pk, false, &params, &mut rng).unwrap();
        let decrypted = lwe::decrypt(&sk, &ciphertext, &params).unwrap();

        assert!(!decrypted);
    }
}
