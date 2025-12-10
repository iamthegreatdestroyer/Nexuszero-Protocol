// Comprehensive tests for dual and multi-exponentiation implementations
#[cfg(test)]
mod tests {
    use nexuszero_crypto::utils::{
        MultiExpConfig, ExpTable, ShamirTrick, VectorExponentiation,
        InterleavedExponentiation, WindowedMultiExponentiation
    };
    use num_bigint::ToBigUint;

    /// Test Shamir's trick basic functionality
    #[test]
    fn test_shamir_trick_basic() {
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());

        let a = 2u32.to_biguint().unwrap();
        let x = 3u32.to_biguint().unwrap();
        let b = 3u32.to_biguint().unwrap();
        let y = 2u32.to_biguint().unwrap();
        let modulus = 7u32.to_biguint().unwrap();

        // 2^3 * 3^2 mod 7 = 8 * 9 mod 7 = 72 mod 7 = 2
        let result = shamir.compute(&a, &x, &b, &y, &modulus).unwrap();
        assert_eq!(result, 2u32.to_biguint().unwrap());
    }

    /// Test Shamir's trick with larger numbers
    #[test]
    fn test_shamir_trick_large_numbers() {
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());

        let a = 123u32.to_biguint().unwrap();
        let x = 45u32.to_biguint().unwrap();
        let b = 456u32.to_biguint().unwrap();
        let y = 67u32.to_biguint().unwrap();
        let modulus = 1000000007u64.to_biguint().unwrap();

        // Should complete without panicking
        let result = shamir.compute(&a, &x, &b, &y, &modulus).unwrap();

        // Verify against naive computation
        let mut expected = 1u64.to_biguint().unwrap();
        for _ in 0..45 {
            expected = (&expected * &a) % &modulus;
        }
        for _ in 0..67 {
            expected = (&expected * &b) % &modulus;
        }

        assert_eq!(result, expected);
    }

    /// Test Shamir's trick with zero exponent
    #[test]
    fn test_shamir_trick_zero_exponent() {
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());

        let a = 2u32.to_biguint().unwrap();
        let x = 0u32.to_biguint().unwrap();
        let b = 3u32.to_biguint().unwrap();
        let y = 2u32.to_biguint().unwrap();
        let modulus = 7u32.to_biguint().unwrap();

        // a^0 * b^2 = 1 * 9 mod 7 = 2
        let result = shamir.compute(&a, &x, &b, &y, &modulus).unwrap();
        assert_eq!(result, 2u32.to_biguint().unwrap());
    }

    /// Test Shamir's trick with both zero exponents
    #[test]
    fn test_shamir_trick_both_zero() {
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());

        let a = 2u32.to_biguint().unwrap();
        let x = 0u32.to_biguint().unwrap();
        let b = 3u32.to_biguint().unwrap();
        let y = 0u32.to_biguint().unwrap();
        let modulus = 7u32.to_biguint().unwrap();

        // a^0 * b^0 = 1 * 1 = 1
        let result = shamir.compute(&a, &x, &b, &y, &modulus).unwrap();
        assert_eq!(result, 1u32.to_biguint().unwrap());
    }

    /// Test exp table creation and lookup
    #[test]
    fn test_exp_table() {
        let base = 2u32.to_biguint().unwrap();
        let modulus = 11u32.to_biguint().unwrap();
        let table = ExpTable::new(base, modulus, 3).unwrap();

        // Verify table contents: 2^i mod 11
        let expected = [1, 2, 4, 8, 5, 10, 9, 7];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(table.lookup(i), Some(&exp.to_biguint().unwrap()));
        }
    }

    /// Test exp table with window size
    #[test]
    fn test_exp_table_window_size() {
        let base = 3u32.to_biguint().unwrap();
        let modulus = 17u32.to_biguint().unwrap();
        let window_size = 2; // Should create 2^2 = 4 entries

        let table = ExpTable::new(base, modulus, window_size).unwrap();
        assert_eq!(table.powers.len(), 4);
    }

    /// Test vector exponentiation basic
    #[test]
    fn test_vector_exponentiation_basic() {
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);

        let bases = vec![
            2u32.to_biguint().unwrap(),
            3u32.to_biguint().unwrap(),
        ];
        let exponents = vec![
            2u32.to_biguint().unwrap(),
            3u32.to_biguint().unwrap(),
        ];
        let modulus = 11u32.to_biguint().unwrap();

        // 2^2 * 3^3 mod 11 = 4 * 27 mod 11 = 108 mod 11 = 9
        let result = vec_exp.compute(&bases, &exponents, &modulus).unwrap();
        assert_eq!(result, 9u32.to_biguint().unwrap());
    }

    /// Test vector exponentiation with mismatched dimensions
    #[test]
    fn test_vector_exponentiation_dimension_mismatch() {
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);

        let bases = vec![2u32.to_biguint().unwrap()];
        let exponents = vec![
            2u32.to_biguint().unwrap(),
            3u32.to_biguint().unwrap(),
        ];
        let modulus = 11u32.to_biguint().unwrap();

        let result = vec_exp.compute(&bases, &exponents, &modulus);
        assert!(result.is_err());
    }

    /// Test vector exponentiation with more bases
    #[test]
    fn test_vector_exponentiation_multiple_bases() {
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);

        let bases = vec![
            2u32.to_biguint().unwrap(),
            3u32.to_biguint().unwrap(),
            5u32.to_biguint().unwrap(),
        ];
        let exponents = vec![
            1u32.to_biguint().unwrap(),
            1u32.to_biguint().unwrap(),
            1u32.to_biguint().unwrap(),
        ];
        let modulus = 31u32.to_biguint().unwrap();

        // 2 * 3 * 5 mod 31 = 30
        let result = vec_exp.compute(&bases, &exponents, &modulus).unwrap();
        assert_eq!(result, 30u32.to_biguint().unwrap());
    }

    /// Test windowed exponentiation with adaptive window size
    #[test]
    fn test_windowed_adaptive_window_size() {
        let config = MultiExpConfig::default();
        let windowed = WindowedMultiExponentiation::new(config, 6);

        let bases = vec![2u32.to_biguint().unwrap()];
        let exponents = vec![100u32.to_biguint().unwrap()];
        let modulus = 997u32.to_biguint().unwrap();

        // Should use window size 4 for 100-bit exponent
        let result = windowed.compute(&bases, &exponents, &modulus).unwrap();

        // Verify: 2^100 mod 997
        let mut expected = 1u32.to_biguint().unwrap();
        for _ in 0..100 {
            expected = (&expected * 2u32.to_biguint().unwrap()) % &modulus;
        }

        assert_eq!(result, expected);
    }

    /// Test interleaved exponentiation basic
    #[test]
    fn test_interleaved_exponentiation_basic() {
        let config = MultiExpConfig::default();
        let interleaved = InterleavedExponentiation::new(config);

        let bases = vec![
            2u32.to_biguint().unwrap(),
            5u32.to_biguint().unwrap(),
        ];
        let exponents = vec![
            3u32.to_biguint().unwrap(),
            2u32.to_biguint().unwrap(),
        ];
        let modulus = 23u32.to_biguint().unwrap();

        // 2^3 * 5^2 mod 23 = 8 * 25 mod 23 = 200 mod 23 = 16
        let result = interleaved.compute(&bases, &exponents, &modulus).unwrap();
        assert_eq!(result, 16u32.to_biguint().unwrap());
    }

    /// Test interleaved exponentiation preprocessing
    #[test]
    fn test_interleaved_preprocessing() {
        let config = MultiExpConfig::default();
        let interleaved = InterleavedExponentiation::new(config);

        let exponents = vec![5u32.to_biguint().unwrap(), 3u32.to_biguint().unwrap()];
        let preprocessed = interleaved.preprocess_exponents(&exponents).unwrap();

        // Should have preprocessed both exponents
        assert_eq!(preprocessed.len(), 2);
        assert!(!preprocessed[0].is_empty());
        assert!(!preprocessed[1].is_empty());
    }

    /// Test multiexp config defaults
    #[test]
    fn test_multiexp_config_defaults() {
        let config = MultiExpConfig::default();

        assert_eq!(config.window_size, 5);
        assert_eq!(config.max_bases, 4);
        assert_eq!(config.table_size, 32);
        assert!(config.simd_enabled);
        assert!(config.cache_tables);
    }

    /// Test custom multiexp config
    #[test]
    fn test_multiexp_custom_config() {
        let config = MultiExpConfig {
            window_size: 6,
            max_bases: 8,
            table_size: 64,
            simd_enabled: false,
            cache_tables: false,
        };

        assert_eq!(config.window_size, 6);
        assert_eq!(config.max_bases, 8);
        assert_eq!(config.table_size, 64);
        assert!(!config.simd_enabled);
        assert!(!config.cache_tables);
    }

    /// Test correctness across multiple computation methods
    #[test]
    fn test_consistency_across_methods() {
        let a = 7u32.to_biguint().unwrap();
        let x = 5u32.to_biguint().unwrap();
        let b = 11u32.to_biguint().unwrap();
        let y = 3u32.to_biguint().unwrap();
        let modulus = 101u32.to_biguint().unwrap();

        // Compute using Shamir's trick
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());
        let shamir_result = shamir.compute(&a, &x, &b, &y, &modulus).unwrap();

        // Compute using vector exponentiation
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);
        let vec_result = vec_exp
            .compute(&vec![a.clone(), b.clone()], &vec![x.clone(), y.clone()], &modulus)
            .unwrap();

        // Results should match
        assert_eq!(shamir_result, vec_result);
    }

    /// Test with prime modulus
    #[test]
    fn test_with_prime_modulus() {
        let bases = vec![2u32.to_biguint().unwrap(), 3u32.to_biguint().unwrap()];
        let exponents = vec![10u32.to_biguint().unwrap(), 20u32.to_biguint().unwrap()];
        let modulus = 1000000007u64.to_biguint().unwrap(); // Large prime

        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);
        let result = vec_exp.compute(&bases, &exponents, &modulus).unwrap();

        // Verify result is less than modulus
        assert!(result < modulus);
    }

    /// Test error handling with zero modulus
    #[test]
    fn test_error_zero_modulus() {
        let mut shamir = ShamirTrick::new(MultiExpConfig::default());
        let a = 2u32.to_biguint().unwrap();
        let x = 3u32.to_biguint().unwrap();
        let b = 3u32.to_biguint().unwrap();
        let y = 2u32.to_biguint().unwrap();
        let zero_modulus = 0u32.to_biguint().unwrap();

        let result = shamir.compute(&a, &x, &b, &y, &zero_modulus);
        assert!(result.is_err());
    }

    /// Test large exponents
    #[test]
    fn test_large_exponents() {
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);

        let bases = vec![2u32.to_biguint().unwrap()];
        let large_exp = (1u128 << 100).to_biguint().unwrap();
        let exponents = vec![large_exp];
        let modulus = 1000000007u64.to_biguint().unwrap();

        // Should handle large exponents without panic
        let result = vec_exp.compute(&bases, &exponents, &modulus);
        assert!(result.is_ok());
    }

    /// Test identity property (base^0 = 1)
    #[test]
    fn test_identity_property() {
        let config = MultiExpConfig::default();
        let vec_exp = VectorExponentiation::new(config);

        let bases = vec![123u32.to_biguint().unwrap()];
        let exponents = vec![0u32.to_biguint().unwrap()];
        let modulus = 1000000007u64.to_biguint().unwrap();

        let result = vec_exp.compute(&bases, &exponents, &modulus).unwrap();
        assert_eq!(result, 1u32.to_biguint().unwrap());
    }
}
