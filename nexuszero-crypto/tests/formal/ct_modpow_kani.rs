//! Kani proof harness for `ct_modpow` functional correctness on small 64-bit inputs.
//!
//! This harness checks that the constant-time Montgomery ladder modular exponentiation
//! returns the same result as the reference `num_bigint::modpow` for arbitrary
//! 64-bit bases, exponents, and moduli (modulus > 1). While this does not prove
//! constant-time properties directly, it ensures semantic equivalence and absence
//! of panics/divergence for the explored state space.
//!
//! Run with: `cargo kani --tests formal::ct_modpow_kani` (after installing cargo-kani)
//! or simply `cargo kani` to explore all proof harnesses.
//!
//! Note: Proving strict constant-time behavior requires either side-channel
//! analysis or specialized tooling; this harness is a first step toward formal
//! assurance by validating functional correctness of the ladder algorithm.

#![cfg(kani)]

use kani::any;
use num_bigint::BigUint;
use nexuszero_crypto::utils::constant_time::ct_modpow;

#[kani::proof]
fn verify_ct_modpow_small_equivalence() {
    let base: u64 = any();
    let exp: u64 = any();
    let modulus: u64 = any();

    // Constrain modulus to avoid division/modulo by zero and trivial group.
    kani::assume(modulus > 1);

    let b = BigUint::from(base);
    let e = BigUint::from(exp);
    let m = BigUint::from(modulus);

    let ct_res = ct_modpow(&b, &e, &m);
    let ref_res = b.modpow(&e, &m);

    // Semantic equivalence check.
    assert!(ct_res == ref_res, "ct_modpow result diverges from reference implementation");
}
