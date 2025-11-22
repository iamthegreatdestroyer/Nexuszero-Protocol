//! Kani proof harness for `ct_dot_product` correctness.
//!
//! Verifies that the constant-time dot product equals a naive implementation
//! for small vector lengths. Length is bounded to keep exploration tractable.

#![cfg(kani)]

use kani::any;
use nexuszero_crypto::utils::constant_time::ct_dot_product;

#[kani::proof]
fn verify_ct_dot_product_equivalence() {
    // Choose a small length bound for tractability in symbolic execution.
    let len: usize = any();
    kani::assume(len > 0 && len <= 6);

    // Generate arrays of length `len`.
    // Kani cannot directly create variable-length vectors with `any()` easily;
    // we approximate by creating fixed-size arrays and slicing.
    let a0: i64 = any();
    let a1: i64 = any();
    let a2: i64 = any();
    let a3: i64 = any();
    let a4: i64 = any();
    let a5: i64 = any();

    let b0: i64 = any();
    let b1: i64 = any();
    let b2: i64 = any();
    let b3: i64 = any();
    let b4: i64 = any();
    let b5: i64 = any();

    let a_full = [a0, a1, a2, a3, a4, a5];
    let b_full = [b0, b1, b2, b3, b4, b5];

    let a = &a_full[..len];
    let b = &b_full[..len];

    let ct_res = ct_dot_product(a, b);
    let ref_res = a.iter().zip(b.iter()).fold(0i64, |acc, (x, y)| acc + x * y);

    assert!(ct_res == ref_res, "ct_dot_product diverges from reference implementation");
}
