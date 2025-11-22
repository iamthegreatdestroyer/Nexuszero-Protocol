//! Kani proof harness for `ct_in_range` correctness.
//!
//! Ensures the constant-time range check matches a straightforward inclusive
//! comparison for arbitrary 64-bit inputs with low bounds < high bounds.
//!
//! Run with Kani in CI (Linux) or locally in a supported environment.

#![cfg(kani)]

use kani::any;
use nexuszero_crypto::utils::constant_time::ct_in_range;

#[kani::proof]
fn verify_ct_in_range_equivalence() {
    let val: u64 = any();
    let lo: u64 = any();
    let hi: u64 = any();

    // Constrain lo < hi to represent a valid range.
    kani::assume(lo < hi);

    let ct = ct_in_range(val, lo, hi);
    let ref_impl = val >= lo && val <= hi;

    assert!(ct == ref_impl, "ct_in_range diverges from reference comparison");
}
