use ndarray::Array2;
use nexuszero_holographic::tensor::decomposition::TensorSVD;

#[test]
fn test_randomized_svd_reconstruction_quality() {
    let m = 40; let n = 25;
    let matrix = Array2::from_shape_fn((m, n), |(i,j)| ((i + 2*j) as f64).sin());
    let mut svd = TensorSVD::compute(&matrix).expect("SVD compute failed");
    let k = 15; // slightly higher target rank for better approximation
    svd.truncate(k);
    // Reconstruct approximate matrix
    let u = &svd.u;
    let vt = &svd.vt;
    let s = ndarray::Array2::from_diag(&ndarray::Array1::from(svd.s.clone()));
    let approx = u.dot(&s).dot(vt);
    // Compute relative Frobenius error
    let frob_orig = matrix.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = &matrix - &approx;
    let frob_diff = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
    let rel_err = frob_diff / frob_orig;
    println!("Computed relative reconstruction error: {rel_err}");
    assert!(rel_err < 0.7, "Reconstruction error too high after improved SVD: {rel_err}");
}

#[test]
fn test_singular_values_monotonic() {
    let matrix = Array2::from_shape_fn((30, 30), |(i,j)| ((i*j + i + j) as f64).cos());
    let svd = TensorSVD::compute(&matrix).expect("SVD compute failed");
    let mut prev = f64::INFINITY;
    for &s in &svd.s { assert!(s <= prev + 1e-8); prev = s; }
}
