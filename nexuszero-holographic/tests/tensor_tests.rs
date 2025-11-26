use ndarray::{Array2, ArrayD, IxDyn, arr2};
use nexuszero_holographic::tensor::network::Tensor;
use nexuszero_holographic::tensor::decomposition::{TensorSVD, SVDParams};
use nexuszero_holographic::tensor::network::TensorError;

#[test]
fn test_tensor_new_and_basic_methods() {
    let data = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0f64);
    let labels = vec!["x".to_string(), "y".to_string()];
    let t = Tensor::new(data.clone(), labels.clone());
    assert_eq!(t.rank(), 2);
    assert_eq!(t.shape(), &[2usize, 2]);
    assert_eq!(t.labels(), labels.as_slice());
    assert_eq!(t.data(), &data);
}

#[test]
fn test_tensor_zeros_and_reshape_success() {
    let t = Tensor::zeros(&[2, 2], vec!["a".into(), "b".into()]);
    let reshaped = t.reshape(&[4]).expect("reshape ok");
    assert_eq!(reshaped.rank(), 1);
    assert_eq!(reshaped.shape(), &[4]);
}

#[test]
fn test_tensor_reshape_error() {
    let t = Tensor::zeros(&[2, 2], vec!["a".into(), "b".into()]);
    let res = t.reshape(&[3]);
    assert!(matches!(res, Err(TensorError::ShapeMismatch)));
}

#[test]
fn test_decomposition_compute_and_truncate() {
    // simple 3x3 matrix with distinct singular values
    let mat = arr2(&[
        [3.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);

    let svd = TensorSVD::compute(&mat).expect("svd should compute");
    assert!(!svd.s.is_empty());
    let mut t = svd;
    let _orig_len = t.s.len();
    t.truncate(1);
    assert_eq!(t.s.len(), 1);
    assert!(t.compression_ratio() > 0.0);
}

#[test]
fn test_decomposition_params_and_rel_error() {
    // create a 4x3 approx matrix
    let mat = arr2(&[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]);
    let params = SVDParams { power_iters: 1, oversampling: 2, max_oversampling: 4, target_rel_error: 0.9 };
    let svd = TensorSVD::compute_with_params(&mat, params).expect("svd should compute");
    assert!(!svd.s.is_empty());
    // verify singular values are non-negative
    for v in svd.s.iter() { assert!(*v >= 0.0); }
}

#[test]
fn test_decomposition_failure_empty() {
    let mat = Array2::<f64>::zeros((0, 0));
    let res = TensorSVD::compute(&mat);
    assert!(res.is_err());
}
