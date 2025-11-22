use nexuszero_holographic::tensor::network::Tensor;

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(&[2,3,4], vec!["i".into(), "j".into(), "k".into()]);
    assert_eq!(t.rank(), 3);
    assert_eq!(t.shape(), &[2,3,4]);
}
