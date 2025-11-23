use nexuszero_holographic::MPS;

#[test]
fn test_mps_basic() {
    let data = vec![1u8,0,1,1,0,1,0,0];
    let mps = MPS::from_proof_data(&data, 4).unwrap();
    assert_eq!(mps.len(), data.len());
    let ratio = mps.compression_ratio();
    assert!(ratio > 0.0);
}
