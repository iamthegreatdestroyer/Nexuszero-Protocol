use nexuszero_holographic::compression::encoder;
use nexuszero_holographic::compression::mps::MPS;
use nexuszero_holographic::compression::decoder;

#[test]
fn test_encode_proof_remapping() {
    let input = vec![0u8, 1, 32, 255];
    let mps_from_encoder = encoder::encode_proof(&input, 8).unwrap();

    // Manually compute expected remapped vector
    let remapped: Vec<u8> = input.iter().map(|b| (b % 2) + ((b / 32) << 1)).collect();
    let mps_from_remapped = MPS::from_proof_data(&remapped, 8).unwrap();

    // Serialization of MPS should be identical
    let ser1 = bincode::serialize(&mps_from_encoder).unwrap();
    let ser2 = bincode::serialize(&mps_from_remapped).unwrap();
    assert_eq!(ser1, ser2);
    // Decoder is a placeholder - ensure behavior matches (empty)
    let decoded = decoder::decode_proof(&mps_from_encoder);
    assert_eq!(decoded.len(), 0);
}

#[test]
fn test_apply_error_correction_noop() {
    let mut data = vec![1u8, 2, 3, 4];
    decoder::apply_error_correction(&mut data);
    assert_eq!(data, vec![1u8, 2, 3, 4]);
}
