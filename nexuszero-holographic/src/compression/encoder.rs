use crate::compression::mps::MPS;
use crate::tensor::network::TensorError;

/// Encode proof bytes into an MPS using simple chunking.
pub fn encode_proof(data: &[u8], max_bond_dim: usize) -> Result<MPS, TensorError> {
    // For now treat each byte as a physical site with dim=256; reduce by mapping to binary parity plus magnitude bucket.
    let mut remapped: Vec<u8> = Vec::with_capacity(data.len());
    for &b in data {
        // Simple reduction: parity bit (b % 2) and bucket (b/32)
        remapped.push((b % 2) + ((b / 32) << 1));
    }
    MPS::from_proof_data(&remapped, max_bond_dim)
}
