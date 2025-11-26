use super::ProverError;

pub struct CpuProver;

impl CpuProver {
    pub fn new() -> Self { Self }

    pub async fn generate_proof(&self, circuit_data: &[u8], privacy_level: u8) -> Result<Vec<u8>, ProverError> {
        // For demo purposes, fake proof by hashing input with privacy level appended
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(circuit_data);
        hasher.update(&[privacy_level]);
        let result = hasher.finalize();
        Ok(result.to_vec())
    }
}
