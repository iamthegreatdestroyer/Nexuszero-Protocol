use ndarray::{Array3, ArrayD, IxDyn};
use rand::Rng;
use crate::tensor::network::TensorError;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MPS {
    tensors: Vec<Array3<f64>>, // rank-3: [left_bond, physical, right_bond]
    bond_dims: Vec<usize>,
    physical_dim: usize,
}

impl MPS {
    pub fn new(length: usize, physical_dim: usize, bond_dim: usize) -> Self {
        let mut tensors = Vec::with_capacity(length);
        let mut bond_dims = Vec::with_capacity(length + 1);
        bond_dims.push(1); // left boundary
        for i in 0..length {
            let left = if i == 0 { 1 } else { bond_dim };
            let right = if i == length - 1 { 1 } else { bond_dim };
            let mut rng = rand::thread_rng();
            let tensor = Array3::from_shape_fn([left, physical_dim, right], |_| rng.gen::<f64>());
            tensors.push(tensor);
            bond_dims.push(right);
        }
        Self { tensors, bond_dims, physical_dim }
    }

    pub fn from_proof_data(data: &[u8], max_bond_dim: usize) -> Result<Self, TensorError> {
        let length = data.len();
        if length == 0 { return Err(TensorError::InvalidContraction); }
        let physical_dim = 4; // coarse bucket encoding (parity + magnitude bucket)
        let mut mps = MPS::new(length, physical_dim, max_bond_dim);
        // Deterministic one-hot encoding per site
        for (i, &byte) in data.iter().enumerate() {
            let bucket = (byte / 64) as usize; // 0..3
            let parity = (byte % 2) as usize;  // 0 or 1
            let encoded_index = (parity + bucket) % physical_dim;
            let tensor = &mut mps.tensors[i];
            for l in 0..tensor.shape()[0] {
                for r in 0..tensor.shape()[2] {
                    for p in 0..physical_dim { tensor[[l,p,r]] = 0.0; }
                    tensor[[l,encoded_index,r]] = 1.0;
                }
            }
        }
        Ok(mps)
    }

    pub fn len(&self) -> usize { self.tensors.len() }

    /// Return true if MPS has no tensors
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }

    pub fn compression_ratio(&self) -> f64 {
        if self.tensors.is_empty() { return 1.0; }
        // Approximate original as raw bytes per site (256 intensity levels)
        let original: usize = self.len() * 256;
        let compressed: usize = self.tensors.iter().map(|t| t.len()).sum();
        compressed as f64 / original as f64
    }

    pub fn verify_boundary(&self, _boundary_data: &[f64]) -> bool {
        // Placeholder: boundary verification would check consistency without full contraction
        true
    }

    pub fn contract_all(&self) -> ArrayD<f64> {
        if self.tensors.is_empty() { return ArrayD::zeros(IxDyn(&[])); }
        // Placeholder: real implementation would sequentially contract along bonds
        // Return a dummy scalar tensor representing contracted norm
        ArrayD::from_elem(IxDyn(&[1]), 0.0)
    }

    /// Approximate serialized size in bytes (heuristic for reporting).
    pub fn approx_serialized_size(&self) -> usize {
        let tensor_bytes: usize = self.tensors.iter().map(|t| t.len() * std::mem::size_of::<f64>()).sum();
        let overhead = self.tensors.len() * 3 * std::mem::size_of::<usize>();
        tensor_bytes + overhead
    }

    /// Expose physical dimension for decode logic
    pub fn physical_dim(&self) -> usize { self.physical_dim }

    /// Return a reference to the raw site tensor for a given index
    pub fn site_tensor(&self, idx: usize) -> Option<&Array3<f64>> {
        self.tensors.get(idx)
    }

    /// Set a site to one-hot encoding for the given physical index. This is a
    /// helper to support lossless encoder/decoder tests.
    pub fn set_site_onehot(&mut self, idx: usize, p_index: usize) {
        if let Some(tensor) = self.tensors.get_mut(idx) {
            let left = tensor.shape()[0];
            let right = tensor.shape()[2];
            for l in 0..left {
                for r in 0..right {
                    for p in 0..self.physical_dim { tensor[[l,p,r]] = 0.0; }
                    if p_index < self.physical_dim { tensor[[l, p_index, r]] = 1.0; }
                }
            }
        }
    }
}
