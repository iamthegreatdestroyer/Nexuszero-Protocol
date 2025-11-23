use ndarray::Array4;
use rand::Rng;

/// Projected Entangled Pair State placeholder
#[derive(Debug, Clone)]
pub struct PEPS {
    pub tensors: Vec<Array4<f64>>, // rank-4: [up, left, physical, right]
    pub grid_dims: (usize, usize),
}

impl PEPS {
    pub fn new(rows: usize, cols: usize, physical_dim: usize, bond_dim: usize) -> Self {
        let mut tensors = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..rows*cols {
            let shape = [bond_dim, bond_dim, physical_dim, bond_dim];
            let t = Array4::from_shape_fn(shape, |_| rng.gen::<f64>());
            tensors.push(t);
        }
        Self { tensors, grid_dims: (rows, cols) }
    }

    pub fn compression_ratio(&self) -> f64 {
        let total: usize = self.tensors.iter().map(|t| t.len()).sum();
        let original = self.grid_dims.0 * self.grid_dims.1 * self.tensors[0].shape()[2];
        total as f64 / original as f64
    }
}
