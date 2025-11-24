use ndarray::Array2;
use crate::tensor::decomposition::TensorSVD;
use crate::tensor::network::TensorError;

pub type SVDResult = (Array2<f64>, Vec<f64>, Array2<f64>);

pub fn truncated_svd(matrix: &Array2<f64>, k: usize) -> Result<SVDResult, TensorError> {
    let mut svd = TensorSVD::compute(matrix)?;
    svd.truncate(k);
    Ok((svd.u, svd.s, svd.vt))
}

// TODO: Implement a proper randomized SVD (e.g., Halko et al. algorithm)
// to replace the placeholder in TensorSVD::compute without external LAPACK.
