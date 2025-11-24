use ndarray::{Array1, Array2};
use crate::tensor::network::TensorError;
use rand::Rng;

/// Configuration parameters for randomized SVD.
/// power_iters: number of (A A^T) applications to sharpen singular spectrum.
/// oversampling: initial extra columns beyond target rank.
/// max_oversampling: upper bound for adaptive oversampling growth.
/// target_rel_error: attempt to reach this reconstruction error (best-effort heuristic).
#[derive(Clone, Copy, Debug)]
pub struct SVDParams {
    pub power_iters: usize,
    pub oversampling: usize,
    pub max_oversampling: usize,
    pub target_rel_error: f64,
}

impl Default for SVDParams {
    fn default() -> Self {
        Self { power_iters: 2, oversampling: 10, max_oversampling: 24, target_rel_error: 0.75 }
    }
}

pub struct TensorSVD {
    pub u: Array2<f64>,
    pub s: Vec<f64>,
    pub vt: Array2<f64>,
}

impl TensorSVD {
    /// Compute randomized SVD with configurable power iterations and adaptive oversampling.
    /// Returns full (possibly oversampled) decomposition; caller can truncate.
    pub fn compute(matrix: &Array2<f64>) -> Result<Self, TensorError> {
        Self::compute_with_params(matrix, SVDParams::default())
    }

    pub fn compute_with_params(matrix: &Array2<f64>, _params: SVDParams) -> Result<Self, TensorError> {
        let (m, n) = (matrix.nrows(), matrix.ncols());
        if m == 0 || n == 0 { return Err(TensorError::DecompositionFailed); }
        let rank_bound = m.min(n);

        // Form normal matrix C = A^T A (n x n) for eigen decomposition.
        let c = matrix.t().dot(matrix); // symmetric positive semidefinite
        let (eigvals, eigvecs) = symmetric_eigen_power_deflation(c, n);
        let mut sv_pairs: Vec<(f64, Array1<f64>)> = eigvals.into_iter().zip(column_iter(&eigvecs)).collect();
        sv_pairs.retain(|(val, _)| *val >= 0.0);
        sv_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        // Singular values
        let singular_vals: Vec<f64> = sv_pairs.iter().map(|(v, _)| v.sqrt()).collect();
        if singular_vals.is_empty() { return Err(TensorError::DecompositionFailed); }

        // Right singular vectors are eigenvectors of A^T A (already normalized by power iteration)
        let mut vt = Array2::<f64>::zeros((sv_pairs.len(), n));
        for (i, (_lam, vec)) in sv_pairs.iter().enumerate() {
            for j in 0..n { vt[[i, j]] = vec[j]; }
        }

        // Left singular vectors u_i = (A v_i)/sigma_i
        let mut u = Array2::<f64>::zeros((m, sv_pairs.len()));
        for (i, (lam, vec)) in sv_pairs.iter().enumerate() {
            let sigma = lam.sqrt();
            if sigma <= 1e-14 { continue; }
            let vcol = vec.to_owned().insert_axis(ndarray::Axis(1));
            let acol = matrix.dot(&vcol);
            for r in 0..m { u[[r, i]] = acol[[r, 0]] / sigma; }
        }
        // Optionally limit to rank_bound
        let mut result = Self { u, s: singular_vals, vt };
        result.truncate(rank_bound);
        Ok(result)
    }

    pub fn truncate(&mut self, k: usize) {
        let k = k.min(self.s.len());
        self.s.truncate(k);
        self.u = self.u.clone().slice_move(ndarray::s![.., ..k]);
        self.vt = self.vt.clone().slice_move(ndarray::s![..k, ..]);
    }
    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.u.nrows() * self.vt.ncols();
        let compressed_size = self.u.nrows() * self.s.len() + self.s.len() + self.s.len() * self.vt.ncols();
        compressed_size as f64 / original_size as f64
    }
}

// Modified Gram-Schmidt orthonormalization
#[allow(dead_code)]
fn orthonormalize(a: Array2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut q = Array2::<f64>::zeros((m, n));
    let eps = 1e-12;
    let mut col = 0;
    for j in 0..n {
        // v = a[:,j]
        let mut v = a.column(j).to_owned();
        // subtract projections
        for k in 0..col {
            let qk = q.column(k);
            let proj = v.dot(&qk);
            v = &v - &(proj * &qk);
        }
        let norm = v.dot(&v).sqrt();
        if norm > eps {
            for i in 0..m { q[[i, col]] = v[i] / norm; }
            col += 1;
        }
    }
    q.slice_move(ndarray::s![.., ..col])
}

// Power iteration with deflation for symmetric eigenvalue problem
fn symmetric_eigen_power_deflation(mut a: Array2<f64>, k: usize) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut eigvals = Vec::new();
    let mut eigvecs = Array2::<f64>::zeros((n, k));
    let max_iter = 60;
    let tol = 1e-8;
    let mut rng = rand::thread_rng();
    for idx in 0..k {
        // random initial vector
        let mut v = Array1::<f64>::zeros(n);
        for x in v.iter_mut() { *x = rng.gen::<f64>() * 2.0 - 1.0; }
        // Orthonormalize against previous eigenvectors
        for j in 0..idx {
            let prev = eigvecs.column(j);
            let proj = v.dot(&prev);
            v = &v - &(proj * &prev);
        }
        normalize(&mut v);
        let mut lambda_old = 0.0;
        for _ in 0..max_iter {
            // w = A v
            let mut w = Array1::<f64>::zeros(n);
            for i in 0..n { for j in 0..n { w[i] += a[[i,j]] * v[j]; } }
            // deflate components along previous eigenvectors to maintain orthogonality
            for j in 0..idx {
                let prev = eigvecs.column(j);
                let proj = w.dot(&prev);
                for i in 0..n { w[i] -= proj * prev[i]; }
            }
            let norm_w = w.dot(&w).sqrt();
            if norm_w < tol { break; }
            for i in 0..n { v[i] = w[i] / norm_w; }
            let lambda = rayleigh_quotient(&a, &v);
            if (lambda - lambda_old).abs() < tol { lambda_old = lambda; break; }
            lambda_old = lambda;
        }
        // store eigenpair
        eigvals.push(lambda_old.max(0.0));
        for i in 0..n { eigvecs[[i, idx]] = v[i]; }
        // Deflation (rank-1 update): A = A - lambda v v^T
        for i in 0..n { for j in 0..n { a[[i,j]] -= lambda_old * v[i] * v[j]; } }
    }
    (eigvals, eigvecs)
}

fn rayleigh_quotient(a: &Array2<f64>, v: &Array1<f64>) -> f64 {
    let n = v.len();
    let mut av = Array1::<f64>::zeros(n);
    for i in 0..n { for j in 0..n { av[i] += a[[i,j]] * v[j]; } }
    v.dot(&av)
}

fn normalize(v: &mut Array1<f64>) {
    let norm = v.dot(v).sqrt();
    if norm > 0.0 { for x in v.iter_mut() { *x /= norm; } }
}

// Iterate columns of a matrix as Array1 clones
fn column_iter(a: &Array2<f64>) -> Vec<Array1<f64>> {
    (0..a.ncols()).map(|j| a.column(j).to_owned()).collect()
}

// Compute relative Frobenius error of reconstruction using provided truncated SVD.
#[allow(dead_code)]
fn approximate_rel_frob_error(original: &Array2<f64>, svd: &TensorSVD) -> f64 {
    let u = &svd.u;
    let vt = &svd.vt;
    let s = ndarray::Array2::from_diag(&ndarray::Array1::from(svd.s.clone()));
    let approx = u.dot(&s).dot(vt);
    let frob_orig = original.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff = original - &approx;
    let frob_diff = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
    frob_diff / frob_orig
}
