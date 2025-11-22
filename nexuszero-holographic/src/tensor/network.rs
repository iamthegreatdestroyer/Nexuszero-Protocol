use ndarray::{ArrayD, IxDyn};

#[derive(Clone, Debug)]
pub struct Tensor {
    data: ArrayD<f64>,
    labels: Vec<String>,
}

impl Tensor {
    pub fn new(data: ArrayD<f64>, labels: Vec<String>) -> Self {
        assert_eq!(data.ndim(), labels.len(), "Number of labels must match tensor rank");
        Tensor { data, labels }
    }
    pub fn zeros(shape: &[usize], labels: Vec<String>) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self::new(data, labels)
    }
    pub fn data(&self) -> &ArrayD<f64> { &self.data }
    pub fn rank(&self) -> usize { self.data.ndim() }
    pub fn shape(&self) -> &[usize] { self.data.shape() }
    pub fn labels(&self) -> &[String] { &self.labels }
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, TensorError> {
        let size: usize = self.shape().iter().product();
        let new_size: usize = new_shape.iter().product();
        if size != new_size { return Err(TensorError::ShapeMismatch); }
        let reshaped = self.data.clone().into_shape(IxDyn(new_shape))
            .map_err(|_| TensorError::ShapeMismatch)?;
        let mut new_labels = self.labels.clone();
        new_labels.resize(new_shape.len(), "unlabeled".to_string());
        Ok(Tensor::new(reshaped, new_labels))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch")] ShapeMismatch,
    #[error("Invalid contraction")] InvalidContraction,
    #[error("Decomposition failed")] DecompositionFailed,
}
