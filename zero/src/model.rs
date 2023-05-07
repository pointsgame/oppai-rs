use ndarray::{Array1, Array3, Array4};
use num_traits::Float;

pub trait Model<N: Float> {
  type E;

  fn predict(&self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E>;
}

pub trait TrainableModel<N: Float>: Model<N> {
  type TE: From<Self::E>;

  fn train(&mut self, inputs: Array4<N>, policies: Array3<N>, values: Array1<N>) -> Result<(), Self::TE>;

  fn save(&self) -> Result<(), Self::TE>;
}
