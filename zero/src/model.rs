use ndarray::{Array, Array1, Array3, Array4, Axis};
use num_traits::Float;

pub trait Model<N: Float> {
  type E;

  fn predict(&self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E>;
}

pub trait TrainableModel<N: Float>: Model<N> + Sized {
  type TE: From<Self::E>;

  fn train(self, inputs: Array4<N>, policies: Array3<N>, values: Array1<N>) -> Result<Self, Self::TE>;
}

impl<T, E, N: Float> Model<N> for T
where
  T: Fn(Array4<N>) -> Result<(Array3<N>, Array1<N>), E>,
{
  type E = E;

  fn predict(&self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E> {
    self(inputs)
  }
}

impl<N: Float> Model<N> for () {
  type E = ();

  fn predict(&self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E> {
    let batch_size = inputs.len_of(Axis(0));
    let height = inputs.len_of(Axis(2));
    let width = inputs.len_of(Axis(3));
    let policy = N::one() / N::from(width * height).unwrap();
    let policies = Array::from_elem((batch_size, height, width), policy);
    let values = Array::from_elem(batch_size, N::one() / (N::one() + N::one()));
    Ok((policies, values))
  }
}
