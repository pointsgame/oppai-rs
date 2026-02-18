use either::Either;
use ndarray::{Array, Array2, Array3, Array4, Axis};
use num_traits::Float;

pub trait Model<N: Float> {
  type E;

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array2<N>), Self::E>;
}

pub trait TrainableModel<N: Float>: Model<N> + Sized {
  type TE: From<Self::E>;

  fn train(self, inputs: Array4<N>, policies: Array3<N>, values: Array2<N>) -> Result<Self, Self::TE>;
}

impl<T, E, N: Float> Model<N> for T
where
  T: Fn(Array4<N>) -> Result<(Array3<N>, Array2<N>), E>,
{
  type E = E;

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    self(inputs)
  }
}

impl<N: Float> Model<N> for () {
  type E = ();

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    let batch_size = inputs.len_of(Axis(0));
    let height = inputs.len_of(Axis(2));
    let width = inputs.len_of(Axis(3));
    let policy = N::one() / N::from(width * height).unwrap();
    let policies = Array::from_elem((batch_size, height, width), policy);
    let values = Array::from_elem((batch_size, 3), N::one() / (N::one() + N::one()));
    Ok((policies, values))
  }
}

impl<N: Float, A: Model<N>, B: Model<N>> Model<N> for Either<A, B> {
  type E = Either<A::E, B::E>;

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    match self {
      Either::Left(a) => a.predict(inputs).map_err(Either::Left),
      Either::Right(b) => b.predict(inputs).map_err(Either::Right),
    }
  }
}
