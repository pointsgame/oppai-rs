use either::Either;
use ndarray::{Array, Array2, Array3, Array4, Axis};
use num_traits::Float;

#[allow(async_fn_in_trait)]
pub trait Model<N: Float> {
  type E;

  /// Whether the third value column holds a real predicted error.
  ///
  /// A model without that head leaves the column at zero, and zero is not a
  /// neutral value there: it claims the evaluation is perfectly certain, which
  /// earns the largest weight the search hands out, not an average one. Such a
  /// model says so here instead and has all of its evaluations count equally.
  fn predicts_uncertainty(&self) -> bool {
    true
  }

  /// Returns the policies and the values for a batch of positions. The values
  /// have 3 columns: the win and loss probabilities, and the predicted
  /// short-term error (standard deviation) of the value - how uncertain the
  /// value estimate is. A model with no estimate to offer leaves it at 0 and
  /// reports [`Model::predicts_uncertainty`] as false.
  async fn predict(&mut self, inputs: Array4<N>, global: Array2<N>) -> Result<(Array3<N>, Array2<N>), Self::E>;
}

pub trait TrainableModel<N: Float>: Model<N> + Sized {
  type TE: From<Self::E>;

  #[allow(clippy::too_many_arguments)]
  fn train(
    self,
    inputs: Array4<N>,
    global: Array2<N>,
    policies: Array3<N>,
    opponent_policies: Array3<N>,
    values: Array2<N>,
    td_values: Array3<N>,
    scores: Array2<N>,
    captured: Array4<N>,
    learning_rate: f64,
  ) -> Result<Self, Self::TE>;
}

impl<T, E, N: Float> Model<N> for T
where
  T: Fn(Array4<N>, Array2<N>) -> Result<(Array3<N>, Array2<N>), E>,
{
  type E = E;

  async fn predict(&mut self, inputs: Array4<N>, global: Array2<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    self(inputs, global)
  }
}

impl<N: Float> Model<N> for () {
  type E = ();

  fn predicts_uncertainty(&self) -> bool {
    false
  }

  async fn predict(&mut self, inputs: Array4<N>, _: Array2<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    let batch_size = inputs.len_of(Axis(0));
    let height = inputs.len_of(Axis(2));
    let width = inputs.len_of(Axis(3));
    let policy = N::one() / N::from(width * height).unwrap();
    let policies = Array::from_elem((batch_size, height, width), policy);
    let mut values = Array::zeros((batch_size, 3));
    values
      .slice_mut(ndarray::s![.., 0..2])
      .fill(N::one() / (N::one() + N::one()));
    Ok((policies, values))
  }
}

impl<N: Float, A: Model<N>, B: Model<N>> Model<N> for Either<A, B> {
  type E = Either<A::E, B::E>;

  fn predicts_uncertainty(&self) -> bool {
    match self {
      Either::Left(a) => a.predicts_uncertainty(),
      Either::Right(b) => b.predicts_uncertainty(),
    }
  }

  async fn predict(&mut self, inputs: Array4<N>, global: Array2<N>) -> Result<(Array3<N>, Array2<N>), Self::E> {
    match self {
      Either::Left(a) => a.predict(inputs, global).await.map_err(Either::Left),
      Either::Right(b) => b.predict(inputs, global).await.map_err(Either::Right),
    }
  }
}
