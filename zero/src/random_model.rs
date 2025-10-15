use std::{
  cmp::Ordering,
  iter::{self, Sum},
};

use ndarray::{Array1, Array3, Array4};
use num_traits::{Float, One};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

use crate::model::Model;

pub struct RandomModel<R>(pub R);

impl<N: Float + SampleUniform + Sum + Clone + One, R: Rng> Model<N> for RandomModel<R>
where
  StandardNormal: Distribution<N>,
{
  type E = ();

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E> {
    let (batch, _, height, width) = inputs.dim();
    let length = height * width;

    let values = Array1::from_shape_simple_fn(batch, || {
      (self.0.random_range::<N, _>(N::one().neg()..=N::one())).powi(3)
    });

    let mut policies = iter::repeat_with(|| self.0.sample(StandardNormal))
      .take(batch * height * width)
      .collect::<Vec<N>>();
    for i in 0..batch {
      let slice = &mut policies[i * length..(i + 1) * length];
      let max = *slice
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .ok_or(())?;
      for p in slice.iter_mut() {
        *p = (*p - max).exp();
      }
      let sum: N = slice.iter().cloned().sum();
      for p in slice {
        *p = *p / sum;
      }
    }
    let policies = Array3::from_shape_vec((batch, height, width), policies).map_err(|_| ())?;

    Ok((policies, values))
  }
}
