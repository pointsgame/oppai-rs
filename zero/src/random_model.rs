use std::{cmp::Ordering, iter};

use ndarray::{Array1, Array3, Array4};
use rand::Rng;
use rand_distr::StandardNormal;

use crate::model::Model;

pub struct RandomModel<R>(R);

macro_rules! model_impl {
  ($t:ty) => {
    impl<R: Rng + Clone> Model<$t> for RandomModel<R> {
      type E = ();

      fn predict(&self, inputs: Array4<$t>) -> Result<(Array3<$t>, Array1<$t>), Self::E> {
        let (batch, _, height, width) = inputs.dim();
        let length = height * width;
        let mut rng = self.0.clone();

        let values = Array1::from_shape_simple_fn(batch, || (rng.random::<$t>() * 2.0 - 1.0).powi(3));

        let mut policies = iter::repeat_with(|| rng.sample(StandardNormal))
          .take(batch * height * width)
          .collect::<Vec<$t>>();
        for i in 0..batch {
          let slice = &mut policies[i * length..(i + 1) * length];
          let max = *slice
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .ok_or(())?;
          for p in slice.iter_mut() {
            *p = (*p - max).exp();
          }
          let sum: $t = slice.iter().sum();
          for p in slice {
            *p /= sum;
          }
        }
        let policies = Array3::from_shape_vec((batch, height, width), policies).map_err(|_| ())?;

        Ok((policies, values))
      }
    }
  };
}

model_impl!(f32);
model_impl!(f64);
