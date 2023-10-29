use either::Either;
use ndarray::{Array, Array1, Array2, Array3, Array4, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::{iter, ops::Add};

#[derive(Clone, Serialize, Deserialize)]
pub struct Examples<N> {
  pub inputs: Vec<Array3<N>>,
  pub policies: Vec<Array2<N>>,
  pub values: Vec<N>,
}

impl<N> Default for Examples<N> {
  fn default() -> Self {
    Self {
      inputs: Default::default(),
      policies: Default::default(),
      values: Default::default(),
    }
  }
}

impl<N> Add for Examples<N> {
  type Output = Self;
  fn add(mut self, rhs: Self) -> Self::Output {
    self.inputs.extend(rhs.inputs);
    self.policies.extend(rhs.policies);
    self.values.extend(rhs.values);
    self
  }
}

impl<N: Clone> Examples<N> {
  #[inline]
  fn inputs_array(inputs: &[Array3<N>]) -> Array4<N> {
    ndarray::stack(Axis(0), inputs.iter().map(|i| i.view()).collect::<Vec<_>>().as_slice()).unwrap()
  }

  #[inline]
  fn policies_array(policies: &[Array2<N>]) -> Array3<N> {
    ndarray::stack(
      Axis(0),
      policies.iter().map(|p| p.view()).collect::<Vec<_>>().as_slice(),
    )
    .unwrap()
  }

  #[inline]
  fn values_array(values: &[N]) -> Array1<N> {
    Array::from(values.to_vec())
  }

  #[inline]
  pub fn inputs(&self) -> Array4<N> {
    Examples::inputs_array(&self.inputs)
  }

  #[inline]
  pub fn policies(&self) -> Array3<N> {
    Examples::policies_array(&self.policies)
  }

  #[inline]
  pub fn values(&self) -> Array1<N> {
    Examples::values_array(&self.values)
  }

  #[inline]
  pub fn clear(&mut self) {
    self.inputs.clear();
    self.policies.clear();
    self.values.clear();
  }

  #[inline]
  pub fn len(&self) -> usize {
    self.values.len()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.values.is_empty()
  }

  pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
    let len = self.len();
    for i in 0..len {
      let j = rng.gen_range(0..len);
      self.inputs.swap(i, j);
      self.policies.swap(i, j);
      self.values.swap(i, j);
    }
  }

  pub fn batches(&self, size: usize) -> impl Iterator<Item = (Array4<N>, Array3<N>, Array1<N>)> + '_ {
    if self.len() <= size {
      Either::Left(iter::once((self.inputs(), self.policies(), self.values())))
    } else {
      Either::Right(
        itertools::izip!(
          self.inputs.chunks(size),
          self.policies.chunks(size),
          self.values.chunks(size)
        )
        .map(|(inputs, policies, values)| {
          (
            Examples::inputs_array(inputs),
            Examples::policies_array(policies),
            Examples::values_array(values),
          )
        })
        .take(self.len() / size),
      )
    }
  }
}
