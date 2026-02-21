use either::Either;
use ndarray::{Array1, Array2, Array3, Array4, Axis};
use rand::Rng;
use std::{iter, ops::Add};

#[derive(Clone, Debug)]
pub struct Examples<N> {
  pub inputs: Vec<Array3<N>>,
  pub policies: Vec<Array2<N>>,
  pub values: Vec<Array1<N>>,
  pub scores: Vec<Array2<N>>,
}

impl<N> Default for Examples<N> {
  fn default() -> Self {
    Self {
      inputs: Default::default(),
      policies: Default::default(),
      values: Default::default(),
      scores: Default::default(),
    }
  }
}

impl<N> Add for Examples<N> {
  type Output = Self;
  fn add(mut self, rhs: Self) -> Self::Output {
    self.inputs.extend(rhs.inputs);
    self.policies.extend(rhs.policies);
    self.values.extend(rhs.values);
    self.scores.extend(rhs.scores);
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
  fn values_array(values: &[Array1<N>]) -> Array2<N> {
    ndarray::stack(Axis(0), values.iter().map(|v| v.view()).collect::<Vec<_>>().as_slice()).unwrap()
  }

  #[inline]
  pub fn scores_array(scores: &[Array2<N>]) -> Array3<N> {
    ndarray::stack(Axis(0), scores.iter().map(|s| s.view()).collect::<Vec<_>>().as_slice()).unwrap()
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
  pub fn values(&self) -> Array2<N> {
    Examples::values_array(&self.values)
  }

  #[inline]
  pub fn scores(&self) -> Array3<N> {
    Examples::scores_array(&self.scores)
  }

  #[inline]
  pub fn clear(&mut self) {
    self.inputs.clear();
    self.policies.clear();
    self.values.clear();
    self.scores.clear();
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
      let j = rng.random_range(0..len);
      self.inputs.swap(i, j);
      self.policies.swap(i, j);
      self.values.swap(i, j);
      self.scores.swap(i, j);
    }
  }

  pub fn batches(&self, size: usize) -> impl Iterator<Item = (Array4<N>, Array3<N>, Array2<N>, Array3<N>)> + '_ {
    if self.len() <= size {
      Either::Left(iter::once((
        self.inputs(),
        self.policies(),
        self.values(),
        self.scores(),
      )))
    } else {
      Either::Right(
        itertools::izip!(
          self.inputs.chunks(size),
          self.policies.chunks(size),
          self.values.chunks(size),
          self.scores.chunks(size),
        )
        .map(|(inputs, policies, values, scores)| {
          (
            Examples::inputs_array(inputs),
            Examples::policies_array(policies),
            Examples::values_array(values),
            Examples::scores_array(scores),
          )
        })
        .take(self.len() / size),
      )
    }
  }
}
