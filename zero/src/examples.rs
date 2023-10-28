use ndarray::{Array, Array1, Array2, Array3, Array4, Axis};
use serde::{Deserialize, Serialize};
use std::ops::Add;

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
  pub fn inputs(&self) -> Array4<N> {
    ndarray::stack(
      Axis(0),
      self.inputs.iter().map(|i| i.view()).collect::<Vec<_>>().as_slice(),
    )
    .unwrap()
  }

  pub fn policies(&self) -> Array3<N> {
    ndarray::stack(
      Axis(0),
      self.policies.iter().map(|p| p.view()).collect::<Vec<_>>().as_slice(),
    )
    .unwrap()
  }

  pub fn values(&self) -> Array1<N> {
    Array::from(self.values.clone())
  }

  pub fn clear(&mut self) {
    self.inputs.clear();
    self.policies.clear();
    self.values.clear();
  }
}
