use std::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PointsVec<T>(pub Vec<T>);

#[cfg(not(feature = "unsafe"))]
impl<T> Index<usize> for PointsVec<T> {
  type Output = T;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

#[cfg(feature = "unsafe")]
impl<T> Index<usize> for PointsVec<T> {
  type Output = T;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    unsafe { self.0.get_unchecked(index) }
  }
}

#[cfg(not(feature = "unsafe"))]
impl<T> IndexMut<usize> for PointsVec<T> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.0[index]
  }
}

#[cfg(feature = "unsafe")]
impl<T> IndexMut<usize> for PointsVec<T> {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    unsafe { self.0.get_unchecked_mut(index) }
  }
}

impl<T> From<Vec<T>> for PointsVec<T> {
  #[inline]
  fn from(vec: Vec<T>) -> Self {
    PointsVec(vec)
  }
}

impl<T> From<PointsVec<T>> for Vec<T> {
  #[inline]
  fn from(vec: PointsVec<T>) -> Self {
    vec.0
  }
}
