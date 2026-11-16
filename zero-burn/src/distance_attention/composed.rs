//! The distance attention composed of burn's operations, in single precision,
//! materializing the scores: for the backends without kernels of their own,
//! and as the reference for those.

use super::distances;
use burn::{
  backend::{Backend, TensorMetadata, tensor::FloatTensor},
  tensor::{FloatDType, Scalar, Shape, TensorData},
};

/// What a score of `on_board`'s zeros gets added: far enough below any other
/// that its exponential is zero, and finite, so that subtracting the maximum
/// of a row does not produce a NaN.
const OFF_BOARD: f64 = -1e30;

fn f32<B: Backend>(tensor: FloatTensor<B>) -> FloatTensor<B> {
  B::float_cast(tensor, FloatDType::F32)
}

fn cast<B: Backend>(tensor: FloatTensor<B>, like: &FloatTensor<B>) -> FloatTensor<B> {
  B::float_cast(tensor, like.dtype().into())
}

fn transpose<B: Backend>(tensor: FloatTensor<B>) -> FloatTensor<B> {
  B::float_swap_dims(tensor, 2, 3)
}

/// `1 / sqrt(channels)`, as burn's attention scales its scores.
pub(super) fn scale(channels: usize) -> f64 {
  1.0 / (channels as f64).sqrt()
}

/// The distances between the cells, `[1, 1, cells, cells]`, and the scores
/// of every query and key, `[batch, heads, cells, cells]`, in single
/// precision.
fn scores<B: Backend>(
  query: FloatTensor<B>,
  key: FloatTensor<B>,
  on_board: FloatTensor<B>,
  slopes: FloatTensor<B>,
  width: usize,
) -> (FloatTensor<B>, FloatTensor<B>) {
  let [batch, heads, cells, channels] = query.shape().dims::<4>();
  let device = query.device();
  let query = B::float_mul_scalar(f32::<B>(query), Scalar::Float(scale(channels)));
  let scores = B::float_matmul(query, transpose::<B>(f32::<B>(key)));
  let distances = B::float_from_data(
    TensorData::new(distances(cells, width), [1, 1, cells, cells]),
    &device,
  );
  let slopes = B::float_reshape(f32::<B>(slopes), Shape::new([1, heads, 1, 1]));
  let scores = B::float_sub(scores, B::float_mul(slopes, distances.clone()));
  let off_board = B::float_mul_scalar(
    B::float_add_scalar(f32::<B>(on_board), Scalar::Float(-1.0)),
    Scalar::Float(-OFF_BOARD),
  );
  let off_board = B::float_reshape(off_board, Shape::new([batch, 1, 1, cells]));
  (distances, B::float_add(scores, off_board))
}

pub fn forward<B: Backend>(
  query: FloatTensor<B>,
  key: FloatTensor<B>,
  value: FloatTensor<B>,
  on_board: FloatTensor<B>,
  slopes: FloatTensor<B>,
  width: usize,
) -> (FloatTensor<B>, FloatTensor<B>) {
  let [batch, heads, cells, _] = query.shape().dims::<4>();
  let (_, scores) = scores::<B>(query, key, on_board, slopes, width);
  let max = B::float_max_dim(scores.clone(), 3);
  let exp = B::float_exp(B::float_sub(scores, max.clone()));
  let sum = B::float_sum_dim(exp.clone(), 3);
  let output = B::float_matmul(B::float_div(exp, sum.clone()), f32::<B>(value.clone()));
  let log_sum_exp = B::float_reshape(
    B::float_add(max, B::float_log(sum)),
    Shape::new([batch, heads, cells]),
  );
  (cast::<B>(output, &value), log_sum_exp)
}

#[allow(clippy::too_many_arguments)]
pub fn backward<B: Backend>(
  query: FloatTensor<B>,
  key: FloatTensor<B>,
  value: FloatTensor<B>,
  on_board: FloatTensor<B>,
  slopes: FloatTensor<B>,
  output: FloatTensor<B>,
  log_sum_exp: FloatTensor<B>,
  grad: FloatTensor<B>,
  width: usize,
) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
  let [batch, heads, cells, channels] = query.shape().dims::<4>();
  let scale = Scalar::Float(scale(channels));
  let (distances, scores) = scores::<B>(query.clone(), key.clone(), on_board, slopes.clone(), width);
  let log_sum_exp = B::float_reshape(log_sum_exp, Shape::new([batch, heads, cells, 1]));
  let probabilities = B::float_exp(B::float_sub(scores, log_sum_exp));
  let grad = f32::<B>(grad);
  // The gradient of the softmax: `p * (dp - sum(p * dp))`, the sum being
  // that of the output and the output's gradient.
  let output_dot_grad = B::float_sum_dim(B::float_mul(f32::<B>(output), grad.clone()), 3);
  let grad_probabilities = B::float_matmul(grad.clone(), transpose::<B>(f32::<B>(value.clone())));
  let grad_scores = B::float_mul(
    probabilities.clone(),
    B::float_sub(grad_probabilities, output_dot_grad),
  );
  let grad_value = B::float_matmul(transpose::<B>(probabilities), grad);
  let grad_query = B::float_mul_scalar(
    B::float_matmul(grad_scores.clone(), f32::<B>(key.clone())),
    scale,
  );
  let grad_key = B::float_mul_scalar(
    B::float_matmul(transpose::<B>(grad_scores.clone()), f32::<B>(query.clone())),
    scale,
  );
  // A score has the slope's distance taken off.
  let grad_slopes = B::float_neg(B::float_mul(grad_scores, distances));
  let grad_slopes = B::float_sum_dim(B::float_sum_dim(B::float_sum_dim(grad_slopes, 3), 2), 0);
  let grad_slopes = cast::<B>(
    B::float_reshape(grad_slopes, Shape::new([heads])),
    &slopes,
  );
  (
    cast::<B>(grad_query, &query),
    cast::<B>(grad_key, &key),
    cast::<B>(grad_value, &value),
    grad_slopes,
  )
}
