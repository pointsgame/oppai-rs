//! Attention over the cells of a board with a penalty on the distance between
//! the query's cell and the key's, computed without materializing the scores.
//!
//! What it computes is burn's attention with the mask of the cells off the
//! board and the bias `-slope_h * distance(i, j)`, `distance` being the
//! euclidean one between the cells in cells. Built from burn's operations, it
//! would hold the scores of all the heads of all the boards of a batch, which
//! on a full board takes gigabytes per layer, and training keeps them for the
//! backward pass. Burn's own memory efficient kernels take no bias, so on the
//! GPU this runs kernels of its own instead, which keep nothing but the output
//! and the logarithm of every query's softmax denominator and recompute the
//! rest in the backward pass, see [`cube`]. On the other backends it is
//! composed of burn's operations, see [`composed`], which also serves as the
//! reference the kernels are tested against.
//!
//! TODO: replace with burn's attention once its memory efficient kernels take
//! a bias, both forward and backward.

mod autodiff;
mod composed;
#[cfg(feature = "cube")]
mod cube;

use burn::{
  backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor},
  tensor::Tensor,
};

/// The attention of the distance penalty, as burn's backends' operations.
///
/// All of them take queries, keys and values `[batch, heads, cells,
/// channels]` of one float type, the cells being those of a board `width`
/// cells wide, row after row; `on_board`, `[batch, cells]` in single
/// precision, one for the cells on the board and zero for those off it, which
/// no query attends to; and `slopes`, `[heads]` in single precision, what
/// every head takes off the score per cell of distance. The scores are scaled
/// by `1 / sqrt(channels)`, as burn's attention does.
#[backend_extension(
  Autodiff,
  Cube: cfg(feature = "cube"),
  Flex: cfg(feature = "flex"),
  Fusion: cfg(feature = "fusion")
)]
pub trait DistanceAttentionBackend: Backend {
  /// The output, `[batch, heads, cells, channels]`, in the float type of the
  /// values.
  #[fusion(dtype = value, shape = value)]
  fn distance_attention(
    query: FloatTensor<Self>,
    key: FloatTensor<Self>,
    value: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    slopes: FloatTensor<Self>,
    width: usize,
  ) -> FloatTensor<Self> {
    Self::distance_attention_forward(query, key, value, on_board, slopes, width).0
  }

  /// The output, and the natural logarithm of every query's softmax
  /// denominator - the scores' log-sum-exp - `[batch, heads, cells]` in
  /// single precision, which the backward pass recomputes the attention from.
  #[fusion(meta = fusion_meta::forward)]
  fn distance_attention_forward(
    query: FloatTensor<Self>,
    key: FloatTensor<Self>,
    value: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    slopes: FloatTensor<Self>,
    width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>) {
    composed::forward::<Self>(query, key, value, on_board, slopes, width)
  }

  /// The gradients of the queries, the keys, the values and the slopes, from
  /// the gradient of the output, given the output and the log-sum-exp of
  /// [`distance_attention_forward`](Self::distance_attention_forward).
  #[allow(clippy::too_many_arguments)]
  #[fusion(meta = fusion_meta::backward)]
  fn distance_attention_backward(
    query: FloatTensor<Self>,
    key: FloatTensor<Self>,
    value: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    slopes: FloatTensor<Self>,
    output: FloatTensor<Self>,
    log_sum_exp: FloatTensor<Self>,
    grad: FloatTensor<Self>,
    width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
    composed::backward::<Self>(query, key, value, on_board, slopes, output, log_sum_exp, grad, width)
  }
}

#[cfg(feature = "flex")]
impl DistanceAttentionBackend for burn::backend::Flex {}

#[cfg(feature = "fusion")]
mod fusion_meta {
  use burn::{backend::fusion::custom::TensorSpec, tensor::DType};

  pub fn forward(
    query: &TensorSpec,
    _key: &TensorSpec,
    value: &TensorSpec,
    _on_board: &TensorSpec,
    _slopes: &TensorSpec,
    _width: &usize,
  ) -> (TensorSpec, TensorSpec) {
    let [batch, heads, cells, _] = query.shape.dims::<4>();
    (
      value.clone(),
      TensorSpec::new([batch, heads, cells].into(), DType::F32),
    )
  }

  #[allow(clippy::too_many_arguments)]
  pub fn backward(
    query: &TensorSpec,
    key: &TensorSpec,
    value: &TensorSpec,
    _on_board: &TensorSpec,
    slopes: &TensorSpec,
    _output: &TensorSpec,
    _log_sum_exp: &TensorSpec,
    _grad: &TensorSpec,
    _width: &usize,
  ) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
    (query.clone(), key.clone(), value.clone(), slopes.clone())
  }
}

/// See [`DistanceAttentionBackend`]: `query`, `key` and `value` are `[batch,
/// heads, cells, channels]`, `on_board` is `[batch, cells]` and `slopes` is
/// `[heads]`, both in single precision. Differentiable in all but `on_board`.
pub fn distance_attention(
  query: Tensor<4>,
  key: Tensor<4>,
  value: Tensor<4>,
  on_board: Tensor<2>,
  slopes: Tensor<1>,
  width: usize,
) -> Tensor<4> {
  Tensor::from_dispatch(Dispatch::distance_attention(
    query.into_dispatch(),
    key.into_dispatch(),
    value.into_dispatch(),
    on_board.into_dispatch(),
    slopes.into_dispatch(),
    width,
  ))
}

/// The euclidean distance between every two cells of a board `width` cells
/// wide, in cells, `cells * cells` of them, the query's cell first.
fn distances(cells: usize, width: usize) -> Vec<f32> {
  (0..cells * cells)
    .map(|index| {
      let (query, key) = (index / cells, index % cells);
      let rows = (query / width) as f32 - (key / width) as f32;
      let columns = (query % width) as f32 - (key % width) as f32;
      (rows * rows + columns * columns).sqrt()
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::{distance_attention, distances};
  use burn::tensor::{
    Device, Distribution, FloatDType, Tensor, TensorData,
    module::attention,
    ops::AttentionModuleOptions,
  };

  /// What [`distance_attention`] computes, as burn's attention with the mask
  /// and the bias materialized.
  fn reference(
    query: Tensor<4>,
    key: Tensor<4>,
    value: Tensor<4>,
    on_board: Tensor<2>,
    slopes: Tensor<1>,
    width: usize,
  ) -> Tensor<4> {
    let [batch, heads, cells, _] = query.dims();
    let device = query.device();
    let distances = Tensor::<4>::from_data(
      TensorData::new(distances(cells, width), [1, 1, cells, cells]),
      &device,
    );
    let bias = (slopes.reshape([1, heads, 1, 1]) * distances)
      .neg()
      .cast(query.dtype())
      .expand([batch, heads, cells, cells]);
    let mask = on_board
      .reshape([batch, 1, 1, cells])
      .lower_elem(0.5)
      .expand([batch, heads, cells, cells]);
    attention(
      query,
      key,
      value,
      Some(mask),
      Some(bias),
      None,
      AttentionModuleOptions::default(),
    )
  }

  /// `[batch, cells]`: a `height x width` grid holding the whole board in
  /// the first sample, and a `rows x columns` one in its corner in the
  /// second.
  fn on_board(device: &Device, (height, width): (usize, usize), (rows, columns): (usize, usize)) -> Tensor<2> {
    let values = (0..2 * height * width)
      .map(|index| {
        let cell = index % (height * width);
        let full = index < height * width;
        (full || (cell / width < rows && cell % width < columns)) as u8 as f32
      })
      .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [2, height * width]), device)
  }

  fn max_difference<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) -> f32 {
    let scale = expected.clone().abs().max().into_scalar::<f32>().max(1e-3);
    (actual - expected).abs().max().into_scalar::<f32>() / scale
  }

  /// The output and the gradients of the queries, the keys, the values and
  /// the slopes on `device` agree with burn's attention, for boards that take
  /// several blocks of cells, with cells off the board.
  fn agrees_with_attention(device: Device, channels: usize, (height, width): (usize, usize), dtype: FloatDType, tolerance: f32) {
    let device = device.autodiff();
    let (batch, heads) = (2, 3);
    let cells = height * width;
    let shape = [batch, heads, cells, channels];
    let random = || Tensor::<4>::random(shape, Distribution::Normal(0.0, 1.0), &device).cast(dtype);
    let (query, key, value) = (random(), random(), random());
    let slopes = Tensor::<1>::from_floats([0.05, 0.3, 1.5], &device);
    let on_board = on_board(&device, (height, width), (height - 2, width - 3));
    let weights = Tensor::<4>::random(shape, Distribution::Normal(0.0, 1.0), &device);

    let run = |custom: bool| {
      let (query, key, value, slopes) = (
        query.clone().detach().require_grad(),
        key.clone().detach().require_grad(),
        value.clone().detach().require_grad(),
        slopes.clone().detach().require_grad(),
      );
      let attend = if custom { distance_attention } else { reference };
      let output = attend(
        query.clone(),
        key.clone(),
        value.clone(),
        on_board.clone(),
        slopes.clone(),
        width,
      );
      let grads = (output.clone().cast(FloatDType::F32) * weights.clone()).sum().backward();
      let grad = |tensor: &Tensor<4>| tensor.grad(&grads).unwrap().cast(FloatDType::F32);
      (
        [
          output.cast(FloatDType::F32).detach(),
          grad(&query),
          grad(&key),
          grad(&value),
        ],
        slopes.grad(&grads).unwrap(),
      )
    };
    let (custom, custom_slopes) = run(true);
    let (expected, expected_slopes) = run(false);
    for (name, (custom, expected)) in ["output", "query", "key", "value"].iter().zip(custom.into_iter().zip(expected)) {
      let difference = max_difference(custom, expected);
      assert!(difference < tolerance, "{name}: {difference}");
    }
    let difference = max_difference(custom_slopes, expected_slopes);
    assert!(difference < tolerance, "slopes: {difference}");
  }

  #[cfg(feature = "flex")]
  #[test]
  fn composed_agrees_with_attention() {
    agrees_with_attention(Device::flex(), 8, (9, 10), FloatDType::F32, 1e-4);
  }

  #[cfg(feature = "vulkan")]
  #[test]
  fn kernels_agree_with_attention_vulkan() {
    let device = crate::device::parse_device("vulkan").unwrap();
    agrees_with_attention(device.clone(), 8, (9, 10), FloatDType::F32, 1e-4);
    agrees_with_attention(device.clone(), 8, (13, 11), FloatDType::F32, 1e-4);
    agrees_with_attention(device.clone(), 32, (7, 11), FloatDType::F32, 1e-4);
    agrees_with_attention(device, 32, (7, 11), FloatDType::F16, 2e-2);
  }

  #[cfg(feature = "rocm")]
  #[test]
  fn kernels_agree_with_attention_rocm() {
    let device = crate::device::parse_device("rocm").unwrap();
    agrees_with_attention(device.clone(), 8, (9, 10), FloatDType::F32, 1e-4);
    agrees_with_attention(device.clone(), 8, (13, 11), FloatDType::F32, 1e-4);
    agrees_with_attention(device.clone(), 32, (7, 11), FloatDType::F32, 1e-4);
    agrees_with_attention(device.clone(), 32, (7, 11), FloatDType::F16, 2e-2);
    agrees_with_attention(device, 32, (7, 11), FloatDType::BF16, 5e-2);
  }
}
