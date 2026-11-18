//! The depthwise convolution and the gated linear unit of a feed-forward
//! layer, fused into kernels of their own on the GPU backends.
//!
//! What it computes, from the hidden values `[batch, cells, channels]` of a
//! `height x width` grid of cells, the activations then their gates: every
//! hidden value plus the `3x3` depthwise convolution of those of the cells on
//! the board around it - `on_board`, `[batch, cells]` in single precision,
//! one on the board and zero off it, takes those off it as zero - and then
//! the SiLU of every such activation times its gate, `[batch, cells,
//! channels / 2]`. That is
//! [`DepthwiseConv::forward`](crate::transformer::DepthwiseConv::forward)
//! with its residual added, followed by
//! [`glu`](crate::transformer::glu).
//!
//! Composed of burn's operations, on the GPU it takes a dozen kernels, each a
//! pass over the hidden values in memory, which the slices of the gated linear
//! unit keep burn's fusion from merging, while the work itself is negligible.
//! Fused, the forward pass is one read of the hidden values and one write of
//! the output; see [`cube`].
//!
//! Only the GPU backends run it, see [`serves`]; everywhere else the feed-forward
//! layer composes it of burn's operations, which is also what the kernels are
//! tested against.
//!
//! To remove it once burn fuses that composition by itself: delete this
//! module and its `mod` line in `lib.rs`, and `DepthwiseConv::glu` with its
//! call in `FeedForward::forward` in `transformer.rs`, which leaves the
//! composition in place of it.

mod autodiff;
#[cfg(feature = "cube")]
mod cube;

use burn::{
  backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor},
  tensor::{Device, Tensor},
};

/// The fused depthwise convolution and gated linear unit, as burn's backends'
/// operations; see the [module](self) for what they compute.
#[backend_extension(Autodiff, Cube: cfg(feature = "cube"), Fusion: cfg(feature = "fusion"))]
pub trait ConvGluBackend: Backend {
  /// The output, `[batch, cells, channels / 2]`, in the float type of
  /// `hidden`.
  #[fusion(meta = fusion_meta::forward)]
  fn conv_glu(
    hidden: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    weight: FloatTensor<Self>,
    height: usize,
    width: usize,
  ) -> FloatTensor<Self>;

  /// The gradients of the hidden values and of the filters from that of the
  /// output.
  #[fusion(meta = fusion_meta::backward)]
  fn conv_glu_backward(
    hidden: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    weight: FloatTensor<Self>,
    grad: FloatTensor<Self>,
    height: usize,
    width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>);
}

#[cfg(feature = "fusion")]
mod fusion_meta {
  use burn::backend::fusion::custom::TensorSpec;

  pub fn forward(
    hidden: &TensorSpec,
    _on_board: &TensorSpec,
    _weight: &TensorSpec,
    _height: &usize,
    _width: &usize,
  ) -> TensorSpec {
    let [batch, cells, channels] = hidden.shape.dims::<3>();
    TensorSpec::new([batch, cells, channels / 2].into(), hidden.dtype)
  }

  pub fn backward(
    hidden: &TensorSpec,
    _on_board: &TensorSpec,
    weight: &TensorSpec,
    _grad: &TensorSpec,
    _height: &usize,
    _width: &usize,
  ) -> (TensorSpec, TensorSpec) {
    (hidden.clone(), weight.clone())
  }
}

/// Whether [`conv_glu`] runs on `device`: whether it is one of a GPU backend.
pub fn serves(device: &Device) -> bool {
  #[cfg(feature = "cube")]
  return device.capabilities().is_some();
  #[cfg(not(feature = "cube"))]
  {
    let _ = device;
    false
  }
}

/// See the [module](self): `hidden` is `[batch, height * width, channels]`,
/// `on_board` is `[batch, height * width]` in single precision and `weight`,
/// the filters, is `[channels, 1, 3, 3]`. Differentiable in `hidden` and
/// `weight`. Only on the devices it [`serves`].
pub fn conv_glu(hidden: Tensor<3>, on_board: Tensor<2>, weight: Tensor<4>, height: usize, width: usize) -> Tensor<3> {
  Tensor::from_dispatch(Dispatch::conv_glu(
    hidden.into_dispatch(),
    on_board.into_dispatch(),
    weight.into_dispatch(),
    height,
    width,
  ))
}

#[cfg(test)]
mod tests {
  use super::conv_glu;
  use crate::transformer::glu;
  use burn::tensor::{Device, Distribution, FloatDType, Tensor, TensorData, module::conv2d, ops::ConvOptions};

  /// What [`conv_glu`] computes, composed of burn's operations.
  fn reference(hidden: Tensor<3>, on_board: Tensor<2>, weight: Tensor<4>, height: usize, width: usize) -> Tensor<3> {
    let [batch, cells, channels] = hidden.dims();
    let masked = hidden.clone() * on_board.reshape([batch, cells, 1]).cast(hidden.dtype());
    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], channels);
    let convolved = conv2d(
      masked.reshape([batch, height, width, channels]).permute([0, 3, 1, 2]),
      weight,
      None,
      options,
    )
    .permute([0, 2, 3, 1])
    .reshape([batch, cells, channels]);
    glu(hidden + convolved)
  }

  /// `[3, cells]`: the whole `height x width` grid on the board in the first
  /// board, a `rows x columns` corner of it in the second, and every other
  /// cell in the third.
  fn on_board(device: &Device, (height, width): (usize, usize), (rows, columns): (usize, usize)) -> Tensor<2> {
    let cells = height * width;
    let values = (0..3 * cells)
      .map(|index| {
        let cell = index % cells;
        let on = match index / cells {
          0 => true,
          1 => cell / width < rows && cell % width < columns,
          _ => cell % 2 == 0,
        };
        on as u8 as f32
      })
      .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [3, cells]), device)
  }

  fn max_difference<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) -> f32 {
    let scale = expected.clone().abs().max().into_scalar::<f32>().max(1e-3);
    (actual - expected).abs().max().into_scalar::<f32>() / scale
  }

  /// How far apart, relative to the largest of them, values in `dtype` may be
  /// from ones computed in single precision.
  fn tolerance(dtype: FloatDType) -> f32 {
    match dtype {
      FloatDType::F16 => 2e-2,
      FloatDType::BF16 => 5e-2,
      _ => 1e-4,
    }
  }

  /// The output and the gradients of the hidden values and of the filters on
  /// `device` agree with the [`reference`] in single precision, for a grid
  /// that takes several chunks of cells, a number of channels that is not
  /// one of whole cubes, and cells off the board; with the hidden values in
  /// `dtype` and the filters in `weight_dtype`, each result as precise as its
  /// type.
  fn agrees_with_reference(device: Device, dtype: FloatDType, weight_dtype: FloatDType) {
    let device = device.autodiff();
    let (height, width, channels) = (9, 11, 2 * 40);
    let cells = height * width;
    let hidden = Tensor::<3>::random([3, cells, channels], Distribution::Normal(0.0, 1.0), &device).cast(dtype);
    let weight = Tensor::<4>::random([channels, 1, 3, 3], Distribution::Normal(0.0, 0.5), &device).cast(weight_dtype);
    let on_board = on_board(&device, (height, width), (height - 2, width - 3));
    let weights = Tensor::<3>::random([3, cells, channels / 2], Distribution::Normal(0.0, 1.0), &device);

    let run = |fused: bool| {
      let (hidden, weight) = if fused {
        (hidden.clone(), weight.clone())
      } else {
        (
          hidden.clone().cast(FloatDType::F32),
          weight.clone().cast(FloatDType::F32),
        )
      };
      let (hidden, weight) = (hidden.detach().require_grad(), weight.detach().require_grad());
      let compute = if fused { conv_glu } else { reference };
      let output = compute(hidden.clone(), on_board.clone(), weight.clone(), height, width);
      assert_eq!(output.dtype(), hidden.dtype());
      let grads = (output.clone().cast(FloatDType::F32) * weights.clone())
        .sum()
        .backward();
      let grad_hidden = hidden.grad(&grads).unwrap();
      let grad_weight = weight.grad(&grads).unwrap();
      assert_eq!(grad_hidden.dtype(), hidden.dtype());
      assert_eq!(grad_weight.dtype(), weight.dtype());
      (
        output.cast(FloatDType::F32).detach(),
        grad_hidden.cast(FloatDType::F32),
        grad_weight.cast(FloatDType::F32),
      )
    };
    let (output, grad_hidden, grad_weight) = run(true);
    let (expected_output, expected_hidden, expected_weight) = run(false);
    for (name, difference, tolerance) in [
      ("output", max_difference(output, expected_output), tolerance(dtype)),
      ("hidden", max_difference(grad_hidden, expected_hidden), tolerance(dtype)),
      (
        "weight",
        max_difference(grad_weight, expected_weight),
        tolerance(weight_dtype),
      ),
    ] {
      assert!(difference < tolerance, "{name}: {difference}");
    }
  }

  #[cfg(feature = "flex")]
  #[test]
  fn serves_no_cpu() {
    assert!(!super::serves(&Device::flex()));
  }

  #[cfg(feature = "vulkan")]
  #[test]
  fn kernels_agree_with_reference_vulkan() {
    let device = crate::device::parse_device("vulkan").unwrap();
    assert!(super::serves(&device));
    agrees_with_reference(device.clone(), FloatDType::F32, FloatDType::F32);
    agrees_with_reference(device, FloatDType::F16, FloatDType::F16);
  }

  #[cfg(feature = "rocm")]
  #[test]
  fn kernels_agree_with_reference_rocm() {
    let device = crate::device::parse_device("rocm").unwrap();
    assert!(super::serves(&device));
    agrees_with_reference(device.clone(), FloatDType::F32, FloatDType::F32);
    // Mixed precision: single precision hidden values, brain floating point
    // filters.
    agrees_with_reference(device.clone(), FloatDType::F32, FloatDType::BF16);
    agrees_with_reference(device.clone(), FloatDType::F16, FloatDType::F16);
    agrees_with_reference(device, FloatDType::BF16, FloatDType::BF16);
  }
}
