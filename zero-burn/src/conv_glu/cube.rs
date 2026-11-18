//! The kernels of [`ConvGluBackend`] for the GPU backends.
//!
//! They use nothing but global memory and registers, with no shared memory,
//! plane operations or atomics, so they run on every CubeCL runtime. A unit
//! takes one channel along one row of a board, the units of a cube
//! consecutive channels, so that every read is coalesced. It slides the `3x3`
//! window of the convolution along the row in its registers, reading only the
//! column of three cells that enters it at every step. Everything is computed
//! in single precision.
//!
//! The backward pass recomputes the convolution instead of keeping it from the
//! forward pass, in three kernels: one for the gradient of the convolution's
//! output - the mixed values - kept in single precision, one for the
//! gradient of the hidden values, which gathers it from the neighbors, and one
//! for the filters', which sums it over a chunk of rows per unit; the chunks
//! are added up afterwards, so that the sum does not depend on the order the
//! units run in.

use super::ConvGluBackend;
use burn::{
  backend::{cubecl::dtype_to_storage_type, ops::FloatTensorOps, tensor::FloatTensor},
  tensor::{DType, FloatDType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// Units per cube, each taking a channel.
const UNITS: usize = 64;

/// Rows and columns of the filter.
const TAPS: usize = 3;

/// Rows of the boards a unit of [`grad_weight_kernel`] sums the gradient of
/// its channel's filter over.
const CHUNK_ROWS: usize = 8;

/// The filter of `channel`, `[row, column]`.
#[cube]
fn load_filter(weight: &Tensor<f32>, channel: usize) -> Array<f32> {
  let mut filter = Array::<f32>::new(TAPS * TAPS);
  #[unroll]
  for tap in 0..TAPS * TAPS {
    filter[tap] = weight[channel * TAPS * TAPS + tap];
  }
  filter
}

/// Loads into the column `slot` of `window`, `[column, row]`, the values of
/// `channel` of the cells of `column` in the rows around `row`, those past
/// the grid zero and, if `masked`, those off the board too; the board
/// starting at the cell `rows`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_column<F: Float>(
  tensor: &Tensor<F>,
  on_board: &Tensor<f32>,
  window: &mut Array<f32>,
  rows: usize,
  row: usize,
  column: usize,
  channel: usize,
  channels: usize,
  height: usize,
  width: usize,
  #[comptime] slot: usize,
  #[comptime] masked: bool,
) {
  #[unroll]
  for row_slot in 0..TAPS {
    // The row plus one, so that it is never negative.
    let other_row = row + row_slot;
    let mut value = f32::new(0.0);
    if column < width && other_row >= 1 && other_row <= height {
      let cell = rows + (other_row - 1) * width + column;
      value = f32::cast_from(tensor[cell * channels + channel]);
      if masked {
        value *= on_board[cell];
      }
    }
    window[slot * TAPS + row_slot] = value;
  }
}

/// Zeroes the column `slot` of `window`.
#[cube]
fn zero_column(window: &mut Array<f32>, #[comptime] slot: usize) {
  #[unroll]
  for row_slot in 0..TAPS {
    window[slot * TAPS + row_slot] = 0.0;
  }
}

/// Moves `window` one column to the right, making room for the next column in
/// the last slot.
#[cube]
fn shift(window: &mut Array<f32>) {
  #[unroll]
  for index in 0..(TAPS - 1) * TAPS {
    window[index] = window[index + TAPS];
  }
}

/// The convolution of `window` with `filter`: the cross-correlation burn's
/// convolutions compute, or, if `flipped`, the transposed convolution that
/// takes its gradient back to its input.
#[cube]
fn convolve(window: &Array<f32>, filter: &Array<f32>, #[comptime] flipped: bool) -> f32 {
  let mut sum = f32::new(0.0);
  #[unroll]
  for tap_row in 0..TAPS {
    #[unroll]
    for tap_column in 0..TAPS {
      let tap = tap_row * TAPS + tap_column;
      if flipped {
        sum += filter[tap] * window[(TAPS - 1 - tap_column) * TAPS + TAPS - 1 - tap_row];
      } else {
        sum += filter[tap] * window[tap_column * TAPS + tap_row];
      }
    }
  }
  sum
}

/// The logistic function.
#[cube]
fn sigmoid(value: f32) -> f32 {
  1.0 / (1.0 + (-value).exp())
}

/// A unit per channel of the output of every row of every board: the SiLU of
/// the mixed activation times the mixed gate, the mixed values being the
/// hidden ones plus the convolution of those on the board around them.
#[cube(launch)]
fn forward_kernel<F: Float>(
  hidden: &Tensor<F>,
  on_board: &Tensor<f32>,
  weight: &Tensor<f32>,
  output: &mut Tensor<F>,
  height: u32,
  width: u32,
  #[define(F)] _dtype: ElemType,
) {
  let cells = hidden.shape(1);
  let channels = hidden.shape(2);
  let half = channels / 2;
  let channel = CUBE_POS_X as usize * UNITS + UNIT_POS_X as usize;
  let row = CUBE_POS_Y as usize;
  let rows = CUBE_POS_Z as usize * cells;
  let (height, width) = (height as usize, width as usize);
  if channel < half {
    let gate_channel = channel + half;
    let activation_filter = load_filter(weight, channel);
    let gate_filter = load_filter(weight, gate_channel);
    let mut activations = Array::<f32>::new(TAPS * TAPS);
    let mut gates = Array::<f32>::new(TAPS * TAPS);
    zero_column(&mut activations, 0usize);
    zero_column(&mut gates, 0usize);
    #[unroll]
    for slot in 1..TAPS {
      load_column::<F>(
        hidden,
        on_board,
        &mut activations,
        rows,
        row,
        slot - 1,
        channel,
        channels,
        height,
        width,
        slot,
        true,
      );
      load_column::<F>(
        hidden,
        on_board,
        &mut gates,
        rows,
        row,
        slot - 1,
        gate_channel,
        channels,
        height,
        width,
        slot,
        true,
      );
    }
    for column in 0..width {
      let cell = rows + row * width + column;
      let activation =
        f32::cast_from(hidden[cell * channels + channel]) + convolve(&activations, &activation_filter, false);
      let gate = f32::cast_from(hidden[cell * channels + gate_channel]) + convolve(&gates, &gate_filter, false);
      output[cell * half + channel] = F::cast_from(activation * sigmoid(activation) * gate);
      shift(&mut activations);
      shift(&mut gates);
      load_column::<F>(
        hidden,
        on_board,
        &mut activations,
        rows,
        row,
        column + 2,
        channel,
        channels,
        height,
        width,
        TAPS - 1,
        true,
      );
      load_column::<F>(
        hidden,
        on_board,
        &mut gates,
        rows,
        row,
        column + 2,
        gate_channel,
        channels,
        height,
        width,
        TAPS - 1,
        true,
      );
    }
  }
}

/// A unit per channel of the output of every row of every board: the
/// gradients of the mixed activation and gate, from the output's.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grad_mixed_kernel<F: Float>(
  hidden: &Tensor<F>,
  on_board: &Tensor<f32>,
  weight: &Tensor<f32>,
  grad: &Tensor<F>,
  grad_mixed: &mut Tensor<f32>,
  height: u32,
  width: u32,
  #[define(F)] _dtype: ElemType,
) {
  let cells = hidden.shape(1);
  let channels = hidden.shape(2);
  let half = channels / 2;
  let channel = CUBE_POS_X as usize * UNITS + UNIT_POS_X as usize;
  let row = CUBE_POS_Y as usize;
  let rows = CUBE_POS_Z as usize * cells;
  let (height, width) = (height as usize, width as usize);
  if channel < half {
    let gate_channel = channel + half;
    let activation_filter = load_filter(weight, channel);
    let gate_filter = load_filter(weight, gate_channel);
    let mut activations = Array::<f32>::new(TAPS * TAPS);
    let mut gates = Array::<f32>::new(TAPS * TAPS);
    zero_column(&mut activations, 0usize);
    zero_column(&mut gates, 0usize);
    #[unroll]
    for slot in 1..TAPS {
      load_column::<F>(
        hidden,
        on_board,
        &mut activations,
        rows,
        row,
        slot - 1,
        channel,
        channels,
        height,
        width,
        slot,
        true,
      );
      load_column::<F>(
        hidden,
        on_board,
        &mut gates,
        rows,
        row,
        slot - 1,
        gate_channel,
        channels,
        height,
        width,
        slot,
        true,
      );
    }
    for column in 0..width {
      let cell = rows + row * width + column;
      let activation =
        f32::cast_from(hidden[cell * channels + channel]) + convolve(&activations, &activation_filter, false);
      let gate = f32::cast_from(hidden[cell * channels + gate_channel]) + convolve(&gates, &gate_filter, false);
      let grad = f32::cast_from(grad[cell * half + channel]);
      let logistic = sigmoid(activation);
      grad_mixed[cell * channels + channel] = grad * gate * logistic * (1.0 + activation * (1.0 - logistic));
      grad_mixed[cell * channels + gate_channel] = grad * activation * logistic;
      shift(&mut activations);
      shift(&mut gates);
      load_column::<F>(
        hidden,
        on_board,
        &mut activations,
        rows,
        row,
        column + 2,
        channel,
        channels,
        height,
        width,
        TAPS - 1,
        true,
      );
      load_column::<F>(
        hidden,
        on_board,
        &mut gates,
        rows,
        row,
        column + 2,
        gate_channel,
        channels,
        height,
        width,
        TAPS - 1,
        true,
      );
    }
  }
}

/// A unit per channel of every row of every board: the gradient of the
/// hidden value, its mixed value's plus, on the board, what it is mixed into
/// its neighbors with.
#[cube(launch)]
fn grad_hidden_kernel<F: Float>(
  grad_mixed: &Tensor<f32>,
  on_board: &Tensor<f32>,
  weight: &Tensor<f32>,
  grad_hidden: &mut Tensor<F>,
  height: u32,
  width: u32,
  #[define(F)] _dtype: ElemType,
) {
  let cells = grad_mixed.shape(1);
  let channels = grad_mixed.shape(2);
  let channel = CUBE_POS_X as usize * UNITS + UNIT_POS_X as usize;
  let row = CUBE_POS_Y as usize;
  let rows = CUBE_POS_Z as usize * cells;
  let (height, width) = (height as usize, width as usize);
  if channel < channels {
    let filter = load_filter(weight, channel);
    let mut grads = Array::<f32>::new(TAPS * TAPS);
    zero_column(&mut grads, 0usize);
    #[unroll]
    for slot in 1..TAPS {
      load_column::<f32>(
        grad_mixed,
        on_board,
        &mut grads,
        rows,
        row,
        slot - 1,
        channel,
        channels,
        height,
        width,
        slot,
        false,
      );
    }
    for column in 0..width {
      let cell = rows + row * width + column;
      // The middle of the window is the cell's own.
      let own = grads[TAPS + 1];
      grad_hidden[cell * channels + channel] = F::cast_from(own + on_board[cell] * convolve(&grads, &filter, true));
      shift(&mut grads);
      load_column::<f32>(
        grad_mixed,
        on_board,
        &mut grads,
        rows,
        row,
        column + 2,
        channel,
        channels,
        height,
        width,
        TAPS - 1,
        false,
      );
    }
  }
}

/// A unit per channel and chunk of [`CHUNK_ROWS`] rows of the boards, taken
/// one after another: the gradient of the channel's filter summed over the
/// chunk, `[chunks, channels, taps]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grad_weight_kernel<F: Float>(
  hidden: &Tensor<F>,
  on_board: &Tensor<f32>,
  grad_mixed: &Tensor<f32>,
  grad_weight: &mut Tensor<f32>,
  height: u32,
  width: u32,
  #[define(F)] _dtype: ElemType,
) {
  let boards = hidden.shape(0);
  let cells = hidden.shape(1);
  let channels = hidden.shape(2);
  let channel = CUBE_POS_X as usize * UNITS + UNIT_POS_X as usize;
  let chunk = CUBE_POS_Y as usize;
  let (height, width) = (height as usize, width as usize);
  if channel < channels {
    let mut sums = Array::<f32>::new(TAPS * TAPS);
    #[unroll]
    for tap in 0..TAPS * TAPS {
      sums[tap] = 0.0;
    }
    let mut window = Array::<f32>::new(TAPS * TAPS);
    let start = chunk * CHUNK_ROWS;
    let end = (start + CHUNK_ROWS).min(boards * height);
    for board_row in start..end {
      let rows = board_row / height * cells;
      let row = board_row % height;
      zero_column(&mut window, 0usize);
      #[unroll]
      for slot in 1..TAPS {
        load_column::<F>(
          hidden,
          on_board,
          &mut window,
          rows,
          row,
          slot - 1,
          channel,
          channels,
          height,
          width,
          slot,
          true,
        );
      }
      for column in 0..width {
        let grad = grad_mixed[(rows + row * width + column) * channels + channel];
        #[unroll]
        for tap_row in 0..TAPS {
          #[unroll]
          for tap_column in 0..TAPS {
            sums[tap_row * TAPS + tap_column] += grad * window[tap_column * TAPS + tap_row];
          }
        }
        shift(&mut window);
        load_column::<F>(
          hidden,
          on_board,
          &mut window,
          rows,
          row,
          column + 2,
          channel,
          channels,
          height,
          width,
          TAPS - 1,
          true,
        );
      }
    }
    #[unroll]
    for tap in 0..TAPS * TAPS {
      grad_weight[(chunk * channels + channel) * TAPS * TAPS + tap] = sums[tap];
    }
  }
}

/// The inputs, contiguous; `on_board` and `weight` in single precision.
struct Inputs {
  hidden: CubeTensor,
  on_board: CubeTensor,
  weight: CubeTensor,
  /// `[batch, cells, channels]`.
  dims: [usize; 3],
  /// Rows of the grid of cells.
  height: usize,
}

impl Inputs {
  fn new(hidden: CubeTensor, on_board: CubeTensor, weight: CubeTensor, height: usize, width: usize) -> Self {
    let dims = hidden.meta.shape().dims::<3>();
    let [batch, cells, channels] = dims;
    assert_eq!(cells, height * width, "the cells are those of a height x width grid");
    assert!(
      channels.is_multiple_of(2),
      "the hidden values are activations and their gates"
    );
    assert_eq!(on_board.meta.shape().dims::<2>(), [batch, cells]);
    assert_eq!(weight.meta.shape().dims::<4>(), [channels, 1, TAPS, TAPS]);
    let single = |tensor: CubeTensor| into_contiguous(CubeBackend::float_cast(tensor, FloatDType::F32));
    Self {
      hidden: into_contiguous(hidden),
      on_board: single(on_board),
      weight: single(weight),
      dims,
      height,
    }
  }

  fn empty(&self, shape: Shape, dtype: DType) -> CubeTensor {
    let buffer = self.hidden.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(
      self.hidden.client.clone(),
      self.hidden.device.clone(),
      shape,
      buffer,
      dtype,
    )
  }

  /// A cube per [`UNITS`] of `channels` of every row of every board.
  fn launch_dims(&self, channels: usize) -> (CubeCount, CubeDim) {
    let [batch, ..] = self.dims;
    (
      CubeCount::Static(channels.div_ceil(UNITS) as u32, self.height as u32, batch as u32),
      CubeDim::new_1d(UNITS as u32),
    )
  }
}

impl ConvGluBackend for CubeBackend {
  fn conv_glu(
    hidden: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    weight: FloatTensor<Self>,
    height: usize,
    width: usize,
  ) -> FloatTensor<Self> {
    let inputs = Inputs::new(hidden, on_board, weight, height, width);
    let [batch, cells, channels] = inputs.dims;
    let dtype = inputs.hidden.dtype;
    let output = inputs.empty(Shape::new([batch, cells, channels / 2]), dtype);
    let (cube_count, cube_dim) = inputs.launch_dims(channels / 2);
    forward_kernel::launch(
      &inputs.hidden.client.clone(),
      cube_count,
      cube_dim,
      inputs.hidden.into_tensor_arg(),
      inputs.on_board.into_tensor_arg(),
      inputs.weight.into_tensor_arg(),
      output.clone().into_tensor_arg(),
      height as u32,
      width as u32,
      dtype_to_storage_type(dtype),
    );
    output
  }

  fn conv_glu_backward(
    hidden: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    weight: FloatTensor<Self>,
    grad: FloatTensor<Self>,
    height: usize,
    width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>) {
    let weight_dtype = weight.dtype;
    let inputs = Inputs::new(hidden, on_board, weight, height, width);
    let [batch, cells, channels] = inputs.dims;
    let dtype = inputs.hidden.dtype;
    let client = inputs.hidden.client.clone();
    let grad = into_contiguous(Self::float_cast(grad, dtype.into()));
    let grad_mixed = inputs.empty(Shape::new([batch, cells, channels]), DType::F32);
    let (cube_count, cube_dim) = inputs.launch_dims(channels / 2);
    grad_mixed_kernel::launch(
      &client,
      cube_count,
      cube_dim,
      inputs.hidden.clone().into_tensor_arg(),
      inputs.on_board.clone().into_tensor_arg(),
      inputs.weight.clone().into_tensor_arg(),
      grad.into_tensor_arg(),
      grad_mixed.clone().into_tensor_arg(),
      height as u32,
      width as u32,
      dtype_to_storage_type(dtype),
    );

    let grad_hidden = inputs.empty(Shape::new([batch, cells, channels]), dtype);
    let chunks = (batch * height).div_ceil(CHUNK_ROWS);
    let grad_weight = inputs.empty(Shape::new([chunks, channels, TAPS * TAPS]), DType::F32);
    let (cube_count, cube_dim) = inputs.launch_dims(channels);
    grad_hidden_kernel::launch(
      &client,
      cube_count,
      cube_dim,
      grad_mixed.clone().into_tensor_arg(),
      inputs.on_board.clone().into_tensor_arg(),
      inputs.weight.into_tensor_arg(),
      grad_hidden.clone().into_tensor_arg(),
      height as u32,
      width as u32,
      dtype_to_storage_type(dtype),
    );

    grad_weight_kernel::launch(
      &client,
      CubeCount::Static(channels.div_ceil(UNITS) as u32, chunks as u32, 1),
      CubeDim::new_1d(UNITS as u32),
      inputs.hidden.into_tensor_arg(),
      inputs.on_board.into_tensor_arg(),
      grad_mixed.into_tensor_arg(),
      grad_weight.clone().into_tensor_arg(),
      height as u32,
      width as u32,
      dtype_to_storage_type(dtype),
    );
    let grad_weight = Self::float_sum_dim(grad_weight, 0);
    let grad_weight = Self::float_reshape(grad_weight, Shape::new([channels, 1, TAPS, TAPS]));
    let grad_weight = Self::float_cast(grad_weight, weight_dtype.into());
    (grad_hidden, grad_weight)
  }
}
