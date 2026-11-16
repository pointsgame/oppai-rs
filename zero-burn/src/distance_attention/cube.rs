//! The kernels of the distance attention, after FlashAttention-2.
//!
//! A unit owns a query's row of the attention - or a key's column in the
//! backward pass for the keys and the values - and streams the other side
//! through shared memory a block at a time, keeping its own vectors and the
//! sums it accumulates in registers; everything is computed in single
//! precision. The forward pass keeps the softmax normalized as it goes, and
//! saves the log-sum-exp of every query's scores along with the output, from
//! which the backward pass recomputes every probability: no score is ever
//! stored. The distance between two cells is computed from their indices.
//!
//! The backward pass takes two kernels, one per side, so that every gradient
//! is summed by the one unit that writes it: one for the queries' gradients,
//! and one for the keys' and the values', which also sums the slopes'
//! gradient over every key's column; those sums are added up over the batch
//! and the cells afterwards.

use super::{DistanceAttentionBackend, composed::scale};
use burn::{
  backend::{cubecl::dtype_to_storage_type, ops::FloatTensorOps, tensor::FloatTensor},
  tensor::{DType, FloatDType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// A score no key gets; finite, as WGSL has no infinity.
const NO_SCORE: f64 = -1e30;

/// The shared memory of a block: that of the keys and values of
/// [`forward_kernel`], or of the queries and output gradients of
/// [`grad_key_value_kernel`], in bytes.
const BLOCK_BYTES: usize = 16 * 1024;

/// How many queries or keys a cube takes, and so how many cells it streams
/// through shared memory at a time: a power of two between 8 and 64 for which
/// two vectors of `channels` per cell fit in [`BLOCK_BYTES`].
fn block(channels: usize) -> usize {
  let fits = BLOCK_BYTES / (2 * channels * size_of::<f32>());
  let mut block = 64;
  while block > 8 && block > fits {
    block /= 2;
  }
  block
}

/// The distance between two cells, by their rows and columns.
#[cube]
fn distance(row: f32, column: f32, other_row: f32, other_column: f32) -> f32 {
  let rows = row - other_row;
  let columns = column - other_column;
  (rows * rows + columns * columns).sqrt()
}

/// The row and column of a cell of a board `width` cells wide.
#[cube]
fn position(cell: usize, width: usize) -> (f32, f32) {
  (f32::cast_from((cell / width) as u32), f32::cast_from((cell % width) as u32))
}

/// The cells of the block `block` of `cells`, every unit of the cube loading
/// that of its own index into shared memory: its two vectors of `channels`,
/// one of `first` and one of `second` from the row `rows` starts at,
/// `on_board` - zero for a cell past the last one - and its position.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_block<F: Float>(
  first: &Tensor<F>,
  second: &Tensor<F>,
  first_scale: f32,
  on_board: &Tensor<f32>,
  on_board_start: usize,
  rows: usize,
  block_index: usize,
  cells: usize,
  width: usize,
  first_block: &mut Shared<[f32]>,
  second_block: &mut Shared<[f32]>,
  block_on_board: &mut Shared<[f32]>,
  block_rows: &mut Shared<[f32]>,
  block_columns: &mut Shared<[f32]>,
  #[comptime] channels: usize,
  #[comptime] block: usize,
) {
  let unit = UNIT_POS_X as usize;
  let cell = block_index * block + unit;
  // A cell past the last one loads the last one, and is off the board.
  let loaded = cell.min(cells - 1);
  #[unroll]
  for channel in 0..channels {
    first_block[unit * channels + channel] = f32::cast_from(first[(rows + loaded) * channels + channel]) * first_scale;
    second_block[unit * channels + channel] = f32::cast_from(second[(rows + loaded) * channels + channel]);
  }
  let mut on = on_board[on_board_start + loaded];
  if cell >= cells {
    on = 0.0;
  }
  block_on_board[unit] = on;
  let (row, column) = position(loaded, width);
  block_rows[unit] = row;
  block_columns[unit] = column;
}

/// One cube per `block` queries of a head of a board: the output and the
/// log-sum-exp of every query.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn forward_kernel<F: Float>(
  query: &Tensor<F>,
  key: &Tensor<F>,
  value: &Tensor<F>,
  on_board: &Tensor<f32>,
  slopes: &Tensor<f32>,
  output: &mut Tensor<F>,
  log_sum_exp: &mut Tensor<f32>,
  width: u32,
  scale: f32,
  #[comptime] channels: usize,
  #[comptime] block: usize,
  #[define(F)] _dtype: ElemType,
) {
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let cell = CUBE_POS_X as usize * block + UNIT_POS_X as usize;
  let own = cell.min(cells - 1);
  let slope = slopes[head];
  let (row, column) = position(own, width);

  let mut own_query = Array::<f32>::new(channels);
  let mut sums = Array::<f32>::new(channels);
  #[unroll]
  for channel in 0..channels {
    own_query[channel] = f32::cast_from(query[(rows + own) * channels + channel]) * scale;
    sums[channel] = 0.0;
  }
  let mut max = f32::new(NO_SCORE);
  let mut denominator = f32::new(0.0);

  let mut keys = Shared::<[f32]>::new_slice(block * channels);
  let mut values = Shared::<[f32]>::new_slice(block * channels);
  let mut keys_on_board = Shared::<[f32]>::new_slice(block);
  let mut key_rows = Shared::<[f32]>::new_slice(block);
  let mut key_columns = Shared::<[f32]>::new_slice(block);

  let blocks = cells.div_ceil(block);
  for block_index in 0..blocks {
    sync_cube();
    load_block::<F>(
      key,
      value,
      1.0,
      on_board,
      batch * cells,
      rows,
      block_index,
      cells,
      width,
      &mut keys,
      &mut values,
      &mut keys_on_board,
      &mut key_rows,
      &mut key_columns,
      channels,
      block,
    );
    sync_cube();
    for other in 0..block {
      if keys_on_board[other] > 0.5 {
        let mut score = f32::new(0.0);
        #[unroll]
        for channel in 0..channels {
          score += own_query[channel] * keys[other * channels + channel];
        }
        score -= slope * distance(row, column, key_rows[other], key_columns[other]);
        if score > max {
          let rescale = (max - score).exp();
          denominator *= rescale;
          #[unroll]
          for channel in 0..channels {
            sums[channel] *= rescale;
          }
          max = score;
        }
        let probability = (score - max).exp();
        denominator += probability;
        #[unroll]
        for channel in 0..channels {
          sums[channel] += probability * values[other * channels + channel];
        }
      }
    }
  }

  if cell < cells {
    // A query with no key on the board, which a board has none of, attends
    // to nothing.
    let mut normalizer = f32::new(0.0);
    let mut total = f32::new(-NO_SCORE);
    if denominator > 0.0 {
      normalizer = 1.0 / denominator;
      total = max + denominator.ln();
    }
    #[unroll]
    for channel in 0..channels {
      output[(rows + cell) * channels + channel] = F::cast_from(sums[channel] * normalizer);
    }
    log_sum_exp[rows + cell] = total;
  }
}

/// The dot product of the output and its gradient of a query, which the
/// gradient of the softmax takes off every probability's.
#[cube]
fn output_dot_grad<F: Float>(output: &Tensor<F>, grad: &Tensor<F>, start: usize, #[comptime] channels: usize) -> f32 {
  let mut sum = f32::new(0.0);
  #[unroll]
  for channel in 0..channels {
    sum += f32::cast_from(output[start + channel]) * f32::cast_from(grad[start + channel]);
  }
  sum
}

/// One cube per `block` queries of a head of a board: the gradients of the
/// queries.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grad_query_kernel<F: Float>(
  query: &Tensor<F>,
  key: &Tensor<F>,
  value: &Tensor<F>,
  on_board: &Tensor<f32>,
  slopes: &Tensor<f32>,
  output: &Tensor<F>,
  log_sum_exp: &Tensor<f32>,
  grad: &Tensor<F>,
  grad_query: &mut Tensor<F>,
  width: u32,
  scale: f32,
  #[comptime] channels: usize,
  #[comptime] block: usize,
  #[define(F)] _dtype: ElemType,
) {
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let cell = CUBE_POS_X as usize * block + UNIT_POS_X as usize;
  let own = cell.min(cells - 1);
  let slope = slopes[head];
  let (row, column) = position(own, width);
  let total = log_sum_exp[rows + own];
  let output_dot_grad = output_dot_grad::<F>(output, grad, (rows + own) * channels, channels);

  let mut own_query = Array::<f32>::new(channels);
  let mut own_grad = Array::<f32>::new(channels);
  let mut sums = Array::<f32>::new(channels);
  #[unroll]
  for channel in 0..channels {
    own_query[channel] = f32::cast_from(query[(rows + own) * channels + channel]) * scale;
    own_grad[channel] = f32::cast_from(grad[(rows + own) * channels + channel]);
    sums[channel] = 0.0;
  }

  let mut keys = Shared::<[f32]>::new_slice(block * channels);
  let mut values = Shared::<[f32]>::new_slice(block * channels);
  let mut keys_on_board = Shared::<[f32]>::new_slice(block);
  let mut key_rows = Shared::<[f32]>::new_slice(block);
  let mut key_columns = Shared::<[f32]>::new_slice(block);

  let blocks = cells.div_ceil(block);
  for block_index in 0..blocks {
    sync_cube();
    load_block::<F>(
      key,
      value,
      1.0,
      on_board,
      batch * cells,
      rows,
      block_index,
      cells,
      width,
      &mut keys,
      &mut values,
      &mut keys_on_board,
      &mut key_rows,
      &mut key_columns,
      channels,
      block,
    );
    sync_cube();
    for other in 0..block {
      if keys_on_board[other] > 0.5 {
        let mut score = f32::new(0.0);
        let mut grad_probability = f32::new(0.0);
        #[unroll]
        for channel in 0..channels {
          score += own_query[channel] * keys[other * channels + channel];
          grad_probability += own_grad[channel] * values[other * channels + channel];
        }
        score -= slope * distance(row, column, key_rows[other], key_columns[other]);
        let grad_score = (score - total).exp() * (grad_probability - output_dot_grad);
        #[unroll]
        for channel in 0..channels {
          sums[channel] += grad_score * keys[other * channels + channel];
        }
      }
    }
  }

  if cell < cells {
    #[unroll]
    for channel in 0..channels {
      grad_query[(rows + cell) * channels + channel] = F::cast_from(sums[channel] * scale);
    }
  }
}

/// One cube per `block` keys of a head of a board: the gradients of the keys
/// and the values, and every key's part of the gradient of the head's slope.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grad_key_value_kernel<F: Float>(
  query: &Tensor<F>,
  key: &Tensor<F>,
  value: &Tensor<F>,
  on_board: &Tensor<f32>,
  slopes: &Tensor<f32>,
  output: &Tensor<F>,
  log_sum_exp: &Tensor<f32>,
  grad: &Tensor<F>,
  grad_key: &mut Tensor<F>,
  grad_value: &mut Tensor<F>,
  grad_slopes: &mut Tensor<f32>,
  width: u32,
  scale: f32,
  #[comptime] channels: usize,
  #[comptime] block: usize,
  #[define(F)] _dtype: ElemType,
) {
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let unit = UNIT_POS_X as usize;
  let cell = CUBE_POS_X as usize * block + unit;
  let own = cell.min(cells - 1);
  let slope = slopes[head];
  let (row, column) = position(own, width);
  let attended = cell < cells && on_board[batch * cells + own] > 0.5;

  let mut own_key = Array::<f32>::new(channels);
  let mut own_value = Array::<f32>::new(channels);
  let mut key_sums = Array::<f32>::new(channels);
  let mut value_sums = Array::<f32>::new(channels);
  #[unroll]
  for channel in 0..channels {
    own_key[channel] = f32::cast_from(key[(rows + own) * channels + channel]);
    own_value[channel] = f32::cast_from(value[(rows + own) * channels + channel]);
    key_sums[channel] = 0.0;
    value_sums[channel] = 0.0;
  }
  let mut slope_sum = f32::new(0.0);

  // The queries, scaled, and the gradients of their outputs, then the
  // log-sum-exp of every query and the dot product of its output and the
  // output's gradient; the queries are all on the board, as a cell off it
  // still has a row of the attention.
  let mut queries = Shared::<[f32]>::new_slice(block * channels);
  let mut grads = Shared::<[f32]>::new_slice(block * channels);
  let mut queries_in = Shared::<[f32]>::new_slice(block);
  let mut query_rows = Shared::<[f32]>::new_slice(block);
  let mut query_columns = Shared::<[f32]>::new_slice(block);
  let mut totals = Shared::<[f32]>::new_slice(block);
  let mut output_dot_grads = Shared::<[f32]>::new_slice(block);

  let blocks = cells.div_ceil(block);
  for block_index in 0..blocks {
    sync_cube();
    let other_cell = block_index * block + unit;
    let loaded = other_cell.min(cells - 1);
    #[unroll]
    for channel in 0..channels {
      queries[unit * channels + channel] = f32::cast_from(query[(rows + loaded) * channels + channel]) * scale;
      grads[unit * channels + channel] = f32::cast_from(grad[(rows + loaded) * channels + channel]);
    }
    let mut query_in = f32::new(1.0);
    if other_cell >= cells {
      query_in = 0.0;
    }
    queries_in[unit] = query_in;
    let (other_row, other_column) = position(loaded, width);
    query_rows[unit] = other_row;
    query_columns[unit] = other_column;
    totals[unit] = log_sum_exp[rows + loaded];
    output_dot_grads[unit] = output_dot_grad::<F>(output, grad, (rows + loaded) * channels, channels);
    sync_cube();
    if attended {
      for other in 0..block {
        if queries_in[other] > 0.5 {
          let mut score = f32::new(0.0);
          let mut grad_probability = f32::new(0.0);
          #[unroll]
          for channel in 0..channels {
            score += queries[other * channels + channel] * own_key[channel];
            grad_probability += grads[other * channels + channel] * own_value[channel];
          }
          let distance = distance(query_rows[other], query_columns[other], row, column);
          score -= slope * distance;
          let probability = (score - totals[other]).exp();
          let grad_score = probability * (grad_probability - output_dot_grads[other]);
          #[unroll]
          for channel in 0..channels {
            value_sums[channel] += probability * grads[other * channels + channel];
            key_sums[channel] += grad_score * queries[other * channels + channel];
          }
          slope_sum -= grad_score * distance;
        }
      }
    }
  }

  if cell < cells {
    #[unroll]
    for channel in 0..channels {
      grad_key[(rows + cell) * channels + channel] = F::cast_from(key_sums[channel]);
      grad_value[(rows + cell) * channels + channel] = F::cast_from(value_sums[channel]);
    }
    grad_slopes[rows + cell] = slope_sum;
  }
}

/// The tensors every kernel reads, contiguous, `on_board` and `slopes` in
/// single precision, and the dimensions of the queries.
struct Inputs {
  query: CubeTensor,
  key: CubeTensor,
  value: CubeTensor,
  on_board: CubeTensor,
  slopes: CubeTensor,
  dims: [usize; 4],
}

impl Inputs {
  fn new(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    on_board: CubeTensor,
    slopes: CubeTensor,
  ) -> Self {
    let dims = query.meta.shape().dims::<4>();
    assert_eq!(key.meta.shape(), query.meta.shape(), "the keys are shaped as the queries");
    assert_eq!(value.meta.shape(), query.meta.shape(), "the values are shaped as the queries");
    assert!(
      key.dtype == query.dtype && value.dtype == query.dtype,
      "the queries, keys and values are of one float type"
    );
    assert_eq!(on_board.meta.shape().dims::<2>(), [dims[0], dims[2]]);
    assert_eq!(slopes.meta.shape().dims::<1>(), [dims[1]]);
    let single = |tensor: CubeTensor| {
      into_contiguous(CubeBackend::float_cast(tensor, FloatDType::F32))
    };
    Self {
      query: into_contiguous(query),
      key: into_contiguous(key),
      value: into_contiguous(value),
      on_board: single(on_board),
      slopes: single(slopes),
      dims,
    }
  }

  fn empty(&self, shape: Shape, dtype: DType) -> CubeTensor {
    let buffer = self.query.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(
      self.query.client.clone(),
      self.query.device.clone(),
      shape,
      buffer,
      dtype,
    )
  }

  /// A cube per block of cells of every head of every board, a unit per cell.
  fn launch_dims(&self) -> (CubeCount, CubeDim, usize) {
    let [batch, heads, cells, channels] = self.dims;
    let block = block(channels);
    (
      CubeCount::Static(cells.div_ceil(block) as u32, heads as u32, batch as u32),
      CubeDim::new_1d(block as u32),
      block,
    )
  }
}

impl DistanceAttentionBackend for CubeBackend {
  fn distance_attention_forward(
    query: FloatTensor<Self>,
    key: FloatTensor<Self>,
    value: FloatTensor<Self>,
    on_board: FloatTensor<Self>,
    slopes: FloatTensor<Self>,
    width: usize,
  ) -> (FloatTensor<Self>, FloatTensor<Self>) {
    let inputs = Inputs::new(query, key, value, on_board, slopes);
    let [batch, heads, cells, channels] = inputs.dims;
    let dtype = inputs.query.dtype;
    let output = inputs.empty(inputs.query.meta.shape().clone(), dtype);
    let log_sum_exp = inputs.empty(Shape::new([batch, heads, cells]), DType::F32);
    let (cube_count, cube_dim, block) = inputs.launch_dims();
    let client = inputs.query.client.clone();
    forward_kernel::launch(
      &client,
      cube_count,
      cube_dim,
      inputs.query.into_tensor_arg(),
      inputs.key.into_tensor_arg(),
      inputs.value.into_tensor_arg(),
      inputs.on_board.into_tensor_arg(),
      inputs.slopes.into_tensor_arg(),
      output.clone().into_tensor_arg(),
      log_sum_exp.clone().into_tensor_arg(),
      width as u32,
      scale(channels) as f32,
      channels,
      block,
      dtype_to_storage_type(dtype),
    );
    (output, log_sum_exp)
  }

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
    let slopes_dtype = slopes.dtype;
    let inputs = Inputs::new(query, key, value, on_board, slopes);
    let [batch, heads, cells, channels] = inputs.dims;
    let dtype = inputs.query.dtype;
    let output = into_contiguous(Self::float_cast(output, dtype.into()));
    let grad = into_contiguous(Self::float_cast(grad, dtype.into()));
    let log_sum_exp = into_contiguous(log_sum_exp);
    let shape = inputs.query.meta.shape().clone();
    let grad_query = inputs.empty(shape.clone(), dtype);
    let grad_key = inputs.empty(shape.clone(), dtype);
    let grad_value = inputs.empty(shape, dtype);
    let grad_slopes = inputs.empty(Shape::new([batch, heads, cells]), DType::F32);
    let (cube_count, cube_dim, block) = inputs.launch_dims();
    let scale = scale(channels) as f32;
    let client = inputs.query.client.clone();
    grad_query_kernel::launch(
      &client,
      cube_count.clone(),
      cube_dim,
      inputs.query.clone().into_tensor_arg(),
      inputs.key.clone().into_tensor_arg(),
      inputs.value.clone().into_tensor_arg(),
      inputs.on_board.clone().into_tensor_arg(),
      inputs.slopes.clone().into_tensor_arg(),
      output.clone().into_tensor_arg(),
      log_sum_exp.clone().into_tensor_arg(),
      grad.clone().into_tensor_arg(),
      grad_query.clone().into_tensor_arg(),
      width as u32,
      scale,
      channels,
      block,
      dtype_to_storage_type(dtype),
    );
    grad_key_value_kernel::launch(
      &client,
      cube_count,
      cube_dim,
      inputs.query.into_tensor_arg(),
      inputs.key.into_tensor_arg(),
      inputs.value.into_tensor_arg(),
      inputs.on_board.into_tensor_arg(),
      inputs.slopes.into_tensor_arg(),
      output.into_tensor_arg(),
      log_sum_exp.into_tensor_arg(),
      grad.into_tensor_arg(),
      grad_key.clone().into_tensor_arg(),
      grad_value.clone().into_tensor_arg(),
      grad_slopes.clone().into_tensor_arg(),
      width as u32,
      scale,
      channels,
      block,
      dtype_to_storage_type(dtype),
    );
    let grad_slopes = Self::float_sum_dim(Self::float_sum_dim(grad_slopes, 2), 0);
    let grad_slopes = Self::float_reshape(grad_slopes, Shape::new([heads]));
    let grad_slopes = Self::float_cast(grad_slopes, slopes_dtype.into());
    (grad_query, grad_key, grad_value, grad_slopes)
  }
}
