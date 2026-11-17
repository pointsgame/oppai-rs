//! The kernels of the distance attention, after FlashAttention-2.
//!
//! A unit owns a query's row of the attention - or a key's column in the
//! backward pass for the keys and the values - and streams the other side
//! through shared memory a block at a time, keeping its own vectors and the
//! sums it accumulates in registers, reading shared memory four channels at a
//! time; everything is computed in single precision. The forward pass keeps the softmax normalized as it goes, and
//! saves the log-sum-exp of every query's scores along with the output, from
//! which the backward pass recomputes every probability: no score is ever
//! stored. The distance between two cells is computed from their indices.
//!
//! The backward pass takes two kernels, one per side, so that every gradient
//! is summed by the one unit that writes it: one for the queries' gradients,
//! and one for the keys' and the values', which also sums the slopes'
//! gradient over every key's column; those sums are added up over the batch
//! and the cells afterwards.

use super::{DistanceAttentionBackend, composed::scale, mma};
use burn::{
  backend::{cubecl::dtype_to_storage_type, ops::FloatTensorOps, tensor::FloatTensor},
  tensor::{DType, FloatDType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, prelude::*};

/// A score no key gets; finite, as WGSL has no infinity.
pub(super) const NO_SCORE: f64 = -1e30;

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

/// How many queries a unit of [`forward_kernel`] takes. Two of them read every
/// key and value from shared memory half as often as one; four take more
/// registers than a unit has and run several times slower.
const FORWARD_QUERIES: usize = 2;

/// How many queries a unit of [`grad_query_kernel`] takes; as for
/// [`FORWARD_QUERIES`], more than two take more registers than a unit has.
const GRAD_QUERIES: usize = 2;

/// The distance between two cells, by their rows and columns.
#[cube]
pub(super) fn distance(row: f32, column: f32, other_row: f32, other_column: f32) -> f32 {
  let rows = row - other_row;
  let columns = column - other_column;
  (rows * rows + columns * columns).sqrt()
}

/// The row and column of a cell of a board `width` cells wide.
#[cube]
pub(super) fn position(cell: usize, width: usize) -> (f32, f32) {
  (f32::cast_from((cell / width) as u32), f32::cast_from((cell % width) as u32))
}

/// Four channels, which the kernels read from shared memory at once.
type Lanes = Vector<f32, Const<4>>;

/// How many channels a [`Lanes`] holds.
const LANES: usize = 4;

/// The four channels of `tensor` from `start` on, times `scale`.
#[cube]
fn read_lanes<F: Float>(tensor: &Tensor<F>, start: usize, scale: f32) -> Lanes {
  let mut lanes = Lanes::empty();
  #[unroll]
  for lane in 0..LANES {
    lanes.insert(lane, f32::cast_from(tensor[start + lane]) * scale);
  }
  lanes
}

/// The sum of the four channels.
#[cube]
fn lanes_sum(lanes: Lanes) -> f32 {
  lanes.extract(0usize) + lanes.extract(1usize) + lanes.extract(2usize) + lanes.extract(3usize)
}

/// The cells of the block `block_index` of `cells`, every unit of the cube
/// loading that of its own index into shared memory: its two vectors of
/// `channels`, one of `first` times `first_scale` and one of `second`, from
/// the row `rows` starts at, `on_board` - zero for a cell past the last one -
/// and its position.
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
  first_block: &mut Shared<[Lanes]>,
  second_block: &mut Shared<[Lanes]>,
  block_on_board: &mut Shared<[f32]>,
  block_rows: &mut Shared<[f32]>,
  block_columns: &mut Shared<[f32]>,
  #[comptime] channels: usize,
  #[comptime] block: usize,
) {
  let vectors = comptime!(channels / LANES);
  let unit = UNIT_POS_X as usize;
  let cell = block_index * block + unit;
  // A cell past the last one loads the last one, and is off the board.
  let loaded = cell.min(cells - 1);
  #[unroll]
  for vector in 0..vectors {
    let start = (rows + loaded) * channels + vector * LANES;
    first_block[unit * vectors + vector] = read_lanes::<F>(first, start, first_scale);
    second_block[unit * vectors + vector] = read_lanes::<F>(second, start, 1.0);
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

/// One cube per `block * queries` queries of a head of a board: the output
/// and the log-sum-exp of every query. Every unit takes `queries` of them, a
/// block apart, so that each key and value it reads from shared memory serves
/// all of them; it reads them four channels at a time.
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
  #[comptime] queries: usize,
  #[define(F)] _dtype: ElemType,
) {
  let vectors = comptime!(channels / LANES);
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let first = CUBE_POS_X as usize * block * queries + UNIT_POS_X as usize;
  let slope = slopes[head];

  let mut own_queries = Array::<Lanes>::new(queries * vectors);
  let mut sums = Array::<Lanes>::new(queries * vectors);
  let mut dots = Array::<Lanes>::new(queries);
  let mut query_rows = Array::<f32>::new(queries);
  let mut query_columns = Array::<f32>::new(queries);
  let mut maxes = Array::<f32>::new(queries);
  let mut denominators = Array::<f32>::new(queries);
  let mut probabilities = Array::<f32>::new(queries);
  #[unroll]
  for index in 0..queries {
    let own = (first + index * block).min(cells - 1);
    let (row, column) = position(own, width);
    query_rows[index] = row;
    query_columns[index] = column;
    maxes[index] = f32::new(NO_SCORE);
    denominators[index] = 0.0;
    #[unroll]
    for vector in 0..vectors {
      own_queries[index * vectors + vector] = read_lanes::<F>(query, (rows + own) * channels + vector * LANES, scale);
      sums[index * vectors + vector] = Lanes::zeroed();
    }
  }

  let mut keys = Shared::<[Lanes]>::new_slice(block * vectors);
  let mut values = Shared::<[Lanes]>::new_slice(block * vectors);
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
        #[unroll]
        for index in 0..queries {
          dots[index] = Lanes::zeroed();
        }
        #[unroll]
        for vector in 0..vectors {
          let key = keys[other * vectors + vector];
          #[unroll]
          for index in 0..queries {
            dots[index] += own_queries[index * vectors + vector] * key;
          }
        }
        let key_row = key_rows[other];
        let key_column = key_columns[other];
        #[unroll]
        for index in 0..queries {
          let score =
            lanes_sum(dots[index]) - slope * distance(query_rows[index], query_columns[index], key_row, key_column);
          if score > maxes[index] {
            let rescale = (maxes[index] - score).exp();
            denominators[index] *= rescale;
            let rescale = Lanes::broadcast(rescale);
            #[unroll]
            for vector in 0..vectors {
              sums[index * vectors + vector] *= rescale;
            }
            maxes[index] = score;
          }
          let probability = (score - maxes[index]).exp();
          denominators[index] += probability;
          probabilities[index] = probability;
        }
        #[unroll]
        for vector in 0..vectors {
          let value = values[other * vectors + vector];
          #[unroll]
          for index in 0..queries {
            sums[index * vectors + vector] += Lanes::broadcast(probabilities[index]) * value;
          }
        }
      }
    }
  }

  #[unroll]
  for index in 0..queries {
    let cell = first + index * block;
    if cell < cells {
      // A query with no key on the board, which a board has none of, attends
      // to nothing.
      let denominator = denominators[index];
      let mut normalizer = f32::new(0.0);
      let mut total = f32::new(-NO_SCORE);
      if denominator > 0.0 {
        normalizer = 1.0 / denominator;
        total = maxes[index] + denominator.ln();
      }
      #[unroll]
      for vector in 0..vectors {
        let sum = sums[index * vectors + vector];
        #[unroll]
        for lane in 0..LANES {
          output[(rows + cell) * channels + vector * LANES + lane] = F::cast_from(sum.extract(lane) * normalizer);
        }
      }
      log_sum_exp[rows + cell] = total;
    }
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

/// One cube per `block * queries` queries of a head of a board: the gradients
/// of the queries. Every unit takes `queries` of them, a block apart, as in
/// [`forward_kernel`].
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
  #[comptime] queries: usize,
  #[define(F)] _dtype: ElemType,
) {
  let vectors = comptime!(channels / LANES);
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let first = CUBE_POS_X as usize * block * queries + UNIT_POS_X as usize;
  let slope = slopes[head];

  let mut own_queries = Array::<Lanes>::new(queries * vectors);
  let mut own_grads = Array::<Lanes>::new(queries * vectors);
  let mut sums = Array::<Lanes>::new(queries * vectors);
  let mut scores = Array::<Lanes>::new(queries);
  let mut grad_probabilities = Array::<Lanes>::new(queries);
  let mut grad_scores = Array::<Lanes>::new(queries);
  let mut query_rows = Array::<f32>::new(queries);
  let mut query_columns = Array::<f32>::new(queries);
  let mut totals = Array::<f32>::new(queries);
  let mut output_dot_grads = Array::<f32>::new(queries);
  #[unroll]
  for index in 0..queries {
    let own = (first + index * block).min(cells - 1);
    let (row, column) = position(own, width);
    query_rows[index] = row;
    query_columns[index] = column;
    totals[index] = log_sum_exp[rows + own];
    output_dot_grads[index] = output_dot_grad::<F>(output, grad, (rows + own) * channels, channels);
    #[unroll]
    for vector in 0..vectors {
      let start = (rows + own) * channels + vector * LANES;
      own_queries[index * vectors + vector] = read_lanes::<F>(query, start, scale);
      own_grads[index * vectors + vector] = read_lanes::<F>(grad, start, 1.0);
      sums[index * vectors + vector] = Lanes::zeroed();
    }
  }

  let mut keys = Shared::<[Lanes]>::new_slice(block * vectors);
  let mut values = Shared::<[Lanes]>::new_slice(block * vectors);
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
        #[unroll]
        for index in 0..queries {
          scores[index] = Lanes::zeroed();
          grad_probabilities[index] = Lanes::zeroed();
        }
        #[unroll]
        for vector in 0..vectors {
          let key = keys[other * vectors + vector];
          let value = values[other * vectors + vector];
          #[unroll]
          for index in 0..queries {
            scores[index] += own_queries[index * vectors + vector] * key;
            grad_probabilities[index] += own_grads[index * vectors + vector] * value;
          }
        }
        let key_row = key_rows[other];
        let key_column = key_columns[other];
        #[unroll]
        for index in 0..queries {
          let score =
            lanes_sum(scores[index]) - slope * distance(query_rows[index], query_columns[index], key_row, key_column);
          grad_scores[index] = Lanes::broadcast(
            (score - totals[index]).exp() * (lanes_sum(grad_probabilities[index]) - output_dot_grads[index]),
          );
        }
        #[unroll]
        for vector in 0..vectors {
          let key = keys[other * vectors + vector];
          #[unroll]
          for index in 0..queries {
            sums[index * vectors + vector] += grad_scores[index] * key;
          }
        }
      }
    }
  }

  #[unroll]
  for index in 0..queries {
    let cell = first + index * block;
    if cell < cells {
      #[unroll]
      for vector in 0..vectors {
        let sum = sums[index * vectors + vector];
        #[unroll]
        for lane in 0..LANES {
          grad_query[(rows + cell) * channels + vector * LANES + lane] = F::cast_from(sum.extract(lane) * scale);
        }
      }
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
  let vectors = comptime!(channels / LANES);
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

  let mut own_key = Array::<Lanes>::new(vectors);
  let mut own_value = Array::<Lanes>::new(vectors);
  let mut key_sums = Array::<Lanes>::new(vectors);
  let mut value_sums = Array::<Lanes>::new(vectors);
  #[unroll]
  for vector in 0..vectors {
    let start = (rows + own) * channels + vector * LANES;
    own_key[vector] = read_lanes::<F>(key, start, 1.0);
    own_value[vector] = read_lanes::<F>(value, start, 1.0);
    key_sums[vector] = Lanes::zeroed();
    value_sums[vector] = Lanes::zeroed();
  }
  let mut slope_sum = f32::new(0.0);

  // The queries, scaled, and the gradients of their outputs, then the
  // log-sum-exp of every query and the dot product of its output and the
  // output's gradient; the queries are all on the board, as a cell off it
  // still has a row of the attention.
  let mut queries = Shared::<[Lanes]>::new_slice(block * vectors);
  let mut grads = Shared::<[Lanes]>::new_slice(block * vectors);
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
    for vector in 0..vectors {
      let start = (rows + loaded) * channels + vector * LANES;
      queries[unit * vectors + vector] = read_lanes::<F>(query, start, scale);
      grads[unit * vectors + vector] = read_lanes::<F>(grad, start, 1.0);
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
          let mut scores = Lanes::zeroed();
          let mut grad_probabilities = Lanes::zeroed();
          #[unroll]
          for vector in 0..vectors {
            scores += queries[other * vectors + vector] * own_key[vector];
            grad_probabilities += grads[other * vectors + vector] * own_value[vector];
          }
          let distance = distance(query_rows[other], query_columns[other], row, column);
          let score = lanes_sum(scores) - slope * distance;
          let probability = (score - totals[other]).exp();
          let grad_score = probability * (lanes_sum(grad_probabilities) - output_dot_grads[other]);
          let probability_lanes = Lanes::broadcast(probability);
          let grad_score_lanes = Lanes::broadcast(grad_score);
          #[unroll]
          for vector in 0..vectors {
            value_sums[vector] += probability_lanes * grads[other * vectors + vector];
            key_sums[vector] += grad_score_lanes * queries[other * vectors + vector];
          }
          slope_sum -= grad_score * distance;
        }
      }
    }
  }

  if cell < cells {
    #[unroll]
    for vector in 0..vectors {
      let key_sum = key_sums[vector];
      let value_sum = value_sums[vector];
      #[unroll]
      for lane in 0..LANES {
        let index = (rows + cell) * channels + vector * LANES + lane;
        grad_key[index] = F::cast_from(key_sum.extract(lane));
        grad_value[index] = F::cast_from(value_sum.extract(lane));
      }
    }
    grad_slopes[rows + cell] = slope_sum;
  }
}

/// The tensors every kernel reads, contiguous, `on_board` and `slopes` in
/// single precision, and the dimensions of the queries.
pub(super) struct Inputs {
  pub(super) query: CubeTensor,
  pub(super) key: CubeTensor,
  pub(super) value: CubeTensor,
  pub(super) on_board: CubeTensor,
  pub(super) slopes: CubeTensor,
  pub(super) dims: [usize; 4],
}

impl Inputs {
  pub(super) fn new(
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
    assert!(
      dims[3].is_multiple_of(LANES),
      "the kernels read the channels of a head four at a time"
    );
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

  pub(super) fn empty(&self, shape: Shape, dtype: DType) -> CubeTensor {
    let buffer = self.query.client.empty(shape.num_elements() * dtype.size());
    CubeTensor::new_contiguous(
      self.query.client.clone(),
      self.query.device.clone(),
      shape,
      buffer,
      dtype,
    )
  }

  /// A cube per `block * per_unit` cells of every head of every board, a
  /// unit per `per_unit` cells.
  fn launch_dims(&self, per_unit: usize) -> (CubeCount, CubeDim, usize) {
    let [batch, heads, cells, channels] = self.dims;
    let block = block(channels);
    (
      CubeCount::Static(cells.div_ceil(block * per_unit) as u32, heads as u32, batch as u32),
      CubeDim::new_1d(block as u32),
      block,
    )
  }
}

/// The output and the log-sum-exp of every query, by [`forward_kernel`].
pub(super) fn forward(inputs: Inputs, width: usize) -> (CubeTensor, CubeTensor) {
  let [batch, heads, cells, channels] = inputs.dims;
  let dtype = inputs.query.dtype;
  let output = inputs.empty(inputs.query.meta.shape().clone(), dtype);
  let log_sum_exp = inputs.empty(Shape::new([batch, heads, cells]), DType::F32);
  let queries = FORWARD_QUERIES;
  let (cube_count, cube_dim, block) = inputs.launch_dims(queries);
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
    queries,
    dtype_to_storage_type(dtype),
  );
  (output, log_sum_exp)
}

/// The gradients of the queries, the keys and the values, and the parts of
/// the gradient of the slopes of every key, `[batch, heads, cells]`, by
/// [`grad_query_kernel`] and [`grad_key_value_kernel`], from the output, the
/// log-sum-exp and the output's gradient, contiguous, the output and its
/// gradient in the float type of the queries.
pub(super) fn backward(
  inputs: Inputs,
  output: CubeTensor,
  log_sum_exp: CubeTensor,
  grad: CubeTensor,
  width: usize,
) -> (CubeTensor, CubeTensor, CubeTensor, CubeTensor) {
  let [batch, heads, cells, channels] = inputs.dims;
  let dtype = inputs.query.dtype;
  let shape = inputs.query.meta.shape().clone();
  let grad_query = inputs.empty(shape.clone(), dtype);
  let grad_key = inputs.empty(shape.clone(), dtype);
  let grad_value = inputs.empty(shape, dtype);
  let grad_slopes = inputs.empty(Shape::new([batch, heads, cells]), DType::F32);
  let scale = scale(channels) as f32;
  let client = inputs.query.client.clone();
  let queries = GRAD_QUERIES;
  let (cube_count, cube_dim, block) = inputs.launch_dims(queries);
  grad_query_kernel::launch(
    &client,
    cube_count,
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
    queries,
    dtype_to_storage_type(dtype),
  );
  let (cube_count, cube_dim, block) = inputs.launch_dims(1);
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
  (grad_query, grad_key, grad_value, grad_slopes)
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
    if mma::serves(&inputs) {
      mma::forward(inputs, width)
    } else {
      forward(inputs, width)
    }
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
    let heads = inputs.dims[1];
    let dtype = inputs.query.dtype;
    let output = into_contiguous(Self::float_cast(output, dtype.into()));
    let grad = into_contiguous(Self::float_cast(grad, dtype.into()));
    let log_sum_exp = into_contiguous(log_sum_exp);
    let (grad_query, grad_key, grad_value, grad_slopes) = if mma::serves(&inputs) {
      mma::backward(inputs, output, log_sum_exp, grad, width)
    } else {
      backward(inputs, output, log_sum_exp, grad, width)
    };
    let grad_slopes = Self::float_sum_dim(Self::float_sum_dim(grad_slopes, 2), 0);
    let grad_slopes = Self::float_reshape(grad_slopes, Shape::new([heads]));
    let grad_slopes = Self::float_cast(grad_slopes, slopes_dtype.into());
    (grad_query, grad_key, grad_value, grad_slopes)
  }
}
