//! The kernels of the distance attention on the matrix units, for queries,
//! keys and values in half precision.
//!
//! A plane of 32 lanes owns 16 queries and multiplies `16 x 16 x 16` tiles,
//! its lanes holding the operands in registers as the GPU lays them out. The
//! kernels are written for the layout where lane `l` holds, as `8` elements
//! `i`:
//!
//! - of `A`, `m x k`: row `l % 16`, columns `8 * (l / 16) + i`;
//! - of `B`, `k x n`: rows `8 * (l / 16) + i`, column `l % 16`;
//! - of the accumulator, `m x n`: rows `8 * (l / 16) + i`, column `l % 16`;
//!
//! the layout of the matrix units of AMD's RDNA4. Every product is taken
//! transposed, the plane's own cells being the columns. In the forward pass
//! the scores are `Sᵀ = K Qᵀ`, so that lane `l` holds the scores of query
//! `l % 16` with 8 of the keys of a tile, which are, element by element, what
//! it has to hold of the probabilities `Pᵀ` as the `B` of `Oᵀ += Vᵀ Pᵀ`, and
//! the accumulator of `Oᵀ` holds 8 of the channels of that same query. The
//! softmax of a query is thus computed by the two lanes `l` and `l ^ 16`
//! alone, out of their own registers and one exchange between them, and its
//! rescaling of the output touches no other lane's. The backward pass takes
//! the same turn: the gradients of the scores come out of the accumulators of
//! the scores and of the probabilities' gradients laid out as the `B` of the
//! products that take them.
//!
//! The operands of a product that are not the plane's own cells' come out of
//! shared memory, every cell's channels in vectors of 8, or transposed, every
//! channel's cells, for the operands whose rows are the channels.
//!
//! Whether a device lays the tiles out so is checked once by
//! [`layout_matches`].
//!
//! The scores, the softmax and the sums are in single precision, as in
//! [`cube`](super::cube), and so is the accumulation of the products; only
//! their operands are in half precision: the queries, keys and values, and
//! the probabilities.

use super::{
  composed::scale,
  cube::{Inputs, NO_SCORE, distance, position},
};
use burn::tensor::{DType, Shape};
use burn_cubecl::{CubeDevice, tensor::CubeTensor};
use cubecl::{
  CubeCount, CubeDim,
  client::Client,
  cmma::{MatrixIdent, MmaDefinition},
  ir::{ElemType, FloatKind, features::MmaConfig},
  prelude::*,
};
use std::{collections::HashMap, sync::Mutex};

/// The lanes of a plane the kernels are written for.
const PLANE: usize = 32;

/// The side of a tile of the matrix units.
const TILE: usize = 16;

/// How many elements of a tile's operand a lane holds: half of a row or a
/// column.
const HALF: usize = 8;

/// How many planes a cube takes, each owning [`TILE`] queries, or keys in
/// [`grad_key_value_kernel`].
const PLANES: usize = 4;

/// How many cells a cube streams through shared memory at a time, [`TILE`] of
/// them per product: keys and values, or queries and the gradients of their
/// outputs in [`grad_key_value_kernel`].
const BLOCK: usize = 64;

/// The elements of a tile's `A` or `B` a lane holds.
type Operand<F> = Vector<F, Const<HALF>>;

/// An element of a tile's accumulator a lane holds; it holds [`HALF`] of
/// them.
type Accumulated = Vector<f32, Const<1>>;

/// Reports the layout of the operands and the accumulator of a `16 x 16 x 16`
/// product: the number of lanes of a plane and the elements a lane holds of
/// each, then the row and the column of every element of every lane, for `A`,
/// `B` and the accumulator in turn.
#[cube(launch)]
fn layout_kernel<F: Float>(output: &mut Tensor<u32>, #[define(F)] _dtype: ElemType) {
  let definition = MmaDefinition::<F, F, f32>::new(TILE, TILE, TILE);
  let lane = UNIT_POS_PLANE;
  let a = definition.elems_per_lane(MatrixIdent::A);
  let b = definition.elems_per_lane(MatrixIdent::B);
  let accumulator = definition.elems_per_lane(MatrixIdent::Accumulator);
  if lane == 0 {
    output[0] = PLANE_DIM;
    output[1] = a as u32;
    output[2] = b as u32;
    output[3] = accumulator as u32;
  }
  #[unroll]
  for i in 0..HALF {
    let start = 4 + (lane as usize * HALF + i) * 2;
    if i < a {
      let (row, column) = definition.position_of_nth(lane, i as u32, MatrixIdent::A);
      output[start] = row;
      output[start + 1] = column;
    }
    if i < b {
      let (row, column) = definition.position_of_nth(lane, i as u32, MatrixIdent::B);
      output[start + 2 * PLANE * HALF] = row;
      output[start + 2 * PLANE * HALF + 1] = column;
    }
    if i < accumulator {
      let (row, column) = definition.position_of_nth(lane, i as u32, MatrixIdent::Accumulator);
      output[start + 4 * PLANE * HALF] = row;
      output[start + 4 * PLANE * HALF + 1] = column;
    }
  }
}

/// What [`layout_kernel`] reports on a device whose layout the kernels are
/// written for.
fn expected_layout() -> Vec<u32> {
  let mut layout = vec![PLANE as u32, HALF as u32, HALF as u32, HALF as u32];
  type Position = fn(usize, usize) -> (usize, usize);
  let tiles: [Position; 3] = [
    |lane, i| (lane % TILE, HALF * (lane / TILE) + i),
    |lane, i| (HALF * (lane / TILE) + i, lane % TILE),
    |lane, i| (HALF * (lane / TILE) + i, lane % TILE),
  ];
  for position in tiles {
    for lane in 0..PLANE {
      for i in 0..HALF {
        let (row, column) = position(lane, i);
        layout.extend([row as u32, column as u32]);
      }
    }
  }
  layout
}

/// The layout [`layout_kernel`] reports on the device of `client`.
fn layout(client: &Client, device: &CubeDevice) -> Vec<u32> {
  let len = 4 + 3 * 2 * PLANE * HALF;
  let output = CubeTensor::new_contiguous(
    client.clone(),
    device.clone(),
    Shape::new([len]),
    client.empty(len * size_of::<u32>()),
    DType::U32,
  );
  layout_kernel::launch(
    client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new_1d(PLANE as u32),
    output.clone().into_tensor_arg(),
    ElemType::Float(FloatKind::F16),
  );
  let bytes = client.read_one(output.handle).expect("the layout is read back");
  bytes
    .as_chunks::<{ size_of::<u32>() }>()
    .0
    .iter()
    .map(|chunk| u32::from_ne_bytes(*chunk))
    .collect()
}

/// An element of the `A` [`product_kernel`] multiplies: a small integer, which
/// the product of two, summed over `16`, keeps exact in any float type.
#[cube]
fn probe_a(row: usize, column: usize) -> f32 {
  f32::cast_from(((row * 3 + column * 7) % 5) as u32) - 2.0
}

/// An element of the `B` [`product_kernel`] multiplies, as [`probe_a`].
#[cube]
fn probe_b(row: usize, column: usize) -> f32 {
  f32::cast_from(((row * 5 + column * 3) % 7) as u32) - 3.0
}

/// Multiplies [`probe_a`] by [`probe_b`] on the matrix units, the operands and
/// the result laid out as the kernels here lay them out, into `output`, `16 x
/// 16` row after row. Unlike [`layout_kernel`], which reports how the compiler
/// believes the tiles to be laid out, this tells whether the device actually
/// lays them out so: the belief is one for every device of a process, and
/// could be another device's.
#[cube(launch)]
fn product_kernel<F: Float>(output: &mut Tensor<f32>, #[define(F)] _dtype: ElemType) {
  let definition = MmaDefinition::<F, F, f32>::new(TILE, TILE, TILE);
  let lane = UNIT_POS_PLANE as usize;
  let mut a = Operand::<F>::empty();
  let mut b = Operand::<F>::empty();
  #[unroll]
  for i in 0..HALF {
    a.insert(i, F::cast_from(probe_a(lane % TILE, HALF * (lane / TILE) + i)));
    b.insert(i, F::cast_from(probe_b(HALF * (lane / TILE) + i, lane % TILE)));
  }
  let mut c = zeroed_tile();
  definition.execute_inplace(&operand::<F>(a), &operand::<F>(b), &mut c);
  #[unroll]
  for i in 0..HALF {
    output[(HALF * (lane / TILE) + i) * TILE + lane % TILE] = c[i].extract(0usize);
  }
}

/// Whether [`product_kernel`] multiplies right on the device of `client`.
fn multiplies(client: &Client, device: &CubeDevice) -> bool {
  let len = TILE * TILE;
  let output = CubeTensor::new_contiguous(
    client.clone(),
    device.clone(),
    Shape::new([len]),
    client.empty(len * size_of::<f32>()),
    DType::F32,
  );
  product_kernel::launch(
    client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new_1d(PLANE as u32),
    output.clone().into_tensor_arg(),
    ElemType::Float(FloatKind::F16),
  );
  let bytes = client.read_one(output.handle).expect("the product is read back");
  let product = bytes
    .as_chunks::<{ size_of::<f32>() }>()
    .0
    .iter()
    .map(|chunk| f32::from_ne_bytes(*chunk));
  let a = |row: usize, column: usize| ((row * 3 + column * 7) % 5) as f32 - 2.0;
  let b = |row: usize, column: usize| ((row * 5 + column * 3) % 7) as f32 - 3.0;
  product.enumerate().all(|(index, value)| {
    let (row, column) = (index / TILE, index % TILE);
    value == (0..TILE).map(|k| a(row, k) * b(k, column)).sum::<f32>()
  })
}

/// The `16 x 16 x 16` product of `dtype` into single precision, as the
/// device's features list it.
fn product(dtype: ElemType) -> MmaConfig {
  MmaConfig {
    a_type: dtype,
    b_type: dtype,
    cd_type: ElemType::Float(FloatKind::F32),
    m: TILE as u32,
    n: TILE as u32,
    k: TILE as u32,
  }
}

/// Whether the device of `client` lays the tiles out as the kernels are
/// written for, checked once per device.
///
/// CubeCL lays the tiles out on every AMD device of a process as on the
/// first one it set up, so in a process with GPUs of both RDNA3 and RDNA4
/// only those of the first one's generation pass.
fn layout_matches(client: &Client, device: &CubeDevice) -> bool {
  static MATCHES: Mutex<Option<HashMap<String, bool>>> = Mutex::new(None);
  let mut matches = MATCHES.lock().unwrap();
  *matches
    .get_or_insert_default()
    .entry(format!("{device:?}"))
    .or_insert_with(|| {
      let properties = client.properties();
      let f16 = product(ElemType::Float(FloatKind::F16));
      if properties.hardware.plane_size_min != PLANE as u32
        || properties.hardware.plane_size_max != PLANE as u32
        || !properties.features.matmul.mma.contains(&f16)
      {
        return false;
      }
      let matches = layout(client, device) == expected_layout() && multiplies(client, device);
      if !matches {
        log::warn!(
          "{device:?} has matrix units, but not laid out as the distance attention's kernels for them are \
           written for; it runs the kernels without them"
        );
      }
      matches
    })
}

/// The element of the matrix units' operands of `dtype`, if they take it.
fn operand_type(dtype: DType) -> Option<ElemType> {
  match dtype {
    DType::F16 => Some(ElemType::Float(FloatKind::F16)),
    DType::BF16 => Some(ElemType::Float(FloatKind::BF16)),
    _ => None,
  }
}

/// Whether the kernels here serve `inputs`: queries, keys and values in half
/// precision, a multiple of [`TILE`] channels per head, and a device that
/// multiplies them as the kernels are written for.
pub(super) fn serves(inputs: &Inputs) -> bool {
  let query = &inputs.query;
  let Some(dtype) = operand_type(query.dtype) else {
    return false;
  };
  inputs.dims[3].is_multiple_of(TILE)
    && query.client.properties().features.matmul.mma.contains(&product(dtype))
    && layout_matches(&query.client, &query.device)
}

/// A tile's accumulator of zeros.
#[cube]
fn zeroed_tile() -> Array<Accumulated> {
  let mut tile = Array::<Accumulated>::new(HALF);
  #[unroll]
  for i in 0..HALF {
    tile[i] = Accumulated::zeroed();
  }
  tile
}

/// Accumulators of zeros, one per chunk of 16 channels.
#[cube]
fn zeroed_tiles(#[comptime] chunks: usize) -> Sequence<Array<Accumulated>> {
  let mut tiles = Sequence::<Array<Accumulated>>::new();
  #[unroll]
  for _ in 0..chunks {
    tiles.push(zeroed_tile());
  }
  tiles
}

/// An operand of the [`HALF`] elements of `vector`.
#[cube]
fn operand<F: Float>(vector: Operand<F>) -> Array<Operand<F>> {
  let mut operand = Array::<Operand<F>>::new(1usize);
  operand[0] = vector;
  operand
}

/// The lane's `B` of the channels of its own cell, which `tensor` holds from
/// `start` on, one per chunk of 16 channels: as the columns of the cells of
/// the tile, the channels of the chunk its rows, it holds their half `half`.
#[cube]
fn own_operands<F: Float>(
  tensor: &Tensor<F>,
  start: usize,
  half: usize,
  #[comptime] chunks: usize,
) -> Sequence<Array<Operand<F>>> {
  let mut operands = Sequence::<Array<Operand<F>>>::new();
  #[unroll]
  for chunk in 0..chunks {
    let mut vector = Operand::<F>::empty();
    #[unroll]
    for i in 0..HALF {
      vector.insert(i, tensor[start + chunk * TILE + half * HALF + i]);
    }
    operands.push(operand::<F>(vector));
  }
  operands
}

/// Loads the channels of the [`BLOCK`] cells of `tensor` from `start` on into
/// `block`, every cell's in vectors of [`HALF`], every unit of the cube a
/// share of them; a cell past the last one loads the last one.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_rows<F: Float>(
  tensor: &Tensor<F>,
  rows: usize,
  start: usize,
  cells: usize,
  unit: usize,
  block: &mut Shared<[Operand<F>]>,
  #[comptime] channels: usize,
  #[comptime] units: usize,
) {
  let vectors = comptime!(channels / HALF);
  #[unroll]
  for round in 0..comptime!((BLOCK * vectors).div_ceil(units)) {
    let index = round * units + unit;
    if index < BLOCK * vectors {
      let loaded = (start + index / vectors).min(cells - 1);
      let from = (rows + loaded) * channels + (index % vectors) * HALF;
      let mut vector = Operand::<F>::empty();
      #[unroll]
      for i in 0..HALF {
        vector.insert(i, tensor[from + i]);
      }
      block[index] = vector;
    }
  }
}

/// Loads the channels of the [`BLOCK`] cells of `tensor` from `start` on into
/// `block` transposed, every channel's cells in vectors of [`HALF`], as
/// [`load_rows`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_columns<F: Float>(
  tensor: &Tensor<F>,
  rows: usize,
  start: usize,
  cells: usize,
  unit: usize,
  block: &mut Shared<[Operand<F>]>,
  #[comptime] channels: usize,
  #[comptime] units: usize,
) {
  let groups = comptime!(BLOCK / HALF);
  #[unroll]
  for round in 0..comptime!((channels * groups).div_ceil(units)) {
    let index = round * units + unit;
    if index < channels * groups {
      let channel = index / groups;
      let first = start + (index % groups) * HALF;
      let mut vector = Operand::<F>::empty();
      #[unroll]
      for i in 0..HALF {
        let loaded = (first + i).min(cells - 1);
        vector.insert(i, tensor[(rows + loaded) * channels + channel]);
      }
      block[index] = vector;
    }
  }
}

/// The `A` of the tile of cells from `first` on of a block that
/// [`load_rows`] loaded, the cells its rows and the chunk `chunk` of their
/// channels its columns.
#[cube]
fn row_operand<F: Float>(
  block: &Shared<[Operand<F>]>,
  first: usize,
  chunk: usize,
  lane: usize,
  #[comptime] channels: usize,
) -> Array<Operand<F>> {
  operand::<F>(block[(first + lane % TILE) * (channels / HALF) + chunk * 2 + lane / TILE])
}

/// The `A` of the tile of cells from `first` on of a block that
/// [`load_columns`] loaded, the chunk `chunk` of their channels its rows and
/// the cells its columns.
#[cube]
fn column_operand<F: Float>(
  block: &Shared<[Operand<F>]>,
  first: usize,
  chunk: usize,
  lane: usize,
) -> Array<Operand<F>> {
  operand::<F>(block[(chunk * TILE + lane % TILE) * (BLOCK / HALF) + first / HALF + lane / TILE])
}

/// Writes the lane's elements of the accumulators `tiles`, one per chunk of 16
/// channels, of the cell whose channels `tensor` holds from `start` on, times
/// `scale`: those of its half `half`.
#[cube]
fn store_tiles<F: Float>(
  tensor: &mut Tensor<F>,
  tiles: &Sequence<Array<Accumulated>>,
  start: usize,
  half: usize,
  scale: f32,
  #[comptime] chunks: usize,
) {
  #[unroll]
  for chunk in 0..chunks {
    let tile = tiles.index(chunk);
    #[unroll]
    for i in 0..HALF {
      tensor[start + chunk * TILE + half * HALF + i] = F::cast_from(tile[i].extract(0usize) * scale);
    }
  }
}

/// One cube per `planes` tiles of queries of a head of a board: the output
/// and the log-sum-exp of every query.
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
  #[comptime] planes: usize,
  #[define(F)] _dtype: ElemType,
) {
  let definition = MmaDefinition::<F, F, f32>::new(TILE, TILE, TILE);
  let chunks = comptime!(channels / TILE);
  let units = comptime!(PLANE * planes);
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let lane = UNIT_POS_X as usize;
  let unit = UNIT_POS_Y as usize * PLANE + lane;
  // The lane's query, and which half of the keys of a tile and of the
  // channels of a chunk it holds.
  let cell = (CUBE_POS_X as usize * planes + UNIT_POS_Y as usize) * TILE + lane % TILE;
  let half = lane / TILE;
  let own = cell.min(cells - 1);
  let (row, column) = position(own, width);
  let slope = slopes[head];

  // `Qᵀ`, the `B` of the scores.
  let queries = own_operands::<F>(query, (rows + own) * channels, half, chunks);
  // `Oᵀ`.
  let mut sums = zeroed_tiles(chunks);
  let mut max = f32::new(NO_SCORE);
  // The lane's part of the denominator: that of its half of the keys.
  let mut denominator = f32::new(0.0);

  let mut keys = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut values = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut keys_on_board = Shared::<[f32]>::new_slice(BLOCK);
  let mut key_rows = Shared::<[f32]>::new_slice(BLOCK);
  let mut key_columns = Shared::<[f32]>::new_slice(BLOCK);

  for block in 0..cells.div_ceil(BLOCK) {
    let start = block * BLOCK;
    sync_cube();
    load_rows::<F>(key, rows, start, cells, unit, &mut keys, channels, units);
    load_columns::<F>(value, rows, start, cells, unit, &mut values, channels, units);
    if unit < BLOCK {
      let other = start + unit;
      let loaded = other.min(cells - 1);
      let mut on = on_board[batch * cells + loaded];
      if other >= cells {
        on = 0.0;
      }
      keys_on_board[unit] = on;
      let (key_row, key_column) = position(loaded, width);
      key_rows[unit] = key_row;
      key_columns[unit] = key_column;
    }
    sync_cube();

    #[unroll]
    for tile in 0..comptime!(BLOCK / TILE) {
      let first = tile * TILE;
      // `Sᵀ = K Qᵀ`.
      let mut scores = zeroed_tile();
      #[unroll]
      for chunk in 0..chunks {
        let keys = row_operand::<F>(&keys, first, chunk, lane, channels);
        definition.execute_inplace(&keys, queries.index(chunk), &mut scores);
      }

      // The lane's scores, its half of the keys of the tile, and their
      // maximum over both halves.
      let mut tile_max = f32::new(NO_SCORE);
      #[unroll]
      for i in 0..HALF {
        let other = first + half * HALF + i;
        let score =
          scores[i].extract(0usize) * scale - slope * distance(row, column, key_rows[other], key_columns[other]);
        let mut kept = f32::new(NO_SCORE);
        if keys_on_board[other] > 0.5 {
          kept = score;
        }
        scores[i] = Accumulated::broadcast(kept);
        if kept > tile_max {
          tile_max = kept;
        }
      }
      let other_max = plane_shuffle_xor(tile_max, 16u32);
      if other_max > tile_max {
        tile_max = other_max;
      }
      let mut rescale = f32::new(1.0);
      if tile_max > max {
        rescale = (max - tile_max).exp();
        max = tile_max;
      }

      // `Pᵀ`, the `B` of the output.
      let mut probabilities = Operand::<F>::empty();
      let mut tile_sum = f32::new(0.0);
      #[unroll]
      for i in 0..HALF {
        let score = scores[i].extract(0usize);
        let mut probability = f32::new(0.0);
        if score > f32::new(NO_SCORE) {
          probability = (score - max).exp();
        }
        tile_sum += probability;
        probabilities.insert(i, F::cast_from(probability));
      }
      let probabilities = operand::<F>(probabilities);
      denominator = denominator * rescale + tile_sum;

      // `Oᵀ += Vᵀ Pᵀ`, rescaled first.
      let rescale = Accumulated::broadcast(rescale);
      #[unroll]
      for chunk in 0..chunks {
        let sum = sums.index_mut(chunk);
        #[unroll]
        for i in 0..HALF {
          sum[i] *= rescale;
        }
        let values = column_operand::<F>(&values, first, chunk, lane);
        definition.execute_inplace(&values, &probabilities, sum);
      }
    }
  }

  let denominator = denominator + plane_shuffle_xor(denominator, 16u32);
  if cell < cells {
    // A query with no key on the board attends to nothing, as in
    // `cube`.
    let mut normalizer = f32::new(0.0);
    let mut total = f32::new(-NO_SCORE);
    if denominator > 0.0 {
      normalizer = 1.0 / denominator;
      total = max + denominator.ln();
    }
    store_tiles::<F>(output, &sums, (rows + cell) * channels, half, normalizer, chunks);
    if half == 0 {
      log_sum_exp[rows + cell] = total;
    }
  }
}

/// One cube per `planes` tiles of queries of a head of a board: the
/// gradients of the queries, and the dot product of the output and its
/// gradient of every query, which [`grad_key_value_kernel`] takes.
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
  output_dot_grads: &mut Tensor<f32>,
  width: u32,
  scale: f32,
  #[comptime] channels: usize,
  #[comptime] planes: usize,
  #[define(F)] _dtype: ElemType,
) {
  let definition = MmaDefinition::<F, F, f32>::new(TILE, TILE, TILE);
  let chunks = comptime!(channels / TILE);
  let units = comptime!(PLANE * planes);
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let lane = UNIT_POS_X as usize;
  let unit = UNIT_POS_Y as usize * PLANE + lane;
  let cell = (CUBE_POS_X as usize * planes + UNIT_POS_Y as usize) * TILE + lane % TILE;
  let half = lane / TILE;
  let own = cell.min(cells - 1);
  let start = (rows + own) * channels;
  let (row, column) = position(own, width);
  let slope = slopes[head];
  let total = log_sum_exp[rows + own];

  // `Qᵀ` and `dOᵀ`, the `B` of the scores and of the probabilities'
  // gradients.
  let queries = own_operands::<F>(query, start, half, chunks);
  let grads = own_operands::<F>(grad, start, half, chunks);
  // `dQᵀ`.
  let mut sums = zeroed_tiles(chunks);
  let mut output_dot_grad = f32::new(0.0);
  #[unroll]
  for chunk in 0..chunks {
    #[unroll]
    for i in 0..HALF {
      let index = start + chunk * TILE + half * HALF + i;
      output_dot_grad += f32::cast_from(output[index]) * f32::cast_from(grad[index]);
    }
  }
  let output_dot_grad = output_dot_grad + plane_shuffle_xor(output_dot_grad, 16u32);

  let mut keys = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut keys_transposed = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut values = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut keys_on_board = Shared::<[f32]>::new_slice(BLOCK);
  let mut key_rows = Shared::<[f32]>::new_slice(BLOCK);
  let mut key_columns = Shared::<[f32]>::new_slice(BLOCK);

  for block in 0..cells.div_ceil(BLOCK) {
    let start = block * BLOCK;
    sync_cube();
    load_rows::<F>(key, rows, start, cells, unit, &mut keys, channels, units);
    load_columns::<F>(key, rows, start, cells, unit, &mut keys_transposed, channels, units);
    load_rows::<F>(value, rows, start, cells, unit, &mut values, channels, units);
    if unit < BLOCK {
      let other = start + unit;
      let loaded = other.min(cells - 1);
      let mut on = on_board[batch * cells + loaded];
      if other >= cells {
        on = 0.0;
      }
      keys_on_board[unit] = on;
      let (key_row, key_column) = position(loaded, width);
      key_rows[unit] = key_row;
      key_columns[unit] = key_column;
    }
    sync_cube();

    #[unroll]
    for tile in 0..comptime!(BLOCK / TILE) {
      let first = tile * TILE;
      // `Sᵀ = K Qᵀ` and `dPᵀ = V dOᵀ`.
      let mut scores = zeroed_tile();
      let mut grad_probabilities = zeroed_tile();
      #[unroll]
      for chunk in 0..chunks {
        let keys = row_operand::<F>(&keys, first, chunk, lane, channels);
        definition.execute_inplace(&keys, queries.index(chunk), &mut scores);
        let values = row_operand::<F>(&values, first, chunk, lane, channels);
        definition.execute_inplace(&values, grads.index(chunk), &mut grad_probabilities);
      }

      // `dSᵀ`, the `B` of the queries' gradients.
      let mut grad_scores = Operand::<F>::empty();
      #[unroll]
      for i in 0..HALF {
        let other = first + half * HALF + i;
        let score =
          scores[i].extract(0usize) * scale - slope * distance(row, column, key_rows[other], key_columns[other]);
        let mut grad_score = f32::new(0.0);
        if keys_on_board[other] > 0.5 {
          grad_score = (score - total).exp() * (grad_probabilities[i].extract(0usize) - output_dot_grad);
        }
        grad_scores.insert(i, F::cast_from(grad_score));
      }
      let grad_scores = operand::<F>(grad_scores);

      // `dQᵀ += Kᵀ dSᵀ`.
      #[unroll]
      for chunk in 0..chunks {
        let keys = column_operand::<F>(&keys_transposed, first, chunk, lane);
        definition.execute_inplace(&keys, &grad_scores, sums.index_mut(chunk));
      }
    }
  }

  if cell < cells {
    store_tiles::<F>(grad_query, &sums, (rows + cell) * channels, half, scale, chunks);
    if half == 0 {
      output_dot_grads[rows + cell] = output_dot_grad;
    }
  }
}

/// One cube per `planes` tiles of keys of a head of a board: the gradients of
/// the keys and the values, and every key's part of the gradient of the
/// head's slope.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn grad_key_value_kernel<F: Float>(
  query: &Tensor<F>,
  key: &Tensor<F>,
  value: &Tensor<F>,
  on_board: &Tensor<f32>,
  slopes: &Tensor<f32>,
  log_sum_exp: &Tensor<f32>,
  output_dot_grads: &Tensor<f32>,
  grad: &Tensor<F>,
  grad_key: &mut Tensor<F>,
  grad_value: &mut Tensor<F>,
  grad_slopes: &mut Tensor<f32>,
  width: u32,
  scale: f32,
  #[comptime] channels: usize,
  #[comptime] planes: usize,
  #[define(F)] _dtype: ElemType,
) {
  let definition = MmaDefinition::<F, F, f32>::new(TILE, TILE, TILE);
  let chunks = comptime!(channels / TILE);
  let units = comptime!(PLANE * planes);
  let heads = query.shape(1);
  let cells = query.shape(2);
  let width = width as usize;
  let batch = CUBE_POS_Z as usize;
  let head = CUBE_POS_Y as usize;
  let rows = (batch * heads + head) * cells;
  let lane = UNIT_POS_X as usize;
  let unit = UNIT_POS_Y as usize * PLANE + lane;
  // The lane's key, and which half of the queries of a tile and of the
  // channels of a chunk it holds.
  let cell = (CUBE_POS_X as usize * planes + UNIT_POS_Y as usize) * TILE + lane % TILE;
  let half = lane / TILE;
  let own = cell.min(cells - 1);
  let start = (rows + own) * channels;
  let (row, column) = position(own, width);
  let slope = slopes[head];
  let attended = cell < cells && on_board[batch * cells + own] > 0.5;

  // `Kᵀ` and `Vᵀ`, the `B` of the scores and of the probabilities'
  // gradients.
  let keys = own_operands::<F>(key, start, half, chunks);
  let values = own_operands::<F>(value, start, half, chunks);
  // `dKᵀ` and `dVᵀ`.
  let mut key_sums = zeroed_tiles(chunks);
  let mut value_sums = zeroed_tiles(chunks);
  let mut slope_sum = f32::new(0.0);

  // The queries, all on the board, as a cell off it still has a row of the
  // attention, and the gradients of their outputs, both as rows and
  // transposed; then the log-sum-exp of every query and the dot product of
  // its output and the output's gradient.
  let mut queries = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut queries_transposed = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut grads = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut grads_transposed = Shared::<[Operand<F>]>::new_slice(BLOCK * channels / HALF);
  let mut queries_in = Shared::<[f32]>::new_slice(BLOCK);
  let mut query_rows = Shared::<[f32]>::new_slice(BLOCK);
  let mut query_columns = Shared::<[f32]>::new_slice(BLOCK);
  let mut totals = Shared::<[f32]>::new_slice(BLOCK);
  let mut dots = Shared::<[f32]>::new_slice(BLOCK);

  for block in 0..cells.div_ceil(BLOCK) {
    let start = block * BLOCK;
    sync_cube();
    load_rows::<F>(query, rows, start, cells, unit, &mut queries, channels, units);
    load_columns::<F>(
      query,
      rows,
      start,
      cells,
      unit,
      &mut queries_transposed,
      channels,
      units,
    );
    load_rows::<F>(grad, rows, start, cells, unit, &mut grads, channels, units);
    load_columns::<F>(grad, rows, start, cells, unit, &mut grads_transposed, channels, units);
    if unit < BLOCK {
      let other = start + unit;
      let loaded = other.min(cells - 1);
      let mut query_in = f32::new(1.0);
      if other >= cells {
        query_in = 0.0;
      }
      queries_in[unit] = query_in;
      let (query_row, query_column) = position(loaded, width);
      query_rows[unit] = query_row;
      query_columns[unit] = query_column;
      totals[unit] = log_sum_exp[rows + loaded];
      dots[unit] = output_dot_grads[rows + loaded];
    }
    sync_cube();

    #[unroll]
    for tile in 0..comptime!(BLOCK / TILE) {
      let first = tile * TILE;
      // `S = Q Kᵀ` and `dP = dO Vᵀ`.
      let mut scores = zeroed_tile();
      let mut grad_probabilities = zeroed_tile();
      #[unroll]
      for chunk in 0..chunks {
        let queries = row_operand::<F>(&queries, first, chunk, lane, channels);
        definition.execute_inplace(&queries, keys.index(chunk), &mut scores);
        let grads = row_operand::<F>(&grads, first, chunk, lane, channels);
        definition.execute_inplace(&grads, values.index(chunk), &mut grad_probabilities);
      }

      // `P` and `dS`, the `B` of the values' and the keys' gradients.
      let mut probabilities = Operand::<F>::empty();
      let mut grad_scores = Operand::<F>::empty();
      #[unroll]
      for i in 0..HALF {
        let other = first + half * HALF + i;
        let distance = distance(query_rows[other], query_columns[other], row, column);
        let score = scores[i].extract(0usize) * scale - slope * distance;
        let mut probability = f32::new(0.0);
        if attended && queries_in[other] > 0.5 {
          probability = (score - totals[other]).exp();
        }
        let grad_score = probability * (grad_probabilities[i].extract(0usize) - dots[other]);
        slope_sum -= grad_score * distance;
        probabilities.insert(i, F::cast_from(probability));
        grad_scores.insert(i, F::cast_from(grad_score));
      }
      let probabilities = operand::<F>(probabilities);
      let grad_scores = operand::<F>(grad_scores);

      // `dVᵀ += dOᵀ P` and `dKᵀ += Qᵀ dS`.
      #[unroll]
      for chunk in 0..chunks {
        let grads = column_operand::<F>(&grads_transposed, first, chunk, lane);
        definition.execute_inplace(&grads, &probabilities, value_sums.index_mut(chunk));
        let queries = column_operand::<F>(&queries_transposed, first, chunk, lane);
        definition.execute_inplace(&queries, &grad_scores, key_sums.index_mut(chunk));
      }
    }
  }

  let slope_sum = slope_sum + plane_shuffle_xor(slope_sum, 16u32);
  if cell < cells {
    store_tiles::<F>(grad_key, &key_sums, (rows + cell) * channels, half, scale, chunks);
    store_tiles::<F>(grad_value, &value_sums, (rows + cell) * channels, half, 1.0, chunks);
    if half == 0 {
      grad_slopes[rows + cell] = slope_sum;
    }
  }
}

/// A cube per [`PLANES`] tiles of cells of every head of every board, a plane
/// per tile.
fn launch_dims(inputs: &Inputs) -> (CubeCount, CubeDim) {
  let [batch, heads, cells, _] = inputs.dims;
  (
    CubeCount::Static(cells.div_ceil(TILE * PLANES) as u32, heads as u32, batch as u32),
    CubeDim::new_2d(PLANE as u32, PLANES as u32),
  )
}

/// The output and the log-sum-exp of every query, by [`forward_kernel`].
pub(super) fn forward(inputs: Inputs, width: usize) -> (CubeTensor, CubeTensor) {
  let [batch, heads, cells, channels] = inputs.dims;
  let dtype = inputs.query.dtype;
  let output = inputs.empty(inputs.query.meta.shape().clone(), dtype);
  let log_sum_exp = inputs.empty(Shape::new([batch, heads, cells]), DType::F32);
  let client = inputs.query.client.clone();
  let (cube_count, cube_dim) = launch_dims(&inputs);
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
    PLANES,
    operand_type(dtype).unwrap(),
  );
  (output, log_sum_exp)
}

/// The gradients of the queries, the keys and the values, and the parts of
/// the gradient of the slopes of every key, by [`grad_query_kernel`] and
/// [`grad_key_value_kernel`]; see [`cube::backward`](super::cube::backward).
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
  let per_cell = Shape::new([batch, heads, cells]);
  let output_dot_grads = inputs.empty(per_cell.clone(), DType::F32);
  let grad_slopes = inputs.empty(per_cell, DType::F32);
  let scale = scale(channels) as f32;
  let operand_type = operand_type(dtype).unwrap();
  let client = inputs.query.client.clone();
  let (cube_count, cube_dim) = launch_dims(&inputs);
  grad_query_kernel::launch(
    &client,
    cube_count.clone(),
    cube_dim,
    inputs.query.clone().into_tensor_arg(),
    inputs.key.clone().into_tensor_arg(),
    inputs.value.clone().into_tensor_arg(),
    inputs.on_board.clone().into_tensor_arg(),
    inputs.slopes.clone().into_tensor_arg(),
    output.into_tensor_arg(),
    log_sum_exp.clone().into_tensor_arg(),
    grad.clone().into_tensor_arg(),
    grad_query.clone().into_tensor_arg(),
    output_dot_grads.clone().into_tensor_arg(),
    width as u32,
    scale,
    channels,
    PLANES,
    operand_type,
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
    log_sum_exp.into_tensor_arg(),
    output_dot_grads.into_tensor_arg(),
    grad.into_tensor_arg(),
    grad_key.clone().into_tensor_arg(),
    grad_value.clone().into_tensor_arg(),
    grad_slopes.clone().into_tensor_arg(),
    width as u32,
    scale,
    channels,
    PLANES,
    operand_type,
  );
  (grad_query, grad_key, grad_value, grad_slopes)
}

#[cfg(all(test, feature = "rocm"))]
mod tests {
  use super::{super::cube, Inputs, layout_matches, serves};
  use burn::{
    backend::ops::FloatTensorOps,
    tensor::{FloatDType, TensorData},
  };
  use burn_cubecl::{CubeBackend, CubeDevice, ops::into_data_sync, tensor::CubeTensor};

  /// The first ROCm device the kernels serve, if any. In a process with
  /// GPUs of both RDNA3 and RDNA4 that is only one of the generation of the
  /// first device set up, so to test on an RDNA4 GPU next to an RDNA3 one,
  /// `HIP_VISIBLE_DEVICES` has to hide the latter.
  fn device() -> Option<CubeDevice> {
    let device = (0..)
      .map_while(|index| CubeDevice::rocm(index).ok())
      .find(|device| layout_matches(&device.client(), device));
    if device.is_none() {
      println!("no ROCm device lays the tiles out as the kernels are written for, skipping");
    }
    device
  }

  /// `len` numbers spread evenly with a standard deviation of one, the same
  /// for the same `seed`.
  fn numbers(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
      .map(|_| {
        state = state
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 12f32.sqrt()
      })
      .collect()
  }

  fn tensor(values: Vec<f32>, shape: &[usize], device: &CubeDevice) -> CubeTensor {
    CubeBackend::float_from_data(TensorData::new(values, shape.to_vec()), device)
  }

  fn values(tensor: CubeTensor) -> Vec<f32> {
    into_data_sync(CubeBackend::float_cast(tensor, FloatDType::F32))
      .convert::<f32>()
      .try_to_vec::<f32>()
      .unwrap()
  }

  /// The largest difference between `actual` and `expected`, relative to the
  /// largest of `expected`.
  fn max_difference(actual: &[f32], expected: &[f32]) -> f32 {
    let scale = expected.iter().fold(1e-3f32, |max, value| max.max(value.abs()));
    actual
      .iter()
      .zip(expected)
      .fold(0f32, |max, (actual, expected)| max.max((actual - expected).abs()))
      / scale
  }

  /// The inputs of a batch of two boards `height x width`, the first one
  /// whole and the second with only its `height - 2 x width - 3` corner on
  /// the board, with three heads of `channels`, in `dtype`.
  fn inputs(device: &CubeDevice, channels: usize, (height, width): (usize, usize), dtype: FloatDType) -> Inputs {
    let (batch, heads, cells) = (2, 3, height * width);
    let shape = [batch, heads, cells, channels];
    let len = batch * heads * cells * channels;
    let random = |seed| CubeBackend::float_cast(tensor(numbers(len, seed), &shape, device), dtype);
    let on_board = (0..batch * cells)
      .map(|index| {
        let cell = index % cells;
        (index < cells || (cell / width < height - 2 && cell % width < width - 3)) as u8 as f32
      })
      .collect();
    Inputs::new(
      random(1),
      random(2),
      random(3),
      tensor(on_board, &[batch, cells], device),
      tensor(vec![0.05, 0.3, 1.5], &[heads], device),
    )
  }

  /// The output and the log-sum-exp of the kernels here agree with those of
  /// the kernels without the matrix units on the same inputs in half
  /// precision, for boards that take several blocks of keys and tiles of
  /// queries, with cells off the board.
  #[test]
  fn forward_agrees() {
    let Some(device) = device() else {
      return;
    };
    for (dtype, tolerance) in [(FloatDType::F16, 2e-3), (FloatDType::BF16, 1e-2)] {
      for (channels, board) in [(16, (13, 11)), (32, (7, 11)), (32, (24, 24)), (48, (9, 10))] {
        let inputs = || inputs(&device, channels, board, dtype);
        assert!(serves(&inputs()));
        let (output, log_sum_exp) = super::forward(inputs(), board.1);
        let (expected_output, expected_log_sum_exp) = cube::forward(inputs(), board.1);
        let case = format!("{dtype:?}, {channels} channels, {board:?}");
        let difference = max_difference(&values(output), &values(expected_output));
        assert!(difference < tolerance, "output, {case}: {difference}");
        let difference = max_difference(&values(log_sum_exp), &values(expected_log_sum_exp));
        assert!(difference < 1e-4, "log-sum-exp, {case}: {difference}");
      }
    }
  }

  /// The gradients of the kernels here agree with those of the kernels
  /// without the matrix units on the same inputs in half precision, as in
  /// [`forward_agrees`].
  #[test]
  fn backward_agrees() {
    let Some(device) = device() else {
      return;
    };
    for (dtype, tolerance) in [(FloatDType::F16, 1e-2), (FloatDType::BF16, 3e-2)] {
      for (channels, board) in [(16, (13, 11)), (32, (7, 11)), (32, (24, 24)), (48, (9, 10))] {
        let inputs = || inputs(&device, channels, board, dtype);
        let (output, log_sum_exp) = cube::forward(inputs(), board.1);
        let shape = inputs().query.meta.shape().clone();
        let grad = CubeBackend::float_cast(tensor(numbers(shape.num_elements(), 4), &shape, &device), dtype);
        let backward = |backward: fn(Inputs, CubeTensor, CubeTensor, CubeTensor, usize) -> _| {
          let (query, key, value, slopes): (CubeTensor, CubeTensor, CubeTensor, CubeTensor) =
            backward(inputs(), output.clone(), log_sum_exp.clone(), grad.clone(), board.1);
          [query, key, value, slopes].map(values)
        };
        let actual = backward(super::backward);
        let expected = backward(cube::backward);
        for (name, (actual, expected)) in ["query", "key", "value", "slopes"]
          .iter()
          .zip(actual.iter().zip(&expected))
        {
          let difference = max_difference(actual, expected);
          let case = format!("{name}, {dtype:?}, {channels} channels, {board:?}");
          println!("{case}: {difference}");
          assert!(difference < tolerance, "{case}: {difference}");
        }
      }
    }
  }

  /// The time the kernels here and those without the matrix units take on a
  /// batch of full boards of the size of the largest training ones.
  #[test]
  #[ignore = "a benchmark"]
  fn forward_speed() {
    let Some(device) = device() else {
      return;
    };
    let client = device.client();
    let (batch, heads, cells, channels, width) = (64, 3, 24 * 24, 32, 24);
    let shape = [batch, heads, cells, channels];
    let len = batch * heads * cells * channels;
    let random = |seed| CubeBackend::float_cast(tensor(numbers(len, seed), &shape, &device), FloatDType::BF16);
    let (query, key, value) = (random(1), random(2), random(3));
    let on_board = tensor(vec![1.0; batch * cells], &[batch, cells], &device);
    let slopes = tensor(vec![0.05, 0.3, 1.5], &[heads], &device);
    let inputs = || {
      Inputs::new(
        query.clone(),
        key.clone(),
        value.clone(),
        on_board.clone(),
        slopes.clone(),
      )
    };
    let time = |forward: fn(Inputs, usize) -> (CubeTensor, CubeTensor)| {
      for _ in 0..3 {
        forward(inputs(), width);
      }
      futures::executor::block_on(client.sync()).unwrap();
      let runs = 20;
      let start = std::time::Instant::now();
      for _ in 0..runs {
        forward(inputs(), width);
      }
      futures::executor::block_on(client.sync()).unwrap();
      start.elapsed() / runs
    };
    let (output, log_sum_exp) = cube::forward(inputs(), width);
    let grad = random(4);
    let time_backward = |backward: fn(Inputs, CubeTensor, CubeTensor, CubeTensor, usize) -> _| {
      let run = || {
        let _: (CubeTensor, CubeTensor, CubeTensor, CubeTensor) =
          backward(inputs(), output.clone(), log_sum_exp.clone(), grad.clone(), width);
      };
      for _ in 0..3 {
        run();
      }
      futures::executor::block_on(client.sync()).unwrap();
      let runs = 20;
      let start = std::time::Instant::now();
      for _ in 0..runs {
        run();
      }
      futures::executor::block_on(client.sync()).unwrap();
      start.elapsed() / runs
    };
    // Alternating, so that neither gets the GPU before its clocks are up.
    for _ in 0..3 {
      println!(
        "forward: matrix units {:?}, without {:?}; backward: matrix units {:?}, without {:?}",
        time(super::forward),
        time(cube::forward),
        time_backward(super::backward),
        time_backward(cube::backward),
      );
    }
  }
}
