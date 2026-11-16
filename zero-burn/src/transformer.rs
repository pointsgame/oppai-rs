//! The attention trunk: nested bottleneck blocks whose inner layers are
//! transformer layers instead of convolutions. Every board cell is a token,
//! and nothing the network learns is tied to a cell: the positions of the
//! tokens enter attention through the offset between two cells only, so a
//! network trained on one board size runs on any other. It is also built to
//! play as well on boards bigger than any it was trained on, where an offset
//! can be longer than any seen in training and there are more tokens to attend
//! to:
//! - the rotary encoding turns at frequencies that complete a full turn
//!   within the smallest training board, so a longer offset gives it no angle
//!   it has not seen, and part of the channels of every head it does not turn
//!   at all, to match tokens by what they hold however far apart they are;
//! - how far apart two cells are is a penalty on the attention score that
//!   grows with the distance, which a bigger board only carries on;
//! - the attention scores are scaled with the logarithm of the number of
//!   cells on the board, so that a head stays as sharp with more of them to
//!   spread its attention over;
//! - optionally, a depthwise convolution in the feed-forward layers tells
//!   every token what its neighbors hold, the same way on every board.

use crate::{
  distance_attention::distance_attention,
  model::{NormActConv, TransformerConfig},
};
use burn::{
  module::{Initializer, Module, Param},
  tensor::{
    Device, Distribution, FloatDType, Tensor, TensorData,
    activation::{sigmoid, silu, softplus},
    module::{conv2d, linear},
    ops::ConvOptions,
    s,
  },
};
use std::f64::consts::PI;

/// The learned rotary frequencies start log-uniformly between the lowest one
/// allowed and this, in radians per cell: a quarter turn per cell, so that the
/// fastest pairs start telling neighbors apart and the slowest ones cells as
/// far apart as the smallest training board allows.
const MAX_INITIAL_ROPE_FREQUENCY: f64 = PI / 2.0;

/// The distance slopes of a layer's heads start geometrically from
/// `2^(-RANGE / heads)` down to `2^-RANGE` per cell: the steepest head mostly
/// attends to its neighborhood, the gentlest one to the whole board.
const INITIAL_DISTANCE_SLOPES_RANGE: f64 = 8.0;

const RMS_NORM_EPSILON: f64 = 1e-6;

/// The inverse of the softplus, to initialize what goes through one.
fn inverse_softplus(value: f64) -> f64 {
  value + (-(-value).exp_m1()).ln()
}

/// A linear layer without a bias, computing in the float type of its weight
/// and returning that type: it casts its input, so that
/// [`Model::with_compute_dtype`](crate::model::Model::with_compute_dtype) can
/// cast its weight alone, and what it returns stays in that type until it is
/// added back to the residual stream.
#[derive(Module, Debug)]
pub struct Projection {
  /// `[d_input, d_output]`.
  weight: Param<Tensor<2>>,
}

impl Projection {
  /// Initialized uniformly with the standard deviation `1 / sqrt(3 * d_input)`,
  /// as a linear layer is by default.
  pub fn new(device: &Device, d_input: usize, d_output: usize) -> Self {
    Self {
      weight: Initializer::KaimingUniform {
        gain: 1.0 / 3.0f64.sqrt(),
        fan_out_only: false,
      }
      .init_with([d_input, d_output], Some(d_input), Some(d_output), device),
    }
  }

  pub fn forward<const D: usize>(&self, inputs: Tensor<D>) -> Tensor<D> {
    let weight = self.weight.val();
    linear(inputs.cast(weight.dtype()), weight, None)
  }
}

/// Root mean square normalization over the last dimension - the channels of
/// a token, or of a head's query or key - followed by a learned scale per
/// channel.
#[derive(Module, Debug)]
pub struct RmsNorm {
  /// Centered at 1: it starts at zero and is applied as `gamma + 1`, so weight
  /// decay pulls the scale toward 1 rather than 0.
  gamma: Param<Tensor<1>>,
}

impl RmsNorm {
  pub fn new(device: &Device, channels: usize) -> Self {
    Self {
      gamma: Param::from_tensor(Tensor::zeros([channels], device)),
    }
  }

  /// The statistic is taken in single precision whatever the input's type:
  /// in half precision squaring overflows from 256 on, and the residual
  /// stream gets there.
  pub fn forward<const D: usize>(&self, inputs: Tensor<D>) -> Tensor<D> {
    let dtype = inputs.dtype();
    let inputs = inputs.cast(FloatDType::F32);
    let rms = (inputs.clone().square().mean_dim(D - 1) + RMS_NORM_EPSILON).sqrt();
    (inputs / rms).cast(dtype) * (self.gamma.val().cast(dtype) + 1.0).unsqueeze::<D>()
  }
}

/// The row and the column of every cell of a board, `[1, 1, cells, 1]` each,
/// that the rotary encodings turn by.
#[derive(Clone, Debug)]
pub struct Cells {
  rows: Tensor<4>,
  columns: Tensor<4>,
}

impl Cells {
  pub fn new(device: &Device, height: usize, width: usize) -> Self {
    let rows = (0..height * width)
      .map(|cell| (cell / width) as f32)
      .collect::<Vec<_>>();
    let columns = (0..height * width)
      .map(|cell| (cell % width) as f32)
      .collect::<Vec<_>>();
    let shape = [1, 1, height * width, 1];
    // In single precision whatever the device's float type, as the rotary
    // frequencies they are multiplied by.
    Self {
      rows: Tensor::<4>::from_data(TensorData::new(rows, shape), device).cast(FloatDType::F32),
      columns: Tensor::<4>::from_data(TensorData::new(columns, shape), device).cast(FloatDType::F32),
    }
  }
}

/// The learned frequencies of a layer's rotary encoding: every pair of
/// rotated channels of every head turns along its own direction of the board,
/// diagonals included, at its own rate.
///
/// The rate never falls below the frequency completing a full turn within the
/// smallest training board, so that every angle a longer offset on a bigger
/// board gives a pair is one training has shown it already. What lies further
/// than that is told by the distance penalty instead, see [`Attention`].
#[derive(Module, Debug)]
pub struct RotaryFrequencies {
  /// `[heads, pairs]`, the direction every pair turns along, in radians from
  /// the rows towards the columns.
  directions: Param<Tensor<2>>,
  /// `[heads, pairs]`, how much faster than `min_frequency` every pair turns,
  /// through a softplus.
  excess: Param<Tensor<2>>,
  /// The lowest frequency, in radians per cell.
  min_frequency: f64,
}

impl RotaryFrequencies {
  /// Directions are uniform, and frequencies log-uniform between
  /// `min_frequency` and [`MAX_INITIAL_ROPE_FREQUENCY`], or twice
  /// `min_frequency` if that is higher.
  pub fn new(device: &Device, heads: usize, pairs: usize, min_frequency: f64) -> Self {
    let shape = [heads, pairs];
    let max_frequency = MAX_INITIAL_ROPE_FREQUENCY.max(2.0 * min_frequency);
    let frequencies = Tensor::<2>::random(
      shape,
      Distribution::Uniform(min_frequency.ln(), max_frequency.ln()),
      device,
    )
    .exp();
    // Kept off the bound itself, where the softplus would have to be undone
    // from zero.
    let excess = (frequencies - min_frequency).clamp_min(min_frequency / 100.0);
    Self {
      directions: Param::from_tensor(Tensor::random(shape, Distribution::Uniform(-PI, PI), device)),
      excess: Param::from_tensor((excess.exp() - 1.0).log()),
      min_frequency,
    }
  }

  /// The radians every pair of every head turns per row and per column,
  /// `[heads, pairs, 2]`, in single precision.
  pub fn frequencies(&self) -> Tensor<3> {
    let directions = self.directions.val().cast(FloatDType::F32);
    let magnitudes = softplus(self.excess.val().cast(FloatDType::F32), 1.0) + self.min_frequency;
    Tensor::stack(
      vec![
        magnitudes.clone() * directions.clone().cos(),
        magnitudes * directions.sin(),
      ],
      2,
    )
  }
}

/// 2D rotary position encoding of the tokens of a board.
///
/// The first channels of each head are split into two halves whose `i`-th
/// channels form a pair, rotated as a complex number by an angle that is a
/// linear function of the cell's row and column, with the pair's own frequency
/// along each; the channels after them are left as they are. Rotating a query
/// and a key by the angles of their cells leaves their dot product depending
/// only on the difference of the angles, i.e. on the offset between the cells.
///
/// TODO: switch to burn's `RotaryEncoding` once it supports 2D positions.
#[derive(Clone, Debug)]
pub struct Rope {
  /// `[1, heads, cells, pairs]`.
  cos: Tensor<4>,
  sin: Tensor<4>,
}

impl Rope {
  /// `frequencies` is `[heads, pairs, 2]`: the radians every pair of every
  /// head turns per row and per column. The angles are computed in single
  /// precision whatever the frequencies' type: in half precision they would
  /// be off by a hundredth of a radian already across a board of 24 cells.
  pub fn new(frequencies: Tensor<3>, cells: &Cells) -> Self {
    let frequencies = frequencies.cast(FloatDType::F32);
    let [heads, pairs, _] = frequencies.dims();
    let along = |axis: usize| {
      frequencies
        .clone()
        .slice(s![.., .., axis..axis + 1])
        .reshape([1, heads, 1, pairs])
    };
    let angles = cells.rows.clone() * along(0) + cells.columns.clone() * along(1);
    Self {
      cos: angles.clone().cos(),
      sin: angles.sin(),
    }
  }

  /// Rotates the first `2 * pairs` channels of `[batch, heads, cells,
  /// head_channels]` queries or keys.
  pub fn apply(&self, inputs: Tensor<4>) -> Tensor<4> {
    let [.., channels] = inputs.dims();
    let [.., pairs] = self.cos.dims();
    let dtype = inputs.dtype();
    let cos = self.cos.clone().cast(dtype);
    let sin = self.sin.clone().cast(dtype);
    let first = inputs.clone().slice(s![.., .., .., 0..pairs]);
    let second = inputs.clone().slice(s![.., .., .., pairs..2 * pairs]);
    let mut parts = vec![
      first.clone() * cos.clone() - second.clone() * sin.clone(),
      first * sin + second * cos,
    ];
    if 2 * pairs < channels {
      parts.push(inputs.slice(s![.., .., .., 2 * pairs..]));
    }
    Tensor::cat(parts, 3)
  }
}

/// What every layer of a forward pass shares.
#[derive(Clone, Debug)]
pub struct AttentionContext {
  height: usize,
  width: usize,
  cells: Cells,
  /// `[batch, 1, 1, 1]`, the natural logarithm of the number of cells on the
  /// board, in single precision.
  log_area: Tensor<4>,
  /// `[batch, cells]`, one on the board and zero off it, in single
  /// precision: no token attends to a cell off the board.
  board: Tensor<2>,
  /// `[batch, cells, 1]`, one on the board and zero off it.
  on_board: Tensor<3>,
}

impl AttentionContext {
  /// For the `[batch, 1, height, width]` mask of the cells on the board.
  pub fn new(mask: &Tensor<4>) -> Self {
    let [batch, _, height, width] = mask.dims();
    let cells = Cells::new(&mask.device(), height, width);
    Self {
      height,
      width,
      cells,
      log_area: mask
        .clone()
        .cast(FloatDType::F32)
        .sum_dim(2)
        .sum_dim(3)
        .clamp_min(1.0)
        .log(),
      board: mask.clone().cast(FloatDType::F32).reshape([batch, height * width]),
      on_board: mask.clone().reshape([batch, height * width, 1]),
    }
  }
}

/// Multi-head self-attention over the tokens of the board, normalized first;
/// it returns the residual and leaves adding it to the caller.
///
/// The queries and the keys are normalized per head, so that how sharp a head
/// is is up to the learned scales rather than to how far the scores happen to
/// grow during training. Besides the rotary encoding, every head takes a
/// learned slope per cell of distance between the query's cell and the key's
/// off their attention score, and multiplies the score by a learned scale per
/// natural logarithm of the number of cells on the board. The penalty makes a
/// far cell count less the further it is, which a bigger board continues the
/// same way, and the scale keeps the head as sharp when there are more cells to
/// spread its attention over.
///
/// What every head reads is then gated channel by channel by a sigmoid of the
/// layer's input: the softmax has to spread a head's attention over the cells
/// whether or not any of them holds what the head looks for, and the gate lets
/// the head pass on nothing then.
#[derive(Module, Debug)]
pub struct Attention {
  norm: RmsNorm,
  /// The queries, keys and values, in this order, in one projection.
  qkv: Projection,
  /// Over the channels of a head, shared by all of them.
  query_norm: RmsNorm,
  key_norm: RmsNorm,
  gate: Projection,
  output: Projection,
  /// `None` when no channel is rotated.
  rope: Option<RotaryFrequencies>,
  /// `[heads]`, the score taken off per cell of distance, through a softplus
  /// so that it never turns into a bonus.
  distance_slopes: Param<Tensor<1>>,
  /// `[heads]`, what the scores are multiplied by per natural logarithm of the
  /// number of cells on the board.
  logit_scales: Param<Tensor<1>>,
  heads: usize,
}

impl Attention {
  /// The logit scales start at one on the smallest training board.
  pub fn new(device: &Device, config: &TransformerConfig) -> Self {
    let heads = config.attention_heads;
    let pairs = config.rotary_channels / 2;
    let slopes = (0..heads)
      .map(|head| {
        inverse_softplus(2.0f64.powf(-INITIAL_DISTANCE_SLOPES_RANGE * (head + 1) as f64 / heads as f64)) as f32
      })
      .collect::<Vec<_>>();
    let min_train_area = (config.min_train_side * config.min_train_side) as f64;
    Self {
      norm: RmsNorm::new(device, config.mid_channels),
      qkv: Projection::new(device, config.mid_channels, 3 * config.mid_channels),
      query_norm: RmsNorm::new(device, config.head_channels()),
      key_norm: RmsNorm::new(device, config.head_channels()),
      gate: Projection::new(device, config.mid_channels, config.mid_channels),
      output: Projection::new(device, config.mid_channels, config.mid_channels),
      rope: (pairs > 0).then(|| RotaryFrequencies::new(device, heads, pairs, config.min_rotary_frequency())),
      distance_slopes: Param::from_tensor(Tensor::from_data(TensorData::new(slopes, [heads]), device)),
      logit_scales: Param::from_tensor(Tensor::full([heads], 1.0 / min_train_area.ln(), device)),
      heads,
    }
  }

  /// `[batch, cells, channels]` tokens.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let [batch, cells, channels] = inputs.dims();
    let heads = self.heads;
    let inputs = self.norm.forward(inputs);
    let qkv = self
      .qkv
      .forward(inputs.clone())
      .reshape([batch, cells, 3 * heads, channels / heads])
      .swap_dims(1, 2);
    let dtype = qkv.dtype();
    let query = self.query_norm.forward(qkv.clone().slice(s![.., 0..heads]));
    let key = self.key_norm.forward(qkv.clone().slice(s![.., heads..2 * heads]));
    let value = qkv.slice(s![.., 2 * heads..]);
    let (query, key) = match &self.rope {
      Some(frequencies) => {
        let rope = Rope::new(frequencies.frequencies(), &context.cells);
        (rope.apply(query), rope.apply(key))
      }
      None => (query, key),
    };
    let per_head = |values: Tensor<1>| values.cast(FloatDType::F32).reshape([1, heads, 1, 1]);
    let query = query * (per_head(self.logit_scales.val()) * context.log_area.clone()).cast(dtype);
    let out = distance_attention(
      query,
      key,
      value,
      context.board.clone(),
      softplus(self.distance_slopes.val().cast(FloatDType::F32), 1.0),
      context.width,
    );
    let gate = sigmoid(self.gate.forward(inputs));
    self
      .output
      .forward(out.swap_dims(1, 2).reshape([batch, cells, channels]) * gate)
  }
}

/// A depthwise `3x3` convolution over the board, computing in the float type
/// of its weight and returning that of its input, like a
/// [`Projection`].
#[derive(Module, Debug)]
pub struct DepthwiseConv {
  /// `[channels, 1, 3, 3]`.
  weight: Param<Tensor<4>>,
}

impl DepthwiseConv {
  /// Zero-initialized.
  pub fn new(device: &Device, channels: usize) -> Self {
    Self {
      weight: Param::from_tensor(Tensor::zeros([channels, 1, 3, 3], device)),
    }
  }

  /// `[batch, cells, channels]` tokens. Those off the board are taken as
  /// zero, so that the edge of the board looks the same to it whatever
  /// padding is around.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let [batch, cells, channels] = inputs.dims();
    let dtype = inputs.dtype();
    let weight = self.weight.val();
    let inputs = (inputs * context.on_board.clone().cast(dtype))
      .cast(weight.dtype())
      .swap_dims(1, 2)
      .reshape([batch, channels, context.height, context.width]);
    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], channels);
    conv2d(inputs, weight, None, options)
      .reshape([batch, channels, cells])
      .swap_dims(1, 2)
      .cast(dtype)
  }
}

/// Feed-forward layer with a SiLU gated linear unit, normalized first; it
/// returns the residual and leaves adding it to the caller.
#[derive(Module, Debug)]
pub struct FeedForward {
  norm: RmsNorm,
  /// The hidden activations and their gates, in this order, in one
  /// projection.
  input: Projection,
  /// Added to the hidden activations and gates, so that every token sees
  /// what its neighbors hold; `None` without it. Zero-initialized, and decayed
  /// towards zero, so the layer starts as a plain feed-forward one.
  conv: Option<DepthwiseConv>,
  output: Projection,
}

impl FeedForward {
  pub fn new(device: &Device, channels: usize, hidden_channels: usize, conv: bool) -> Self {
    Self {
      norm: RmsNorm::new(device, channels),
      input: Projection::new(device, channels, 2 * hidden_channels),
      conv: conv.then(|| DepthwiseConv::new(device, 2 * hidden_channels)),
      output: Projection::new(device, hidden_channels, channels),
    }
  }

  /// `[batch, cells, channels]` tokens.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let hidden = self.input.forward(self.norm.forward(inputs));
    let hidden = match &self.conv {
      Some(conv) => hidden.clone() + conv.forward(hidden, context),
      None => hidden,
    };
    let [.., hidden_channels] = hidden.dims();
    let activations = hidden.clone().slice(s![.., .., 0..hidden_channels / 2]);
    let gates = hidden.slice(s![.., .., hidden_channels / 2..]);
    self.output.forward(silu(activations) * gates)
  }
}

/// Attention followed by a feed-forward layer, each adding its residual.
#[derive(Module, Debug)]
pub struct TransformerLayer {
  attention: Attention,
  ffn: FeedForward,
}

impl TransformerLayer {
  pub fn new(device: &Device, config: &TransformerConfig) -> Self {
    Self {
      attention: Attention::new(device, config),
      ffn: FeedForward::new(device, config.mid_channels, config.ffn_channels, config.ffn_conv),
    }
  }

  /// `[batch, cells, channels]` tokens. The residuals are computed in the
  /// projections' float type and added in the stream's.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let dtype = inputs.dtype();
    let out = inputs.clone() + self.attention.forward(inputs, context).cast(dtype);
    out.clone() + self.ffn.forward(out, context).cast(dtype)
  }
}

/// A nested bottleneck block of transformer layers: a `1x1` convolution down
/// to `mid_channels`, the transformer layers over the board's cells, and a
/// `1x1` convolution back up, added to the trunk.
#[derive(Module, Debug)]
pub struct TransformerBlock {
  pub(crate) normactconvp: NormActConv,
  pub(crate) layers: Vec<TransformerLayer>,
  pub(crate) normactconvq: NormActConv,
}

impl TransformerBlock {
  pub fn new(device: &Device, inner_channels: usize, config: &TransformerConfig) -> Self {
    assert!(config.layers > 0, "a transformer block needs a layer");
    assert!(
      config.mid_channels.is_multiple_of(config.attention_heads),
      "the channels must split evenly between the attention heads"
    );
    assert!(
      config.rotary_channels.is_multiple_of(2) && config.rotary_channels <= config.head_channels(),
      "the 2D rotary encoding turns pairs of channels of an attention head"
    );
    assert!(
      config.min_train_side > 1,
      "the smallest training board needs two cells along a side"
    );
    Self {
      normactconvp: NormActConv::new(device, false, None, inner_channels, config.mid_channels, [1, 1]),
      layers: (0..config.layers)
        .map(|_| TransformerLayer::new(device, config))
        .collect(),
      normactconvq: NormActConv::new(device, true, None, config.mid_channels, inner_channels, [1, 1]),
    }
  }

  /// The convolution in is scaled like a residual block's, and the one out is
  /// zero-initialized so that the block starts as the identity. The
  /// transformer layers keep their default initialization: each of them
  /// normalizes what it reads, so the depth of the trunk does not change the
  /// scale they see.
  pub(crate) fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    let inner_scale = fixup_scale.powf(1.0 / (1.0 + self.layers.len() as f64));
    self.normactconvp.initialize(inner_scale, device);
    self.normactconvq.initialize(0.0, device);
  }

  pub fn forward(
    &self,
    inputs: Tensor<4>,
    mask: Tensor<4>,
    mask_sum_hw: Tensor<4>,
    context: &AttentionContext,
  ) -> Tensor<4> {
    let out = self
      .normactconvp
      .forward(inputs.clone(), mask.clone(), mask_sum_hw.clone());
    let [batch, channels, height, width] = out.dims();
    let mut tokens = out.reshape([batch, channels, height * width]).swap_dims(1, 2);
    for layer in &self.layers {
      tokens = layer.forward(tokens, context);
    }
    let out = tokens.swap_dims(1, 2).reshape([batch, channels, height, width]);
    let out = self.normactconvq.forward(out, mask, mask_sum_hw);
    inputs + out
  }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
  use super::{Attention, AttentionContext, Cells, FeedForward, Rope, RotaryFrequencies};
  use crate::model::TransformerConfig;
  use burn::tensor::{Device, Distribution, Tensor, TensorData, s};

  fn config(rotary_channels: usize, ffn_conv: bool) -> TransformerConfig {
    TransformerConfig {
      mid_channels: 16,
      layers: 1,
      attention_heads: 2,
      ffn_channels: 24,
      rotary_channels,
      min_train_side: 4,
      ffn_conv,
    }
  }

  /// The `[batch, 1, height, width]` mask of a `rows x columns` board in the
  /// corner of every one of `batch` grids of `height x width` cells.
  fn mask(
    device: &Device,
    batch: usize,
    (height, width): (usize, usize),
    (rows, columns): (usize, usize),
  ) -> Tensor<4> {
    let values = (0..batch * height * width)
      .map(|i| {
        let cell = i % (height * width);
        if cell / width < rows && cell % width < columns {
          1.0
        } else {
          0.0
        }
      })
      .collect::<Vec<f32>>();
    Tensor::from_data(TensorData::new(values, [batch, 1, height, width]), device)
  }

  /// A rotated query and a rotated key have the dot product that depends on the
  /// offset between their cells only: moving both cells by the same amount
  /// keeps it, at no offset it is the dot product before the rotation, and
  /// another offset changes it. The channels after the rotated ones are kept.
  #[test]
  fn rope_depends_on_the_offset_only() {
    let device = Device::flex();
    let (height, width, channels) = (5, 6, 10);
    let frequencies = Tensor::<1>::from_floats([0.9, 0.4, -0.3, 0.7, 0.05, -0.6, 0.5, 0.5], &device).reshape([1, 4, 2]);
    let rope = Rope::new(frequencies, &Cells::new(&device, height, width));
    let rotated = |vector: [f32; 10]| {
      let tokens = Tensor::<1>::from_floats(vector, &device)
        .reshape([1, 1, 1, channels])
        .repeat_dim(2, height * width);
      let rotated = rope.apply(tokens).into_data().try_to_vec::<f32>().unwrap();
      move |y: usize, x: usize| rotated[(y * width + x) * channels..][..channels].to_vec()
    };
    let query = [0.3, -1.2, 0.7, 0.1, 2.0, -0.4, 0.9, -0.6, 1.5, -2.5];
    let key = [-0.5, 0.8, 1.1, -0.3, 0.2, 1.4, -0.7, 0.6, 0.4, 0.3];
    let queries = rotated(query);
    let keys = rotated(key);
    let dot = |a: Vec<f32>, b: Vec<f32>| a.iter().zip(&b).map(|(a, b)| a * b).sum::<f32>();

    let at = |(qy, qx), (ky, kx)| dot(queries(qy, qx), keys(ky, kx));
    let base = at((1, 2), (3, 1));
    assert!((at((2, 4), (4, 3)) - base).abs() < 1e-4);
    assert!((at((0, 3), (2, 2)) - base).abs() < 1e-4);
    assert!((at((2, 2), (3, 1)) - base).abs() > 1e-2);
    assert!((at((1, 2), (3, 2)) - base).abs() > 1e-2);
    assert!((at((4, 5), (4, 5)) - dot(query.to_vec(), key.to_vec())).abs() < 1e-4);

    // The rotation keeps the length of every vector, and the channels after
    // the rotated ones as they are.
    let length = |v: Vec<f32>| dot(v.clone(), v).sqrt();
    let expected = length(query.to_vec());
    assert!((length(queries(3, 4)) - expected).abs() < 1e-4);
    assert_eq!(queries(3, 4)[8..], query[8..]);
  }

  /// The initial learned frequencies are within their range, and turn along
  /// both signs of both axes.
  #[test]
  fn initial_learned_frequencies() {
    let device = Device::flex();
    let min_frequency = 0.4;
    let frequencies = RotaryFrequencies::new(&device, 3, 8, min_frequency).frequencies();
    assert_eq!(frequencies.dims(), [3, 8, 2]);
    let magnitudes = frequencies
      .clone()
      .square()
      .sum_dim(2)
      .sqrt()
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    assert!(magnitudes.iter().all(|magnitude| {
      (min_frequency as f32 * 1.001..=super::MAX_INITIAL_ROPE_FREQUENCY as f32 * 1.02).contains(magnitude)
    }));
    for axis in 0..2 {
      let values = frequencies
        .clone()
        .slice(s![.., .., axis..axis + 1])
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
      assert!(values.iter().any(|value| *value < 0.0));
      assert!(values.iter().any(|value| *value > 0.0));
    }
  }

  /// No frequency ever goes below the lowest one, however far its parameter
  /// is pushed.
  #[test]
  fn learned_frequencies_keep_their_bound() {
    let device = Device::flex();
    let mut frequencies = RotaryFrequencies::new(&device, 1, 2, 0.4);
    frequencies.excess = frequencies.excess.map(|excess| excess - 1000.0);
    let magnitudes = frequencies.frequencies().square().sum_dim(2).sqrt();
    let values = magnitudes.into_data().try_to_vec::<f32>().unwrap();
    assert!(values.iter().all(|value| (value - 0.4).abs() < 1e-5));
  }

  /// Every learned parameter of the attention and of the feed-forward layer
  /// gets a gradient: what they return depends on all of them.
  #[test]
  fn every_parameter_gets_a_gradient() {
    use burn::module::{Module, ModuleVisitor, Param};
    struct Gradients<'a>(&'a burn::tensor::Gradients, Vec<f32>);
    impl ModuleVisitor for Gradients<'_> {
      fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let grad = param.val().grad(self.0).unwrap();
        self.1.push(grad.abs().sum().into_scalar::<f32>());
      }
    }
    let device = Device::flex().autodiff();
    let config = config(6, true);
    let (height, width) = (3, 4);
    let attention = Attention::new(&device, &config);
    let ffn = FeedForward::new(&device, config.mid_channels, config.ffn_channels, true);
    let context = AttentionContext::new(&mask(&device, 1, (height, width), (height, width)));
    let inputs = Tensor::<3>::random([1, height * width, config.mid_channels], Distribution::Default, &device);
    // The convolution starts at zero, which still lets the gradient through to
    // its weight; through the other parameters of the feed-forward layer it
    // has to be away from zero.
    let attended = attention.forward(inputs, &context);
    let ffn = ffn.map(&mut ShiftConv);
    let grads = (attended.clone() + ffn.forward(attended, &context)).sum().backward();
    let mut gradients = Gradients(&grads, Vec::new());
    attention.visit(&mut gradients);
    ffn.visit(&mut gradients);
    assert_eq!(gradients.1.len(), 14);
    assert!(gradients.1.iter().all(|grad| *grad > 0.0), "{:?}", gradients.1);

    /// Moves the depthwise convolution's weight away from zero.
    struct ShiftConv;
    impl burn::module::ModuleMapper for ShiftConv {
      fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if D == 4 {
          param.map(|tensor| (tensor + 0.1).detach().require_grad())
        } else {
          param
        }
      }
    }
  }

  /// The distance penalty steers attention away from far cells: with a steep
  /// slope a token attends to its own cell alone, so changing another token
  /// leaves what it gets as it is, and without a slope it does not.
  #[test]
  fn distance_penalty_keeps_attention_near() {
    let device = Device::flex();
    let config = config(0, false);
    let (height, width) = (3, 3);
    let context = AttentionContext::new(&mask(&device, 1, (height, width), (height, width)));
    let inputs = Tensor::<3>::random([1, height * width, config.mid_channels], Distribution::Default, &device);
    let changed = Tensor::cat(
      vec![
        inputs.clone().slice(s![.., 0..8]),
        Tensor::random([1, 1, config.mid_channels], Distribution::Default, &device),
      ],
      1,
    );
    let attention = Attention::new(&device, &config);
    let with_slope = |slope: f32| {
      let attention = Attention {
        distance_slopes: attention
          .distance_slopes
          .clone()
          .map(|slopes| slopes.zeros_like() + slope),
        ..attention.clone()
      };
      let difference = attention.forward(inputs.clone(), &context) - attention.forward(changed.clone(), &context);
      difference.slice(s![.., 0..8]).abs().max().into_scalar::<f32>()
    };
    assert!(with_slope(100.0) < 1e-5);
    assert!(with_slope(-100.0) > 1e-3);
  }
}
