//! The attention trunk: nested bottleneck blocks whose inner layers are
//! transformer layers instead of convolutions. Every board cell is a token,
//! and the only thing telling the tokens' positions apart is the 2D rotary
//! encoding of the queries and keys, so attention depends on the offset between
//! two cells and never on where they are - a network trained on one board size
//! carries over to another. The rotary frequencies are either fixed or learned
//! by every attention layer, see [`TransformerConfig::learnable_rope`].

use crate::model::{NormActConv, TransformerConfig, TrunkBlock};
use burn::{
  module::{Initializer, Module, Param},
  tensor::{
    Bool, Device, Distribution, FloatDType, Tensor, TensorData,
    activation::silu,
    module::{attention, linear},
    ops::AttentionModuleOptions,
    s,
  },
};

/// Base of the rotary encoding's frequencies, which fall geometrically from
/// one radian per cell towards `1 / ROPE_THETA`. The slow ones turn by well
/// under a full circle across the largest board, so no two offsets on it look
/// alike to every frequency at once.
const ROPE_THETA: f64 = 100.0;

/// Learned rotary frequencies start log-uniformly between these, in radians
/// per cell: from a full turn every 6 cells to one over 300 cells, wider than
/// any board, so that some pairs tell nearby cells apart and others whole
/// regions.
const LEARNED_ROPE_FREQUENCIES: (f64, f64) = (1.0 / 50.0, 1.0);

const RMS_NORM_EPSILON: f64 = 1e-6;

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

/// Root mean square normalization over the channels of each token, followed
/// by a learned per channel scale.
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
  pub fn forward(&self, inputs: Tensor<3>) -> Tensor<3> {
    let dtype = inputs.dtype();
    let inputs = inputs.cast(FloatDType::F32);
    let rms = (inputs.clone().square().mean_dim(2) + RMS_NORM_EPSILON).sqrt();
    (inputs / rms).cast(dtype) * (self.gamma.val().cast(dtype) + 1.0).unsqueeze::<3>()
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
    Self {
      rows: Tensor::from_data(TensorData::new(rows, shape), device),
      columns: Tensor::from_data(TensorData::new(columns, shape), device),
    }
  }
}

/// 2D rotary position encoding of the tokens of a board.
///
/// Each head's channels are split into two halves whose `i`-th channels form
/// a pair, rotated as a complex number by an angle that is a linear function
/// of the cell's row and column, with the pair's own frequency along each.
/// Rotating a query and a key by the angles of their cells leaves their dot
/// product depending only on the difference of the angles, i.e. on the offset
/// between the cells.
///
/// TODO: switch to burn's `RotaryEncoding` once it supports 2D positions.
#[derive(Clone, Debug)]
pub struct Rope {
  /// `[1, heads, cells, head_channels / 2]`, with a single head when all heads
  /// share their frequencies.
  cos: Tensor<4>,
  sin: Tensor<4>,
}

impl Rope {
  /// `frequencies` is `[heads, head_channels / 2, 2]`: the radians every pair
  /// of every head turns per row and per column. A single head's frequencies
  /// apply to all of them. The angles are computed in single precision
  /// whatever the frequencies' type: in half precision they would be off by
  /// a hundredth of a radian already across a board of 24 cells.
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

  /// The fixed frequencies, `[1, head_channels / 2, 2]`: the first half of the
  /// pairs turns with the row only and the second half with the column only,
  /// each half at the frequencies [`ROPE_THETA`] sets. Diagonal offsets are
  /// then seen only as a row offset and a column offset together.
  pub fn fixed_frequencies(device: &Device, head_channels: usize) -> Tensor<3> {
    let pairs = head_channels / 2;
    let mut frequencies = Vec::with_capacity(2 * pairs);
    for axis in 0..2 {
      for i in 0..pairs / 2 {
        let frequency = ROPE_THETA.powf(-(i as f64) / (pairs / 2) as f64) as f32;
        frequencies.extend(if axis == 0 { [frequency, 0.0] } else { [0.0, frequency] });
      }
    }
    Tensor::from_data(TensorData::new(frequencies, [1, pairs, 2]), device)
  }

  /// Initial learned frequencies, `[heads, head_channels / 2, 2]`: every pair
  /// of every head turns along an arbitrary direction of the board, so a head
  /// can learn to follow diagonals as well as rows and columns. Magnitudes are
  /// log-uniform in [`LEARNED_ROPE_FREQUENCIES`] and signs random.
  pub fn initial_frequencies(device: &Device, heads: usize, head_channels: usize) -> Tensor<3> {
    let shape = [heads, head_channels / 2, 2];
    let (low, high) = LEARNED_ROPE_FREQUENCIES;
    let magnitudes = Tensor::<3>::random(shape, Distribution::Uniform(low.ln(), high.ln()), device).exp();
    let signs = Tensor::<3>::random(shape, Distribution::Bernoulli(0.5), device) * 2.0 - 1.0;
    magnitudes * signs
  }

  /// Rotates `[batch, heads, cells, head_channels]` queries or keys.
  pub fn apply(&self, inputs: Tensor<4>) -> Tensor<4> {
    let [.., channels] = inputs.dims();
    let dtype = inputs.dtype();
    let cos = self.cos.clone().cast(dtype);
    let sin = self.sin.clone().cast(dtype);
    let first = inputs.clone().slice(s![.., .., .., 0..channels / 2]);
    let second = inputs.slice(s![.., .., .., channels / 2..]);
    Tensor::cat(
      vec![
        first.clone() * cos.clone() - second.clone() * sin.clone(),
        first * sin + second * cos,
      ],
      3,
    )
  }
}

/// What every attention layer of a forward pass shares.
#[derive(Clone, Debug)]
pub struct AttentionContext {
  cells: Cells,
  /// The fixed rotary encoding, `None` when every layer learns its own.
  rope: Option<Rope>,
  /// `[batch, 1, 1, cells]`, true for the cells off the board, which no token
  /// may attend to.
  off_board: Tensor<4, Bool>,
}

/// Multi-head self-attention over the tokens of the board, normalized first;
/// it returns the residual and leaves adding it to the caller.
#[derive(Module, Debug)]
pub struct Attention {
  norm: RmsNorm,
  /// The queries, keys and values, in this order, in one projection.
  qkv: Projection,
  output: Projection,
  /// This layer's rotary frequencies, `[heads, head_channels / 2, 2]`, or
  /// `None` for the fixed ones. See [`Rope::new`].
  rope_frequencies: Option<Param<Tensor<3>>>,
  heads: usize,
}

impl Attention {
  pub fn new(device: &Device, channels: usize, heads: usize, learnable_rope: bool) -> Self {
    Self {
      norm: RmsNorm::new(device, channels),
      qkv: Projection::new(device, channels, 3 * channels),
      output: Projection::new(device, channels, channels),
      rope_frequencies: learnable_rope
        .then(|| Param::from_tensor(Rope::initial_frequencies(device, heads, channels / heads))),
      heads,
    }
  }

  /// `[batch, cells, channels]` tokens.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let [batch, cells, channels] = inputs.dims();
    let heads = self.heads;
    let qkv = self
      .qkv
      .forward(self.norm.forward(inputs))
      .reshape([batch, cells, 3 * heads, channels / heads])
      .swap_dims(1, 2);
    let learned = self
      .rope_frequencies
      .as_ref()
      .map(|frequencies| Rope::new(frequencies.val(), &context.cells));
    let rope = learned
      .as_ref()
      .or(context.rope.as_ref())
      .expect("the context has the fixed rotary encoding when a layer has no learned one");
    let query = rope.apply(qkv.clone().slice(s![.., 0..heads]));
    let key = rope.apply(qkv.clone().slice(s![.., heads..2 * heads]));
    let value = qkv.slice(s![.., 2 * heads..]);
    // Broadcast rather than materialized: the mask is only on the keys.
    let mask = context.off_board.clone().expand([batch, heads, cells, cells]);
    let out = attention(query, key, value, Some(mask), None, AttentionModuleOptions::default());
    self
      .output
      .forward(out.swap_dims(1, 2).reshape([batch, cells, channels]))
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
  output: Projection,
}

impl FeedForward {
  pub fn new(device: &Device, channels: usize, hidden_channels: usize) -> Self {
    Self {
      norm: RmsNorm::new(device, channels),
      input: Projection::new(device, channels, 2 * hidden_channels),
      output: Projection::new(device, hidden_channels, channels),
    }
  }

  /// `[batch, cells, channels]` tokens.
  pub fn forward(&self, inputs: Tensor<3>) -> Tensor<3> {
    let hidden = self.input.forward(self.norm.forward(inputs));
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
      attention: Attention::new(
        device,
        config.mid_channels,
        config.attention_heads,
        config.learnable_rope,
      ),
      ffn: FeedForward::new(device, config.mid_channels, config.ffn_channels),
    }
  }

  /// `[batch, cells, channels]` tokens. The residuals are computed in the
  /// projections' float type and added in the stream's.
  pub fn forward(&self, inputs: Tensor<3>, context: &AttentionContext) -> Tensor<3> {
    let dtype = inputs.dtype();
    let out = inputs.clone() + self.attention.forward(inputs, context).cast(dtype);
    out.clone() + self.ffn.forward(out).cast(dtype)
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
  pub fn new(device: &Device, config: &TransformerConfig) -> Self {
    assert!(config.transformer_size > 0, "a transformer block needs a layer");
    assert!(
      config.mid_channels.is_multiple_of(config.attention_heads),
      "the channels must split evenly between the attention heads"
    );
    assert!(
      config
        .head_channels()
        .is_multiple_of(if config.learnable_rope { 2 } else { 4 }),
      "the 2D rotary encoding needs an even number of channels per attention head, and a multiple of 4 for the \
       fixed one"
    );
    Self {
      normactconvp: NormActConv::new(device, false, None, config.inner_channels, config.mid_channels, [1, 1]),
      layers: (0..config.transformer_size)
        .map(|_| TransformerLayer::new(device, config))
        .collect(),
      normactconvq: NormActConv::new(device, true, None, config.mid_channels, config.inner_channels, [1, 1]),
    }
  }
}

impl TrunkBlock for TransformerBlock {
  type Context = AttentionContext;

  fn context(&self, mask: &Tensor<4>) -> AttentionContext {
    let [batch, _, height, width] = mask.dims();
    let device = mask.device();
    let attention = &self.layers[0].attention;
    let [channels] = attention.norm.gamma.val().dims();
    let cells = Cells::new(&device, height, width);
    let rope = attention
      .rope_frequencies
      .is_none()
      .then(|| Rope::new(Rope::fixed_frequencies(&device, channels / attention.heads), &cells));
    AttentionContext {
      cells,
      rope,
      off_board: mask.clone().reshape([batch, 1, 1, height * width]).lower_elem(0.5),
    }
  }

  /// The convolution in is scaled like a residual block's, and the one out is
  /// zero-initialized so that the block starts as the identity. The
  /// transformer layers keep their default initialization: each of them
  /// normalizes what it reads, so the depth of the trunk does not change the
  /// scale they see.
  fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    let inner_scale = fixup_scale.powf(1.0 / (1.0 + self.layers.len() as f64));
    self.normactconvp.initialize(inner_scale, device);
    self.normactconvq.initialize(0.0, device);
  }

  fn forward(
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
  use super::{Attention, AttentionContext, Cells, Rope};
  use burn::tensor::{Device, Distribution, Tensor};

  /// A rotated query and a rotated key have the dot product that depends on the
  /// offset between their cells only: moving both cells by the same amount
  /// keeps it, at no offset it is the dot product before the rotation, and
  /// another offset changes it.
  fn depends_on_the_offset_only(device: &Device, frequencies: Tensor<3>) {
    let (height, width, channels) = (5, 6, 8);
    let rope = Rope::new(frequencies, &Cells::new(device, height, width));
    let rotated = |vector: [f32; 8]| {
      let tokens = Tensor::<1>::from_floats(vector, device)
        .reshape([1, 1, 1, channels])
        .repeat_dim(2, height * width);
      let rotated = rope.apply(tokens).into_data().try_to_vec::<f32>().unwrap();
      move |y: usize, x: usize| rotated[(y * width + x) * channels..][..channels].to_vec()
    };
    let query = [0.3, -1.2, 0.7, 0.1, 2.0, -0.4, 0.9, -0.6];
    let key = [-0.5, 0.8, 1.1, -0.3, 0.2, 1.4, -0.7, 0.6];
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

    // The rotation keeps the length of every vector.
    let length = |v: Vec<f32>| dot(v.clone(), v).sqrt();
    let expected = length(query.to_vec());
    assert!((length(queries(3, 4)) - expected).abs() < 1e-4);
  }

  #[test]
  fn fixed_rope_depends_on_the_offset_only() {
    let device = Device::flex();
    depends_on_the_offset_only(&device, Rope::fixed_frequencies(&device, 8));
  }

  /// Learned frequencies may turn along any direction, diagonals included.
  #[test]
  fn learned_rope_depends_on_the_offset_only() {
    let device = Device::flex();
    let frequencies = Tensor::<1>::from_floats([0.9, 0.4, -0.3, 0.7, 0.05, -0.6, 0.5, 0.5], &device).reshape([1, 4, 2]);
    depends_on_the_offset_only(&device, frequencies);
  }

  /// The initial learned frequencies are within their range along each axis,
  /// with both signs.
  #[test]
  fn initial_learned_frequencies() {
    let device = Device::flex();
    let frequencies = Rope::initial_frequencies(&device, 3, 16);
    assert_eq!(frequencies.dims(), [3, 8, 2]);
    let values = frequencies.into_data().try_to_vec::<f32>().unwrap();
    let (low, high) = super::LEARNED_ROPE_FREQUENCIES;
    assert!(
      values
        .iter()
        .all(|value| (low as f32..=high as f32).contains(&value.abs()))
    );
    assert!(values.iter().any(|value| *value < 0.0));
    assert!(values.iter().any(|value| *value > 0.0));
  }

  /// The learned frequencies are trained: the attention's output depends on
  /// them, so they get a gradient.
  #[test]
  fn learned_rope_frequencies_get_gradients() {
    let device = Device::flex().autodiff();
    let (height, width, channels) = (3, 4, 16);
    let attention = Attention::new(&device, channels, 2, true);
    let context = AttentionContext {
      cells: Cells::new(&device, height, width),
      rope: None,
      off_board: Tensor::<4>::ones([1, 1, 1, height * width], &device).lower_elem(0.5),
    };
    let inputs = Tensor::<3>::random([1, height * width, channels], Distribution::Default, &device);
    let grads = attention.forward(inputs, &context).sum().backward();
    let frequencies = attention.rope_frequencies.as_ref().unwrap().val();
    let grad = frequencies.grad(&grads).unwrap();
    assert!(grad.abs().sum().into_scalar::<f32>() > 0.0);
  }
}
