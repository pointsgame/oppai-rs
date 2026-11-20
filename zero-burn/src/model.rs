use crate::{
  chains::{Chains, ChainsConfig},
  resnet::ResidualBlock,
};
use burn::{
  module::{Initializer, Module, ModuleMapper, ModuleVisitor, Param},
  nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    loss::HuberLossConfig,
  },
  optim::{GradientsParams, ModuleOptimizer},
  store::{ModuleRecord, RecordError},
  tensor::{
    DType, DataError, Device, ExecutionError, FloatDType, Int, Tensor, TensorData, Transaction,
    activation::{log_softmax, mish, sigmoid, softmax, softplus, tanh},
    module::{conv2d, linear},
    ops::ConvOptions,
    s,
  },
};
use derive_more::From;
use ndarray::{Array, Array1, Array2, Array3, Array4, Dimension, ShapeError};
use oppai_zero::{
  examples::{FUTURE_POSITIONS, TD_VALUES},
  field_features::{CHANNELS, FEATURE_CHANNELS, GLOBAL_FEATURES, GLOBAL_KOMI, SCORE_ONE_HOT_SIZE},
  model::{Model as OppaiModel, TrainableModel as OppaiTrainableModel, VALUE_COLUMNS},
};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::Path};
use thiserror::Error;

/// Sizes of the policy, value and captured heads, which every trunk shares.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct HeadsConfig {
  pub v1_channels: usize,
  pub p1_channels: usize,
  pub g1_channels: usize,
  pub v2_size: usize,
  pub sbv2_size: usize,
  pub num_scorebeliefs: usize,
}

/// A nested bottleneck residual block of convolutions: a `1x1` convolution
/// down to `mid_channels`, inner residual blocks of two `3x3` convolutions
/// each, and a `1x1` convolution back up.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NbtConfig {
  pub mid_channels: usize,
  /// Inner residual blocks.
  pub size: usize,
  /// Global pooling in the first convolution of the first inner block, if
  /// any.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub gpool: Option<GpoolConfig>,
  /// Pooling over the chains of dots after the first inner block, if any.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub chains: Option<ChainsConfig>,
}

/// A convolution part of whose channels is pooled over the whole board rather
/// than convolved, and conditions the rest of them: see [`ConvAndGPool`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpoolConfig {
  /// The channels pooled.
  pub channels: usize,
  /// Whether the pooled channels gate the rest multiplicatively as well as
  /// add a bias to them.
  pub gate: bool,
}

/// A block of the trunk, which adds its residual to the trunk's channels.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockConfig {
  Nbt(NbtConfig),
}

/// Model architecture hyperparameters: the trunk's blocks, in order, and the
/// heads reading it. A model must be created with the same config it was
/// trained with for the weights to load.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
  /// Channels of the trunk the blocks add their residuals to.
  pub inner_channels: usize,
  pub blocks: Vec<BlockConfig>,
  #[serde(flatten)]
  pub heads: HeadsConfig,
}

/// Five nested bottleneck blocks of 192 channels, every second one pooling
/// globally.
impl Default for ModelConfig {
  fn default() -> Self {
    Self {
      inner_channels: 192,
      blocks: (0..5)
        .map(|i| {
          BlockConfig::Nbt(NbtConfig {
            mid_channels: 96,
            size: 2,
            gpool: (i % 2 == 1).then_some(GpoolConfig {
              channels: 32,
              gate: true,
            }),
            chains: None,
          })
        })
        .collect(),
      heads: HeadsConfig {
        v1_channels: 32,
        p1_channels: 32,
        g1_channels: 32,
        v2_size: 80,
        sbv2_size: 80,
        num_scorebeliefs: 6,
      },
    }
  }
}

impl ModelConfig {
  /// Whether any block pools over the chains of dots.
  pub fn pools_chains(&self) -> bool {
    self.blocks.iter().any(|block| match block {
      BlockConfig::Nbt(config) => config.chains.is_some(),
    })
  }

  /// The same architecture without any pooling over chains: the one a model
  /// [`Model::adding_chains`] starts from.
  pub fn without_chains(&self) -> Self {
    let mut config = self.clone();
    for block in &mut config.blocks {
      match block {
        BlockConfig::Nbt(config) => config.chains = None,
      }
    }
    config
  }

  pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ModelConfigError> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
  }
}

#[derive(Error, Debug, From)]
pub enum ModelConfigError {
  #[error("io error")]
  Io(std::io::Error),
  #[error("json error")]
  Json(serde_json::Error),
}

// Activation gain for mish, used to keep activation variance stable through the deep residual trunk.
pub(crate) fn mish_gain() -> f64 {
  2.210277_f64.sqrt()
}

/// Reinitialize a weight tensor:
/// sample from `N(0, (scale * gain / sqrt(fan_in))^2)`, or fill with zeros when `scale == 0`.
/// Zero scale is how Fixup makes a residual branch start as the identity function.
pub(crate) fn init_weight<const D: usize>(
  shape: [usize; D],
  fan_in: usize,
  scale: f64,
  gain: f64,
  device: &Device,
) -> Param<Tensor<D>> {
  if scale <= 0.0 {
    Initializer::Zeros.init(shape, device)
  } else {
    let std = scale * gain / (fan_in as f64).sqrt();
    Initializer::Normal { mean: 0.0, std }.init(shape, device)
  }
}

/// Reinitialize a convolution's weights. `fan_in = in_channels * kernel_h * kernel_w`.
pub(crate) fn init_conv(conv: &mut Conv2d, scale: f64, gain: f64, device: &Device) {
  let [out_c, in_c, kh, kw] = conv.weight.val().dims();
  conv.weight = init_weight([out_c, in_c, kh, kw], in_c * kh * kw, scale, gain, device);
}

/// A stride 1 convolution padded to keep the spatial size, which is every
/// convolution of the network. It computes in the float type of its weight and
/// returns the float type of its input: a model normally has both in one type,
/// but [`Model::with_compute_dtype`] casts the convolution weights alone.
pub(crate) fn conv2d_same(inputs: Tensor<4>, weight: Tensor<4>) -> Tensor<4> {
  let [_, _, kernel_height, kernel_width] = weight.dims();
  let options = ConvOptions::new([1, 1], [kernel_height / 2, kernel_width / 2], [1, 1], 1);
  let dtype = inputs.dtype();
  conv2d(inputs.cast(weight.dtype()), weight, None, options).cast(dtype)
}

/// Casts the weights of the [`Conv2d`] modules it maps - the ones nearly all
/// the work of the network is done by - leaving every other parameter as it
/// is.
///
/// TODO: switch to burn's automatic mixed precision once it has one, which
/// would also make the casts around every convolution unnecessary.
struct ComputeCaster {
  dtype: FloatDType,
  /// Whether each module entered and not yet exited is one to cast.
  in_compute: Vec<bool>,
}

impl ModuleMapper for ComputeCaster {
  fn enter_module(&mut self, _name: &str, container_type: &str) {
    self.in_compute.push(container_type == "Struct:Conv2d");
  }

  fn exit_module(&mut self, _name: &str, _container_type: &str) {
    self.in_compute.pop();
  }

  fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
    if self.in_compute.last() == Some(&true) {
      param.map(|tensor| tensor.cast(self.dtype))
    } else {
      param
    }
  }
}

/// [`Conv2d::forward`] through [`conv2d_same`].
pub(crate) fn conv_forward(conv: &Conv2d, inputs: Tensor<4>) -> Tensor<4> {
  debug_assert!(
    conv.stride == [1, 1]
      && conv.dilation == [1, 1]
      && conv.groups == 1
      && conv.bias.is_none()
      && conv.kernel_size.iter().all(|size| size % 2 == 1),
    "every convolution of the network is unbiased, odd sized, and of stride 1"
  );
  conv2d_same(inputs, conv.weight.val())
}

/// Reinitialize a linear layer's weights (and bias, if present). The burn weight layout is
/// `[d_input, d_output]`, so `fan_in = d_input`.
fn init_linear(
  linear: &mut Linear,
  weight_scale: f64,
  weight_gain: f64,
  bias_scale: f64,
  bias_gain: f64,
  device: &Device,
) {
  let [d_in, d_out] = linear.weight.val().dims();
  linear.weight = init_weight([d_in, d_out], d_in, weight_scale, weight_gain, device);
  if linear.bias.is_some() {
    linear.bias = Some(init_weight([d_out], d_in, bias_scale, bias_gain, device));
  }
}

#[derive(Module, Debug)]
pub struct NormMask {
  beta: Param<Tensor<4>>,
  gamma: Option<Param<Tensor<4>>>,
}

impl NormMask {
  pub fn new(device: &Device, channels: usize, gamma: bool) -> Self {
    Self {
      beta: Param::from_tensor(Tensor::zeros([1, channels, 1, 1], device)),
      // Centered at 1: gamma starts at zero and is applied as `gamma + 1`,
      // so the layer begins as a unit affine and weight decay pulls the
      // effective scale toward 1 rather than 0.
      gamma: if gamma {
        Some(Param::from_tensor(Tensor::zeros([1, channels, 1, 1], device)))
      } else {
        None
      },
    }
  }

  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>) -> Tensor<4> {
    match self.gamma {
      Some(ref gamma) => (inputs * (gamma.val() + 1.0) + self.beta.val()) * mask,
      None => (inputs + self.beta.val()) * mask,
    }
  }
}

/// The smallest and the biggest boards the [`board_size_feature`] spans from
/// -1 to 1: the range of sizes the network is meant to play on, whatever
/// sizes it is trained on. They only keep the feature near one over that
/// range: the weights reading it would absorb any other choice.
const BOARD_SIZE_FEATURE_AREAS: (f64, f64) = (16.0 * 16.0, 40.0 * 40.0);

/// The board's size, as global pooling scales the means with it, from the
/// `[batch, 1, 1, 1]` number of cells on the board: the logarithm of the
/// area, going from -1 to 1 over [`BOARD_SIZE_FEATURE_AREAS`] - 0 for an area
/// of 640, and 0.73 for the default 39x32 board. A logarithm grows slowly, so
/// that a board bigger than any trained on takes the network less far from
/// what it learned than the side or the area would.
pub(crate) fn board_size_feature(mask_sum_hw: Tensor<4>) -> Tensor<4> {
  let (min, max) = BOARD_SIZE_FEATURE_AREAS;
  let center = (min.ln() + max.ln()) / 2.0;
  let half_range = (max.ln() - min.ln()) / 2.0;
  (mask_sum_hw.log() - center) / half_range
}

/// The features the value head pools from each of its channels.
const VALUE_POOLED: usize = 2;

#[derive(Module, Debug)]
pub struct ConvAndGPool {
  conv1r: Conv2d,
  conv1g: Conv2d,
  normg: NormMask,
  linearg: Linear,
  /// The gate's logits; zero-initialized, so that every gate starts at one,
  /// and decayed towards zero, so towards one.
  lineargate: Option<Linear>,
}

impl ConvAndGPool {
  /// A `3x3` convolution from `channels` to `channels - gpool.channels`,
  /// whose output gets a bias computed from the globally pooled
  /// `gpool.channels` added to it - and with `gpool.gate`, is first scaled
  /// channel by channel by twice the sigmoid of another function of them,
  /// which lets a block turn its channels up or down by what the whole board
  /// holds.
  pub fn new(device: &Device, channels: usize, gpool: GpoolConfig) -> Self {
    let gpool_channels = gpool.channels;
    Self {
      conv1r: Conv2dConfig::new([channels, channels - gpool_channels], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      conv1g: Conv2dConfig::new([channels, gpool_channels], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      normg: NormMask::new(device, gpool_channels, false),
      linearg: LinearConfig::new(3 * gpool_channels, channels - gpool_channels)
        .with_bias(false)
        .init(device),
      lineargate: gpool.gate.then(|| {
        let mut linear = LinearConfig::new(3 * gpool_channels, channels - gpool_channels)
          .with_bias(false)
          .init(device);
        linear.weight = Param::from_tensor(linear.weight.val().zeros_like());
        linear
      }),
    }
  }

  /// Splits the input variance between the regular (`r`) and global-pooling (`g`) branches
  /// so they add back up to roughly `scale`.
  fn initialize(&mut self, scale: f64, device: &Device) {
    let gain = mish_gain();
    let r_scale = 0.8_f64;
    let g_scale = 0.6_f64;
    init_conv(&mut self.conv1r, scale * r_scale, gain, device);
    init_conv(&mut self.conv1g, scale.sqrt() * g_scale.sqrt(), gain, device);
    init_linear(
      &mut self.linearg,
      scale.sqrt() * g_scale.sqrt(),
      gain,
      0.0,
      gain,
      device,
    );
    // `normg` stays a learnable affine (fixup uses no fixed scale here).
    if let Some(lineargate) = &mut self.lineargate {
      lineargate.weight = Param::from_tensor(lineargate.weight.val().zeros_like());
    }
  }

  /// The mean of every channel over the board, the mean scaled by the
  /// [`board_size_feature`], and the maximum.
  fn gpool(inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let size = board_size_feature(mask_sum_hw.clone());
    let layer_mean = inputs.clone().sum_dim(2).sum_dim(3) / mask_sum_hw;
    // Activation functions is always greater than -1.0, and map 0 -> 0
    let layer_max = (inputs + (mask - 1.0)).max_dim(2).max_dim(3);

    Tensor::cat(vec![layer_mean.clone(), layer_mean * size, layer_max], 1)
  }

  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let outr = conv_forward(&self.conv1r, inputs.clone());
    let outg = conv_forward(&self.conv1g, inputs);
    let outg = self.normg.forward(outg, mask.clone());
    let outg = mish(outg);
    let pooled = Self::gpool(outg, mask, mask_sum_hw).squeeze_dims::<2>(&[2, 3]);
    let outr = match &self.lineargate {
      Some(lineargate) => {
        let gate = sigmoid(lineargate.forward(pooled.clone())) * 2.0;
        let dtype = outr.dtype();
        outr * gate.unsqueeze_dims::<4>(&[-1, -1]).cast(dtype)
      }
      None => outr,
    };
    let outg = self.linearg.forward(pooled).unsqueeze_dims(&[-1, -1]);
    outr + outg
  }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum ConvOrGpool {
  Conv(Conv2d),
  Gpool(ConvAndGPool),
}

impl ConvOrGpool {
  /// A global pooling convolution when `gpool` is given, whose output
  /// channels are `in_channels - gpool.channels` rather than `out_channels`.
  pub fn new(
    device: &Device,
    gpool: Option<GpoolConfig>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    if let Some(gpool) = gpool {
      Self::Gpool(ConvAndGPool::new(device, in_channels, gpool))
    } else {
      Self::Conv(
        Conv2dConfig::new([in_channels, out_channels], kernel_size)
          .with_padding(PaddingConfig2d::Same)
          .with_bias(false)
          .init(device),
      )
    }
  }

  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    match self {
      Self::Conv(conv) => conv_forward(conv, inputs),
      Self::Gpool(gpool) => gpool.forward(inputs, mask, mask_sum_hw),
    }
  }
}

#[derive(Module, Debug)]
pub struct NormActConv {
  pub(crate) norm: NormMask,
  pub(crate) convgpool: ConvOrGpool,
}

impl NormActConv {
  pub fn new(
    device: &Device,
    gamma: bool,
    gpool: Option<GpoolConfig>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    Self {
      norm: NormMask::new(device, in_channels, gamma),
      convgpool: ConvOrGpool::new(device, gpool, in_channels, out_channels, kernel_size),
    }
  }

  /// only the convolution is rescaled; the norm stays a learnable affine
  /// since fixup applies no fixed scale to it.
  pub(crate) fn initialize(&mut self, scale: f64, device: &Device) {
    match &mut self.convgpool {
      ConvOrGpool::Conv(conv) => init_conv(conv, scale, mish_gain(), device),
      ConvOrGpool::Gpool(gpool) => gpool.initialize(scale, device),
    }
  }

  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let out = self.norm.forward(inputs, mask.clone());
    let out = mish(out);
    self.convgpool.forward(out, mask, mask_sum_hw)
  }
}

/// Squared softplus with a gradient floor: the forward value is
/// `softplus(x / 2)^2`, but the backward pass is the gradient of the plain
/// `softplus(x)` floored at `grad_floor`, so the output can't get stuck with a
/// vanishing gradient when `x` goes very negative. The detach trick keeps the
/// forward value exact while routing gradients through the floored path,
/// whose derivative is `grad_floor + (1 - grad_floor) * sigmoid(x)`.
fn squared_softplus_with_gradient_floor(x: Tensor<2>, grad_floor: f64) -> Tensor<2> {
  let value = softplus(x.clone() * 0.5, 1.0).powi_scalar(2);
  let grad_path = x.clone() * grad_floor + softplus(x, 1.0) * (1.0 - grad_floor);
  grad_path.clone() + (value - grad_path).detach()
}

/// Output scale of the TD score head, in points: the layer's raw output is
/// multiplied by this, so a raw output of about one is a score of a normal size.
const TD_SCORE_SCALE: f64 = 20.0;

/// Output scale of the short-term score error head, in points squared.
const SCORE_ERROR_SCALE: f64 = 150.0;

/// Everything [`ValueHead`] predicts about a batch of positions.
pub struct ValuePredictions {
  /// `(win, loss)` logit pairs: the main value trained towards the final result
  /// first, then one pair per TD horizon, shortest horizon last.
  pub value: Tensor<2>,
  /// Predicted squared error of the shortest-horizon TD value.
  pub value_error: Tensor<2>,
  /// Predicted score at each TD horizon, in points, shortest horizon last.
  pub td_score: Tensor<2>,
  /// Predicted squared error of the shortest-horizon TD score, in points squared.
  pub score_error: Tensor<2>,
  /// Log-distribution of the terminal score over the score bins.
  pub score: Tensor<2>,
}

/// Value head output channels of `linear_valuehead`:
/// 0..2 - main value (win, loss) logits trained towards the final result,
/// then `(win, loss)` logit pairs for each of the `TD_VALUES` horizons,
/// shortest horizon last.
#[derive(Module, Debug)]
pub struct ValueHead {
  conv1: Conv2d,
  bias1: NormMask,
  linear2: Linear,
  linear_valuehead: Linear,
  /// Predicts the squared error of the shortest-horizon TD value, i.e. how
  /// uncertain the value estimate of this position is in the short term.
  linear_error: Linear,
  /// Predicts the score at each TD horizon, in points.
  linear_td_score: Linear,
  /// Predicts the squared error of the shortest-horizon TD score, i.e. how
  /// uncertain that score is - the score's counterpart of `linear_error`.
  linear_score_error: Linear,
  // Score belief components
  linear_s2: Linear,
  linear_s2off: Linear,
  linear_s3: Linear,
  linear_smix: Linear,
}

/// How much the score belief's per-bin input grows per point of score. The
/// head tells bins apart only through this input, so neighbouring scores have
/// to differ by enough of it for the head to resolve single points: at 0.05,
/// the scores a game realistically ends at, within some 30 points, span about
/// the unit range its layers are scaled for. At 0.002 neighbouring scores
/// differed so little that the belief stayed nearly flat - some 26 points
/// wide even at a game's final position - for a whole run.
const SCORE_BELIEF_OFFSET_SCALE: f64 = 0.05;

/// The score belief's inputs for every bin, see [`score_belief_offsets`].
pub const SCORE_BELIEF_OFFSETS: usize = 2;

/// The score of every bin of the score belief, in points: the bins are whole
/// points.
fn score_belief_bin_scores(device: &Device) -> Tensor<1> {
  Tensor::<1, Int>::arange(0..SCORE_ONE_HOT_SIZE as i64, device)
    .float()
    .sub_scalar((SCORE_ONE_HOT_SIZE / 2) as f64)
}

/// The score belief's inputs for every bin, `[batch, SCORE_ONE_HOT_SIZE, 2]`
/// for the `[batch, 1]` komi in points:
///
/// 0. the bin's score times [`SCORE_BELIEF_OFFSET_SCALE`];
/// 1. how much of a level board's result falls into the bin under this komi -
///    the target a game ending level trains it towards: all of it in the bin
///    of 0 under a whole komi, where it is a draw, and half of it in each of
///    the two bins around a half one.
///
/// The first only sets bins apart smoothly, so on its own the head cannot put
/// most of the belief into a single bin, and the bin of a draw is the one bin
/// it needs that for: games are drawn often, and the probability of a draw is
/// read from that bin. The second marks it, and it is added to the term from
/// the position before the activation, so that the head can raise it by how
/// likely the position is to end level. Both are constants of the
/// architecture rather than parameters, so that they aren't stored in model
/// files and can change without them.
fn score_belief_offsets(komi: Tensor<2>) -> Tensor<3> {
  let [batch, _] = komi.dims();
  let scores = score_belief_bin_scores(&komi.device()).reshape([1, SCORE_ONE_HOT_SIZE]);
  let komi = komi.cast(scores.dtype());
  let level = (-(scores.clone() - komi).abs() + 1.0).clamp_min(0.0);
  let offsets = (scores * SCORE_BELIEF_OFFSET_SCALE).expand([batch, SCORE_ONE_HOT_SIZE]);
  Tensor::stack(vec![offsets, level], 2)
}

impl ValueHead {
  pub fn new(device: &Device, inner_channels: usize, config: &HeadsConfig) -> Self {
    Self {
      conv1: Conv2dConfig::new([inner_channels, config.v1_channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      bias1: NormMask::new(device, config.v1_channels, false),
      linear2: LinearConfig::new(VALUE_POOLED * config.v1_channels, config.v2_size).init(device),
      linear_valuehead: LinearConfig::new(config.v2_size, 2 + 2 * TD_VALUES).init(device),
      linear_error: LinearConfig::new(config.v2_size, 1).init(device),
      linear_td_score: LinearConfig::new(config.v2_size, TD_VALUES).init(device),
      linear_score_error: LinearConfig::new(config.v2_size, 1).init(device),

      linear_s2: LinearConfig::new(VALUE_POOLED * config.v1_channels, config.sbv2_size).init(device),
      linear_s2off: LinearConfig::new(SCORE_BELIEF_OFFSETS, config.sbv2_size)
        .with_bias(false)
        .init(device),
      linear_s3: LinearConfig::new(config.sbv2_size, config.num_scorebeliefs).init(device),
      linear_smix: LinearConfig::new(VALUE_POOLED * config.v1_channels, config.num_scorebeliefs).init(device),
    }
  }

  /// Pre-pooling layers keep unit-ish variance while the output
  /// projections are scaled down so the head starts near-neutral.
  fn initialize(&mut self, device: &Device) {
    let gain = mish_gain();
    let bias_scale = 0.2_f64;
    let scorebelief_output_scale = 0.5_f64;

    init_conv(&mut self.conv1, 1.0, gain, device);
    init_linear(&mut self.linear2, 1.0, gain, bias_scale, gain, device);
    // Identity gain (1.0) for output projections.
    init_linear(&mut self.linear_valuehead, 1.0, 1.0, bias_scale, 1.0, device);
    init_linear(&mut self.linear_error, 1.0, 1.0, bias_scale, 1.0, device);
    init_linear(&mut self.linear_td_score, 1.0, 1.0, bias_scale, 1.0, device);
    init_linear(&mut self.linear_score_error, 1.0, 1.0, bias_scale, 1.0, device);

    init_linear(&mut self.linear_s2, 1.0, gain, 1.0, gain, device);
    // `linear_s2off` has only a couple of input features, so it borrows `linear_s2`'s fan-in to
    // avoid a huge std; it has no bias.
    let s2off_dims = self.linear_s2off.weight.val().dims();
    let s2_fan_in = self.linear_s2.weight.val().dims()[0];
    self.linear_s2off.weight = init_weight(s2off_dims, s2_fan_in, 1.0, gain, device);
    init_linear(
      &mut self.linear_s3,
      scorebelief_output_scale,
      1.0,
      scorebelief_output_scale * bias_scale,
      1.0,
      device,
    );
    init_linear(&mut self.linear_smix, 1.0, 1.0, bias_scale, 1.0, device);
    // `bias1` stays a learnable affine.
  }

  /// The mean of every channel over the board, and the mean scaled by the
  /// [`board_size_feature`]. Nothing grows faster with the size than that: a
  /// square of it would take a board bigger than any trained on several times
  /// as far from what training showed.
  fn gpool(inputs: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let size = board_size_feature(mask_sum_hw.clone());
    let layer_mean = inputs.sum_dim(2).sum_dim(3) / mask_sum_hw;
    Tensor::cat(vec![layer_mean.clone(), layer_mean * size], 1)
  }

  /// The value outputs, with the score belief under the `[batch, 1]` komi
  /// of the player to move, in points.
  pub fn forward(
    &self,
    inputs: Tensor<4>,
    mask: Tensor<4>,
    mask_sum_hw: Tensor<4>,
    komi: Tensor<2>,
  ) -> ValuePredictions {
    let outv1 = conv_forward(&self.conv1, inputs);
    let outv1 = self.bias1.forward(outv1, mask.clone());
    let outv1 = mish(outv1);
    let outpooled = Self::gpool(outv1, mask_sum_hw).reshape([0, -1]);

    // Main Value Head

    let outv2 = self.linear2.forward(outpooled.clone());
    let outv2 = mish(outv2);
    let out_value = self.linear_valuehead.forward(outv2.clone());
    let out_value_error = self.value_error(outv2.clone());
    let out_td_score = self.linear_td_score.forward(outv2.clone()) * TD_SCORE_SCALE;
    // Squared softplus keeps the predicted squared error positive, as for the
    // value error above.
    let out_score_error =
      squared_softplus_with_gradient_floor(self.linear_score_error.forward(outv2), 0.05) * SCORE_ERROR_SCALE;

    ValuePredictions {
      value: out_value,
      value_error: out_value_error,
      td_score: out_td_score,
      score_error: out_score_error,
      score: self.score_belief(outpooled, komi),
    }
  }

  /// The log-probabilities of the score belief, `[batch, SCORE_ONE_HOT_SIZE]`,
  /// from the pooled features under the `[batch, 1]` komi in points.
  fn score_belief(&self, outpooled: Tensor<2>, komi: Tensor<2>) -> Tensor<2> {
    // Term 1: Linear from pooled
    let s2_term = self.linear_s2.forward(outpooled.clone()).reshape([0, 1, -1]);

    // Term 2: the bins' own inputs
    let offsets = score_belief_offsets(komi).cast(self.linear_s2off.weight.val().dtype());
    let s2off_term = self.linear_s2off.forward(offsets);

    let outsv2 = s2_term + s2off_term;
    let outsv2 = mish(outsv2);
    let outsv3 = self.linear_s3.forward(outsv2);

    let outsmix = self.linear_smix.forward(outpooled);
    let outsmix_logweights = log_softmax(outsmix, 1);

    let out_scorebelief_logprobs = log_softmax(outsv3, 1);

    // Take the mixture distribution weighted by outsmix_logweights, as a
    // LogSumExp stabilized by subtracting the max: the terms are
    // log-probabilities, so the naive form can underflow to -inf for far-tail
    // score bins. The max is detached since its gradient cancels analytically.
    // TODO: replace with LogSumExp once it's implemented in burn
    let log_terms = out_scorebelief_logprobs + outsmix_logweights.unsqueeze_dim(1);
    let max = log_terms.clone().max_dim(2).detach();
    ((log_terms - max.clone()).exp().sum_dim(2).log() + max).squeeze_dim(2)
  }

  /// The value outputs inference reads: the win/loss logits, the predicted
  /// short-term squared value error, the longest-horizon TD score (the head
  /// whose target converges to the final score), and the probability of a
  /// draw under the `[batch, 1]` komi in points - the score belief's bin of a
  /// lead of exactly 0. The output projections run on slices of their
  /// weights, so the TD value distributions and the shorter score horizons -
  /// auxiliary training targets all - are never computed.
  pub fn forward_inference(
    &self,
    inputs: Tensor<4>,
    mask: Tensor<4>,
    mask_sum_hw: Tensor<4>,
    komi: Tensor<2>,
  ) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
    let outv1 = conv_forward(&self.conv1, inputs);
    let outv1 = self.bias1.forward(outv1, mask.clone());
    let outv1 = mish(outv1);
    let outpooled = Self::gpool(outv1, mask_sum_hw).reshape([0, -1]);

    // Main Value Head

    let outv2 = self.linear2.forward(outpooled.clone());
    let outv2 = mish(outv2);
    let out_value = linear(
      outv2.clone(),
      self.linear_valuehead.weight.val().slice(s![.., 0..2]),
      self
        .linear_valuehead
        .bias
        .as_ref()
        .map(|bias| bias.val().slice(s![0..2])),
    );
    let out_value_error = self.value_error(outv2.clone());
    // The TD score is a single small layer on features this path computes
    // anyway. The longest horizon comes first, see `td_lookaheads`.
    let out_td_score = linear(
      outv2,
      self.linear_td_score.weight.val().slice(s![.., 0..1]),
      self
        .linear_td_score
        .bias
        .as_ref()
        .map(|bias| bias.val().slice(s![0..1])),
    ) * TD_SCORE_SCALE;
    // The whole belief is needed to normalize its one bin, but it is a few
    // small layers on the pooled features next to the trunk's.
    let center = SCORE_ONE_HOT_SIZE / 2;
    let out_draw = self
      .score_belief(outpooled, komi)
      .slice(s![.., center..center + 1])
      .exp();
    (out_value, out_value_error, out_td_score, out_draw)
  }

  /// Predicted squared short-term value error, from the second value layer.
  /// Shared by both forward paths so that the value the search weights by is
  /// exactly the one the error head was trained to produce.
  fn value_error(&self, outv2: Tensor<2>) -> Tensor<2> {
    // Squared softplus keeps the predicted squared error positive.
    squared_softplus_with_gradient_floor(self.linear_error.forward(outv2), 0.05) * 0.25
  }
}

/// Number of channels [`PolicyHead`] predicts.
const POLICY_OUTPUTS: usize = 8;

/// Index of the short-term optimistic policy, the one the search interpolates
/// its priors towards. It sits right after the policy itself so that the two
/// channels inference reads form a prefix and the final policy conv can run on
/// a plain slice of its weight.
const OPTIMISTIC_POLICY: usize = 1;

/// Number of policy channels inference needs: the policy and the short-term
/// optimistic policy it is interpolated towards. Every channel past these is
/// an auxiliary training target.
const INFERENCE_POLICY_OUTPUTS: usize = OPTIMISTIC_POLICY + 1;

/// Index of the opponent policy, predicting the reply to the move played.
const OPPONENT_POLICY: usize = 2;

/// Indices of the soft policy and soft opponent policy, trained on flattened
/// (higher entropy) versions of the same targets.
const SOFT_POLICY: usize = 3;
const SOFT_OPPONENT_POLICY: usize = 4;

/// Index of the long-term optimistic policy, trained on the games that were won
/// or whose final score beat the prediction. Nothing reads it outside training:
/// it is an auxiliary target, there to make the trunk carry what telling those
/// games apart takes.
const LONG_OPTIMISTIC_POLICY: usize = 5;

/// Index of the per-move q values: for every cell, the pre-tanh value the
/// search would settle on for playing there - a whole board of move
/// evaluations from one forward pass, where the value head only says how good
/// the position is and the policy only which reply the search would prefer.
/// Nothing reads it outside training: it is an auxiliary target, there to make
/// the trunk carry an evaluation of every reply rather than only the chosen
/// one's.
const Q_VALUE_POLICY: usize = 6;

/// Index of the per-move q scores: for every cell, the score in points the
/// search would settle on for playing there, divided by [`TD_SCORE_SCALE`].
/// The score's counterpart of [`Q_VALUE_POLICY`], and an auxiliary target all
/// the same.
const Q_SCORE_POLICY: usize = 7;

/// Policy head output channels:
/// 0 - policy, 1 - short-term optimistic policy,
/// 2 - opponent policy, 3 - soft policy, 4 - soft opponent policy,
/// 5 - long-term optimistic policy,
/// 6 - per-move q values (pre-tanh), 7 - per-move q scores (pre-scale).
#[derive(Module, Debug)]
pub struct PolicyHead {
  conv1p: Conv2d,
  conv1g: Conv2d,
  biasg: NormMask,
  linearg: Linear,
  bias2: NormMask,
  conv2p: Conv2d,
}

impl PolicyHead {
  pub fn new(device: &Device, inner_channels: usize, config: &HeadsConfig) -> Self {
    Self {
      conv1p: Conv2dConfig::new([inner_channels, config.p1_channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      conv1g: Conv2dConfig::new([inner_channels, config.g1_channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      biasg: NormMask::new(device, config.g1_channels, false),
      linearg: LinearConfig::new(3 * config.g1_channels, config.p1_channels)
        .with_bias(false)
        .init(device),
      bias2: NormMask::new(device, config.p1_channels, false),
      conv2p: Conv2dConfig::new([config.p1_channels, POLICY_OUTPUTS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
    }
  }

  /// Split variance between the regular and global-pool branches,
  /// and scale down the final policy conv (identity gain) so initial logits are small.
  fn initialize(&mut self, device: &Device) {
    let gain = mish_gain();
    let scale_output = 0.3_f64;
    init_conv(&mut self.conv1p, 0.8, gain, device);
    init_conv(&mut self.conv1g, 1.0, gain, device);
    init_linear(&mut self.linearg, 0.6, gain, 0.0, gain, device);
    init_conv(&mut self.conv2p, scale_output, 1.0, device);
    // `biasg` and `bias2` stay learnable affines.
  }

  /// The head up to (not including) the final policy conv.
  fn features(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let outp = conv_forward(&self.conv1p, inputs.clone());
    let outg = conv_forward(&self.conv1g, inputs);
    let outg = self.biasg.forward(outg, mask.clone());
    let outg = mish(outg);
    let outg = ConvAndGPool::gpool(outg, mask.clone(), mask_sum_hw).reshape([0, -1]);
    let outg = self.linearg.forward(outg).unsqueeze_dims(&[-1, -1]);

    let outp = outp + outg;
    let outp = self.bias2.forward(outp, mask);
    mish(outp)
  }

  fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let outp = self.features(inputs, mask.clone(), mask_sum_hw);
    let outp = conv_forward(&self.conv2p, outp);
    outp - (1.0 - mask) * 5000.0
  }

  /// Only the first [`INFERENCE_POLICY_OUTPUTS`] channels: everything past
  /// them is an auxiliary training target, so the final conv runs on a slice
  /// of its weight instead of computing planes nothing would read.
  fn forward_inference(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let outp = self.features(inputs, mask.clone(), mask_sum_hw);
    let weight = self.conv2p.weight.val().slice(s![0..INFERENCE_POLICY_OUTPUTS]);
    let outp = conv2d_same(outp, weight);
    outp - (1.0 - mask) * 5000.0
  }
}

/// Output channels of the captured head: the logits of the cells captured by
/// the current player and by the opponent at the end of the game, then for
/// each future position the pre-tanh prediction of its dots, see
/// [`Batch::future`](oppai_zero::examples::Batch::future).
const CAPTURED_OUTPUTS: usize = 2 + FUTURE_POSITIONS;

/// How much each future position counts towards its loss, nearest first: the
/// farther one is much harder to predict, so it gets less of the weight.
const FUTURE_POSITION_WEIGHTS: [f32; FUTURE_POSITIONS] = [1.0, 0.25];

/// The per-cell predictions of the board ahead: the captured cells at the end
/// of the game and the future positions, see [`CAPTURED_OUTPUTS`]. Search
/// reads neither, they are there to be learned.
#[derive(Module, Debug)]
pub struct CapturedHead {
  conv: Conv2d,
}

impl CapturedHead {
  pub fn new(device: &Device, inner_channels: usize) -> Self {
    Self {
      conv: Conv2dConfig::new([inner_channels, CAPTURED_OUTPUTS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
    }
  }

  /// Scale down the output conv (identity gain) so initial logits are small.
  fn initialize(&mut self, device: &Device) {
    init_conv(&mut self.conv, 0.2, 1.0, device);
  }

  fn forward(&self, inputs: Tensor<4>) -> Tensor<4> {
    conv_forward(&self.conv, inputs)
  }
}

/// A block of the trunk, which adds its residual to the trunk's channels.
#[derive(Module, Debug)]
pub enum Block {
  Nbt(ResidualBlock),
}

impl Block {
  pub fn new(device: &Device, inner_channels: usize, config: &BlockConfig) -> Self {
    match config {
      BlockConfig::Nbt(config) => Self::Nbt(ResidualBlock::new(device, inner_channels, config)),
    }
  }

  /// Fixup initialization of the block: it has to start as the identity, and
  /// `fixup_scale` is the scale its residual branch gets otherwise.
  fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    match self {
      Self::Nbt(block) => block.initialize(fixup_scale, device),
    }
  }

  fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>, chains: Option<&Chains>) -> Tensor<4> {
    match self {
      Self::Nbt(block) => block.forward(inputs, mask, mask_sum_hw, chains),
    }
  }

  /// Whether the block pools over the chains of dots.
  fn pools_chains(&self) -> bool {
    match self {
      Self::Nbt(block) => block.chainpool.is_some(),
    }
  }
}

/// The network: the input embedding, the trunk of residual blocks, and the
/// heads reading the trunk's output.
#[derive(Module, Debug)]
pub struct Model {
  pub(crate) conv_spatial: Conv2d,
  pub(crate) linear_global: Linear,
  pub(crate) residuals: Vec<Block>,
  pub(crate) norm_trunkfinal: NormMask,
  pub(crate) value_head: ValueHead,
  pub(crate) policy_head: PolicyHead,
  pub(crate) captured_head: CapturedHead,
}

impl Model {
  pub fn new(device: &Device, config: &ModelConfig) -> Self {
    let inner_channels = config.inner_channels;
    Self {
      conv_spatial: Conv2dConfig::new([FEATURE_CHANNELS, inner_channels], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      linear_global: LinearConfig::new(GLOBAL_FEATURES, inner_channels)
        .with_bias(false)
        .init(device),
      residuals: config
        .blocks
        .iter()
        .map(|block| Block::new(device, inner_channels, block))
        .collect(),
      norm_trunkfinal: NormMask::new(device, inner_channels, false),
      value_head: ValueHead::new(device, inner_channels, &config.heads),
      policy_head: PolicyHead::new(device, inner_channels, &config.heads),
      captured_head: CapturedHead::new(device, inner_channels),
    }
  }

  /// Creates the model and loads its weights from `path`. The weights are cast
  /// to the device's default float type, so a model saved in full precision
  /// can be evaluated in half precision on a device configured for it.
  pub fn load<P: AsRef<Path>>(device: &Device, config: &ModelConfig, path: P) -> Result<Self, RecordError> {
    Self::new(device, config).try_load_record(ModuleRecord::load(path)?.cast_to_module_dtype())
  }

  /// Creates the model of `config`, which may pool over the chains of dots,
  /// from the weights of a model trained with the same architecture without
  /// that pooling: `base`, of [`ModelConfig::without_chains`]. The pooling
  /// starts initialized, so as the identity - the new model computes what
  /// `base` does - and grows in with training.
  pub fn adding_chains(device: &Device, config: &ModelConfig, base: Self) -> Result<Self, RecordError> {
    let mut model = Self::new(device, config);
    model.initialize(device);
    model.try_load_record(base.into_record().allow_partial(true))
  }

  /// [`Self::adding_chains`] to the model saved at `path`, which has to load
  /// with [`ModelConfig::without_chains`] of `config`.
  pub fn load_adding_chains<P: AsRef<Path>>(
    device: &Device,
    config: &ModelConfig,
    path: P,
  ) -> Result<Self, RecordError> {
    let base = Self::load(device, &config.without_chains(), path)?;
    Self::adding_chains(device, config, base)
  }

  /// The model with the weights of its convolutions cast to `dtype`, and with
  /// them the convolutions themselves (see [`conv2d_same`]), while everything
  /// else stays in the type it is in. Those are nearly all
  /// the work of the network, and in single precision many GPUs cannot run
  /// them on their matrix units, so a half precision type makes them many
  /// times faster. The casts are part of the autodiff graph, so the gradients
  /// still reach the original weights, and the backward pass of what was cast
  /// runs in `dtype` as well.
  pub fn with_compute_dtype(self, dtype: FloatDType) -> Self {
    self.map(&mut ComputeCaster {
      dtype,
      in_compute: Vec::new(),
    })
  }

  /// Fixup initialization for the residual trunk and heads. Every residual branch is
  /// zero-initialized so the network starts as a shallow function and each block's first conv
  /// is scaled by `1/sqrt(num_blocks)`, keeping activation and gradient variance stable
  /// through depth without any explicit normalization. Must be called once on a freshly
  /// created model before training; it is a no-op to call again before loading weights.
  pub fn initialize(&mut self, device: &Device) {
    let gain = mish_gain();
    init_conv(&mut self.conv_spatial, 0.8, gain, device);
    {
      let dims = self.linear_global.weight.val().dims();
      self.linear_global.weight = init_weight(dims, dims[0], 0.6, gain, device);
    }

    let fixup_scale = 1.0 / (self.residuals.len() as f64).sqrt();
    for residual in &mut self.residuals {
      residual.initialize(fixup_scale, device);
    }
    // `norm_trunkfinal` stays a learnable affine (fixup applies no fixed scale).

    self.policy_head.initialize(device);
    self.value_head.initialize(device);
    self.captured_head.initialize(device);
  }

  /// The embedding and the trunk: the features the heads read, the board
  /// mask, and its area.
  ///
  /// `spatial` holds all the input planes: the first convolution reads the
  /// features, and the chain planes after them only tell the blocks that pool
  /// over chains which cells to pool together - a model without such blocks
  /// ignores them.
  fn trunk(&self, spatial: Tensor<4>, global: Tensor<2>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let chains = self
      .residuals
      .iter()
      .any(Block::pools_chains)
      .then(|| Chains::new(spatial.clone().slice(s![.., FEATURE_CHANNELS..CHANNELS])));
    let spatial = spatial.slice(s![.., 0..FEATURE_CHANNELS]);
    let mask = spatial.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let x_spatial = conv_forward(&self.conv_spatial, spatial);
    let x_global = self.linear_global.forward(global).unsqueeze_dims(&[-1, -1]);
    let mut x = x_spatial + x_global;
    for residual in &self.residuals {
      x = residual.forward(x, mask.clone(), mask_sum_hw.clone(), chains.as_ref());
    }
    x = self.norm_trunkfinal.forward(x, mask.clone());
    (mish(x), mask, mask_sum_hw)
  }

  /// The policy logits, the value outputs and the outputs of the captured
  /// head, see [`CAPTURED_OUTPUTS`].
  pub fn forward(&self, spatial: Tensor<4>, global: Tensor<2>) -> (Tensor<4>, ValuePredictions, Tensor<4>) {
    let komi = global.clone().slice(s![.., GLOBAL_KOMI..GLOBAL_KOMI + 1]);
    let (x, mask, mask_sum_hw) = self.trunk(spatial, global);
    let policy = self.policy_head.forward(x.clone(), mask.clone(), mask_sum_hw.clone());
    let captured = self.captured_head.forward(x.clone());
    let value = self.value_head.forward(x, mask, mask_sum_hw, komi);
    (policy, value, captured)
  }

  /// The outputs inference reads: the policy logits, then the value outputs
  /// of [`ValueHead::forward_inference`].
  pub fn forward_inference(
    &self,
    spatial: Tensor<4>,
    global: Tensor<2>,
  ) -> (Tensor<4>, Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
    let komi = global.clone().slice(s![.., GLOBAL_KOMI..GLOBAL_KOMI + 1]);
    let (x, mask, mask_sum_hw) = self.trunk(spatial, global);
    let policy = self
      .policy_head
      .forward_inference(x.clone(), mask.clone(), mask_sum_hw.clone());
    let (value, value_error, td_score, draw) = self.value_head.forward_inference(x, mask, mask_sum_hw, komi);
    (policy, value, value_error, td_score, draw)
  }

  /// The float type the network takes its inputs in.
  fn input_dtype(&self) -> DType {
    self.conv_spatial.weight.val().dtype()
  }
}

/// How much a `surprise` counts towards an optimistic policy: how far the outcome
/// beat the net's own prediction, measured in the standard deviations the net
/// predicted for itself (`predicted_sq_error` is that squared, and
/// `variance_floor` is added to it), through a soft threshold at 1.5 of them. An
/// outcome that went as predicted barely counts, one that beat the prediction by
/// three standard deviations counts almost in full, and one that went worse
/// counts for practically nothing.
///
/// Measuring the surprise against the net's own predicted error is what keeps
/// the threshold meaningful everywhere: the same swing is a shrug in a wild
/// position and a revelation in a quiet one. The floor on the variance keeps a
/// supremely confident prediction from turning a rounding error into a huge
/// surprise.
fn stdevs_excess_weight(surprise: Tensor<2>, predicted_sq_error: Tensor<2>, variance_floor: f64) -> Tensor<2> {
  let stdevs_excess = surprise / (predicted_sq_error + variance_floor).sqrt();
  sigmoid((stdevs_excess - 1.5) * 3.0)
}

/// The variance floor of a surprise measured in win probability, whose scale is
/// `[-1, 1]`, and of one measured in points.
const VALUE_VARIANCE_FLOOR: f64 = 1e-4;
const SCORE_VARIANCE_FLOOR: f64 = 0.25;

/// The score each bin of the score belief stands for, in points, as a row to
/// broadcast over a batch of distributions.
fn score_bins(device: &Device) -> Tensor<2> {
  let center = (SCORE_ONE_HOT_SIZE / 2) as f32;
  let bins = (0..SCORE_ONE_HOT_SIZE).map(|i| i as f32 - center).collect::<Vec<_>>();
  Tensor::from_data(TensorData::new(bins, [1, SCORE_ONE_HOT_SIZE]), device)
}

/// Mean and variance, in points, of a distribution over the score bins - of the
/// score belief the net predicts, or of the one-hot target it is trained on,
/// whose mean is then simply the score the game ended at.
fn score_mean_and_variance(probs: Tensor<2>, bins: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
  let mean = (probs.clone() * bins.clone()).sum_dim(1);
  let mean_sq = (probs * bins.powi_scalar(2)).sum_dim(1);
  // Guard against numerical imprecision producing a negative variance.
  let variance = (mean_sq - mean.clone().powi_scalar(2)).clamp_min(0.0);
  (mean, variance)
}

/// Interpolates a batch's policy logits from the trained policy towards the
/// optimistic one and returns the single plane of logits to take the softmax of.
/// `optimism` holds one weight per position, shaped `[batch, 1, 1]` so that it
/// broadcasts over the board.
///
/// Interpolating the logits rather than the probabilities makes the blend the
/// weighted geometric mean of the two policies, which needs both heads to like a
/// move: a move the trained policy gives almost no mass to stays almost
/// massless however much the optimistic head likes it. Averaging the
/// probabilities instead would put a floor of `optimism * p` under every such
/// move, so even a small weight would promote whatever the optimistic head
/// happened to be confident about. The two normalizing constants dropped along
/// the way are per position, so the softmax that follows divides them out
/// anyway.
fn interpolate_policy(policy_logits: Tensor<4>, optimism: Tensor<3>) -> Tensor<3> {
  let policy: Tensor<3> = policy_logits.clone().slice(s![.., 0..1, .., ..]).squeeze_dim(1);
  let optimistic: Tensor<3> = policy_logits
    .slice(s![.., OPTIMISTIC_POLICY..OPTIMISTIC_POLICY + 1, .., ..])
    .squeeze_dim(1);
  policy.clone() + (optimistic - policy) * optimism
}

#[derive(Clone)]
pub struct Predictor {
  pub model: Model,
  pub device: Device,
}

#[derive(Clone)]
pub struct Learner {
  pub predictor: Predictor,
  pub optimizer: ModuleOptimizer,
  /// The bound every training step scales the gradient down to when its norm -
  /// the norm of the whole gradient, see [`gradient_norm`] - exceeds it. `None`
  /// leaves the gradients as they come out of the backward pass.
  pub gradient_clipping: Option<f32>,
  /// The float type the convolutions compute in during
  /// training, `None` for the type of the weights. See
  /// [`Model::with_compute_dtype`]; brain floating point keeps the range of
  /// single precision, so the small gradients of a loss averaged over a batch
  /// do not underflow the way they do in the 16 bit IEEE type.
  pub compute_dtype: Option<FloatDType>,
}

#[derive(Error, Debug, From)]
pub enum ModelError {
  #[error("shape error")]
  ShapeError(ShapeError),
  #[error("data error")]
  DataError(DataError),
  #[error("execution error")]
  ExecutionError(ExecutionError),
}

fn into_data_vec<A: Clone, D: Dimension>(array: Array<A, D>) -> Vec<A> {
  let len = array.len();
  let (mut vec, offset) = if array.is_standard_layout() {
    array.into_raw_vec_and_offset()
  } else {
    array.as_standard_layout().to_owned().into_raw_vec_and_offset()
  };
  if let Some(offset) = offset {
    vec.drain(0..offset);
  }
  vec.truncate(len);
  vec
}

impl OppaiModel<f32> for Predictor {
  type E = ModelError;

  async fn predict(
    &self,
    inputs: Array4<f32>,
    global: Array2<f32>,
    optimism: Array1<f32>,
  ) -> Result<(Array3<f32>, Array2<f32>), Self::E> {
    let (batch, channels, height, width) = inputs.dim();
    // The network runs in the float type of its weights, which may be half
    // precision (see [`Model::load`]); everything after it runs in single
    // precision, so the softmaxes and what the search reads back are as precise
    // as they would be anyway.
    let dtype = self.model.input_dtype();
    let inputs = Tensor::<4>::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.device,
    )
    .cast(dtype);
    let global = Tensor::<2>::from_data(
      TensorData::new(into_data_vec(global), [batch, GLOBAL_FEATURES]),
      &self.device,
    )
    .cast(dtype);
    let optimism = Tensor::<3>::from_data(TensorData::new(into_data_vec(optimism), [batch, 1, 1]), &self.device)
      .cast(FloatDType::F32);
    let (policy_logits, value_logits, value_error, td_score, draw) = self.model.forward_inference(inputs, global);
    let (policy_logits, value_logits, value_error, td_score, draw) = (
      policy_logits.cast(FloatDType::F32),
      value_logits.cast(FloatDType::F32),
      value_error.cast(FloatDType::F32),
      td_score.cast(FloatDType::F32),
      draw.cast(FloatDType::F32),
    );
    let policy_logits = interpolate_policy(policy_logits, optimism);
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    // The predicted squared error becomes a standard deviation for the
    // search's uncertainty weighting. The score estimate is the
    // longest-horizon TD score, the head whose target converges to the final
    // score; the search averages it over each subtree into the per-move q
    // score targets. The probability of a draw tells the search which
    // positions are settled draws.
    let values = Tensor::cat(vec![softmax(value_logits, 1), value_error.sqrt(), td_score, draw], 1);
    // Both are read back in one transaction: a separate read for each would
    // wait on the device twice.
    let [policies, values]: [TensorData; 2] = Transaction::default()
      .register(policies)
      .register(values)
      .execute_async()
      .await?
      .try_into()
      .expect("a transaction returns one data per registered tensor");
    let policies = Array3::from_shape_vec((batch, height, width), policies.try_into_vec()?)?;
    let values = Array2::from_shape_vec((batch, VALUE_COLUMNS), values.try_into_vec()?)?;
    Ok((policies, values))
  }
}

impl OppaiModel<f32> for Learner {
  type E = ModelError;

  async fn predict(
    &self,
    inputs: Array4<f32>,
    global: Array2<f32>,
    optimism: Array1<f32>,
  ) -> Result<(Array3<f32>, Array2<f32>), Self::E> {
    self.predictor.predict(inputs, global, optimism).await
  }
}

/// Collects the parameters as flattened tensors, which stay on the device: a
/// visitor sees parameters of every rank, and flattening gives them one type.
struct ParamCollector {
  params: Vec<Tensor<1>>,
}

impl ModuleVisitor for ParamCollector {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    self.params.push(param.val().reshape([-1]));
  }
}

struct EmaMapper<'a> {
  params: std::collections::VecDeque<Tensor<1>>,
  factor: f64,
  device: &'a Device,
}

impl ModuleMapper for EmaMapper<'_> {
  fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
    let current = self.params.pop_front().expect("model architectures must match");
    let (id, average, mapper) = param.consume();
    let current = current.to_device(self.device).reshape(average.shape());
    let average = average.clone() + (current - average) * self.factor;
    Param::from_mapped_value(id, average, mapper)
  }
}

/// Update an exponential moving average of the model weights:
/// `average += factor * (model - average)` for every parameter, pairing the
/// parameters of the two models by module traversal order. With
/// `factor = 1 / swa_scale`, `swa_scale` snapshots are averaged in
/// expectation, smoothing out the noise of the last training steps.
pub fn ema_update(average: Model, model: &Model, factor: f64, device: &Device) -> Model {
  let mut collector = ParamCollector { params: Vec::new() };
  model.visit(&mut collector);
  let mut mapper = EmaMapper {
    params: collector.params.into(),
    factor,
    device,
  };
  average.map(&mut mapper)
}

struct ParamNormVisitor {
  sum_sq: Tensor<1>,
}

impl ParamNormVisitor {
  fn new(device: &Device) -> Self {
    Self {
      sum_sq: Tensor::zeros([1], device),
    }
  }

  fn l2_norm(self) -> f32 {
    self.sum_sq.sqrt().into_scalar()
  }
}

impl ModuleVisitor for ParamNormVisitor {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    let tensor = param.val();
    self.sum_sq = self.sum_sq.clone() + (tensor.clone() * tensor).sum();
  }
}

struct GradNormVisitor<'a> {
  grads: &'a GradientsParams,
  sum_sq: Tensor<1>,
}

impl<'a> GradNormVisitor<'a> {
  fn new(grads: &'a GradientsParams, device: &Device) -> Self {
    Self {
      grads,
      sum_sq: Tensor::zeros([1], &device.clone().without_autodiff()),
    }
  }
}

impl ModuleVisitor for GradNormVisitor<'_> {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    if let Some(grad) = self.grads.get::<D>(param.id) {
      self.sum_sq = self.sum_sq.clone() + grad.square().sum();
    }
  }
}

struct GradScaleVisitor<'a> {
  grads: &'a mut GradientsParams,
  scale: f32,
}

impl ModuleVisitor for GradScaleVisitor<'_> {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
    if let Some(grad) = self.grads.remove::<D>(param.id) {
      self.grads.register::<D>(param.id, grad.mul_scalar(self.scale));
    }
  }
}

/// The L2 norm of the model's whole gradient, every parameter's gradient taken
/// as a single vector. It is the norm of the gradient of the loss as it is
/// defined here, a mean over the batch, so a norm stated for a loss summed over
/// the batch is this one times the batch size.
fn gradient_norm(model: &Model, grads: &GradientsParams, device: &Device) -> f32 {
  let mut visitor = GradNormVisitor::new(grads, device);
  model.visit(&mut visitor);
  visitor.sum_sq.sqrt().into_scalar()
}

/// Multiply every gradient by the same factor, which scaling the whole gradient
/// down to a bound needs: it leaves the direction of the update alone, unlike
/// bounding each parameter's gradient on its own, which would also make the
/// bound depend on how the parameters happen to be split into tensors.
///
/// TODO: switch to burn's gradient clipping once it can clip by the norm of
/// the whole gradient; its `GradientClipping::Norm` bounds every parameter's
/// gradient on its own.
fn scale_gradients(model: &Model, grads: GradientsParams, scale: f32) -> GradientsParams {
  let mut grads = grads;
  let mut visitor = GradScaleVisitor {
    grads: &mut grads,
    scale,
  };
  model.visit(&mut visitor);
  grads
}

/// Every term of the training loss, each already weighted and averaged over the
/// batch, named as they are logged and in that order.
pub struct Losses(pub [(&'static str, Tensor<1>); 17]);

impl Losses {
  /// The terms as numbers, reading them back from the device.
  pub fn values(self) -> Vec<(&'static str, f32)> {
    self
      .0
      .into_iter()
      .map(|(name, loss)| (name, loss.into_scalar::<f32>()))
      .collect()
  }
}

impl Predictor {
  /// A copy that evaluates without recording anything for a backward pass, for
  /// computing losses that are only read, like on a validation set: on an
  /// autodiff device every forward pass keeps its activations for the backward
  /// pass that never comes.
  pub fn valid(&self) -> Self {
    Predictor {
      model: self.model.valid(),
      device: self.device.clone().without_autodiff(),
    }
  }

  /// What the captured head predicts for each cell: the probabilities of it
  /// being captured at the end of the game by the player to move and by the
  /// opponent, then the predicted dots of each future position, see
  /// [`CAPTURED_OUTPUTS`]. Search never reads these, they are for checking
  /// what the head has learned.
  pub fn predict_board_ahead(&self, inputs: Array4<f32>, global: Array2<f32>) -> Result<Array4<f32>, ModelError> {
    let (batch, channels, height, width) = inputs.dim();
    let dtype = self.model.input_dtype();
    let inputs = Tensor::<4>::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.device,
    )
    .cast(dtype);
    let global = Tensor::<2>::from_data(
      TensorData::new(into_data_vec(global), [batch, GLOBAL_FEATURES]),
      &self.device,
    )
    .cast(dtype);
    let (_, _, captured) = self.model.forward(inputs, global);
    let captured = captured.cast(FloatDType::F32);
    let outputs = Tensor::cat(
      vec![
        sigmoid(captured.clone().slice(s![.., 0..2])),
        tanh(captured.slice(s![.., 2..CAPTURED_OUTPUTS])),
      ],
      1,
    );
    Ok(Array4::from_shape_vec(
      (batch, CAPTURED_OUTPUTS, height, width),
      outputs.into_data().try_into_vec()?,
    )?)
  }

  /// The loss terms of a batch for the model as it is, with the convolutions
  /// computing in `compute_dtype` (see
  /// [`Model::with_compute_dtype`]).
  #[allow(clippy::too_many_arguments)]
  pub fn losses(
    &self,
    compute_dtype: Option<FloatDType>,
    inputs: Array4<f32>,
    global: Array2<f32>,
    policies: Array3<f32>,
    opponent_policies: Array3<f32>,
    values: Array2<f32>,
    td_values: Array3<f32>,
    td_scores: Array2<f32>,
    scores: Array2<f32>,
    captured: Array4<f32>,
    future: Array4<f32>,
    q_values: Array4<f32>,
    outcome_weights: Array1<f32>,
  ) -> Losses {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.device,
    );
    let global = Tensor::from_data(
      TensorData::new(into_data_vec(global), [batch, GLOBAL_FEATURES]),
      &self.device,
    );
    let policies = Tensor::from_data(
      TensorData::new(into_data_vec(policies), [batch, height * width]),
      &self.device,
    );
    let opponent_policies = Tensor::from_data(
      TensorData::new(into_data_vec(opponent_policies), [batch, height * width]),
      &self.device,
    );
    let values = Tensor::from_data(TensorData::new(into_data_vec(values), [batch, 2]), &self.device);
    let td_values: Tensor<3> = Tensor::from_data(
      TensorData::new(into_data_vec(td_values), [batch, TD_VALUES, 2]),
      &self.device,
    );
    let td_scores: Tensor<2> = Tensor::from_data(
      TensorData::new(into_data_vec(td_scores), [batch, TD_VALUES]),
      &self.device,
    );
    let scores = Tensor::from_data(
      TensorData::new(into_data_vec(scores), [batch, SCORE_ONE_HOT_SIZE]),
      &self.device,
    );
    let scores_cdf = scores.clone().cumsum(1);
    let captured = Tensor::from_data(
      TensorData::new(into_data_vec(captured), [batch, 2, height, width]),
      &self.device,
    );
    let future: Tensor<4> = Tensor::from_data(
      TensorData::new(into_data_vec(future), [batch, FUTURE_POSITIONS, height, width]),
      &self.device,
    );
    let q_values: Tensor<4> = Tensor::from_data(
      TensorData::new(into_data_vec(q_values), [batch, 3, height, width]),
      &self.device,
    );
    // 0 for the side positions, which have none of the targets only the end of
    // the game gives, see `Batch::outcome_weights`.
    let outcome_weights: Tensor<2> = Tensor::from_data(
      TensorData::new(into_data_vec(outcome_weights), [batch, 1]),
      &self.device,
    );
    let q_targets = q_values.clone().slice(s![.., 0..1]).reshape([0, -1]);
    let q_score_targets = q_values.clone().slice(s![.., 1..2]).reshape([0, -1]);
    let q_weights = q_values.slice(s![.., 2..3]).reshape([0, -1]);
    // The captured head predicts the terminal captured state and the future
    // dots of every board cell, so its losses are masked only by the board
    // mask.
    let mask = inputs.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let model = match compute_dtype {
      Some(dtype) => self.model.clone().with_compute_dtype(dtype),
      None => self.model.clone(),
    };
    let (out_policy_logits, out_value, out_captured) = model.forward(inputs, global);
    let out_captured_logits = out_captured.clone().slice(s![.., 0..2]);
    let out_future = tanh(out_captured.slice(s![.., 2..CAPTURED_OUTPUTS]));
    let ValuePredictions {
      value: out_value_logits,
      value_error: out_value_error,
      td_score: out_td_scores,
      score_error: out_score_error,
      score: out_scores,
    } = out_value;
    let out_policies = log_softmax(
      out_policy_logits.clone().slice(s![.., 0..1, .., ..]).reshape([0, -1]),
      1,
    );
    let out_opponent_policies = log_softmax(
      out_policy_logits
        .clone()
        .slice(s![.., OPPONENT_POLICY..OPPONENT_POLICY + 1, .., ..])
        .reshape([0, -1]),
      1,
    );
    let out_soft_policies = log_softmax(
      out_policy_logits
        .clone()
        .slice(s![.., SOFT_POLICY..SOFT_POLICY + 1, .., ..])
        .reshape([0, -1]),
      1,
    );
    let out_soft_opponent_policies = log_softmax(
      out_policy_logits
        .clone()
        .slice(s![.., SOFT_OPPONENT_POLICY..SOFT_OPPONENT_POLICY + 1, .., ..])
        .reshape([0, -1]),
      1,
    );
    let out_long_optimistic_policies = log_softmax(
      out_policy_logits
        .clone()
        .slice(s![.., LONG_OPTIMISTIC_POLICY..LONG_OPTIMISTIC_POLICY + 1, .., ..])
        .reshape([0, -1]),
      1,
    );
    let out_q_pretanh = out_policy_logits
      .clone()
      .slice(s![.., Q_VALUE_POLICY..Q_VALUE_POLICY + 1, .., ..])
      .reshape([0, -1]);
    let out_q_score_prescale = out_policy_logits
      .clone()
      .slice(s![.., Q_SCORE_POLICY..Q_SCORE_POLICY + 1, .., ..])
      .reshape([0, -1]);
    let out_optimistic_policies = log_softmax(
      out_policy_logits
        .slice(s![.., OPTIMISTIC_POLICY..OPTIMISTIC_POLICY + 1, .., ..])
        .reshape([0, -1]),
      1,
    );
    let out_values = log_softmax(out_value_logits.clone().slice(s![.., 0..2]), 1);
    let out_scores_cdf = out_scores.clone().exp().cumsum(1);

    // Auxiliary soft policy target: the policy target raised to the power 1/4
    // and renormalized over the board, so it's a flattened (higher entropy)
    // version of the same distribution. The epsilon gives unvisited on-board
    // cells a small uniform mass; off-board cells are zeroed by the mask.
    let policy_mask = mask.clone().reshape([0, -1]);
    let soft_policies = ((policies.clone() + 1e-7) * policy_mask.clone()).powf_scalar(0.25);
    let soft_policies = soft_policies.clone() / soft_policies.sum_dim(1);
    let soft_opponent_policies = ((opponent_policies.clone() + 1e-7) * policy_mask).powf_scalar(0.25);
    let soft_opponent_policies = soft_opponent_policies.clone() / soft_opponent_policies.sum_dim(1);
    // A game's final position has no reply and carries an all-zero opponent
    // target (see `Examples::batch`), which silences the plain opponent loss by
    // itself. The soft target above is renormalized back into a distribution,
    // so such rows have to be gated explicitly by the target's total mass -
    // which is 1 for every ordinary row.
    let opponent_weight = opponent_policies.clone().sum_dim(1);

    let batch = <f32 as num_traits::NumCast>::from(batch).unwrap();
    let values_loss = -(out_values * values.clone()).sum() * 0.72 / batch;
    let td_values_loss = (0..TD_VALUES)
      .map(|i| {
        let logits = out_value_logits.clone().slice(s![.., 2 + 2 * i..4 + 2 * i]);
        let target = td_values.clone().slice(s![.., i..i + 1, ..]).reshape([0, -1]);
        -(log_softmax(logits, 1) * target).sum()
      })
      .reduce(|a, b| a + b)
      .unwrap()
      * 0.72
      / batch;
    // The short-term value error head is trained towards the actual squared
    // error of the model's own shortest-horizon TD value, with the prediction
    // detached so only the error head learns from this loss. The epsilon adds
    // a tiny irreducible error for regularization.
    let td_short_pred = softmax(
      out_value_logits
        .clone()
        .slice(s![.., 2 + 2 * (TD_VALUES - 1)..])
        .detach(),
      1,
    );
    let pred_value = td_short_pred.clone().slice(s![.., 0..1]) - td_short_pred.slice(s![.., 1..2]);
    let td_short_target = td_values.clone().slice(s![.., TD_VALUES - 1.., ..]).reshape([0, -1]);
    let real_value = td_short_target.clone().slice(s![.., 0..1]) - td_short_target.slice(s![.., 1..2]);
    // Signed, so it also says which way the net was wrong: positive means the
    // position turned out better in the short term than it predicted.
    let value_surprise = real_value - pred_value;
    let sq_error = value_surprise.clone().square() + 1e-8;
    let value_error_loss = HuberLossConfig::new(0.4)
      .init()
      .forward_no_reduction(out_value_error.clone(), sq_error)
      .sum()
      * 2.0
      / batch;
    // The TD score head is the score's counterpart of the TD value head: how the
    // score stands at each horizon, in points. Huber rather than squared error so
    // that a game whose score runs away does not dominate the gradient.
    let td_scores_loss = (HuberLossConfig::new(12.0)
      .init()
      .forward_no_reduction(out_td_scores.clone(), td_scores.clone())
      * outcome_weights.clone())
    .sum()
      * 0.0004
      / batch;
    // And the short-term score error head is trained exactly like the value one:
    // towards the squared error of the model's own shortest-horizon TD score,
    // that prediction detached so only the error head learns from it. The epsilon
    // is a hundredth of a point squared of irreducible error.
    let pred_score = out_td_scores.slice(s![.., TD_VALUES - 1..]).detach();
    let real_score = td_scores.slice(s![.., TD_VALUES - 1..]);
    let score_surprise = real_score - pred_score;
    let score_sq_error = score_surprise.clone().square() + 1e-4;
    let score_error_loss = (HuberLossConfig::new(100.0)
      .init()
      .forward_no_reduction(out_score_error.clone(), score_sq_error)
      * outcome_weights.clone())
    .sum()
      * 0.00002
      / batch;

    let policies_loss = -(out_policies * policies.clone()).sum() / batch;
    // Both optimistic policies learn the very same target as the policy above,
    // only from the positions that turned out better than the net expected. Such
    // a policy is what a search wants as its prior: it ranks the moves that pay
    // off in the lines that beat the net's expectations, which is what the search
    // is there to find, while the plain policy - an average over how those
    // positions really went - ranks them where their average says they belong.
    // Nothing here trains the value, the score or their error heads, so all of
    // them enter detached.
    //
    // The short-term one, which is the policy the search reads, counts a position
    // by how far its near-future value or score beat what the net predicted for
    // itself. Either one is enough, so the two thresholds add up - a move can pay
    // off in points while the game stays as won or lost as it was, and a decided
    // position has no win probability left to be surprised in.
    // A side position has no score to have been surprised by.
    let optimistic_weight = (stdevs_excess_weight(value_surprise, out_value_error.detach(), VALUE_VARIANCE_FLOOR)
      + stdevs_excess_weight(score_surprise, out_score_error.detach(), SCORE_VARIANCE_FLOOR) * outcome_weights.clone())
    .clamp_max(1.0);
    // The scale is about a fifth of the main policy term above: the same target
    // seen through a filter only some of the samples pass.
    let optimistic_policies_loss =
      -(out_optimistic_policies * policies.clone() * optimistic_weight).sum() * 0.215 / batch;
    // The long-term one asks the same of the whole game instead of the next few
    // turns: it counts a position by whether the game was won - squared, so that
    // a draw counts for a quarter rather than half and the target leans on the
    // wins - or by how far the final score beat the score the net believed in,
    // measured against the spread of that belief. Nothing reads this policy at
    // inference; it is here because telling those games apart is worth learning.
    let bins = score_bins(&self.device);
    let (pred_score_mean, pred_score_variance) =
      score_mean_and_variance(out_scores.clone().exp().detach(), bins.clone());
    let (real_score_mean, _) = score_mean_and_variance(scores.clone(), bins);
    let win = values.slice(s![.., 0..1]);
    // And a side position has no game to have won.
    let long_optimistic_weight = (win.powi_scalar(2)
      + stdevs_excess_weight(
        real_score_mean - pred_score_mean,
        pred_score_variance,
        SCORE_VARIANCE_FLOOR,
      ))
    .clamp_max(1.0)
      * outcome_weights.clone();
    let long_optimistic_policies_loss =
      -(out_long_optimistic_policies * policies * long_optimistic_weight).sum() * 0.108 / batch;
    let opponent_policies_loss = -(out_opponent_policies * opponent_policies).sum() * 0.15 / batch;
    let soft_policies_loss = -(out_soft_policies * soft_policies).sum() * 8.0 / batch;
    let soft_opponent_policies_loss =
      -(out_soft_opponent_policies * soft_opponent_policies * opponent_weight).sum() * 1.2 / batch;
    let pdf_loss = -(out_scores * scores * outcome_weights.clone()).sum() * 0.02 / batch;
    let cdf_loss = ((out_scores_cdf - scores_cdf).square() * outcome_weights.clone()).sum() * 0.02 / batch;
    // Binary cross-entropy with logits in the numerically stable form
    // `max(z, 0) - z * t + ln(1 + exp(-|z|))`, normalized by the board area
    // like KataGo's ownership loss.
    let captured_bce = out_captured_logits.clone().clamp_min(0.0) - out_captured_logits.clone() * captured
      + (-out_captured_logits.abs()).exp().log1p();
    let captured_loss = ((captured_bce * mask.clone()).sum_dim(2).sum_dim(3) / mask_sum_hw.clone()
      * outcome_weights.clone().reshape([0, 1, 1, 1]))
    .sum()
      * 1.5
      / batch;
    // The squared error of the predicted dots at each future position. The
    // future positions are a fixed number of moves ahead, so about the same
    // number of cells changes by then on every board - which would make it a
    // sum over the cells - but on a bigger board where those moves land is
    // less certain, so it is normalized by the side of the board, between the
    // sum and the mean over the area.
    let future_weights =
      Tensor::<1>::from_floats(FUTURE_POSITION_WEIGHTS, &self.device).reshape([1, FUTURE_POSITIONS as i32, 1, 1]);
    let future_loss = (((out_future - future).square() * mask * future_weights)
      .sum_dim(1)
      .sum_dim(2)
      .sum_dim(3)
      / mask_sum_hw.sqrt()
      * outcome_weights.reshape([0, 1, 1, 1]))
    .sum()
      * 0.25
      / batch;

    // Per-move q loss: each explored move's predicted value against the value
    // the search settled on for it, as a binary cross-entropy over the implied
    // win probability - the output is pre-tanh, and `tanh(x) = 2*sigmoid(2x) - 1`
    // makes `2x` the logit of `(1 + q) / 2`. Each move counts by the square
    // root of the search weight behind it, so the well-searched replies
    // dominate without the thin tail vanishing. Unexplored moves carry zero
    // weight; their logits are also zeroed so that the off-board wall the
    // policy head subtracts stays out of the arithmetic before the weight
    // cancels it. The denominator's +1 keeps rows without any recorded q
    // values (older data) finite - they contribute nothing.
    let q_mask = q_weights.clone().greater_elem(0.0).float();
    let q_sqrt_weights = q_weights.sqrt();
    let q_logits = out_q_pretanh * q_mask.clone() * 2.0;
    let q_target_probs = (q_targets + 1.0) / 2.0;
    let q_bce = q_logits.clone().clamp_min(0.0) - q_logits.clone() * q_target_probs + (-q_logits.abs()).exp().log1p();
    let q_values_loss =
      ((q_bce * q_sqrt_weights.clone()).sum_dim(1) / (q_sqrt_weights.clone().sum_dim(1) + 1.0)).sum() * 1.5 / batch;

    // The score's counterpart of the q loss above: each explored move's
    // predicted score against the score the search settled on for it. Huber
    // rather than squared error so that a runaway line does not dominate the
    // gradient, and the same masking and square-root weighting as above - the
    // mask again keeps the off-board wall out of the arithmetic before the
    // zero weight cancels it.
    let q_score_pred = out_q_score_prescale * q_mask * TD_SCORE_SCALE;
    let q_score_huber = HuberLossConfig::new(12.0)
      .init()
      .forward_no_reduction(q_score_pred, q_score_targets);
    let q_scores_loss =
      ((q_score_huber * q_sqrt_weights.clone()).sum_dim(1) / (q_sqrt_weights.sum_dim(1) + 1.0)).sum() * 0.0008 / batch;

    Losses([
      ("value", values_loss),
      ("td value", td_values_loss),
      ("value error", value_error_loss),
      ("td score", td_scores_loss),
      ("score error", score_error_loss),
      ("policy", policies_loss),
      ("opponent policy", opponent_policies_loss),
      ("soft policy", soft_policies_loss),
      ("soft opponent policy", soft_opponent_policies_loss),
      ("optimistic policy", optimistic_policies_loss),
      ("long optimistic policy", long_optimistic_policies_loss),
      ("pdf", pdf_loss),
      ("cdf", cdf_loss),
      ("captured", captured_loss),
      ("future position", future_loss),
      ("q", q_values_loss),
      ("q score", q_scores_loss),
    ])
  }
}

impl Learner {
  /// The loss terms of a batch for the model as it is, which is what
  /// [`OppaiTrainableModel::train`] minimizes.
  #[allow(clippy::too_many_arguments)]
  pub fn losses(
    &self,
    inputs: Array4<f32>,
    global: Array2<f32>,
    policies: Array3<f32>,
    opponent_policies: Array3<f32>,
    values: Array2<f32>,
    td_values: Array3<f32>,
    td_scores: Array2<f32>,
    scores: Array2<f32>,
    captured: Array4<f32>,
    future: Array4<f32>,
    q_values: Array4<f32>,
    outcome_weights: Array1<f32>,
  ) -> Losses {
    self.predictor.losses(
      self.compute_dtype,
      inputs,
      global,
      policies,
      opponent_policies,
      values,
      td_values,
      td_scores,
      scores,
      captured,
      future,
      q_values,
      outcome_weights,
    )
  }
}

impl OppaiTrainableModel<f32> for Learner {
  type TE = ModelError;

  fn train(
    mut self,
    inputs: Array4<f32>,
    global: Array2<f32>,
    policies: Array3<f32>,
    opponent_policies: Array3<f32>,
    values: Array2<f32>,
    td_values: Array3<f32>,
    td_scores: Array2<f32>,
    scores: Array2<f32>,
    captured: Array4<f32>,
    future: Array4<f32>,
    q_values: Array4<f32>,
    outcome_weights: Array1<f32>,
    learning_rate: f64,
  ) -> Result<Self, Self::TE> {
    let Losses(losses) = self.losses(
      inputs,
      global,
      policies,
      opponent_policies,
      values,
      td_values,
      td_scores,
      scores,
      captured,
      future,
      q_values,
      outcome_weights,
    );

    let mut norm_visitor = ParamNormVisitor::new(&self.predictor.device);
    self.predictor.model.visit(&mut norm_visitor);
    let param_l2_norm = norm_visitor.l2_norm();

    let mut message = String::from("Loss:");
    for (name, loss) in &losses {
      message += &format!(" {} {}", name, loss.clone().into_scalar::<f32>());
    }
    log::info!("{} L2 norm {}", message, param_l2_norm);

    let loss = losses.into_iter().map(|(_, loss)| loss).reduce(|a, b| a + b).unwrap();

    let mut grads = GradientsParams::from_grads(loss.backward(), &self.predictor.model);
    // Logged every batch next to the losses, because whether the bound is set
    // well is only visible from how often it engages: it is there to catch the
    // occasional outlier batch, so it should stay clear of the ordinary ones.
    let grad_norm = gradient_norm(&self.predictor.model, &grads, &self.predictor.device);
    let max_norm = self
      .gradient_clipping
      .and_then(<f32 as num_traits::NumCast>::from)
      .filter(|&max_norm| grad_norm > max_norm);
    match max_norm {
      Some(max_norm) => {
        log::info!("Gradient norm: {} clipped to {}", grad_norm, max_norm);
        grads = scale_gradients(&self.predictor.model, grads, max_norm / grad_norm);
      }
      None => log::info!("Gradient norm: {}", grad_norm),
    }
    self.predictor.model = self.optimizer.step(learning_rate, self.predictor.model, grads);

    Ok(self)
  }
}

#[cfg(all(test, any(feature = "flex", feature = "vulkan", feature = "webgpu")))]
mod tests {
  use super::{Block, BlockConfig, Learner, Model, ModelConfig, NbtConfig, Predictor};
  #[cfg(feature = "flex")]
  use super::{
    ConvOrGpool, GLOBAL_FEATURES, OPTIMISTIC_POLICY, POLICY_OUTPUTS, SCORE_VARIANCE_FLOOR, TD_SCORE_SCALE,
    VALUE_VARIANCE_FLOOR, gradient_norm, interpolate_policy, scale_gradients, score_bins, score_mean_and_variance,
    squared_softplus_with_gradient_floor, stdevs_excess_weight,
  };
  use crate::chains::ChainsConfig;
  #[cfg(feature = "flex")]
  use crate::optimizer::{OptimizerConfig, OptimizerKind};
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  use burn::tensor::DeviceKind;
  #[cfg(feature = "flex")]
  use burn::{
    optim::GradientsParams,
    tensor::{Tensor, TensorData, activation::softmax, s},
  };
  use burn::{optim::SgdConfig, tensor::Device};
  use ndarray::{Array2, Array3, Array4, Axis, array};
  #[cfg(feature = "flex")]
  use oppai_zero::field_features::FEATURE_CHANNELS;
  #[cfg(feature = "flex")]
  use oppai_zero::field_features::score_one_hot;
  use oppai_zero::{
    examples::{FUTURE_POSITIONS, TD_VALUES},
    field_features::{CHANNELS, SCORE_ONE_HOT_SIZE},
    model::{Model as OppaiModel, TrainableModel, VALUE_COLUMNS},
  };

  /// The default architecture with pooling over chains in its second and
  /// fourth blocks, the ones that pool globally too.
  fn chains_config() -> ModelConfig {
    let mut config = ModelConfig::default();
    for block in config.blocks.iter_mut().skip(1).step_by(2) {
      let BlockConfig::Nbt(block) = block;
      block.chains = Some(ChainsConfig { channels: 16 });
    }
    config
  }

  /// A chain plane of a board for the tests: chains labelled as the features
  /// label them, by their rank. The player to move has one chain on the
  /// cells of even `x + y`, and the opponent one on the odd columns below the
  /// first row and another on the last cell of the first row.
  #[cfg(feature = "flex")]
  fn chain_plane(channel: usize, y: usize, x: usize, width: usize) -> f32 {
    match channel - FEATURE_CHANNELS {
      0 if (x + y).is_multiple_of(2) => 1.0,
      1 if x % 2 == 1 && y >= 1 => 1.0,
      1 if y == 0 && x == width - 1 => 2.0,
      _ => 0.0,
    }
  }

  #[test]
  fn default_config_file_matches_default() {
    let config = ModelConfig::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/configs/b5c192nbt.json")).unwrap();
    assert_eq!(config, ModelConfig::default());
  }

  /// Every config file reads, and a config reads back as what was written.
  #[test]
  fn config_files_read() {
    for entry in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/configs")).unwrap() {
      let path = entry.unwrap().path();
      let config = ModelConfig::from_file(&path).unwrap_or_else(|error| panic!("{path:?}: {error:?}"));
      assert!(!config.blocks.is_empty(), "{path:?}");
    }
    let json = serde_json::to_string(&ModelConfig::default()).unwrap();
    assert_eq!(
      serde_json::from_str::<ModelConfig>(&json).unwrap(),
      ModelConfig::default()
    );
    // A misspelled field of a block is an error rather than a default.
    assert!(serde_json::from_str::<ModelConfig>(&json.replace("\"mid_channels\"", "\"mid_channel\"")).is_err());
  }

  // The forward value is squared softplus, while the gradient is that of the
  // plain softplus, never dropping below the floor even for very negative
  // inputs.
  #[test]
  #[cfg(feature = "flex")]
  fn squared_softplus_gradient_floor() {
    let device = Device::flex();
    let x = Tensor::<2>::from_floats([[-20.0, 0.0, 20.0]], &device.clone().autodiff()).require_grad();
    let out = squared_softplus_with_gradient_floor(x.clone(), 0.05);

    let values = out.clone().into_data().try_to_vec::<f32>().unwrap();
    let expected = [0.0f32, 0.480453, 100.000908];
    for (value, expected) in values.into_iter().zip(expected) {
      assert!((value - expected).abs() < 1e-3);
    }

    let grads = out.sum().backward();
    let grads = x.grad(&grads).unwrap().into_data().try_to_vec::<f32>().unwrap();
    let expected = [0.05f32, 0.525, 1.0];
    for (grad, expected) in grads.into_iter().zip(expected) {
      assert!((grad - expected).abs() < 1e-3);
    }
  }

  /// The norm being clipped is that of the whole gradient rather than of each
  /// parameter on its own, so scaling the gradient by the ratio of a bound to
  /// that norm brings it to exactly the bound - and every parameter's gradient
  /// takes the same factor, leaving the direction of the update as it was.
  #[test]
  #[cfg(feature = "flex")]
  fn scaling_brings_the_whole_gradient_to_the_bound() {
    let config = ModelConfig::default();
    let device = Device::flex();
    let model = Model::new(&device.clone().autodiff(), &config);
    let inputs = Tensor::ones([1, CHANNELS, 4, 8], &device);
    let global = Tensor::zeros([1, GLOBAL_FEATURES], &device);
    let (policy_logits, value_logits, ..) = model.forward_inference(inputs, global);
    let grads = GradientsParams::from_grads((policy_logits.sum() + value_logits.sum()).backward(), &model);

    let unclipped = gradient_norm(&model, &grads, &device);
    assert!(unclipped > 0.0);

    let max_norm = unclipped / 4.0;
    let clipped = scale_gradients(&model, grads, max_norm / unclipped);
    assert!((gradient_norm(&model, &clipped, &device) - max_norm).abs() < unclipped * 1e-4);
  }

  /// Only the samples that beat the net's own short-term prediction train the
  /// optimistic policy, and by how many of its own predicted standard deviations
  /// they beat it by. A position that went as predicted - or worse - must count
  /// for next to nothing, or the head just relearns the plain policy.
  #[test]
  #[cfg(feature = "flex")]
  fn optimistic_weight_follows_the_surprise() {
    let device = Device::flex();
    // A predicted error of 0.25 is a predicted standard deviation of 0.5, so
    // the threshold sits at a surprise of 0.75.
    let surprise = Tensor::<2>::from_floats([[-1.0], [0.0], [0.75], [1.5], [3.0]], &device);
    let predicted_sq_error = Tensor::<2>::from_floats([[0.25], [0.25], [0.25], [0.25], [0.25]], &device);
    let weights = stdevs_excess_weight(surprise, predicted_sq_error, VALUE_VARIANCE_FLOOR)
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();

    // Worse than predicted, and as predicted: all but ignored.
    assert!(weights[0] < 0.001, "got {}", weights[0]);
    assert!(weights[1] < 0.02, "got {}", weights[1]);
    // Exactly at the threshold: half weight, but for the floor on the variance.
    assert!((weights[2] - 0.5).abs() < 1e-3, "got {}", weights[2]);
    // Well past it: counted nearly in full, and never more than in full.
    assert!(weights[3] > 0.9 && weights[3] < 1.0, "got {}", weights[3]);
    assert!(weights[4] > 0.99 && weights[4] <= 1.0, "got {}", weights[4]);

    // A surprise in points goes through the same threshold, and the same
    // predicted error means something entirely different there: a predicted
    // squared error of 100 points is a standard deviation of 10, so 15 points
    // more than predicted is what half weight takes.
    let surprise = Tensor::<2>::from_floats([[0.75], [15.0]], &device);
    let predicted_sq_error = Tensor::<2>::from_floats([[100.0], [100.0]], &device);
    let weights = stdevs_excess_weight(surprise, predicted_sq_error, SCORE_VARIANCE_FLOOR)
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    assert!(weights[0] < 0.02, "got {}", weights[0]);
    assert!((weights[1] - 0.5).abs() < 1e-2, "got {}", weights[1]);
  }

  /// The long-term filter and the score half of the short-term one both measure
  /// the score against the belief head's own spread, so that spread has to come
  /// out of the predicted distribution: a confident belief must give a small
  /// variance and a hedged one a large variance, and the mean of the one-hot
  /// target has to be exactly the score the game ended at.
  #[test]
  #[cfg(feature = "flex")]
  fn score_belief_mean_and_variance() {
    let device = Device::flex();
    let bins = score_bins(&device);
    let center = SCORE_ONE_HOT_SIZE / 2;

    // Row 0: all of the mass on a score of 7, the way the training target
    // encodes a whole-point score. Row 1: half on -3 and half on 11.
    let mut probs = [vec![0.0f32; SCORE_ONE_HOT_SIZE], vec![0.0f32; SCORE_ONE_HOT_SIZE]];
    probs[0][center + 7] = 1.0;
    probs[1][center - 3] = 0.5;
    probs[1][center + 11] = 0.5;
    let probs = Tensor::<2>::from_data(TensorData::new(probs.concat(), [2, SCORE_ONE_HOT_SIZE]), &device);

    let (mean, variance) = score_mean_and_variance(probs, bins);
    let mean = mean.into_data().try_to_vec::<f32>().unwrap();
    let variance = variance.into_data().try_to_vec::<f32>().unwrap();

    assert!((mean[0] - 7.0).abs() < 1e-3, "got {}", mean[0]);
    assert!(variance[0] < 1e-3, "got {}", variance[0]);
    // Mean of -3 and 11, each 7 points away from it.
    assert!((mean[1] - 4.0).abs() < 1e-3, "got {}", mean[1]);
    assert!((variance[1] - 49.0).abs() < 1e-2, "got {}", variance[1]);
  }

  /// The policy the search gets is the trained one at optimism 0, the
  /// optimistic one at optimism 1, and their weighted geometric mean in
  /// between - so a move the two heads disagree about lands between the
  /// probabilities they give it, and every position of a batch is blended by
  /// its own weight.
  #[test]
  #[cfg(feature = "flex")]
  fn interpolate_policy_between_the_heads() {
    let device = Device::flex();
    // Two positions of a single cell each, so the softmax is over one channel
    // pair per row and the logits are the policies up to normalization.
    let mut logits = [[0.0f32; 2]; POLICY_OUTPUTS];
    logits[0] = [1.0, -1.0];
    logits[OPTIMISTIC_POLICY] = [-3.0, 5.0];
    let logits: Tensor<4> = Tensor::<2>::from_floats(logits, &device).reshape([1, POLICY_OUTPUTS, 1, 2]);

    let blend = |optimism: f32| {
      let optimism = Tensor::<3>::from_floats([[[optimism]]], &device);
      interpolate_policy(logits.clone(), optimism)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
    };

    assert_eq!(blend(0.0), vec![1.0, -1.0]);
    assert_eq!(blend(1.0), vec![-3.0, 5.0]);
    // A quarter of the way there, in logits.
    assert_eq!(blend(0.25), vec![0.0, 0.5]);

    // Each row is blended by its own weight, not by the batch's.
    let logits = logits.clone().repeat_dim(0, 2);
    let optimism = Tensor::<3>::from_floats([[[0.0]], [[1.0]]], &device);
    let blended = interpolate_policy(logits, optimism)
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    assert_eq!(blended, vec![1.0, -1.0, -3.0, 5.0]);
  }

  #[test]
  #[cfg(feature = "flex")]
  fn forward() {
    let config = ModelConfig::default();
    let model = Model::new(&Device::flex(), &config);
    let (policy_logits, predictions, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &Device::flex()),
      Tensor::ones([1, GLOBAL_FEATURES], &Device::flex()),
    );
    let values = predictions.value;
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    assert!(
      policies
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|p| (0.0..=1.0).contains(p))
    );
    assert!(
      policies
        .iter_dim(0)
        .all(|p| (p.sum().into_scalar::<f32>() - 1.0) < 0.001)
    );
    assert!(
      values
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| (-1.0..=1.0).contains(v))
    );
  }

  /// The probability of a draw inference reads is the score belief's bin of a
  /// lead of 0, as training sees it, under each row's own komi.
  #[test]
  #[cfg(feature = "flex")]
  fn inference_draw_is_the_belief_of_a_level_lead() {
    let device = Device::flex();
    let mut model = Model::new(&device, &ModelConfig::default());
    model.initialize(&device);
    let inputs = Tensor::<4>::random([3, CHANNELS, 5, 6], burn::tensor::Distribution::Default, &device);
    let global = Tensor::<2>::from_floats(
      [
        [0.1, 1.0, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.3],
        [-0.1, 1.0, -1.0, 0.1, 0.1],
      ],
      &device,
    );
    let (_, predictions, _) = model.forward(inputs.clone(), global.clone());
    let center = SCORE_ONE_HOT_SIZE / 2;
    let expected = predictions
      .score
      .slice(s![.., center..center + 1])
      .exp()
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    let (.., draw) = model.forward_inference(inputs, global);
    let draw = draw.into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(draw.len(), 3);
    for (draw, expected) in draw.iter().zip(&expected) {
      assert!((0.0..=1.0).contains(draw));
      assert!((draw - expected).abs() < 1e-6, "{draw} {expected}");
    }
  }

  // Verifies the core Fixup invariant: after `initialize`, every residual branch ends in a
  // zero-initialized conv so each block starts as the identity, and the model still produces a
  // valid, finite policy distribution.
  #[test]
  #[cfg(feature = "flex")]
  fn initialize_zeroes_residual_branches() {
    let device = Device::flex();
    let mut model = Model::new(&device, &ModelConfig::default());
    model.initialize(&device);

    let assert_zero = |convgpool: &ConvOrGpool| match convgpool {
      ConvOrGpool::Conv(conv) => {
        let abs_sum = conv.weight.val().abs().sum().into_scalar::<f32>();
        assert_eq!(abs_sum, 0.0, "residual branch output conv must be zero-initialized");
      }
      ConvOrGpool::Gpool(_) => panic!("residual branch output should be a plain conv"),
    };

    for residual in &model.residuals {
      let Block::Nbt(residual) = residual;
      assert_zero(&residual.normactconvq.convgpool);
      for inner in &residual.inner {
        assert_zero(&inner.normactconv2.convgpool);
      }
    }

    let (policy_logits, predictions, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &device),
      Tensor::ones([1, GLOBAL_FEATURES], &device),
    );
    let values = predictions.value;
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    assert!(
      policies
        .iter_dim(0)
        .all(|p| (p.sum().into_scalar::<f32>() - 1.0).abs() < 0.001)
    );
    assert!(
      values
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
    );
  }

  /// The paths of every float parameter, in traversal order.
  #[cfg(feature = "flex")]
  #[derive(Default)]
  struct PathCollector {
    path: Vec<String>,
    paths: Vec<String>,
  }

  #[cfg(feature = "flex")]
  impl burn::module::ModuleVisitor for PathCollector {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
      self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
      self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, _param: &burn::module::Param<Tensor<D>>) {
      self.paths.push(self.path.join("."));
    }
  }

  // A block's parameters are under its index and its kind, and a record loads
  // back into a model of the same config.
  #[test]
  #[cfg(feature = "flex")]
  fn records_have_the_paths_of_the_blocks() {
    use burn::module::Module;
    let device = Device::flex();
    let model = Model::new(&device, &ModelConfig::default());
    let mut paths = PathCollector::default();
    model.visit(&mut paths);
    for path in [
      "conv_spatial.weight",
      "residuals.0.Nbt.normactconvp.norm.beta",
      "residuals.0.Nbt.inner.1.normactconv2.convgpool.Conv.weight",
      "residuals.1.Nbt.inner.0.normactconv1.convgpool.Gpool.conv1g.weight",
      "residuals.4.Nbt.normactconvq.norm.gamma",
      "value_head.linear2.weight",
    ] {
      assert!(paths.paths.iter().any(|known| known == path), "{path}");
    }

    let inputs = Tensor::ones([1, CHANNELS, 4, 8], &device);
    let global = Tensor::ones([1, GLOBAL_FEATURES], &device);
    let loaded = Model::new(&device, &ModelConfig::default())
      .try_load_record(model.clone().into_record())
      .unwrap();
    loaded
      .forward_inference(inputs.clone(), global.clone())
      .0
      .into_data()
      .assert_eq(&model.forward_inference(inputs, global).0.into_data(), true);
  }

  // The board size feature goes from -1 on a 16x16 board to 1 on a 40x40
  // one, and grows like the logarithm of the area.
  #[test]
  #[cfg(feature = "flex")]
  fn board_size_feature_values() {
    let device = Device::flex();
    let areas = Tensor::<1>::from_floats([256.0, 1600.0, 640.0, 1248.0], &device).reshape([4, 1, 1, 1]);
    let values = super::board_size_feature(areas)
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    for (value, expected) in values.into_iter().zip([-1.0, 1.0, 0.0, 0.7288]) {
      assert!((value - expected).abs() < 1e-3, "{value} vs {expected}");
    }
  }

  // The score belief's second input for every bin is the target of a game
  // that ends with the board level, whatever the komi: the bin of a draw under
  // a whole komi, and the two bins around a half one.
  #[test]
  #[cfg(feature = "flex")]
  fn score_belief_level_input_is_the_level_target() {
    let device = Device::flex();
    let komis_x_2 = [-1, 0, 1, 4];
    let komi = Tensor::<1>::from_floats(komis_x_2.map(|komi_x_2| komi_x_2 as f32 / 2.0), &device).reshape([4, 1]);
    let offsets = super::score_belief_offsets(komi);
    assert_eq!(offsets.dims(), [4, SCORE_ONE_HOT_SIZE, 2]);
    let level = offsets
      .clone()
      .slice(s![.., .., 1..2])
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    for (i, komi_x_2) in komis_x_2.into_iter().enumerate() {
      let expected = score_one_hot::<f32>(0, komi_x_2);
      assert_eq!(
        &level[i * SCORE_ONE_HOT_SIZE..(i + 1) * SCORE_ONE_HOT_SIZE],
        expected.as_slice().unwrap()
      );
    }
    let scaled = offsets
      .slice(s![0..1, .., 0..1])
      .into_data()
      .try_to_vec::<f32>()
      .unwrap();
    let center = SCORE_ONE_HOT_SIZE / 2;
    assert_eq!(scaled[center], 0.0);
    assert!((scaled[center + 1] - super::SCORE_BELIEF_OFFSET_SCALE as f32).abs() < 1e-6);
  }

  // The global pooling gates start at one: a network with them computes what
  // the same weights do without them, which is what lets a trained network
  // gain them by adding their zero weights to its record.
  #[test]
  #[cfg(feature = "flex")]
  fn zero_gates_leave_the_network_as_it_is() {
    use burn::module::Module;
    let device = Device::flex();
    let without_gates = |mut config: ModelConfig| {
      for block in &mut config.blocks {
        if let BlockConfig::Nbt(NbtConfig { gpool: Some(gpool), .. }) = block {
          gpool.gate = false;
        }
      }
      config
    };
    let config = ModelConfig::default();
    let mut gated = Model::new(&device, &config);
    gated.initialize(&device);
    let plain = Model::new(&device, &without_gates(config))
      .try_load_record(gated.clone().into_record().allow_unused(true))
      .unwrap();
    let inputs = Tensor::<4>::random([2, CHANNELS, 5, 6], burn::tensor::Distribution::Default, &device);
    let global = Tensor::ones([2, GLOBAL_FEATURES], &device);
    let (policy, value, ..) = gated.forward_inference(inputs.clone(), global.clone());
    let (plain_policy, plain_value, ..) = plain.forward_inference(inputs, global);
    policy.into_data().assert_eq(&plain_policy.into_data(), true);
    value.into_data().assert_eq(&plain_value.into_data(), true);
  }

  // A trained network gains pooling over chains as the identity: the grown
  // network predicts what the one it grew from does, and its pooling is in
  // place to train, its output convolution zero but its input one not.
  #[test]
  #[cfg(feature = "flex")]
  fn adding_chains_leaves_the_network_as_it_is() {
    let device = Device::flex();
    let config = chains_config();
    let mut base = Model::new(&device, &config.without_chains());
    base.initialize(&device);
    let grown = Model::adding_chains(&device, &config, base.clone()).unwrap();
    let pools = grown
      .residuals
      .iter()
      .filter_map(|block| {
        let Block::Nbt(block) = block;
        block.chainpool.as_ref()
      })
      .count();
    assert_eq!(pools, 2);
    let mut paths = PathCollector::default();
    burn::module::Module::visit(&grown, &mut paths);
    assert!(
      paths
        .paths
        .iter()
        .any(|path| path == "residuals.1.Nbt.chainpool.conv1.weight")
    );

    let (height, width) = (4, 5);
    let values = (0..CHANNELS * height * width)
      .map(|i| {
        let (channel, y, x) = (i / (height * width), i / width % height, i % width);
        if channel == 0 {
          1.0
        } else if channel >= FEATURE_CHANNELS {
          chain_plane(channel, y, x, width)
        } else {
          ((channel * 5 + y * 7 + x * 3) % 13) as f32 / 13.0
        }
      })
      .collect::<Vec<_>>();
    let inputs = Tensor::<4>::from_data(TensorData::new(values, [1, CHANNELS, height, width]), &device);
    let global = Tensor::ones([1, GLOBAL_FEATURES], &device) * 0.5;
    let (policy, value, ..) = grown.forward_inference(inputs.clone(), global.clone());
    let (base_policy, base_value, ..) = base.forward_inference(inputs, global);
    policy.into_data().assert_eq(&base_policy.into_data(), true);
    value.into_data().assert_eq(&base_value.into_data(), true);
  }

  // A board predicts the same in a grid of its own size as padded into a
  // bigger one, which is how boards of several sizes share a batch: nothing
  // off the board reaches the cells on it, nor the board's value.
  #[test]
  #[cfg(feature = "flex")]
  fn padding_leaves_the_board_as_it_is() {
    for config in [ModelConfig::default(), chains_config()] {
      padding_leaves_the_board_of(&config);
    }
  }

  #[cfg(feature = "flex")]
  fn padding_leaves_the_board_of(config: &ModelConfig) {
    let device = Device::flex();
    let model = Model::new(&device, config);
    let (height, width) = (4, 5);
    let board = |padded_height: usize, padded_width: usize| {
      let values = (0..CHANNELS * padded_height * padded_width)
        .map(|i| {
          let (channel, cell) = (i / (padded_height * padded_width), i % (padded_height * padded_width));
          let (y, x) = (cell / padded_width, cell % padded_width);
          if y >= height || x >= width {
            0.0
          } else if channel == 0 {
            1.0
          } else if channel >= FEATURE_CHANNELS {
            chain_plane(channel, y, x, width)
          } else {
            ((channel * 7 + y * 3 + x * 5) % 11) as f32 / 11.0
          }
        })
        .collect::<Vec<_>>();
      Tensor::<4>::from_data(
        TensorData::new(values, [1, CHANNELS, padded_height, padded_width]),
        &device,
      )
    };
    let global = Tensor::<2>::ones([1, GLOBAL_FEATURES], &device) * 0.5;
    let (policy, value, ..) = model.forward_inference(board(height, width), global.clone());
    let (padded_policy, padded_value, ..) = model.forward_inference(board(height + 3, width + 2), global);
    padded_policy
      .slice(s![.., .., 0..height, 0..width])
      .into_data()
      .assert_approx_eq::<f32>(&policy.into_data(), Default::default());
    padded_value
      .into_data()
      .assert_approx_eq::<f32>(&value.into_data(), Default::default());
  }

  // With factor 1 the average must become an exact copy of the model, and with
  // factor 0 it must stay unchanged - the two endpoints of the EMA lerp.
  #[test]
  #[cfg(feature = "flex")]
  fn ema_update_endpoints() {
    let config = ModelConfig::default();
    use super::ema_update;

    let device = Device::flex();
    let mut average = Model::new(&device, &config);
    average.initialize(&device);
    let mut model = Model::new(&device, &config);
    model.initialize(&device);

    let inputs = Tensor::ones([1, CHANNELS, 4, 8], &device);
    let global = Tensor::ones([1, GLOBAL_FEATURES], &device);

    let (model_policy, _, _) = model.forward(inputs.clone(), global.clone());
    let (average_policy, _, _) = average.forward(inputs.clone(), global.clone());

    let average = ema_update(average, &model, 0.0, &device);
    let (unchanged_policy, _, _) = average.forward(inputs.clone(), global.clone());
    unchanged_policy
      .into_data()
      .assert_eq(&average_policy.into_data(), true);

    // `a + (c - a) * 1` is only c up to rounding, so compare approximately.
    let average = ema_update(average, &model, 1.0, &device);
    let (copied_policy, _, _) = average.forward(inputs, global);
    copied_policy
      .into_data()
      .assert_approx_eq::<f32>(&model_policy.into_data(), burn::tensor::Tolerance::default());
  }

  macro_rules! predict_test {
    ($name:ident, $device:expr, $config:expr) => {
      #[test]
      fn $name() {
        let device = $device;
        let model = Model::new(&device, &$config);
        let predictor = Predictor { model, device };
        let (policies, values) = futures::executor::block_on(predictor.predict(
          Array4::from_elem((1, CHANNELS, 4, 8), 1.0),
          array![[0.2, 1.0, 0.0, 0.3, 0.1]],
          array![0.0],
        ))
        .unwrap();
        // Win and loss probabilities, the predicted short-term error as a
        // standard deviation, which the search turns into a playout weight,
        // the predicted score in points, and the probability of a draw.
        assert_eq!(values.dim(), (1, VALUE_COLUMNS));
        assert!((values[(0, 0)] + values[(0, 1)] - 1.0).abs() < 1e-4);
        assert!(values[(0, 2)] >= 0.0 && values[(0, 2)].is_finite());
        assert!(values[(0, 3)].is_finite());
        assert!((0.0..=1.0).contains(&values[(0, 4)]));

        // The same position three times, each row asking for its own optimism:
        // a batch is what the search actually submits, and every row of it has
        // to be blended by its own weight rather than the batch's first.
        let (batched, _) = futures::executor::block_on(predictor.predict(
          Array4::from_elem((3, CHANNELS, 4, 8), 1.0),
          array![
            [0.2, 1.0, 0.0, 0.3, 0.1],
            [0.2, 1.0, 0.0, 0.3, 0.1],
            [0.2, 1.0, 0.0, 0.3, 0.1]
          ],
          array![0.0, 1.0, 0.5],
        ))
        .unwrap();
        let plain = batched.index_axis(Axis(0), 0);
        let optimistic = batched.index_axis(Axis(0), 1);
        let half = batched.index_axis(Axis(0), 2);
        // Row 0 asked for no optimism, so it is the policy predicted above.
        assert!(
          (&plain - &policies.index_axis(Axis(0), 0))
            .iter()
            .all(|p| p.abs() < 1e-6)
        );
        // Row 1 is the optimistic head instead, which is a distribution of its
        // own and not the same one.
        assert!((optimistic.sum() - 1.0).abs() < 1e-4);
        assert!((&optimistic - &plain).iter().any(|p| p.abs() > 1e-6));
        // Row 2 is halfway between them in logit space, i.e. the geometric mean
        // of the two policies - so the ratio it keeps to that mean is the same
        // for every cell of the board, whatever the normalization works out to.
        let mut ratios = plain
          .iter()
          .zip(optimistic.iter())
          .zip(half.iter())
          .map(|((&plain, &optimistic), &half)| half / (plain * optimistic).sqrt());
        let first = ratios.next().unwrap();
        assert!(ratios.all(|ratio| (ratio - first).abs() < 1e-3 * first));
      }
    };
  }

  #[cfg(feature = "flex")]
  predict_test!(predict_flex, Device::flex(), ModelConfig::default());
  #[cfg(feature = "flex")]
  predict_test!(predict_chains_flex, Device::flex(), chains_config());

  /// Casts every float parameter, which is what loading a model on a device
  /// configured for half precision does to it.
  #[cfg(feature = "flex")]
  struct CastMapper(burn::tensor::FloatDType);

  #[cfg(feature = "flex")]
  impl burn::module::ModuleMapper for CastMapper {
    fn map_float<const D: usize>(&mut self, param: burn::module::Param<Tensor<D>>) -> burn::module::Param<Tensor<D>> {
      let (id, tensor, mapper) = param.consume();
      burn::module::Param::from_mapped_value(id, tensor.cast(self.0), mapper)
    }
  }

  // A model in half precision takes the same single precision inputs and gives
  // back the same single precision outputs, up to the rounding of its weights
  // and activations.
  #[test]
  #[cfg(feature = "flex")]
  fn half_precision_predicts_what_single_precision_does() {
    let config = ModelConfig::default();
    use burn::{module::Module, tensor::FloatDType};
    let device = Device::flex();
    let mut model = Model::new(&device, &config);
    model.initialize(&device);
    let half = model.clone().map(&mut CastMapper(FloatDType::F16));
    assert_eq!(half.input_dtype(), burn::tensor::DType::F16);
    let single = Predictor {
      model,
      device: device.clone(),
    };
    let half = Predictor { model: half, device };
    let inputs = Array4::from_shape_fn((2, CHANNELS, 4, 8), |(b, c, y, x)| {
      if c == 0 || (b + c + y * 3 + x) % 4 == 0 {
        1.0
      } else {
        0.0
      }
    });
    let global = Array2::from_elem((2, GLOBAL_FEATURES), 0.2);
    let optimism = array![0.0, 0.5];
    let (single_policies, single_values) =
      futures::executor::block_on(single.predict(inputs.clone(), global.clone(), optimism.clone())).unwrap();
    let (half_policies, half_values) = futures::executor::block_on(half.predict(inputs, global, optimism)).unwrap();
    assert!(
      (&half_policies - &single_policies).iter().all(|d| d.abs() < 1e-2),
      "{half_policies} vs {single_policies}"
    );
    // Half precision keeps about three significant digits of the activations,
    // so an output is off by a fraction of the unit its layer's raw output is
    // stated in, however small the output itself: the score, that raw output
    // times TD_SCORE_SCALE points, is off by a few hundredths of a point even
    // when it is near zero. Outputs are compared relative to their size, but
    // never more tightly than that unit allows.
    let units = [1.0, 1.0, 1.0, TD_SCORE_SCALE as f32];
    assert!(
      half_values
        .rows()
        .into_iter()
        .zip(single_values.rows())
        .all(|(half, single)| {
          half
            .iter()
            .zip(single)
            .zip(units)
            .all(|((half, single), unit)| (half - single).abs() < 1e-2 * single.abs().max(unit))
        }),
      "{half_values} vs {single_values}"
    );
  }
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  predict_test!(
    predict_wgpu,
    Device::wgpu(DeviceKind::DefaultDevice),
    ModelConfig::default()
  );

  macro_rules! train_test {
    ($name:ident, $device:expr, $config:expr) => {
      train_test!($name, $device, $config, SgdConfig::new().init());
    };
    ($name:ident, $device:expr, $config:expr, $optimizer:expr) => {
      #[test]
      fn $name() {
        let device = $device.autodiff();
        let model = Model::new(&device, &$config);
        let predictor = Predictor { model, device };
        let optimizer = $optimizer;
        // Clipped, so that the whole step - including scaling the gradients on
        // the device - is what gets exercised here.
        let learner = Learner {
          predictor,
          optimizer,
          gradient_clipping: Some(1.0),
          compute_dtype: None,
        };

        let inputs = Array4::from_elem((1, CHANNELS, 4, 8), 1.0);
        let global = Array2::from_elem((1, GLOBAL_FEATURES), 0.2);
        let policies = Array3::from_elem((1, 4, 8), 0.5);
        let opponent_policies = Array3::from_elem((1, 4, 8), 0.7);
        let values = array![[1.0, 0.0]];
        let td_values = Array3::from_elem((1, TD_VALUES, 2), 0.5);
        let td_scores = Array2::from_elem((1, TD_VALUES), 3.0);
        let mut scores = Array2::from_elem((1, SCORE_ONE_HOT_SIZE), 0.0);
        scores[(0, 0)] = 1.0;
        let captured = Array4::from_elem((1, 2, 4, 8), 1.0);
        let future = Array4::from_elem((1, FUTURE_POSITIONS, 4, 8), 0.5);
        // One explored move with a recorded q value, q score and search
        // weight; every other cell keeps zero weight and stays out of the q
        // losses.
        let mut q_values = Array4::from_elem((1, 3, 4, 8), 0.0);
        q_values[(0, 0, 1, 2)] = 0.5;
        q_values[(0, 1, 1, 2)] = 2.5;
        q_values[(0, 2, 1, 2)] = 4.0;

        let (out_policies_1, out_values_1) =
          futures::executor::block_on(learner.predict(inputs.clone(), global.clone(), array![0.0])).unwrap();
        let learner = learner
          .train(
            inputs.clone(),
            global.clone(),
            policies,
            opponent_policies,
            values,
            td_values,
            td_scores,
            scores,
            captured,
            future,
            q_values,
            array![1.0],
            0.01,
          )
          .unwrap();
        let (out_policies_2, out_values_2) =
          futures::executor::block_on(learner.predict(inputs, global, array![0.0])).unwrap();

        assert!((out_policies_1 - out_policies_2).iter().all(|v| v.abs() > 0.0));
        assert!((out_values_1 - out_values_2).iter().all(|v| v.abs() > 0.0));
      }
    };
  }

  // The loss terms validation reads are those of training, named as the
  // training log names them: computing them leaves the model as it is, and a
  // training step on the batch brings them down.
  #[test]
  #[cfg(feature = "flex")]
  fn losses_are_the_training_terms() {
    let config = ModelConfig::default();
    let device = Device::flex().autodiff();
    let model = Model::new(&device, &config);
    let predictor = Predictor { model, device };
    let mut learner = Learner {
      predictor,
      optimizer: SgdConfig::new().init(),
      gradient_clipping: None,
      compute_dtype: None,
    };

    let mut q_values = Array4::from_elem((1, 3, 4, 8), 0.0);
    q_values[(0, 0, 1, 2)] = 0.5;
    q_values[(0, 2, 1, 2)] = 4.0;
    let mut scores = Array2::from_elem((1, SCORE_ONE_HOT_SIZE), 0.0);
    scores[(0, SCORE_ONE_HOT_SIZE / 2)] = 1.0;
    let batch = (
      Array4::from_elem((1, CHANNELS, 4, 8), 1.0),
      array![[0.2, 1.0, 0.0, 0.3, 0.1]],
      Array3::from_elem((1, 4, 8), 1.0 / 32.0),
      Array3::from_elem((1, 4, 8), 1.0 / 32.0),
      array![[1.0, 0.0]],
      Array3::from_elem((1, TD_VALUES, 2), 0.5),
      Array2::from_elem((1, TD_VALUES), 3.0),
      scores,
      Array4::from_elem((1, 2, 4, 8), 0.0),
      Array4::from_elem((1, FUTURE_POSITIONS, 4, 8), -1.0),
      q_values,
      array![1.0],
    );
    let losses = |learner: &Learner| {
      let b = batch.clone();
      learner
        .losses(b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7, b.8, b.9, b.10, b.11)
        .values()
    };

    let before = losses(&learner);
    assert_eq!(before, losses(&learner));
    // The copy for validation computes the same losses, only without autodiff.
    let valid = learner.predictor.valid();
    assert!(!valid.device.is_autodiff());
    let b = batch.clone();
    let valid_losses = valid
      .losses(None, b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7, b.8, b.9, b.10, b.11)
      .values();
    for (&(name, training), &(_, validation)) in before.iter().zip(&valid_losses) {
      assert!(
        (training - validation).abs() <= 1e-5 * training.abs().max(1.0),
        "{name}"
      );
    }
    let names = before.iter().map(|&(name, _)| name).collect::<Vec<_>>();
    assert_eq!(names.len(), 17);
    assert_eq!(names.iter().collect::<std::collections::HashSet<_>>().len(), 17);
    assert_eq!(&names[..2], &["value", "td value"]);
    assert!(before.iter().all(|&(_, loss)| loss.is_finite()));

    for _ in 0..4 {
      let b = batch.clone();
      learner = learner
        .train(b.0, b.1, b.2, b.3, b.4, b.5, b.6, b.7, b.8, b.9, b.10, b.11, 0.01)
        .unwrap();
    }
    let total = |losses: &[(&str, f32)]| losses.iter().map(|&(_, loss)| loss).sum::<f32>();
    assert!(total(&losses(&learner)) < total(&before));
  }

  // A side position has no game played out from it, so the targets of the
  // game's end - the final score, the TD scores, the captured cells, the
  // future positions and the policy of the won games - are left out of its losses, whatever they hold,
  // while the targets of its own search still count.
  #[cfg(feature = "flex")]
  #[test]
  fn side_positions_have_no_outcome_losses() {
    let device = Device::flex().autodiff();
    let model = Model::new(&device, &ModelConfig::default());
    let predictor = Predictor { model, device };
    let mut scores = Array2::from_elem((1, SCORE_ONE_HOT_SIZE), 0.0);
    scores[(0, SCORE_ONE_HOT_SIZE / 2 + 5)] = 1.0;
    let losses = |outcome_weight: f32| {
      predictor
        .losses(
          None,
          Array4::from_elem((1, CHANNELS, 4, 8), 1.0),
          array![[0.2, 1.0, 0.0, 0.3, 0.1]],
          Array3::from_elem((1, 4, 8), 1.0 / 32.0),
          Array3::from_elem((1, 4, 8), 0.0),
          array![[1.0, 0.0]],
          Array3::from_elem((1, TD_VALUES, 2), 0.5),
          Array2::from_elem((1, TD_VALUES), 3.0),
          scores.clone(),
          Array4::from_elem((1, 2, 4, 8), 1.0),
          Array4::from_elem((1, FUTURE_POSITIONS, 4, 8), 1.0),
          Array4::from_elem((1, 3, 4, 8), 0.0),
          array![outcome_weight],
        )
        .values()
    };
    let outcome = [
      "td score",
      "score error",
      "long optimistic policy",
      "pdf",
      "cdf",
      "captured",
      "future position",
    ];
    let played_out = losses(1.0);
    let side = losses(0.0);
    for (&(name, played_out), &(_, side)) in played_out.iter().zip(&side) {
      if outcome.contains(&name) {
        assert!(played_out > 0.0, "{name}");
        assert_eq!(side, 0.0, "{name}");
      } else if name != "optimistic policy" {
        assert_eq!(played_out, side, "{name}");
      }
    }
  }

  // Convolutions in brain floating point give the losses and the gradient that
  // single precision ones do, up to rounding - and do round, so they really run
  // in the other type - while the weights stay in single precision.
  #[test]
  #[cfg(feature = "flex")]
  fn mixed_precision_trains_like_single_precision() {
    let config = ModelConfig::default();
    use burn::{
      module::Module,
      tensor::{DType, FloatDType},
    };
    let device = Device::flex().autodiff();
    let model = Model::new(&device, &config);
    let learner = |compute_dtype| Learner {
      predictor: Predictor {
        model: model.clone(),
        device: device.clone(),
      },
      optimizer: SgdConfig::new().init(),
      gradient_clipping: None,
      compute_dtype,
    };
    let single = learner(None);
    let mixed = learner(Some(FloatDType::BF16));

    let inputs = Array4::from_shape_fn((2, CHANNELS, 4, 8), |(b, c, y, x)| {
      if c == 0 || (b + c + y * 3 + x) % 4 == 0 {
        1.0
      } else {
        0.0
      }
    });
    let mut scores = Array2::from_elem((2, SCORE_ONE_HOT_SIZE), 0.0);
    scores[(0, SCORE_ONE_HOT_SIZE / 2)] = 1.0;
    scores[(1, SCORE_ONE_HOT_SIZE / 2 + 3)] = 1.0;
    let losses = |learner: &Learner| {
      learner.losses(
        inputs.clone(),
        array![[0.2, 1.0, 0.0, 0.3, 0.1], [-0.2, 0.0, 0.5, 0.0, 0.4]],
        Array3::from_elem((2, 4, 8), 1.0 / 32.0),
        Array3::from_elem((2, 4, 8), 1.0 / 32.0),
        array![[1.0, 0.0], [0.0, 1.0]],
        Array3::from_elem((2, TD_VALUES, 2), 0.5),
        Array2::from_elem((2, TD_VALUES), 3.0),
        scores.clone(),
        Array4::from_elem((2, 2, 4, 8), 0.0),
        Array4::from_shape_fn((2, FUTURE_POSITIONS, 4, 8), |(b, _, y, x)| {
          [-1.0, 0.0, 1.0][(b + y + x) % 3]
        }),
        Array4::from_elem((2, 3, 4, 8), 0.0),
        array![1.0, 1.0],
      )
    };
    let gradient = |learner: &Learner| {
      let super::Losses(losses) = losses(learner);
      let loss = losses.into_iter().map(|(_, loss)| loss).reduce(|a, b| a + b).unwrap();
      let model = &learner.predictor.model;
      let grads = GradientsParams::from_grads(loss.backward(), model);
      let mut collector = GradientCollector {
        grads: &grads,
        values: Vec::new(),
      };
      model.visit(&mut collector);
      collector.values
    };

    let single_losses = losses(&single).values();
    let mixed_losses = losses(&mixed).values();
    assert_ne!(single_losses, mixed_losses);
    for (&(name, single), &(_, mixed)) in single_losses.iter().zip(&mixed_losses) {
      assert!(
        (single - mixed).abs() <= 2e-2 * single.abs().max(1e-2),
        "{name}: {single} vs {mixed}"
      );
    }

    let single_gradient = gradient(&single);
    let mixed_gradient = gradient(&mixed);
    // Every weight gets its gradient through the casts, the convolutions' too.
    assert_eq!(single_gradient.len(), mixed_gradient.len());
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a * b) as f64).sum::<f64>();
    let cosine = dot(&single_gradient, &mixed_gradient)
      / (dot(&single_gradient, &single_gradient) * dot(&mixed_gradient, &mixed_gradient)).sqrt();
    assert!(cosine > 0.999, "{cosine}");
    let norms = dot(&mixed_gradient, &mixed_gradient).sqrt() / dot(&single_gradient, &single_gradient).sqrt();
    assert!((norms - 1.0).abs() < 1e-2, "{norms}");

    // The learner keeps its weights in single precision; only the copy the
    // forward pass runs on has the weights of its convolutions and projections
    // - and nothing else - cast.
    let dtypes = |model: &Model| {
      let mut collector = DTypeCollector::default();
      model.visit(&mut collector);
      collector.dtypes
    };
    assert!(
      dtypes(&mixed.predictor.model)
        .iter()
        .all(|&(_, dtype)| dtype == DType::F32)
    );
    let cast = dtypes(&model.with_compute_dtype(FloatDType::BF16));
    for (container, dtype) in &cast {
      let computes = matches!(
        container.as_str(),
        "Struct:Conv2d" | "Struct:Projection" | "Struct:DepthwiseConv"
      );
      assert_eq!(*dtype, if computes { DType::BF16 } else { DType::F32 }, "{container}");
    }
    assert!(cast.iter().any(|(container, _)| container == "Struct:Conv2d"));
    assert!(cast.iter().any(|(container, _)| container == "Struct:Linear"));
  }

  /// The float type of every float parameter, with the type of the module
  /// holding it.
  #[cfg(feature = "flex")]
  #[derive(Default)]
  struct DTypeCollector {
    containers: Vec<String>,
    dtypes: Vec<(String, burn::tensor::DType)>,
  }

  #[cfg(feature = "flex")]
  impl burn::module::ModuleVisitor for DTypeCollector {
    fn enter_module(&mut self, _name: &str, container_type: &str) {
      self.containers.push(container_type.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
      self.containers.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
      let container = self.containers.last().cloned().unwrap_or_default();
      self.dtypes.push((container, param.val().dtype()));
    }
  }

  /// Every float parameter's gradient, flattened into one vector in traversal
  /// order.
  #[cfg(feature = "flex")]
  struct GradientCollector<'a> {
    grads: &'a GradientsParams,
    values: Vec<f32>,
  }

  #[cfg(feature = "flex")]
  impl burn::module::ModuleVisitor for GradientCollector<'_> {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
      if let Some(grad) = self.grads.get::<D>(param.id) {
        self
          .values
          .extend(grad.into_data().convert::<f32>().try_into_vec::<f32>().unwrap());
      }
    }
  }

  #[cfg(feature = "flex")]
  train_test!(train_flex, Device::flex(), ModelConfig::default());
  #[cfg(feature = "flex")]
  train_test!(train_chains_flex, Device::flex(), chains_config());
  // Every optimizer, with its groups, steps the network.
  #[cfg(feature = "flex")]
  train_test!(
    train_adamw_flex,
    Device::flex(),
    ModelConfig::default(),
    OptimizerConfig::new(OptimizerKind::AdamW, 1).init()
  );
  #[cfg(feature = "flex")]
  train_test!(
    train_muon_flex,
    Device::flex(),
    ModelConfig::default(),
    OptimizerConfig::new(OptimizerKind::Muon, 1).init()
  );
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  train_test!(
    train_wgpu,
    Device::wgpu(DeviceKind::DefaultDevice),
    ModelConfig::default()
  );
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  train_test!(
    train_chains_wgpu,
    Device::wgpu(DeviceKind::DefaultDevice),
    chains_config()
  );
}
