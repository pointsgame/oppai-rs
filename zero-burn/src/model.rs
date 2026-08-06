use burn::{
  module::{Initializer, Module, ModuleVisitor, Param},
  nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
    loss::HuberLossConfig,
  },
  optim::{GradientsParams, Optimizer},
  tensor::{
    DataError, Tensor, TensorData,
    activation::{log_softmax, mish, sigmoid, softmax, softplus},
    backend::{AutodiffBackend, Backend, ExecutionError},
    module::{conv2d, linear},
    ops::{ConvOptions, FloatElem},
    s,
  },
};
use derive_more::From;
use ndarray::{Array, Array1, Array2, Array3, Array4, Dimension, ShapeError};
use num_traits::Float;
use oppai_zero::{
  examples::TD_VALUES,
  field_features::{CHANNELS, GLOBAL_FEATURES, SCORE_ONE_HOT_SIZE},
  model::{Model as OppaiModel, TrainableModel as OppaiTrainableModel},
};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader, path::Path};
use thiserror::Error;

/// Model architecture hyperparameters. A model must be created with the same
/// config it was trained with for the weights to load.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
  pub inner_channels: usize,
  pub residual_blocks: usize,
  pub residual_size: usize,
  pub gpool_every: usize,
  pub gpool_channels: usize,
  pub v1_channels: usize,
  pub p1_channels: usize,
  pub g1_channels: usize,
  pub v2_size: usize,
  pub sbv2_size: usize,
  pub num_scorebeliefs: usize,
}

impl Default for ModelConfig {
  fn default() -> Self {
    Self {
      inner_channels: 192,
      residual_blocks: 5,
      residual_size: 2,
      gpool_every: 2,
      gpool_channels: 32,
      v1_channels: 32,
      p1_channels: 32,
      g1_channels: 32,
      v2_size: 80,
      sbv2_size: 80,
      num_scorebeliefs: 6,
    }
  }
}

impl ModelConfig {
  pub fn residual_inner_channels(&self) -> usize {
    self.inner_channels / 2
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
fn mish_gain() -> f64 {
  2.210277_f64.sqrt()
}

/// Reinitialize a weight tensor:
/// sample from `N(0, (scale * gain / sqrt(fan_in))^2)`, or fill with zeros when `scale == 0`.
/// Zero scale is how Fixup makes a residual branch start as the identity function.
fn init_weight<B: Backend, const D: usize>(
  shape: [usize; D],
  fan_in: usize,
  scale: f64,
  gain: f64,
  device: &B::Device,
) -> Param<Tensor<B, D>> {
  if scale <= 0.0 {
    Initializer::Zeros.init(shape, device)
  } else {
    let std = scale * gain / (fan_in as f64).sqrt();
    Initializer::Normal { mean: 0.0, std }.init(shape, device)
  }
}

/// Reinitialize a convolution's weights. `fan_in = in_channels * kernel_h * kernel_w`.
fn init_conv<B: Backend>(conv: &mut Conv2d<B>, scale: f64, gain: f64, device: &B::Device) {
  let [out_c, in_c, kh, kw] = conv.weight.val().dims();
  conv.weight = init_weight([out_c, in_c, kh, kw], in_c * kh * kw, scale, gain, device);
}

/// Reinitialize a linear layer's weights (and bias, if present). The burn weight layout is
/// `[d_input, d_output]`, so `fan_in = d_input`.
fn init_linear<B: Backend>(
  linear: &mut Linear<B>,
  weight_scale: f64,
  weight_gain: f64,
  bias_scale: f64,
  bias_gain: f64,
  device: &B::Device,
) {
  let [d_in, d_out] = linear.weight.val().dims();
  linear.weight = init_weight([d_in, d_out], d_in, weight_scale, weight_gain, device);
  if linear.bias.is_some() {
    linear.bias = Some(init_weight([d_out], d_in, bias_scale, bias_gain, device));
  }
}

#[derive(Module, Debug)]
pub struct NormMask<B: Backend> {
  beta: Param<Tensor<B, 4>>,
  gamma: Option<Param<Tensor<B, 4>>>,
}

impl<B: Backend> NormMask<B> {
  pub fn new(device: &B::Device, channels: usize, gamma: bool) -> Self {
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

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>) -> Tensor<B, 4> {
    match self.gamma {
      Some(ref gamma) => (inputs * (gamma.val() + 1.0) + self.beta.val()) * mask,
      None => (inputs + self.beta.val()) * mask,
    }
  }
}

#[derive(Module, Debug)]
pub struct ConvAndGPool<B: Backend> {
  conv1r: Conv2d<B>,
  conv1g: Conv2d<B>,
  normg: NormMask<B>,
  linearg: Linear<B>,
}

impl<B: Backend> ConvAndGPool<B> {
  pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
    let residual_inner_channels = config.residual_inner_channels();
    Self {
      conv1r: Conv2dConfig::new(
        [residual_inner_channels, residual_inner_channels - config.gpool_channels],
        [3, 3],
      )
      .with_padding(PaddingConfig2d::Same)
      .with_bias(false)
      .init(device),
      conv1g: Conv2dConfig::new([residual_inner_channels, config.gpool_channels], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      normg: NormMask::new(device, config.gpool_channels, false),
      linearg: LinearConfig::new(
        3 * config.gpool_channels,
        residual_inner_channels - config.gpool_channels,
      )
      .with_bias(false)
      .init(device),
    }
  }

  /// Splits the input variance between the regular (`r`) and global-pooling (`g`) branches
  /// so they add back up to roughly `scale`.
  fn initialize(&mut self, scale: f64, device: &B::Device) {
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
  }

  fn gpool(inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    // min size: 16, max size: 40, avg: (40 + 16) / 2 = 28
    let mask_sum_hw_sqrt_offset = mask_sum_hw.clone().sqrt() - 28.0;

    let layer_mean = inputs.clone().sum_dim(2).sum_dim(3) / mask_sum_hw;
    // Activation functions is always greater than -1.0, and map 0 -> 0
    let layer_max = (inputs + (mask - 1.0)).max_dim(2).max_dim(3);

    let out_pool1 = layer_mean.clone();
    let out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0);
    let out_pool3 = layer_max;

    Tensor::cat(vec![out_pool1, out_pool2, out_pool3], 1)
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let outr = self.conv1r.forward(inputs.clone());
    let outg = self.conv1g.forward(inputs);
    let outg = self.normg.forward(outg, mask.clone());
    let outg = mish(outg);
    let outg = Self::gpool(outg, mask, mask_sum_hw);
    let outg = self
      .linearg
      .forward(outg.squeeze_dims::<2>(&[2, 3]))
      .unsqueeze_dims(&[-1, -1]);
    outr + outg
  }
}

#[allow(clippy::large_enum_variant)]
#[derive(Module, Debug)]
pub enum ConvOrGpool<B: Backend> {
  Conv(Conv2d<B>),
  Gpool(ConvAndGPool<B>),
}

impl<B: Backend> ConvOrGpool<B> {
  pub fn new(
    device: &B::Device,
    config: &ModelConfig,
    gpool: bool,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    if gpool {
      Self::Gpool(ConvAndGPool::new(device, config))
    } else {
      Self::Conv(
        Conv2dConfig::new([in_channels, out_channels], kernel_size)
          .with_padding(PaddingConfig2d::Same)
          .with_bias(false)
          .init(device),
      )
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    match self {
      Self::Conv(conv) => conv.forward(inputs),
      Self::Gpool(gpool) => gpool.forward(inputs, mask, mask_sum_hw),
    }
  }
}

#[derive(Module, Debug)]
pub struct NormActConv<B: Backend> {
  norm: NormMask<B>,
  convgpool: ConvOrGpool<B>,
}

impl<B: Backend> NormActConv<B> {
  pub fn new(
    device: &B::Device,
    config: &ModelConfig,
    gamma: bool,
    gpool: bool,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    Self {
      norm: NormMask::new(device, in_channels, gamma),
      convgpool: ConvOrGpool::new(device, config, gpool, in_channels, out_channels, kernel_size),
    }
  }

  /// only the convolution is rescaled; the norm stays a learnable affine
  /// since fixup applies no fixed scale to it.
  fn initialize(&mut self, scale: f64, device: &B::Device) {
    match &mut self.convgpool {
      ConvOrGpool::Conv(conv) => init_conv(conv, scale, mish_gain(), device),
      ConvOrGpool::Gpool(gpool) => gpool.initialize(scale, device),
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let out = self.norm.forward(inputs, mask.clone());
    let out = mish(out);
    self.convgpool.forward(out, mask, mask_sum_hw)
  }
}

#[derive(Module, Debug)]
pub struct InnerResidualBlock<B: Backend> {
  normactconv1: NormActConv<B>,
  normactconv2: NormActConv<B>,
}

impl<B: Backend> InnerResidualBlock<B> {
  pub fn new(device: &B::Device, config: &ModelConfig, gpool: bool) -> Self {
    let residual_inner_channels = config.residual_inner_channels();
    Self {
      normactconv1: NormActConv::new(
        device,
        config,
        false,
        gpool,
        residual_inner_channels,
        residual_inner_channels,
        [3, 3],
      ),
      normactconv2: NormActConv::new(
        device,
        config,
        true,
        false,
        if gpool {
          residual_inner_channels - config.gpool_channels
        } else {
          residual_inner_channels
        },
        residual_inner_channels,
        [3, 3],
      ),
    }
  }

  /// Scale the first conv, and zero-initialize the second conv so the block starts
  /// as the identity and only gradually learns a residual.
  fn initialize(&mut self, fixup_scale: f64, device: &B::Device) {
    self.normactconv1.initialize(fixup_scale, device);
    self.normactconv2.initialize(0.0, device);
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let out = self
      .normactconv1
      .forward(inputs.clone(), mask.clone(), mask_sum_hw.clone());
    let out = self.normactconv2.forward(out, mask, mask_sum_hw);
    inputs + out
  }
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
  normactconvp: NormActConv<B>,
  inner: Vec<InnerResidualBlock<B>>,
  normactconvq: NormActConv<B>,
}

impl<B: Backend> ResidualBlock<B> {
  pub fn new(device: &B::Device, config: &ModelConfig, gpool: bool) -> Self {
    Self {
      normactconvp: NormActConv::new(
        device,
        config,
        false,
        false,
        config.inner_channels,
        config.residual_inner_channels(),
        [1, 1],
      ),
      inner: (0..config.residual_size)
        .map(|i| InnerResidualBlock::new(device, config, gpool && i == 0))
        .collect(),
      normactconvq: NormActConv::new(
        device,
        config,
        true,
        false,
        config.residual_inner_channels(),
        config.inner_channels,
        [1, 1],
      ),
    }
  }

  /// Each of the `1 + residual_size` stages gets the geometric share
  /// `fixup_scale^(1/(1+residual_size))` of the block's scale, and the final `1x1`
  /// conv is zero-initialized so the whole nested block starts as the identity.
  fn initialize(&mut self, fixup_scale: f64, device: &B::Device) {
    let inner_scale = fixup_scale.powf(1.0 / (1.0 + self.inner.len() as f64));
    self.normactconvp.initialize(inner_scale, device);
    for inner in &mut self.inner {
      inner.initialize(inner_scale, device);
    }
    self.normactconvq.initialize(0.0, device);
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let mut out = self
      .normactconvp
      .forward(inputs.clone(), mask.clone(), mask_sum_hw.clone());
    for inner in &self.inner {
      out = inner.forward(out, mask.clone(), mask_sum_hw.clone());
    }
    let out = self.normactconvq.forward(out, mask, mask_sum_hw);
    inputs + out
  }
}

/// Squared softplus with a gradient floor: the forward value is
/// `softplus(x / 2)^2`, but the backward pass is the gradient of the plain
/// `softplus(x)` floored at `grad_floor`, so the output can't get stuck with a
/// vanishing gradient when `x` goes very negative. The detach trick keeps the
/// forward value exact while routing gradients through the floored path,
/// whose derivative is `grad_floor + (1 - grad_floor) * sigmoid(x)`.
fn squared_softplus_with_gradient_floor<B: Backend>(x: Tensor<B, 2>, grad_floor: f64) -> Tensor<B, 2> {
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
pub struct ValuePredictions<B: Backend> {
  /// `(win, loss)` logit pairs: the main value trained towards the final result
  /// first, then one pair per TD horizon, shortest horizon last.
  pub value: Tensor<B, 2>,
  /// Predicted squared error of the shortest-horizon TD value.
  pub value_error: Tensor<B, 2>,
  /// Predicted score at each TD horizon, in points, shortest horizon last.
  pub td_score: Tensor<B, 2>,
  /// Predicted squared error of the shortest-horizon TD score, in points squared.
  pub score_error: Tensor<B, 2>,
  /// Log-distribution of the terminal score over the score bins.
  pub score: Tensor<B, 2>,
}

/// Value head output channels of `linear_valuehead`:
/// 0..2 - main value (win, loss) logits trained towards the final result,
/// then `(win, loss)` logit pairs for each of the `TD_VALUES` horizons,
/// shortest horizon last.
#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
  conv1: Conv2d<B>,
  bias1: NormMask<B>,
  linear2: Linear<B>,
  linear_valuehead: Linear<B>,
  /// Predicts the squared error of the shortest-horizon TD value, i.e. how
  /// uncertain the value estimate of this position is in the short term.
  linear_error: Linear<B>,
  /// Predicts the score at each TD horizon, in points.
  linear_td_score: Linear<B>,
  /// Predicts the squared error of the shortest-horizon TD score, i.e. how
  /// uncertain that score is - the score's counterpart of `linear_error`.
  linear_score_error: Linear<B>,
  // Score belief components
  linear_s2: Linear<B>,
  linear_s2off: Linear<B>,
  linear_s3: Linear<B>,
  linear_smix: Linear<B>,
  score_belief_offset_bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> ValueHead<B> {
  pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
    let offset_bias_data: Vec<f32> = (0..SCORE_ONE_HOT_SIZE as i32)
      .map(|i| 0.002 * ((i - (SCORE_ONE_HOT_SIZE - 1) as i32 / 2) as f32))
      .collect();
    let offset_bias_tensor: Tensor<B, 1> =
      Tensor::from_data(TensorData::new(offset_bias_data, [SCORE_ONE_HOT_SIZE]), device);

    Self {
      conv1: Conv2dConfig::new([config.inner_channels, config.v1_channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      bias1: NormMask::new(device, config.v1_channels, false),
      linear2: LinearConfig::new(3 * config.v1_channels, config.v2_size).init(device),
      linear_valuehead: LinearConfig::new(config.v2_size, 2 + 2 * TD_VALUES).init(device),
      linear_error: LinearConfig::new(config.v2_size, 1).init(device),
      linear_td_score: LinearConfig::new(config.v2_size, TD_VALUES).init(device),
      linear_score_error: LinearConfig::new(config.v2_size, 1).init(device),

      linear_s2: LinearConfig::new(3 * config.v1_channels, config.sbv2_size).init(device),
      linear_s2off: LinearConfig::new(1, config.sbv2_size).with_bias(false).init(device),
      linear_s3: LinearConfig::new(config.sbv2_size, config.num_scorebeliefs).init(device),
      linear_smix: LinearConfig::new(3 * config.v1_channels, config.num_scorebeliefs).init(device),
      score_belief_offset_bias: Param::from_tensor(offset_bias_tensor).no_grad(),
    }
  }

  /// Pre-pooling layers keep unit-ish variance while the output
  /// projections are scaled down so the head starts near-neutral.
  fn initialize(&mut self, device: &B::Device) {
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
    // `linear_s2off` has a single input feature, so KataGo borrows `linear_s2`'s fan-in to avoid a
    // huge std; it has no bias.
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

  fn gpool(inputs: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    // min size: 16, max size: 40, avg: (40 + 16) / 2 = 28
    let mask_sum_hw_sqrt_offset = mask_sum_hw.clone().sqrt() - 28.0;

    let layer_mean = inputs.clone().sum_dim(2).sum_dim(3) / mask_sum_hw;

    let out_pool1 = layer_mean.clone();
    let out_pool2 = layer_mean.clone() * (mask_sum_hw_sqrt_offset.clone() / 10.0);
    // (sum $ map (\x -> (x - 28) ** 2) [16..40]) / (40 - 16 + 1) / 100
    let out_pool3 = layer_mean * (mask_sum_hw_sqrt_offset.clone() * mask_sum_hw_sqrt_offset / 100.0 - 0.52);

    Tensor::cat(vec![out_pool1, out_pool2, out_pool3], 1)
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> ValuePredictions<B> {
    let outv1 = self.conv1.forward(inputs);
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

    // Score Belief Head

    // Term 1: Linear from pooled
    let s2_term = self.linear_s2.forward(outpooled.clone()).reshape([0, 1, -1]);

    // Term 2: Offset bias
    let offset_bias = self.score_belief_offset_bias.val().reshape([1, SCORE_ONE_HOT_SIZE, 1]);
    let s2off_term = self.linear_s2off.forward(offset_bias);

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
    let out_score_log_dist = ((log_terms - max.clone()).exp().sum_dim(2).log() + max).squeeze_dim(2);

    ValuePredictions {
      value: out_value,
      value_error: out_value_error,
      td_score: out_td_score,
      score_error: out_score_error,
      score: out_score_log_dist,
    }
  }

  /// The value outputs inference reads: the win/loss logits, the predicted
  /// short-term squared value error, and the longest-horizon TD score (the
  /// head whose target converges to the final score). The output projections
  /// run on slices of their weights, so the TD value distributions and the
  /// shorter score horizons - auxiliary training targets all - are never
  /// computed.
  pub fn forward_no_score(
    &self,
    inputs: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    mask_sum_hw: Tensor<B, 4>,
  ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let outv1 = self.conv1.forward(inputs);
    let outv1 = self.bias1.forward(outv1, mask.clone());
    let outv1 = mish(outv1);
    let outpooled = Self::gpool(outv1, mask_sum_hw).reshape([0, -1]);

    // Main Value Head

    let outv2 = self.linear2.forward(outpooled.clone());
    let outv2 = mish(outv2);
    let out_value = linear(
      outv2.clone(),
      self.linear_valuehead.weight.val().slice(s![.., 0..2]),
      self.linear_valuehead.bias.as_ref().map(|bias| bias.val().slice(s![0..2])),
    );
    let out_value_error = self.value_error(outv2.clone());
    // The TD score is a single small layer on features this path computes
    // anyway, unlike the score belief this path is there to skip. The longest
    // horizon comes first in [`TD_VALUE_COEFFS`].
    let out_td_score = linear(
      outv2,
      self.linear_td_score.weight.val().slice(s![.., 0..1]),
      self.linear_td_score.bias.as_ref().map(|bias| bias.val().slice(s![0..1])),
    ) * TD_SCORE_SCALE;
    (out_value, out_value_error, out_td_score)
  }

  /// Predicted squared short-term value error, from the second value layer.
  /// Shared by both forward paths so that the value the search weights by is
  /// exactly the one the error head was trained to produce.
  fn value_error(&self, outv2: Tensor<B, 2>) -> Tensor<B, 2> {
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
pub struct PolicyHead<B: Backend> {
  conv1p: Conv2d<B>,
  conv1g: Conv2d<B>,
  biasg: NormMask<B>,
  linearg: Linear<B>,
  bias2: NormMask<B>,
  conv2p: Conv2d<B>,
}

impl<B: Backend> PolicyHead<B> {
  pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
    Self {
      conv1p: Conv2dConfig::new([config.inner_channels, config.p1_channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      conv1g: Conv2dConfig::new([config.inner_channels, config.g1_channels], [1, 1])
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
  fn initialize(&mut self, device: &B::Device) {
    let gain = mish_gain();
    let scale_output = 0.3_f64;
    init_conv(&mut self.conv1p, 0.8, gain, device);
    init_conv(&mut self.conv1g, 1.0, gain, device);
    init_linear(&mut self.linearg, 0.6, gain, 0.0, gain, device);
    init_conv(&mut self.conv2p, scale_output, 1.0, device);
    // `biasg` and `bias2` stay learnable affines.
  }

  /// The head up to (not including) the final policy conv.
  fn features(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let outp = self.conv1p.forward(inputs.clone());
    let outg = self.conv1g.forward(inputs);
    let outg = self.biasg.forward(outg, mask.clone());
    let outg = mish(outg);
    let outg = ConvAndGPool::<B>::gpool(outg, mask.clone(), mask_sum_hw).reshape([0, -1]);
    let outg = self.linearg.forward(outg).unsqueeze_dims(&[-1, -1]);

    let outp = outp + outg;
    let outp = self.bias2.forward(outp, mask);
    mish(outp)
  }

  fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let outp = self.features(inputs, mask.clone(), mask_sum_hw);
    let outp = self.conv2p.forward(outp);
    outp - (1.0 - mask) * 5000.0
  }

  /// Only the first [`INFERENCE_POLICY_OUTPUTS`] channels: everything past
  /// them is an auxiliary training target, so the final conv runs on a slice
  /// of its weight instead of computing planes nothing would read.
  fn forward_inference(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let outp = self.features(inputs, mask.clone(), mask_sum_hw);
    let weight = self.conv2p.weight.val().slice(s![0..INFERENCE_POLICY_OUTPUTS]);
    let outp = conv2d(outp, weight, None, ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
    outp - (1.0 - mask) * 5000.0
  }
}

#[derive(Module, Debug)]
pub struct CapturedHead<B: Backend> {
  conv: Conv2d<B>,
}

impl<B: Backend> CapturedHead<B> {
  pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
    Self {
      conv: Conv2dConfig::new([config.inner_channels, 2], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
    }
  }

  /// Scale down the output conv (identity gain) so initial logits are small.
  fn initialize(&mut self, device: &B::Device) {
    init_conv(&mut self.conv, 0.2, 1.0, device);
  }

  fn forward(&self, inputs: Tensor<B, 4>) -> Tensor<B, 4> {
    self.conv.forward(inputs)
  }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
  conv_spatial: Conv2d<B>,
  linear_global: Linear<B>,
  residuals: Vec<ResidualBlock<B>>,
  norm_trunkfinal: NormMask<B>,
  value_head: ValueHead<B>,
  policy_head: PolicyHead<B>,
  captured_head: CapturedHead<B>,
}

impl<B: Backend> Model<B> {
  pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
    Self {
      conv_spatial: Conv2dConfig::new([CHANNELS, config.inner_channels], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      linear_global: LinearConfig::new(GLOBAL_FEATURES, config.inner_channels)
        .with_bias(false)
        .init(device),
      residuals: (0..config.residual_blocks)
        .map(|i| ResidualBlock::new(device, config, (i + 1) % config.gpool_every == 0))
        .collect(),
      norm_trunkfinal: NormMask::new(device, config.inner_channels, false),
      value_head: ValueHead::new(device, config),
      policy_head: PolicyHead::new(device, config),
      captured_head: CapturedHead::new(device, config),
    }
  }

  /// Fixup initialization for the residual trunk and heads. Every residual branch is
  /// zero-initialized so the network starts as a shallow function and each block's first conv
  /// is scaled by `1/sqrt(num_blocks)`, keeping activation and gradient variance stable
  /// through depth without any explicit normalization. Must be called once on a freshly
  /// created model before training; it is a no-op to call again before loading weights.
  pub fn initialize(&mut self, device: &B::Device) {
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

  pub fn forward(
    &self,
    spatial: Tensor<B, 4>,
    global: Tensor<B, 2>,
  ) -> (Tensor<B, 4>, ValuePredictions<B>, Tensor<B, 4>) {
    let mask = spatial.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let x_spatial = self.conv_spatial.forward(spatial);
    let x_global = self.linear_global.forward(global).unsqueeze_dims(&[-1, -1]);
    let mut x = x_spatial + x_global;
    for residual in &self.residuals {
      x = residual.forward(x, mask.clone(), mask_sum_hw.clone());
    }
    x = self.norm_trunkfinal.forward(x, mask.clone());
    x = mish(x);
    let policy = self.policy_head.forward(x.clone(), mask.clone(), mask_sum_hw.clone());
    let captured = self.captured_head.forward(x.clone());
    let value = self.value_head.forward(x, mask, mask_sum_hw);
    (policy, value, captured)
  }

  pub fn forward_no_score(
    &self,
    spatial: Tensor<B, 4>,
    global: Tensor<B, 2>,
  ) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let mask = spatial.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let x_spatial = self.conv_spatial.forward(spatial);
    let x_global = self.linear_global.forward(global).unsqueeze_dims(&[-1, -1]);
    let mut x = x_spatial + x_global;
    for residual in &self.residuals {
      x = residual.forward(x, mask.clone(), mask_sum_hw.clone());
    }
    x = self.norm_trunkfinal.forward(x, mask.clone());
    x = mish(x);
    let policy = self.policy_head.forward_inference(x.clone(), mask.clone(), mask_sum_hw.clone());
    let (value, value_error, td_score) = self.value_head.forward_no_score(x, mask, mask_sum_hw);
    (policy, value, value_error, td_score)
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
fn stdevs_excess_weight<B: Backend>(
  surprise: Tensor<B, 2>,
  predicted_sq_error: Tensor<B, 2>,
  variance_floor: f64,
) -> Tensor<B, 2> {
  let stdevs_excess = surprise / (predicted_sq_error + variance_floor).sqrt();
  sigmoid((stdevs_excess - 1.5) * 3.0)
}

/// The variance floor of a surprise measured in win probability, whose scale is
/// `[-1, 1]`, and of one measured in points.
const VALUE_VARIANCE_FLOOR: f64 = 1e-4;
const SCORE_VARIANCE_FLOOR: f64 = 0.25;

/// The score each bin of the score belief stands for, in points, as a row to
/// broadcast over a batch of distributions.
fn score_bins<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
  let center = (SCORE_ONE_HOT_SIZE / 2) as f32;
  let bins = (0..SCORE_ONE_HOT_SIZE).map(|i| i as f32 - center).collect::<Vec<_>>();
  Tensor::from_data(TensorData::new(bins, [1, SCORE_ONE_HOT_SIZE]), device)
}

/// Mean and variance, in points, of a distribution over the score bins - of the
/// score belief the net predicts, or of the one-hot target it is trained on,
/// whose mean is then simply the score the game ended at.
fn score_mean_and_variance<B: Backend>(probs: Tensor<B, 2>, bins: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
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
fn interpolate_policy<B: Backend>(policy_logits: Tensor<B, 4>, optimism: Tensor<B, 3>) -> Tensor<B, 3> {
  let policy: Tensor<B, 3> = policy_logits.clone().slice(s![.., 0..1, .., ..]).squeeze_dim(1);
  let optimistic: Tensor<B, 3> = policy_logits
    .slice(s![.., OPTIMISTIC_POLICY..OPTIMISTIC_POLICY + 1, .., ..])
    .squeeze_dim(1);
  policy.clone() + (optimistic - policy) * optimism
}

#[derive(Clone)]
pub struct Predictor<B: Backend> {
  pub model: Model<B>,
  pub device: B::Device,
}

#[derive(Clone)]
pub struct Learner<B: AutodiffBackend, O> {
  pub predictor: Predictor<B>,
  pub optimizer: O,
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

impl<B> OppaiModel<FloatElem<B>> for Predictor<B>
where
  B: Backend,
  FloatElem<B>: Float,
{
  type E = ModelError;

  async fn predict(
    &mut self,
    inputs: Array4<FloatElem<B>>,
    global: Array2<FloatElem<B>>,
    optimism: Array1<FloatElem<B>>,
  ) -> Result<(Array3<FloatElem<B>>, Array2<FloatElem<B>>), Self::E> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.device,
    );
    let global = Tensor::from_data(
      TensorData::new(into_data_vec(global), [batch, GLOBAL_FEATURES]),
      &self.device,
    );
    let optimism = Tensor::from_data(TensorData::new(into_data_vec(optimism), [batch, 1, 1]), &self.device);
    let (policy_logits, value_logits, value_error, td_score) = self.model.forward_no_score(inputs, global);
    let policy_logits = interpolate_policy(policy_logits, optimism);
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    // The predicted squared error becomes a standard deviation for the
    // search's uncertainty weighting. The score estimate is the
    // longest-horizon TD score, the head whose target converges to the final
    // score; the search averages it over each subtree into the per-move q
    // score targets.
    let values = Tensor::cat(vec![softmax(value_logits, 1), value_error.sqrt(), td_score], 1);
    let policies = Array3::from_shape_vec((batch, height, width), policies.into_data_async().await?.into_vec()?)?;
    let values = Array2::from_shape_vec((batch, 4), values.into_data_async().await?.into_vec()?)?;
    Ok((policies, values))
  }
}

impl<B, O> OppaiModel<FloatElem<B>> for Learner<B, O>
where
  B: Backend + AutodiffBackend,
  FloatElem<B>: Float,
{
  type E = ModelError;

  async fn predict(
    &mut self,
    inputs: Array4<FloatElem<B>>,
    global: Array2<FloatElem<B>>,
    optimism: Array1<FloatElem<B>>,
  ) -> Result<(Array3<FloatElem<B>>, Array2<FloatElem<B>>), Self::E> {
    self.predictor.predict(inputs, global, optimism).await
  }
}

struct ParamNormVisitor<B: Backend> {
  sum_sq: Tensor<B, 1>,
}

impl<B: Backend> ParamNormVisitor<B> {
  fn new(device: &B::Device) -> Self {
    Self {
      sum_sq: Tensor::zeros([1], device),
    }
  }

  fn l2_norm(self) -> FloatElem<B> {
    self.sum_sq.sqrt().into_scalar()
  }
}

impl<B: Backend> ModuleVisitor<B> for ParamNormVisitor<B> {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
    let tensor = param.val();
    self.sum_sq = self.sum_sq.clone() + (tensor.clone() * tensor).sum();
  }
}

impl<B, O> OppaiTrainableModel<FloatElem<B>> for Learner<B, O>
where
  B: Backend + AutodiffBackend,
  FloatElem<B>: Float,
  O: Optimizer<Model<B>, B>,
{
  type TE = ModelError;

  fn train(
    mut self,
    inputs: Array4<FloatElem<B>>,
    global: Array2<FloatElem<B>>,
    policies: Array3<FloatElem<B>>,
    opponent_policies: Array3<FloatElem<B>>,
    values: Array2<FloatElem<B>>,
    td_values: Array3<FloatElem<B>>,
    td_scores: Array2<FloatElem<B>>,
    scores: Array2<FloatElem<B>>,
    captured: Array4<FloatElem<B>>,
    q_values: Array4<FloatElem<B>>,
    learning_rate: f64,
  ) -> Result<Self, Self::TE> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.predictor.device,
    );
    let global = Tensor::from_data(
      TensorData::new(into_data_vec(global), [batch, GLOBAL_FEATURES]),
      &self.predictor.device,
    );
    let policies = Tensor::from_data(
      TensorData::new(into_data_vec(policies), [batch, height * width]),
      &self.predictor.device,
    );
    let opponent_policies = Tensor::from_data(
      TensorData::new(into_data_vec(opponent_policies), [batch, height * width]),
      &self.predictor.device,
    );
    let values = Tensor::from_data(
      TensorData::new(into_data_vec(values), [batch, 2]),
      &self.predictor.device,
    );
    let td_values: Tensor<B, 3> = Tensor::from_data(
      TensorData::new(into_data_vec(td_values), [batch, TD_VALUES, 2]),
      &self.predictor.device,
    );
    let td_scores: Tensor<B, 2> = Tensor::from_data(
      TensorData::new(into_data_vec(td_scores), [batch, TD_VALUES]),
      &self.predictor.device,
    );
    let scores = Tensor::from_data(
      TensorData::new(into_data_vec(scores), [batch, SCORE_ONE_HOT_SIZE]),
      &self.predictor.device,
    );
    let scores_cdf = scores.clone().cumsum(1);
    let captured = Tensor::from_data(
      TensorData::new(into_data_vec(captured), [batch, 2, height, width]),
      &self.predictor.device,
    );
    let q_values: Tensor<B, 4> = Tensor::from_data(
      TensorData::new(into_data_vec(q_values), [batch, 3, height, width]),
      &self.predictor.device,
    );
    let q_targets = q_values.clone().slice(s![.., 0..1]).reshape([0, -1]);
    let q_score_targets = q_values.clone().slice(s![.., 1..2]).reshape([0, -1]);
    let q_weights = q_values.slice(s![.., 2..3]).reshape([0, -1]);
    // The captured head predicts the terminal captured state of every board
    // cell, so the loss is masked only by the board mask.
    let mask = inputs.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let (out_policy_logits, out_value, out_captured_logits) = self.predictor.model.forward(inputs, global);
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

    let batch = <FloatElem<B> as num_traits::NumCast>::from(batch).unwrap();
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
    let td_scores_loss = HuberLossConfig::new(12.0)
      .init()
      .forward_no_reduction(out_td_scores.clone(), td_scores.clone())
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
    let score_error_loss = HuberLossConfig::new(100.0)
      .init()
      .forward_no_reduction(out_score_error.clone(), score_sq_error)
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
    let optimistic_weight = (stdevs_excess_weight(value_surprise, out_value_error.detach(), VALUE_VARIANCE_FLOOR)
      + stdevs_excess_weight(score_surprise, out_score_error.detach(), SCORE_VARIANCE_FLOOR))
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
    let bins = score_bins(&self.predictor.device);
    let (pred_score_mean, pred_score_variance) =
      score_mean_and_variance(out_scores.clone().exp().detach(), bins.clone());
    let (real_score_mean, _) = score_mean_and_variance(scores.clone(), bins);
    let win = values.slice(s![.., 0..1]);
    let long_optimistic_weight = (win.powi_scalar(2)
      + stdevs_excess_weight(
        real_score_mean - pred_score_mean,
        pred_score_variance,
        SCORE_VARIANCE_FLOOR,
      ))
    .clamp_max(1.0);
    let long_optimistic_policies_loss =
      -(out_long_optimistic_policies * policies * long_optimistic_weight).sum() * 0.108 / batch;
    let opponent_policies_loss = -(out_opponent_policies * opponent_policies).sum() * 0.15 / batch;
    let soft_policies_loss = -(out_soft_policies * soft_policies).sum() * 8.0 / batch;
    let soft_opponent_policies_loss =
      -(out_soft_opponent_policies * soft_opponent_policies * opponent_weight).sum() * 1.2 / batch;
    let pdf_loss = -(out_scores * scores).sum() * 0.02 / batch;
    let cdf_loss = (out_scores_cdf - scores_cdf).square().sum() * 0.02 / batch;
    // Binary cross-entropy with logits in the numerically stable form
    // `max(z, 0) - z * t + ln(1 + exp(-|z|))`, normalized by the board area
    // like KataGo's ownership loss.
    let captured_bce = out_captured_logits.clone().clamp_min(0.0) - out_captured_logits.clone() * captured
      + (-out_captured_logits.abs()).exp().log1p();
    let captured_loss = ((captured_bce * mask).sum_dim(2).sum_dim(3) / mask_sum_hw).sum() * 1.5 / batch;

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

    let mut norm_visitor = ParamNormVisitor::new(&self.predictor.device);
    self.predictor.model.visit(&mut norm_visitor);
    let param_l2_norm = norm_visitor.l2_norm();

    log::info!(
      "Loss: value {} td value {} value error {} td score {} score error {} policy {} opponent policy {} soft policy {} soft opponent policy {} optimistic policy {} long optimistic policy {} pdf {} cdf {} captured {} q {} q score {} L2 norm {}",
      values_loss.clone().into_scalar(),
      td_values_loss.clone().into_scalar(),
      value_error_loss.clone().into_scalar(),
      td_scores_loss.clone().into_scalar(),
      score_error_loss.clone().into_scalar(),
      policies_loss.clone().into_scalar(),
      opponent_policies_loss.clone().into_scalar(),
      soft_policies_loss.clone().into_scalar(),
      soft_opponent_policies_loss.clone().into_scalar(),
      optimistic_policies_loss.clone().into_scalar(),
      long_optimistic_policies_loss.clone().into_scalar(),
      pdf_loss.clone().into_scalar(),
      cdf_loss.clone().into_scalar(),
      captured_loss.clone().into_scalar(),
      q_values_loss.clone().into_scalar(),
      q_scores_loss.clone().into_scalar(),
      param_l2_norm,
    );

    let loss = values_loss
      + td_values_loss
      + value_error_loss
      + policies_loss
      + opponent_policies_loss
      + soft_policies_loss
      + soft_opponent_policies_loss
      + optimistic_policies_loss
      + long_optimistic_policies_loss
      + td_scores_loss
      + score_error_loss
      + pdf_loss
      + cdf_loss
      + captured_loss
      + q_values_loss
      + q_scores_loss;

    let grads = GradientsParams::from_grads(loss.backward(), &self.predictor.model);
    self.predictor.model = self.optimizer.step(learning_rate, self.predictor.model, grads);

    Ok(self)
  }
}

#[cfg(all(
  test,
  any(feature = "flex", feature = "ndarray", feature = "vulkan", feature = "webgpu")
))]
mod tests {
  #[cfg(feature = "ndarray")]
  use super::{
    ConvOrGpool, OPTIMISTIC_POLICY, POLICY_OUTPUTS, SCORE_VARIANCE_FLOOR, VALUE_VARIANCE_FLOOR, interpolate_policy,
    score_bins, score_mean_and_variance, squared_softplus_with_gradient_floor, stdevs_excess_weight,
  };
  use super::{Learner, Model, ModelConfig, Predictor};
  #[cfg(feature = "flex")]
  use burn::backend::{Flex, flex::FlexDevice};
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  use burn::backend::{Wgpu, wgpu::WgpuDevice};
  use burn::{backend::Autodiff, optim::SgdConfig};
  #[cfg(feature = "ndarray")]
  use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::{Tensor, TensorData, activation::softmax},
  };
  use ndarray::{Array2, Array3, Array4, Axis, array};
  use oppai_zero::{
    examples::TD_VALUES,
    field_features::{CHANNELS, SCORE_ONE_HOT_SIZE},
    model::{Model as OppaiModel, TrainableModel},
  };

  #[test]
  fn default_config_file_matches_default() {
    let config = ModelConfig::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/configs/b5c192nbt.json")).unwrap();
    assert_eq!(config, ModelConfig::default());
  }

  // The forward value is squared softplus, while the gradient is that of the
  // plain softplus, never dropping below the floor even for very negative
  // inputs.
  #[test]
  #[cfg(feature = "ndarray")]
  fn squared_softplus_gradient_floor() {
    let device = NdArrayDevice::Cpu;
    let x = Tensor::<Autodiff<NdArray>, 2>::from_floats([[-20.0, 0.0, 20.0]], &device).require_grad();
    let out = squared_softplus_with_gradient_floor(x.clone(), 0.05);

    let values = out.clone().into_data().to_vec::<f32>().unwrap();
    let expected = [0.0f32, 0.480453, 100.000908];
    for (value, expected) in values.into_iter().zip(expected) {
      assert!((value - expected).abs() < 1e-3);
    }

    let grads = out.sum().backward();
    let grads = x.grad(&grads).unwrap().into_data().to_vec::<f32>().unwrap();
    let expected = [0.05f32, 0.525, 1.0];
    for (grad, expected) in grads.into_iter().zip(expected) {
      assert!((grad - expected).abs() < 1e-3);
    }
  }

  /// Only the samples that beat the net's own short-term prediction train the
  /// optimistic policy, and by how many of its own predicted standard deviations
  /// they beat it by. A position that went as predicted - or worse - must count
  /// for next to nothing, or the head just relearns the plain policy.
  #[test]
  #[cfg(feature = "ndarray")]
  fn optimistic_weight_follows_the_surprise() {
    let device = NdArrayDevice::Cpu;
    // A predicted error of 0.25 is a predicted standard deviation of 0.5, so
    // the threshold sits at a surprise of 0.75.
    let surprise = Tensor::<NdArray, 2>::from_floats([[-1.0], [0.0], [0.75], [1.5], [3.0]], &device);
    let predicted_sq_error = Tensor::<NdArray, 2>::from_floats([[0.25], [0.25], [0.25], [0.25], [0.25]], &device);
    let weights = stdevs_excess_weight(surprise, predicted_sq_error, VALUE_VARIANCE_FLOOR)
      .into_data()
      .to_vec::<f32>()
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
    let surprise = Tensor::<NdArray, 2>::from_floats([[0.75], [15.0]], &device);
    let predicted_sq_error = Tensor::<NdArray, 2>::from_floats([[100.0], [100.0]], &device);
    let weights = stdevs_excess_weight(surprise, predicted_sq_error, SCORE_VARIANCE_FLOOR)
      .into_data()
      .to_vec::<f32>()
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
  #[cfg(feature = "ndarray")]
  fn score_belief_mean_and_variance() {
    let device = NdArrayDevice::Cpu;
    let bins = score_bins::<NdArray>(&device);
    let center = SCORE_ONE_HOT_SIZE / 2;

    // Row 0: all of the mass on a score of 7, the way the training target
    // encodes a whole-point score. Row 1: half on -3 and half on 11.
    let mut probs = [vec![0.0f32; SCORE_ONE_HOT_SIZE], vec![0.0f32; SCORE_ONE_HOT_SIZE]];
    probs[0][center + 7] = 1.0;
    probs[1][center - 3] = 0.5;
    probs[1][center + 11] = 0.5;
    let probs = Tensor::<NdArray, 2>::from_data(TensorData::new(probs.concat(), [2, SCORE_ONE_HOT_SIZE]), &device);

    let (mean, variance) = score_mean_and_variance(probs, bins);
    let mean = mean.into_data().to_vec::<f32>().unwrap();
    let variance = variance.into_data().to_vec::<f32>().unwrap();

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
  #[cfg(feature = "ndarray")]
  fn interpolate_policy_between_the_heads() {
    let device = NdArrayDevice::Cpu;
    // Two positions of a single cell each, so the softmax is over one channel
    // pair per row and the logits are the policies up to normalization.
    let mut logits = [[0.0f32; 2]; POLICY_OUTPUTS];
    logits[0] = [1.0, -1.0];
    logits[OPTIMISTIC_POLICY] = [-3.0, 5.0];
    let logits: Tensor<NdArray, 4> =
      Tensor::<NdArray, 2>::from_floats(logits, &device).reshape([1, POLICY_OUTPUTS, 1, 2]);

    let blend = |optimism: f32| {
      let optimism = Tensor::<NdArray, 3>::from_floats([[[optimism]]], &device);
      interpolate_policy(logits.clone(), optimism)
        .into_data()
        .to_vec::<f32>()
        .unwrap()
    };

    assert_eq!(blend(0.0), vec![1.0, -1.0]);
    assert_eq!(blend(1.0), vec![-3.0, 5.0]);
    // A quarter of the way there, in logits.
    assert_eq!(blend(0.25), vec![0.0, 0.5]);

    // Each row is blended by its own weight, not by the batch's.
    let logits = logits.clone().repeat_dim(0, 2);
    let optimism = Tensor::<NdArray, 3>::from_floats([[[0.0]], [[1.0]]], &device);
    let blended = interpolate_policy(logits, optimism)
      .into_data()
      .to_vec::<f32>()
      .unwrap();
    assert_eq!(blended, vec![1.0, -1.0, -3.0, 5.0]);
  }

  #[test]
  #[cfg(feature = "ndarray")]
  fn forward() {
    let model = Model::<NdArray>::new(&NdArrayDevice::Cpu, &ModelConfig::default());
    let (policy_logits, predictions, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &NdArrayDevice::Cpu),
      Tensor::ones([1, 1], &NdArrayDevice::Cpu),
    );
    let values = predictions.value;
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    assert!(
      policies
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|p| (0.0..=1.0).contains(p))
    );
    assert!(policies.iter_dim(0).all(|p| (p.sum().into_scalar() - 1.0) < 0.001));
    assert!(
      values
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| (-1.0..=1.0).contains(v))
    );
  }

  // Verifies the core Fixup invariant: after `initialize`, every residual branch ends in a
  // zero-initialized conv so each block starts as the identity, and the model still produces a
  // valid, finite policy distribution.
  #[test]
  #[cfg(feature = "ndarray")]
  fn initialize_zeroes_residual_branches() {
    let device = NdArrayDevice::Cpu;
    let mut model = Model::<NdArray>::new(&device, &ModelConfig::default());
    model.initialize(&device);

    let assert_zero = |convgpool: &ConvOrGpool<NdArray>| match convgpool {
      ConvOrGpool::Conv(conv) => {
        let abs_sum = conv.weight.val().abs().sum().into_scalar();
        assert_eq!(abs_sum, 0.0, "residual branch output conv must be zero-initialized");
      }
      ConvOrGpool::Gpool(_) => panic!("residual branch output should be a plain conv"),
    };

    for residual in &model.residuals {
      assert_zero(&residual.normactconvq.convgpool);
      for inner in &residual.inner {
        assert_zero(&inner.normactconv2.convgpool);
      }
    }

    let (policy_logits, predictions, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &device),
      Tensor::ones([1, 1], &device),
    );
    let values = predictions.value;
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    assert!(
      policies
        .iter_dim(0)
        .all(|p| (p.sum().into_scalar() - 1.0).abs() < 0.001)
    );
    assert!(
      values
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
    );
  }

  macro_rules! predict_test {
    ($name:ident, $backend:ty, $device:expr) => {
      #[test]
      fn $name() {
        let model = Model::<$backend>::new(&$device, &ModelConfig::default());
        let mut predictor = Predictor {
          model,
          device: $device,
        };
        let (policies, values) = futures::executor::block_on(predictor.predict(
          Array4::from_elem((1, CHANNELS, 4, 8), 1.0),
          array![[0.2]],
          array![0.0],
        ))
        .unwrap();
        // Win and loss probabilities, the predicted short-term error as a
        // standard deviation, which the search turns into a playout weight,
        // and the predicted score in points.
        assert_eq!(values.dim(), (1, 4));
        assert!((values[(0, 0)] + values[(0, 1)] - 1.0).abs() < 1e-4);
        assert!(values[(0, 2)] >= 0.0 && values[(0, 2)].is_finite());
        assert!(values[(0, 3)].is_finite());

        // The same position three times, each row asking for its own optimism:
        // a batch is what the search actually submits, and every row of it has
        // to be blended by its own weight rather than the batch's first.
        let (batched, _) = futures::executor::block_on(predictor.predict(
          Array4::from_elem((3, CHANNELS, 4, 8), 1.0),
          Array2::from_elem((3, 1), 0.2),
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
  predict_test!(predict_flex, Flex, FlexDevice);
  #[cfg(feature = "ndarray")]
  predict_test!(predict_ndarray, NdArray, NdArrayDevice::Cpu);
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  predict_test!(predict_wgpu, Wgpu, WgpuDevice::DefaultDevice);

  macro_rules! train_test {
    ($name:ident, $backend:ty, $device:expr) => {
      #[test]
      fn $name() {
        let model = Model::<Autodiff<$backend>>::new(&$device, &ModelConfig::default());
        let predictor = Predictor {
          model,
          device: $device,
        };
        let optimizer = SgdConfig::new().init::<Autodiff<$backend>, Model<_>>();
        let mut learner = Learner { predictor, optimizer };

        let inputs = Array4::from_elem((1, CHANNELS, 4, 8), 1.0);
        let global = array![[0.2]];
        let policies = Array3::from_elem((1, 4, 8), 0.5);
        let opponent_policies = Array3::from_elem((1, 4, 8), 0.7);
        let values = array![[1.0, 0.0]];
        let td_values = Array3::from_elem((1, TD_VALUES, 2), 0.5);
        let td_scores = Array2::from_elem((1, TD_VALUES), 3.0);
        let mut scores = Array2::from_elem((1, SCORE_ONE_HOT_SIZE), 0.0);
        scores[(0, 0)] = 1.0;
        let captured = Array4::from_elem((1, 2, 4, 8), 1.0);
        // One explored move with a recorded q value, q score and search
        // weight; every other cell keeps zero weight and stays out of the q
        // losses.
        let mut q_values = Array4::from_elem((1, 3, 4, 8), 0.0);
        q_values[(0, 0, 1, 2)] = 0.5;
        q_values[(0, 1, 1, 2)] = 2.5;
        q_values[(0, 2, 1, 2)] = 4.0;

        let (out_policies_1, out_values_1) =
          futures::executor::block_on(learner.predict(inputs.clone(), global.clone(), array![0.0])).unwrap();
        let mut learner = learner
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
            q_values,
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

  #[cfg(feature = "flex")]
  train_test!(train_flex, Flex, FlexDevice);
  #[cfg(feature = "ndarray")]
  train_test!(train_ndarray, NdArray, NdArrayDevice::Cpu);
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  train_test!(train_wgpu, Wgpu, WgpuDevice::DefaultDevice);
}
