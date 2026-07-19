use burn::{
  module::{Initializer, Module, ModuleMapper, ModuleVisitor, Param},
  nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
  },
  optim::{GradientsParams, Optimizer},
  tensor::{
    DataError, Tensor, TensorData,
    activation::{log_softmax, mish, softmax, softplus},
    backend::{AutodiffBackend, Backend, ExecutionError},
    ops::FloatElem,
    s,
  },
};
use derive_more::From;
use ndarray::{Array, Array2, Array3, Array4, Dimension, ShapeError};
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

  pub fn forward(
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
    let out_value = self.linear_valuehead.forward(outv2.clone());
    // Softplus keeps the predicted squared error positive; the multiplier is
    // KataGo's, keeping the initial output small relative to the [0, 4] range.
    let out_value_error = softplus(self.linear_error.forward(outv2), 1.0) * 0.25;

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

    // Take the mixture distribution weighted by outsmix_logweights
    // TODO: might be numerically unstable, but burn doesn't have LogSumExp operator
    // See https://en.wikipedia.org/wiki/LogSumExp
    let out_score_log_dist = (out_scorebelief_logprobs + outsmix_logweights.unsqueeze_dim(1))
      .exp()
      .sum_dim(2)
      .log()
      .squeeze_dim(2);

    (out_value, out_value_error, out_score_log_dist)
  }

  pub fn forward_no_score(
    &self,
    inputs: Tensor<B, 4>,
    mask: Tensor<B, 4>,
    mask_sum_hw: Tensor<B, 4>,
  ) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let outv1 = self.conv1.forward(inputs);
    let outv1 = self.bias1.forward(outv1, mask.clone());
    let outv1 = mish(outv1);
    let outpooled = Self::gpool(outv1, mask_sum_hw).reshape([0, -1]);

    // Main Value Head

    let outv2 = self.linear2.forward(outpooled.clone());
    let outv2 = mish(outv2);
    let out_value = self.linear_valuehead.forward(outv2.clone());
    let out_value_error = softplus(self.linear_error.forward(outv2), 1.0) * 0.25;
    (out_value, out_value_error)
  }
}

/// Policy head output channels:
/// 0 - policy, 1 - opponent policy,
/// 2 - soft policy, 3 - soft opponent policy.
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
      conv2p: Conv2dConfig::new([config.p1_channels, 4], [1, 1])
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

  fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let outp = self.conv1p.forward(inputs.clone());
    let outg = self.conv1g.forward(inputs);
    let outg = self.biasg.forward(outg, mask.clone());
    let outg = mish(outg);
    let outg = ConvAndGPool::<B>::gpool(outg, mask.clone(), mask_sum_hw).reshape([0, -1]);
    let outg = self.linearg.forward(outg).unsqueeze_dims(&[-1, -1]);

    let outp = outp + outg;
    let outp = self.bias2.forward(outp, mask.clone());
    let outp = mish(outp);
    let outp = self.conv2p.forward(outp);
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
  ) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 4>) {
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
    let (value, value_error, score) = self.value_head.forward(x, mask, mask_sum_hw);
    (policy, value, value_error, score, captured)
  }

  pub fn forward_no_score(
    &self,
    spatial: Tensor<B, 4>,
    global: Tensor<B, 2>,
  ) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
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
    let (value, value_error) = self.value_head.forward_no_score(x, mask, mask_sum_hw);
    (policy, value, value_error)
  }
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
    let (policy_logits, value_logits, value_error) = self.model.forward_no_score(inputs, global);
    // TODO: lightweight model that doesn't calculate second layer
    let policy_logits: Tensor<B, 3> = policy_logits.slice(s![.., 0..1, .., ..]).squeeze_dim(1);
    let policies = softmax(policy_logits.reshape([0, -1]), 1);
    // The predicted squared error becomes a standard deviation for the
    // search's uncertainty weighting.
    let values = Tensor::cat(vec![softmax(value_logits.slice(s![.., 0..2]), 1), value_error.sqrt()], 1);
    let policies = Array3::from_shape_vec((batch, height, width), policies.into_data_async().await?.into_vec()?)?;
    let values = Array2::from_shape_vec((batch, 3), values.into_data_async().await?.into_vec()?)?;
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
  ) -> Result<(Array3<FloatElem<B>>, Array2<FloatElem<B>>), Self::E> {
    self.predictor.predict(inputs, global).await
  }
}

/// Huber loss: quadratic within `delta` of the target, linear beyond it.
fn huber_loss<B: Backend>(pred: Tensor<B, 2>, target: Tensor<B, 2>, delta: f64) -> Tensor<B, 2> {
  let diff = pred - target;
  let abs = diff.clone().abs();
  let quadratic = diff.square() * 0.5;
  let linear = abs.clone() * delta - delta * delta * 0.5;
  quadratic.mask_where(abs.greater_elem(delta), linear)
}

struct ParamCollector {
  params: Vec<TensorData>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector {
  fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
    self.params.push(param.val().into_data());
  }
}

struct EmaMapper<'a, B: Backend> {
  params: std::collections::VecDeque<TensorData>,
  factor: f64,
  device: &'a B::Device,
}

impl<B: Backend> ModuleMapper<B> for EmaMapper<'_, B> {
  fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
    let data = self.params.pop_front().expect("model architectures must match");
    let current = Tensor::from_data(data, self.device);
    let (id, average, mapper) = param.consume();
    let average = average.clone() + (current - average) * self.factor;
    Param::from_mapped_value(id, average, mapper)
  }
}

/// Update an exponential moving average of the model weights:
/// `average += factor * (model - average)` for every parameter, pairing the
/// parameters of the two models by module traversal order. This is KataGo's
/// SWA scheme with `factor = 1 / swa_scale`, so `1 / factor` snapshots are
/// averaged in expectation.
pub fn ema_update<B: Backend>(average: Model<B>, model: &Model<B>, factor: f64, device: &B::Device) -> Model<B> {
  let mut collector = ParamCollector { params: Vec::new() };
  model.visit(&mut collector);
  let mut mapper = EmaMapper {
    params: collector.params.into(),
    factor,
    device,
  };
  average.map(&mut mapper)
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
    scores: Array2<FloatElem<B>>,
    captured: Array4<FloatElem<B>>,
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
    let scores = Tensor::from_data(
      TensorData::new(into_data_vec(scores), [batch, SCORE_ONE_HOT_SIZE]),
      &self.predictor.device,
    );
    let scores_cdf = scores.clone().cumsum(1);
    let captured = Tensor::from_data(
      TensorData::new(into_data_vec(captured), [batch, 2, height, width]),
      &self.predictor.device,
    );
    // The captured head predicts the terminal captured state of every board
    // cell, so the loss is masked only by the board mask.
    let mask = inputs.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let (out_policy_logits, out_value_logits, out_value_error, out_scores, out_captured_logits) =
      self.predictor.model.forward(inputs, global);
    let out_policies = log_softmax(
      out_policy_logits.clone().slice(s![.., 0..1, .., ..]).reshape([0, -1]),
      1,
    );
    let out_opponent_policies = log_softmax(
      out_policy_logits.clone().slice(s![.., 1..2, .., ..]).reshape([0, -1]),
      1,
    );
    let out_soft_policies = log_softmax(
      out_policy_logits.clone().slice(s![.., 2..3, .., ..]).reshape([0, -1]),
      1,
    );
    let out_soft_opponent_policies = log_softmax(out_policy_logits.slice(s![.., 3..4, .., ..]).reshape([0, -1]), 1);
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

    let batch = <FloatElem<B> as num_traits::NumCast>::from(batch).unwrap();
    // KataGo's 1.20 samplewise factor times the 0.6 value_loss_scale, for the
    // main value head and each TD horizon alike.
    let values_loss = -(out_values * values).sum() * 0.72 / batch;
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
    // a tiny irreducible error for regularization, as in KataGo.
    let td_short_pred = softmax(
      out_value_logits.clone().slice(s![.., 2 + 2 * (TD_VALUES - 1)..]).detach(),
      1,
    );
    let pred_value = td_short_pred.clone().slice(s![.., 0..1]) - td_short_pred.slice(s![.., 1..2]);
    let td_short_target = td_values.clone().slice(s![.., TD_VALUES - 1.., ..]).reshape([0, -1]);
    let real_value = td_short_target.clone().slice(s![.., 0..1]) - td_short_target.slice(s![.., 1..2]);
    let sq_error = (pred_value - real_value).square() + 1e-8;
    let value_error_loss = huber_loss(out_value_error, sq_error, 0.4).sum() * 2.0 / batch;
    let policies_loss = -(out_policies * policies).sum() / batch;
    let opponent_policies_loss = -(out_opponent_policies * opponent_policies).sum() * 0.15 / batch;
    // KataGo's soft_policy_weight_scale is 8.0; the opponent variant keeps the
    // same extra 0.15 factor as the hard opponent policy loss.
    let soft_policies_loss = -(out_soft_policies * soft_policies).sum() * 8.0 / batch;
    let soft_opponent_policies_loss = -(out_soft_opponent_policies * soft_opponent_policies).sum() * 1.2 / batch;
    let pdf_loss = -(out_scores * scores).sum() * 0.02 / batch;
    let cdf_loss = (out_scores_cdf - scores_cdf).square().sum() * 0.02 / batch;
    // Binary cross-entropy with logits in the numerically stable form
    // `max(z, 0) - z * t + ln(1 + exp(-|z|))`, normalized by the board area
    // like KataGo's ownership loss.
    let captured_bce = out_captured_logits.clone().clamp_min(0.0) - out_captured_logits.clone() * captured
      + (-out_captured_logits.abs()).exp().log1p();
    let captured_loss = ((captured_bce * mask).sum_dim(2).sum_dim(3) / mask_sum_hw).sum() * 1.5 / batch;

    let mut norm_visitor = ParamNormVisitor::new(&self.predictor.device);
    self.predictor.model.visit(&mut norm_visitor);
    let param_l2_norm = norm_visitor.l2_norm();

    log::info!(
      "Loss: value {} td value {} value error {} policy {} opponent policy {} soft policy {} soft opponent policy {} pdf {} cdf {} captured {} L2 norm {}",
      values_loss.clone().into_scalar(),
      td_values_loss.clone().into_scalar(),
      value_error_loss.clone().into_scalar(),
      policies_loss.clone().into_scalar(),
      opponent_policies_loss.clone().into_scalar(),
      soft_policies_loss.clone().into_scalar(),
      soft_opponent_policies_loss.clone().into_scalar(),
      pdf_loss.clone().into_scalar(),
      cdf_loss.clone().into_scalar(),
      captured_loss.clone().into_scalar(),
      param_l2_norm,
    );

    let loss = values_loss
      + td_values_loss
      + value_error_loss
      + policies_loss
      + opponent_policies_loss
      + soft_policies_loss
      + soft_opponent_policies_loss
      + pdf_loss
      + cdf_loss
      + captured_loss;

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
  use super::ConvOrGpool;
  use super::{Learner, Model, ModelConfig, Predictor};
  #[cfg(feature = "flex")]
  use burn::backend::{Flex, flex::FlexDevice};
  #[cfg(any(feature = "vulkan", feature = "webgpu"))]
  use burn::backend::{Wgpu, wgpu::WgpuDevice};
  use burn::{backend::Autodiff, optim::SgdConfig};
  #[cfg(feature = "ndarray")]
  use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    tensor::{Tensor, activation::softmax},
  };
  use ndarray::{Array2, Array3, Array4, array};
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

  #[test]
  #[cfg(feature = "ndarray")]
  fn forward() {
    let model = Model::<NdArray>::new(&NdArrayDevice::Cpu, &ModelConfig::default());
    let (policy_logits, values, _, _, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &NdArrayDevice::Cpu),
      Tensor::ones([1, 1], &NdArrayDevice::Cpu),
    );
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

    let (policy_logits, values, _, _, _) = model.forward(
      Tensor::ones([1, CHANNELS, 4, 8], &device),
      Tensor::ones([1, 1], &device),
    );
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

  // With factor 1 the average must become an exact copy of the model, and with
  // factor 0 it must stay unchanged - the two endpoints of the EMA lerp.
  #[test]
  #[cfg(feature = "ndarray")]
  fn ema_update_endpoints() {
    use super::ema_update;

    let device = NdArrayDevice::Cpu;
    let config = ModelConfig::default();
    let mut average = Model::<NdArray>::new(&device, &config);
    average.initialize(&device);
    let mut model = Model::<NdArray>::new(&device, &config);
    model.initialize(&device);

    let inputs = Tensor::ones([1, CHANNELS, 4, 8], &device);
    let global = Tensor::ones([1, 1], &device);

    let (model_policy, _, _, _, _) = model.forward(inputs.clone(), global.clone());
    let (average_policy, _, _, _, _) = average.forward(inputs.clone(), global.clone());

    let average = ema_update(average, &model, 0.0, &device);
    let (unchanged_policy, _, _, _, _) = average.forward(inputs.clone(), global.clone());
    unchanged_policy
      .into_data()
      .assert_eq(&average_policy.into_data(), true);

    // `a + (c - a) * 1` is only c up to rounding, so compare approximately.
    let average = ema_update(average, &model, 1.0, &device);
    let (copied_policy, _, _, _, _) = average.forward(inputs, global);
    copied_policy
      .into_data()
      .assert_approx_eq::<f32>(&model_policy.into_data(), burn::tensor::Tolerance::default());
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
        futures::executor::block_on(predictor.predict(Array4::from_elem((1, CHANNELS, 4, 8), 1.0), array![[0.2]]))
          .unwrap();
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
        let mut scores = Array2::from_elem((1, SCORE_ONE_HOT_SIZE), 0.0);
        scores[(0, 0)] = 1.0;
        let captured = Array4::from_elem((1, 2, 4, 8), 1.0);

        let (out_policies_1, out_values_1) =
          futures::executor::block_on(learner.predict(inputs.clone(), global.clone())).unwrap();
        let mut learner = learner
          .train(
            inputs.clone(),
            global.clone(),
            policies,
            opponent_policies,
            values,
            td_values,
            scores,
            captured,
            0.01,
          )
          .unwrap();
        let (out_policies_2, out_values_2) = futures::executor::block_on(learner.predict(inputs, global)).unwrap();

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
