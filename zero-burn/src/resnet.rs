//! The convolutional trunk: nested bottleneck residual blocks of `3x3`
//! convolutions, some of which also pool the board globally.

use crate::model::{NormActConv, ResNetConfig, TrunkBlock};
use burn::{
  module::Module,
  tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
pub struct InnerResidualBlock {
  pub(crate) normactconv1: NormActConv,
  pub(crate) normactconv2: NormActConv,
}

impl InnerResidualBlock {
  pub fn new(device: &Device, config: &ResNetConfig, gpool: bool) -> Self {
    let residual_inner_channels = config.residual_inner_channels();
    Self {
      normactconv1: NormActConv::new(
        device,
        false,
        gpool.then_some(config.gpool_channels),
        residual_inner_channels,
        residual_inner_channels,
        [3, 3],
      ),
      normactconv2: NormActConv::new(
        device,
        true,
        None,
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
  fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    self.normactconv1.initialize(fixup_scale, device);
    self.normactconv2.initialize(0.0, device);
  }

  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>) -> Tensor<4> {
    let out = self
      .normactconv1
      .forward(inputs.clone(), mask.clone(), mask_sum_hw.clone());
    let out = self.normactconv2.forward(out, mask, mask_sum_hw);
    inputs + out
  }
}

#[derive(Module, Debug)]
pub struct ResidualBlock {
  pub(crate) normactconvp: NormActConv,
  pub(crate) inner: Vec<InnerResidualBlock>,
  pub(crate) normactconvq: NormActConv,
}

impl ResidualBlock {
  pub fn new(device: &Device, config: &ResNetConfig, gpool: bool) -> Self {
    Self {
      normactconvp: NormActConv::new(
        device,
        false,
        None,
        config.inner_channels,
        config.residual_inner_channels(),
        [1, 1],
      ),
      inner: (0..config.residual_size)
        .map(|i| InnerResidualBlock::new(device, config, gpool && i == 0))
        .collect(),
      normactconvq: NormActConv::new(
        device,
        true,
        None,
        config.residual_inner_channels(),
        config.inner_channels,
        [1, 1],
      ),
    }
  }
}

impl TrunkBlock for ResidualBlock {
  type Context = ();

  fn context(&self, _mask: &Tensor<4>) {}

  /// Each of the `1 + residual_size` stages gets the geometric share
  /// `fixup_scale^(1/(1+residual_size))` of the block's scale, and the final `1x1`
  /// conv is zero-initialized so the whole nested block starts as the identity.
  fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    let inner_scale = fixup_scale.powf(1.0 / (1.0 + self.inner.len() as f64));
    self.normactconvp.initialize(inner_scale, device);
    for inner in &mut self.inner {
      inner.initialize(inner_scale, device);
    }
    self.normactconvq.initialize(0.0, device);
  }

  fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, mask_sum_hw: Tensor<4>, _context: &()) -> Tensor<4> {
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
