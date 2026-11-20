//! The convolutional trunk: nested bottleneck residual blocks of `3x3`
//! convolutions, some of which also pool the board globally or over the chains
//! of dots.

use crate::{
  chains::{ChainPool, Chains},
  model::{GpoolConfig, NbtConfig, NormActConv},
};
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
  /// With `gpool`, the first convolution pools part of its channels over the
  /// board.
  pub fn new(device: &Device, channels: usize, gpool: Option<GpoolConfig>) -> Self {
    Self {
      normactconv1: NormActConv::new(device, false, gpool, channels, channels, [3, 3]),
      normactconv2: NormActConv::new(
        device,
        true,
        None,
        channels - gpool.map_or(0, |gpool| gpool.channels),
        channels,
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
  /// Pooling over the chains of dots after the first inner block, if any.
  pub(crate) chainpool: Option<ChainPool>,
  pub(crate) normactconvq: NormActConv,
}

impl ResidualBlock {
  pub fn new(device: &Device, inner_channels: usize, config: &NbtConfig) -> Self {
    assert!(config.size > 0, "a nested bottleneck block needs an inner block");
    Self {
      normactconvp: NormActConv::new(device, false, None, inner_channels, config.mid_channels, [1, 1]),
      inner: (0..config.size)
        .map(|i| InnerResidualBlock::new(device, config.mid_channels, config.gpool.filter(|_| i == 0)))
        .collect(),
      chainpool: config
        .chains
        .map(|chains| ChainPool::new(device, config.mid_channels, chains)),
      normactconvq: NormActConv::new(device, true, None, config.mid_channels, inner_channels, [1, 1]),
    }
  }

  /// Each of the `1 + size` stages gets the geometric share
  /// `fixup_scale^(1/(1+size))` of the block's scale, and the final `1x1`
  /// conv is zero-initialized so the whole nested block starts as the identity.
  pub(crate) fn initialize(&mut self, fixup_scale: f64, device: &Device) {
    let inner_scale = fixup_scale.powf(1.0 / (1.0 + self.inner.len() as f64));
    self.normactconvp.initialize(inner_scale, device);
    for inner in &mut self.inner {
      inner.initialize(inner_scale, device);
    }
    if let Some(chainpool) = &mut self.chainpool {
      chainpool.initialize(inner_scale, device);
    }
    self.normactconvq.initialize(0.0, device);
  }

  /// `chains` are the batch's chains, which a block with chain pooling needs.
  pub fn forward(
    &self,
    inputs: Tensor<4>,
    mask: Tensor<4>,
    mask_sum_hw: Tensor<4>,
    chains: Option<&Chains>,
  ) -> Tensor<4> {
    let mut out = self
      .normactconvp
      .forward(inputs.clone(), mask.clone(), mask_sum_hw.clone());
    for (i, inner) in self.inner.iter().enumerate() {
      out = inner.forward(out, mask.clone(), mask_sum_hw.clone());
      if i == 0
        && let Some(chainpool) = &self.chainpool
      {
        let chains = chains.expect("a block pooling over chains needs the chains");
        out = out.clone() + chainpool.forward(out, mask.clone(), chains);
      }
    }
    let out = self.normactconvq.forward(out, mask, mask_sum_hw);
    inputs + out
  }
}
