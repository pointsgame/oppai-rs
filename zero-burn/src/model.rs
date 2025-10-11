use burn::{
  module::{Module, Param},
  nn::{
    Gelu, Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
  },
  optim::{GradientsParams, Optimizer},
  tensor::{
    DataError, Tensor, TensorData,
    activation::log_softmax,
    backend::{AutodiffBackend, Backend},
    s,
  },
};
use derive_more::From;
use ndarray::{Array, Array1, Array3, Array4, Dimension, ShapeError};
use num_traits::{Float, NumCast};
use oppai_zero::{
  field_features::CHANNELS,
  model::{Model as OppaiModel, TrainableModel as OppaiTrainableModel},
};
use thiserror::Error;

const INPUT_CHANNELS: usize = CHANNELS;
// KataGo uses 512, but we don't have so many features.
const INNER_CHANNELS: usize = 256;
const RESIDUAL_INNER_CHANNELS: usize = INNER_CHANNELS / 2;
const RESIDUAL_BLOCKS: usize = 28;
const RESIDUAL_SIZE: usize = 2;
const GPOOL_CHANNELS: usize = 64;
const V1_CHANNELS: usize = 128;
const P1_CHANNELS: usize = 64;
const G1_CHANNELS: usize = 64;

#[derive(Module, Debug)]
pub struct NormMask<B: Backend> {
  beta: Param<Tensor<B, 4>>,
  gamma: Option<Param<Tensor<B, 4>>>,
}

impl<B: Backend> NormMask<B> {
  pub fn new(device: &B::Device, channels: usize, gamma: bool) -> Self {
    Self {
      beta: Param::from_tensor(Tensor::zeros([1, channels, 1, 1], device)),
      gamma: if gamma {
        Some(Param::from_tensor(Tensor::ones([1, channels, 1, 1], device)))
      } else {
        None
      },
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>) -> Tensor<B, 4> {
    match self.gamma {
      Some(ref gamma) => (inputs * gamma.val() + self.beta.val()) * mask,
      None => (inputs + self.beta.val()) * mask,
    }
  }
}

#[derive(Module, Debug)]
pub struct ConvAndGPool<B: Backend> {
  conv1r: Conv2d<B>,
  conv1g: Conv2d<B>,
  normg: NormMask<B>,
  actg: Gelu,
  linearg: Linear<B>,
}

impl<B: Backend> ConvAndGPool<B> {
  pub fn new(device: &B::Device) -> Self {
    Self {
      conv1r: Conv2dConfig::new(
        [RESIDUAL_INNER_CHANNELS, RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS],
        [3, 3],
      )
      .with_padding(PaddingConfig2d::Same)
      .init(device),
      conv1g: Conv2dConfig::new([RESIDUAL_INNER_CHANNELS, GPOOL_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      normg: NormMask::new(device, GPOOL_CHANNELS, false),
      actg: Gelu::new(),
      linearg: LinearConfig::new(3 * GPOOL_CHANNELS, RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS).init(device),
    }
  }

  fn gpool(inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let mask_sum_hw_sqrt_offset = mask_sum_hw.clone().sqrt() - 14.0;

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
    let outg = self.actg.forward(outg);
    let outg = Self::gpool(outg, mask, mask_sum_hw);
    let outg = self
      .linearg
      .forward(outg.squeeze_dims::<2>(&[2, 3]))
      .unsqueeze_dims(&[-1, -1]);
    outr + outg
  }
}

#[derive(Module, Debug)]
pub enum ConvOrGpool<B: Backend> {
  Conv(Conv2d<B>),
  Gpool(ConvAndGPool<B>),
}

impl<B: Backend> ConvOrGpool<B> {
  pub fn new(
    device: &B::Device,
    gpool: bool,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    if gpool {
      Self::Gpool(ConvAndGPool::new(device))
    } else {
      Self::Conv(
        Conv2dConfig::new([in_channels, out_channels], kernel_size)
          .with_padding(PaddingConfig2d::Same)
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
  act: Gelu,
  convgpool: ConvOrGpool<B>,
}

impl<B: Backend> NormActConv<B> {
  pub fn new(
    device: &B::Device,
    gamma: bool,
    gpool: bool,
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
  ) -> Self {
    Self {
      norm: NormMask::new(device, in_channels, gamma),
      act: Gelu::new(),
      convgpool: ConvOrGpool::new(device, gpool, in_channels, out_channels, kernel_size),
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let out = self.norm.forward(inputs, mask.clone());
    let out = self.act.forward(out);
    self.convgpool.forward(out, mask, mask_sum_hw)
  }
}

#[derive(Module, Debug)]
pub struct InnerResidualBlock<B: Backend> {
  normactconv1: NormActConv<B>,
  normactconv2: NormActConv<B>,
}

impl<B: Backend> InnerResidualBlock<B> {
  pub fn new(device: &B::Device, gpool: bool) -> Self {
    Self {
      normactconv1: NormActConv::new(
        device,
        false,
        gpool,
        RESIDUAL_INNER_CHANNELS,
        RESIDUAL_INNER_CHANNELS,
        [3, 3],
      ),
      normactconv2: NormActConv::new(
        device,
        true,
        false,
        if gpool {
          RESIDUAL_INNER_CHANNELS - GPOOL_CHANNELS
        } else {
          RESIDUAL_INNER_CHANNELS
        },
        RESIDUAL_INNER_CHANNELS,
        [3, 3],
      ),
    }
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
  pub fn new(device: &B::Device, gpool: bool) -> Self {
    Self {
      normactconvp: NormActConv::new(device, false, false, INNER_CHANNELS, RESIDUAL_INNER_CHANNELS, [1, 1]),
      inner: (0..RESIDUAL_SIZE)
        .map(|i| InnerResidualBlock::new(device, gpool && i == 0))
        .collect(),
      normactconvq: NormActConv::new(device, true, false, RESIDUAL_INNER_CHANNELS, INNER_CHANNELS, [1, 1]),
    }
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

#[derive(Module, Debug)]
pub struct ValueHead<B: Backend> {
  conv1: Conv2d<B>,
  bias1: NormMask<B>,
  act1: Gelu,
  linear2: Linear<B>,
}

impl<B: Backend> ValueHead<B> {
  pub fn new(device: &B::Device) -> Self {
    Self {
      conv1: Conv2dConfig::new([INNER_CHANNELS, V1_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      bias1: NormMask::new(device, V1_CHANNELS, false),
      act1: Gelu::new(),
      linear2: LinearConfig::new(3 * V1_CHANNELS, 1).init(device),
    }
  }

  fn gpool(inputs: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 4> {
    let mask_sum_hw_sqrt_offset = mask_sum_hw.clone().sqrt() - 141.0;

    let layer_mean = inputs.clone().sum_dim(2).sum_dim(3) / mask_sum_hw;

    let out_pool1 = layer_mean.clone();
    let out_pool2 = layer_mean.clone() * (mask_sum_hw_sqrt_offset.clone() / 10.0);
    let out_pool3 = layer_mean * (mask_sum_hw_sqrt_offset.clone() * mask_sum_hw_sqrt_offset / 100.0 - 0.1);

    Tensor::cat(vec![out_pool1, out_pool2, out_pool3], 1)
  }

  pub fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 1> {
    let outv1 = self.conv1.forward(inputs);
    let outv1 = self.bias1.forward(outv1, mask);
    let outv1 = self.act1.forward(outv1);
    let outpooled = Self::gpool(outv1, mask_sum_hw).reshape([0, -1]);
    let outv2 = self.linear2.forward(outpooled).squeeze(1);
    outv2.tanh()
  }
}

#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
  conv1p: Conv2d<B>,
  conv1g: Conv2d<B>,
  biasg: NormMask<B>,
  actg: Gelu,
  linearg: Linear<B>,
  bias2: NormMask<B>,
  act2: Gelu,
  conv2p: Conv2d<B>,
}

impl<B: Backend> PolicyHead<B> {
  pub fn new(device: &B::Device) -> Self {
    Self {
      conv1p: Conv2dConfig::new([INNER_CHANNELS, P1_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      conv1g: Conv2dConfig::new([INNER_CHANNELS, G1_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      biasg: NormMask::new(device, G1_CHANNELS, false),
      actg: Gelu::new(),
      linearg: LinearConfig::new(3 * G1_CHANNELS, P1_CHANNELS).init(device),
      bias2: NormMask::new(device, P1_CHANNELS, false),
      act2: Gelu::new(),
      conv2p: Conv2dConfig::new([P1_CHANNELS, 1], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
    }
  }

  fn forward(&self, inputs: Tensor<B, 4>, mask: Tensor<B, 4>, mask_sum_hw: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, _, height, width] = inputs.dims();

    let outp = self.conv1p.forward(inputs.clone());
    let outg = self.conv1g.forward(inputs);
    let outg = self.biasg.forward(outg, mask.clone());
    let outg = self.actg.forward(outg);
    let outg = ConvAndGPool::<B>::gpool(outg, mask.clone(), mask_sum_hw).reshape([0, -1]);
    let outg = self.linearg.forward(outg).unsqueeze_dims(&[-1, -1]);

    let outp = outp + outg;
    let outp = self.bias2.forward(outp, mask.clone());
    let outp = self.act2.forward(outp);
    let outp = self.conv2p.forward(outp);
    let outp: Tensor<B, 4> = outp - (1.0 - mask) * 5000.0;
    let outp = log_softmax(outp.reshape([0, -1]), 1);
    outp.reshape([batch, height, width])
  }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
  initial_conv: Conv2d<B>,
  residuals: Vec<ResidualBlock<B>>,
  value_head: ValueHead<B>,
  policy_head: PolicyHead<B>,
}

impl<B: Backend> Model<B> {
  pub fn new(device: &B::Device) -> Self {
    Self {
      initial_conv: Conv2dConfig::new([INPUT_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      residuals: (0..RESIDUAL_BLOCKS)
        .map(|i| ResidualBlock::new(device, (i + 1) % 3 == 0))
        .collect(),
      value_head: ValueHead::new(device),
      policy_head: PolicyHead::new(device),
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let mask = inputs.clone().slice(s![.., 0..1]);
    let mask_sum_hw = mask.clone().sum_dim(2).sum_dim(3);
    let mut x = self.initial_conv.forward(inputs);
    for residual in &self.residuals {
      x = residual.forward(x, mask.clone(), mask_sum_hw.clone());
    }
    let policy = self.policy_head.forward(x.clone(), mask.clone(), mask_sum_hw.clone());
    let value = self.value_head.forward(x, mask, mask_sum_hw);
    (policy, value)
  }
}

pub struct Predictor<B: Backend> {
  pub model: Model<B>,
  pub device: B::Device,
}

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
}

fn into_data_vec<A: Clone, D: Dimension>(array: Array<A, D>) -> Vec<A> {
  let (mut vec, offset) = if array.is_standard_layout() {
    array.into_raw_vec_and_offset()
  } else {
    array.as_standard_layout().to_owned().into_raw_vec_and_offset()
  };
  if let Some(offset) = offset {
    vec.drain(0..offset);
  }
  vec
}

impl<B> OppaiModel<<B as Backend>::FloatElem> for Predictor<B>
where
  B: Backend,
  <B as Backend>::FloatElem: Float,
{
  type E = ModelError;

  fn predict(
    &self,
    inputs: Array4<<B as Backend>::FloatElem>,
  ) -> Result<(Array3<<B as Backend>::FloatElem>, Array1<<B as Backend>::FloatElem>), Self::E> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.device,
    );
    let (policies, values) = self.model.forward(inputs);
    let policies = Array3::from_shape_vec((batch, height, width), policies.into_data().into_vec()?)?;
    let values = Array1::from_vec(values.into_data().into_vec()?);
    Ok((policies, values))
  }
}

impl<B, O> OppaiModel<<B as Backend>::FloatElem> for Learner<B, O>
where
  B: Backend + AutodiffBackend,
  <B as Backend>::FloatElem: Float,
{
  type E = ModelError;

  fn predict(
    &self,
    inputs: Array4<<B as Backend>::FloatElem>,
  ) -> Result<(Array3<<B as Backend>::FloatElem>, Array1<<B as Backend>::FloatElem>), Self::E> {
    self.predictor.predict(inputs)
  }
}

impl<B, O> OppaiTrainableModel<<B as Backend>::FloatElem> for Learner<B, O>
where
  B: Backend + AutodiffBackend,
  <B as Backend>::FloatElem: Float,
  O: Optimizer<Model<B>, B>,
{
  type TE = ModelError;

  fn train(
    mut self,
    inputs: Array4<<B as Backend>::FloatElem>,
    policies: Array3<<B as Backend>::FloatElem>,
    values: Array1<<B as Backend>::FloatElem>,
  ) -> Result<Self, Self::TE> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(
      TensorData::new(into_data_vec(inputs), [batch, channels, height, width]),
      &self.predictor.device,
    );
    let policies = Tensor::from_data(
      TensorData::new(into_data_vec(policies), [batch, height, width]),
      &self.predictor.device,
    );
    let values = Tensor::from_data(TensorData::new(into_data_vec(values), [batch]), &self.predictor.device);
    let (out_policies, out_values) = self.predictor.model.forward(inputs);

    let batch = <<B as Backend>::FloatElem as NumCast>::from(batch).unwrap();
    let values_loss = (out_values - values)
      .powf(Tensor::from_data(
        [<<B as Backend>::FloatElem as NumCast>::from(2.0).unwrap()],
        &self.predictor.device,
      ))
      .sum()
      / batch;
    let policies_loss = -(out_policies * policies).sum() / batch;
    let loss = values_loss + policies_loss;

    log::info!("Loss: {}", loss.clone().into_scalar());

    let grads = GradientsParams::from_grads(loss.backward(), &self.predictor.model);
    self.predictor.model = self.optimizer.step(0.0001, self.predictor.model, grads);

    Ok(self)
  }
}

#[cfg(test)]
mod tests {
  use super::{Learner, Model, Predictor};
  use burn::{
    backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice},
    optim::SgdConfig,
    tensor::Tensor,
  };
  use ndarray::{Array, Array3, Array4};
  use oppai_zero::{
    field_features::CHANNELS,
    model::{Model as OppaiModel, TrainableModel},
  };

  #[test]
  fn forward() {
    let model = Model::<NdArray>::new(&NdArrayDevice::Cpu);
    let (policies, values) = model.forward(Tensor::ones([1, CHANNELS, 4, 8], &NdArrayDevice::Cpu));
    let policies = policies.exp();
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

  macro_rules! predict_test {
    ($name:ident, $backend:ty, $device:expr) => {
      #[test]
      fn $name() {
        let model = Model::<$backend>::new(&$device);
        let predictor = Predictor {
          model,
          device: $device,
        };
        predictor
          .predict(Array4::from_elem((1, CHANNELS, 4, 8), 1.0))
          .unwrap();
      }
    };
  }

  predict_test!(predict_ndarray, NdArray, NdArrayDevice::Cpu);
  predict_test!(predict_wgpu, Wgpu, WgpuDevice::DefaultDevice);

  macro_rules! train_test {
    ($name:ident, $backend:ty, $device:expr) => {
      #[test]
      fn $name() {
        let model = Model::<Autodiff<$backend>>::new(&$device);
        let predictor = Predictor {
          model,
          device: $device,
        };
        let optimizer = SgdConfig::new().init::<Autodiff<$backend>, Model<_>>();
        let learner = Learner { predictor, optimizer };

        let inputs = Array4::from_elem((1, CHANNELS, 4, 8), 1.0);
        let policies = Array3::from_elem((1, 4, 8), 0.5);
        let values = Array::from_elem(1, 0.5);

        let (out_policies_1, out_values_1) = learner.predict(inputs.clone()).unwrap();
        let learner = learner.train(inputs.clone(), policies, values).unwrap();
        let (out_policies_2, out_values_2) = learner.predict(inputs).unwrap();

        assert!((out_policies_1 - out_policies_2).iter().all(|v| v.abs() > 0.0));
        assert!((out_values_1 - out_values_2).iter().all(|v| v.abs() > 0.0));
      }
    };
  }

  train_test!(train_ndarray, NdArray, NdArrayDevice::Cpu);
  train_test!(train_wgpu, Wgpu, WgpuDevice::DefaultDevice);
}
