use burn::{
  module::Module,
  nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
  },
  optim::{GradientsParams, Optimizer},
  tensor::{
    DataError, Tensor, TensorData,
    activation::log_softmax,
    backend::{AutodiffBackend, Backend},
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
const INNER_CHANNELS: usize = 32; // AlphaGo uses 256
const RESIDUAL_BLOCKS: usize = 5; // AlphaGo uses 19 or 39
const POLICY_CHANNELS: usize = 2;
const VALUE_CHANNELS: usize = 1;
const VALUE_HIDDEN_SIZE: usize = 256;

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
  conv1: Conv2d<B>,
  bn1: BatchNorm<B, 2>,
  conv2: Conv2d<B>,
  bn2: BatchNorm<B, 2>,
  activation: Relu,
}

impl<B: Backend> ResidualBlock<B> {
  pub fn forward(&self, inputs: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv1.forward(inputs.clone());
    let x = self.bn1.forward(x);
    let x = self.activation.forward(x);
    let x = self.conv2.forward(x);
    let x = self.bn2.forward(x);
    self.activation.forward(inputs + x)
  }

  pub fn new(device: &B::Device) -> Self {
    Self {
      conv1: Conv2dConfig::new([INNER_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      bn1: BatchNormConfig::new(INNER_CHANNELS).init(device),
      conv2: Conv2dConfig::new([INNER_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      bn2: BatchNormConfig::new(INNER_CHANNELS).init(device),
      activation: Default::default(),
    }
  }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
  initial_conv: Conv2d<B>,
  initial_bn: BatchNorm<B, 2>,
  residuals: Vec<ResidualBlock<B>>,
  policy_conv: Conv2d<B>,
  policy_bn: BatchNorm<B, 2>,
  policy_fc: Linear<B>,
  value_conv: Conv2d<B>,
  value_bn: BatchNorm<B, 2>,
  value_fc1: Linear<B>,
  value_fc2: Linear<B>,
  activation: Relu,
}

impl<B: Backend> Model<B> {
  pub fn new(width: u32, height: u32, device: &B::Device) -> Self {
    let length = (width * height) as usize;
    Self {
      initial_conv: Conv2dConfig::new([INPUT_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      initial_bn: BatchNormConfig::new(INNER_CHANNELS).init(device),
      residuals: vec![ResidualBlock::new(device); RESIDUAL_BLOCKS],
      policy_conv: Conv2dConfig::new([INNER_CHANNELS, POLICY_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      policy_bn: BatchNormConfig::new(POLICY_CHANNELS).init(device),
      policy_fc: LinearConfig::new(POLICY_CHANNELS * length, length).init(device),
      value_conv: Conv2dConfig::new([INNER_CHANNELS, VALUE_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(device),
      value_bn: BatchNormConfig::new(VALUE_CHANNELS).init(device),
      value_fc1: LinearConfig::new(VALUE_CHANNELS * length, VALUE_HIDDEN_SIZE).init(device),
      value_fc2: LinearConfig::new(VALUE_HIDDEN_SIZE, 1).init(device),
      activation: Default::default(),
    }
  }

  pub fn forward(&self, inputs: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 1>) {
    let [batch, _, height, width] = inputs.dims();

    let x = self.initial_conv.forward(inputs);
    let x = self.initial_bn.forward(x);
    let mut x = self.activation.forward(x);
    for residual in &self.residuals {
      x = residual.forward(x);
    }

    let p = self.policy_conv.forward(x.clone());
    let p = self.policy_bn.forward(p);
    let p = self.activation.forward(p);
    let p = p.reshape([batch, POLICY_CHANNELS * height * width]);
    let p = self.policy_fc.forward(p);
    let p = log_softmax(p, 1);
    let p = p.reshape([batch, height, width]);

    let v = self.value_conv.forward(x);
    let v = self.value_bn.forward(v);
    let v = self.activation.forward(v);
    let v = v.reshape([batch, VALUE_CHANNELS * height * width]);
    let v = self.value_fc1.forward(v);
    let v = self.activation.forward(v);
    let v = self.value_fc2.forward(v);
    let v = v.tanh();
    let v = v.reshape([batch]);

    (p, v)
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
    self.predictor.model = self.optimizer.step(0.01, self.predictor.model, grads);

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
    let model = Model::<NdArray>::new(4, 8, &NdArrayDevice::Cpu);
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
        let model = Model::<$backend>::new(8, 4, &$device);
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
        let model = Model::<Autodiff<$backend>>::new(8, 4, &$device);
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
