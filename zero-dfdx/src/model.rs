use ::safetensors::SafeTensorError;
use dfdx::{optim::Adam, prelude::*};
use ndarray::{Array1, Array3, Array4, Axis, ShapeError};
use num_traits::Float;
use oppai_zero::{
  field_features::CHANNELS,
  model::{Model, TrainableModel},
};
use thiserror::Error;

const WIDTH: usize = 8;
const HEIGHT: usize = 8;
const LENGTH: usize = WIDTH * HEIGHT;
const INNER_CHANNELS: usize = 8;
const KERNEL_SIZE: usize = 3;
const LINEAR_INPUT: usize = INNER_CHANNELS * (WIDTH - 4) * (HEIGHT - 4);
const LINEAR_FIRST: usize = 1024;
const LINEAR_SECOND: usize = 512;

#[derive(Default, Debug, Clone, Copy)]
pub struct LogSoftmax;

impl ZeroSizedModule for LogSoftmax {}
impl NonMutableModule for LogSoftmax {}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<S, E, D, T>> for LogSoftmax {
  type Output = Tensor<S, E, D, T>;
  type Error = D::Err;

  fn try_forward(&self, input: Tensor<S, E, D, T>) -> Result<Self::Output, D::Err> {
    input.try_log_softmax::<S::LastAxis>()
  }
}

type DfdxModule = (
  (
    (
      Conv2D<CHANNELS, INNER_CHANNELS, KERNEL_SIZE, 1, 1>,
      BatchNorm2D<INNER_CHANNELS>,
      ReLU,
    ),
    (
      Conv2D<INNER_CHANNELS, INNER_CHANNELS, KERNEL_SIZE, 1, 1>,
      BatchNorm2D<INNER_CHANNELS>,
      ReLU,
    ),
    (
      Conv2D<INNER_CHANNELS, INNER_CHANNELS, KERNEL_SIZE>,
      BatchNorm2D<INNER_CHANNELS>,
      ReLU,
    ),
    (
      Conv2D<INNER_CHANNELS, INNER_CHANNELS, KERNEL_SIZE>,
      BatchNorm2D<INNER_CHANNELS>,
      ReLU,
    ),
  ),
  Flatten2D,
  (
    (
      Linear<LINEAR_INPUT, LINEAR_FIRST>,
      BatchNorm1D<LINEAR_FIRST>,
      DropoutOneIn<3>,
    ),
    (
      Linear<LINEAR_FIRST, LINEAR_SECOND>,
      BatchNorm1D<LINEAR_SECOND>,
      DropoutOneIn<3>,
    ),
  ),
  SplitInto<(
    (Linear<LINEAR_SECOND, LENGTH>, LogSoftmax),
    (Linear<LINEAR_SECOND, 1>, Tanh),
  )>,
);

type InputShape = (
  usize, // batch size
  Const<CHANNELS>,
  Const<WIDTH>,
  Const<HEIGHT>,
);

type PolicyShape = (
  usize, // batch size
  Const<LENGTH>,
);

type ValueShape = (
  usize, // batch size
  Const<1>,
);

pub struct DfdxModel<N>
where
  N: Float + Dtype,
  AutoDevice: Device<N>,
{
  device: AutoDevice,
  model: <DfdxModule as BuildOnDevice<AutoDevice, N>>::Built,
  adam: Adam<<DfdxModule as BuildOnDevice<AutoDevice, N>>::Built, N, AutoDevice>,
}

impl<N> Default for DfdxModel<N>
where
  N: Float + Dtype,
  AutoDevice: Device<N>,
{
  fn default() -> Self {
    let device = AutoDevice::default();
    let model = device.build_module::<DfdxModule, N>();
    let adam = Adam::new(&model, Default::default());
    Self { device, model, adam }
  }
}

impl<N> Clone for DfdxModel<N>
where
  N: Float + Dtype,
  AutoDevice: Device<N>,
{
  fn clone(&self) -> Self {
    let device = self.device.clone();
    let model = self.model.clone();
    let adam = Adam::new(&model, Default::default());
    Self { device, model, adam }
  }
}

#[derive(Clone, Error, Debug)]
pub enum PredictError {
  #[error("Ndarray error")]
  Ndarray(ShapeError),
  #[error("Dfdx error")]
  Dfdx(<AutoDevice as HasErr>::Err),
}

impl From<ShapeError> for PredictError {
  fn from(value: ShapeError) -> Self {
    PredictError::Ndarray(value)
  }
}

impl Model<f32> for DfdxModel<f32> {
  type E = PredictError;

  fn predict(&self, inputs: Array4<f32>) -> Result<(Array3<f32>, Array1<f32>), Self::E> {
    let batch_size = inputs.len_of(Axis(0));
    let inputs: Tensor<InputShape, f32, _> = self
      .device
      .try_tensor_from_vec(inputs.into_raw_vec(), (batch_size, Const, Const, Const))
      .map_err(PredictError::Dfdx)?;

    let (policies, values) = self.model.forward(inputs);

    let policies = exp(policies);

    let policies = Array3::from_shape_vec((batch_size, WIDTH, HEIGHT), policies.as_vec())?;
    let values = Array1::from_shape_vec(batch_size, values.as_vec())?;

    Ok((policies, values))
  }
}

#[derive(Error, Debug)]
pub enum TrainError {
  #[error("Predict error")]
  Predict(PredictError),
  #[error("Train error")]
  Train(OptimizerUpdateError<AutoDevice>),
  #[error("Save error")]
  Save(SafeTensorError),
}

impl From<PredictError> for TrainError {
  fn from(value: PredictError) -> Self {
    TrainError::Predict(value)
  }
}

impl From<OptimizerUpdateError<AutoDevice>> for TrainError {
  fn from(value: OptimizerUpdateError<AutoDevice>) -> Self {
    TrainError::Train(value)
  }
}

impl From<SafeTensorError> for TrainError {
  fn from(value: SafeTensorError) -> Self {
    TrainError::Save(value)
  }
}

impl TrainableModel<f32> for DfdxModel<f32> {
  type TE = TrainError;

  fn train(&mut self, inputs: Array4<f32>, policies: Array3<f32>, values: Array1<f32>) -> Result<(), Self::TE> {
    let batch_size = inputs.len_of(Axis(0));
    let inputs: Tensor<InputShape, f32, _> = self
      .device
      .try_tensor_from_vec(inputs.into_raw_vec(), (batch_size, Const, Const, Const))
      .map_err(PredictError::Dfdx)?;
    let policies: Tensor<PolicyShape, f32, _> = self
      .device
      .try_tensor_from_vec(policies.into_raw_vec(), (batch_size, Const))
      .map_err(PredictError::Dfdx)?;
    let values: Tensor<ValueShape, f32, _> = self
      .device
      .try_tensor_from_vec(values.into_raw_vec(), (batch_size, Const))
      .map_err(PredictError::Dfdx)?;

    let mut grads = self.model.alloc_grads();

    let (out_policies, out_values) = self.model.forward_mut(inputs.traced(grads));

    let policy_loss = -(out_policies * policies).sum::<Rank0, _>() / batch_size as f32;
    let value_loss = (out_values - values).powi(2).sum::<Rank0, _>() / batch_size as f32;
    let loss = policy_loss + value_loss;

    grads = loss.backward();
    self.adam.update(&mut self.model, &grads)?;
    self.model.zero_grads(&mut grads);

    Ok(())
  }

  fn save(&self) -> Result<(), Self::TE> {
    self.model.save_safetensors("model.safetensors").map_err(From::from)
  }
}
