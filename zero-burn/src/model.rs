use burn::{
  module::Module,
  nn::{
    conv::{Conv2d, Conv2dConfig},
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, ReLU,
  },
  optim::{GradientsParams, Optimizer},
  tensor::{
    activation::log_softmax,
    backend::{ADBackend, Backend},
    Data, Tensor,
  },
};
use ndarray::{Array1, Array3, Array4, ShapeError};
use num_traits::{Float, NumCast};
use oppai_zero::{
  field_features::CHANNELS,
  model::{Model as OppaiModel, TrainableModel as OppaiTrainableModel},
};

const INPUT_CHANNELS: usize = CHANNELS;
const INNER_CHANNELS: usize = 256;
const RESIDUAL_BLOCKS: usize = 19;
const POLICY_CHANNELS: usize = 2;
const VALUE_CHANNELS: usize = 1;
const VALUE_HIDDEN_SIZE: usize = 256;

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
  conv1: Conv2d<B>,
  bn1: BatchNorm<B, 2>,
  conv2: Conv2d<B>,
  bn2: BatchNorm<B, 2>,
  activation: ReLU,
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
}

impl<B: Backend> Default for ResidualBlock<B> {
  fn default() -> Self {
    Self {
      conv1: Conv2dConfig::new([INNER_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(),
      bn1: BatchNormConfig::new(INNER_CHANNELS).init(),
      conv2: Conv2dConfig::new([INNER_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(),
      bn2: BatchNormConfig::new(INNER_CHANNELS).init(),
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
  activation: ReLU,
}

impl<B: Backend> Model<B> {
  pub fn new(width: u32, height: u32) -> Self {
    let length = (width * height) as usize;
    Self {
      initial_conv: Conv2dConfig::new([INPUT_CHANNELS, INNER_CHANNELS], [3, 3])
        .with_padding(PaddingConfig2d::Same)
        .init(),
      initial_bn: BatchNormConfig::new(INNER_CHANNELS).init(),
      residuals: vec![Default::default(); RESIDUAL_BLOCKS],
      policy_conv: Conv2dConfig::new([INNER_CHANNELS, POLICY_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(),
      policy_bn: BatchNormConfig::new(POLICY_CHANNELS).init(),
      policy_fc: LinearConfig::new(POLICY_CHANNELS * length, length).init(),
      value_conv: Conv2dConfig::new([INNER_CHANNELS, VALUE_CHANNELS], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .init(),
      value_bn: BatchNormConfig::new(VALUE_CHANNELS).init(),
      value_fc1: LinearConfig::new(VALUE_CHANNELS * length, VALUE_HIDDEN_SIZE).init(),
      value_fc2: LinearConfig::new(VALUE_HIDDEN_SIZE, 1).init(),
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

struct Learner<B, O>
where
  B: ADBackend,
{
  model: Model<B>,
  optimizer: O,
}

impl<B> OppaiModel<<B as Backend>::FloatElem> for Model<B>
where
  B: Backend,
  <B as Backend>::FloatElem: Float,
{
  type E = ShapeError;

  fn predict(
    &self,
    inputs: Array4<<B as Backend>::FloatElem>,
  ) -> Result<(Array3<<B as Backend>::FloatElem>, Array1<<B as Backend>::FloatElem>), Self::E> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(Data::new(
      if inputs.is_standard_layout() {
        inputs.into_raw_vec()
      } else {
        inputs.as_standard_layout().to_owned().into_raw_vec()
      },
      [batch, channels, height, width].into(),
    ));
    let (policies, values) = self.forward(inputs);
    let policies = Array3::from_shape_vec((batch, height, width), policies.into_data().value)?;
    let values = Array1::from_vec(values.into_data().value);
    Ok((policies, values))
  }
}

impl<B, O> OppaiModel<<B as Backend>::FloatElem> for Learner<B, O>
where
  B: Backend + ADBackend,
  <B as Backend>::FloatElem: Float,
{
  type E = ShapeError;

  fn predict(
    &self,
    inputs: Array4<<B as Backend>::FloatElem>,
  ) -> Result<(Array3<<B as Backend>::FloatElem>, Array1<<B as Backend>::FloatElem>), Self::E> {
    self.model.predict(inputs)
  }
}

impl<B, O> OppaiTrainableModel<<B as Backend>::FloatElem> for Learner<B, O>
where
  B: Backend + ADBackend,
  <B as Backend>::FloatElem: Float,
  O: Optimizer<Model<B>, B>,
{
  type TE = ShapeError;

  fn train(
    mut self,
    inputs: Array4<<B as Backend>::FloatElem>,
    policies: Array3<<B as Backend>::FloatElem>,
    values: Array1<<B as Backend>::FloatElem>,
  ) -> Result<Self, Self::TE> {
    let (batch, channels, height, width) = inputs.dim();
    let inputs = Tensor::from_data(Data::new(
      if inputs.is_standard_layout() {
        inputs.into_raw_vec()
      } else {
        inputs.as_standard_layout().to_owned().into_raw_vec()
      },
      [batch, channels, height, width].into(),
    ));
    let policies = Tensor::from_data(Data::new(
      if policies.is_standard_layout() {
        policies.into_raw_vec()
      } else {
        policies.as_standard_layout().to_owned().into_raw_vec()
      },
      [batch, height, width].into(),
    ));
    let values = Tensor::from_data(Data::new(values.into_raw_vec(), [batch].into()));
    let (out_policies, out_values) = self.model.forward(inputs);

    let batch = <<B as Backend>::FloatElem as NumCast>::from(batch).unwrap();
    let values_loss = (out_values - values).powf(2.0).sum() / batch;
    let policies_loss = -(out_policies * policies).sum() / batch;
    let loss = values_loss + policies_loss;

    let grads = GradientsParams::from_grads(loss.backward(), &self.model);
    self.model = self.optimizer.step(0.01, self.model, grads);

    Ok(self)
  }
}

#[cfg(test)]
mod tests {
  use super::{Learner, Model};
  use burn::{
    autodiff::ADBackendDecorator,
    backend::{NdArrayBackend, WgpuBackend},
    optim::SgdConfig,
    tensor::Tensor,
  };
  use ndarray::{Array, Array3, Array4, Axis};
  use oppai_zero::{
    field_features::CHANNELS,
    model::{Model as OppaiModel, TrainableModel},
  };

  #[test]
  fn forward() {
    let model = Model::<NdArrayBackend>::new(4, 8);
    let (policies, values) = model.forward(Tensor::ones([1, CHANNELS, 4, 8]));
    let policies = policies.exp().into_primitive().array;
    let values = values.into_primitive().array;
    assert!(policies.iter().all(|p| (0.0..=1.0).contains(p)));
    assert!(policies.axis_iter(Axis(0)).all(|p| (p.sum() - 1.0) < 0.001));
    assert!(values.iter().all(|v| (-1.0..=1.0).contains(v)));
  }

  macro_rules! predict_test {
    ($name:ident, $backend:ty) => {
      #[test]
      fn $name() {
        let model = Model::<$backend>::new(8, 4);
        model
          .predict(Array4::from_elem((1, CHANNELS, 4, 8), 1.0))
          .unwrap();
      }
    };
  }

  predict_test!(predict_ndarray, NdArrayBackend);
  predict_test!(predict_wgpu, WgpuBackend);

  macro_rules! train_test {
    ($name:ident, $backend:ty) => {
      #[test]
      fn $name() {
        let model = Model::<ADBackendDecorator<$backend>>::new(8, 4);
        let optimizer = SgdConfig::new().init::<ADBackendDecorator<$backend>, Model<_>>();
        let learner = Learner { model, optimizer };

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

  train_test!(train_ndarray, NdArrayBackend);
  train_test!(train_wgpu, WgpuBackend);
}
