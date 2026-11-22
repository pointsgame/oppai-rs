//! Pooling over the chains of dots: every dot gets the mean of features over
//! the whole chain it is part of, and optionally a soft max of them, however
//! far the chain reaches, see [`ChainPool`].

use crate::model::{NormMask, conv_forward, init_conv, mish_gain};
use burn::{
  module::Module,
  nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
  },
  tensor::{DType, Device, FloatDType, Int, Tensor, activation::mish, s},
};
use oppai_zero::field_features::CHAIN_CHANNELS;
use serde::{Deserialize, Serialize};

/// Pooling over the chains of dots in a nested bottleneck block, see
/// [`ChainPool`].
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainsConfig {
  /// The channels pooled, for each player's chains.
  pub channels: usize,
  /// The temperature of a soft max of the pooled channels over every chain,
  /// next to their mean, if any: see [`Chains::pool`]. The higher, the closer
  /// to the max - at most `ln(size) / soft_max` below it.
  #[serde(default, skip_serializing_if = "Option::is_none")]
  pub soft_max: Option<f64>,
}

/// The players whose chains are pooled: the player to move and the opponent.
const PLAYERS: usize = 2;

/// The chains of each player pooled: the biggest ones, see `push_chains` of the
/// features. The dots of smaller chains get nothing from the pooling, as if
/// they were in no chain. The positions of training have about 8 chains a
/// player at most on average on a 24x24 board, and the biggest ones hold most
/// of the dots.
pub const MAX_CHAINS: usize = 64;

/// How the size of a chain enters the pooled features: its logarithm, scaled
/// down so that the biggest chains of a 39x32 board stay near one.
const SIZE_FEATURE_SCALE: f64 = 0.25;

/// The least mean of the exponentials of a soft max taken the logarithm of:
/// the least normal number of single precision and of bfloat16 alike. Only a
/// chain whose every value lies further below the board's max than about 87
/// over the temperature gets less - far beyond what the normalized channels
/// pooled reach at the temperatures of use - while the chains with no cells,
/// which reach no cell, get zero.
const MIN_MEAN_EXPONENTIAL: f64 = f32::MIN_POSITIVE as f64;

/// The chains of a batch, taken from the chain planes of its inputs, as the
/// matrices pooling multiplies by. They are built once per forward pass and
/// shared by every block that pools over chains.
///
/// Pooling is two batched matrix products with the one-hot membership of the
/// cells in [`MAX_CHAINS`] chains rather than a scatter to the chains and a
/// gather back: the matrices are small, and a matrix product runs as one fast
/// kernel where a scatter adds atomically, cell by cell.
pub struct Chains {
  /// `[batch, PLAYERS, cells, MAX_CHAINS]`: whether every cell is in each
  /// chain, divided by the size of the chain - which turns the sum over a
  /// chain into its mean. The player to move's chains first.
  mean: Tensor<4>,
  /// `[batch, PLAYERS, MAX_CHAINS, cells]`: whether every cell is in each
  /// chain.
  member: Tensor<4>,
  /// `[batch, PLAYERS, height, width]`: the logarithm of one plus the size of
  /// the chain every dot is in, scaled by [`SIZE_FEATURE_SCALE`], and zero
  /// elsewhere.
  size_feature: Tensor<4>,
}

impl Chains {
  /// From the `[batch, CHAIN_CHANNELS, height, width]` chain planes, which
  /// label every dot with the rank of its chain. Ranks are small numbers,
  /// exact in every float type.
  pub fn new(labels: Tensor<4>) -> Self {
    let [batch, channels, height, width] = labels.dims();
    assert_eq!(channels, CHAIN_CHANNELS, "a plane for each player's chains");
    assert_eq!(channels, PLAYERS);
    let cells = height * width;
    let device = labels.device();
    let labels = labels.cast(FloatDType::F32).reshape([batch, PLAYERS, 1, cells]);
    let ranks = Tensor::<1, Int>::arange(1..MAX_CHAINS as i64 + 1, &device)
      .float()
      .cast(FloatDType::F32)
      .reshape([1, 1, MAX_CHAINS, 1]);
    let member = labels.equal(ranks).float().cast(FloatDType::F32);
    let sizes = member.clone().sum_dim(3);
    let mean = (member.clone() / sizes.clone().clamp_min(1.0)).swap_dims(2, 3);
    let size_feature = (sizes + 1.0).log() * SIZE_FEATURE_SCALE;
    let size_feature = size_feature.swap_dims(2, 3).matmul(member.clone());
    Self {
      mean,
      member,
      size_feature: size_feature.reshape([batch, PLAYERS, height, width]),
    }
  }

  /// The mean of the `[batch, channels, height, width]` `features` over the
  /// chain of every cell - and with a `soft_max` temperature, their soft max
  /// over it after the means - for the player to move's chains and then for
  /// the opponent's, followed by the size feature of both:
  /// `[batch, PLAYERS * (channels + 1), height, width]`, or `[batch, PLAYERS *
  /// (2 * channels + 1), height, width]` with the soft max, zero on the cells
  /// outside the chains. Computed in the type of `features`, but with the soft
  /// max in bfloat16 for a half precision type, see below.
  ///
  /// The mean dilutes what holds at a single dot of a chain by the chain's
  /// size, while the max keeps it - but a max is no matrix product. The soft
  /// max is: the logarithm of the mean of the exponentials over the chain,
  /// divided by the temperature, `ln(mean(exp(t·x))) / t`. It lies between the
  /// mean and the max, at most `ln(size) / t` below the max.
  ///
  /// The exponentials are of the values less the max of their channel over the
  /// board, which changes nothing but keeps them at most one. Their means over
  /// the chains are taken by the same matrix product as the means of the
  /// values. That product runs in bfloat16 for a half precision type: half
  /// precision cannot hold the small exponentials, and in single precision the
  /// product is several times slower. Only the logarithm of their mean takes
  /// the rounding of bfloat16, and divided by the temperature, it stays small.
  pub fn pool(&self, features: Tensor<4>, soft_max: Option<f64>) -> Tensor<4> {
    let [batch, channels, height, width] = features.dims();
    let cells = height * width;
    let dtype = features.dtype();
    let features = features.reshape([batch, 1, channels, cells]);
    let member = self.member.clone().cast(dtype);
    let (pooled, channels) = match soft_max {
      None => {
        let features = features.expand([batch, PLAYERS, channels, cells]);
        // `[batch, PLAYERS, channels, MAX_CHAINS]`: the means over the chains.
        let means = features.matmul(self.mean.clone().cast(dtype));
        (means.matmul(member), channels)
      }
      Some(temperature) => {
        let pool_dtype = if dtype == DType::F32 {
          FloatDType::F32
        } else {
          FloatDType::BF16
        };
        let values = features.clone().cast(FloatDType::F32);
        // `[batch, 1, channels, 1]`
        let board_max = values.clone().max_dim(3);
        let exponentials = ((values - board_max.clone()) * temperature).exp();
        let features = Tensor::cat(vec![features.cast(pool_dtype), exponentials.cast(pool_dtype)], 2).expand([
          batch,
          PLAYERS,
          2 * channels,
          cells,
        ]);
        // `[batch, PLAYERS, 2 * channels, MAX_CHAINS]`: the means over the
        // chains of the values and of their exponentials.
        let means = features
          .matmul(self.mean.clone().cast(pool_dtype))
          .cast(FloatDType::F32);
        let soft_maxes = means
          .clone()
          .slice(s![.., .., channels..])
          .clamp_min(MIN_MEAN_EXPONENTIAL)
          .log()
          / temperature
          + board_max;
        let means = Tensor::cat(vec![means.slice(s![.., .., ..channels]), soft_maxes], 2);
        (means.cast(dtype).matmul(member), 2 * channels)
      }
    };
    Tensor::cat(
      vec![
        pooled.reshape([batch, PLAYERS * channels, height, width]),
        self.size_feature.clone().cast(dtype),
      ],
      1,
    )
  }
}

/// Pooling over the chains of dots: a `1x1` convolution picks the features to
/// pool, every dot gets their mean over its chain - and optionally their soft
/// max, see [`Chains::pool`] - and the size of the chain, for the player to
/// move's chains and for the opponent's, and another `1x1` convolution turns
/// those into the residual added to the block.
///
/// Convolutions only see a few cells further with every block, while a chain
/// can span the board. A chain is what a capture has to enclose and what
/// grounding connects to the edge, so what any part of it faces matters to the
/// rest of it: pooling hands that to every dot of the chain in one step. The
/// grouping comes from the rules rather than being learned, see `push_chains`
/// of the features.
#[derive(Module, Debug)]
pub struct ChainPool {
  norm: NormMask,
  conv1: Conv2d,
  normc: NormMask,
  /// Zero-initialized, so that a block starts as it would be without the
  /// pooling - which also lets a trained network gain it, see
  /// [`Model::adding_chains`](crate::model::Model::adding_chains).
  conv2: Conv2d,
  /// The temperature of the soft max, if any.
  soft_max: Option<f64>,
}

impl ChainPool {
  pub fn new(device: &Device, channels: usize, config: ChainsConfig) -> Self {
    assert!(
      config.soft_max.is_none_or(|temperature| temperature > 0.0),
      "the temperature of the soft max is positive"
    );
    let pooled = if config.soft_max.is_some() {
      2 * config.channels
    } else {
      config.channels
    };
    Self {
      norm: NormMask::new(device, channels, false),
      conv1: Conv2dConfig::new([channels, config.channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      normc: NormMask::new(device, config.channels, false),
      conv2: Conv2dConfig::new([PLAYERS * (pooled + 1), channels], [1, 1])
        .with_padding(PaddingConfig2d::Same)
        .with_bias(false)
        .init(device),
      soft_max: config.soft_max,
    }
  }

  pub(crate) fn initialize(&mut self, scale: f64, device: &Device) {
    init_conv(&mut self.conv1, scale, mish_gain(), device);
    init_conv(&mut self.conv2, 0.0, 1.0, device);
  }

  /// The residual of `inputs`, masked to the board.
  pub fn forward(&self, inputs: Tensor<4>, mask: Tensor<4>, chains: &Chains) -> Tensor<4> {
    let out = mish(self.norm.forward(inputs, mask.clone()));
    let out = conv_forward(&self.conv1, out);
    let out = mish(self.normc.forward(out, mask.clone()));
    conv_forward(&self.conv2, chains.pool(out, self.soft_max)) * mask
  }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
  use super::Chains;
  use burn::tensor::{Device, Tensor, TensorData};
  use oppai_zero::field_features::CHAIN_CHANNELS;

  // Every dot gets the mean over its chain and the size of the chain, the
  // player to move's chains first, and every other cell zeros.
  #[test]
  fn pool_takes_the_means_over_the_chains() {
    let device = Device::flex();
    // A 2x3 board: the player to move has a chain on the first two cells of
    // the first row and one on the last cell of the second; the opponent one
    // on the first two cells of the second row, and one ranked past
    // `MAX_CHAINS`, which is not pooled.
    #[rustfmt::skip]
    let labels = [
      // own ranks
      1., 1., 0., 0., 0., 2.,
      // enemy ranks
      0., 0., super::MAX_CHAINS as f32 + 1.0, 1., 1., 0.,
    ];
    let labels = Tensor::<4>::from_data(TensorData::new(labels.to_vec(), [1, CHAIN_CHANNELS, 2, 3]), &device);
    let features = Tensor::<4>::from_data(TensorData::new(vec![1., 2., 3., 4., 5., 6.], [1, 1, 2, 3]), &device);
    let pooled = Chains::new(labels).pool(features, None);
    let size = |size: f32| (size + 1.0).ln() * super::SIZE_FEATURE_SCALE as f32;
    #[rustfmt::skip]
    let expected = [
      // own means
      1.5, 1.5, 0., 0., 0., 6.,
      // enemy means
      0., 0., 0., 4.5, 4.5, 0.,
      // own sizes
      size(2.), size(2.), 0., 0., 0., size(1.),
      // enemy sizes
      0., 0., 0., size(2.), size(2.), 0.,
    ];
    pooled
      .into_data()
      .assert_approx_eq::<f32>(&TensorData::new(expected.to_vec(), [1, 4, 2, 3]), Default::default());
  }

  // With a soft max, every dot gets the soft max over its chain after the
  // means: between the mean and the max, and at a high temperature all but
  // the max. Other cells get zeros.
  #[test]
  fn pool_takes_the_soft_maxes_over_the_chains() {
    let device = Device::flex();
    // A 2x3 board: the player to move has a chain on the first row and one on
    // the last cell of the second; the opponent one on the first two cells of
    // the second row.
    #[rustfmt::skip]
    let labels = [
      // own ranks
      1., 1., 1., 0., 0., 2.,
      // enemy ranks
      0., 0., 0., 1., 1., 0.,
    ];
    let labels = Tensor::<4>::from_data(TensorData::new(labels.to_vec(), [1, CHAIN_CHANNELS, 2, 3]), &device);
    let features = Tensor::<4>::from_data(TensorData::new(vec![0., 3., 1., -1., 2., 6.], [1, 1, 2, 3]), &device);
    for temperature in [0.5, 2.0, 20.0] {
      let pooled = Chains::new(labels.clone()).pool(features.clone(), Some(temperature));
      let soft_max = |values: &[f64]| {
        let max = values.iter().copied().fold(f64::MIN, f64::max);
        let mean = values
          .iter()
          .map(|value| ((value - max) * temperature).exp())
          .sum::<f64>()
          / values.len() as f64;
        (max + mean.ln() / temperature) as f32
      };
      let (own, enemy) = (soft_max(&[0., 3., 1.]), soft_max(&[-1., 2.]));
      let size = |size: f32| (size + 1.0).ln() * super::SIZE_FEATURE_SCALE as f32;
      #[rustfmt::skip]
      let expected = [
        // own means
        4. / 3., 4. / 3., 4. / 3., 0., 0., 6.,
        // own soft maxes
        own, own, own, 0., 0., 6.,
        // enemy means
        0., 0., 0., 0.5, 0.5, 0.,
        // enemy soft maxes
        0., 0., 0., enemy, enemy, 0.,
        // own sizes
        size(3.), size(3.), size(3.), 0., 0., size(1.),
        // enemy sizes
        0., 0., 0., size(2.), size(2.), 0.,
      ];
      pooled
        .into_data()
        .assert_approx_eq::<f32>(&TensorData::new(expected.to_vec(), [1, 6, 2, 3]), Default::default());
      assert!(own > 4. / 3. && own < 3.);
      if temperature >= 20.0 {
        assert!(own > 2.9, "{own}");
      }
    }
  }

  // In half precision the soft max is what it is in single precision, up to
  // rounding: its exponentials, which half precision cannot hold, are pooled
  // in bfloat16.
  #[test]
  fn soft_max_pools_in_half_precision() {
    use burn::tensor::FloatDType;
    let device = Device::flex();
    let (height, width) = (4, 5);
    let labels = (0..CHAIN_CHANNELS * height * width)
      .map(|i| match i % 7 {
        0 | 3 => 1.0,
        5 => 2.0,
        _ => 0.0,
      })
      .collect::<Vec<f32>>();
    let labels = Tensor::<4>::from_data(TensorData::new(labels, [1, CHAIN_CHANNELS, height, width]), &device);
    let features = (0..3 * height * width)
      .map(|i| ((i * 37 % 23) as f32 - 6.0) / 4.0)
      .collect::<Vec<f32>>();
    let features = Tensor::<4>::from_data(TensorData::new(features, [1, 3, height, width]), &device);
    let chains = Chains::new(labels);
    let single = chains.pool(features.clone(), Some(8.0));
    let half = chains.pool(features.cast(FloatDType::F16), Some(8.0));
    half
      .cast(FloatDType::F32)
      .into_data()
      .assert_approx_eq::<f32>(&single.into_data(), burn::tensor::Tolerance::absolute(3e-2));
  }
}
