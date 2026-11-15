//! The optimizers training can use, each set up per group of parameters: the
//! groups differ in weight decay and learning rate, and with Muon in the
//! update rule too, since orthogonalizing an update only makes sense for the
//! weight matrices of the hidden layers.
//!
//! Every optimizer is wrapped in [`Lookahead`] by default: the weights take
//! `k` fast steps, then move back toward where they were by `1 - alpha`.

use std::{
  fmt,
  str::FromStr,
  sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
  },
};

use burn::{
  module::ParamGroup,
  optim::{
    AdamWConfig, AdjustLrFn, LearningRate, ModuleOptimizer, Muon, MuonConfig, MuonState, Optimizer, RecordState,
    SgdConfig, StateSink, StateSource, decay::WeightDecayConfig, join_path, momentum::MomentumConfig,
  },
  tensor::{Device, Tensor},
};
use burn_pack::Scalar;

/// The update rule of training.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OptimizerKind {
  /// Stochastic gradient descent with momentum.
  Sgd,
  /// Adam with decoupled weight decay.
  AdamW,
  /// Muon for the weight matrices of the hidden layers - convolution filters
  /// flattened to `[out, in * kernel]` - and AdamW for everything else.
  Muon,
}

impl OptimizerKind {
  pub const NAMES: [&str; 3] = ["sgd", "adamw", "muon"];

  /// Weight decay of the hidden layers' weights, see [`OptimizerConfig`].
  ///
  /// With SGD it is an L2 penalty added to the gradient of a loss averaged
  /// over the batch, so it does not depend on the batch size. AdamW and Muon
  /// decay the weights by `learning_rate * weight_decay` every step whatever
  /// the gradient's scale is, and since their learning rate grows with the
  /// square root of the batch size, so does this.
  pub fn default_weight_decay(self, batch_size: usize) -> f64 {
    match self {
      Self::Sgd => 0.000000004,
      Self::AdamW | Self::Muon => 0.005 * batch_scale(batch_size),
    }
  }

  /// The learning rate of a trained network. SGD's is that of a loss averaged
  /// over the batch, so it is linear in the batch size: this is the one of a
  /// batch of 512. Adam's step does not depend on the gradient's scale, and
  /// its learning rate grows with the square root of the batch size instead.
  pub fn default_learning_rate(self, batch_size: usize) -> f64 {
    match self {
      Self::Sgd => 0.0001,
      Self::AdamW | Self::Muon => 0.00004 * batch_scale(batch_size),
    }
  }
}

/// The square root of the batch size relative to 256, by which the steps of
/// Adam and Muon have to grow to keep pace with bigger batches.
fn batch_scale(batch_size: usize) -> f64 {
  (batch_size as f64 / 256.0).sqrt()
}

impl FromStr for OptimizerKind {
  type Err = String;

  fn from_str(name: &str) -> Result<Self, String> {
    match name {
      "sgd" => Ok(Self::Sgd),
      "adamw" => Ok(Self::AdamW),
      "muon" => Ok(Self::Muon),
      _ => Err(format!("unknown optimizer {name}, expected one of {:?}", Self::NAMES)),
    }
  }
}

impl fmt::Display for OptimizerKind {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Self::Sgd => "sgd",
      Self::AdamW => "adamw",
      Self::Muon => "muon",
    })
  }
}

/// Lookahead: after every `steps` steps of the wrapped optimizer, the weights
/// move from where they were `steps` steps ago only `alpha` of the way to where
/// those steps took them. The fast steps get the learning rate divided by
/// `alpha`, so that a whole cycle goes as far as plain steps would, but along
/// an average of their noisy directions.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Lookahead {
  pub steps: usize,
  pub alpha: f64,
}

impl Default for Lookahead {
  fn default() -> Self {
    Self { steps: 6, alpha: 0.5 }
  }
}

/// How many steps the optimizer has taken, as of its latest one. The count is
/// part of the optimizer's saved state, so it goes on from where the loaded
/// state left off: it counts the steps of the whole run, not just those of one
/// training. See [`OptimizerConfig::init_with_clock`].
#[derive(Clone, Default, Debug)]
pub struct StepClock(Arc<AtomicUsize>);

impl StepClock {
  pub fn steps(&self) -> usize {
    self.0.load(Ordering::Relaxed)
  }
}

/// How to optimize a model's parameters.
#[derive(Clone, PartialEq, Debug)]
pub struct OptimizerConfig {
  pub kind: OptimizerKind,
  /// Weight decay of the weights of the hidden layers, the input layer, the
  /// heads and the scales of the trunk's normalizations; the attention's
  /// projections get half of it and the offsets, the biases and the rotary
  /// frequencies nearly none.
  pub weight_decay: f64,
  /// `None` steps the wrapped optimizer alone.
  pub lookahead: Option<Lookahead>,
  /// The loss is a mean over the batch, so its gradient is this many times
  /// smaller than that of the batch's summed loss, which Adam's epsilon is
  /// stated for: the epsilon has to shrink as much to stay as small next to
  /// the gradient.
  pub batch_size: usize,
}

/// The groups of a model's parameters, which differ in how they are
/// regularized and updated.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Group {
  /// Everything no other group takes: the offsets of the trunk's
  /// normalizations, the scales of the transformer's ones, and the rotary
  /// frequencies, distance slopes and logit scales of the attention, none of
  /// which has a reason to be pulled to zero.
  NoDecay,
  /// The weights of the trunk's convolutions and feed-forward layers.
  Hidden,
  /// The weights of the attention's projections.
  Attention,
  /// The weights of the depthwise convolutions of the feed-forward layers,
  /// decayed like the hidden ones but no matrices: every channel's filter is
  /// on its own.
  Depthwise,
  /// The scales of the trunk's normalizations.
  Scale,
  /// The weights reading the input features.
  Input,
  /// The biases and offsets of the heads and of the trunk's final
  /// normalization.
  OutputNoDecay,
  /// The weights of the heads.
  Output,
}

impl Group {
  /// In the order the optimizer has to take them: a parameter is in the last
  /// group matching it, and the first one has to match all of them.
  const ALL: [Group; 8] = [
    Group::NoDecay,
    Group::Hidden,
    Group::Attention,
    Group::Depthwise,
    Group::Scale,
    Group::Input,
    Group::OutputNoDecay,
    Group::Output,
  ];

  /// The parameters of the group by their paths in the model's record.
  fn pattern(self) -> &'static str {
    match self {
      Self::NoDecay => "",
      Self::Hidden => r"^residuals\..*\.weight$",
      Self::Attention => r"^residuals\..*\.attention\.(qkv|gate|output)\.weight$",
      Self::Depthwise => r"^residuals\..*\.ffn\.conv\.weight$",
      Self::Scale => r"^residuals\..*\.normactconv[^.]*\.norm\.gamma$",
      Self::Input => r"^(conv_spatial|linear_global)\.weight$",
      Self::OutputNoDecay => r"^(policy_head|value_head|captured_head|norm_trunkfinal)\.",
      Self::Output => r"^((policy_head|value_head|captured_head)\..*\.weight|norm_trunkfinal\.gamma)$",
    }
  }

  fn param_group(self) -> ParamGroup {
    match self {
      Self::NoDecay => ParamGroup::all(),
      _ => ParamGroup::from_regex(self.pattern()).expect("the patterns are valid"),
    }
  }

  /// The weight matrices that Muon orthogonalizes the updates of.
  fn is_matrix(self) -> bool {
    matches!(self, Self::Hidden | Self::Attention)
  }

  /// The weight decay of the group relative to that of the hidden layers.
  fn weight_decay_factor(self, kind: OptimizerKind) -> f64 {
    match self {
      Self::Hidden | Self::Depthwise | Self::Scale | Self::Input | Self::Output => 1.0,
      Self::Attention => 0.5,
      // Next to nothing: a hundredth with SGD, and with Adam and Muon a
      // decay of 1e-8 at the default weight decay.
      Self::NoDecay | Self::OutputNoDecay => match kind {
        OptimizerKind::Sgd => 0.01,
        OptimizerKind::AdamW | OptimizerKind::Muon => 0.000002,
      },
    }
  }

  /// The learning rate of the group relative to the base one: the heads
  /// learn at half of it, and Muon's steps on the weight matrices are twice
  /// as long as Adam's.
  fn learning_rate_factor(self, kind: OptimizerKind) -> f64 {
    match self {
      Self::Output | Self::OutputNoDecay => 0.5,
      _ if kind == OptimizerKind::Muon && self.is_matrix() => 2.0,
      _ => 1.0,
    }
  }
}

impl OptimizerConfig {
  /// Whether the weights after the optimizer's `steps`-th step are where a
  /// lookahead cycle ends rather than fast weights part way through one, which
  /// have wandered off along that cycle's noise.
  pub fn is_synced(&self, steps: usize) -> bool {
    self
      .lookahead
      .is_none_or(|lookahead| steps.is_multiple_of(lookahead.steps))
  }

  pub fn new(kind: OptimizerKind, batch_size: usize) -> Self {
    Self {
      kind,
      weight_decay: kind.default_weight_decay(batch_size),
      lookahead: Some(Lookahead::default()),
      batch_size,
    }
  }

  pub fn init(&self) -> ModuleOptimizer {
    self.init_with_clock().0
  }

  /// The optimizer, and a clock telling how many steps it has taken.
  pub fn init_with_clock(&self) -> (ModuleOptimizer, StepClock) {
    let clock = StepClock::default();
    let mut groups = Group::ALL.into_iter();
    let first = groups.next().expect("there are groups");
    debug_assert_eq!(first, Group::NoDecay, "the first group has to match every parameter");
    let optimizer = groups.fold(self.optimizer(first, None, &clock), |optimizer, group| {
      self.optimizer(group, Some(optimizer), &clock)
    });
    (optimizer, clock)
  }

  /// Adds the optimizer of `group` to `optimizer`, or creates the optimizer
  /// with it as the fallback when there is none yet.
  fn optimizer(&self, group: Group, optimizer: Option<ModuleOptimizer>, clock: &StepClock) -> ModuleOptimizer {
    let weight_decay = self.weight_decay * group.weight_decay_factor(self.kind);
    let decay = (weight_decay > 0.0).then(|| WeightDecayConfig::new(weight_decay as f32));
    match self.kind {
      OptimizerKind::Sgd => self.add(
        optimizer,
        group,
        clock,
        SgdConfig::new()
          .with_weight_decay(decay)
          // Without dampening, which would only scale the learning rate down.
          .with_momentum(Some(MomentumConfig::new().with_dampening(0.0)))
          .build(),
      ),
      OptimizerKind::AdamW => self.add(
        optimizer,
        group,
        clock,
        AdamWConfig::new()
          .with_beta_1(0.9)
          .with_beta_2(0.999)
          .with_epsilon(self.epsilon(1e-8))
          .with_weight_decay(weight_decay as f32)
          .build(),
      ),
      OptimizerKind::Muon if group.is_matrix() => self.add(
        optimizer,
        group,
        clock,
        MatrixMuon(
          MuonConfig::new()
            .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
            .with_weight_decay(decay)
            .build(),
        ),
      ),
      // Adam next to Muon averages over longer horizons than on its own.
      OptimizerKind::Muon => self.add(
        optimizer,
        group,
        clock,
        AdamWConfig::new()
          .with_beta_1(0.95)
          .with_beta_2(0.995)
          .with_epsilon(self.epsilon(1e-6))
          .with_weight_decay(weight_decay as f32)
          .build(),
      ),
    }
  }

  /// Adam's `epsilon` for a loss summed over the batch, restated for the mean.
  fn epsilon(&self, epsilon: f32) -> f32 {
    epsilon / self.batch_size as f32
  }

  fn add<O: Optimizer>(
    &self,
    optimizer: Option<ModuleOptimizer>,
    group: Group,
    clock: &StepClock,
    inner: O,
  ) -> ModuleOptimizer {
    let tuned = Tuned {
      inner,
      learning_rate_factor: group.learning_rate_factor(self.kind),
      lookahead: self.lookahead,
      clock: clock.clone(),
    };
    match optimizer {
      Some(optimizer) => optimizer.with_group(group.param_group(), tuned, None),
      None => tuned.into(),
    }
  }
}

/// Muon for weights of any rank above 1: a convolution's filters are
/// orthogonalized as the matrix `[out, in * kernel]`, mapping the patch under
/// the kernel to the output channels.
///
/// TODO: drop once burn's `Muon` takes convolution filters itself; it panics
/// on anything but matrices.
#[derive(Clone)]
struct MatrixMuon(Muon);

impl Optimizer for MatrixMuon {
  type State<const D: usize> = MuonState<2>;

  fn step<const D: usize>(
    &self,
    lr: LearningRate,
    tensor: Tensor<D>,
    grad: Tensor<D>,
    state: Option<MuonState<2>>,
  ) -> (Tensor<D>, Option<MuonState<2>>) {
    let dims = tensor.dims();
    let matrix = [dims[0], dims[1..].iter().product()];
    let (tensor, state) = self.0.step(lr, tensor.reshape(matrix), grad.reshape(matrix), state);
    (tensor.reshape(dims), state)
  }

  fn to_device<const D: usize>(state: MuonState<2>, device: &Device) -> MuonState<2> {
    <Muon as Optimizer>::to_device(state, device)
  }
}

/// An optimizer with the learning rate scaled by a factor, and stepping under
/// [`Lookahead`] when it is set.
///
/// TODO: give the groups their learning rates through burn's
/// `ModuleLearningRate` once it can be built with groups other than by a
/// learning rate scheduler, and drop the lookahead here once burn has one.
#[derive(Clone)]
struct Tuned<O> {
  inner: O,
  learning_rate_factor: f64,
  lookahead: Option<Lookahead>,
  /// Where the count of steps is published. Every parameter keeps a count of
  /// its own, and they all agree as each step steps every parameter.
  clock: StepClock,
}

#[derive(Clone)]
struct TunedState<S, const D: usize> {
  inner: Option<S>,
  /// Where the weights move back toward at the end of the lookahead cycle.
  slow: Option<Tensor<D>>,
  /// Steps taken so far.
  steps: usize,
}

// TODO: derive once burn's `RecordState` derive works outside of burn's
// optimizer crate - it refers to `crate::RecordState` - and drop the direct
// dependency on `burn_pack`, needed only for the scalar here.
impl<S: RecordState, const D: usize> RecordState for TunedState<S, D> {
  fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
    if let Some(inner) = &self.inner {
      inner.state_flatten(&join_path(prefix, "inner"), out);
    }
    if let Some(slow) = &self.slow {
      out.push_tensor(prefix, "slow", slow.to_data());
    }
    out.push_scalar(prefix, "steps", Scalar::from(self.steps));
  }

  fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
    let steps = usize::try_from(src.take_scalar(prefix, "steps")?).ok()?;
    let inner_prefix = join_path(prefix, "inner");
    let inner = if src.has_under(&inner_prefix) {
      Some(S::state_unflatten(&inner_prefix, src, device)?)
    } else {
      None
    };
    let slow = src
      .take_tensor(prefix, "slow")
      .map(|data| Tensor::from_data(data, device));
    Some(Self { inner, slow, steps })
  }
}

impl<O: Optimizer> Optimizer for Tuned<O> {
  type State<const D: usize> = TunedState<O::State<D>, D>;

  fn step<const D: usize>(
    &self,
    lr: LearningRate,
    tensor: Tensor<D>,
    grad: Tensor<D>,
    state: Option<Self::State<D>>,
  ) -> (Tensor<D>, Option<Self::State<D>>) {
    let (inner, slow, steps) = state.map_or((None, None, 0), |state| (state.inner, state.slow, state.steps));
    let alpha = self.lookahead.map_or(1.0, |lookahead| lookahead.alpha);
    let slow = self.lookahead.and(slow.or_else(|| Some(tensor.clone())));
    let (tensor, inner) = self
      .inner
      .step(lr * self.learning_rate_factor / alpha, tensor, grad, inner);
    let steps = steps + 1;
    self.clock.0.fetch_max(steps, Ordering::Relaxed);
    let (tensor, slow) = match (self.lookahead, slow) {
      (Some(lookahead), Some(slow)) if steps.is_multiple_of(lookahead.steps) => {
        let slow = slow.clone() + (tensor - slow) * lookahead.alpha;
        (slow.clone(), Some(slow))
      }
      (_, slow) => (tensor, slow),
    };
    (tensor, Some(TunedState { inner, slow, steps }))
  }

  fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
    TunedState {
      inner: state.inner.map(|inner| O::to_device(inner, device)),
      slow: state.slow.map(|slow| slow.to_device(device)),
      steps: state.steps,
    }
  }
}

#[cfg(all(test, feature = "flex"))]
mod tests {
  use super::{Group, Lookahead, MatrixMuon, OptimizerConfig, OptimizerKind, StepClock, Tuned};
  use crate::model::{BlockConfig, GpoolConfig, Model, ModelConfig, NbtConfig, TransformerConfig};
  use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::{GradientsParams, MuonConfig, Optimizer, SgdConfig},
    tensor::{Device, Distribution, Tensor},
  };

  fn group_of(path: &str) -> Group {
    *Group::ALL
      .iter()
      .rev()
      .find(|group| group.param_group().matches(&ParamId::new(), Some(path)))
      .unwrap()
  }

  #[test]
  fn parameters_fall_into_their_groups() {
    for (path, group) in [
      ("conv_spatial.weight", Group::Input),
      ("linear_global.weight", Group::Input),
      (
        "residuals.0.Nbt.inner.1.normactconv1.convgpool.Conv.weight",
        Group::Hidden,
      ),
      (
        "residuals.1.Nbt.inner.0.normactconv1.convgpool.Gpool.conv1g.weight",
        Group::Hidden,
      ),
      (
        "residuals.1.Nbt.inner.0.normactconv1.convgpool.Gpool.linearg.weight",
        Group::Hidden,
      ),
      (
        "residuals.1.Nbt.inner.0.normactconv1.convgpool.Gpool.lineargate.weight",
        Group::Hidden,
      ),
      (
        "residuals.1.Nbt.inner.0.normactconv1.convgpool.Gpool.normg.beta",
        Group::NoDecay,
      ),
      ("residuals.0.Nbt.normactconvp.convgpool.Conv.weight", Group::Hidden),
      ("residuals.0.Nbt.normactconvp.norm.beta", Group::NoDecay),
      ("residuals.0.Nbt.normactconvq.norm.gamma", Group::Scale),
      ("residuals.0.Nbt.inner.0.normactconv2.norm.gamma", Group::Scale),
      (
        "residuals.2.Transformer.layers.1.attention.qkv.weight",
        Group::Attention,
      ),
      (
        "residuals.2.Transformer.layers.1.attention.gate.weight",
        Group::Attention,
      ),
      (
        "residuals.2.Transformer.layers.1.attention.query_norm.gamma",
        Group::NoDecay,
      ),
      (
        "residuals.2.Transformer.layers.1.attention.key_norm.gamma",
        Group::NoDecay,
      ),
      (
        "residuals.2.Transformer.layers.1.attention.output.weight",
        Group::Attention,
      ),
      ("residuals.2.Transformer.layers.1.attention.norm.gamma", Group::NoDecay),
      (
        "residuals.2.Transformer.layers.1.attention.rope.directions",
        Group::NoDecay,
      ),
      ("residuals.2.Transformer.layers.1.attention.rope.excess", Group::NoDecay),
      (
        "residuals.2.Transformer.layers.1.attention.distance_slopes",
        Group::NoDecay,
      ),
      (
        "residuals.2.Transformer.layers.1.attention.logit_scales",
        Group::NoDecay,
      ),
      ("residuals.2.Transformer.layers.1.ffn.input.weight", Group::Hidden),
      ("residuals.2.Transformer.layers.1.ffn.conv.weight", Group::Depthwise),
      ("residuals.2.Transformer.layers.1.ffn.output.weight", Group::Hidden),
      ("residuals.2.Transformer.layers.1.ffn.norm.gamma", Group::NoDecay),
      ("norm_trunkfinal.beta", Group::OutputNoDecay),
      ("policy_head.conv1p.weight", Group::Output),
      ("policy_head.biasg.beta", Group::OutputNoDecay),
      ("value_head.linear2.weight", Group::Output),
      ("value_head.linear2.bias", Group::OutputNoDecay),
      ("captured_head.conv.weight", Group::Output),
    ] {
      assert_eq!(group_of(path), group, "{path}");
    }
  }

  /// The paths of every trained parameter of a model.
  #[derive(Default)]
  struct Paths {
    path: Vec<String>,
    paths: Vec<String>,
  }

  impl ModuleVisitor for Paths {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
      self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
      self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
      if param.is_require_grad() {
        self.paths.push(self.path.join("."));
      }
    }
  }

  // Every weight of both networks is decayed, and only the weights of the
  // trunk are matrices for Muon, but for the depthwise convolutions'.
  #[test]
  fn every_weight_is_decayed() {
    let device = Device::flex();
    let hybrid = ModelConfig {
      blocks: vec![
        BlockConfig::Transformer(TransformerConfig {
          mid_channels: 16,
          layers: 1,
          attention_heads: 2,
          ffn_channels: 24,
          rotary_channels: 6,
          min_train_side: 4,
          ffn_conv: true,
        }),
        BlockConfig::Nbt(NbtConfig {
          mid_channels: 16,
          size: 1,
          gpool: Some(GpoolConfig {
            channels: 8,
            gate: true,
          }),
        }),
      ],
      ..ModelConfig::default()
    };
    for config in [ModelConfig::default(), hybrid] {
      let mut paths = Paths::default();
      Model::new(&device, &config).visit(&mut paths);
      for path in &paths.paths {
        let group = group_of(path);
        assert_eq!(
          group == Group::NoDecay || group == Group::OutputNoDecay,
          !path.ends_with(".weight"),
          "{path}"
        );
        assert_eq!(
          group.is_matrix(),
          path.starts_with("residuals.") && path.ends_with(".weight") && !path.ends_with(".ffn.conv.weight"),
          "{path}"
        );
      }
    }
  }

  // A convolution's filters are orthogonalized as one matrix per output
  // channel row.
  #[test]
  fn muon_flattens_convolutions() {
    let device = Device::flex();
    let muon = MuonConfig::new().build();
    let tensor = Tensor::<4>::random([4, 3, 3, 3], Distribution::Default, &device);
    let grad = Tensor::<4>::random([4, 3, 3, 3], Distribution::Default, &device);
    let (stepped, _) = MatrixMuon(muon.clone()).step(0.1, tensor.clone(), grad.clone(), None);
    let (expected, _) = muon.step(0.1, tensor.reshape([4, 27]), grad.reshape([4, 27]), None);
    stepped
      .reshape([4, 27])
      .into_data()
      .assert_approx_eq::<f32>(&expected.into_data(), Default::default());
  }

  // With a constant gradient, every lookahead cycle moves as far as plain
  // steps at the given learning rate would, and in between the fast steps
  // are `1 / alpha` times longer.
  #[test]
  fn lookahead_cycles() {
    let device = Device::flex();
    let tuned = Tuned {
      inner: SgdConfig::new().build(),
      learning_rate_factor: 1.0,
      lookahead: Some(Lookahead { steps: 2, alpha: 0.5 }),
      clock: StepClock::default(),
    };
    let mut tensor = Tensor::<1>::zeros([1], &device);
    let mut state = None;
    let mut values = Vec::new();
    for _ in 0..4 {
      let (stepped, next) = tuned.step(0.1, tensor, Tensor::ones([1], &device), state);
      values.push(stepped.clone().into_scalar::<f32>());
      (tensor, state) = (stepped, next);
    }
    for (value, expected) in values.into_iter().zip([-0.2, -0.2, -0.4, -0.4]) {
      assert!((value - expected).abs() < 1e-6, "{value} {expected}");
    }
  }

  #[derive(Module, Debug)]
  struct Weight {
    weight: Param<Tensor<4>>,
  }

  /// A module with a parameter in a group of Muon and one of Adam.
  #[derive(Module, Debug)]
  struct Tiny {
    conv_spatial: Weight,
    residuals: Vec<Weight>,
  }

  fn grads(module: &Tiny) -> GradientsParams {
    let loss = (module.conv_spatial.weight.val().sin().sum() + module.residuals[0].weight.val().cos().sum()) * 3.0;
    GradientsParams::from_grads(loss.backward(), module)
  }

  // The optimizer's state - Muon's momentum on a flattened convolution,
  // Adam's moments and where the lookahead cycle is - survives saving, so a
  // training continues as if it never stopped.
  #[test]
  fn state_survives_saving() {
    let device = Device::flex().autodiff();
    let weight = |shape: [usize; 4]| Weight {
      weight: Param::from_tensor(Tensor::random(shape, Distribution::Default, &device)),
    };
    let mut module = Tiny {
      conv_spatial: weight([4, 2, 3, 3]),
      residuals: vec![weight([5, 4, 3, 3])],
    };
    let config = OptimizerConfig::new(OptimizerKind::Muon, 256);
    let mut optimizer = config.init();
    for _ in 0..4 {
      let grads = grads(&module);
      module = optimizer.step(0.01, module, grads);
    }
    let mut loaded = config.init().load_record(optimizer.to_record());
    let mut copy = module.clone();
    for _ in 0..3 {
      let grads_module = grads(&module);
      module = optimizer.step(0.01, module, grads_module);
      let grads_copy = grads(&copy);
      copy = loaded.step(0.01, copy, grads_copy);
    }
    for (a, b) in [
      (module.conv_spatial.weight.val(), copy.conv_spatial.weight.val()),
      (module.residuals[0].weight.val(), copy.residuals[0].weight.val()),
    ] {
      a.into_data()
        .assert_approx_eq::<f32>(&b.into_data(), Default::default());
    }
  }

  // The clock counts the steps of the whole run: an optimizer loaded from a
  // saved state goes on from the count it was saved with.
  #[test]
  fn the_clock_goes_on_after_loading() {
    let device = Device::flex().autodiff();
    let weight = |shape: [usize; 4]| Weight {
      weight: Param::from_tensor(Tensor::random(shape, Distribution::Default, &device)),
    };
    let mut module = Tiny {
      conv_spatial: weight([4, 2, 3, 3]),
      residuals: vec![weight([5, 4, 3, 3])],
    };
    let config = OptimizerConfig::new(OptimizerKind::Sgd, 256);
    let (mut optimizer, clock) = config.init_with_clock();
    for _ in 0..4 {
      let grads = grads(&module);
      module = optimizer.step(0.01, module, grads);
    }
    assert_eq!(clock.steps(), 4);
    let (loaded, clock) = config.init_with_clock();
    let mut loaded = loaded.load_record(optimizer.to_record());
    for _ in 0..3 {
      let grads = grads(&module);
      module = loaded.step(0.01, module, grads);
    }
    assert_eq!(clock.steps(), 7);
    assert!(!config.is_synced(7));
    assert!(config.is_synced(12));
  }

  #[test]
  fn kinds_parse_back() {
    for name in OptimizerKind::NAMES {
      assert_eq!(name.parse::<OptimizerKind>().unwrap().to_string(), name);
    }
  }
}
