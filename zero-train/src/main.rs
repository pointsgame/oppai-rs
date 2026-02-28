#![allow(clippy::too_many_arguments)]

mod config;

use anyhow::{Error, Result};
use burn::{
  backend::{Autodiff, NdArray, Wgpu, ndarray::NdArrayDevice, wgpu::WgpuDevice},
  module::Module,
  optim::{Optimizer, SgdConfig, decay::WeightDecayConfig, momentum::MomentumConfig},
  record::{DefaultFileRecorder, FullPrecisionSettings, Record, Recorder},
  tensor::backend::{AutodiffBackend, Backend},
};
use config::{Action, Backend as ConfigBackend, Config, InitParams, PitParams, PlayParams, TrainParams, cli_parse};
use either::Either;
use num_traits::Float;
use oppai_field::{any_field::AnyField, field::Field, player::Player};
use oppai_initial::initial::InitialPosition;
use oppai_sgf::{from_sgf, to_sgf};
use oppai_zero::{
  episode::{self, episode},
  examples::Examples,
  model::TrainableModel,
  pit,
  random_model::RandomModel,
};
use oppai_zero_burn::model::{Learner, Model as BurnModel, Predictor};
use oppai_zero_sgf::{sgf_to_visits, visits_to_sgf};
use rand::{Rng, SeedableRng, distr::uniform::SampleUniform, rngs::SmallRng};
use rand_distr::{Distribution, Exp1, Open01, StandardNormal};
use sgf_parse::{GameTree, SimpleText, serialize, unknown_game::Prop};
use std::{
  cmp::Ordering,
  fmt::{Debug, Display},
  fs::{self, File},
  io::Write,
  iter::{self, Sum},
  process::ExitCode,
};

fn init<B>(params: InitParams, device: B::Device) -> Result<ExitCode>
where
  B: AutodiffBackend,
{
  let model = BurnModel::<B>::new(&device);
  model.save_file(params.model, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let optimizer = SgdConfig::new()
    .with_weight_decay(Some(WeightDecayConfig::new(0.00003)))
    .with_momentum(Some(MomentumConfig::new()))
    .init::<B, BurnModel<_>>();
  let record = optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    params.optimizer,
  )?;

  Ok(ExitCode::SUCCESS)
}

fn play<B, R: Rng>(params: PlayParams, device: B::Device, rng: &mut R) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let mut model = match params.model {
    Some(model_path) => {
      let model = BurnModel::<B>::new(&device);
      let model = model.load_file(
        model_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
      )?;
      Either::Right(Predictor { model, device })
    }
    None => Either::Left(RandomModel(SmallRng::from_seed(rng.random()))),
  };

  let mut file = File::options().append(true).create(true).open(&params.game)?;

  for _ in 0..params.count {
    let width = params.width[rng.random_range(0..params.width.len())];
    let height = params.height[rng.random_range(0..params.height.len())];
    let komi_x_2 = params.komi_x_2[rng.random_range(0..params.komi_x_2.len())];
    let player = Player::Red;
    let mut field = Field::new_from_rng(width, height, rng);

    for (pos, player) in InitialPosition::Cross.points(width, height, player) {
      // TODO: random shift
      field.put_point(pos, player);
    }

    let visits = episode(&mut field, player, &mut model, komi_x_2, rng)
      .map_err(|e| e.either(|()| anyhow::anyhow!("random model failed"), Error::from))?;

    let field = field.into();
    if let Some(mut node) = to_sgf(&field) {
      visits_to_sgf(&mut node, &visits, field.field().stride, field.field().moves_count());
      let score = field.field().score(Player::Red);
      node.properties.push(Prop::RE(match score.cmp(&0) {
        Ordering::Equal => "0".into(),
        Ordering::Greater => SimpleText {
          text: format!("W+{}", score),
        },
        Ordering::Less => SimpleText {
          text: format!("B+{}", score.abs()),
        },
      }));
      let sgf = serialize(iter::once(&GameTree::Unknown(node)));
      writeln!(&mut file, "{sgf}")?;
    }
  }

  Ok(ExitCode::SUCCESS)
}

fn train<B, R: Rng>(params: TrainParams, device: B::Device, rng: &mut R) -> Result<ExitCode>
where
  B: AutodiffBackend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let optimizer = SgdConfig::new()
    .with_weight_decay(Some(WeightDecayConfig::new(0.00003)))
    .with_momentum(Some(MomentumConfig::new()))
    .init::<B, BurnModel<_>>();
  let item = Recorder::<B>::load_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &mut params.optimizer.clone(),
  )?;
  let record = Record::from_item::<FullPrecisionSettings>(item, &device);
  let optimizer = optimizer.load_record(record);
  let predictor = Predictor { model, device };
  let mut learner = Learner {
    predictor,
    optimizer,
    lr: params.learning_rate,
  };

  let mut examples: Examples<<B as Backend>::FloatElem> = Default::default();
  for path in params.games {
    let sgf = fs::read_to_string(path)?;
    let trees = sgf_parse::parse(&sgf)?;
    for node in trees.iter().filter_map(|tree| match tree {
      GameTree::Unknown(node) => Some(node),
      GameTree::GoGame(_) => None,
    }) {
      let field = from_sgf::<Field, _>(node, rng).ok_or(anyhow::anyhow!("invalid sgf"))?;
      let visits = sgf_to_visits(node, field.stride);

      if field.width() > params.width || field.height() > params.height {
        return Err(anyhow::anyhow!(
          "Game is bigger than config: {}:{}",
          field.width(),
          field.height()
        ));
      }

      examples = examples
        + episode::examples(
          params.width,
          params.height,
          0,
          field.zobrist_arc(),
          &visits,
          &field.colored_moves().collect::<Vec<_>>(),
        );
    }
  }

  for epoch in 0..params.epochs {
    log::info!("Training {} epoch", epoch);
    examples.shuffle(rng);
    for batch in examples.batches(params.batch_size) {
      learner = learner.train(
        batch.inputs,
        batch.global,
        batch.policies,
        batch.opponent_policies,
        batch.values,
        batch.scores,
      )?;
    }
  }

  learner
    .predictor
    .model
    .save_file(params.model_new, &DefaultFileRecorder::<FullPrecisionSettings>::new())?;

  let record = learner.optimizer.to_record();
  let item = record.into_item::<FullPrecisionSettings>();
  Recorder::<B>::save_item(
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    item,
    params.optimizer_new,
  )?;

  Ok(ExitCode::SUCCESS)
}

fn pit<B, R: Rng>(params: PitParams, device: B::Device, rng: &mut R) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
{
  let model = BurnModel::<B>::new(&device);
  let model = model.load_file(
    params.model,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let mut predictor = Predictor {
    model,
    device: device.clone(),
  };

  let model_new = BurnModel::<B>::new(&device);
  let model_new = model_new.load_file(
    params.model_new,
    &DefaultFileRecorder::<FullPrecisionSettings>::new(),
    &device,
  )?;
  let mut predictor_new = Predictor {
    model: model_new,
    device,
  };

  let player = Player::Red;
  let field = Field::new_from_rng(params.width, params.height, rng);

  let games = params.games;
  let result = if pit::pit(&field, player, &mut predictor_new, &mut predictor, 0, rng, &|field| {
    if let Some(ref games) = games
      && let Some(node) = to_sgf(&field.into())
    {
      let sgf = serialize(iter::once(&GameTree::Unknown(node)));
      let mut file = File::options().append(true).create(true).open(games).unwrap();
      writeln!(&mut file, "{sgf}").unwrap();
    }
  })? {
    ExitCode::SUCCESS
  } else {
    2.into()
  };

  Ok(result)
}

fn run<B>(config: Config, action: Action, device: B::Device) -> Result<ExitCode>
where
  B: Backend,
  <B as Backend>::FloatElem: Float + Sum + SampleUniform + Display + Debug,
  StandardNormal: Distribution<<B as Backend>::FloatElem>,
  Exp1: Distribution<<B as Backend>::FloatElem>,
  Open01: Distribution<<B as Backend>::FloatElem>,
{
  let mut rng = config.seed.map_or_else(SmallRng::from_os_rng, SmallRng::seed_from_u64);

  match action {
    Action::Init(params) => init::<Autodiff<B>>(params, device),
    Action::Play(params) => play::<B, _>(params, device, &mut rng),
    Action::Train(params) => train::<Autodiff<B>, _>(params, device, &mut rng),
    Action::Pit(params) => pit::<B, _>(params, device, &mut rng),
  }
}

fn main() -> Result<ExitCode> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let (config, action) = cli_parse();

  match config.backend {
    ConfigBackend::Ndarray => run::<NdArray>(config, action, NdArrayDevice::Cpu),
    ConfigBackend::Wgpu => run::<Wgpu>(config, action, WgpuDevice::DefaultDevice),
  }
}
