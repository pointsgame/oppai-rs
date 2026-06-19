#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

mod config;

use crate::config::{Backend as ConfigBackend, Config, cli_parse};
use anyhow::Result;
#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
#[cfg(feature = "ndarray")]
use burn::backend::{NdArray, ndarray::NdArrayDevice};
#[cfg(feature = "rocm")]
use burn::backend::{Rocm, rocm::RocmDevice};
#[cfg(any(feature = "vulkan", feature = "webgpu"))]
use burn::backend::{Wgpu, wgpu::WgpuDevice};
use burn::{
  module::Module,
  record::{DefaultFileRecorder, FullPrecisionSettings},
  tensor::{backend::Backend, ops::FloatElem},
};
use either::Either;
use num_traits::Float;
use oppai_ai::{ai::AI, analysis::Analysis};
use oppai_ais::{
  oppai::{InConfidence, Oppai},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::field::Field;
use oppai_patterns::patterns::Patterns;
use oppai_protocol::{Constraint, Coords, Move, Request, Response};
use oppai_zero_burn::model::{Model as BurnModel, Predictor};
use rand::{make_rng, rngs::SmallRng};
use std::{
  default::Default,
  fmt::{Debug, Display},
  fs::File,
  io::{self, BufRead, BufReader, Read, Write},
  iter::Sum,
  path::Path,
  sync::Arc,
};

type CliModel<B> = Either<(), Predictor<B>>;

struct State<B: Backend>
where
  FloatElem<B>: Float + Sum + Display + Debug,
{
  field: Field,
  rng: SmallRng,
  oppai: Oppai<FloatElem<B>, CliModel<B>>,
}

fn run<B>(config: Config, patterns: Arc<Patterns>, device: B::Device) -> Result<()>
where
  B: Backend,
  FloatElem<B>: Float + Sum + Display + Debug,
{
  let model = config.model.as_ref().map(|model_path| {
    let model = BurnModel::<B>::new(&device);
    model
      .load_file(
        model_path,
        &DefaultFileRecorder::<FullPrecisionSettings>::new(),
        &device,
      )
      .expect("Failed to load model file.")
  });
  let mut input = BufReader::new(io::stdin());
  let mut output = io::stdout();
  let mut state_option = None;
  let mut s = String::new();
  loop {
    s.clear();
    input.read_line(&mut s)?;
    let request = serde_json::from_str(&s)?;

    let response = match request {
      Request::Init { width, height } => {
        let mut rng = make_rng::<SmallRng>();
        let model: CliModel<B> = match &model {
          Some(model) => Either::Right(Predictor {
            model: model.clone(),
            device: device.clone(),
          }),
          None => Either::Left(()),
        };
        state_option = Some(State::<B> {
          field: Field::new_from_rng(width, height, &mut rng),
          rng,
          oppai: Oppai::new(width, height, config.ai.clone(), patterns.clone(), model),
        });
        Response::Init
      }
      Request::PutPoint { coords, player } => {
        let state = state_option.as_mut().ok_or(anyhow::anyhow!("Not initialized"))?;
        let pos = state.field.to_pos(coords.x, coords.y);
        let put = state.field.put_point(pos, player);
        Response::PutPoint { put }
      }
      Request::Undo => {
        let state = state_option.as_mut().ok_or(anyhow::anyhow!("Not initialized"))?;
        let undone = state.field.undo();
        Response::Undo { undone }
      }
      Request::Analyze {
        player,
        constraint: Constraint::Time(time),
      } => {
        let state = state_option.as_mut().ok_or(anyhow::anyhow!("Not initialized"))?;
        let mut oppai = TimeLimitedAI(time, &mut state.oppai);
        let analysis = oppai.analyze(&mut state.rng, &mut state.field, player, None, &|| false);
        let mut moves: Vec<Move> = analysis
          .moves()
          .map(|(pos, weight)| Move {
            coords: Coords {
              x: state.field.to_x(pos),
              y: state.field.to_y(pos),
            },
            weight: weight.to_f64().unwrap_or_default(),
          })
          .collect();
        moves.sort_unstable_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        Response::Analyze { moves }
      }
      Request::Analyze {
        player,
        constraint: Constraint::Complexity(complexity),
      } => {
        let state = state_option.as_mut().ok_or(anyhow::anyhow!("Not initialized"))?;
        let confidence = InConfidence {
          minimax_depth: (8.0 * complexity).round() as u32,
          uct_iterations: (100_000.0 * complexity).round() as usize,
          zero_iterations: (1_000.0 * complexity).round() as usize,
        };
        let analysis = state
          .oppai
          .analyze(&mut state.rng, &mut state.field, player, Some(confidence), &|| false);
        let mut moves: Vec<Move> = analysis
          .moves()
          .map(|(pos, weight)| Move {
            coords: Coords {
              x: state.field.to_x(pos),
              y: state.field.to_y(pos),
            },
            weight: weight.to_f64().unwrap_or_default(),
          })
          .collect();
        moves.sort_unstable_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
        Response::Analyze { moves }
      }
    };

    writeln!(&mut output, "{}", serde_json::to_string(&response)?)?;
    output.flush()?;
  }
}

fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();
  let config = cli_parse();
  let patterns = config
    .patterns_cache
    .as_ref()
    .filter(|patterns_cache| Path::new(patterns_cache).exists())
    .map(|patterns_cache| {
      let mut file = File::open(patterns_cache).expect("Failed to open patterns cache file.");
      let mut buffer = Vec::new();
      file
        .read_to_end(&mut buffer)
        .expect("Failed to read patterns cache file.");
      postcard::from_bytes(&buffer).expect("Failed to deserialize patterns cache file.")
    })
    .unwrap_or_else(|| {
      if config.patterns.is_empty() {
        Patterns::default()
      } else {
        Patterns::from_files(
          config
            .patterns
            .iter()
            .map(|path| File::open(path).expect("Failed to open patterns file.")),
        )
        .expect("Failed to read patterns file.")
      }
    });
  if let Some(patterns_cache) = config.patterns_cache.as_ref()
    && !Path::new(patterns_cache).exists()
  {
    let buffer = postcard::to_stdvec(&patterns).expect("Failed to serialize patterns cache file.");
    std::fs::write(patterns_cache, buffer).expect("Failed to write patterns cache file.");
  }
  let patterns_arc = Arc::new(patterns);

  match config.backend {
    #[cfg(feature = "cuda")]
    ConfigBackend::Cuda => run::<Cuda>(config, patterns_arc, CudaDevice::default()),
    #[cfg(feature = "ndarray")]
    ConfigBackend::Ndarray => run::<NdArray>(config, patterns_arc, NdArrayDevice::Cpu),
    #[cfg(feature = "rocm")]
    ConfigBackend::Rocm => run::<Rocm>(config, patterns_arc, RocmDevice::default()),
    #[cfg(any(feature = "vulkan", feature = "webgpu"))]
    ConfigBackend::Wgpu => run::<Wgpu>(config, patterns_arc, WgpuDevice::DefaultDevice),
  }
}
