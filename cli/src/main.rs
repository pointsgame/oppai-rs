#![recursion_limit = "256"]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

mod config;

use crate::config::{Config, cli_parse};
use anyhow::Result;
use burn::{
  module::Module,
  tensor::{DeviceConfig, FloatDType},
};
use either::Either;
use oppai_ai::{ai::AI, analysis::Analysis};
use oppai_ais::{
  oppai::{Config as AIConfig, InConfidence, Oppai},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::field::Field;
use oppai_patterns::patterns::Patterns;
use oppai_protocol::{Constraint, Coords, Move, Request, Response};
use oppai_zero_burn::model::{Model as BurnModel, Predictor};
use rand::{make_rng, rngs::SmallRng};
use std::{
  default::Default,
  fs::File,
  io::{self, BufRead, Read, Write},
  path::Path,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc,
  },
  thread,
};

type CliModel = Either<(), Predictor>;

struct State {
  field: Field,
  rng: SmallRng,
  oppai: Oppai<f32, CliModel>,
}

fn run(config: Config, patterns: Arc<Patterns>) -> Result<()> {
  let mut device = config.device.clone();
  if config.half {
    // Before the model is loaded, so that its weights are converted too.
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F16))?;
  }
  let model = config.model.as_ref().map(|model_path| {
    BurnModel::new(&device, &config.model_config)
      .try_load_file(model_path)
      .expect("Failed to load model file.")
  });
  // Requests are read on a separate thread, so that `Stop` takes effect while
  // an analysis is running rather than after it. Analyses are numbered in the
  // order they are read, and `Stop` marks all of them up to the last one read
  // as stopped - this way it's never lost or applied to a later analysis,
  // however the requests and the responses interleave.
  let stopped = Arc::new(AtomicU64::new(0));
  let (sender, receiver) = mpsc::channel::<Result<(Request, u64)>>();
  thread::spawn({
    let stopped = stopped.clone();
    move || {
      let mut requested = 0;
      for line in io::stdin().lock().lines() {
        let request = line
          .map_err(anyhow::Error::from)
          .and_then(|line| serde_json::from_str::<Request>(&line).map_err(anyhow::Error::from));
        match request {
          Ok(Request::Stop) => stopped.store(requested, Ordering::Relaxed),
          Ok(request) => {
            if let Request::Analyze { .. } = request {
              requested += 1;
            }
            if sender.send(Ok((request, requested))).is_err() {
              return;
            }
          }
          Err(error) => {
            let _ = sender.send(Err(error));
            return;
          }
        }
      }
      // Stdin is closed, so nobody is waiting for the running analysis.
      stopped.store(u64::MAX, Ordering::Relaxed);
    }
  });
  let mut output = io::stdout();
  let mut state_option = None;
  for request in receiver {
    let (request, analysis_id) = request?;
    let should_stop = || stopped.load(Ordering::Relaxed) >= analysis_id;

    let response = match request {
      Request::Init {
        width,
        height,
        komi_x_2,
      } => {
        let mut rng = make_rng::<SmallRng>();
        let model: CliModel = match &model {
          Some(model) => Either::Right(Predictor {
            model: model.clone(),
            device: device.clone(),
          }),
          None => Either::Left(()),
        };
        state_option = Some(State {
          field: Field::new_from_rng(width, height, &mut rng),
          rng,
          oppai: Oppai::new(
            width,
            height,
            AIConfig {
              komi_x_2,
              ..config.ai.clone()
            },
            patterns.clone(),
            model,
          ),
        });
        Response::Init
      }
      Request::PutPoint { coords, player } => {
        let state = state_option.as_mut().ok_or(anyhow::anyhow!("Not initialized"))?;
        let pos = state.field.to_pos(coords.x, coords.y);
        let put = state.field.put_point(pos, player);
        if put {
          state.field.update_grounded();
        }
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
        let analysis =
          futures::executor::block_on(oppai.analyze(&mut state.rng, &mut state.field, player, None, &should_stop));
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
          uct_iterations: (100_000.0 * complexity).round() as u32,
          zero_iterations: (1_000.0 * complexity).round() as u32,
        };
        let analysis = futures::executor::block_on(state.oppai.analyze(
          &mut state.rng,
          &mut state.field,
          player,
          Some(confidence),
          &should_stop,
        ));
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
      // Handled by the stdin thread as soon as it's read.
      Request::Stop => continue,
    };

    writeln!(&mut output, "{}", serde_json::to_string(&response)?)?;
    output.flush()?;
  }
  Ok(())
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

  run(config, patterns_arc)
}
