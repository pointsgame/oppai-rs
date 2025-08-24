#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

mod config;

use crate::config::cli_parse;
use anyhow::Result;
use oppai_ai::{ai::AI, analysis::Analysis};
use oppai_ais::{
  oppai::{InConfidence, Oppai},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::{
  field::{Field, length},
  zobrist::Zobrist,
};
use oppai_patterns::patterns::Patterns;
use oppai_protocol::{Constraint, Coords, Move, Request, Response};
use rand::{SeedableRng, rngs::SmallRng};
use std::{
  default::Default,
  fs::File,
  io::{self, BufRead, BufReader, Read, Write},
  path::Path,
  sync::Arc,
};

struct State {
  field: Field,
  rng: SmallRng,
  oppai: Oppai<f32, ()>,
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
        let mut rng = SmallRng::from_os_rng();
        let zobrist = Arc::new(Zobrist::new(length(width, height) * 2, &mut rng));
        state_option = Some(State {
          field: Field::new(width, height, zobrist),
          rng,
          oppai: Oppai::new(width, height, config.ai.clone(), patterns_arc.clone(), ()),
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
        let moves = analysis
          .moves()
          .map(|(pos, weight)| Move {
            coords: Coords {
              x: state.field.to_x(pos),
              y: state.field.to_y(pos),
            },
            weight: weight.to_f64().unwrap_or_default(),
          })
          .collect();
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
        let moves = analysis
          .moves()
          .map(|(pos, weight)| Move {
            coords: Coords {
              x: state.field.to_x(pos),
              y: state.field.to_y(pos),
            },
            weight: weight.to_f64().unwrap_or_default(),
          })
          .collect();
        Response::Analyze { moves }
      }
    };

    writeln!(&mut output, "{}", serde_json::to_string(&response)?)?;
    output.flush()?;
  }
}
