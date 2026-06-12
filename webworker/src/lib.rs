#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

use anyhow::{Result, anyhow};
use oppai_ai::{ai::AI, analysis::Analysis};
use oppai_ais::{
  oppai::{Config as AIConfig, InConfidence, Oppai},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::field::Field;
use oppai_patterns::patterns::Patterns;
use oppai_protocol::{Constraint, Coords, Move, Request, Response};
use rand::{make_rng, rngs::SmallRng};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

struct State {
  field: Field,
  rng: SmallRng,
  oppai: Oppai<f32, ()>,
}

fn handle(state_option: &mut Option<State>, patterns: &Arc<Patterns>, request: Request) -> Result<Response> {
  Ok(match request {
    Request::Init { width, height } => {
      let mut rng = make_rng::<SmallRng>();
      *state_option = Some(State {
        field: Field::new_from_rng(width, height, &mut rng),
        rng,
        oppai: Oppai::new(width, height, AIConfig::default(), patterns.clone(), ()),
      });
      Response::Init
    }
    Request::PutPoint { coords, player } => {
      let state = state_option.as_mut().ok_or(anyhow!("Not initialized"))?;
      let pos = state.field.to_pos(coords.x, coords.y);
      let put = state.field.put_point(pos, player);
      Response::PutPoint { put }
    }
    Request::Undo => {
      let state = state_option.as_mut().ok_or(anyhow!("Not initialized"))?;
      let undone = state.field.undo();
      Response::Undo { undone }
    }
    Request::Analyze {
      player,
      constraint: Constraint::Time(time),
    } => {
      let state = state_option.as_mut().ok_or(anyhow!("Not initialized"))?;
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
      let state = state_option.as_mut().ok_or(anyhow!("Not initialized"))?;
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
  })
}

#[wasm_bindgen(start)]
pub fn run() {
  console_error_panic_hook::set_once();
  wasm_logger::init(wasm_logger::Config::default());

  log::info!("Initializing OpPAI worker");

  let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));
  let scope_clone = scope.clone();

  let patterns = Arc::new(Patterns::default());
  let mut state_option = None;

  let callback = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
    let result = event
      .data()
      .as_string()
      .ok_or(anyhow!("Expected a string message"))
      .and_then(|s| serde_json::from_str(&s).map_err(anyhow::Error::from))
      .and_then(|request| handle(&mut state_option, &patterns, request))
      .and_then(|response| serde_json::to_string(&response).map_err(anyhow::Error::from));
    match result {
      Ok(response) => scope_clone.post_message(&JsValue::from_str(&response)).unwrap(),
      Err(error) => web_sys::console::error_1(&error.to_string().into()),
    }
  });
  scope.set_onmessage(Some(callback.as_ref().unchecked_ref()));
  callback.forget();
}
