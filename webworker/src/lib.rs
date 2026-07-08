#![allow(clippy::too_many_arguments)]
#![allow(clippy::cognitive_complexity)]

use anyhow::{Result, anyhow};
use burn::backend::wgpu::{Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async};
use burn::{
  module::Module,
  record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
};
use futures::{StreamExt, channel::mpsc};
use oppai_ai::{ai::AI, analysis::Analysis};
use oppai_ais::{
  oppai::{Config as AIConfig, InConfidence, Oppai, Solver},
  time_limited_ai::TimeLimitedAI,
};
use oppai_field::field::Field;
use oppai_patterns::patterns::Patterns;
use oppai_protocol::{Constraint, Coords, Move, Request, Response};
use oppai_zero_burn::model::{Model as BurnModel, ModelConfig, Predictor};
use rand::{make_rng, rngs::SmallRng};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

struct State {
  field: Field,
  rng: SmallRng,
  oppai: Oppai<f32, Predictor<Wgpu>>,
}

async fn download_bytes(url: &str) -> Result<Vec<u8>> {
  let bytes = reqwest::get(url).await?.bytes().await?;
  Ok(bytes.to_vec())
}

async fn download_config(url: &str) -> Result<ModelConfig> {
  Ok(reqwest::get(url).await?.json().await?)
}

async fn handle(
  state_option: &mut Option<State>,
  patterns: &Arc<Patterns>,
  request: Request,
  config: &ModelConfig,
  model_bytes: &[u8],
) -> Result<Response> {
  Ok(match request {
    Request::Init { width, height } => {
      let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
        .load(model_bytes.to_vec(), &WgpuDevice::DefaultDevice)?;
      let model = BurnModel::<Wgpu>::new(&WgpuDevice::DefaultDevice, config).load_record(record);
      let predictor = Predictor {
        model,
        device: WgpuDevice::DefaultDevice,
      };
      let mut rng = make_rng::<SmallRng>();
      let config = AIConfig {
        solver: Solver::Zero,
        ladders: false,
        ..AIConfig::default()
      };
      *state_option = Some(State {
        field: Field::new_from_rng(width, height, &mut rng),
        rng,
        oppai: Oppai::new(width, height, config, patterns.clone(), predictor),
      });
      Response::Init
    }
    Request::PutPoint { coords, player } => {
      let state = state_option.as_mut().ok_or(anyhow!("Not initialized"))?;
      let pos = state.field.to_pos(coords.x, coords.y);
      let put = state.field.put_point(pos, player);
      if put {
        state.field.update_grounded();
      }
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
      let analysis = oppai
        .analyze(&mut state.rng, &mut state.field, player, None, &|| false)
        .await;
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
        .analyze(&mut state.rng, &mut state.field, player, Some(confidence), &|| false)
        .await;
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

async fn process(
  scope: &DedicatedWorkerGlobalScope,
  state: &mut Option<State>,
  patterns: &Arc<Patterns>,
  config: &ModelConfig,
  model_bytes: &[u8],
  message: String,
) {
  let result = match serde_json::from_str(&message) {
    Ok(request) => handle(state, patterns, request, config, model_bytes).await,
    Err(error) => Err(anyhow::Error::from(error)),
  };
  let result = result.and_then(|response| serde_json::to_string(&response).map_err(anyhow::Error::from));
  match result {
    Ok(response) => scope.post_message(&JsValue::from_str(&response)).unwrap(),
    Err(error) => web_sys::console::error_1(&error.to_string().into()),
  }
}

#[wasm_bindgen(start)]
pub fn run() {
  console_error_panic_hook::set_once();
  wasm_logger::init(wasm_logger::Config::default());

  log::info!("Initializing OpPAI worker");

  let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));

  // Messages arriving before the model is downloaded are buffered in the
  // channel and processed strictly in order by the single consumer task below.
  let (sender, mut receiver) = mpsc::unbounded();

  let callback = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
    let Some(message) = event.data().as_string() else {
      web_sys::console::error_1(&"Expected a string message".into());
      return;
    };
    let _ = sender.unbounded_send(message);
  });
  scope.set_onmessage(Some(callback.as_ref().unchecked_ref()));
  callback.forget();

  wasm_bindgen_futures::spawn_local(async move {
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), Default::default()).await;

    let (config, model_bytes) = futures::future::join(
      download_config("https://kropki.org/model.json"),
      download_bytes("https://kropki.org/model.mpk"),
    )
    .await;
    let config = match config {
      Ok(config) => config,
      Err(error) => {
        web_sys::console::error_1(&format!("Failed to download the model config: {}", error).into());
        return;
      }
    };
    let model_bytes = match model_bytes {
      Ok(model_bytes) => model_bytes,
      Err(error) => {
        web_sys::console::error_1(&format!("Failed to download the model: {}", error).into());
        return;
      }
    };
    log::info!("Model is loaded");

    let patterns = Arc::new(Patterns::default());
    let mut state = None;
    while let Some(message) = receiver.next().await {
      process(&scope, &mut state, &patterns, &config, &model_bytes, message).await;
    }
  });
}
