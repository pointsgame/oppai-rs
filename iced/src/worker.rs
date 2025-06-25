#[cfg(target_arch = "wasm32")]
mod worker_message;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
  panic!("not available");
}

#[cfg(target_arch = "wasm32")]
fn main() {
  use oppai_ai::{ai::AI, analysis::Analysis};
  use oppai_ais::{
    oppai::{Config as AIConfig, Oppai},
    time_limited_ai::TimeLimitedAI,
  };
  use oppai_field::{
    field::{Field, length},
    zobrist::Zobrist,
  };
  use oppai_patterns::patterns::Patterns;
  use rand::SeedableRng;
  use rand::rngs::SmallRng;
  use std::sync::Arc;
  use std::time::Duration;
  use wasm_bindgen::prelude::*;
  use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};
  use worker_message::{Request, Response};

  struct State {
    field: Field,
    rng: SmallRng,
    oppai: Oppai<f32, ()>,
  }

  console_error_panic_hook::set_once();
  web_sys::console::log_1(&"Initializing OpPAI worker".into());

  let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));
  let scope_clone = scope.clone();

  let mut state: Option<State> = None;

  let callback = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
    let request: Request = serde_wasm_bindgen::from_value(event.data()).unwrap();

    if let Request::New(width, height) = request {
      let mut rng = SmallRng::from_seed([1; 32]);
      let zobrist = Arc::new(Zobrist::new(length(width, height) * 2, &mut rng));
      state = Some(State {
        field: Field::new(width, height, zobrist),
        rng,
        oppai: Oppai::new(width, height, AIConfig::default(), Arc::new(Patterns::default()), ()),
      })
    }

    let state = if let Some(state) = state.as_mut() {
      state
    } else {
      scope_clone
        .post_message(&serde_wasm_bindgen::to_value(&Response::Init).unwrap())
        .unwrap();
      return;
    };

    match request {
      Request::PutPoint(pos, player) => {
        state.field.put_point(pos, player);
      }
      Request::Undo => {
        state.field.undo();
      }
      Request::UndoAll => {
        state.field.undo_all();
      }
      Request::BestMove(player) => {
        let mut oppai = TimeLimitedAI(Duration::from_secs(5), &mut state.oppai);
        let pos = oppai
          .analyze(&mut state.rng, &mut state.field, player, None, &|| false)
          .best_move(&mut state.rng)
          .map_or(0, |pos| pos.get());
        scope_clone
          .post_message(&serde_wasm_bindgen::to_value(&Response::BestMove(pos)).unwrap())
          .unwrap();
      }
      Request::New(_, _) => unreachable!(),
    }
  });
  scope.set_onmessage(Some(callback.as_ref().unchecked_ref()));
  callback.forget();

  scope
    .post_message(&serde_wasm_bindgen::to_value(&Response::Init).unwrap())
    .unwrap();
}
