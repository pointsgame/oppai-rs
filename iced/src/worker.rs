#[cfg(not(target_arch = "wasm32"))]
fn main() {
  panic!("not available");
}

#[cfg(target_arch = "wasm32")]
fn main() {
  mod worker_message;

  use oppai_bot::bot::Bot;
  use oppai_bot::config::Config as BotConfig;
  use oppai_bot::patterns::Patterns;
  use rand::rngs::SmallRng;
  use rand::SeedableRng;
  use std::sync::Arc;
  use std::{sync::atomic::AtomicBool, unreachable};
  use wasm_bindgen::prelude::*;
  use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};
  use worker_message::{Request, Response};

  console_error_panic_hook::set_once();
  web_sys::console::log_1(&"Initializing OpPAI worker".into());

  let scope = DedicatedWorkerGlobalScope::from(JsValue::from(js_sys::global()));
  let scope_clone = scope.clone();

  let mut bot: Option<Bot<SmallRng>> = None;

  let callback = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
    let request: Request = serde_wasm_bindgen::from_value(event.data()).unwrap();

    if let Request::New(width, height) = request {
      bot = Some(Bot::new(
        width,
        height,
        SmallRng::from_seed([1; 16]),
        Arc::new(Patterns::default()),
        BotConfig::default(),
      ))
    }

    let bot = if let Some(bot) = bot.as_mut() {
      bot
    } else {
      scope_clone
        .post_message(&serde_wasm_bindgen::to_value(&Response::Init).unwrap())
        .unwrap();
      return;
    };

    match request {
      Request::PutPoint(pos, player) => {
        bot.field.put_point(pos, player);
      }
      Request::Undo => {
        bot.field.undo();
      }
      Request::BestMove(player) => {
        let pos = bot
          .best_move(player, 100000, 6, &AtomicBool::new(false))
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
