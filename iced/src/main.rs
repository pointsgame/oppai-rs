mod canvas_config;
mod canvas_field;
mod config;
#[cfg(target_arch = "wasm32")]
mod worker_message;

use crate::config::{cli_parse, Config};
#[cfg(target_arch = "wasm32")]
use crate::worker_message::{Request, Response};
use canvas_field::{CanvasField, CanvasMessage, Label};
#[cfg(target_arch = "wasm32")]
use iced::subscription;
use iced::theme::Palette;
use iced::widget::{Canvas, Column, Container, Row, Text};
use iced::{
  event, executor, keyboard, window, Application, Color, Command, Element, Event, Length, Settings, Subscription, Theme,
};
#[cfg(not(target_arch = "wasm32"))]
use oppai_ai::ai::AI;
#[cfg(not(target_arch = "wasm32"))]
use oppai_ai::analysis::Analysis;
#[cfg(not(target_arch = "wasm32"))]
use oppai_ais::{oppai::Oppai, time_limited_ai::TimeLimitedAI};
use oppai_field::extended_field::ExtendedField;
use oppai_field::field::{NonZeroPos, Pos};
use oppai_field::player::Player;
#[cfg(not(target_arch = "wasm32"))]
use oppai_patterns::patterns::Patterns;
use oppai_sgf::visits::sgf_to_visits;
use oppai_sgf::{from_sgf, to_sgf_str};
use oppai_zero::episode::Visits;
use rand::rngs::SmallRng;
#[cfg(not(target_arch = "wasm32"))]
use rand::Rng;
use rand::SeedableRng;
use rfd::AsyncFileDialog;
#[cfg(not(target_arch = "wasm32"))]
use rfd::FileHandle;
use sgf_parse::GameTree;
use std::io::Read;
use std::iter;
#[cfg(not(target_arch = "wasm32"))]
use std::ops::DerefMut;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
  atomic::{AtomicBool, Ordering},
  Arc,
};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, fs::File, sync::Mutex};

pub fn main() -> iced::Result {
  #[cfg(not(target_arch = "wasm32"))]
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  #[cfg(not(target_arch = "wasm32"))]
  env_logger::Builder::from_env(env).init();

  #[cfg(target_arch = "wasm32")]
  console_error_panic_hook::set_once();
  #[cfg(target_arch = "wasm32")]
  wasm_logger::init(wasm_logger::Config::default());

  let config = cli_parse();

  Game::run(Settings {
    antialiasing: true,
    flags: config,
    window: window::Settings {
      icon: Some(window::icon::from_file_data(include_bytes!("../../resources/Logo.png"), None).unwrap()),
      ..Default::default()
    },
    ..Settings::default()
  })
}

struct Game {
  config: Config,
  rng: SmallRng,
  moves: Vec<(Pos, Player, Visits)>,
  canvas_field: CanvasField<Vec<Label>>,
  #[cfg(not(target_arch = "wasm32"))]
  oppai: Arc<Mutex<Oppai<f32, ()>>>,
  #[cfg(target_arch = "wasm32")]
  worker: web_sys::Worker,
  ai: bool,
  thinking: bool,
  coordinates: Option<(u32, u32)>,
  #[cfg(not(target_arch = "wasm32"))]
  should_stop: Arc<AtomicBool>,
}

impl Game {
  #[cfg(target_arch = "wasm32")]
  fn send_worker_message(&self, message: Request) {
    self
      .worker
      .post_message(&serde_wasm_bindgen::to_value(&message).unwrap())
      .unwrap();
  }

  pub fn put_point(&mut self, pos: Pos) -> bool {
    let player = self.canvas_field.extended_field.player;
    self.put_players_point(pos, player)
  }

  fn refresh(&mut self) {
    self.canvas_field.extra.clear();
    let moves_count = self.canvas_field.extended_field.field.moves_count();
    if moves_count > 0 {
      let visits = &self.moves[moves_count - 1].2;
      let max = visits.max() as f32;
      self
        .canvas_field
        .extra
        .extend(visits.0.iter().map(|&(pos, visits)| Label {
          pos,
          text: visits.to_string(),
          color: Color {
            r: 0.0,
            g: visits as f32 / max,
            b: 0.0,
            a: 1.0,
          },
          scale: 0.5,
        }));
    }
    self.canvas_field.field_cache.clear();
  }

  pub fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.canvas_field.extended_field.put_players_point(pos, player) {
      let moves_count = self.canvas_field.extended_field.field.moves_count();
      if self
        .moves
        .get(moves_count - 1)
        .map_or(true, |&(cur_pos, cur_player, _)| (cur_pos, cur_player) != (pos, player))
      {
        self.moves.truncate(moves_count - 1);
        self.moves.push((pos, player, Default::default()));
      }
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::PutPoint(pos, player));
      self.refresh();
      true
    } else {
      false
    }
  }

  pub fn undo(&mut self) -> bool {
    if self.canvas_field.extended_field.undo() {
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::Undo);
      self.refresh();
      true
    } else {
      false
    }
  }

  pub fn redo(&mut self) -> bool {
    let moves_count = self.canvas_field.extended_field.field.moves_count();
    if self.moves.len() > moves_count {
      let (pos, player, _) = self.moves[moves_count];
      self.put_players_point(pos, player);
      true
    } else {
      false
    }
  }

  pub fn redo_all(&mut self) -> bool {
    let moves_count = self.canvas_field.extended_field.field.moves_count();
    if self.moves.len() > moves_count {
      for &(pos, player, _) in &self.moves[moves_count..] {
        self.canvas_field.extended_field.put_players_point(pos, player);
      }
      self.put_all_bot_points();
      self.refresh();
      true
    } else {
      false
    }
  }

  pub fn undo_all(&mut self) -> bool {
    if self.canvas_field.extended_field.undo() {
      self.canvas_field.extended_field.clear();
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::UndoAll);
      self.refresh();
      true
    } else {
      false
    }
  }

  #[cfg(not(target_arch = "wasm32"))]
  pub fn put_all_bot_points(&self) {}

  #[cfg(target_arch = "wasm32")]
  pub fn put_all_bot_points(&self) {
    for &pos in self.canvas_field.extended_field.field.moves() {
      let player = self.canvas_field.extended_field.field.cell(pos).get_player();
      self.send_worker_message(Request::PutPoint(pos, player));
    }
  }

  pub fn is_locked(&self) -> bool {
    self.thinking
  }
}

#[derive(Debug)]
enum Message {
  Canvas(CanvasMessage),
  Undo,
  UndoAll,
  Redo,
  RedoAll,
  New,
  Open,
  Save,
  ToggleEditMode,
  ToggleAI,
  Interrupt,
  BotMove(Option<NonZeroPos>),
  #[cfg(not(target_arch = "wasm32"))]
  OpenFile(Option<FileHandle>),
  #[cfg(target_arch = "wasm32")]
  OpenFile(Option<Vec<u8>>),
  #[cfg(not(target_arch = "wasm32"))]
  SaveFile(Option<FileHandle>),
  #[cfg(target_arch = "wasm32")]
  SetWorkerListener(iced::futures::channel::mpsc::UnboundedSender<Message>),
  #[cfg(target_arch = "wasm32")]
  InitWorker,
}

impl Application for Game {
  type Executor = executor::Default;
  type Message = Message;
  type Flags = Config;
  type Theme = Theme;

  fn new(flags: Config) -> (Self, Command<Self::Message>) {
    let mut rng = SmallRng::from_entropy();
    let mut extended_field = ExtendedField::new_from_rng(flags.width, flags.height, &mut rng);
    // TODO: store patterns to use them when new Oppai is created
    #[cfg(not(target_arch = "wasm32"))]
    let patterns = flags
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
        if flags.patterns.is_empty() {
          Patterns::default()
        } else {
          Patterns::from_files(
            flags
              .patterns
              .iter()
              .map(|path| File::open(path).expect("Failed to open patterns file.")),
          )
          .expect("Failed to read patterns file.")
        }
      });
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(patterns_cache) = flags.patterns_cache.as_ref() {
      if !Path::new(patterns_cache).exists() {
        let buffer = postcard::to_stdvec(&patterns).expect("Failed to serialize patterns cache file.");
        std::fs::write(patterns_cache, buffer).expect("Failed to write patterns cache file.");
      }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let oppai = Oppai::new(
      flags.width,
      flags.height,
      flags.ai_config.clone(),
      Arc::new(patterns),
      (),
    );
    let moves = flags.initial_position.points(
      extended_field.field.width(),
      extended_field.field.height(),
      extended_field.player,
    );
    extended_field.put_points(moves.clone());
    let game = Game {
      config: flags.clone(),
      rng,
      moves: moves.map(|(pos, player)| (pos, player, Default::default())).collect(),
      canvas_field: CanvasField {
        extended_field,
        field_cache: Default::default(),
        edit_mode: false,
        // TODO: split configs
        config: flags.canvas_config,
        extra: Vec::new(),
      },
      #[cfg(not(target_arch = "wasm32"))]
      oppai: Arc::new(Mutex::new(oppai)),
      #[cfg(target_arch = "wasm32")]
      worker: {
        const NAME: &'static str = "worker";
        let origin = web_sys::window().unwrap().location().origin().unwrap();
        let script = js_sys::Array::new();
        script.push(&format!(r#"importScripts("{origin}/{NAME}.js");wasm_bindgen("{origin}/{NAME}_bg.wasm");"#).into());
        let blob = web_sys::Blob::new_with_str_sequence_and_options(
          &script,
          web_sys::BlobPropertyBag::new().type_("text/javascript"),
        )
        .unwrap();

        let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();
        web_sys::Worker::new(&url).unwrap()
      },
      ai: true,
      thinking: false,
      coordinates: None,
      #[cfg(not(target_arch = "wasm32"))]
      should_stop: Arc::new(AtomicBool::new(false)),
    };
    game.put_all_bot_points();

    (game, Command::none())
  }

  fn title(&self) -> String {
    "OpPAI".into()
  }

  fn subscription(&self) -> Subscription<Message> {
    #[cfg(target_arch = "wasm32")]
    let worker_subscription = {
      struct WorkerListener;
      enum State {
        Starting,
        Ready(iced::futures::channel::mpsc::UnboundedReceiver<Message>),
      }
      subscription::channel(std::any::TypeId::of::<WorkerListener>(), 16, |mut output| async move {
        use iced::futures::{sink::SinkExt, StreamExt};
        let mut state = State::Starting;
        loop {
          match &mut state {
            State::Starting => {
              let (tx, rx) = iced::futures::channel::mpsc::unbounded();
              output.send(Message::SetWorkerListener(tx)).await.unwrap();
              state = State::Ready(rx);
            }
            State::Ready(rx) => {
              let message = rx.select_next_some().await;
              output.send(message).await.unwrap();
            }
          }
        }
      })
    };

    let keys_subscription = event::listen_with(|event, _| match event {
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Named(keyboard::key::Named::ArrowLeft),
        ..
      }) => Some(Message::Undo),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Named(keyboard::key::Named::ArrowDown),
        ..
      }) => Some(Message::UndoAll),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Named(keyboard::key::Named::ArrowRight),
        ..
      }) => Some(Message::Redo),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Named(keyboard::key::Named::ArrowUp),
        ..
      }) => Some(Message::RedoAll),
      #[cfg(not(target_arch = "wasm32"))]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        modifiers,
        ..
      }) if modifiers.control() && c.as_str() == "n" => Some(Message::New),
      #[cfg(not(target_arch = "wasm32"))]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        modifiers,
        ..
      }) if modifiers.control() && c.as_str() == "o" => Some(Message::Open),
      #[cfg(not(target_arch = "wasm32"))]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        modifiers,
        ..
      }) if modifiers.control() && c.as_str() == "s" => Some(Message::Save),
      #[cfg(target_arch = "wasm32")]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        ..
      }) if c.as_str() == "n" => Some(Message::New),
      #[cfg(target_arch = "wasm32")]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        ..
      }) if c.as_str() == "s" => Some(Message::Save),
      #[cfg(target_arch = "wasm32")]
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        ..
      }) if c.as_str() == "o" => Some(Message::Open),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        ..
      }) if c.as_str() == "e" => Some(Message::ToggleEditMode),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Character(c),
        ..
      }) if c.as_str() == "a" => Some(Message::ToggleAI),
      Event::Keyboard(keyboard::Event::KeyPressed {
        key: keyboard::Key::Named(keyboard::key::Named::Escape),
        ..
      }) => Some(Message::Interrupt),
      _ => None,
    });

    #[cfg(target_arch = "wasm32")]
    let subscription = Subscription::batch([keys_subscription, worker_subscription]);
    #[cfg(not(target_arch = "wasm32"))]
    let subscription = keys_subscription;

    subscription
  }

  fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
    match message {
      Message::Canvas(CanvasMessage::PutPoint(pos)) => {
        if self.is_locked() {
          return Command::none();
        }
        if self.put_point(pos) && self.ai {
          self.thinking = true;

          let player = self.canvas_field.extended_field.player;

          #[cfg(not(target_arch = "wasm32"))]
          {
            let oppai = self.oppai.clone();
            let mut rng = SmallRng::from_seed(self.rng.gen());
            let mut field = self.canvas_field.extended_field.field.clone();
            let time = self.config.time;
            let should_stop = self.should_stop.clone();
            return Command::perform(
              async move {
                let mut oppai = oppai.lock().unwrap();
                let mut oppai = TimeLimitedAI(time, oppai.deref_mut());
                oppai
                  .analyze(&mut rng, &mut field, player, None, &|| {
                    should_stop.load(Ordering::Relaxed)
                  })
                  .best_move(&mut rng)
              },
              Message::BotMove,
            );
          }

          #[cfg(target_arch = "wasm32")]
          self.send_worker_message(Request::BestMove(player));
        }
      }
      Message::Canvas(CanvasMessage::PutPlayersPoint(pos, player)) => {
        if self.is_locked() {
          return Command::none();
        }
        self.put_players_point(pos, player);
      }
      Message::Canvas(CanvasMessage::ChangeCoordinates(x, y)) => {
        self.coordinates = Some((x, y));
      }
      Message::Canvas(CanvasMessage::ClearCoordinates) => {
        self.coordinates = None;
      }
      Message::Undo => {
        if self.is_locked() {
          return Command::none();
        }
        self.undo();
      }
      Message::UndoAll => {
        if self.is_locked() {
          return Command::none();
        }
        self.undo_all();
      }
      Message::Redo => {
        if self.is_locked() {
          return Command::none();
        }
        self.redo();
      }
      Message::RedoAll => {
        if self.is_locked() {
          return Command::none();
        }
        self.redo_all();
      }
      Message::New => {
        if self.is_locked() {
          return Command::none();
        }
        self.canvas_field.extended_field =
          ExtendedField::new_from_rng(self.config.width, self.config.height, &mut self.rng);
        #[cfg(not(target_arch = "wasm32"))]
        {
          self.oppai = Arc::new(Mutex::new(Oppai::new(
            self.config.width,
            self.config.height,
            self.config.ai_config.clone(),
            Arc::new(Patterns::default()),
            (),
          )));
        }
        #[cfg(target_arch = "wasm32")]
        self.send_worker_message(Request::New(
          self.canvas_field.extended_field.field.width(),
          self.canvas_field.extended_field.field.height(),
        ));
        self
          .canvas_field
          .extended_field
          .put_points(self.config.initial_position.points(
            self.canvas_field.extended_field.field.width(),
            self.canvas_field.extended_field.field.height(),
            self.canvas_field.extended_field.player,
          ));
        self.put_all_bot_points();
        self.refresh();
      }
      Message::Open => {
        if self.is_locked() {
          return Command::none();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          return Command::perform(
            AsyncFileDialog::new().add_filter("SGF", &["sgf"]).pick_file(),
            Message::OpenFile,
          );
        }
        #[cfg(target_arch = "wasm32")]
        {
          return Command::perform(
            async {
              let maybe_file = AsyncFileDialog::new().add_filter("SGF", &["sgf"]).pick_file().await;
              if let Some(file) = maybe_file {
                Some(file.read().await)
              } else {
                None
              }
            },
            Message::OpenFile,
          );
        }
      }
      #[cfg(not(target_arch = "wasm32"))]
      Message::Save => {
        return Command::perform(
          AsyncFileDialog::new().add_filter("SGF", &["sgf"]).save_file(),
          Message::SaveFile,
        );
      }
      #[cfg(target_arch = "wasm32")]
      Message::Save => {
        if let Some(s) = to_sgf_str(&self.canvas_field.extended_field) {
          use wasm_bindgen::JsCast;
          let window = web_sys::window().unwrap();
          let document = window.document().unwrap();
          let element = document.create_element("a").unwrap();
          let element = element.dyn_into::<web_sys::HtmlElement>().unwrap();
          element
            .set_attribute(
              "href",
              &(String::from("data:text/plain;charset=utf-8,") + &String::from(js_sys::encode_uri_component(&s))),
            )
            .unwrap();
          element.set_attribute("download", "game.sgf").unwrap();
          element.style().set_property("display", "none").unwrap();
          let body = document.body().unwrap();
          let element = element.dyn_into::<web_sys::Node>().unwrap();
          body.append_child(&element).unwrap();
          let element = element.dyn_into::<web_sys::HtmlElement>().unwrap();
          element.click();
          let element = element.dyn_into::<web_sys::Node>().unwrap();
          body.remove_child(&element).unwrap();
        }
      }
      Message::ToggleEditMode => {
        self.canvas_field.edit_mode = !self.canvas_field.edit_mode;
      }
      Message::ToggleAI => {
        self.ai = !self.ai;
      }
      Message::Interrupt =>
      {
        #[cfg(not(target_arch = "wasm32"))]
        if self.thinking {
          self.should_stop.store(true, Ordering::Relaxed);
        }
      }
      Message::BotMove(maybe_pos) => {
        if let Some(pos) = maybe_pos {
          self.put_point(pos.get());
        }
        self.thinking = false;
        #[cfg(not(target_arch = "wasm32"))]
        self.should_stop.store(false, Ordering::Relaxed);
      }
      #[cfg(not(target_arch = "wasm32"))]
      Message::OpenFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Ok(sgf) = fs::read_to_string(file.path()) {
            if let Ok(trees) = sgf_parse::parse(sgf.as_str()) {
              if let Some(node) = trees.iter().find_map(|tree| match tree {
                GameTree::Unknown(node) => Some(node),
                GameTree::GoGame(_) => None,
              }) {
                if let Some(extended_field) = from_sgf::<ExtendedField, _>(node, &mut self.rng) {
                  let visits = sgf_to_visits(node, extended_field.field.width());
                  self.moves = extended_field
                    .field
                    .colored_moves()
                    .zip(
                      iter::repeat(Default::default())
                        .take(extended_field.field.moves_count() - visits.len() - 1)
                        .chain(visits)
                        .chain(iter::repeat(Default::default())),
                    )
                    .map(|((pos, player), visits)| (pos, player, visits))
                    .collect();
                  self.canvas_field.extended_field = extended_field;
                  self.oppai = Arc::new(Mutex::new(Oppai::new(
                    self.config.width,
                    self.config.height,
                    self.config.ai_config.clone(),
                    Arc::new(Patterns::default()),
                    (),
                  )));
                  self.put_all_bot_points();
                  self.refresh();
                }
              }
            }
          }
        }
      }
      #[cfg(target_arch = "wasm32")]
      Message::OpenFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Ok(sgf) = std::str::from_utf8(&file) {
            if let Ok(trees) = sgf_parse::parse(sgf) {
              if let Some(node) = trees.iter().find_map(|tree| match tree {
                GameTree::Unknown(node) => Some(node),
                GameTree::GoGame(_) => None,
              }) {
                if let Some(extended_field) = from_sgf::<ExtendedField, _>(node, &mut self.rng) {
                  let visits = sgf_to_visits(node, extended_field.field.width());
                  self.moves = extended_field
                    .field
                    .colored_moves()
                    .zip(
                      iter::repeat(Default::default())
                        .take(extended_field.field.moves_count() - visits.len() - 1)
                        .chain(visits)
                        .chain(iter::repeat(Default::default())),
                    )
                    .map(|((pos, player), visits)| (pos, player, visits))
                    .collect();
                  self.canvas_field.extended_field = extended_field;
                  self.send_worker_message(Request::New(
                    self.canvas_field.extended_field.field.width(),
                    self.canvas_field.extended_field.field.height(),
                  ));
                  self.put_all_bot_points();
                  self.refresh();
                }
              }
            }
          }
        }
      }
      #[cfg(not(target_arch = "wasm32"))]
      Message::SaveFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Some(s) = to_sgf_str(&self.canvas_field.extended_field) {
            fs::write(file.path(), s).ok();
          }
        }
      }
      #[cfg(target_arch = "wasm32")]
      Message::SetWorkerListener(tx) => {
        use wasm_bindgen::JsCast;
        let callback = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
          move |event: web_sys::MessageEvent| {
            let response: Response = serde_wasm_bindgen::from_value(event.data()).unwrap();
            let message = match response {
              Response::BestMove(pos) => Message::BotMove(NonZeroPos::new(pos as usize)),
              Response::Init => Message::InitWorker,
            };
            tx.unbounded_send(message).unwrap();
          },
        );
        self.worker.set_onmessage(Some(callback.as_ref().unchecked_ref()));
        callback.forget();
      }
      #[cfg(target_arch = "wasm32")]
      Message::InitWorker => {
        self.send_worker_message(Request::New(
          self.canvas_field.extended_field.field.width(),
          self.canvas_field.extended_field.field.height(),
        ));
        self.put_all_bot_points();
        if self.thinking {
          self.send_worker_message(Request::BestMove(self.canvas_field.extended_field.player));
        }
      }
    }

    Command::none()
  }

  fn view(&self) -> iced::Element<'_, Self::Message> {
    let mode = Text::new(if self.canvas_field.edit_mode {
      "Mode: Editing"
    } else {
      "Mode: Playing"
    });

    let ai = Text::new(if self.thinking {
      "AI: Thinking"
    } else if self.ai {
      "AI: Idle"
    } else {
      "AI: Off"
    });

    let score = Row::new()
      .push(Text::new("Score: "))
      .push(
        Text::new(
          self
            .canvas_field
            .extended_field
            .field
            .captured_count(Player::Red)
            .to_string(),
        )
        .style(Color::from(self.config.canvas_config.red_color)),
      )
      .push(Text::new(":"))
      .push(
        Text::new(
          self
            .canvas_field
            .extended_field
            .field
            .captured_count(Player::Black)
            .to_string(),
        )
        .style(Color::from(self.config.canvas_config.black_color)),
      );

    let moves_count = Text::new(format!(
      "Moves: {}",
      self.canvas_field.extended_field.field.moves_count()
    ));

    let coordinates = Text::new(if let Some((x, y)) = self.coordinates {
      format!("Coords: {}-{}", x, y)
    } else {
      "Coords: -".to_owned()
    });

    let canvas = Canvas::new(&self.canvas_field).height(Length::Fill).width(Length::Fill);
    let canvas_element = Element::<CanvasMessage>::from(canvas).map(Message::Canvas);

    let info = Column::new()
      .push(mode)
      .push(ai)
      .push(score)
      .push(moves_count)
      .push(coordinates)
      .width(Length::Fixed(130.0))
      .padding(2);

    let content = Row::new().push(canvas_element).push(info);

    Container::new(content).width(Length::Fill).height(Length::Fill).into()
  }

  fn theme(&self) -> Theme {
    Theme::custom(
      "oppai".to_string(),
      Palette {
        background: self.config.canvas_config.background_color.into(),
        text: self.config.canvas_config.grid_color.into(),
        ..Palette::LIGHT
      },
    )
  }
}
