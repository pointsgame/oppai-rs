#[macro_use]
extern crate log;

mod config;
mod extended_field;
mod sgf;
#[cfg(target_arch = "wasm32")]
mod worker_message;

use crate::config::{cli_parse, Config, Rgb};
use crate::extended_field::ExtendedField;
#[cfg(target_arch = "wasm32")]
use crate::worker_message::{Request, Response};
use iced::theme::Palette;
use iced::widget::{canvas, Canvas, Column, Container, Row, Text};
use iced::{
  executor, keyboard, mouse, Application, Color, Command, Element, Length, Point, Rectangle, Settings, Size, Theme,
  Vector,
};
#[cfg(not(target_arch = "wasm32"))]
use oppai_bot::bot::Bot;
use oppai_bot::field::{to_pos, NonZeroPos, Pos};
#[cfg(not(target_arch = "wasm32"))]
use oppai_bot::patterns::Patterns;
use oppai_bot::player::Player;
use rand::rngs::SmallRng;
#[cfg(not(target_arch = "wasm32"))]
use rand::Rng;
use rand::SeedableRng;
use rfd::AsyncFileDialog;
#[cfg(not(target_arch = "wasm32"))]
use rfd::FileHandle;
use std::sync::{
  atomic::{AtomicBool, Ordering},
  Arc,
};
#[cfg(not(target_arch = "wasm32"))]
use std::{fs, fs::File, sync::Mutex};

impl From<Rgb> for Color {
  fn from(rgb: Rgb) -> Self {
    Self::from_rgb8(rgb.r, rgb.g, rgb.b)
  }
}

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
    ..Settings::default()
  })
}

struct Game {
  config: Config,
  rng: SmallRng,
  extended_field: ExtendedField,
  field_cache: canvas::Cache,
  #[cfg(not(target_arch = "wasm32"))]
  bot: Arc<Mutex<Bot<SmallRng>>>,
  #[cfg(target_arch = "wasm32")]
  worker: web_sys::Worker,
  edit_mode: bool,
  ai: bool,
  thinking: bool,
  #[cfg(not(target_arch = "wasm32"))]
  file_choosing: bool,
  coordinates: Option<(u32, u32)>,
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
    let player = self.extended_field.player;
    if self.extended_field.put_point(pos) {
      #[cfg(not(target_arch = "wasm32"))]
      self.bot.lock().unwrap().field.put_point(pos, player);
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::PutPoint(pos, player));
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  pub fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.extended_field.put_players_point(pos, player) {
      #[cfg(not(target_arch = "wasm32"))]
      self.bot.lock().unwrap().field.put_point(pos, player);
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::PutPoint(pos, player));
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  pub fn undo(&mut self) -> bool {
    if self.extended_field.undo() {
      #[cfg(not(target_arch = "wasm32"))]
      self.bot.lock().unwrap().field.undo();
      #[cfg(target_arch = "wasm32")]
      self.send_worker_message(Request::Undo);
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  #[cfg(not(target_arch = "wasm32"))]
  pub fn put_all_bot_points(&self) {
    let mut bot = self.bot.lock().unwrap();
    for &pos in self.extended_field.field.points_seq() {
      let player = self.extended_field.field.cell(pos).get_player();
      bot.field.put_point(pos, player);
    }
  }

  #[cfg(target_arch = "wasm32")]
  pub fn put_all_bot_points(&self) {
    for &pos in self.extended_field.field.points_seq() {
      let player = self.extended_field.field.cell(pos).get_player();
      self.send_worker_message(Request::PutPoint(pos, player));
    }
  }

  #[cfg(not(target_arch = "wasm32"))]
  pub fn is_locked(&self) -> bool {
    self.thinking || self.file_choosing
  }

  #[cfg(target_arch = "wasm32")]
  pub fn is_locked(&self) -> bool {
    self.thinking
  }
}

#[derive(Debug, Clone, Copy)]
enum CanvasMessage {
  PutPoint(Pos),
  PutPlayersPoint(Pos, Player),
  Undo,
  New,
  Open,
  Save,
  ToggleEditMode,
  ToggleAI,
  ChangeCoordinates(u32, u32),
  ClearCoordinates,
  Interrupt,
}

#[derive(Debug)]
enum Message {
  Canvas(CanvasMessage),
  BotMove(Option<NonZeroPos>),
  #[cfg(not(target_arch = "wasm32"))]
  OpenFile(Option<FileHandle>),
  #[cfg(not(target_arch = "wasm32"))]
  SaveFile(Option<FileHandle>),
  #[cfg(target_arch = "wasm32")]
  OpenFile(Option<Vec<u8>>),
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
    let mut extended_field = ExtendedField::new(flags.width, flags.height, &mut rng);
    #[cfg(not(target_arch = "wasm32"))]
    let patterns = if flags.patterns.is_empty() {
      Patterns::default()
    } else {
      Patterns::from_files(
        flags
          .patterns
          .iter()
          .map(|path| File::open(path).expect("Failed to open patterns file.")),
      )
      .expect("Failed to read patterns file.")
    };
    #[cfg(not(target_arch = "wasm32"))]
    let bot = Bot::new(
      flags.width,
      flags.height,
      SmallRng::from_seed(rng.gen()),
      Arc::new(patterns),
      flags.bot_config.clone(),
    );
    extended_field.place_initial_position(flags.initial_position);
    let game = Game {
      config: flags,
      rng,
      extended_field,
      field_cache: Default::default(),
      #[cfg(not(target_arch = "wasm32"))]
      bot: Arc::new(Mutex::new(bot)),
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
      edit_mode: false,
      ai: true,
      thinking: false,
      #[cfg(not(target_arch = "wasm32"))]
      file_choosing: false,
      coordinates: None,
      should_stop: Arc::new(AtomicBool::new(false)),
    };
    game.put_all_bot_points();

    #[cfg(target_arch = "wasm32")]
    {
      // hack to resize canvas
      // https://github.com/iced-rs/iced/issues/1265
      use iced_native::{command, window};
      let window = web_sys::window().unwrap();
      let (width, height) = (
        (window.inner_width().unwrap().as_f64().unwrap()) as u32,
        (window.inner_height().unwrap().as_f64().unwrap()) as u32,
      );
      (
        game,
        Command::single(command::Action::Window(window::Action::Resize { width, height })),
      )
    }

    #[cfg(not(target_arch = "wasm32"))]
    (game, Command::none())
  }

  fn title(&self) -> String {
    "OpPAI".into()
  }

  #[cfg(target_arch = "wasm32")]
  fn subscription(&self) -> iced::Subscription<Message> {
    struct WorkerListener;
    enum State {
      Starting,
      Ready(iced::futures::channel::mpsc::UnboundedReceiver<Message>),
    }
    iced::subscription::channel(std::any::TypeId::of::<WorkerListener>(), 16, |mut output| async move {
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
  }

  fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
    match message {
      Message::BotMove(maybe_pos) => {
        if let Some(pos) = maybe_pos {
          self.put_point(pos.get());
        }
        self.thinking = false;
        self.should_stop.store(false, Ordering::Relaxed);
      }
      #[cfg(not(target_arch = "wasm32"))]
      Message::OpenFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Ok(text) = fs::read_to_string(file.path()) {
            if let Ok(game_tree) = sgf_parser::parse(&text) {
              if let Some(extended_field) = sgf::from_sgf(game_tree).and_then(|game| {
                let width = game.width;
                ExtendedField::from_moves(
                  game.width,
                  game.height,
                  &mut self.rng,
                  game
                    .moves
                    .into_iter()
                    .map(|(player, x, y)| (to_pos(width, x, y), player)),
                )
              }) {
                self.extended_field = extended_field;
                self.bot = Arc::new(Mutex::new(Bot::new(
                  self.config.width,
                  self.config.height,
                  SmallRng::from_seed(self.rng.gen()),
                  Arc::new(Patterns::default()),
                  self.config.bot_config.clone(),
                )));
                self.put_all_bot_points();
                self.field_cache.clear();
              }
            }
          }
        }
        self.file_choosing = false;
      }
      #[cfg(target_arch = "wasm32")]
      Message::OpenFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Ok(text) = std::str::from_utf8(&file) {
            if let Ok(game_tree) = sgf_parser::parse(&text) {
              if let Some(extended_field) = sgf::from_sgf(game_tree).and_then(|game| {
                let width = game.width;
                ExtendedField::from_moves(
                  game.width,
                  game.height,
                  &mut self.rng,
                  game
                    .moves
                    .into_iter()
                    .map(|(player, x, y)| (to_pos(width, x, y), player)),
                )
              }) {
                self.extended_field = extended_field;
                self.send_worker_message(Request::New(
                  self.extended_field.field.width(),
                  self.extended_field.field.height(),
                ));
                self.put_all_bot_points();
                self.field_cache.clear();
              }
            }
          }
        }
      }
      #[cfg(not(target_arch = "wasm32"))]
      Message::SaveFile(maybe_file) => {
        if let Some(file) = maybe_file {
          let game_tree = sgf::to_sgf(sgf::SgfGame::from(&self.extended_field.field));
          let s: String = game_tree.into();
          fs::write(file.path(), s).ok();
        }
        self.file_choosing = false;
      }

      Message::Canvas(CanvasMessage::PutPoint(pos)) => {
        if self.is_locked() {
          return Command::none();
        }
        if self.put_point(pos) && self.ai {
          self.thinking = true;

          let player = self.extended_field.player;

          #[cfg(not(target_arch = "wasm32"))]
          {
            let bot = self.bot.clone();
            let time = self.config.time;
            let should_stop = self.should_stop.clone();
            return Command::perform(
              async move { bot.lock().unwrap().best_move_with_time(player, time, &should_stop) },
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
      Message::Canvas(CanvasMessage::Undo) => {
        if self.is_locked() {
          return Command::none();
        }
        self.undo();
      }
      Message::Canvas(CanvasMessage::New) => {
        if self.is_locked() {
          return Command::none();
        }
        self.extended_field = ExtendedField::new(self.config.width, self.config.height, &mut self.rng);
        #[cfg(not(target_arch = "wasm32"))]
        {
          self.bot = Arc::new(Mutex::new(Bot::new(
            self.config.width,
            self.config.height,
            SmallRng::from_seed(self.rng.gen()),
            Arc::new(Patterns::default()),
            self.config.bot_config.clone(),
          )));
        }
        #[cfg(target_arch = "wasm32")]
        self.send_worker_message(Request::New(
          self.extended_field.field.width(),
          self.extended_field.field.height(),
        ));
        self.extended_field.place_initial_position(self.config.initial_position);
        self.put_all_bot_points();
        self.field_cache.clear();
      }
      Message::Canvas(CanvasMessage::Open) => {
        if self.is_locked() {
          return Command::none();
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          self.file_choosing = true;
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
      Message::Canvas(CanvasMessage::Save) => {
        if self.file_choosing {
          return Command::none();
        }
        self.file_choosing = true;
        return Command::perform(
          AsyncFileDialog::new().add_filter("SGF", &["sgf"]).save_file(),
          Message::SaveFile,
        );
      }
      #[cfg(target_arch = "wasm32")]
      Message::Canvas(CanvasMessage::Save) => {
        let game_tree = sgf::to_sgf(sgf::SgfGame::from(&self.extended_field.field));
        let s: String = game_tree.into();

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
      Message::Canvas(CanvasMessage::ToggleEditMode) => {
        self.edit_mode = !self.edit_mode;
      }
      Message::Canvas(CanvasMessage::ToggleAI) => {
        self.ai = !self.ai;
      }
      Message::Canvas(CanvasMessage::ChangeCoordinates(x, y)) => {
        self.coordinates = Some((x, y));
      }
      Message::Canvas(CanvasMessage::ClearCoordinates) => {
        self.coordinates = None;
      }
      Message::Canvas(CanvasMessage::Interrupt) => {
        if self.thinking {
          self.should_stop.store(true, Ordering::Relaxed);
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
          self.extended_field.field.width(),
          self.extended_field.field.height(),
        ));
        self.put_all_bot_points();
        if self.thinking {
          self.send_worker_message(Request::BestMove(self.extended_field.player));
        }
      }
    }

    Command::none()
  }

  fn view(&self) -> iced::Element<'_, Self::Message> {
    let mode = Text::new(if self.edit_mode {
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
        Text::new(self.extended_field.field.captured_count(Player::Red).to_string())
          .style(Color::from(self.config.red_color)),
      )
      .push(Text::new(":"))
      .push(
        Text::new(self.extended_field.field.captured_count(Player::Black).to_string())
          .style(Color::from(self.config.black_color)),
      );

    let moves_count = Text::new(format!("Moves: {}", self.extended_field.field.moves_count()));

    let coordinates = Text::new(if let Some((x, y)) = self.coordinates {
      format!("Coords: {}-{}", x, y)
    } else {
      "Coords: -".to_owned()
    });

    let canvas = Canvas::new(self).height(Length::Fill).width(Length::Fill);
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
    Theme::custom(Palette {
      background: self.config.background_color.into(),
      text: self.config.grid_color.into(),
      ..Palette::LIGHT
    })
  }
}

impl canvas::Program<CanvasMessage> for Game {
  type State = ();

  fn update(
    &self,
    _state: &mut (),
    event: canvas::Event,
    bounds: Rectangle,
    cursor: canvas::Cursor,
  ) -> (canvas::event::Status, Option<CanvasMessage>) {
    match event {
      canvas::Event::Mouse(event) => {
        match event {
          mouse::Event::ButtonPressed(mouse::Button::Left) => {}
          mouse::Event::ButtonPressed(mouse::Button::Right) => {
            if !self.edit_mode {
              return (canvas::event::Status::Ignored, None);
            }
          }
          mouse::Event::CursorMoved { .. } => {}
          mouse::Event::CursorLeft => {
            if self.coordinates.is_some() {
              return (canvas::event::Status::Captured, Some(CanvasMessage::ClearCoordinates));
            } else {
              return (canvas::event::Status::Ignored, None);
            }
          }
          _ => return (canvas::event::Status::Ignored, None),
        }

        let cursor_position = if let Some(position) = cursor.position_in(&bounds) {
          position
        } else {
          return (canvas::event::Status::Ignored, None);
        };

        let field_width = self.extended_field.field.width();
        let field_height = self.extended_field.field.height();
        let width = bounds
          .width
          .min(bounds.height / field_height as f32 * field_width as f32);
        let height = bounds
          .height
          .min(bounds.width / field_width as f32 * field_height as f32);
        let step_x = width / field_width as f32;
        let step_y = height / field_height as f32;
        let shift = Vector::new(
          ((bounds.width - width) / 2.0).round(),
          ((bounds.height - height) / 2.0).round(),
        );
        let cursor_shift = Vector::new(step_x / 2.0, step_y / 2.0);

        let point = cursor_position - shift;
        if point.x >= 0.0 && point.x <= width && point.y >= 0.0 && point.y <= height {
          let point = point - cursor_shift;
          let x = (point.x / step_x).round() as u32;
          let y = (point.y / step_y).round() as u32;

          match event {
            mouse::Event::ButtonPressed(button) => {
              let pos = self.extended_field.field.to_pos(x, y);
              match button {
                mouse::Button::Left => (
                  canvas::event::Status::Captured,
                  Some(if self.edit_mode {
                    CanvasMessage::PutPlayersPoint(pos, Player::Red)
                  } else {
                    CanvasMessage::PutPoint(pos)
                  }),
                ),
                mouse::Button::Right => (
                  canvas::event::Status::Captured,
                  Some(CanvasMessage::PutPlayersPoint(pos, Player::Black)),
                ),
                _ => (canvas::event::Status::Ignored, None),
              }
            }
            mouse::Event::CursorMoved { .. } => (
              canvas::event::Status::Captured,
              if self.coordinates != Some((x, y)) {
                Some(CanvasMessage::ChangeCoordinates(x, y))
              } else {
                None
              },
            ),
            _ => (canvas::event::Status::Ignored, None),
          }
        } else {
          (
            canvas::event::Status::Captured,
            if self.coordinates.is_some() {
              Some(CanvasMessage::ClearCoordinates)
            } else {
              None
            },
          )
        }
      }
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::Backspace,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Undo)),
      #[cfg(not(target_arch = "wasm32"))]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::N,
        modifiers,
      }) if modifiers.control() => (canvas::event::Status::Captured, Some(CanvasMessage::New)),
      #[cfg(not(target_arch = "wasm32"))]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::O,
        modifiers,
      }) if modifiers.control() => (canvas::event::Status::Captured, Some(CanvasMessage::Open)),
      #[cfg(not(target_arch = "wasm32"))]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::S,
        modifiers,
      }) if modifiers.control() => (canvas::event::Status::Captured, Some(CanvasMessage::Save)),
      #[cfg(target_arch = "wasm32")]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::N,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::New)),
      #[cfg(target_arch = "wasm32")]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::S,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Save)),
      #[cfg(target_arch = "wasm32")]
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::O,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Open)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::E,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::ToggleEditMode)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::A,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::ToggleAI)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::Escape,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Interrupt)),
      _ => (canvas::event::Status::Ignored, None),
    }
  }

  fn draw(&self, _state: &(), _theme: &Theme, bounds: Rectangle, cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
    fn color(config: &Config, player: Player) -> Color {
      (match player {
        Player::Red => config.red_color,
        Player::Black => config.black_color,
      })
      .into()
    }

    let field_width = self.extended_field.field.width();
    let field_height = self.extended_field.field.height();
    let width = bounds
      .width
      .min(bounds.height / field_height as f32 * field_width as f32);
    let height = bounds
      .height
      .min(bounds.width / field_width as f32 * field_height as f32);
    let step_x = width / field_width as f32;
    let step_y = height / field_height as f32;
    let shift = Vector::new(
      ((bounds.width - width) / 2.0).round(),
      ((bounds.height - height) / 2.0).round(),
    );
    let cursor_shift = Vector::new(step_x / 2.0, step_y / 2.0);

    let xy_to_point = |x: u32, y: u32| {
      let offset_x = (step_x * x as f32 + step_x / 2.0).round() + 0.5;
      let offset_y = (step_y * y as f32 + step_y / 2.0).round() + 0.5;
      Point::new(offset_x, offset_y) + shift
    };
    let pos_to_point = |pos: Pos| {
      let x = self.extended_field.field.to_x(pos);
      let y = self.extended_field.field.to_y(pos);
      xy_to_point(x, y)
    };

    let point_radius = width / field_width as f32 * self.config.point_radius;

    let field = self.field_cache.draw(bounds.size(), |frame| {
      // draw grid

      let grid = canvas::Path::new(|path| {
        for x in 0..field_width {
          let offset = (step_x * x as f32 + step_x / 2.0).round() + 0.5;
          path.move_to(Point::new(offset, 0.0) + shift);
          path.line_to(Point::new(offset, height) + shift);
        }
        for y in 0..field_height {
          let offset = (step_y * y as f32 + step_y / 2.0).round() + 0.5;
          path.move_to(Point::new(0.0, offset) + shift);
          path.line_to(Point::new(width, offset) + shift);
        }
      });

      frame.stroke(
        &grid,
        canvas::Stroke {
          width: self.config.grid_thickness,
          style: canvas::Style::Solid(self.config.grid_color.into()),
          ..canvas::Stroke::default()
        },
      );

      // draw points

      for &player in &[Player::Red, Player::Black] {
        let points = canvas::Path::new(|path| {
          for &pos in self
            .extended_field
            .field
            .points_seq()
            .iter()
            .filter(|&&pos| self.extended_field.field.cell(pos).is_players_point(player))
          {
            path.circle(pos_to_point(pos), point_radius)
          }
        });

        frame.fill(&points, color(&self.config, player));
      }

      // fill extended area to display connecting lines

      if self.config.extended_filling {
        for &pos in self.extended_field.field.points_seq() {
          let player = self.extended_field.field.cell(pos).get_player();
          let mut color = color(&self.config, player);
          color.a = self.config.filling_alpha;
          let p = pos_to_point(pos);
          let captured = self.extended_field.captured[pos];
          let is_owner = |pos: Pos| -> bool {
            self.extended_field.field.cell(pos).is_players_point(player)
              || self.extended_field.captured[pos] > 0
                && (captured == 0 || self.extended_field.captured[pos] < captured)
          };

          // draw vertical lines

          if self
            .extended_field
            .field
            .cell(self.extended_field.field.s(pos))
            .is_players_point(player)
          {
            if !is_owner(self.extended_field.field.w(pos)) && !is_owner(self.extended_field.field.sw(pos)) {
              frame.fill_rectangle(p, Size::new(-point_radius, step_y), color);
            }

            if !is_owner(self.extended_field.field.e(pos)) && !is_owner(self.extended_field.field.se(pos)) {
              frame.fill_rectangle(p, Size::new(point_radius, step_y), color);
            }
          }

          // draw horizontal lines

          if self
            .extended_field
            .field
            .cell(self.extended_field.field.e(pos))
            .is_players_point(player)
          {
            if !is_owner(self.extended_field.field.n(pos)) && !is_owner(self.extended_field.field.ne(pos)) {
              frame.fill_rectangle(p, Size::new(step_x, -point_radius), color);
            }

            if !is_owner(self.extended_field.field.s(pos)) && !is_owner(self.extended_field.field.se(pos)) {
              frame.fill_rectangle(p, Size::new(step_x, point_radius), color);
            }
          }

          // draw \ diagonal lines

          let diag_width = point_radius / 2f32.sqrt();

          if self
            .extended_field
            .field
            .cell(self.extended_field.field.se(pos))
            .is_players_point(player)
          {
            let p2 = pos_to_point(self.extended_field.field.se(pos));

            if is_owner(self.extended_field.field.e(pos)) && !is_owner(self.extended_field.field.s(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(-diag_width, diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }

            if is_owner(self.extended_field.field.s(pos)) && !is_owner(self.extended_field.field.e(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(diag_width, -diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }
          }

          // draw / diagonal lines

          if self
            .extended_field
            .field
            .cell(self.extended_field.field.ne(pos))
            .is_players_point(player)
          {
            let p2 = pos_to_point(self.extended_field.field.ne(pos));

            if is_owner(self.extended_field.field.e(pos)) && !is_owner(self.extended_field.field.n(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(-diag_width, -diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }

            if is_owner(self.extended_field.field.n(pos)) && !is_owner(self.extended_field.field.e(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(diag_width, diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }
          }
        }
      }

      // fill captures

      for (chain, player, _) in &self.extended_field.captures {
        if !self.config.maximum_area_filling && chain.len() < 4 {
          continue;
        }

        let path = canvas::Path::new(|path| {
          path.move_to(pos_to_point(chain[0]));
          for &pos in chain.iter().skip(1) {
            path.line_to(pos_to_point(pos));
          }
        });

        let mut color = color(&self.config, *player);
        color.a = self.config.filling_alpha;

        frame.fill(&path, color);
      }

      // mark last point

      if self.config.last_point_mark {
        if let Some(&pos) = self.extended_field.field.points_seq().last() {
          let last_point = canvas::Path::new(|path| path.circle(pos_to_point(pos), point_radius * 1.5));

          let color = color(&self.config, self.extended_field.field.cell(pos).get_player());

          frame.stroke(
            &last_point,
            canvas::Stroke {
              width: 2.0,
              style: canvas::Style::Solid(color),
              ..canvas::Stroke::default()
            },
          );
        }
      }
    });

    let mut frame = canvas::Frame::new(bounds.size());

    if let Some(point) = cursor.position().and_then(|cursor_position| {
      let point = cursor_position - shift;
      if point.x >= 0.0 && point.x <= width && point.y >= 0.0 && point.y <= height {
        let point = point - cursor_shift;
        let x = (point.x / step_x).round() as u32;
        let y = (point.y / step_y).round() as u32;
        let pos = self.extended_field.field.to_pos(x, y);
        if self.extended_field.field.is_putting_allowed(pos) {
          Some(xy_to_point(x, y))
        } else {
          None
        }
      } else {
        None
      }
    }) {
      let cursor_point = canvas::Path::new(|path| path.circle(point, point_radius));

      let mut color = color(&self.config, self.extended_field.player);
      color.a = 0.5;

      frame.fill(&cursor_point, color);
    }

    vec![field, frame.into_geometry()]
  }
}
