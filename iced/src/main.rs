mod config;
mod extended_field;
mod sgf;

use crate::config::{cli_parse, Config, RGB};
use crate::extended_field::ExtendedField;
use iced::{
  canvas, container, executor, keyboard, mouse, Application, Background, Canvas, Color, Column, Command, Container,
  Element, Length, Point, Rectangle, Row, Settings, Size, Text, Vector,
};
use oppai_bot::bot::Bot;
use oppai_bot::config::Config as BotConfig;
use oppai_field::field::{NonZeroPos, Pos};
use oppai_field::player::Player;
use oppai_patterns::patterns::Patterns;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rfd::{AsyncFileDialog, FileHandle};
use std::{
  fs,
  sync::{Arc, Mutex},
};

impl From<RGB> for Color {
  fn from(rgb: RGB) -> Self {
    Self::from_rgb8(rgb.r, rgb.g, rgb.b)
  }
}

pub fn main() -> iced::Result {
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
  bot: Arc<Mutex<Bot<SmallRng>>>,
  edit_mode: bool,
  ai: bool,
  thinking: bool,
  file_choosing: bool,
  coordinates: Option<(u32, u32)>,
}

impl Game {
  pub fn put_point(&mut self, pos: Pos) -> bool {
    let player = self.extended_field.player;
    if self.extended_field.put_point(pos) {
      self.bot.lock().unwrap().field.put_point(pos, player);
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  pub fn put_players_point(&mut self, pos: Pos, player: Player) -> bool {
    if self.extended_field.put_players_point(pos, player) {
      self.bot.lock().unwrap().field.put_point(pos, player);
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  pub fn undo(&mut self) -> bool {
    if self.extended_field.undo() {
      self.bot.lock().unwrap().field.undo();
      self.field_cache.clear();
      true
    } else {
      false
    }
  }

  pub fn put_all_bot_points(&self) {
    let mut bot = self.bot.lock().unwrap();
    for &pos in self.extended_field.field.points_seq() {
      let player = self.extended_field.field.cell(pos).get_player();
      bot.field.put_point(pos, player);
    }
  }

  pub fn is_locked(&self) -> bool {
    self.thinking || self.file_choosing
  }
}

#[derive(Debug, Clone, Copy)]
enum CanvasMessage {
  PutPoint(Pos),
  PutPlayersPoint(Pos, Player),
  Undo,
  New,
  Open,
  ToggleEditMode,
  ToggleAI,
  ChangeCoordinates(u32, u32),
  ClearCoordinates,
}

#[derive(Debug)]
enum Message {
  Canvas(CanvasMessage),
  BotMove(Option<NonZeroPos>),
  OpenFile(Option<FileHandle>),
}

impl Application for Game {
  type Executor = executor::Default;
  type Message = Message;
  type Flags = Config;

  fn new(flags: Config) -> (Self, Command<Self::Message>) {
    let mut rng = SmallRng::from_entropy();
    let mut extended_field = ExtendedField::new(flags.width, flags.height, &mut rng);
    let bot = Bot::new(
      flags.width,
      flags.height,
      SmallRng::from_seed(rng.gen()),
      Arc::new(Patterns::default()),
      BotConfig::default(),
    );
    extended_field.place_initial_position(flags.initial_position);
    let game = Game {
      config: flags,
      rng,
      extended_field,
      field_cache: Default::default(),
      bot: Arc::new(Mutex::new(bot)),
      edit_mode: false,
      ai: true,
      thinking: false,
      file_choosing: false,
      coordinates: None,
    };
    game.put_all_bot_points();
    (game, Command::none())
  }

  fn title(&self) -> String {
    "OpPAI".into()
  }

  fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
    match message {
      Message::BotMove(maybe_pos) => {
        if let Some(pos) = maybe_pos {
          self.put_point(pos.get());
        }
        self.thinking = false;
      }
      Message::OpenFile(maybe_file) => {
        if let Some(file) = maybe_file {
          if let Ok(text) = fs::read_to_string(file.inner()) {
            if let Ok(game_tree) = sgf_parser::parse(&text) {
              if let Some(extended_field) = sgf::from_sgf(game_tree, &mut self.rng) {
                self.extended_field = extended_field;
                self.bot = Arc::new(Mutex::new(Bot::new(
                  self.config.width,
                  self.config.height,
                  SmallRng::from_seed(self.rng.gen()),
                  Arc::new(Patterns::default()),
                  BotConfig::default(),
                )));
                self.put_all_bot_points();
                self.field_cache.clear();
              }
            }
          }
        }
        self.file_choosing = false;
      }
      Message::Canvas(CanvasMessage::PutPoint(pos)) => {
        if self.is_locked() {
          return Command::none();
        }
        if self.put_point(pos) {
          if self.ai {
            self.thinking = true;
            let bot = self.bot.clone();
            let player = self.extended_field.player;
            return async move { Message::BotMove(bot.lock().unwrap().best_move(player)) }.into();
          }
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
        self.bot = Arc::new(Mutex::new(Bot::new(
          self.config.width,
          self.config.height,
          SmallRng::from_seed(self.rng.gen()),
          Arc::new(Patterns::default()),
          BotConfig::default(),
        )));
        self.extended_field.place_initial_position(self.config.initial_position);
        self.put_all_bot_points();
        self.field_cache.clear();
      }
      Message::Canvas(CanvasMessage::Open) => {
        if self.is_locked() {
          return Command::none();
        }
        self.file_choosing = true;
        return Command::perform(AsyncFileDialog::new().add_filter("SGF", &["sgf"]).pick_file(), Message::OpenFile);
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
    }

    Command::none()
  }

  fn view(&mut self) -> iced::Element<'_, Self::Message> {
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
      .push(Text::new(self.extended_field.field.captured_count(Player::Red).to_string()).color(self.config.red_color))
      .push(Text::new(":"))
      .push(
        Text::new(self.extended_field.field.captured_count(Player::Black).to_string()).color(self.config.black_color),
      );

    let moves_count = Text::new(format!("Moves: {}", self.extended_field.field.moves_count()));

    let coordinates = Text::new(if let Some((x, y)) = self.coordinates {
      format!("Coords: {}-{}", x, y)
    } else {
      "Coords: -".to_owned()
    });

    let background_color = self.config.background_color;
    let text_color = self.config.grid_color;

    let canvas = Canvas::new(self).height(Length::Fill).width(Length::Fill);
    let canvas_element = Element::<CanvasMessage>::from(canvas).map(Message::Canvas);

    let info = Column::new()
      .push(mode)
      .push(ai)
      .push(score)
      .push(moves_count)
      .push(coordinates)
      .width(Length::Units(130))
      .padding(2);

    let content = Row::new().push(canvas_element).push(info);

    Container::new(content)
      .width(Length::Fill)
      .height(Length::Fill)
      .style(ContainerStyle {
        background: background_color.into(),
        text: text_color.into(),
      })
      .into()
  }
}

pub struct ContainerStyle {
  background: Color,
  text: Color,
}

impl container::StyleSheet for ContainerStyle {
  fn style(&self) -> container::Style {
    container::Style {
      background: Some(Background::Color(self.background)),
      text_color: Some(self.text),
      ..container::Style::default()
    }
  }
}

impl canvas::Program<CanvasMessage> for Game {
  fn update(
    &mut self,
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
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::N,
        modifiers: keyboard::Modifiers { control: true, .. },
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::New)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::O,
        modifiers: keyboard::Modifiers { control: true, .. },
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Open)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::E,
        modifiers: keyboard::Modifiers { control: true, .. },
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::ToggleEditMode)),
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::A,
        modifiers: keyboard::Modifiers { control: true, .. },
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::ToggleAI)),
      _ => (canvas::event::Status::Ignored, None),
    }
  }

  fn draw(&self, bounds: Rectangle, cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
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
          color: self.config.grid_color.into(),
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
              color,
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
