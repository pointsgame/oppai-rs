mod config;

use crate::config::{cli_parse, Config, RGB};
use iced::{
  canvas, container, executor, keyboard, mouse, Application, Background, Canvas, Color, Command, Container, Element,
  Length, Point, Rectangle, Row, Settings, Size, Text, Vector,
};
use oppai_field::field::{self, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

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
  player: Player,
  field: Field,
  captures: Vec<(Vec<Pos>, Player, usize)>,
  captured: Vec<usize>,
  field_cache: canvas::Cache,
}

#[derive(Debug, Clone, Copy)]
enum CanvasMessage {
  PutPoint(Pos),
  Undo,
}

#[derive(Debug, Clone, Copy)]
enum Message {
  Canvas(CanvasMessage),
}

impl Application for Game {
  type Executor = executor::Default;
  type Message = Message;
  type Flags = Config;

  fn new(flags: Config) -> (Self, Command<Self::Message>) {
    let mut rng = XorShiftRng::from_entropy();
    let zobrist = Zobrist::new(field::length(flags.width, flags.height) * 2, &mut rng);
    let field = Field::new(flags.width, flags.height, std::sync::Arc::new(zobrist));
    let length = field.length();
    (
      Game {
        config: flags,
        player: Player::Red,
        field,
        captures: Vec::new(),
        captured: vec![0; length],
        field_cache: Default::default(),
      },
      Command::none(),
    )
  }

  fn title(&self) -> String {
    "OpPAI".into()
  }

  fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
    match message {
      Message::Canvas(CanvasMessage::PutPoint(pos)) => {
        if self.field.put_point(pos, self.player) {
          let last_chain = self.field.get_last_chain();
          if let Some(&pos) = last_chain.first() {
            let player = self.field.cell(pos).get_player();
            self.captures.push((last_chain, player, self.field.moves_count()));
            for (pos, _) in self.field.last_changed_cells() {
              if self.captured[pos] == 0 && self.field.cell(pos).is_captured() {
                self.captured[pos] = self.field.moves_count();
              }
            }
          }

          if self.config.maximum_area_filling {
            let n = self.field.n(pos);
            let s = self.field.s(pos);
            let w = self.field.w(pos);
            let e = self.field.e(pos);
            let nw = self.field.nw(pos);
            let ne = self.field.ne(pos);
            let sw = self.field.sw(pos);
            let se = self.field.se(pos);

            let mut check = |pos1: Pos, pos2: Pos| {
              if self.field.cell(pos1).get_players_point() == Some(self.player)
                && self.field.cell(pos2).get_players_point() == Some(self.player)
              {
                self
                  .captures
                  .push((vec![pos, pos1, pos2], self.player, self.field.moves_count()));
                true
              } else {
                false
              }
            };

            let _ = !check(s, e) && (check(s, se) || check(e, se));
            let _ = !check(e, n) && (check(e, ne) || check(n, ne));
            let _ = !check(n, w) && (check(n, nw) || check(w, nw));
            let _ = !check(w, s) && (check(w, sw) || check(s, sw));
          }

          self.player = self.player.next();

          self.field_cache.clear();
        }
      }
      Message::Canvas(CanvasMessage::Undo) => {
        if let Some(player) = self.field.last_player() {
          let moves_count = self.field.moves_count();
          for (pos, _) in self.field.last_changed_cells() {
            if self.captured[pos] == moves_count {
              self.captured[pos] = 0;
            }
          }

          self.player = player;
          self.field.undo();

          while self
            .captures
            .last()
            .map_or(false, |&(_, _, c)| c > self.field.moves_count())
          {
            self.captures.pop();
          }

          self.field_cache.clear();
        }
      }
    }

    Command::none()
  }

  fn view(&mut self) -> iced::Element<'_, Self::Message> {
    let score = Row::new()
      .push(Text::new(self.field.captured_count(Player::Red).to_string()).color(self.config.red_color))
      .push(Text::new(":"))
      .push(Text::new(self.field.captured_count(Player::Black).to_string()).color(self.config.black_color));

    let background_color = self.config.background_color;
    let text_color = self.config.grid_color;

    let canvas = Canvas::new(self).height(Length::Fill).width(Length::Fill);
    let canvas_element = Element::<CanvasMessage>::from(canvas).map(Message::Canvas);

    let content = Row::new().push(canvas_element).push(score);

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
      canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
        let cursor_position = if let Some(position) = cursor.position_in(&bounds) {
          position
        } else {
          return (canvas::event::Status::Ignored, None);
        };

        let field_width = self.field.width();
        let field_height = self.field.height();
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
          (
            canvas::event::Status::Captured,
            Some(CanvasMessage::PutPoint(self.field.to_pos(x, y))),
          )
        } else {
          (canvas::event::Status::Captured, None)
        }
      }
      canvas::Event::Keyboard(keyboard::Event::KeyPressed {
        key_code: keyboard::KeyCode::Backspace,
        ..
      }) => (canvas::event::Status::Captured, Some(CanvasMessage::Undo)),
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

    let field_width = self.field.width();
    let field_height = self.field.height();
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
      let x = self.field.to_x(pos);
      let y = self.field.to_y(pos);
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
            .field
            .points_seq()
            .iter()
            .filter(|&&pos| self.field.cell(pos).is_players_point(player))
          {
            path.circle(pos_to_point(pos), point_radius)
          }
        });

        frame.fill(&points, color(&self.config, player));
      }

      // fill extended area to display connecting lines

      if self.config.extended_filling {
        for &pos in self.field.points_seq() {
          let player = self.field.cell(pos).get_player();
          let mut color = color(&self.config, player);
          color.a = self.config.filling_alpha;
          let p = pos_to_point(pos);
          let captured = self.captured[pos];
          let is_owner = |pos: Pos| -> bool {
            self.field.cell(pos).is_players_point(player)
              || self.captured[pos] > 0 && (captured == 0 || self.captured[pos] < captured)
          };

          // draw vertical lines

          if self.field.cell(self.field.s(pos)).is_players_point(player) {
            if !is_owner(self.field.w(pos)) && !is_owner(self.field.sw(pos)) {
              frame.fill_rectangle(p, Size::new(-point_radius, step_y), color);
            }

            if !is_owner(self.field.e(pos)) && !is_owner(self.field.se(pos)) {
              frame.fill_rectangle(p, Size::new(point_radius, step_y), color);
            }
          }

          // draw horizontal lines

          if self.field.cell(self.field.e(pos)).is_players_point(player) {
            if !is_owner(self.field.n(pos)) && !is_owner(self.field.ne(pos)) {
              frame.fill_rectangle(p, Size::new(step_x, -point_radius), color);
            }

            if !is_owner(self.field.s(pos)) && !is_owner(self.field.se(pos)) {
              frame.fill_rectangle(p, Size::new(step_x, point_radius), color);
            }
          }

          // draw \ diagonal lines

          let diag_width = point_radius / 2f32.sqrt();

          if self.field.cell(self.field.se(pos)).is_players_point(player) {
            let p2 = pos_to_point(self.field.se(pos));

            if is_owner(self.field.e(pos)) && !is_owner(self.field.s(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(-diag_width, diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }

            if is_owner(self.field.s(pos)) && !is_owner(self.field.e(pos)) {
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

          if self.field.cell(self.field.ne(pos)).is_players_point(player) {
            let p2 = pos_to_point(self.field.ne(pos));

            if is_owner(self.field.e(pos)) && !is_owner(self.field.n(pos)) {
              let path = canvas::Path::new(|path| {
                let vec = Vector::new(-diag_width, -diag_width);
                path.move_to(p);
                path.line_to(p + vec);
                path.line_to(p2 + vec);
                path.line_to(p2);
              });
              frame.fill(&path, color);
            }

            if is_owner(self.field.n(pos)) && !is_owner(self.field.e(pos)) {
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

      for (chain, player, _) in &self.captures {
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

      if let Some(&pos) = self.field.points_seq().last() {
        let last_point = canvas::Path::new(|path| path.circle(pos_to_point(pos), point_radius * 1.5));

        let color = color(&self.config, self.field.cell(pos).get_player());

        frame.stroke(
          &last_point,
          canvas::Stroke {
            width: 2.0,
            color,
            ..canvas::Stroke::default()
          },
        );
      }
    });

    let mut frame = canvas::Frame::new(bounds.size());

    if let Some(point) = cursor.position().and_then(|cursor_position| {
      let point = cursor_position - shift;
      if point.x >= 0.0 && point.x <= width && point.y >= 0.0 && point.y <= height {
        let point = point - cursor_shift;
        let x = (point.x / step_x).round() as u32;
        let y = (point.y / step_y).round() as u32;
        let pos = self.field.to_pos(x, y);
        if self.field.is_putting_allowed(pos) {
          Some(xy_to_point(x, y))
        } else {
          None
        }
      } else {
        None
      }
    }) {
      let cursor_point = canvas::Path::new(|path| path.circle(point, point_radius));

      let mut color = color(&self.config, self.player);
      color.a = 0.5;

      frame.fill(&cursor_point, color);
    }

    vec![field, frame.into_geometry()]
  }
}
