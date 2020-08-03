use iced::{canvas, mouse, Canvas, Color, Element, Length, Point, Rectangle, Sandbox, Settings, Vector};
use oppai_field::field::{self, Field, Pos};
use oppai_field::player::Player;
use oppai_field::zobrist::Zobrist;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

const FIELD_WIDTH: u32 = 39;
const FIELD_HEIGHT: u32 = 32;

pub fn main() {
  Game::run(Settings {
    antialiasing: true,
    ..Settings::default()
  });
}

struct Game {
  player: Player,
  field: Field,
  captures: Vec<(Vec<Pos>, Player)>,
}

#[derive(Debug, Clone, Copy)]
enum Message {
  PutPoint(Pos),
}

impl Sandbox for Game {
  type Message = Message;

  fn new() -> Self {
    let mut rng = XorShiftRng::from_seed(SEED);
    let zobrist = Zobrist::new(field::length(FIELD_WIDTH, FIELD_HEIGHT) * 2, &mut rng);
    let field = Field::new(FIELD_WIDTH, FIELD_HEIGHT, std::sync::Arc::new(zobrist));
    Game {
      player: Player::Red,
      field,
      captures: Vec::new(),
    }
  }

  fn title(&self) -> String {
    "OpPAI".into()
  }

  fn update(&mut self, message: Self::Message) {
    let Message::PutPoint(pos) = message;
    if self.field.put_point(pos, self.player) {
      self.player = self.player.next();
      let last_chain = self.field.get_last_chain();
      if let Some(&pos) = last_chain.first() {
        let player = self.field.cell(pos).get_player();
        self.captures.push((last_chain, player));
      }
    }
  }

  fn view(&mut self) -> iced::Element<'_, Self::Message> {
    let canvas = Canvas::new(self).height(Length::Fill).width(Length::Fill);
    Element::<Pos>::from(canvas).map(Message::PutPoint)
  }
}

impl canvas::Program<Pos> for Game {
  fn update(&mut self, event: canvas::Event, bounds: Rectangle, cursor: canvas::Cursor) -> Option<Pos> {
    let cursor_position = cursor.position_in(&bounds)?;
    let canvas::Event::Mouse(mouse_event) = event;
    match mouse_event {
      mouse::Event::ButtonPressed(mouse::Button::Left) => {
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
          ((bounds.width - width) / 2.0).round() + step_x / 2.0,
          ((bounds.height - height) / 2.0).round() + step_y / 2.0,
        );

        let point = cursor_position - shift;
        if point.x >= 0.0 && point.x <= width && point.y >= 0.0 && point.y <= height {
          let x = (point.x / step_x).round() as u32;
          let y = (point.y / step_y).round() as u32;
          Some(self.field.to_pos(x, y))
        } else {
          None
        }
      }
      _ => None,
    }
  }

  fn draw(&self, bounds: Rectangle, _cursor: canvas::Cursor) -> Vec<canvas::Geometry> {
    let mut frame = canvas::Frame::new(bounds.size());

    let field_width = self.field.width();
    let field_height = self.field.height();
    let width = frame
      .width()
      .min(frame.height() / field_height as f32 * field_width as f32);
    let height = frame
      .height()
      .min(frame.width() / field_width as f32 * field_height as f32);
    let step_x = width / field_width as f32;
    let step_y = height / field_height as f32;
    let shift = Vector::new(
      ((frame.width() - width) / 2.0).round(),
      ((frame.height() - height) / 2.0).round(),
    );

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
        width: 1.0,
        color: Color::BLACK,
        ..canvas::Stroke::default()
      },
    );

    fn color(player: Player) -> Color {
      match player {
        Player::Red => Color::from_rgb8(0xFF, 0x00, 0x00),
        Player::Black => Color::BLACK,
      }
    }

    let pos_to_point = |pos: Pos| {
      let x = self.field.to_x(pos);
      let y = self.field.to_y(pos);
      let offset_x = (step_x * x as f32 + step_x / 2.0).round() + 0.5;
      let offset_y = (step_y * y as f32 + step_y / 2.0).round() + 0.5;
      Point::new(offset_x, offset_y) + shift
    };

    for &player in &[Player::Red, Player::Black] {
      let points = canvas::Path::new(|path| {
        for &pos in self
          .field
          .points_seq()
          .iter()
          .filter(|&&pos| self.field.cell(pos).is_players_point(player))
        {
          path.circle(pos_to_point(pos), 5.0)
        }
      });

      frame.fill(&points, color(player));
    }

    for (chain, player) in &self.captures {
      let path = canvas::Path::new(|path| {
        path.move_to(pos_to_point(chain[0]));
        for &pos in chain.iter().skip(1) {
          path.line_to(pos_to_point(pos));
        }
      });

      let mut color = color(*player);
      color.a = 0.5;

      frame.fill(&path, color);
    }

    vec![frame.into_geometry()]
  }
}
