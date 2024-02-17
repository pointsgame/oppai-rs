use crate::canvas_config::{CanvasConfig, Rgb};
use iced::alignment::{Horizontal, Vertical};
use iced::mouse::Cursor;
use iced::widget::canvas::{self, Frame, Text};
use iced::{mouse, Color, Pixels, Point, Rectangle, Renderer, Size, Theme, Vector};
use oppai_field::extended_field::ExtendedField;
use oppai_field::field::Pos;
use oppai_field::player::Player;

impl From<Rgb> for Color {
  fn from(rgb: Rgb) -> Self {
    Self::from_rgb8(rgb.r, rgb.g, rgb.b)
  }
}

#[derive(Debug, Clone, Copy)]
pub enum CanvasMessage {
  PutPoint(Pos),
  PutPlayersPoint(Pos, Player),
  ChangeCoordinates(u32, u32),
  ClearCoordinates,
}

pub trait Extra {
  fn render<F: Fn(Pos) -> Point>(&self, frame: &mut Frame, bounds: Rectangle, field: &ExtendedField, pos_to_point: &F);
}

impl Extra for () {
  fn render<F: Fn(Pos) -> Point>(&self, _: &mut Frame, _: Rectangle, _: &ExtendedField, _: &F) {}
}

impl<E: Extra, const N: usize> Extra for [E; N] {
  fn render<F: Fn(Pos) -> Point>(&self, frame: &mut Frame, bounds: Rectangle, field: &ExtendedField, pos_to_point: &F) {
    for e in self {
      e.render(frame, bounds, field, pos_to_point);
    }
  }
}

impl<E: Extra> Extra for Vec<E> {
  fn render<F: Fn(Pos) -> Point>(&self, frame: &mut Frame, bounds: Rectangle, field: &ExtendedField, pos_to_point: &F) {
    for e in self {
      e.render(frame, bounds, field, pos_to_point);
    }
  }
}

impl<E: Extra> Extra for Option<E> {
  fn render<F: Fn(Pos) -> Point>(&self, frame: &mut Frame, bounds: Rectangle, field: &ExtendedField, pos_to_point: &F) {
    if let Some(e) = self {
      e.render(frame, bounds, field, pos_to_point);
    }
  }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Label {
  pub pos: Pos,
  pub text: String,
  pub color: Color,
  pub scale: f32,
}

impl Extra for Label {
  fn render<F: Fn(Pos) -> Point>(&self, frame: &mut Frame, bounds: Rectangle, field: &ExtendedField, pos_to_point: &F) {
    let mut text: Text = self.text.as_str().into();
    text.horizontal_alignment = Horizontal::Center;
    text.vertical_alignment = Vertical::Center;
    text.size = Pixels(self.scale * bounds.width / field.field.width() as f32);
    text.color = self.color;
    text.position = pos_to_point(self.pos);
    frame.fill_text(text);
  }
}

pub struct CanvasField<E: Extra = ()> {
  pub extended_field: ExtendedField,
  pub field_cache: canvas::Cache,
  pub edit_mode: bool,
  pub config: CanvasConfig,
  pub extra: E,
}

impl<E: Extra> canvas::Program<CanvasMessage> for CanvasField<E> {
  type State = Option<(u32, u32)>;

  fn update(
    &self,
    state: &mut Option<(u32, u32)>,
    event: canvas::Event,
    bounds: Rectangle,
    cursor: Cursor,
  ) -> (canvas::event::Status, Option<CanvasMessage>) {
    match event {
      canvas::Event::Mouse(event) => {
        match event {
          mouse::Event::ButtonReleased(mouse::Button::Left) => {}
          mouse::Event::ButtonReleased(mouse::Button::Right) => {
            if !self.edit_mode {
              return (canvas::event::Status::Ignored, None);
            }
          }
          mouse::Event::CursorMoved { .. } => {}
          mouse::Event::CursorLeft => {
            if state.is_some() {
              *state = None;
              return (canvas::event::Status::Captured, Some(CanvasMessage::ClearCoordinates));
            } else {
              return (canvas::event::Status::Ignored, None);
            }
          }
          _ => return (canvas::event::Status::Ignored, None),
        }

        let cursor_position = if let Some(position) = cursor.position_in(bounds) {
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
            mouse::Event::ButtonReleased(button) => {
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
              if *state != Some((x, y)) {
                *state = Some((x, y));
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
            if state.is_some() {
              *state = None;
              Some(CanvasMessage::ClearCoordinates)
            } else {
              None
            },
          )
        }
      }
      _ => (canvas::event::Status::Ignored, None),
    }
  }

  fn draw(
    &self,
    _state: &Option<(u32, u32)>,
    renderer: &Renderer,
    _theme: &Theme,
    bounds: Rectangle,
    cursor: Cursor,
  ) -> Vec<canvas::Geometry> {
    fn color(config: &CanvasConfig, player: Player) -> Color {
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

    let field = self.field_cache.draw(renderer, bounds.size(), |frame| {
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
            .moves()
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
        for &pos in self.extended_field.field.moves() {
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
        if let Some(&pos) = self.extended_field.field.moves().last() {
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

      // extra

      self.extra.render(
        frame,
        Rectangle {
          x: shift.x,
          y: shift.y,
          width,
          height,
        },
        &self.extended_field,
        &pos_to_point,
      );
    });

    let mut frame = canvas::Frame::new(renderer, bounds.size());

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
