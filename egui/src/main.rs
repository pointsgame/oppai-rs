mod extended_field;

use eframe::{
  egui::{self, Vec2},
  epaint::{Color32, Pos2, Stroke, Rounding, Rect, PathShape},
};
use extended_field::ExtendedField;
use oppai_bot::{player::Player, field::Pos};
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub fn field_ui(ui: &mut egui::Ui, field: &ExtendedField) -> egui::Response {
  let size = ui.available_size();
  let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());

  let field_width = field.field.width();
  let field_height = field.field.height();

  let width = rect
    .width()
    .min(rect.height() / field_height as f32 * field_width as f32);
  let height = rect
    .height()
    .min(rect.width() / field_width as f32 * field_height as f32);
  let shift = Vec2::new(
    ((rect.width() - width) / 2.0).round(),
    ((rect.height() - height) / 2.0).round(),
  );

  let rect = Rect {
    min: rect.min + shift,
    max: rect.min + shift + Vec2::new(width, height)
  };

  let step_x = rect.width() / field_width as f32;
  let step_y = rect.height() / field_height as f32;


  let xy_to_pos2 = |x: u32, y: u32| {
    let offset_x = (rect.left() + step_x * x as f32 + step_x / 2.0).round() + 0.5;
    let offset_y = (rect.top() + step_y * y as f32 + step_y / 2.0).round() + 0.5;
    Pos2::new(offset_x, offset_y)
  };
  let pos_to_pos2 = |pos: Pos| {
    let x = field.field.to_x(pos);
    let y = field.field.to_y(pos);
    xy_to_pos2(x, y)
  };

  // draw background
  ui.painter().rect_filled(rect, Rounding::ZERO, Color32::WHITE);

  // draw grid
  for x in 0 .. field_width {
    let x = (rect.left() + step_x * x as f32 + step_x / 2.0).round() + 0.5;
    ui.painter().line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, Color32::BLACK));
  }

  for y in 0 .. field_height {
    let y = (rect.top() + step_y * y as f32 + step_y / 2.0).round() + 0.5;
    ui.painter().line_segment([Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(1.0, Color32::BLACK));
  }

  // draw points

  for &player in &[Player::Red, Player::Black] {
    let color = match player {
      Player::Red => Color32::RED,
      Player::Black => Color32::BLACK,
    };

    for &pos in field
      .field
      .points_seq()
      .iter()
      .filter(|&&pos| field.field.cell(pos).is_players_point(player))
          {
            ui.painter().circle_filled(pos_to_pos2(pos), 5.0, color)
          }
  }

  // fill captures

  for (chain, player, _) in &field.captures {
    if chain.len() < 4 {
      continue;
    }

    let mut color = Color32::from_rgba_unmultiplied(255, 0, 0, 127);

    let path = PathShape {
      points: chain.iter().map(|&pos| pos_to_pos2(pos)).collect(),
      closed: true,
      fill: color,
      stroke: Stroke::NONE,
    };

    ui.painter().add(path);
  }

  response
}

fn main() -> Result<(), eframe::Error> {
  let options = eframe::NativeOptions {
    initial_window_size: Some(egui::vec2(640.0, 480.0)),
    ..Default::default()
  };

  let mut rng = SmallRng::from_entropy();
  let mut field = ExtendedField::new(39, 32, &mut rng);
  // field.put_players_point(field.field.to_pos(10, 10), Player::Black);
  // field.put_players_point(field.field.to_pos(9, 10), Player::Red);
  // field.put_players_point(field.field.to_pos(10, 9), Player::Red);
  // field.put_players_point(field.field.to_pos(10, 11), Player::Red);
  // field.put_players_point(field.field.to_pos(11, 10), Player::Red);
  // field.put_players_point(field.field.to_pos(11, 11), Player::Black);
  // field.put_players_point(field.field.to_pos(12, 11), Player::Red);
  // field.put_players_point(field.field.to_pos(11, 12), Player::Red);
  field.put_players_point(field.field.to_pos(10, 10), Player::Black);
  field.put_players_point(field.field.to_pos(12, 10), Player::Black);

  field.put_players_point(field.field.to_pos(9, 10), Player::Red);
  field.put_players_point(field.field.to_pos(10, 9), Player::Red);
  field.put_players_point(field.field.to_pos(10, 11), Player::Red);
  field.put_players_point(field.field.to_pos(13, 10), Player::Red);
  field.put_players_point(field.field.to_pos(12, 9), Player::Red);
  field.put_players_point(field.field.to_pos(12, 11), Player::Red);

  field.put_players_point(field.field.to_pos(11, 10), Player::Red);

  eframe::run_simple_native("OpPAI", options, move |ctx, _frame| {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.heading("My egui Application");

      field_ui(ui, &field);

    });
  })
}
