#[cfg(test)]
mod test;

use oppai_field::extended_field::ExtendedField;
use oppai_field::field::Pos;
use oppai_field::player::Player;
use svg::Document;
use svg::node::element::path::Data;
use svg::node::element::{Animate, Circle, Definitions, Group, Path, Rectangle, Use};

/// The configuration options for a drawing.
pub struct Config {
  /// The width of the drawing area in pixels.
  pub width: u32,
  /// The height of the drawing area in pixels.
  pub height: u32,
  /// The color used for red points as an SVG color string.
  pub red_color: String,
  /// The color used for black points as an SVG color string.
  pub black_color: String,
  /// The color used for the grid lines as an SVG color string.
  pub grid_color: String,
  /// The color used for the background as an SVG color string.
  pub background_color: String,
  /// The thickness of the grid lines in pixels.
  pub grid_thickness: u32,
  /// The radius of points.
  ///
  /// It's automatically scaled according to the filed and the drawing area sizes.
  pub point_radius: f32,
  /// The alpha value (transparency) used for captured areas.
  pub filling_alpha: f32,
  /// Whither to fill slighter wider area which allows to mark sticks.
  ///
  /// It is ignored if `maximum_area_filling` is disabled.
  pub extended_filling: bool,
  /// Whether to fill the maximum possible area.
  pub maximum_area_filling: bool,
  /// Whether to mark the last point.
  pub last_point_mark: bool,
  /// Whether to animate the point under the cursor.
  pub pointer: bool,
  /// If `true`, share pointer group with animations via defs.
  ///
  /// This drastically reduces SVG size but works only in Firefox.
  pub shared_pointer: bool,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 1024,
      height: 1024,
      red_color: "red".to_owned(),
      black_color: "black".to_owned(),
      grid_color: "black".to_owned(),
      background_color: "white".to_owned(),
      grid_thickness: 1,
      point_radius: 0.2,
      filling_alpha: 0.5,
      extended_filling: true,
      maximum_area_filling: true,
      last_point_mark: true,
      pointer: true,
      shared_pointer: true,
    }
  }
}

const DOT_ID: &str = "dot";
const DOT_REF: &str = "#dot";

const VERTICAL_LINE_ID: &str = "verticalLine";
const VERTICAL_LINE_REF: &str = "#verticalLine";

const HORIZONTAL_LINE_ID: &str = "horizontalLine";
const HORIZONTAL_LINE_REF: &str = "#horizontalLine";

const POINTER_ID: &str = "pointer";
const POINTER_REF: &str = "#pointer";

pub fn field_to_svg(config: &Config, extended_field: &ExtendedField) -> Document {
  let field_width = extended_field.field.width();
  let field_height = extended_field.field.height();
  let offset = if config.grid_thickness % 2 == 0 { 0.0 } else { 0.5 };
  let width = (config.width as f32).min(config.height as f32 / field_height as f32 * field_width as f32);
  let height = (config.height as f32).min(config.width as f32 / field_width as f32 * field_height as f32);
  let step_x = width / field_width as f32;
  let step_y = height / field_height as f32;
  let xy_to_point = |x: u32, y: u32| {
    let offset_x = (step_x * x as f32 + step_x / 2.0).round() + offset;
    let offset_y = (step_y * y as f32 + step_y / 2.0).round() + offset;
    (offset_x, offset_y)
  };
  let pos_to_point = |pos: Pos| {
    let x = extended_field.field.to_x(pos);
    let y = extended_field.field.to_y(pos);
    xy_to_point(x, y)
  };
  let color = |player: Player| match player {
    Player::Red => config.red_color.as_ref(),
    Player::Black => config.black_color.as_ref(),
  };
  let point_radius = width / field_width as f32 * config.point_radius;

  let mut defs = Definitions::new();
  if !extended_field.field.is_empty() {
    defs = defs
      .add(
        Circle::new()
          .set("id", DOT_ID)
          .set("r", point_radius)
          .set("shape-rendering", "geometricPrecision"),
      )
      .add(
        Rectangle::new()
          .set("id", VERTICAL_LINE_ID)
          .set("width", point_radius)
          .set("height", step_y)
          .set("fill-opacity", config.filling_alpha),
      )
      .add(
        Rectangle::new()
          .set("id", HORIZONTAL_LINE_ID)
          .set("width", step_x)
          .set("height", point_radius)
          .set("fill-opacity", config.filling_alpha),
      );
  }
  if config.pointer && config.shared_pointer {
    defs = defs.add(
      Group::new()
        .set("id", POINTER_ID)
        .set("fill-opacity", 0)
        .set("pointer-events", "bounding-box")
        .add(
          Circle::new()
            .set("r", point_radius)
            .set("cx", (step_x / 2.0).round() + offset)
            .set("cy", (step_y / 2.0).round() + offset)
            .set("fill", color(extended_field.player))
            .set("shape-rendering", "geometricPrecision"),
        )
        .add(
          Rectangle::new()
            .set("width", step_x)
            .set("height", step_y)
            .set("fill-opacity", 0)
            .set("shape-rendering", "crispEdges"),
        )
        .add(
          Animate::new()
            .set("attributeName", "fill-opacity")
            .set("values", "0;0.5")
            .set("dur", "100ms")
            .set("repeatCount", 1)
            .set("begin", "mouseover")
            .set("fill", "freeze"),
        )
        .add(
          Animate::new()
            .set("attributeName", "fill-opacity")
            .set("values", "0.5;0")
            .set("dur", "100ms")
            .set("repeatCount", 1)
            .set("begin", "mouseout")
            .set("fill", "freeze"),
        ),
    );
  }
  let mut document = Document::new()
    .set("viewBox", (0, 0, width, height))
    .set("width", width)
    .set("height", height)
    .add(defs);

  // background

  let background = Rectangle::new()
    .set("width", width)
    .set("height", height)
    .set("fill", config.background_color.as_ref());
  document = document.add(background);

  // grid

  let mut data = Data::new();
  for x in 0..field_width {
    let offset = (step_x * x as f32 + step_x / 2.0).round() + offset;
    data = data.move_to((offset, 0));
    data = data.vertical_line_to(height.round());
  }
  for y in 0..field_height {
    let offset = (step_y * y as f32 + step_y / 2.0).round() + offset;
    data = data.move_to((0, offset));
    data = data.horizontal_line_to(width.round());
  }
  let grid = Path::new()
    .set("fill", "none")
    .set("stroke", config.grid_color.as_ref())
    .set("stroke-width", config.grid_thickness)
    .set("shape-rendering", "crispEdges")
    .set("d", data);
  document = document.add(grid);

  // points

  for &pos in &extended_field.field.moves {
    let color = color(extended_field.field.cell(pos).get_player());
    let (x, y) = pos_to_point(pos);
    let circle = Use::new()
      .set("href", DOT_REF)
      .set("fill", color)
      .set("x", x)
      .set("y", y);
    document = document.add(circle);
  }

  // fill extended area to display connecting lines

  if config.extended_filling {
    for &pos in &extended_field.field.moves {
      let player = extended_field.field.cell(pos).get_player();
      let color = color(player);
      let (x, y) = pos_to_point(pos);
      let captured = extended_field.captured[pos];
      let is_owner = |pos: Pos| -> bool {
        extended_field.field.cell(pos).is_players_point(player)
          || extended_field.captured[pos] > 0 && (captured == 0 || extended_field.captured[pos] < captured)
      };

      // draw vertical lines

      if extended_field
        .field
        .cell(extended_field.field.s(pos))
        .is_players_point(player)
      {
        if !is_owner(extended_field.field.w(pos)) && !is_owner(extended_field.field.sw(pos)) {
          let rectangle = Use::new()
            .set("href", VERTICAL_LINE_REF)
            .set("x", x - point_radius)
            .set("y", y)
            .set("fill", color);
          document = document.add(rectangle);
        }

        if !is_owner(extended_field.field.e(pos)) && !is_owner(extended_field.field.se(pos)) {
          let rectangle = Use::new()
            .set("href", VERTICAL_LINE_REF)
            .set("x", x)
            .set("y", y)
            .set("fill", color);
          document = document.add(rectangle);
        }
      }

      // draw horizontal lines

      if extended_field
        .field
        .cell(extended_field.field.e(pos))
        .is_players_point(player)
      {
        if !is_owner(extended_field.field.n(pos)) && !is_owner(extended_field.field.ne(pos)) {
          let rectangle = Use::new()
            .set("href", HORIZONTAL_LINE_REF)
            .set("x", x)
            .set("y", y - point_radius)
            .set("fill", color);
          document = document.add(rectangle);
        }

        if !is_owner(extended_field.field.s(pos)) && !is_owner(extended_field.field.se(pos)) {
          let rectangle = Use::new()
            .set("href", HORIZONTAL_LINE_REF)
            .set("x", x)
            .set("y", y)
            .set("fill", color);
          document = document.add(rectangle);
        }
      }

      // draw \ diagonal lines

      let diag_width = point_radius / 2f32.sqrt();

      if extended_field
        .field
        .cell(extended_field.field.se(pos))
        .is_players_point(player)
      {
        let (x2, y2) = pos_to_point(extended_field.field.se(pos));

        if is_owner(extended_field.field.e(pos)) && !is_owner(extended_field.field.s(pos)) {
          let data = Data::new()
            .move_to((x, y))
            .line_to((x - diag_width, y + diag_width))
            .line_to((x2 - diag_width, y2 + diag_width))
            .line_to((x2, y2))
            .close();
          let path = Path::new()
            .set("fill", color)
            .set("fill-opacity", config.filling_alpha)
            .set("shape-rendering", "crispEdges")
            .set("d", data);
          document = document.add(path);
        }

        if is_owner(extended_field.field.s(pos)) && !is_owner(extended_field.field.e(pos)) {
          let data = Data::new()
            .move_to((x, y))
            .line_to((x + diag_width, y - diag_width))
            .line_to((x2 + diag_width, y2 - diag_width))
            .line_to((x2, y2))
            .close();
          let path = Path::new()
            .set("fill", color)
            .set("fill-opacity", config.filling_alpha)
            .set("shape-rendering", "crispEdges")
            .set("d", data);
          document = document.add(path);
        }
      }

      // draw / diagonal lines

      if extended_field
        .field
        .cell(extended_field.field.ne(pos))
        .is_players_point(player)
      {
        let (x2, y2) = pos_to_point(extended_field.field.ne(pos));

        if is_owner(extended_field.field.e(pos)) && !is_owner(extended_field.field.n(pos)) {
          let data = Data::new()
            .move_to((x, y))
            .line_to((x - diag_width, y - diag_width))
            .line_to((x2 - diag_width, y2 - diag_width))
            .line_to((x2, y2))
            .close();
          let path = Path::new()
            .set("fill", color)
            .set("fill-opacity", config.filling_alpha)
            .set("shape-rendering", "crispEdges")
            .set("d", data);
          document = document.add(path);
        }

        if is_owner(extended_field.field.n(pos)) && !is_owner(extended_field.field.e(pos)) {
          let data = Data::new()
            .move_to((x, y))
            .line_to((x + diag_width, y + diag_width))
            .line_to((x2 + diag_width, y2 + diag_width))
            .line_to((x2, y2))
            .close();
          let path = Path::new()
            .set("fill", color)
            .set("fill-opacity", config.filling_alpha)
            .set("shape-rendering", "crispEdges")
            .set("d", data);
          document = document.add(path);
        }
      }
    }
  }

  // captures

  for (chain, player, _) in &extended_field.captures {
    if !config.maximum_area_filling && chain.len() < 4 {
      continue;
    }

    let mut data = Data::new().move_to(pos_to_point(chain[0]));
    for &pos in chain.iter().skip(1) {
      data = data.line_to(pos_to_point(pos));
    }

    let path = Path::new()
      .set("fill", color(*player))
      .set("fill-opacity", config.filling_alpha)
      .set("shape-rendering", "crispEdges")
      .set("d", data);
    document = document.add(path);
  }

  // last point

  if config.last_point_mark {
    if let Some(&pos) = extended_field.field.moves.last() {
      let (x, y) = pos_to_point(pos);
      let color = color(extended_field.field.cell(pos).get_player());
      let circle = Circle::new()
        .set("cx", x)
        .set("cy", y)
        .set("r", point_radius * 1.5)
        .set("fill", "none")
        .set("stroke", color)
        .set("stroke-width", 2)
        .set("shape-rendering", "geometricPrecision");
      document = document.add(circle);
    }
  }

  // pointer

  if config.pointer {
    for pos in (extended_field.field.min_pos()..=extended_field.field.max_pos())
      .filter(|&pos| extended_field.field.is_putting_allowed(pos))
    {
      let x = step_x * extended_field.field.to_x(pos) as f32;
      let y = step_y * extended_field.field.to_y(pos) as f32;

      if config.shared_pointer {
        let pointer = Use::new().set("href", POINTER_REF).set("x", x).set("y", y);
        document = document.add(pointer);
      } else {
        let pointer = Group::new()
          .set("fill-opacity", 0)
          .set("pointer-events", "bounding-box")
          .add(
            Circle::new()
              .set("r", point_radius)
              .set("cx", (x + step_x / 2.0).round() + offset)
              .set("cy", (y + step_y / 2.0).round() + offset)
              .set("fill", color(extended_field.player))
              .set("shape-rendering", "geometricPrecision"),
          )
          .add(
            Rectangle::new()
              .set("x", x)
              .set("y", y)
              .set("width", step_x)
              .set("height", step_y)
              .set("fill-opacity", 0)
              .set("shape-rendering", "crispEdges"),
          )
          .add(
            Animate::new()
              .set("attributeName", "fill-opacity")
              .set("values", "0;0.5")
              .set("dur", "100ms")
              .set("repeatCount", 1)
              .set("begin", "mouseover")
              .set("fill", "freeze"),
          )
          .add(
            Animate::new()
              .set("attributeName", "fill-opacity")
              .set("values", "0.5;0")
              .set("dur", "100ms")
              .set("repeatCount", 1)
              .set("begin", "mouseout")
              .set("fill", "freeze"),
          );
        document = document.add(pointer);
      }
    }
  }

  document
}
