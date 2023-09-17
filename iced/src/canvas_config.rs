use std::num::ParseIntError;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub struct Rgb {
  pub r: u8,
  pub g: u8,
  pub b: u8,
}

impl FromStr for Rgb {
  type Err = ParseIntError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    let r: u8 = u8::from_str_radix(&s[1..3], 16)?;
    let g: u8 = u8::from_str_radix(&s[3..5], 16)?;
    let b: u8 = u8::from_str_radix(&s[5..7], 16)?;

    Ok(Rgb { r, g, b })
  }
}

pub const RED: Rgb = Rgb {
  r: 0xFF,
  g: 0x00,
  b: 0x00,
};

pub const BLACK: Rgb = Rgb {
  r: 0x00,
  g: 0x00,
  b: 0x00,
};

pub const WHITE: Rgb = Rgb {
  r: 0xFF,
  g: 0xFF,
  b: 0xFF,
};

#[derive(Debug, Clone)]
pub struct CanvasConfig {
  pub red_color: Rgb,
  pub black_color: Rgb,
  pub grid_color: Rgb,
  pub background_color: Rgb,
  pub grid_thickness: f32,
  pub point_radius: f32,
  pub filling_alpha: f32,
  pub extended_filling: bool,
  pub maximum_area_filling: bool,
  pub last_point_mark: bool,
}

impl Default for CanvasConfig {
  fn default() -> Self {
    Self {
      red_color: RED,
      black_color: BLACK,
      grid_color: BLACK,
      background_color: WHITE,
      grid_thickness: 1.0,
      point_radius: 1.0 / 6.0,
      filling_alpha: 0.5,
      extended_filling: true,
      maximum_area_filling: true,
      last_point_mark: true,
    }
  }
}
