use clap::{crate_authors, crate_description, crate_name, crate_version, value_t, App, Arg};
use std::num::ParseIntError;
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
pub struct RGB {
  pub r: u8,
  pub g: u8,
  pub b: u8,
}

impl FromStr for RGB {
  type Err = ParseIntError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    let r: u8 = u8::from_str_radix(&s[1..3], 16)?;
    let g: u8 = u8::from_str_radix(&s[3..5], 16)?;
    let b: u8 = u8::from_str_radix(&s[5..7], 16)?;

    Ok(RGB { r, g, b })
  }
}

pub const RED: RGB = RGB {
  r: 0xFF,
  g: 0x00,
  b: 0x00,
};

pub const BLACK: RGB = RGB {
  r: 0x00,
  g: 0x00,
  b: 0x00,
};

#[derive(Debug, Clone)]
pub struct Config {
  pub width: u32,
  pub height: u32,
  pub red_color: RGB,
  pub black_color: RGB,
  pub grid_color: RGB,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 39,
      height: 32,
      red_color: RED,
      black_color: BLACK,
      grid_color: BLACK,
    }
  }
}

pub fn cli_parse() -> Config {
  let matches = App::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .arg(
      Arg::with_name("width")
        .long("width")
        .help("Field width")
        .takes_value(true)
        .default_value("39"),
    )
    .arg(
      Arg::with_name("height")
        .long("height")
        .help("Field height")
        .takes_value(true)
        .default_value("32"),
    )
    .arg(
      Arg::with_name("red-color")
        .long("red-color")
        .help("The color of first player")
        .takes_value(true)
        .default_value("#FF0000"),
    )
    .arg(
      Arg::with_name("black-color")
        .long("black-color")
        .help("The color of second player")
        .takes_value(true)
        .default_value("#000000"),
    )
    .arg(
      Arg::with_name("grid-color")
        .long("grid-color")
        .help("The color of grid")
        .takes_value(true)
        .default_value("#000000"),
    )
    .get_matches();

  let width = value_t!(matches.value_of("width"), u32).unwrap_or_else(|e| e.exit());
  let height = value_t!(matches.value_of("height"), u32).unwrap_or_else(|e| e.exit());
  let red_color = value_t!(matches.value_of("red-color"), RGB).unwrap_or_else(|e| e.exit());
  let black_color = value_t!(matches.value_of("black-color"), RGB).unwrap_or_else(|e| e.exit());
  let grid_color = value_t!(matches.value_of("grid-color"), RGB).unwrap_or_else(|e| e.exit());

  Config {
    width,
    height,
    red_color,
    black_color,
    grid_color,
  }
}
