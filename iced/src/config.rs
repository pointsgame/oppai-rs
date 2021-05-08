use crate::extended_field::InitialPosition;
use clap::{crate_authors, crate_description, crate_name, crate_version, value_t, App, Arg};
use oppai_bot::cli::*;
use oppai_bot::config::Config as BotConfig;
use std::num::ParseIntError;
use std::str::FromStr;
use std::time::Duration;

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
pub struct Config {
  pub width: u32,
  pub height: u32,
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
  pub initial_position: InitialPosition,
  pub patterns: Vec<String>,
  pub bot_config: BotConfig,
  pub time: Duration,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 39,
      height: 32,
      red_color: RED,
      black_color: BLACK,
      grid_color: BLACK,
      background_color: WHITE,
      grid_thickness: 1.0,
      point_radius: 0.166667,
      filling_alpha: 0.5,
      extended_filling: true,
      maximum_area_filling: true,
      last_point_mark: true,
      initial_position: InitialPosition::Cross,
      patterns: Vec::new(),
      bot_config: BotConfig::default(),
      time: Duration::from_secs(5),
    }
  }
}

pub fn cli_parse() -> Config {
  let matches = App::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .groups(&groups())
    .args(&args())
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
    .arg(
      Arg::with_name("background-color")
        .long("background-color")
        .help("The background color")
        .takes_value(true)
        .default_value("#FFFFFF"),
    )
    .arg(
      Arg::with_name("grid-thickness")
        .long("grid-thickness")
        .help("The grid thickness")
        .takes_value(true)
        .default_value("1"),
    )
    .arg(
      Arg::with_name("point-radius")
        .long("point-radius")
        .help("The point radius")
        .takes_value(true)
        .default_value("0.166667"),
    )
    .arg(
      Arg::with_name("filling-alpha")
        .long("filling-alpha")
        .help("The degree of filling transparency")
        .takes_value(true)
        .default_value("0.5"),
    )
    .arg(
      Arg::with_name("no-extended-filling")
        .long("no-extended-filling")
        .help("Disable extended area filling, changes appearance only"),
    )
    .arg(
      Arg::with_name("no-maximum-area-filling")
        .long("no-maximum-area-filling")
        .help("Disable filling captures by maximum area, changes appearance only")
        .requires("no-extended-filling"),
    )
    .arg(
      Arg::with_name("no-last-point-mark")
        .long("no-last-point-mark")
        .help("Don't mark last point"),
    )
    .arg(
      Arg::with_name("initial-position")
        .long("initial-position")
        .help("Initial position on the field")
        .takes_value(true)
        .possible_values(&InitialPosition::variants())
        .case_insensitive(true)
        .default_value("Cross"),
    )
    .arg(
      Arg::with_name("patterns-file")
        .short("p")
        .long("patterns-file")
        .help("Patterns file to use")
        .takes_value(true)
        .multiple(true),
    )
    .arg(
      Arg::with_name("time")
        .long("time")
        .help("Time to think that AI will use for one move")
        .takes_value(true)
        .default_value("5s"),
    )
    .get_matches();

  let width = value_t!(matches.value_of("width"), u32).unwrap_or_else(|e| e.exit());
  let height = value_t!(matches.value_of("height"), u32).unwrap_or_else(|e| e.exit());
  let red_color = value_t!(matches.value_of("red-color"), Rgb).unwrap_or_else(|e| e.exit());
  let black_color = value_t!(matches.value_of("black-color"), Rgb).unwrap_or_else(|e| e.exit());
  let grid_color = value_t!(matches.value_of("grid-color"), Rgb).unwrap_or_else(|e| e.exit());
  let background_color = value_t!(matches.value_of("background-color"), Rgb).unwrap_or_else(|e| e.exit());
  let grid_thickness = value_t!(matches.value_of("grid-thickness"), f32).unwrap_or_else(|e| e.exit());
  let point_radius = value_t!(matches.value_of("point-radius"), f32).unwrap_or_else(|e| e.exit());
  let filling_alpha = value_t!(matches.value_of("filling-alpha"), f32).unwrap_or_else(|e| e.exit());
  let extended_filling = !matches.is_present("no-extended-filling");
  let maximum_area_filling = !matches.is_present("no-maximum-area-filling");
  let last_point_mark = !matches.is_present("no-last-point-mark");
  let initial_position = value_t!(matches.value_of("initial-position"), InitialPosition).unwrap_or_else(|e| e.exit());
  let patterns = if matches.is_present("patterns-file") {
    clap::values_t!(matches.values_of("patterns-file"), String).unwrap_or_else(|e| e.exit())
  } else {
    Vec::new()
  };
  let bot_config = parse_config(&matches);
  let time = value_t!(matches.value_of("time"), humantime::Duration)
    .unwrap_or_else(|e| e.exit())
    .into();

  Config {
    width,
    height,
    red_color,
    black_color,
    grid_color,
    background_color,
    grid_thickness,
    point_radius,
    filling_alpha,
    extended_filling,
    maximum_area_filling,
    last_point_mark,
    initial_position,
    patterns,
    bot_config,
    time,
  }
}
