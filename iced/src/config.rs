use clap::{crate_authors, crate_description, crate_name, crate_version, value_parser, Arg, ArgAction, Command};
use oppai_bot::cli::*;
use oppai_bot::config::Config as BotConfig;
use oppai_initial::initial::InitialPosition;
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
  let command = Command::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .groups(groups())
    .args(&args())
    .arg(
      Arg::new("width")
        .long("width")
        .help("Field width")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("39"),
    )
    .arg(
      Arg::new("height")
        .long("height")
        .help("Field height")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("32"),
    )
    .arg(
      Arg::new("red-color")
        .long("red-color")
        .help("The color of first player")
        .num_args(1)
        .value_parser(value_parser!(Rgb))
        .default_value("#FF0000"),
    )
    .arg(
      Arg::new("black-color")
        .long("black-color")
        .help("The color of second player")
        .num_args(1)
        .value_parser(value_parser!(Rgb))
        .default_value("#000000"),
    )
    .arg(
      Arg::new("grid-color")
        .long("grid-color")
        .help("The color of grid")
        .num_args(1)
        .value_parser(value_parser!(Rgb))
        .default_value("#000000"),
    )
    .arg(
      Arg::new("background-color")
        .long("background-color")
        .help("The background color")
        .num_args(1)
        .value_parser(value_parser!(Rgb))
        .default_value("#FFFFFF"),
    )
    .arg(
      Arg::new("grid-thickness")
        .long("grid-thickness")
        .help("The grid thickness")
        .num_args(1)
        .value_parser(value_parser!(f32))
        .default_value("1"),
    )
    .arg(
      Arg::new("point-radius")
        .long("point-radius")
        .help("The point radius")
        .num_args(1)
        .value_parser(value_parser!(f32))
        .default_value("0.166667"),
    )
    .arg(
      Arg::new("filling-alpha")
        .long("filling-alpha")
        .help("The degree of filling transparency")
        .num_args(1)
        .value_parser(value_parser!(f32))
        .default_value("0.5"),
    )
    .arg(
      Arg::new("no-extended-filling")
        .long("no-extended-filling")
        .help("Disable extended area filling, changes appearance only")
        .action(ArgAction::SetFalse),
    )
    .arg(
      Arg::new("no-maximum-area-filling")
        .long("no-maximum-area-filling")
        .help("Disable filling captures by maximum area, changes appearance only")
        .requires("no-extended-filling")
        .action(ArgAction::SetFalse),
    )
    .arg(
      Arg::new("no-last-point-mark")
        .long("no-last-point-mark")
        .help("Don't mark last point")
        .action(ArgAction::SetFalse),
    )
    .arg(
      Arg::new("initial-position")
        .long("initial-position")
        .help("Initial position on the field")
        .num_args(1)
        .value_parser(value_parser!(InitialPosition))
        .ignore_case(true)
        .default_value("Cross"),
    )
    .arg(
      Arg::new("patterns-file")
        .short('p')
        .long("patterns-file")
        .help("Patterns file to use")
        .num_args(1..),
    )
    .arg(
      Arg::new("time")
        .long("time")
        .help("Time to think that AI will use for one move")
        .num_args(1)
        .value_parser(value_parser!(humantime::Duration))
        .default_value("5s"),
    );

  #[cfg(not(target_arch = "wasm32"))]
  let matches = command.get_matches();

  #[cfg(target_arch = "wasm32")]
  let matches = {
    let mut args = vec!["oppai".to_owned()];

    let window = web_sys::window().unwrap();
    let search = window.location().search().unwrap();
    for pair in search.trim_start_matches('?').split('&') {
      let mut it = pair.split('=').take(2);
      if let Some(k) = it.next() {
        let mut k = k.to_owned();
        k.insert_str(0, "--");
        args.push(k);
      }
      if let Some(v) = it.next() {
        args.push(v.to_owned());
      }
    }

    command.get_matches_from(args)
  };

  let width = matches.get_one("width").copied().unwrap();
  let height = matches.get_one("height").copied().unwrap();
  let red_color = matches.get_one("red-color").copied().unwrap();
  let black_color = matches.get_one("black-color").copied().unwrap();
  let grid_color = matches.get_one("grid-color").copied().unwrap();
  let background_color = matches.get_one("background-color").copied().unwrap();
  let grid_thickness = matches.get_one("grid-thickness").copied().unwrap();
  let point_radius = matches.get_one("point-radius").copied().unwrap();
  let filling_alpha = matches.get_one("filling-alpha").copied().unwrap();
  let extended_filling = matches.get_flag("no-extended-filling");
  let maximum_area_filling = matches.get_flag("no-maximum-area-filling");
  let last_point_mark = matches.get_flag("no-last-point-mark");
  let initial_position = matches.get_one("initial-position").copied().unwrap();
  let patterns = matches
    .get_many("patterns-file")
    .map_or_else(Vec::new, |patterns| patterns.cloned().collect());
  let bot_config = parse_config(&matches);
  let time = matches.get_one::<humantime::Duration>("time").copied().unwrap().into();

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
