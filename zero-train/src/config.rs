use clap::{crate_authors, crate_description, crate_name, crate_version, value_parser, Arg, ArgAction, Command};

pub struct Config {
  pub width: u32,
  pub height: u32,
  pub device: String,
  pub library: Option<String>,
  pub double: bool,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      width: 16,
      height: 16,
      device: "cpu".to_string(),
      library: None,
      double: false,
    }
  }
}

pub fn cli_parse() -> Config {
  let matches = Command::new(crate_name!())
    .version(crate_version!())
    .author(crate_authors!("\n"))
    .about(crate_description!())
    .arg(
      Arg::new("width")
        .long("width")
        .help("Field width")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("16"),
    )
    .arg(
      Arg::new("height")
        .long("height")
        .help("Field height")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .default_value("16"),
    )
    .arg(
      Arg::new("device")
        .long("device")
        .help("Device to run pytorch network")
        .num_args(1)
        .default_value("cpu"),
    )
    .arg(
      Arg::new("library")
        .long("library")
        .help("Load pytorch dynamic library")
        .num_args(1),
    )
    .arg(
      Arg::new("double")
        .long("double")
        .help("Use double precision type (float64) for calculations")
        .action(ArgAction::SetTrue),
    )
    .get_matches();

  let width = matches.get_one("width").copied().unwrap();
  let height = matches.get_one("height").copied().unwrap();
  let device = matches.get_one("device").cloned().unwrap();
  let library = matches.get_one("library").cloned();
  let double = matches.get_flag("double");

  Config {
    width,
    height,
    device,
    library,
    double,
  }
}
