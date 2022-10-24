use clap::{crate_authors, crate_description, crate_name, crate_version, Arg, Command};

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
        .takes_value(true)
        .default_value("16"),
    )
    .arg(
      Arg::new("height")
        .long("height")
        .help("Field height")
        .takes_value(true)
        .default_value("16"),
    )
    .arg(
      Arg::new("device")
        .long("device")
        .help("Device to run pytorch network")
        .takes_value(true)
        .default_value("cpu"),
    )
    .arg(
      Arg::new("library")
        .long("library")
        .help("Load pytorch dynamic library")
        .takes_value(true),
    )
    .arg(
      Arg::new("double")
        .long("double")
        .help("Use double precision type (float64) for calculations"),
    )
    .get_matches();

  let width = matches.value_of_t("width").unwrap_or_else(|e| e.exit());
  let height = matches.value_of_t("height").unwrap_or_else(|e| e.exit());
  let device = matches.value_of_t("device").unwrap_or_else(|e| e.exit());
  let library = matches.get_one::<String>("library").cloned();
  let double = matches.is_present("double");

  Config {
    width,
    height,
    device,
    library,
    double,
  }
}
