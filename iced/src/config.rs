use clap::{crate_authors, crate_description, crate_name, crate_version, value_t, App, Arg};

pub struct Config {
  pub width: u32,
  pub height: u32,
}

impl Default for Config {
  fn default() -> Self {
    Self { width: 39, height: 32 }
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
    .get_matches();

  let width = value_t!(matches.value_of("width"), u32).unwrap_or_else(|e| e.exit());
  let height = value_t!(matches.value_of("height"), u32).unwrap_or_else(|e| e.exit());

  Config { width, height }
}
