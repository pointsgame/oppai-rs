use clap::{Arg, Command};
use cookie::Key;
use openidconnect::{ClientId, ClientSecret, url::Url};

#[derive(Clone, Debug)]
pub struct OidcConfig {
  pub issuer_url: Url,
  pub client_id: ClientId,
  pub client_secret: Option<ClientSecret>,
}

#[derive(Clone, Debug)]
pub struct Config {
  pub oidc: OidcConfig,
  #[cfg(not(feature = "in-memory"))]
  pub postgres_socket: String,
  pub cookie_key: Key,
}

pub fn cli_parse() -> Config {
  let command = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .arg(
      Arg::new("oidc-issuer-url")
        .long("oidc-issuer-url")
        .help("OIDC issuer URL")
        .num_args(1)
        .required(true)
        .env("OIDC_ISSUER_URL"),
    )
    .arg(
      Arg::new("oidc-client-id")
        .long("oidc-client-id")
        .help("OIDC client ID")
        .num_args(1)
        .required(true)
        .env("OIDC_CLIENT_ID"),
    )
    .arg(
      Arg::new("oidc-client-secret")
        .long("oidc-client-secret")
        .help("OIDC client secret")
        .num_args(1)
        .env("OIDC_CLIENT_SECRET"),
    )
    .arg(
      Arg::new("cookie-key")
        .long("cookie-key")
        .help("Cookie secret key")
        .num_args(1)
        .env("COOKIE_KEY"),
    );
  #[cfg(not(feature = "in-memory"))]
  let command = command.arg(
    Arg::new("postgres-socket")
      .long("postgres-socket")
      .help("Postgres UNIX socket")
      .num_args(1)
      .required(true)
      .env("POSTGRES_SOCKET"),
  );
  let matches = command.get_matches();
  let issuer_url = matches.get_one::<String>("oidc-issuer-url").cloned().unwrap();
  let client_id = matches.get_one::<String>("oidc-client-id").cloned().unwrap();
  let client_secret = matches.get_one::<String>("oidc-client-secret").cloned();

  let oidc = OidcConfig {
    issuer_url: Url::parse(&issuer_url).expect("Invalid OIDC issuer URL"),
    client_id: ClientId::new(client_id),
    client_secret: client_secret.map(ClientSecret::new),
  };

  let cookie_key = matches.get_one::<String>("cookie-key").map_or_else(Key::generate, |s| {
    Key::from(hex::decode(s.as_str()).unwrap().as_slice())
  });
  Config {
    oidc,
    #[cfg(not(feature = "in-memory"))]
    postgres_socket: matches.get_one("postgres-socket").cloned().unwrap(),
    cookie_key,
  }
}
