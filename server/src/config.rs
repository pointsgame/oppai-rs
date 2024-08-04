use clap::{Arg, Command};
use cookie::Key;
use openidconnect::{ClientId, ClientSecret};

pub struct OidcConfig {
  pub client_id: ClientId,
  pub client_secret: ClientSecret,
}

pub struct Config {
  pub google_oidc: OidcConfig,
  pub gitlab_oidc: OidcConfig,
  pub postgres_socket: String,
  pub cookie_key: Key,
}

pub fn cli_parse() -> Config {
  let matches = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .arg(
      Arg::new("google-oidc-client-id")
        .long("google-oidc-client-id")
        .help("Google OIDC client ID")
        .num_args(1)
        .required(true)
        .env("GOOGLE_OIDC_CLIENT_ID"),
    )
    .arg(
      Arg::new("google-oidc-client-secret")
        .long("google-oidc-client-secret")
        .help("Google OIDC client secret")
        .num_args(1)
        .required(true)
        .env("GOOGLE_OIDC_CLIENT_SECRET"),
    )
    .arg(
      Arg::new("gitlab-oidc-client-id")
        .long("gitlab-oidc-client-id")
        .help("GitLab OIDC client ID")
        .num_args(1)
        .required(true)
        .env("GITLAB_OIDC_CLIENT_ID"),
    )
    .arg(
      Arg::new("gitlab-oidc-client-secret")
        .long("gitlab-oidc-client-secret")
        .help("GitLab OIDC client secret")
        .num_args(1)
        .required(true)
        .env("GITLAB_OIDC_CLIENT_SECRET"),
    )
    .arg(
      Arg::new("postgres-socket")
        .long("postgres-socket")
        .help("Postgres UNIX socket")
        .num_args(1)
        .required(true)
        .env("POSTGRES_SOCKET"),
    )
    .arg(
      Arg::new("cookie-key")
        .long("cookie-key")
        .help("Cookie secret key")
        .num_args(1)
        .required(true)
        .env("COOKIE_KEY"),
    )
    .get_matches();
  Config {
    google_oidc: OidcConfig {
      client_id: ClientId::new(matches.get_one("google-oidc-client-id").cloned().unwrap()),
      client_secret: ClientSecret::new(matches.get_one("google-oidc-client-secret").cloned().unwrap()),
    },
    gitlab_oidc: OidcConfig {
      client_id: ClientId::new(matches.get_one("gitlab-oidc-client-id").cloned().unwrap()),
      client_secret: ClientSecret::new(matches.get_one("gitlab-oidc-client-secret").cloned().unwrap()),
    },
    postgres_socket: matches.get_one("postgres-socket").cloned().unwrap(),
    cookie_key: Key::from(
      hex::decode(matches.get_one::<String>("cookie-key").cloned().unwrap().as_str())
        .unwrap()
        .as_slice(),
    ),
  }
}
