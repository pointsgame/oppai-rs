use anyhow::{Error, Result};
use cookie::time::{Duration, OffsetDateTime};
use cookie::{Cookie, CookieJar, Expiration, Key, SameSite};
use futures::channel::mpsc::{self, Sender};
use futures_util::{select, FutureExt, SinkExt, StreamExt};
use ids::*;
use im::HashSet;
use openidconnect::{
  core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata},
  AccessTokenHash, AuthorizationCode, CsrfToken, EndpointMaybeSet, EndpointNotSet, EndpointSet, IssuerUrl, Nonce,
  OAuth2TokenResponse, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, Scope, TokenResponse,
};
use oppai_field::{field::Field, player::Player};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use state::{FieldSize, Game, OpenGame, State};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::{
  net::{TcpListener, TcpStream},
  sync::RwLock,
};
use tokio_tungstenite::tungstenite::handshake::server::Request;
use tokio_tungstenite::tungstenite::Message;
use uuid::Builder;

mod config;
mod db;
mod ids;
mod message;
mod state;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
struct CookieData {
  player_id: PlayerId,
  expires_at: SystemTime,
}

type OidcClient =
  CoreClient<EndpointSet, EndpointNotSet, EndpointNotSet, EndpointNotSet, EndpointMaybeSet, EndpointMaybeSet>;

struct AuthState {
  provider: message::AuthProvider,
  pkce_verifier: PkceCodeVerifier,
  nonce: Nonce,
  csrf_state: CsrfToken,
  remember_me: bool,
}

struct OidcClients {
  google_client: Arc<OidcClient>,
  gitlab_client: Arc<OidcClient>,
}

impl OidcClients {
  fn oidc_client(&self, provider: message::AuthProvider) -> &OidcClient {
    match provider {
      message::AuthProvider::Google => &self.google_client,
      message::AuthProvider::GitLab => &self.gitlab_client,
    }
  }
}

struct Session<R: Rng> {
  db: Arc<db::SqlxDb>,
  http_client: Arc<reqwest::Client>,
  oidc_clients: OidcClients,
  rng: R,
  connection_id: ConnectionId,
  player_id: Option<PlayerId>,
  watching: HashSet<GameId>,
  auth_state: Option<AuthState>,
  cookie_key: Arc<Key>,
}

impl<R: Rng> Session<R> {
  fn new(
    mut rng: R,
    db: Arc<db::SqlxDb>,
    http_client: Arc<reqwest::Client>,
    google_client: Arc<OidcClient>,
    gitlab_client: Arc<OidcClient>,
    cookie_key: Arc<Key>,
  ) -> Self {
    let connection_id = ConnectionId(Builder::from_random_bytes(rng.gen()).into_uuid());
    Session {
      db,
      http_client,
      oidc_clients: {
        OidcClients {
          google_client,
          gitlab_client,
        }
      },
      rng,
      connection_id,
      player_id: None,
      watching: HashSet::new(),
      auth_state: None,
      cookie_key,
    }
  }

  async fn get_auth_url(&mut self, state: &State, provider: message::AuthProvider, remember_me: bool) -> Result<()> {
    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
    let (auth_url, csrf_state, nonce) = self
      .oidc_clients
      .oidc_client(provider)
      .authorize_url(
        CoreAuthenticationFlow::AuthorizationCode,
        CsrfToken::new_random,
        Nonce::new_random,
      )
      .add_scope(Scope::new("email".to_string()))
      .add_scope(Scope::new("profile".to_string()))
      .set_pkce_challenge(pkce_challenge)
      .url();

    self.auth_state = Some(AuthState {
      provider,
      pkce_verifier,
      nonce,
      csrf_state,
      remember_me,
    });

    state
      .send_to_connection(
        self.connection_id,
        message::Response::AuthUrl {
          url: auth_url.to_string(),
        },
      )
      .await?;

    Ok(())
  }

  async fn auth(&mut self, state: &State, oidc_code: String, oidc_state: String) -> Result<()> {
    let auth_state = self
      .auth_state
      .take()
      .ok_or_else(|| anyhow::anyhow!("no auth state forconnection {}", self.connection_id))?;

    if auth_state.csrf_state.secret() != CsrfToken::new(oidc_state).secret() {
      anyhow::bail!("invalid csrf token for connection {}", self.connection_id);
    }

    let token_response = self
      .oidc_clients
      .oidc_client(auth_state.provider)
      .exchange_code(AuthorizationCode::new(oidc_code))?
      .set_pkce_verifier(auth_state.pkce_verifier)
      .request_async(self.http_client.as_ref())
      .await?;

    let id_token = token_response.id_token().ok_or_else(|| {
      anyhow::anyhow!(
        "server did not return an ID token for connection {}",
        self.connection_id
      )
    })?;
    let id_token_verifier = self.oidc_clients.oidc_client(auth_state.provider).id_token_verifier();
    let claims = id_token.claims(&id_token_verifier, &auth_state.nonce)?;

    if let Some(expected_access_token_hash) = claims.access_token_hash() {
      let actual_access_token_hash = AccessTokenHash::from_token(
        token_response.access_token(),
        id_token.signing_alg()?,
        id_token.signing_key(&id_token_verifier)?,
      )?;
      if actual_access_token_hash != *expected_access_token_hash {
        anyhow::bail!("invalid access token for connection {}", self.connection_id);
      }
    }

    let db_provider = match auth_state.provider {
      message::AuthProvider::Google => db::Provider::Google,
      message::AuthProvider::GitLab => db::Provider::GitLab,
    };

    let player_id = self
      .db
      .get_or_create_player(db::OidcPlayer {
        provider: db_provider,
        subject: claims.subject().to_string(),
        email: claims.email().map(|email| email.to_string()),
        email_verified: claims.email_verified(),
        name: claims
          .name()
          .and_then(|name| name.get(None))
          .map(|name| name.to_string()),
        nickname: claims
          .nickname()
          .and_then(|nickname| nickname.get(None))
          .map(|nickname| nickname.to_string()),
        preferred_username: claims
          .preferred_username()
          .map(|preferred_username| preferred_username.to_string()),
      })
      .await?;
    let player_id = PlayerId(player_id);

    self.player_id = Some(player_id);
    state.insert_players_connection(player_id, self.connection_id);

    let duration = if auth_state.remember_me {
      Duration::weeks(12)
    } else {
      Duration::weeks(1)
    };
    let mut jar = CookieJar::new();
    let cookie = Cookie::build((
      "kropki",
      serde_json::to_string(&CookieData {
        player_id,
        expires_at: SystemTime::now() + duration,
      })
      .unwrap(),
    ))
    .expires(if auth_state.remember_me {
      Expiration::DateTime(OffsetDateTime::now_utc() + duration)
    } else {
      Expiration::Session
    })
    .same_site(SameSite::Strict)
    .secure(true)
    .build();
    jar.private_mut(&self.cookie_key).add(cookie);

    state
      .send_to_connection(
        self.connection_id,
        message::Response::Auth {
          player_id,
          cookie: jar.get("kropki").unwrap().to_string(),
        },
      )
      .await?;

    Ok(())
  }

  #[cfg(feature = "test")]
  async fn auth_test(&mut self, state: &State, name: String) -> Result<()> {
    let player_id = self.db.get_or_create_test_player(name).await?;
    let player_id = PlayerId(player_id);

    self.player_id = Some(player_id);
    state.insert_players_connection(player_id, self.connection_id);

    let mut jar = CookieJar::new();
    let cookie = Cookie::build((
      "kropki",
      serde_json::to_string(&CookieData {
        player_id,
        expires_at: SystemTime::now() + Duration::weeks(1),
      })
      .unwrap(),
    ))
    .expires(Expiration::Session)
    .same_site(SameSite::Strict)
    .secure(true)
    .build();
    jar.private_mut(&self.cookie_key).add(cookie);

    state
      .send_to_connection(
        self.connection_id,
        message::Response::Auth {
          player_id,
          cookie: jar.get("kropki").unwrap().to_string(),
        },
      )
      .await?;

    Ok(())
  }

  async fn init(&self, state: &State, tx: Sender<message::Response>) -> Result<()> {
    // lock connection before inserting so we can be sure we send init message before any update
    let connection = Arc::new(RwLock::new(tx));
    let connection_c = connection.clone();
    let mut connection_c_lock = connection_c.write().await;

    state.connections.pin().insert(self.connection_id, connection);

    let open_games = state
      .open_games
      .pin()
      .iter()
      .map(|(&game_id, open_game)| message::OpenGame {
        game_id,
        player_id: open_game.player_id,
        size: message::FieldSize {
          width: open_game.size.width,
          height: open_game.size.height,
        },
      })
      .collect();
    let games = state
      .games
      .pin()
      .iter()
      .map(|(&game_id, game)| message::Game {
        game_id,
        red_player_id: game.red_player_id,
        black_player_id: game.black_player_id,
        size: message::FieldSize {
          width: game.size.width,
          height: game.size.height,
        },
      })
      .collect();
    let init = message::Response::Init {
      player_id: self.player_id,
      open_games,
      games,
    };
    connection_c_lock.send(init).await?;

    Ok(())
  }

  fn finalize(&self, state: &State) {
    for &game_id in &self.watching {
      state.unsubscribe(self.connection_id, game_id);
    }

    if let Some(player_id) = self.player_id {
      state.remove_players_connection(player_id, self.connection_id);
    }
  }

  async fn create(&mut self, state: &State, size: message::FieldSize) -> Result<()> {
    if !size.is_valid() {
      anyhow::bail!(
        "invalid filed size {}:{} from connection {}",
        size.width,
        size.height,
        self.connection_id
      );
    }

    let player_id = if let Some(player_id) = self.player_id {
      player_id
    } else {
      anyhow::bail!(
        "attempt to create a game from an unauthorized connection {}",
        self.connection_id
      )
    };

    if state
      .open_games
      .pin()
      .values()
      .filter(|open_game| open_game.player_id == player_id)
      .count()
      > 2
    {
      anyhow::bail!("too many open games for player {}", player_id);
    }

    let game_id = GameId(Builder::from_random_bytes(self.rng.gen()).into_uuid());
    let open_game = OpenGame {
      player_id,
      size: FieldSize {
        width: size.width,
        height: size.height,
      },
    };

    state.open_games.pin().insert(game_id, open_game);

    state
      .send_to_all(message::Response::Create {
        game_id,
        player_id,
        size,
      })
      .await;

    Ok(())
  }

  async fn close(&mut self, state: &State, game_id: GameId) -> Result<()> {
    let player_id = if let Some(player_id) = self.player_id {
      player_id
    } else {
      anyhow::bail!(
        "attempt to close a game from an unauthorized connection {}",
        self.connection_id
      )
    };

    if let Some(open_game) = state.open_games.pin().get(&game_id) {
      if player_id != open_game.player_id {
        anyhow::bail!(
          "attempt to close a wrong game {} from connection {}",
          game_id,
          self.connection_id
        )
      }
    } else {
      return Ok(());
    }

    if state.open_games.pin().remove(&game_id).is_some() {
      state.send_to_all(message::Response::Close { game_id }).await;
    }

    Ok(())
  }

  async fn join(&mut self, state: &State, game_id: GameId) -> Result<()> {
    let player_id = if let Some(player_id) = self.player_id {
      player_id
    } else {
      anyhow::bail!(
        "attempt to join a game from an unauthorized connection {}",
        self.connection_id
      )
    };

    let open_game = if let Some(open_game) = state.open_games.pin().remove(&game_id) {
      open_game.clone()
    } else {
      log::warn!(
        "Player {} attempted to join a game {} which dosn't exist",
        player_id,
        game_id
      );
      return Ok(());
    };

    if open_game.player_id == player_id {
      anyhow::bail!("attempt to join own game from player {}", player_id);
    }

    let field = Field::new_from_rng(open_game.size.width, open_game.size.height, &mut self.rng);
    let game = Game {
      red_player_id: open_game.player_id,
      black_player_id: player_id,
      size: open_game.size,
      field: Arc::new(RwLock::new(field)),
    };

    state.games.pin().insert(game_id, game);

    state
      .send_to_all(message::Response::Start {
        game_id,
        red_player_id: open_game.player_id,
        black_player_id: player_id,
      })
      .await;

    Ok(())
  }

  async fn subscribe(&mut self, state: &State, game_id: GameId) -> Result<()> {
    if self.watching.len() > 2 {
      anyhow::bail!("too many subscriptions from a connection {}", self.connection_id);
    }
    if self.watching.insert(game_id).is_some() {
      anyhow::bail!(
        "connection {} already watching the game {}",
        self.connection_id,
        game_id
      );
    }

    state.subscribe(self.connection_id, game_id);

    let field = if let Some(game) = state.games.pin().get(&game_id) {
      game.field.clone()
    } else {
      // TODO: log
      return Ok(());
    };
    let field = field.read().await;
    state
      .send_to_connection(
        self.connection_id,
        message::Response::GameInit {
          game_id,
          moves: field
            .colored_moves()
            .map(|(pos, player)| message::Move {
              coordinate: message::Coordinate {
                x: field.to_x(pos),
                y: field.to_y(pos),
              },
              player,
            })
            .collect(),
        },
      )
      .await
  }

  fn unsubscribe(&mut self, state: &State, game_id: GameId) -> Result<()> {
    if self.watching.remove(&game_id).is_none() {
      anyhow::bail!("connection {} not watching the game {}", self.connection_id, game_id);
    }

    state.unsubscribe(self.connection_id, game_id);

    Ok(())
  }

  async fn put_point(&self, state: &State, game_id: GameId, coordinate: message::Coordinate) -> Result<()> {
    let player_id = if let Some(player_id) = self.player_id {
      player_id
    } else {
      anyhow::bail!(
        "attempt to put a point from an unauthorized connection {}",
        self.connection_id
      )
    };

    let (field, player) = if let Some(game) = state.games.pin().get(&game_id) {
      let player = if let Some(player) = game.color(player_id) {
        player
      } else {
        anyhow::bail!(
          "player {} attempted to put point in a wrong game {}",
          player_id,
          game_id,
        );
      };
      (game.field.clone(), player)
    } else {
      anyhow::bail!(
        "player {} attempted to put point in a game {} that don't exist",
        player_id,
        game_id,
      );
    };

    let mut field = field.write().await;
    let pos = field.to_pos(coordinate.x, coordinate.y);

    if field.last_player().map_or(Player::Red, |player| player.next()) != player {
      anyhow::bail!(
        "player {} attempted to put point on opponent's turn in a game {}",
        player_id,
        game_id,
      );
    }

    if !field.put_point(pos, player) {
      anyhow::bail!(
        "player {} attempted tp put point on a wrong position {:?} in game {}",
        player_id,
        (coordinate.x, coordinate.y),
        game_id,
      );
    }
    drop(field);

    state
      .send_to_watchers(
        game_id,
        message::Response::PutPoint {
          game_id,
          coordinate,
          player,
        },
      )
      .await;

    Ok(())
  }

  async fn accept_connection(mut self, state: Arc<State>, stream: TcpStream) -> Result<()> {
    let ws_stream = tokio_tungstenite::accept_hdr_async(stream, |request: &Request, response| {
      let mut jar = CookieJar::new();
      if let Some(cookie) = request
        .headers()
        .get("Cookie")
        .and_then(|cookie| cookie.to_str().ok())
        .and_then(|cookie| {
          Cookie::split_parse(cookie)
            .flat_map(|cookie| cookie.into_iter())
            .find(|cookie| cookie.name() == "kropki")
            .map(|cookie| cookie.into_owned())
        })
      {
        jar.add(cookie);
      }
      self.player_id = jar
        .private(&self.cookie_key)
        .get("kropki")
        .and_then(|cookie| serde_json::from_str(cookie.value()).ok())
        .filter(|data: &CookieData| data.expires_at >= SystemTime::now())
        .map(|data| data.player_id);
      Ok(response)
    })
    .await?;

    let (mut tx_ws, mut rx_ws) = ws_stream.split();

    let (tx, mut rx) = mpsc::channel::<message::Response>(32);

    self.init(&state, tx).await?;

    let future1 = async {
      while let Some(message) = rx.next().await {
        tx_ws.send(Message::Text(serde_json::to_string(&message)?)).await?;
      }

      Ok::<(), Error>(())
    };

    let future2 = async {
      while let Some(message) = rx_ws.next().await {
        if let Message::Text(message) = message? {
          let message: message::Request = serde_json::from_str(message.as_str())?;
          match message {
            message::Request::GetAuthUrl { provider, remember_me } => {
              self.get_auth_url(&state, provider, remember_me).await?
            }
            message::Request::Auth {
              code: oidc_code,
              state: oidc_state,
            } => self.auth(&state, oidc_code, oidc_state).await?,
            #[cfg(feature = "test")]
            message::Request::AuthTest { name } => self.auth_test(&state, name).await?,
            message::Request::Create { size } => self.create(&state, size).await?,
            message::Request::Close { game_id } => self.close(&state, game_id).await?,
            message::Request::Join { game_id } => self.join(&state, game_id).await?,
            message::Request::Subscribe { game_id } => self.subscribe(&state, game_id).await?,
            message::Request::Unsubscribe { game_id } => self.unsubscribe(&state, game_id)?,
            message::Request::PutPoint { game_id, coordinate } => self.put_point(&state, game_id, coordinate).await?,
          }
        }
      }

      Ok::<(), Error>(())
    };

    let result = select! {
      r = future1.fuse() => r,
      r = future2.fuse() => r,
    };

    self.finalize(&state);

    result
  }
}

#[tokio::main]
async fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let config = config::cli_parse();

  let listener = TcpListener::bind("127.0.0.1:8080").await?;
  let state = Arc::new(State::default());

  let mut rng = StdRng::from_entropy();

  let pool = PgPoolOptions::new().connect(&config.postgres_url).await?;
  sqlx::migrate!("./migrations").run(&pool).await?;
  let db = Arc::new(db::SqlxDb::from(pool));

  let http_client = reqwest::ClientBuilder::new()
    .redirect(reqwest::redirect::Policy::none()) // Following redirects opens the client up to SSRF vulnerabilities.
    .build()?;

  let provider_metadata =
    CoreProviderMetadata::discover_async(IssuerUrl::new("https://accounts.google.com".to_string())?, &http_client)
      .await?;
  let google_client = CoreClient::from_provider_metadata(
    provider_metadata,
    config.google_oidc.client_id,
    Some(config.google_oidc.client_secret),
  )
  .set_redirect_uri(RedirectUrl::new("https://kropki.org".to_string())?);

  let provider_metadata =
    CoreProviderMetadata::discover_async(IssuerUrl::new("https://gitlab.com".to_string())?, &http_client).await?;
  let gitlab_client = CoreClient::from_provider_metadata(
    provider_metadata,
    config.gitlab_oidc.client_id,
    Some(config.gitlab_oidc.client_secret),
  )
  .set_redirect_uri(RedirectUrl::new("https://kropki.org".to_string())?);

  let http_client = Arc::new(http_client);
  let google_client = Arc::new(google_client);
  let gitlab_client = Arc::new(gitlab_client);
  let key = Arc::new(Key::generate());

  loop {
    let (stream, addr) = listener.accept().await?;
    let session = Session::new(
      StdRng::from_rng(&mut rng)?,
      db.clone(),
      http_client.clone(),
      google_client.clone(),
      gitlab_client.clone(),
      key.clone(),
    );
    tokio::spawn(session.accept_connection(state.clone(), stream).map(move |result| {
      if let Err(error) = result {
        log::warn!("Closed a connection from {} with an error: {}", addr, error);
      }
    }));
  }
}
