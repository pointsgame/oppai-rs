use crate::db::Db;
use anyhow::{Error, Result};
use cookie::time::{Duration as CookieDuration, OffsetDateTime};
use cookie::{Cookie, CookieJar, Expiration, Key, SameSite};
use futures::channel::mpsc::{self, Sender};
use futures_util::{FutureExt, SinkExt, StreamExt, select};
use ids::*;
use itertools::Itertools;
use openidconnect::{
  AccessTokenHash, AuthorizationCode, CsrfToken, EndpointMaybeSet, EndpointNotSet, EndpointSet, IssuerUrl, Nonce,
  OAuth2TokenResponse, PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, Scope, TokenResponse,
  core::{CoreAuthenticationFlow, CoreClient, CoreProviderMetadata},
};
use oppai_field::{field::Field, player::Player};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
#[cfg(not(feature = "in-memory"))]
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use state::{FieldSize, Game, GameConfig, GameState, GameTime, OpenGame, State};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::SystemTime;
use time::PrimitiveDateTime;
use tokio::sync::Mutex;
use tokio::{
  net::{TcpListener, TcpStream},
  sync::RwLock,
};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::handshake::server::Request;
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

#[derive(Debug)]
struct AuthState {
  oidc_client: OidcClient,
  pkce_verifier: PkceCodeVerifier,
  nonce: Nonce,
  csrf_state: CsrfToken,
  remember_me: bool,
}

struct SessionShared {
  #[cfg(not(feature = "in-memory"))]
  db: db::SqlxDb,
  #[cfg(feature = "in-memory")]
  db: db::InMemoryDb,
  http_client: reqwest::Client,
  cookie_key: Key,
  oidc: config::OidcConfig,
}

struct Session<R: Rng> {
  shared: Arc<SessionShared>,
  rng: R,
  connection_id: ConnectionId,
  player_id: Option<PlayerId>,
  watching: HashSet<GameId>,
  auth_state: Option<AuthState>,
}

impl<R: Rng> Session<R> {
  fn new(shared: Arc<SessionShared>, mut rng: R) -> Self {
    let connection_id = ConnectionId(Builder::from_random_bytes(rng.random()).into_uuid());
    Session {
      shared,
      rng,
      connection_id,
      player_id: None,
      watching: HashSet::new(),
      auth_state: None,
    }
  }

  fn player_id(&self) -> Result<PlayerId> {
    self
      .player_id
      .ok_or_else(|| anyhow::anyhow!("unauthorized connection {}", self.connection_id))
  }

  async fn oidc_client(&self) -> Result<OidcClient> {
    let redirect_url = RedirectUrl::new("https://kropki.org/".to_string())?;
    let provider_metadata = CoreProviderMetadata::discover_async(
      IssuerUrl::new(self.shared.oidc.issuer_url.to_string())?,
      &self.shared.http_client,
    )
    .await?;
    let client = CoreClient::from_provider_metadata(
      provider_metadata,
      self.shared.oidc.client_id.clone(),
      self.shared.oidc.client_secret.clone(),
    )
    .set_redirect_uri(redirect_url);
    Ok(client)
  }

  async fn get_auth_url(&mut self, state: &State, remember_me: bool) -> Result<()> {
    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
    let oidc_client = self.oidc_client().await?;
    let (auth_url, csrf_state, nonce) = oidc_client
      .authorize_url(
        CoreAuthenticationFlow::AuthorizationCode,
        CsrfToken::new_random,
        Nonce::new_random,
      )
      .add_scope(Scope::new("email".to_string()))
      .add_scope(Scope::new("profile".to_string()))
      .set_pkce_challenge(pkce_challenge)
      .url();

    state
      .send_to_connection(
        self.connection_id,
        message::Response::AuthUrl {
          url: auth_url.to_string(),
        },
      )
      .await?;

    self.auth_state = Some(AuthState {
      oidc_client,
      pkce_verifier,
      nonce,
      csrf_state,
      remember_me,
    });

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

    let token_response = auth_state
      .oidc_client
      .exchange_code(AuthorizationCode::new(oidc_code))?
      .set_pkce_verifier(auth_state.pkce_verifier)
      .request_async(&self.shared.http_client)
      .await?;

    let id_token = token_response.id_token().ok_or_else(|| {
      anyhow::anyhow!(
        "server did not return an ID token for connection {}",
        self.connection_id
      )
    })?;
    let id_token_verifier = auth_state.oidc_client.id_token_verifier();
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

    let player = self
      .shared
      .db
      .get_or_create_player(
        db::OidcPlayer {
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
        },
        &mut self.rng,
      )
      .await?;
    let player_id = PlayerId(player.id);

    self.player_id = Some(player_id);
    state.insert_players_connection(player_id, self.connection_id);

    state
      .send_to_all(message::Response::PlayerJoined {
        player_id,
        player: message::Player {
          nickname: player.nickname,
        },
      })
      .await;

    let duration = if auth_state.remember_me {
      CookieDuration::weeks(12)
    } else {
      CookieDuration::weeks(1)
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
    .path("/")
    .expires(if auth_state.remember_me {
      Expiration::DateTime(OffsetDateTime::now_utc() + duration)
    } else {
      Expiration::Session
    })
    .same_site(SameSite::Strict)
    .secure(true)
    .build();
    jar.private_mut(&self.shared.cookie_key).add(cookie);

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
    let player = self.shared.db.get_or_create_test_player(name).await?;
    let player_id = PlayerId(player.id);

    self.player_id = Some(player_id);
    state.insert_players_connection(player_id, self.connection_id);

    state
      .send_to_all(message::Response::PlayerJoined {
        player_id,
        player: message::Player {
          nickname: player.nickname,
        },
      })
      .await;

    let mut jar = CookieJar::new();
    let cookie = Cookie::build((
      "kropki",
      serde_json::to_string(&CookieData {
        player_id,
        expires_at: SystemTime::now() + CookieDuration::weeks(1),
      })
      .unwrap(),
    ))
    .path("/")
    .expires(Expiration::Session)
    .same_site(SameSite::Strict)
    .secure(true)
    .build();
    jar.private_mut(&self.shared.cookie_key).add(cookie);

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
    let player = if let Some(player_id) = self.player_id {
      Some(self.shared.db.get_player(player_id.0).await?)
    } else {
      None
    };

    // lock connection before inserting so we can be sure we send init message before any update
    let connection = Arc::new(Mutex::new(tx));
    let connection_c = connection.clone();
    let mut connection_c_lock = connection_c.lock().await;

    state.connections.pin().insert(self.connection_id, connection);

    if let Some(player) = player {
      let player_id = PlayerId(player.id);
      state.insert_players_connection(player_id, self.connection_id);
      state
        .send_to_all_except(
          self.connection_id,
          message::Response::PlayerJoined {
            player_id,
            player: message::Player {
              nickname: player.nickname,
            },
          },
        )
        .await;
    }

    let player_ids = state
      .players
      .pin()
      .keys()
      .chain(state.open_games.pin().values().map(|open_game| &open_game.player_id))
      .chain(
        state
          .games
          .pin()
          .values()
          .flat_map(|game| [&game.black_player_id, &game.red_player_id].into_iter()),
      )
      .map(|player_id| player_id.0)
      .unique()
      .collect::<Vec<_>>();
    let mut players = self
      .shared
      .db
      .get_players(&player_ids)
      .await?
      .into_iter()
      .map(|player| {
        (
          player.id,
          message::Player {
            nickname: player.nickname,
          },
        )
      })
      .collect::<HashMap<_, _>>();

    let open_games = state
      .open_games
      .pin()
      .iter()
      .flat_map(|(&game_id, open_game)| {
        players
          .get(&open_game.player_id.0)
          .map(|player| {
            (
              game_id,
              message::OpenGame {
                player_id: open_game.player_id,
                player: message::Player {
                  nickname: player.nickname.clone(),
                },
                config: message::GameConfig {
                  size: message::FieldSize {
                    width: open_game.config.size.width,
                    height: open_game.config.size.height,
                  },
                  time: message::GameTime {
                    total: open_game.config.time.total,
                    increment: open_game.config.time.increment,
                  },
                },
              },
            )
          })
          .into_iter()
      })
      .collect();
    let games = state
      .games
      .pin()
      .iter()
      .flat_map(|(&game_id, game)| {
        players
          .get(&game.red_player_id.0)
          .zip(players.get(&game.black_player_id.0))
          .map(|(red_player, black_player)| {
            (
              game_id,
              message::Game {
                red_player_id: game.red_player_id,
                black_player_id: game.black_player_id,
                red_player: message::Player {
                  nickname: red_player.nickname.clone(),
                },
                black_player: message::Player {
                  nickname: black_player.nickname.clone(),
                },
                config: message::GameConfig {
                  size: message::FieldSize {
                    width: game.config.size.width,
                    height: game.config.size.height,
                  },
                  time: message::GameTime {
                    total: game.config.time.total,
                    increment: game.config.time.increment,
                  },
                },
              },
            )
          })
          .into_iter()
      })
      .collect();
    let players = state
      .players
      .pin()
      .keys()
      .flat_map(|&player_id| {
        players
          .remove(&player_id.0)
          .map(|player| (player_id, player))
          .into_iter()
      })
      .collect();

    let init = message::Response::Init {
      player_id: self.player_id,
      players,
      open_games,
      games,
    };
    connection_c_lock.send(init).await?;

    Ok(())
  }

  async fn finalize(&self, state: &State) {
    state.connections.pin().remove(&self.connection_id);

    for &game_id in &self.watching {
      state.unsubscribe(self.connection_id, game_id);
    }

    if let Some(player_id) = self.player_id
      && state.remove_players_connection(player_id, self.connection_id)
    {
      state.send_to_all(message::Response::PlayerLeft { player_id }).await;
    }
  }

  async fn sign_out(&mut self, state: &State) {
    if let Some(player_id) = self.player_id {
      self.player_id = None;
      if state.remove_players_connection(player_id, self.connection_id) {
        state.send_to_all(message::Response::PlayerLeft { player_id }).await;
      }
    }
  }

  async fn create(&mut self, state: &State, config: message::GameConfig) -> Result<()> {
    if !config.is_valid() {
      anyhow::bail!(
        "invalid game config {:?} from connection {}",
        config,
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

    let game_id = GameId(Builder::from_random_bytes(self.rng.random()).into_uuid());
    let open_game = OpenGame {
      player_id,
      config: GameConfig {
        size: FieldSize {
          width: config.size.width,
          height: config.size.height,
        },
        time: GameTime {
          total: config.time.total,
          increment: config.time.increment,
        },
      },
    };

    state.open_games.pin().insert(game_id, open_game);

    let player = self.shared.db.get_player(player_id.0).await?;

    state
      .send_to_all(message::Response::Create {
        game_id,
        open_game: message::OpenGame {
          player_id,
          player: message::Player {
            nickname: player.nickname,
          },
          config,
        },
      })
      .await;

    Ok(())
  }

  async fn close(&mut self, state: &State, game_id: GameId) -> Result<()> {
    let player_id = self.player_id()?;

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
    let player_id = self.player_id()?;

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

    let now = SystemTime::now();
    let now_offset = OffsetDateTime::from(now);
    let now_primitive = PrimitiveDateTime::new(now_offset.date(), now_offset.time());

    self
      .shared
      .db
      .create_game(db::Game {
        id: game_id.0,
        red_player_id: open_game.player_id.0,
        black_player_id: player_id.0,
        start_time: now_primitive,
      })
      .await?;

    let field = Field::new_from_rng(open_game.config.size.width, open_game.config.size.height, &mut self.rng);
    let game_state = GameState {
      field,
      red_time: open_game.config.time.total,
      black_time: open_game.config.time.total,
      last_move_time: now,
      draw_offer: None,
    };
    let game = Game {
      red_player_id: open_game.player_id,
      black_player_id: player_id,
      config: open_game.config.clone(),
      state: Arc::new(RwLock::new(game_state)),
    };

    state.games.pin().insert(game_id, game);

    let [player_1, player_2] = self
      .shared
      .db
      .get_players(&[open_game.player_id.0, player_id.0])
      .await?
      .try_into()
      .map_err(|_| anyhow::anyhow!("can't find players {} and {}", open_game.player_id.0, player_id.0))?;
    let [red_player, black_player] = if player_1.id == open_game.player_id.0 {
      [player_1, player_2]
    } else {
      [player_2, player_1]
    };

    state
      .send_to_all(message::Response::Start {
        game_id,
        game: message::Game {
          red_player_id: PlayerId(red_player.id),
          black_player_id: PlayerId(black_player.id),
          red_player: message::Player {
            nickname: red_player.nickname,
          },
          black_player: message::Player {
            nickname: black_player.nickname,
          },
          config: message::GameConfig {
            size: message::FieldSize {
              width: open_game.config.size.width,
              height: open_game.config.size.height,
            },
            time: message::GameTime {
              total: open_game.config.time.total,
              increment: open_game.config.time.increment,
            },
          },
        },
      })
      .await;

    Ok(())
  }

  async fn subscribe(&mut self, state: &State, game_id: GameId) -> Result<()> {
    if self.watching.len() > 2 {
      anyhow::bail!("too many subscriptions from a connection {}", self.connection_id);
    }
    if !self.watching.insert(game_id) {
      anyhow::bail!(
        "connection {} already watching the game {}",
        self.connection_id,
        game_id
      );
    }

    state.subscribe(self.connection_id, game_id);

    let (game_state, red_player_id, black_player_id, config) = if let Some(game) = state.games.pin().get(&game_id) {
      (
        game.state.clone(),
        game.red_player_id,
        game.black_player_id,
        game.config.clone(),
      )
    } else {
      // TODO: log
      return Ok(());
    };
    let game_state = game_state.read().await;

    let moves = game_state
      .field
      .colored_moves()
      .map(|(pos, player)| message::Move {
        coordinate: message::Coordinate {
          x: game_state.field.to_x(pos),
          y: game_state.field.to_y(pos),
        },
        player,
      })
      .collect();

    let draw_offer = game_state.draw_offer;

    let now = SystemTime::now();
    let now_epoch = now.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let elapsed = now.duration_since(game_state.last_move_time).unwrap_or_default();

    let time_left = match game_state
      .field
      .last_player()
      .map_or(Player::Red, |player| player.next())
    {
      Player::Red => message::TimeLeft {
        red: game_state.red_time.saturating_sub(elapsed),
        black: game_state.black_time,
      },
      Player::Black => message::TimeLeft {
        red: game_state.red_time,
        black: game_state.black_time.saturating_sub(elapsed),
      },
    };

    drop(game_state);

    let [player_1, player_2] = self
      .shared
      .db
      .get_players(&[red_player_id.0, black_player_id.0])
      .await?
      .try_into()
      .map_err(|_| anyhow::anyhow!("can't find players {} and {}", red_player_id.0, black_player_id.0))?;
    let [red_player, black_player] = if player_1.id == red_player_id.0 {
      [player_1, player_2]
    } else {
      [player_2, player_1]
    };

    state
      .send_to_connection(
        self.connection_id,
        message::Response::GameInit {
          game_id,
          game: message::Game {
            red_player_id: PlayerId(red_player.id),
            black_player_id: PlayerId(black_player.id),
            red_player: message::Player {
              nickname: red_player.nickname,
            },
            black_player: message::Player {
              nickname: black_player.nickname,
            },
            config: message::GameConfig {
              size: message::FieldSize {
                width: config.size.width,
                height: config.size.height,
              },
              time: message::GameTime {
                total: config.time.total,
                increment: config.time.increment,
              },
            },
          },
          moves,
          init_time: now_epoch,
          time_left,
          draw_offer,
          result: None,
        },
      )
      .await
  }

  fn unsubscribe(&mut self, state: &State, game_id: GameId) -> Result<()> {
    if !self.watching.remove(&game_id) {
      anyhow::bail!("connection {} not watching the game {}", self.connection_id, game_id);
    }

    state.unsubscribe(self.connection_id, game_id);

    Ok(())
  }

  async fn put_point(&self, state: &State, game_id: GameId, coordinate: message::Coordinate) -> Result<()> {
    let player_id = self.player_id()?;

    let (game_state, player, increment) = if let Some(game) = state.games.pin().get(&game_id) {
      let player = if let Some(player) = game.color(player_id) {
        player
      } else {
        anyhow::bail!(
          "player {} attempted to put point in a wrong game {}",
          player_id,
          game_id,
        );
      };
      (game.state.clone(), player, game.config.time.increment)
    } else {
      log::warn!(
        "player {} attempted to put point in a game {} that don't exist",
        player_id,
        game_id,
      );

      return Ok(());
    };

    let mut game_state = game_state.write().await;
    let pos = game_state.field.to_pos(coordinate.x, coordinate.y);

    if game_state
      .field
      .last_player()
      .map_or(Player::Red, |player| player.next())
      != player
    {
      anyhow::bail!(
        "player {} attempted to put point on opponent's turn in a game {}",
        player_id,
        game_id,
      );
    }

    if !game_state.field.put_point(pos, player) {
      anyhow::bail!(
        "player {} attempted tp put point on a wrong position {:?} in game {}",
        player_id,
        (coordinate.x, coordinate.y),
        game_id,
      );
    }

    let now = SystemTime::now();
    let now_offset = OffsetDateTime::from(now);
    let now_primitive = PrimitiveDateTime::new(now_offset.date(), now_offset.time());
    let now_epoch = now.duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();

    let elapsed = now.duration_since(game_state.last_move_time).unwrap_or_default();
    match player {
      Player::Red => game_state.red_time = game_state.red_time.saturating_sub(elapsed) + increment,
      Player::Black => game_state.black_time = game_state.black_time.saturating_sub(elapsed) + increment,
    }

    game_state.last_move_time = now;

    self
      .shared
      .db
      .create_move(db::Move {
        game_id: game_id.0,
        player: player.into(),
        number: (game_state.field.moves_count() - 1) as i16,
        x: coordinate.x as i16,
        y: coordinate.y as i16,
        timestamp: now_primitive,
      })
      .await
      .inspect_err(|_| {
        game_state.field.undo();
      })?;

    let time_left = message::TimeLeft {
      red: game_state.red_time,
      black: game_state.black_time,
    };

    drop(game_state);

    state
      .send_to_watchers(
        game_id,
        message::Response::PutPoint {
          game_id,
          _move: message::Move { coordinate, player },
          putting_time: now_epoch,
          time_left,
        },
      )
      .await;

    Ok(())
  }

  async fn resign(&self, state: &State, game_id: GameId) -> Result<()> {
    let player_id = self.player_id()?;

    let player = {
      let pin = state.games.pin();

      let player = if let Some(game) = pin.get(&game_id) {
        if player_id == game.red_player_id {
          Player::Red
        } else if player_id == game.black_player_id {
          Player::Black
        } else {
          anyhow::bail!("player {} attempted to resign in a wrong game {}", player_id, game_id,)
        }
      } else {
        log::warn!(
          "player {} attempted to resign in a game {} that don't exist",
          player_id,
          game_id,
        );

        return Ok(());
      };

      if pin.remove(&game_id).is_none() {
        log::warn!("Game {} is already finished", game_id);
        return Ok(());
      };

      player
    };

    let now = SystemTime::now();
    let now_offset = OffsetDateTime::from(now);
    let now_primitive = PrimitiveDateTime::new(now_offset.date(), now_offset.time());

    self
      .shared
      .db
      .set_result(
        game_id.0,
        now_primitive,
        match player {
          Player::Red => db::GameResult::ResignedBlack,
          Player::Black => db::GameResult::ResignedRed,
        },
      )
      .await?;

    state
      .send_to_watchers(
        game_id,
        message::Response::GameResult {
          game_id,
          result: message::GameResult::Win {
            winner: player.next(),
            reason: message::WinReason::Resigned,
          },
        },
      )
      .await;

    Ok(())
  }

  async fn draw(&self, state: &State, game_id: GameId) -> Result<()> {
    let player_id = self.player_id()?;

    let (game_state, player) = if let Some(game) = state.games.pin().get(&game_id) {
      let player = if let Some(player) = game.color(player_id) {
        player
      } else {
        anyhow::bail!("player {} attempted to draw in a wrong game {}", player_id, game_id,);
      };
      (game.state.clone(), player)
    } else {
      log::warn!(
        "player {} attempted to draw in a game {} that don't exist",
        player_id,
        game_id,
      );

      return Ok(());
    };

    let mut game_state = game_state.write().await;

    match game_state.draw_offer {
      None => {
        game_state.draw_offer = Some(player);
        drop(game_state);

        let now = SystemTime::now();
        let now_offset = OffsetDateTime::from(now);
        let now_primitive = PrimitiveDateTime::new(now_offset.date(), now_offset.time());

        self
          .shared
          .db
          .create_draw_offer(db::DrawOffer {
            game_id: game_id.0,
            player: player.into(),
            offer: true,
            timestamp: now_primitive,
          })
          .await?;
        state
          .send_to_watchers(game_id, message::Response::Draw { game_id, player })
          .await;
      }
      Some(draw_offer) => {
        if draw_offer == player.next() {
          if state.games.pin().remove(&game_id).is_none() {
            log::warn!("Game {} is already finished", game_id);
            return Ok(());
          };

          let now = SystemTime::now();
          let now_offset = OffsetDateTime::from(now);
          let now_primitive = PrimitiveDateTime::new(now_offset.date(), now_offset.time());

          self
            .shared
            .db
            .set_result(game_id.0, now_primitive, db::GameResult::DrawAgreement)
            .await?;

          state
            .send_to_watchers(
              game_id,
              message::Response::GameResult {
                game_id,
                result: message::GameResult::Draw {
                  reason: message::DrawReason::Agreement,
                },
              },
            )
            .await;
        }
      }
    }

    Ok(())
  }

  fn is_nickname_valid(nickname: &str) -> bool {
    if nickname.len() < 3 || nickname.len() > 32 {
      return false;
    }

    if !nickname.chars().all(|c| c.is_alphanumeric() || c == '_') {
      return false;
    }

    true
  }

  async fn change_nickname(&mut self, state: &State, nickname: String) -> Result<()> {
    if !Self::is_nickname_valid(&nickname) {
      anyhow::bail!("Invalid nickname format");
    }

    let player_id = self.player_id()?;

    self
      .shared
      .db
      .update_player_nickname(player_id.0, nickname.clone())
      .await?;

    let player = self.shared.db.get_player(player_id.0).await?;

    state
      .send_to_all(message::Response::NicknameChanged {
        player_id,
        player: message::Player {
          nickname: player.nickname,
        },
      })
      .await;

    Ok(())
  }

  async fn check_nickname(&self, state: &State, nickname: String) -> Result<()> {
    let available =
      Self::is_nickname_valid(&nickname) && self.shared.db.is_nickname_available(nickname.clone()).await?;

    state
      .send_to_connection(
        self.connection_id,
        message::Response::NicknameAvailable { nickname, available },
      )
      .await?;

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
        .private(&self.shared.cookie_key)
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
        tx_ws
          .send(Message::Text(serde_json::to_string(&message)?.into()))
          .await?;
      }

      Ok::<(), Error>(())
    };

    let future2 = async {
      while let Some(message) = rx_ws.next().await {
        if let Message::Text(message) = message? {
          let message: message::Request = serde_json::from_str(message.as_str())?;
          match message {
            message::Request::GetAuthUrl { remember_me } => self.get_auth_url(&state, remember_me).await?,
            message::Request::Auth {
              code: oidc_code,
              state: oidc_state,
            } => self.auth(&state, oidc_code, oidc_state).await?,
            #[cfg(feature = "test")]
            message::Request::AuthTest { name } => self.auth_test(&state, name).await?,
            message::Request::SignOut => self.sign_out(&state).await,
            message::Request::Create { config } => self.create(&state, config).await?,
            message::Request::Close { game_id } => self.close(&state, game_id).await?,
            message::Request::Join { game_id } => self.join(&state, game_id).await?,
            message::Request::Subscribe { game_id } => self.subscribe(&state, game_id).await?,
            message::Request::Unsubscribe { game_id } => self.unsubscribe(&state, game_id)?,
            message::Request::PutPoint { game_id, coordinate } => self.put_point(&state, game_id, coordinate).await?,
            message::Request::Resign { game_id } => self.resign(&state, game_id).await?,
            message::Request::Draw { game_id } => self.draw(&state, game_id).await?,
            message::Request::ChangeNickname { nickname } => self.change_nickname(&state, nickname).await?,
            message::Request::CheckNickname { nickname } => self.check_nickname(&state, nickname).await?,
          }
        }
      }

      Ok::<(), Error>(())
    };

    let result = select! {
      r = future1.fuse() => r,
      r = future2.fuse() => r,
    };

    self.finalize(&state).await;

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

  let mut rng = StdRng::from_os_rng();

  #[cfg(not(feature = "in-memory"))]
  let options = PgConnectOptions::new_without_pgpass().socket(&config.postgres_socket);
  #[cfg(not(feature = "in-memory"))]
  let pool = PgPoolOptions::new().connect_with(options).await?;
  #[cfg(not(feature = "in-memory"))]
  sqlx::migrate!("./migrations").run(&pool).await?;

  let http_client = reqwest::ClientBuilder::new()
    .redirect(reqwest::redirect::Policy::none()) // Following redirects opens the client up to SSRF vulnerabilities.
    .build()?;

  let session_shared = Arc::new(SessionShared {
    #[cfg(not(feature = "in-memory"))]
    db: db::SqlxDb::from(pool),
    #[cfg(feature = "in-memory")]
    db: db::InMemoryDb::default(),
    http_client,
    cookie_key: config.cookie_key,
    oidc: config.oidc,
  });

  loop {
    let (stream, addr) = listener.accept().await?;
    let session = Session::new(session_shared.clone(), StdRng::from_rng(&mut rng));
    tokio::spawn(session.accept_connection(state.clone(), stream).map(move |result| {
      if let Err(error) = result {
        log::warn!("Closed a connection from {} with an error: {}", addr, error);
      }
    }));
  }
}
