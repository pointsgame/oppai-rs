use oppai_field::player::Player as Color;
use serde::{Deserialize, Serialize};

use crate::ids::*;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct FieldSize {
  pub width: u32,
  pub height: u32,
}

impl FieldSize {
  const MIN_SIZE: u32 = 10;
  const MAX_SIZE: u32 = 50;

  pub fn is_valid(&self) -> bool {
    self.width >= Self::MIN_SIZE
      && self.width <= Self::MAX_SIZE
      && self.height >= Self::MIN_SIZE
      && self.height <= Self::MAX_SIZE
  }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Coordinate {
  pub x: u32,
  pub y: u32,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Move {
  pub coordinate: Coordinate,
  pub player: Color,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Player {
  pub player_id: PlayerId,
  pub nickname: String,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenGame {
  pub game_id: GameId,
  pub player: Player,
  pub size: FieldSize,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Game {
  pub game_id: GameId,
  pub red_player: Player,
  pub black_player: Player,
  pub size: FieldSize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum AuthProvider {
  Portier,
  Google,
  GitLab,
  #[cfg(feature = "test")]
  Test,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
#[serde(rename_all_fields = "camelCase")]
pub enum Request {
  GetAuthUrl {
    provider: AuthProvider,
    remember_me: bool,
  },
  Auth {
    code: String,
    state: String,
  },
  #[cfg(feature = "test")]
  AuthTest {
    name: String,
  },
  SignOut,
  /// Create a new game in a lobby.
  Create {
    size: FieldSize,
  },
  /// Close an open game.
  Close {
    game_id: GameId,
  },
  /// Join a game from lobby.
  Join {
    game_id: GameId,
  },
  /// Subscribe to game moves.
  Subscribe {
    game_id: GameId,
  },
  /// Subscribe from game moves.
  Unsubscribe {
    game_id: GameId,
  },
  /// Put a point in a game.
  PutPoint {
    game_id: GameId,
    coordinate: Coordinate,
  },
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
#[serde(rename_all_fields = "camelCase")]
pub enum Response {
  /// First message when connection is established.
  Init {
    auth_providers: Vec<AuthProvider>,
    player_id: Option<PlayerId>,
    players: Vec<Player>,
    open_games: Vec<OpenGame>,
    games: Vec<Game>,
  },
  /// First message after subscription.
  GameInit {
    game_id: GameId,
    moves: Vec<Move>,
  },
  AuthUrl {
    url: String,
  },
  Auth {
    player_id: PlayerId,
    cookie: String,
  },
  PlayerJoined {
    player: Player,
  },
  PlayerLeft {
    player_id: PlayerId,
  },
  /// A new game was created in a lobby.
  Create {
    open_game: OpenGame,
  },
  /// An open game was closed.
  Close {
    game_id: GameId,
  },
  /// A new game started.
  Start {
    game: Game,
  },
  /// A point in a game was put.
  PutPoint {
    game_id: GameId,
    #[serde(rename = "move")]
    _move: Move,
  },
}
