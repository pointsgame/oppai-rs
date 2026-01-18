use std::{collections::HashMap, time::Duration};

use oppai_field::player::Player as Color;
use serde::{Deserialize, Serialize};
use serde_with::{DurationMilliSeconds, DurationSeconds, serde_as};

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

#[serde_as]
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameTime {
  #[serde_as(as = "DurationSeconds")]
  pub total: Duration,
  #[serde_as(as = "DurationSeconds")]
  pub increment: Duration,
}

#[serde_as]
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeLeft {
  #[serde_as(as = "DurationMilliSeconds")]
  pub red: Duration,
  #[serde_as(as = "DurationMilliSeconds")]
  pub black: Duration,
}

impl GameTime {
  const MIN_TOTAL: Duration = Duration::from_secs(30);
  const MAX_TOTAL: Duration = Duration::from_secs(4 * 60 * 60);
  const MAX_INCREMENT: Duration = Duration::from_secs(60);

  pub fn is_valid(&self) -> bool {
    self.total >= Self::MIN_TOTAL && self.total <= Self::MAX_TOTAL && self.increment <= Self::MAX_INCREMENT
  }
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameConfig {
  pub size: FieldSize,
  pub time: GameTime,
}

impl GameConfig {
  pub fn is_valid(&self) -> bool {
    self.size.is_valid() && self.time.is_valid()
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
  pub nickname: String,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenGame {
  pub player_id: PlayerId,
  pub player: Player,
  pub config: GameConfig,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Game {
  pub red_player_id: PlayerId,
  pub black_player_id: PlayerId,
  pub red_player: Player,
  pub black_player: Player,
  pub config: GameConfig,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum DrawReason {
  Agreement,
  Grounded,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum WinReason {
  Resigned,
  Grounded,
  TimeOut,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all_fields = "camelCase")]
pub enum GameResult {
  Win { winner: Color, reason: WinReason },
  Draw { reason: DrawReason },
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
#[serde(rename_all_fields = "camelCase")]
pub enum Request {
  GetAuthUrl {
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
    config: GameConfig,
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
  /// Resign a game.
  Resign {
    game_id: GameId,
  },
  /// Offer or accept a draw.
  Draw {
    game_id: GameId,
  },
  /// Change user's own nickname.
  ChangeNickname {
    nickname: String,
  },
  /// Check if a nickname is available.
  CheckNickname {
    nickname: String,
  },
}

#[serde_as]
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
#[serde(rename_all_fields = "camelCase")]
pub enum Response {
  /// First message when connection is established.
  Init {
    player_id: Option<PlayerId>,
    players: HashMap<PlayerId, Player>,
    open_games: HashMap<GameId, OpenGame>,
    games: HashMap<GameId, Game>,
  },
  /// First message after subscription.
  GameInit {
    game_id: GameId,
    game: Game,
    moves: Vec<Move>,
    #[serde_as(as = "DurationMilliSeconds")]
    init_time: Duration,
    time_left: TimeLeft,
    draw_offer: Option<Color>,
    result: Option<GameResult>,
  },
  AuthUrl {
    url: String,
  },
  Auth {
    player_id: PlayerId,
    cookie: String,
  },
  PlayerJoined {
    player_id: PlayerId,
    player: Player,
  },
  PlayerLeft {
    player_id: PlayerId,
  },
  /// A new game was created in a lobby.
  Create {
    game_id: GameId,
    open_game: OpenGame,
  },
  /// An open game was closed.
  Close {
    game_id: GameId,
  },
  /// A new game started.
  Start {
    game_id: GameId,
    game: Game,
  },
  /// A point in a game was put.
  PutPoint {
    game_id: GameId,
    #[serde(rename = "move")]
    _move: Move,
    #[serde_as(as = "DurationMilliSeconds")]
    putting_time: Duration,
    time_left: TimeLeft,
  },
  /// Offer a draw.
  Draw {
    game_id: GameId,
    player: Color,
  },
  GameResult {
    game_id: GameId,
    time_left: TimeLeft,
    result: GameResult,
  },
  /// A player changed their nickname.
  NicknameChanged {
    player_id: PlayerId,
    player: Player,
  },
  /// Response to nickname availability check.
  NicknameAvailable {
    nickname: String,
    available: bool,
  },
}
