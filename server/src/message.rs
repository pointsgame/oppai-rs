use serde::{Deserialize, Serialize};

use crate::ids::*;

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct FieldSize {
  pub width: u32,
  pub height: u32,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Coordinate {
  pub x: u32,
  pub y: u32,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct OpenGame {
  pub game_id: GameId,
  pub size: FieldSize,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Game {
  pub game_id: GameId,
  pub size: FieldSize,
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum Request {
  /// Create a new game in a lobby.
  Create { size: FieldSize },
  /// Join a game from lobby.
  Join { game_id: GameId },
  /// Subscribe to game moves.
  Subscribe { game_id: GameId },
  /// Subscribe from game moves.
  Unsubscribe { game_id: GameId },
  /// Put a point in a game.
  PutPoint { game_id: GameId, coordinate: Coordinate },
}

#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(tag = "command")]
pub enum Response {
  /// First message when connection is established.
  Init {
    open_games: Vec<OpenGame>,
    games: Vec<Game>,
  },
  /// First message after subscription.
  GameInit {},
  /// A new game was created in a lobby.
  Create { game_id: GameId, size: FieldSize },
  /// A new game started.
  Start { game_id: GameId },
  /// A point in a game was put.
  PutPoint { game_id: GameId, coordinate: Coordinate },
}
