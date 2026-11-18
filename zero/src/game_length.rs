//! How long self-play games are, for the constants that are fractions of a
//! typical game.

/// The part of the board a typical self-play game fills by its end, in moves
/// per board cell. Measured on self-play games: about 100 moves on boards
/// with a mean side of 20.
pub const TYPICAL_FILL: f64 = 0.25;

/// The length in moves of a typical self-play game on a board of `area`
/// cells.
pub fn typical_game_length(area: f64) -> f64 {
  TYPICAL_FILL * area
}
