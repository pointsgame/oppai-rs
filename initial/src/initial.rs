use crate::opening::{cross, quadruple_cross, quadruple_cross_fits, triple_cross};
use oppai_field::field::{Pos, to_pos};
use oppai_field::player::Player;
use rand::Rng;
use strum::{EnumString, VariantNames};

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum InitialPosition {
  Empty,
  Cross,
  TripleCross,
  QuadrupleCross,
}

impl InitialPosition {
  /// Whether the position can be placed on a field of the given size.
  pub fn fits(self, width: u32, height: u32) -> bool {
    self != InitialPosition::QuadrupleCross || quadruple_cross_fits(width, height)
  }

  /// Picks random moves of the position as `(x, y)` coordinates. Moves
  /// alternate between the players starting with the first one, and there is
  /// an even number of them, so the first player moves next.
  ///
  /// Must only be called when [`InitialPosition::fits`] holds.
  pub fn moves<R: Rng>(self, width: u32, height: u32, rng: &mut R) -> Vec<(u32, u32)> {
    match self {
      InitialPosition::Empty => Vec::new(),
      InitialPosition::Cross => cross(width, height, rng),
      InitialPosition::TripleCross => triple_cross(width, height, rng),
      InitialPosition::QuadrupleCross => quadruple_cross(width, height, rng),
    }
  }

  /// The same as [`InitialPosition::moves`], with the moves converted to
  /// positions on the field and paired with the players making them, starting
  /// with `player`.
  pub fn points<R: Rng>(self, width: u32, height: u32, player: Player, rng: &mut R) -> Vec<(Pos, Player)> {
    let mut player = player;
    self
      .moves(width, height, rng)
      .into_iter()
      .map(|(x, y)| {
        let point = (to_pos(width + 1, x, y), player);
        player = player.next();
        point
      })
      .collect()
  }
}
