//! Tactical features of a position: what the dots of either player can
//! capture within a move or two.
//!
//! Whether a dot captures depends on a closed loop around the enemy's dots,
//! which can run across much of the board. A network sees that only as far as
//! its layers reach, and only approximately, so the cells where a capture is
//! possible are given to it directly.
//!
//! The meaning of every plane is fixed once a network has been trained on it.
//! A new tactical feature becomes a plane of its own rather than a change to
//! what an existing plane marks.

use oppai_common::trajectory::{Trajectory, build_trajectories};
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;

/// The number of input planes [`Tactics`] provides, see [`Tactics::plane`].
pub const TACTICAL_CHANNELS: usize = 4;

/// The longest trajectory the planes mark.
const MAX_LENGTH: u32 = 2;

/// For every cell of a field, the tactical planes it is marked in, as bits:
///
/// 0. the player to move captures with a dot here;
/// 1. the opponent captures with a dot here;
/// 2. the cell is one of two where the player to move captures with a dot in
///    each;
/// 3. the same for the opponent.
///
/// The sets of cells are the trajectories of length 1 and 2 of
/// [`build_trajectories`]: the cells where a player's dots capture, every one
/// of them needed, so that a pair is never one capturing cell next to one that
/// does not matter. A pair whose second dot captures alone too, and more with
/// the first, is still listed.
#[derive(Clone, PartialEq, Debug)]
pub struct Tactics {
  planes: Vec<u8>,
}

impl Tactics {
  /// The tactical planes of `field` with `player` to move. The trajectories
  /// are built by playing and undoing moves, so `field` is left as it was.
  pub fn new(field: &mut Field, player: Player) -> Self {
    let mut planes = vec![0; field.length()];
    let mut empty_board = vec![0; field.length()];
    for (i, player) in [player, player.next()].into_iter().enumerate() {
      let trajectories: Vec<Trajectory<2>> = build_trajectories(field, player, MAX_LENGTH, &mut empty_board, &|| false);
      for trajectory in &trajectories {
        let plane = (trajectory.points.len() - 1) * 2 + i;
        for &pos in &trajectory.points {
          planes[pos] |= 1 << plane;
        }
      }
    }
    Tactics { planes }
  }

  /// Whether `pos` is marked in the plane with the index `plane`, which is
  /// less than [`TACTICAL_CHANNELS`].
  #[inline]
  pub fn plane(&self, plane: usize, pos: Pos) -> bool {
    self.planes[pos] & (1 << plane) != 0
  }
}
