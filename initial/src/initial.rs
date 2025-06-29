use oppai_field::field::{Pos, to_pos};
use oppai_field::player::Player;
use strum::{EnumString, VariantNames};

fn cross(width: u32, height: u32, player: Player) -> [(Pos, Player); 4] {
  let w2 = width / 2;
  let h2 = height / 2;
  [
    (to_pos(width + 1, w2 - 1, h2 - 1), player),
    (to_pos(width + 1, w2 - 1, h2), player.next()),
    (to_pos(width + 1, w2, h2), player),
    (to_pos(width + 1, w2, h2 - 1), player.next()),
  ]
}

fn two_crosses(width: u32, height: u32, player: Player) -> [(Pos, Player); 8] {
  let w2 = width / 2;
  let h2 = height / 2;
  [
    (to_pos(width + 1, w2 - 2, h2 - 1), player),
    (to_pos(width + 1, w2 - 2, h2), player.next()),
    (to_pos(width + 1, w2 - 1, h2), player),
    (to_pos(width + 1, w2 - 1, h2 - 1), player.next()),
    (to_pos(width + 1, w2, h2), player),
    (to_pos(width + 1, w2, h2 - 1), player.next()),
    (to_pos(width + 1, w2 + 1, h2 - 1), player),
    (to_pos(width + 1, w2 + 1, h2), player.next()),
  ]
}

fn triple_cross(width: u32, height: u32, player: Player) -> [(Pos, Player); 8] {
  let w2 = width / 2;
  let h2 = height / 2;
  [
    (to_pos(width + 1, w2 - 1, h2 - 1), player),
    (to_pos(width + 1, w2 - 1, h2), player.next()),
    (to_pos(width + 1, w2, h2), player),
    (to_pos(width + 1, w2, h2 - 1), player.next()),
    (to_pos(width + 1, w2 + 1, h2 - 1), player),
    (to_pos(width + 1, w2, h2 - 2), player.next()),
    (to_pos(width + 1, w2, h2 + 1), player),
    (to_pos(width + 1, w2 + 1, h2), player.next()),
  ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum InitialPosition {
  Empty,
  Cross,
  TwoCrosses,
  TripleCross,
}

impl InitialPosition {
  pub fn points(self, width: u32, height: u32, player: Player) -> impl Iterator<Item = (Pos, Player)> + Clone {
    let cross = if self == InitialPosition::Cross {
      Some(cross(width, height, player))
    } else {
      None
    };
    let two_crosses = if self == InitialPosition::TwoCrosses {
      Some(two_crosses(width, height, player))
    } else {
      None
    };
    let triple_cross = if self == InitialPosition::TripleCross {
      Some(triple_cross(width, height, player))
    } else {
      None
    };
    cross
      .into_iter()
      .flatten()
      .chain(two_crosses.into_iter().flatten())
      .chain(triple_cross.into_iter().flatten())
  }
}
