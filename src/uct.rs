use std::iter::*;
use std::sync::atomic::*;
use atomic_option::*;
use types::*;
use player::*;

struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicOption<UctNode>,
  sibling: Option<Box<UctNode>>
}

pub struct UctRoot {
  node: Option<Box<UctNode>>,
  moves: Vec<Pos>,
  moves_field: Vec<bool>,
  player: Player,
  points_seq: Vec<Pos>
}

impl UctRoot {
  pub fn new(length: Pos) -> UctRoot {
    UctRoot {
      node: None,
      moves: Vec::new(),
      moves_field: repeat(false).take(length).collect(),
      player: Player::Red,
      points_seq: Vec::new()
    }
  }

  //pub fn best_move
  //pub fn estimates
}
