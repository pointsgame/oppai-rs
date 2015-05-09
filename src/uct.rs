use std::iter::*;
use std::sync::atomic::*;
use atomic_option::*;
use types::*;
use config::*;
use player::*;
use field::*;

struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicOption<UctNode>,
  sibling: Option<Box<UctNode>>
}

impl UctNode {
  pub fn new() -> UctNode {
    UctNode {
      wins: AtomicUsize::new(0),
      draws: AtomicUsize::new(0),
      visits: AtomicUsize::new(0),
      pos: 0,
      child: AtomicOption::empty(),
      sibling: None
    }
  }
}

pub struct UctRoot {
  node: Option<Box<UctNode>>,
  moves: Vec<Pos>,
  moves_field: Vec<Pos>,
  player: Option<Player>,
  moves_count: usize,
  hash: u64
}

impl UctRoot {
  fn clear(&mut self) {
    self.node = None;
    self.moves.clear();
    for i in self.moves_field.iter_mut() {
      *i = 0;
    }
    self.player = None;
    self.moves_count = 0;
    self.hash = 0;
  }

  fn init(&mut self, field: &Field) {
    self.node = Some(Box::new(UctNode::new()));
    self.player = None;
    let width = field.width();
    for &start_pos in field.points_seq() {
      wave(width, start_pos, |pos| {
        if self.moves_field[pos] != start_pos && field.is_putting_allowed(pos) && manhattan(width, start_pos, pos) <= UCT_RADIUS {
          self.moves_field[pos] = start_pos;
          true
        } else {
          false
        }
      });
    }
  }

  fn update(&mut self, field: &Field, player: Player) {
    if !self.node.is_none() && field.hash_at(self.moves_count) != Some(self.hash) {
      self.clear();
    }
    loop {
      
    }
  }

  pub fn new(length: Pos) -> UctRoot {
    UctRoot {
      node: None,
      moves: Vec::new(),
      moves_field: repeat(0).take(length).collect(),
      player: None,
      moves_count: 0,
      hash: 0
    }
  }

  pub fn best_move(&mut self) -> Pos {
    0
  }

  //pub fn estimates
}
