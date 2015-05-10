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
  player: Player,
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
    self.player = Player::Red;
    self.moves_count = 0;
    self.hash = 0;
  }

  fn init(&mut self, field: &Field, player: Player) {
    self.node = Some(Box::new(UctNode::new()));
    self.player = Player::Red;
    let width = field.width();
    for &start_pos in field.points_seq() {
      wave(width, start_pos, |pos| {
        if self.moves_field[pos] != start_pos && field.is_putting_allowed(pos) && manhattan(width, start_pos, pos) <= UCT_RADIUS {
          if self.moves_field[pos] == 0 {
            self.moves.push(pos);
          }
          self.moves_field[pos] = start_pos;
          true
        } else {
          false
        }
      });
    }
    self.player = player;
  }

  fn update(&mut self, field: &Field, player: Player) {
    if !self.node.is_none() && field.hash_at(self.moves_count) != Some(self.hash) {
      self.clear();
    }
    if self.node.is_none() {
      self.init(field, player);
    } else {
      let points_seq = field.points_seq();
      let moves_count = field.moves_count();
      loop {
        if self.moves_count == moves_count {
          break;
        }
        let next_pos = points_seq[self.moves_count];
        if !field.is_players_point(next_pos, self.player) {
          self.clear();
          self.init(field, player);
          break;
        }
        let mut next = self.node.as_ref().unwrap().child.take(Ordering::Relaxed);
        while next.is_some() && next.as_ref().unwrap().pos != next_pos {
          next = next.unwrap().sibling;
        }
        match next.as_mut() {
          Some(node) => {
            node.sibling = None;
          },
          None => { }
        }
        self.node = next;
        let moves_field = &mut self.moves_field;
        self.moves.retain(|&pos| {
          if field.is_putting_allowed(pos) {
            true
          } else {
            moves_field[pos] = 0;
            false
          }
        });
        
      }
    }
  }

  pub fn new(length: Pos) -> UctRoot {
    UctRoot {
      node: None,
      moves: Vec::new(),
      moves_field: repeat(0).take(length).collect(),
      player: Player::Red,
      moves_count: 0,
      hash: 0
    }
  }

  pub fn best_move(&mut self, field: &Field, player: Player) -> Pos {
    self.update(field, player);
    0
  }

  //pub fn estimates
}
