use std::*;
use std::iter::*;
use std::sync::atomic::*;
use types::*;
use config::*;
use player::*;
use field::*;

#[unsafe_no_drop_flag]
struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicPtr<UctNode>,
  sibling: Option<Box<UctNode>>
}

unsafe impl Send for UctNode { }

impl Drop for UctNode {
  fn drop(&mut self) {
    self.clear_child();
  }
}

impl UctNode {
  pub fn new(pos: Pos) -> UctNode {
    UctNode {
      wins: AtomicUsize::new(0),
      draws: AtomicUsize::new(0),
      visits: AtomicUsize::new(0),
      pos: pos,
      child: AtomicPtr::new(ptr::null_mut()),
      sibling: None
    }
  }

  pub fn get_pos(&self) -> Pos {
    self.pos
  }

  pub fn set_pos(&mut self, pos: Pos) {
    self.pos = pos;
  }

  pub fn get_sibling(&mut self) -> Option<Box<UctNode>> {
    mem::replace(&mut self.sibling, None)
  }

  pub fn get_sibling_ref<'a>(&'a self) -> Option<&'a UctNode> {
    self.sibling.as_ref().map(|b| &**b)
  }

  pub fn get_sibling_mut<'a>(&'a mut self) -> Option<&'a mut UctNode> {
    self.sibling.as_mut().map(|b| &mut **b)
  }

  pub fn clear_sibling(&mut self) {
    self.sibling = None;
  }

  pub fn set_sibling(&mut self, sibling: Box<UctNode>) {
    self.sibling = Some(sibling);
  }

  pub fn get_child(&self) -> Option<Box<UctNode>> {
    let ptr = self.child.swap(ptr::null_mut(), Ordering::Relaxed);
    if !ptr.is_null() {
      Some(unsafe { mem::transmute(ptr) })
    } else {
      None
    }
  }

  pub fn get_child_ref<'a>(&'a self) -> Option<&'a UctNode> {
    let ptr = self.child.load(Ordering::Relaxed);
    if !ptr.is_null() {
      Some(unsafe { &*ptr })
    } else {
      None
    }
  }

  pub fn get_child_mut<'a>(&'a mut self) -> Option<&'a mut UctNode> {
    let ptr = self.child.load(Ordering::Relaxed);
    if !ptr.is_null() {
      Some(unsafe { &mut *ptr })
    } else {
      None
    }
  }

  pub fn clear_child(&self) {
    let ptr = self.child.swap(ptr::null_mut(), Ordering::Relaxed);
    if !ptr.is_null() {
      drop::<Box<UctNode>>(unsafe { mem::transmute(ptr) });
    }
  }

  pub fn set_child(&self, child: Box<UctNode>) {
    let ptr = self.child.swap(unsafe { mem::transmute(child) }, Ordering::Relaxed);
    if !ptr.is_null() {
      drop::<Box<UctNode>>(unsafe { mem::transmute(ptr) });
    }
  }

  pub fn set_child_if_empty(&self, child: Box<UctNode>) {
    let child_ptr = unsafe { mem::transmute(child) };
    let ptr = self.child.compare_and_swap(ptr::null_mut(), child_ptr, Ordering::Relaxed);
    if !ptr.is_null() {
      drop::<Box<UctNode>>(unsafe { mem::transmute(child_ptr) });
    }
  }

  pub fn get_visits(&self) -> usize {
    self.visits.load(Ordering::Relaxed)
  }

  pub fn get_wins(&self) -> usize {
    self.wins.load(Ordering::Relaxed)
  }

  pub fn get_draws(&self) -> usize {
    self.draws.load(Ordering::Relaxed)
  }

  pub fn add_win(&self) {
    self.visits.fetch_add(1, Ordering::Relaxed);
    self.wins.fetch_add(1, Ordering::Relaxed);
  }

  pub fn add_draw(&self) {
    self.visits.fetch_add(1, Ordering::Relaxed);
    self.draws.fetch_add(1, Ordering::Relaxed);
  }

  pub fn add_loose(&self) {
    self.visits.fetch_add(1, Ordering::Relaxed);
  }

  pub fn clear_stats(&self) {
    self.wins.store(0, Ordering::Relaxed);
    self.draws.store(0, Ordering::Relaxed);
    self.visits.store(0, Ordering::Relaxed);
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
    self.node = Some(Box::new(UctNode::new(0)));
    self.player = player;
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
  }

  fn expand_node(node: &mut UctNode, moves: &Vec<Pos>) {
    if node.get_child_ref().is_none() {
      if node.get_visits() == usize::max_value() {
        node.clear_stats();
      }
    } else {
      let mut next = node.get_child_mut();
      while next.as_ref().unwrap().get_sibling_ref().is_some() {
        UctRoot::expand_node(*next.as_mut().unwrap(), moves);
        next = next.unwrap().get_sibling_mut();
      }
      UctRoot::expand_node(*next.as_mut().unwrap(), moves);
      for &pos in moves {
        next.as_mut().unwrap().set_sibling(Box::new(UctNode::new(pos)));
        next = next.unwrap().get_sibling_mut();
      }
    }
  }

  fn update(&mut self, field: &Field, player: Player) {
    if self.node.is_some() && field.hash_at(self.moves_count) != Some(self.hash) {
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
        let mut next = self.node.as_ref().unwrap().get_child();
        while next.is_some() && next.as_ref().unwrap().pos != next_pos {
          next = next.unwrap().get_sibling();
        }
        match next.as_mut() {
          Some(node) => {
            node.clear_sibling();
          },
          None => {
            self.clear();
            self.init(field, player);
            break;
          }
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
        let mut added_moves = Vec::new();
        let width = field.width();
        wave(width, next_pos, |pos| {
          if moves_field[pos] != next_pos && field.is_putting_allowed(pos) && manhattan(width, next_pos, pos) <= UCT_RADIUS {
            if moves_field[pos] == 0 {
              added_moves.push(pos);
            }
            moves_field[pos] = next_pos;
            true
          } else {
            false
          }
        });
        UctRoot::expand_node(self.node.as_mut().unwrap(), &added_moves);
        self.moves_count += 1;
        self.player = self.player.next();
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
