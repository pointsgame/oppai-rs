use std::*;
use std::iter::*;
use std::sync::atomic::*;
use rand::{Rng, XorShiftRng};
use types::*;
use config::*;
use player::*;
use field::*;

static UCT_STR: &'static str = "uct";

#[unsafe_no_drop_flag]
struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicPtr<UctNode>,
  sibling: Option<Box<UctNode>>
}

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

  pub fn get_sibling_ref(&self) -> Option<&UctNode> {
    self.sibling.as_ref().map(|b| &**b)
  }

  pub fn get_sibling_mut(&mut self) -> Option<&mut UctNode> {
    self.sibling.as_mut().map(|b| &mut **b)
  }

  pub fn clear_sibling(&mut self) {
    self.sibling = None;
  }

  pub fn set_sibling(&mut self, sibling: Box<UctNode>) {
    self.sibling = Some(sibling);
  }

  pub fn set_sibling_option(&mut self, sibling: Option<Box<UctNode>>) {
    self.sibling = sibling;
  }

  pub fn get_child(&self) -> Option<Box<UctNode>> {
    let ptr = self.child.swap(ptr::null_mut(), Ordering::Relaxed);
    if !ptr.is_null() {
      Some(unsafe { mem::transmute(ptr) })
    } else {
      None
    }
  }

  pub fn get_child_ref(&self) -> Option<&UctNode> {
    let ptr = self.child.load(Ordering::Relaxed);
    if !ptr.is_null() {
      Some(unsafe { &*ptr })
    } else {
      None
    }
  }

  pub fn get_child_mut(&mut self) -> Option<&mut UctNode> {
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

  pub fn loose_node(&self) {
    self.wins.store(0, Ordering::Relaxed);
    self.draws.store(0, Ordering::Relaxed);
    self.visits.store(usize::max_value(), Ordering::Relaxed);
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
    info!(target: UCT_STR, "Initialization.");
    self.node = Some(Box::new(UctNode::new(0)));
    self.player = player;
    self.moves_count = field.moves_count();
    self.hash = field.hash();
    let width = field.width();
    for &start_pos in field.points_seq() {
      wave(width, start_pos, |pos| {
        if pos == start_pos && self.moves_field[pos] == 0 {
          self.moves_field[pos] = 1;
          true
        } else if self.moves_field[pos] != start_pos && field.is_putting_allowed(pos) && manhattan(width, start_pos, pos) <= UCT_RADIUS {
          if self.moves_field[pos] == 0 {
            self.moves.push(pos);
          }
          self.moves_field[pos] = start_pos;
          true
        } else {
          false
        }
      });
      self.moves_field[start_pos] = 0;
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
      info!(target: UCT_STR, "Updation.");
      let points_seq = field.points_seq();
      let moves_count = field.moves_count();
      loop {
        if self.moves_count == moves_count {
          break;
        }
        let next_pos = points_seq[self.moves_count];
        info!(target: UCT_STR, "Next move is ({0}, {1}), player {2}.", field.to_x(next_pos), field.to_y(next_pos), self.player);
        if !field.is_players_point(next_pos, self.player) {
          self.clear();
          self.init(field, player);
          break;
        }
        let mut next = self.node.as_ref().unwrap().get_child();
        while next.is_some() && next.as_ref().unwrap().pos != next_pos {
          next = next.unwrap().get_sibling();
        }
        if let Some(ref mut node) = next {
          let pos = node.get_pos();
          info!(target: UCT_STR, "Node found for move ({0}, {1}).", field.to_x(pos), field.to_y(pos));
          node.clear_sibling();
        } else {
          self.clear();
          self.init(field, player);
          break;
        }
        self.node = next;
        let moves_field = &mut self.moves_field;
        let moves = &mut self.moves;
        moves.retain(|&pos| {
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
          if pos == next_pos && moves_field[pos] == 0 {
            moves_field[pos] = 1;
            true
          } else if moves_field[pos] != next_pos && field.is_putting_allowed(pos) && manhattan(width, next_pos, pos) <= UCT_RADIUS {
            if moves_field[pos] == 0 && pos != next_pos {
              moves.push(pos);
              added_moves.push(pos);
            }
            moves_field[pos] = next_pos;
            true
          } else {
            false
          }
        });
        moves_field[next_pos] = 0;
        UctRoot::expand_node(self.node.as_mut().unwrap(), &added_moves);
        self.moves_count += 1;
        self.player = self.player.next();
        self.hash = field.hash();
      }
    }
  }

  pub fn new(length: Pos) -> UctRoot {
    UctRoot {
      node: None,
      moves: Vec::with_capacity(length),
      moves_field: repeat(0).take(length).collect(),
      player: Player::Red,
      moves_count: 0,
      hash: 0
    }
  }

  fn play_random_game<T: Rng>(field: &mut Field, mut player: Player, rng: &mut T, possible_moves: &mut Vec<Pos>) -> Option<Player> {
    rng.shuffle(possible_moves);
    let mut putted: CoordProd = 0;
    for &pos in possible_moves.iter() {
      if field.is_putting_allowed(pos) && !field.is_empty_base(pos) {
        field.put_point(pos, player);
        player = player.next();
        putted += 1;
      }
    }
    let red_score = field.score(Player::Red);
    let result = if red_score > 0 {
      Some(Player::Red)
    } else if red_score < 0 {
      Some(Player::Black)
    } else {
      None
    };
    for _ in 0 .. putted {
      field.undo();
    }
    result
  }

  fn ucb(parent: &UctNode, node: &UctNode) -> f32 {
    let wins = node.get_wins() as f32;
    let draws = node.get_draws() as f32;
    let visits = node.get_visits() as f32;
    let parent_visits = parent.get_visits() as f32;
    let win_rate = (wins + draws * UCT_DRAW_WEIGHT) / visits;
    let uct = match UCB_TYPE {
      UcbType::Ucb1 => UCTK * f32::sqrt(2.0 * f32::ln(parent_visits) / visits),
      UcbType::Ucb1Tuned => {
        let v = (wins + draws * UCT_DRAW_WEIGHT * UCT_DRAW_WEIGHT) / visits - win_rate * win_rate + f32::sqrt(2.0 * f32::ln(parent_visits) / visits);
        UCTK * f32::sqrt(v.min(0.25) * f32::ln(parent_visits) / visits)
      }
    };
    win_rate + uct
  }

  fn create_children(field: &Field, possible_moves: &Vec<Pos>, node: &UctNode) {
    let mut children = None;
    for &pos in possible_moves {
      if field.is_putting_allowed(pos) {
        let mut cur_child = Box::new(UctNode::new(pos));
        cur_child.set_sibling_option(children);
        children = Some(cur_child);
      }
    }
    if let Some(child) = children {
      node.set_child_if_empty(child)
    }
  }

  fn uct_select(node: &UctNode) -> Option<&UctNode> {
    let mut best_uct = 0f32;
    let mut result = None;
    let mut next = node.get_child_ref();
    while let Some(next_node) = next {
      let visits = next_node.get_visits();
      let wins = next_node.get_wins();
      if visits == usize::max_value() {
        if wins == usize::max_value() {
          return Some(next_node);
        } else {
          next = next_node.get_sibling_ref();
          continue;
        }
      } else if visits == 0 {
        return Some(next_node);
      }
      let uct_value = UctRoot::ucb(node, next_node);
      if uct_value > best_uct {
        best_uct = uct_value;
        result = Some(next_node);
      }
      next = next_node.get_sibling_ref();
    }
    result
  }

  fn play_simulation_rec<T: Rng>(field: &mut Field, player: Player, node: &UctNode, possible_moves: &mut Vec<Pos>, rng: &mut T, depth: Depth) -> Option<Player> {
    let mut random_result;
    if node.get_visits() < UCT_WHEN_CREATE_CHILDREN || depth == UCT_DEPTH {
      random_result = UctRoot::play_random_game(field, player, rng, possible_moves);
    } else {
      if node.get_child_ref().is_none() {
        UctRoot::create_children(field, possible_moves, node);
      }
      if let Some(next) = UctRoot::uct_select(node) {
        field.put_point(next.get_pos(), player);
        if field.get_delta_score(player) < 0 {
          field.undo();
          next.loose_node();
          return UctRoot::play_simulation_rec(field, player, node, possible_moves, rng, depth);
        }
        random_result = UctRoot::play_simulation_rec(field, player.next(), next, possible_moves, rng, depth + 1);
        field.undo();
      } else {
        let red_score = field.score(Player::Red);
        random_result = if red_score > 0 {
          Some(Player::Red)
        } else if red_score < 0 {
          Some(Player::Black)
        } else {
          None
        };
      }
    }
    if let Some(player_random_result) = random_result {
      if player_random_result == player {
        node.add_loose();
      } else {
        node.add_win();
      }
    } else {
      node.add_draw();
    }
    random_result
  }

  fn play_simulation<T: Rng>(field: &mut Field, player: Player, node: &UctNode, possible_moves: &mut Vec<Pos>, rng: &mut T) {
    UctRoot::play_simulation_rec(field, player, node, possible_moves, rng, 0);
  }

  fn best_move_generic<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T, should_stop: &AtomicBool) -> Option<Pos> {
    self.update(field, player);
    let mut guards = Vec::with_capacity(4);
    for _ in 0 .. 4 {
      let xor_shift_rng = rng.gen::<XorShiftRng>();
      guards.push(thread::scoped(|| {
        let mut local_field = field.clone();
        let mut local_rng = xor_shift_rng;
        let mut possible_moves = self.moves.clone();
        while !should_stop.load(Ordering::Relaxed) {
          UctRoot::play_simulation(&mut local_field, player, self.node.as_ref().unwrap(), &mut possible_moves, &mut local_rng);
        }
      }));
    }
    drop(guards);
    let mut best_uct = 0f32;
    let mut result = None;
    if let Some(ref root) = self.node {
      let mut next = root.get_child_ref();
      while let Some(next_node) = next {
        let uct_value = UctRoot::ucb(root, next_node);
        let pos = next_node.get_pos();
        info!(target: UCT_STR, "Uct for move ({0}, {1}) is {2}, {3} wins, {4} draws, {5} visits.", field.to_x(pos), field.to_y(pos), uct_value, next_node.get_wins(), next_node.get_draws(), next_node.get_visits());
        if uct_value > best_uct {
          best_uct = uct_value;
          result = Some(pos);
        }
        next = next_node.get_sibling_ref();
      }
    }
    if let Some(pos) = result {
      info!(target: UCT_STR, "Best move is ({0}, {1}), uct is {2}.", field.to_x(pos), field.to_y(pos), best_uct);
    }
    result
  }

  pub fn best_move<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T, time: Time) -> Option<Pos> {
    let should_stop = AtomicBool::new(false);
    let guard = thread::scoped(|| {
      thread::sleep_ms(time);
      should_stop.store(true, Ordering::Relaxed);
    });
    let result = self.best_move_generic(field, player, rng, &should_stop);
    drop(guard);
    result
  }

  //pub fn estimates
}
