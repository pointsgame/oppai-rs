use std::{ptr, thread, mem};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, AtomicPtr, Ordering};
use rand::{Rng, XorShiftRng};
use types::{Pos, Coord, Time, Depth, Score};
use config;
use config::{UcbType, UctKomiType};
use player::Player;
use field::Field;
use wave_pruning::WavePruning;

const UCT_STR: &'static str = "uct";

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
  player: Player,
  moves_count: usize,
  hash: u64,
  wave_pruning: WavePruning,
  komi: AtomicIsize,
  komi_visits: AtomicUsize,
  komi_wins: AtomicUsize,
  komi_draws: AtomicUsize
}

impl UctRoot {
  fn clear(&mut self) {
    self.node = None;
    self.wave_pruning.clear();
    self.player = Player::Red;
    self.moves_count = 0;
    self.hash = 0;
    self.komi = AtomicIsize::new(0);
    self.komi_visits = AtomicUsize::new(0);
    self.komi_wins = AtomicUsize::new(0);
    self.komi_draws = AtomicUsize::new(0);
  }

  fn init(&mut self, field: &Field, player: Player) {
    debug!(target: UCT_STR, "Initialization.");
    self.node = Some(Box::new(UctNode::new(0)));
    self.player = player;
    self.moves_count = field.moves_count();
    self.hash = field.hash();
    if config::uct_komi_type() != UctKomiType::None {
      self.komi = AtomicIsize::new(field.score(player) as isize);
    }
    self.wave_pruning.init(field, config::uct_radius());
  }

  fn expand_node<T: Rng>(node: &mut UctNode, moves: &mut Vec<Pos>, rng: &mut T) {
    if node.get_child_ref().is_none() {
      if node.get_visits() == usize::max_value() {
        node.clear_stats();
      }
    } else {
      let mut next = node.get_child_mut();
      while next.as_ref().unwrap().get_sibling_ref().is_some() {
        UctRoot::expand_node(*next.as_mut().unwrap(), moves, rng);
        next = next.unwrap().get_sibling_mut();
      }
      UctRoot::expand_node(*next.as_mut().unwrap(), moves, rng);
      rng.shuffle(moves);
      for &pos in moves.iter() {
        next.as_mut().unwrap().set_sibling(Box::new(UctNode::new(pos)));
        next = next.unwrap().get_sibling_mut();
      }
    }
  }

  fn update<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T) {
    if self.node.is_some() && field.hash_at(self.moves_count) != Some(self.hash) {
      self.clear();
    }
    if self.node.is_none() {
      self.init(field, player);
    } else {
      debug!(target: UCT_STR, "Updation.");
      let points_seq = field.points_seq();
      let moves_count = field.moves_count();
      let last_moves_count = self.moves_count;
      loop {
        if self.moves_count == moves_count {
          if self.player != player {
            self.clear();
            self.init(field, player);
          } else if let Some(node) = self.node.as_mut() {
            let mut added_moves = self.wave_pruning.update(field, last_moves_count, config::uct_radius());
            debug!(target: UCT_STR, "Added  into consideration moves: {:?}.", added_moves.iter().map(|&pos| (field.to_x(pos), field.to_y(pos))).collect::<Vec<(Coord, Coord)>>());
            UctRoot::expand_node(node, &mut added_moves, rng);
            match config::uct_komi_type() {
              UctKomiType::Static => self.komi = AtomicIsize::new(field.score(self.player) as isize),
              UctKomiType::Dynamic => {
                self.komi_visits = AtomicUsize::new(node.get_visits());
                self.komi_wins = AtomicUsize::new(node.get_wins());
                self.komi_draws = AtomicUsize::new(node.get_draws());
              },
              UctKomiType::None => { }
            }
          }
          break;
        }
        let next_pos = points_seq[self.moves_count];
        debug!(target: UCT_STR, "Next move is ({0}, {1}), player {2}.", field.to_x(next_pos), field.to_y(next_pos), self.player);
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
          debug!(target: UCT_STR, "Node found for move ({0}, {1}).", field.to_x(pos), field.to_y(pos));
          node.clear_sibling();
        } else {
          self.clear();
          self.init(field, player);
          break;
        }
        self.node = next;
        self.moves_count += 1;
        self.player = self.player.next();
        self.hash = field.hash();
        if config::uct_komi_type() == UctKomiType::Dynamic {
          self.komi = AtomicIsize::new(-self.komi.load(Ordering::Relaxed));
        }
      }
    }
  }

  pub fn new(length: Pos) -> UctRoot {
    UctRoot {
      node: None,
      player: Player::Red,
      moves_count: 0,
      hash: 0,
      wave_pruning: WavePruning::new(length),
      komi: AtomicIsize::new(0),
      komi_visits: AtomicUsize::new(0),
      komi_wins: AtomicUsize::new(0),
      komi_draws: AtomicUsize::new(0)
    }
  }

  fn random_result(field: &Field, player: Player, komi: Score) -> Option<Player> {
    let red_komi = if player == Player::Red { komi } else { -komi };
    let red_score = field.score(Player::Red);
    if red_score > red_komi {
      Some(Player::Red)
    } else if red_score < red_komi {
      Some(Player::Black)
    } else {
      None
    }
  }

  fn play_random_game<T: Rng>(field: &mut Field, player: Player, rng: &mut T, possible_moves: &mut Vec<Pos>, komi: Score) -> Option<Player> {
    rng.shuffle(possible_moves);
    let mut cur_player = player;
    for &pos in possible_moves.iter() {
      if field.is_putting_allowed(pos) && !field.is_empty_base(pos) {
        field.put_point(pos, cur_player);
        cur_player = cur_player.next();
      }
    }
    UctRoot::random_result(field, player, komi)
  }

  fn ucb(parent: &UctNode, node: &UctNode, ucb_type: UcbType) -> f32 {
    let wins = node.get_wins() as f32;
    let draws = node.get_draws() as f32;
    let visits = node.get_visits() as f32;
    let parent_visits = parent.get_visits() as f32;
    let uct_draw_weight = config::uct_draw_weight();
    let uctk = config::uctk();
    let win_rate = (wins + draws * uct_draw_weight) / visits;
    let uct = match ucb_type {
      UcbType::Winrate => 0f32,
      UcbType::Ucb1 => uctk * f32::sqrt(2.0 * f32::ln(parent_visits) / visits),
      UcbType::Ucb1Tuned => {
        let v = (wins + draws * uct_draw_weight * uct_draw_weight) / visits - win_rate * win_rate + f32::sqrt(2.0 * f32::ln(parent_visits) / visits);
        uctk * f32::sqrt(v.min(0.25) * f32::ln(parent_visits) / visits)
      }
    };
    win_rate + uct
  }

  fn create_children<T: Rng>(field: &Field, possible_moves: &mut Vec<Pos>, node: &UctNode, rng: &mut T) {
    rng.shuffle(possible_moves);
    let mut children = None;
    for &pos in possible_moves.iter() {
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
        }
      } else if visits == 0 {
        return Some(next_node);
      } else {
        let uct_value = UctRoot::ucb(node, next_node, config::ucb_type());
        if uct_value > best_uct {
          best_uct = uct_value;
          result = Some(next_node);
        }
      }
      next = next_node.get_sibling_ref();
    }
    result
  }

  fn is_last_move_stupid(field: &Field, pos: Pos, player: Player) -> bool {
    let delta_score = field.get_delta_score(player);
    delta_score < 0 || delta_score == 0 && {
      let enemy = player.next();
      let mut enemies_around = 0u8;
      if field.is_players_point(field.n(pos), enemy) {
        enemies_around += 1;
      }
      if field.is_players_point(field.s(pos), enemy) {
        enemies_around += 1;
      }
      if field.is_players_point(field.w(pos), enemy) {
        enemies_around += 1;
      }
      if field.is_players_point(field.e(pos), enemy) {
        enemies_around += 1;
      }
      enemies_around == 3
    } && {
      field.is_putting_allowed(field.n(pos)) || field.is_putting_allowed(field.s(pos)) || field.is_putting_allowed(field.w(pos)) || field.is_putting_allowed(field.e(pos))
    }
  }

  fn play_simulation_rec<T: Rng>(field: &mut Field, player: Player, node: &UctNode, possible_moves: &mut Vec<Pos>, rng: &mut T, komi: Score, depth: Depth) -> Option<Player> {
    let random_result = if node.get_visits() < config::uct_when_create_children() || depth == config::uct_depth() {
      UctRoot::play_random_game(field, player, rng, possible_moves, komi)
    } else {
      if node.get_child_ref().is_none() {
        UctRoot::create_children(field, possible_moves, node, rng)
      }
      if let Some(next) = UctRoot::uct_select(node) {
        let pos = next.get_pos();
        field.put_point(pos, player);
        if UctRoot::is_last_move_stupid(field, pos, player) {
          field.undo();
          next.loose_node();
          return UctRoot::play_simulation_rec(field, player, node, possible_moves, rng, komi, depth);
        }
        UctRoot::play_simulation_rec(field, player.next(), next, possible_moves, rng, -komi, depth + 1)
      } else {
        UctRoot::random_result(field, player, komi)
      }
    };
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

  fn play_simulation<T: Rng>(&self, field: &mut Field, player: Player, possible_moves: &mut Vec<Pos>, rng: &mut T, ratched: &AtomicIsize) {
    if let Some(node) = self.node.as_ref() {
      UctRoot::play_simulation_rec(field, player, node, possible_moves, rng, self.komi.load(Ordering::Relaxed) as Score, 0);
      if config::uct_komi_type() == UctKomiType::Dynamic {
        let visits = node.get_visits();
        let komi_visits = self.komi_visits.load(Ordering::Relaxed);
        let delta_visits = visits - komi_visits;
        if delta_visits > config::uct_komi_min_iterations() {
          let wins = node.get_wins();
          let delta_wins = wins - self.komi_wins.load(Ordering::Relaxed);
          let draws = node.get_draws();
          let delta_draws = draws - self.komi_draws.load(Ordering::Relaxed);
          let win_rate = 1f32 - (delta_wins as f32 + delta_draws as f32 * config::uct_draw_weight()) / delta_visits as f32;
          let komi = self.komi.load(Ordering::Relaxed);
          let red = config::uct_red();
          if win_rate < red || win_rate > config::uct_green() && komi < ratched.load(Ordering::Relaxed) {
            let old_komi_visits = self.komi_visits.compare_and_swap(komi_visits, visits, Ordering::Relaxed);
            if old_komi_visits == komi_visits {
              self.komi_wins.store(wins, Ordering::Relaxed);
              self.komi_draws.store(draws, Ordering::Relaxed);
              if win_rate < red {
                if komi > 0 {
                  ratched.store(komi - 1, Ordering::Relaxed);
                }
                self.komi.fetch_sub(1, Ordering::Relaxed);
                info!(target: UCT_STR, "Komi decreased after {1} visits: {0}. Winrate is {2}.", komi - 1, visits, win_rate);
              } else {
                self.komi.fetch_add(1, Ordering::Relaxed);
                info!(target: UCT_STR, "Komi increased after {1} visits: {0}. Winrate is {2}.", komi + 1, visits, win_rate);
              }
            }
          }
        }
      }
    }
  }

  fn best_move_generic<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T, should_stop: &AtomicBool, max_iterations_count: usize) -> Option<Pos> {
    info!(target: UCT_STR, "Generating best move for player {0}.", player);
    debug!(target: UCT_STR, "Moves history: {:?}.", field.points_seq().iter().map(|&pos| (field.to_x(pos), field.to_y(pos), field.get_player(pos))).collect::<Vec<(Coord, Coord, Player)>>());
    debug!(target: UCT_STR, "Next random u64: {0}.", rng.gen::<u64>());
    self.update(field, player, rng);
    info!(target: UCT_STR, "Komi is {0}, type is {1}.", self.komi.load(Ordering::Relaxed), config::uct_komi_type());
    let threads_count = config::threads_count();
    let iterations = AtomicUsize::new(0);
    let ratched = AtomicIsize::new(isize::max_value());
    let mut guards = Vec::with_capacity(threads_count);
    for _ in 0 .. threads_count {
      let xor_shift_rng = rng.gen::<XorShiftRng>();
      guards.push(thread::scoped(|| {
        let mut local_field = field.clone();
        let mut local_rng = xor_shift_rng;
        let mut possible_moves = self.wave_pruning.moves().clone();
        while !should_stop.load(Ordering::Relaxed) && iterations.load(Ordering::Relaxed) < max_iterations_count {
          self.play_simulation(&mut local_field, player, &mut possible_moves, &mut local_rng, &ratched);
          for _ in 0 .. local_field.moves_count() - self.moves_count {
            local_field.undo();
          }
          iterations.fetch_add(1, Ordering::Relaxed);
        }
      }));
    }
    drop(guards);
    info!(target: UCT_STR, "Iterations count: {0}.", iterations.load(Ordering::Relaxed));
    let mut best_uct = 0f32;
    let mut result = None;
    if let Some(ref root) = self.node {
      let mut next = root.get_child_ref();
      while let Some(next_node) = next {
        let uct_value = UctRoot::ucb(root, next_node, config::final_ucb_type());
        let pos = next_node.get_pos();
        info!(target: UCT_STR, "Uct for move ({0}, {1}) is {2}, {3} wins, {4} draws, {5} visits.", field.to_x(pos), field.to_y(pos), uct_value, next_node.get_wins(), next_node.get_draws(), next_node.get_visits());
        if uct_value > best_uct || uct_value == best_uct && rng.gen() {
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

  pub fn best_move_with_time<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T, time: Time) -> Option<Pos> {
    let should_stop = AtomicBool::new(false);
    let guard = thread::scoped(|| {
      thread::sleep_ms(time);
      should_stop.store(true, Ordering::Relaxed);
    });
    let result = self.best_move_generic(field, player, rng, &should_stop, usize::max_value());
    drop(guard);
    result
  }

  pub fn best_move_with_iterations_count<T: Rng>(&mut self, field: &Field, player: Player, rng: &mut T, iterations: usize) -> Option<Pos> {
    let should_stop = AtomicBool::new(false);
    self.best_move_generic(field, player, rng, &should_stop, iterations)
  }
}
