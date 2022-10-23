use crate::wave_pruning::WavePruning;
use oppai_common::common;
use oppai_field::field::{Field, NonZeroPos, Pos};
use oppai_field::player::Player;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::{
  mem, ptr,
  sync::atomic::{AtomicBool, AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
};
use strum::{EnumString, EnumVariantNames};

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, EnumVariantNames)]
pub enum UcbType {
  Winrate,
  Ucb1,
  Ucb1Tuned,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, EnumVariantNames)]
pub enum UctKomiType {
  None,
  Static,
  Dynamic,
}

#[derive(Clone, PartialEq, Debug)]
pub struct UctConfig {
  pub threads_count: usize,
  pub radius: u32,
  pub ucb_type: UcbType,
  pub draw_weight: f64,
  pub uctk: f64,
  pub when_create_children: usize,
  pub depth: u32,
  pub komi_type: UctKomiType,
  pub red: f64,
  pub green: f64,
  pub komi_min_iterations: usize,
  pub fpu: f64,
}

impl Default for UctConfig {
  fn default() -> Self {
    Self {
      threads_count: num_cpus::get(),
      radius: 3,
      ucb_type: UcbType::Ucb1Tuned,
      draw_weight: 0.4,
      uctk: 1.0,
      when_create_children: 2,
      depth: 8,
      komi_type: UctKomiType::Dynamic,
      red: 0.45,
      green: 0.5,
      komi_min_iterations: 3000,
      fpu: 1.1,
    }
  }
}

struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicPtr<UctNode>,
  sibling: Option<Box<UctNode>>,
}

impl Drop for UctNode {
  fn drop(&mut self) {
    unsafe {
      self.clear_child();
    }
  }
}

impl UctNode {
  pub fn new(pos: Pos) -> UctNode {
    UctNode {
      wins: AtomicUsize::new(0),
      draws: AtomicUsize::new(0),
      visits: AtomicUsize::new(0),
      pos,
      child: AtomicPtr::new(ptr::null_mut()),
      sibling: None,
    }
  }

  pub fn get_pos(&self) -> Pos {
    self.pos
  }

  pub fn get_sibling(&mut self) -> Option<Box<UctNode>> {
    mem::replace(&mut self.sibling, None)
  }

  pub fn get_sibling_ref(&self) -> Option<&UctNode> {
    self.sibling.as_deref()
  }

  pub fn get_sibling_mut(&mut self) -> Option<&mut UctNode> {
    self.sibling.as_deref_mut()
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
    if ptr.is_null() {
      None
    } else {
      Some(unsafe { Box::from_raw(ptr) })
    }
  }

  pub fn get_child_ref(&self) -> Option<&UctNode> {
    let ptr = self.child.load(Ordering::Relaxed);
    if ptr.is_null() {
      None
    } else {
      Some(unsafe { &*ptr })
    }
  }

  pub fn get_child_mut(&mut self) -> Option<&mut UctNode> {
    let ptr = self.child.load(Ordering::Relaxed);
    if ptr.is_null() {
      None
    } else {
      Some(unsafe { &mut *ptr })
    }
  }

  unsafe fn clear_child(&self) {
    let ptr = self.child.swap(ptr::null_mut(), Ordering::Relaxed);
    if !ptr.is_null() {
      drop::<Box<UctNode>>(Box::from_raw(ptr));
    }
  }

  pub fn set_child(&self, child: Box<UctNode>) {
    let child_ptr = Box::into_raw(child);
    let result = self
      .child
      .compare_exchange(ptr::null_mut(), child_ptr, Ordering::Relaxed, Ordering::Relaxed);
    if result.is_err() {
      drop::<Box<UctNode>>(unsafe { Box::from_raw(child_ptr) });
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

  pub fn lose_node(&self) {
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
  config: UctConfig,
  node: Option<Box<UctNode>>,
  player: Player,
  moves_count: usize,
  hash: u64,
  wave_pruning: WavePruning,
  komi: AtomicIsize,
  komi_visits: AtomicUsize,
  komi_wins: AtomicUsize,
  komi_draws: AtomicUsize,
}

impl UctRoot {
  pub fn clear(&mut self) {
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
    debug!("Initialization.");
    self.node = Some(Box::new(UctNode::new(0)));
    self.player = player;
    self.moves_count = field.moves_count();
    self.hash = field.hash();
    if self.config.komi_type != UctKomiType::None {
      self.komi = AtomicIsize::new(field.score(player) as isize);
    }
    self.wave_pruning.init(field, self.config.radius);
  }

  fn expand_node<R: Rng>(node: &mut UctNode, moves: &mut Vec<Pos>, rng: &mut R) {
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
      moves.shuffle(rng);
      for &pos in moves.iter() {
        next.as_mut().unwrap().set_sibling(Box::new(UctNode::new(pos)));
        next = next.unwrap().get_sibling_mut();
      }
    }
  }

  fn update<R: Rng>(&mut self, field: &Field, player: Player, rng: &mut R) {
    if self.node.is_some() && field.hash_at(self.moves_count) != Some(self.hash) {
      self.clear();
    }
    if self.node.is_none() {
      self.init(field, player);
    } else {
      debug!("Updation.");
      let points_seq = field.points_seq();
      let moves_count = field.moves_count();
      let last_moves_count = self.moves_count;
      loop {
        if self.moves_count == moves_count {
          if self.player != player {
            self.clear();
            self.init(field, player);
          } else if let Some(node) = self.node.as_mut() {
            let mut added_moves = self.wave_pruning.update(field, last_moves_count, self.config.radius);
            debug!(
              "Added into consideration moves: {:?}.",
              added_moves
                .iter()
                .map(|&pos| (field.to_x(pos), field.to_y(pos)))
                .collect::<Vec<(u32, u32)>>()
            );
            UctRoot::expand_node(node, &mut added_moves, rng);
            match self.config.komi_type {
              UctKomiType::Static => self.komi = AtomicIsize::new(field.score(self.player) as isize),
              UctKomiType::Dynamic => {
                self.komi_visits = AtomicUsize::new(node.get_visits());
                self.komi_wins = AtomicUsize::new(node.get_wins());
                self.komi_draws = AtomicUsize::new(node.get_draws());
              }
              UctKomiType::None => {}
            }
          }
          break;
        }
        let next_pos = points_seq[self.moves_count];
        debug!(
          "Next move is ({}, {}), player {}.",
          field.to_x(next_pos),
          field.to_y(next_pos),
          self.player
        );
        if !field.cell(next_pos).is_players_point(self.player) {
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
          debug!("Node found for move ({}, {}).", field.to_x(pos), field.to_y(pos));
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
        if self.config.komi_type == UctKomiType::Dynamic {
          self.komi = AtomicIsize::new(-self.komi.load(Ordering::Relaxed));
        }
      }
    }
  }

  pub fn new(config: UctConfig, length: Pos) -> UctRoot {
    UctRoot {
      config,
      node: None,
      player: Player::Red,
      moves_count: 0,
      hash: 0,
      wave_pruning: WavePruning::new(length),
      komi: AtomicIsize::new(0),
      komi_visits: AtomicUsize::new(0),
      komi_wins: AtomicUsize::new(0),
      komi_draws: AtomicUsize::new(0),
    }
  }

  fn random_result(field: &Field, player: Player, komi: i32) -> Option<Player> {
    use std::cmp::Ordering;
    let red_komi = if player == Player::Red { komi } else { -komi };
    let red_score = field.score(Player::Red);
    match red_score.cmp(&red_komi) {
      Ordering::Greater => Some(Player::Red),
      Ordering::Less => Some(Player::Black),
      Ordering::Equal => None,
    }
  }

  fn play_random_game<R: Rng>(
    field: &mut Field,
    player: Player,
    rng: &mut R,
    possible_moves: &mut Vec<Pos>,
    komi: i32,
  ) -> Option<Player> {
    possible_moves.shuffle(rng);
    let mut cur_player = player;
    for &pos in possible_moves.iter() {
      let cell = field.cell(pos);
      if cell.is_putting_allowed() && !cell.is_empty_base() {
        field.put_point(pos, cur_player);
        cur_player = cur_player.next();
      }
    }
    UctRoot::random_result(field, player, komi)
  }

  fn ucb(&self, parent_visits_ln: f64, node: &UctNode, ucb_type: UcbType) -> f64 {
    let wins = node.get_wins() as f64;
    let draws = node.get_draws() as f64;
    let visits = node.get_visits() as f64;
    let win_rate = (wins + draws * self.config.draw_weight) / visits;
    let uct = match ucb_type {
      UcbType::Winrate => 0f64,
      UcbType::Ucb1 => self.config.uctk * (2.0 * parent_visits_ln / visits).sqrt(),
      UcbType::Ucb1Tuned => {
        let v = (wins + draws * self.config.draw_weight * self.config.draw_weight) / visits - win_rate * win_rate
          + (2.0 * parent_visits_ln / visits).sqrt();
        self.config.uctk * (v.min(0.25) * parent_visits_ln / visits).sqrt()
      }
    };
    win_rate + uct
  }

  fn create_children<R: Rng>(field: &Field, possible_moves: &mut Vec<Pos>, node: &UctNode, rng: &mut R) {
    possible_moves.shuffle(rng);
    let mut children = None;
    for &pos in possible_moves.iter() {
      if field.cell(pos).is_putting_allowed() {
        let mut cur_child = Box::new(UctNode::new(pos));
        cur_child.set_sibling_option(children);
        children = Some(cur_child);
      }
    }
    if let Some(child) = children {
      node.set_child(child)
    }
  }

  fn uct_select<'a>(&self, node: &'a UctNode) -> Option<&'a UctNode> {
    let node_visits_ln = (node.get_visits() as f64).ln();
    let mut best_uct = 0f64;
    let mut result = None;
    let mut next = node.get_child_ref();
    while let Some(next_node) = next {
      let visits = next_node.get_visits();
      let wins = next_node.get_wins();
      let uct_value = if visits == usize::max_value() {
        if wins == usize::max_value() {
          return Some(next_node);
        }
        -1f64
      } else if visits == 0 {
        self.config.fpu
      } else {
        self.ucb(node_visits_ln, next_node, self.config.ucb_type)
      };
      if uct_value > best_uct {
        best_uct = uct_value;
        result = Some(next_node);
      }
      next = next_node.get_sibling_ref();
    }
    result
  }

  fn play_simulation_rec<R: Rng>(
    &self,
    field: &mut Field,
    player: Player,
    node: &UctNode,
    possible_moves: &mut Vec<Pos>,
    rng: &mut R,
    komi: i32,
    depth: u32,
  ) -> Option<Player> {
    let random_result = if node.get_visits() < self.config.when_create_children || depth == self.config.depth {
      UctRoot::play_random_game(field, player, rng, possible_moves, komi)
    } else {
      if node.get_child_ref().is_none() {
        UctRoot::create_children(field, possible_moves, node, rng)
      }
      if let Some(next) = self.uct_select(node) {
        let pos = next.get_pos();
        field.put_point(pos, player);
        if common::is_last_move_stupid(field, pos, player) {
          field.undo();
          next.lose_node();
          return self.play_simulation_rec(field, player, node, possible_moves, rng, komi, depth);
        }
        if common::is_penult_move_stupid(field) {
          // Theoretically, visits in this node may be overflowed by another thread, but
          // there's nothing to worry about. In this case this node will be
          // marked as losing on the next visit
          // because uct_select method selects
          // child determined.
          node.lose_node();
          return Some(player);
        }
        self.play_simulation_rec(field, player.next(), next, possible_moves, rng, -komi, depth + 1)
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

  fn play_simulation<R: Rng>(
    &self,
    field: &mut Field,
    player: Player,
    possible_moves: &mut Vec<Pos>,
    rng: &mut R,
    ratched: &AtomicIsize,
  ) {
    if let Some(node) = self.node.as_ref() {
      self.play_simulation_rec(
        field,
        player,
        node,
        possible_moves,
        rng,
        self.komi.load(Ordering::Relaxed) as i32,
        0,
      );
      if self.config.komi_type == UctKomiType::Dynamic {
        let visits = node.get_visits();
        let komi_visits = self.komi_visits.load(Ordering::Relaxed);
        let delta_visits = visits - komi_visits;
        if delta_visits > self.config.komi_min_iterations {
          let wins = node.get_wins();
          let delta_wins = wins - self.komi_wins.load(Ordering::Relaxed);
          let draws = node.get_draws();
          let delta_draws = draws - self.komi_draws.load(Ordering::Relaxed);
          let win_rate =
            1f64 - (delta_wins as f64 + delta_draws as f64 * self.config.draw_weight) / delta_visits as f64;
          let komi = self.komi.load(Ordering::Relaxed);
          if win_rate < self.config.red || win_rate > self.config.green && komi < ratched.load(Ordering::Relaxed) {
            let result = self
              .komi_visits
              .compare_exchange(komi_visits, visits, Ordering::Relaxed, Ordering::Relaxed);
            if result.is_ok() {
              self.komi_wins.store(wins, Ordering::Relaxed);
              self.komi_draws.store(draws, Ordering::Relaxed);
              if win_rate < self.config.red {
                if komi > 0 {
                  ratched.store(komi - 1, Ordering::Relaxed);
                }
                self.komi.fetch_sub(1, Ordering::Relaxed);
                info!(
                  "Komi decreased after {} visits: {}. Winrate is {}.",
                  visits,
                  komi - 1,
                  win_rate
                );
              } else {
                self.komi.fetch_add(1, Ordering::Relaxed);
                info!(
                  "Komi increased after {} visits: {}. Winrate is {}.",
                  visits,
                  komi + 1,
                  win_rate
                );
              }
            }
          }
        }
      }
    }
  }

  pub fn best_move<S, R>(
    &mut self,
    field: &Field,
    player: Player,
    rng: &mut R,
    should_stop: &AtomicBool,
    max_iterations_count: usize,
  ) -> Option<NonZeroPos>
  where
    R: Rng + SeedableRng<Seed = S> + Send,
    Standard: Distribution<S>,
  {
    info!("Generating best move for player {}.", player);
    debug!(
      "Moves history: {:?}.",
      field
        .points_seq()
        .iter()
        .map(|&pos| (field.to_x(pos), field.to_y(pos), field.cell(pos).get_player()))
        .collect::<Vec<(u32, u32, Player)>>()
    );
    debug!("Next random u64: {}.", rng.gen::<u64>());
    self.update(field, player, rng);
    info!(
      "Komi is {}, type is {:?}.",
      self.komi.load(Ordering::Relaxed),
      self.config.komi_type
    );
    let iterations = AtomicUsize::new(0);
    let ratched = AtomicIsize::new(isize::max_value());
    crossbeam::scope(|scope| {
      for _ in 0..self.config.threads_count {
        let new_rng = R::from_seed(rng.gen());
        scope.spawn(|_| {
          let mut local_field = field.clone();
          let mut local_rng = new_rng;
          let mut possible_moves = self.wave_pruning.moves().clone();
          while !should_stop.load(Ordering::Relaxed) && iterations.load(Ordering::Relaxed) < max_iterations_count {
            self.play_simulation(&mut local_field, player, &mut possible_moves, &mut local_rng, &ratched);
            for _ in 0..local_field.moves_count() - self.moves_count {
              local_field.undo();
            }
            iterations.fetch_add(1, Ordering::Relaxed);
          }
        });
      }
    })
    .expect("UCT best_move_generic panic");
    info!("Iterations count: {}.", iterations.load(Ordering::Relaxed));
    let mut best_uct = 0f64;
    let mut result = None;
    if let Some(ref root) = self.node {
      let mut next = root.get_child_ref();
      let root_visits_ln = (root.get_visits() as f64).ln();
      while let Some(next_node) = next {
        let uct_value = if next_node.get_visits() > 0 {
          self.ucb(root_visits_ln, next_node, UcbType::Winrate)
        } else {
          0f64
        };
        let pos = next_node.get_pos();
        info!(
          "Uct for move ({}, {}) is {}, {} wins, {} draws, {} visits.",
          field.to_x(pos),
          field.to_y(pos),
          uct_value,
          next_node.get_wins(),
          next_node.get_draws(),
          next_node.get_visits()
        );
        #[allow(clippy::float_cmp)]
        if uct_value > best_uct || uct_value == best_uct && rng.gen::<bool>() {
          best_uct = uct_value;
          result = NonZeroPos::new(pos);
        }
        next = next_node.get_sibling_ref();
      }
    }
    if let Some(pos) = result {
      info!(
        "Best move is ({}, {}), uct is {}.",
        field.to_x(pos.get()),
        field.to_y(pos.get()),
        best_uct
      );
    }
    result
  }
}
