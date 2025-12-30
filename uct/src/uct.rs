use crate::wave_pruning::WavePruning;
use oppai_common::common;
use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;
use rand::distr::{Distribution, StandardUniform};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::{
  mem::{self, ManuallyDrop},
  ptr,
  sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
};
use strum::{EnumString, VariantNames};
use thin_vec::ThinVec;

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
pub enum UcbType {
  Winrate,
  Ucb1,
  Ucb1Tuned,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, EnumString, VariantNames)]
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
      #[cfg(not(target_arch = "wasm32"))]
      threads_count: num_cpus::get(),
      #[cfg(target_arch = "wasm32")]
      threads_count: 1,
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
  children: AtomicPtr<()>,
}

impl Drop for UctNode {
  fn drop(&mut self) {
    self.clear_children();
  }
}

impl Clone for UctNode {
  fn clone(&self) -> Self {
    Self {
      wins: AtomicUsize::new(self.wins.load(Ordering::SeqCst)),
      draws: AtomicUsize::new(self.draws.load(Ordering::SeqCst)),
      visits: AtomicUsize::new(self.visits.load(Ordering::SeqCst)),
      pos: self.pos,
      children: unsafe {
        self.get_children().map_or(AtomicPtr::default(), |children| {
          let children = ManuallyDrop::new(ThinVec::from(children));
          AtomicPtr::new(mem::transmute::<ManuallyDrop<ThinVec<UctNode>>, *mut ()>(children))
        })
      },
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
      children: AtomicPtr::default(),
    }
  }

  pub fn get_pos(&self) -> Pos {
    self.pos
  }

  unsafe fn get_children<'a>(&'a self) -> Option<&'a [UctNode]> {
    let ptr = self.children.load(Ordering::Relaxed);
    if ptr.is_null() {
      None
    } else {
      unsafe {
        let children = mem::transmute::<*mut (), ManuallyDrop<ThinVec<UctNode>>>(ptr);
        Some(mem::transmute::<&[UctNode], &'a [UctNode]>(children.as_slice()))
      }
    }
  }

  fn clear_children(&self) -> Option<ThinVec<UctNode>> {
    let ptr = self.children.swap(ptr::null_mut(), Ordering::Relaxed);
    if ptr.is_null() {
      None
    } else {
      Some(unsafe { mem::transmute::<*mut (), ThinVec<UctNode>>(ptr) })
    }
  }

  pub fn set_children(&self, children: ThinVec<UctNode>) {
    let ptr = unsafe { mem::transmute::<ThinVec<UctNode>, *mut ()>(children) };
    let result = self
      .children
      .compare_exchange(ptr::null_mut(), ptr, Ordering::Relaxed, Ordering::Relaxed);
    if result.is_err() {
      unsafe {
        mem::transmute::<*mut (), ThinVec<UctNode>>(ptr);
      }
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
    self.visits.store(usize::MAX, Ordering::Relaxed);
  }

  pub fn clear_stats(&self) {
    self.wins.store(0, Ordering::Relaxed);
    self.draws.store(0, Ordering::Relaxed);
    self.visits.store(0, Ordering::Relaxed);
  }
}

pub struct UctRoot {
  config: UctConfig,
  node: Option<UctNode>,
  player: Player,
  moves_count: usize,
  hash: u64,
  wave_pruning: WavePruning,
  komi: AtomicIsize,
  komi_visits: AtomicUsize,
  komi_wins: AtomicUsize,
  komi_draws: AtomicUsize,
}

impl Clone for UctRoot {
  fn clone(&self) -> Self {
    Self {
      config: self.config.clone(),
      node: self.node.clone(),
      player: self.player,
      moves_count: self.moves_count,
      hash: self.hash,
      wave_pruning: self.wave_pruning.clone(),
      komi: AtomicIsize::new(self.komi.load(Ordering::SeqCst)),
      komi_visits: AtomicUsize::new(self.komi_visits.load(Ordering::SeqCst)),
      komi_wins: AtomicUsize::new(self.komi_wins.load(Ordering::SeqCst)),
      komi_draws: AtomicUsize::new(self.komi_draws.load(Ordering::SeqCst)),
    }
  }
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

  fn init(&mut self, field: &mut Field, player: Player) {
    debug!("Initialization.");
    self.node = Some(UctNode::new(0));
    self.player = player;
    self.moves_count = field.moves_count();
    self.hash = field.hash;
    if self.config.komi_type != UctKomiType::None {
      self.komi = AtomicIsize::new(field.score(player) as isize);
    }
    self.wave_pruning.init(field, self.config.radius);
  }

  fn expand_node<R: Rng>(node: &mut UctNode, moves: &mut Vec<Pos>, rng: &mut R) {
    if let Some(mut children) = node.clear_children() {
      for child in children.iter_mut() {
        UctRoot::expand_node(child, moves, rng);
      }
      moves.shuffle(rng);
      children.extend(moves.iter().copied().map(UctNode::new));
      node.set_children(children);
    } else if node.get_visits() == usize::MAX {
      node.clear_stats();
    }
  }

  fn update<R: Rng>(&mut self, field: &mut Field, player: Player, rng: &mut R) {
    if self.node.is_some() && field.hash_at(self.moves_count) != Some(self.hash) {
      self.clear();
    }
    if self.node.is_none() {
      self.init(field, player);
    } else {
      let moves = &field.moves;
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
        let next_pos = moves[self.moves_count];
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
        if let Some(children) = self.node.take().and_then(|node| node.clear_children()) {
          if let Some(node) = children.into_iter().find(|node| node.pos == next_pos) {
            debug!(
              "Node found for move ({}, {}).",
              field.to_x(node.pos),
              field.to_y(node.pos)
            );
            self.node = Some(node);
            self.moves_count += 1;
            self.player = self.player.next();
            self.hash = field.hash;
            if self.config.komi_type == UctKomiType::Dynamic {
              self.komi = AtomicIsize::new(-self.komi.load(Ordering::Relaxed));
            }
          } else {
            self.clear();
            self.init(field, player);
            break;
          }
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
    possible_moves: &mut [Pos],
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

  fn create_children<R: Rng>(field: &Field, possible_moves: &mut [Pos], node: &UctNode, rng: &mut R) {
    possible_moves.shuffle(rng);
    let mut moves = possible_moves
      .iter()
      .copied()
      .filter(|&pos| field.cell(pos).is_putting_allowed())
      .peekable();
    if moves.peek().is_none() {
      return;
    }
    let children = moves.map(UctNode::new).collect::<ThinVec<UctNode>>();
    node.set_children(children);
  }

  fn uct_select<'a>(&self, node: &'a UctNode) -> Option<&'a UctNode> {
    let node_visits_ln = (node.get_visits() as f64).ln();
    let mut best_uct = 0f64;
    let mut result = None;
    let children = unsafe { node.get_children() };
    for child in children.iter().flat_map(|children| children.iter()) {
      let visits = child.get_visits();
      let wins = child.get_wins();
      let uct_value = if visits == usize::MAX {
        if wins == usize::MAX {
          return Some(child);
        }
        -1f64
      } else if visits == 0 {
        self.config.fpu
      } else {
        self.ucb(node_visits_ln, child, self.config.ucb_type)
      };
      if uct_value > best_uct {
        best_uct = uct_value;
        result = Some(child);
      }
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
      if unsafe { node.get_children() }.is_none() {
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

  pub fn best_moves<S, R, SS>(
    &mut self,
    field: &mut Field,
    player: Player,
    rng: &mut R,
    should_stop: &SS,
    max_iterations_count: usize,
  ) -> (Vec<(Pos, f64)>, usize, f64)
  where
    R: Rng + SeedableRng<Seed = S> + Send,
    StandardUniform: Distribution<S>,
    SS: Fn() -> bool + Sync,
  {
    info!("Generating best move for player {}.", player);
    debug!(
      "Moves history: {:?}.",
      field
        .moves
        .iter()
        .map(|&pos| (field.to_x(pos), field.to_y(pos), field.cell(pos).get_player()))
        .collect::<Vec<(u32, u32, Player)>>()
    );
    debug!("Next random u64: {}.", rng.random::<u64>());
    self.update(field, player, rng);
    info!(
      "Komi is {}, type is {:?}.",
      self.komi.load(Ordering::Relaxed),
      self.config.komi_type
    );
    let ratched = AtomicIsize::new(isize::MAX);
    #[cfg(not(target_arch = "wasm32"))]
    let iterations = {
      let iterations = AtomicUsize::new(0);
      crossbeam::scope(|scope| {
        for _ in 0..self.config.threads_count {
          let new_rng = R::from_seed(rng.random());
          scope.spawn(|_| {
            let mut local_field = field.clone();
            let mut local_rng = new_rng;
            let mut possible_moves = self.wave_pruning.moves().clone();
            while !should_stop() && iterations.load(Ordering::Relaxed) < max_iterations_count {
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
      iterations.load(Ordering::Relaxed)
    };
    #[cfg(target_arch = "wasm32")]
    let iterations = {
      let mut iterations = 0;
      let mut local_field = field.clone();
      let mut possible_moves = self.wave_pruning.moves().clone();
      while !should_stop() && iterations < max_iterations_count {
        self.play_simulation(&mut local_field, player, &mut possible_moves, rng, &ratched);
        for _ in 0..local_field.moves_count() - self.moves_count {
          local_field.undo();
        }
        iterations += 1;
      }
      info!("Iterations count: {}.", iterations);
      iterations
    };
    let mut moves = Vec::new();
    let winrate = if let Some(ref root) = self.node {
      let root_visits_ln = (root.get_visits() as f64).ln();
      let children = unsafe { root.get_children() };
      for child in children.iter().flat_map(|children| children.iter()) {
        let uct_value = if child.get_visits() > 0 {
          self.ucb(root_visits_ln, child, UcbType::Winrate)
        } else {
          0f64
        };
        let pos = child.get_pos();
        info!(
          "Uct for move ({}, {}) is {}, {} wins, {} draws, {} visits.",
          field.to_x(pos),
          field.to_y(pos),
          uct_value,
          child.get_wins(),
          child.get_draws(),
          child.get_visits()
        );
        moves.push((pos, uct_value));
      }
      (root.get_wins() as f64 + root.get_draws() as f64 / 2.0) / root.get_visits() as f64
    } else {
      0.0
    };
    (moves, iterations, winrate)
  }
}
