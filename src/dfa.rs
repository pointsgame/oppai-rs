use std::hash::Hash;
use std::collections::vec_deque::VecDeque;
use std::collections::{HashSet, HashMap};
use player::Player;
use cell::Cell;

#[derive(Clone, Debug)]
pub struct DfaState<P: Eq + Hash + Clone> {
  pub empty: usize,
  pub red: usize,
  pub black: usize,
  pub bad: usize,
  pub is_final: bool,
  pub patterns: HashSet<P>
}

impl<P: Eq + Hash + Clone> DfaState<P> {
  pub fn new(empty: usize, red: usize, black: usize, bad: usize, is_final: bool, patterns: HashSet<P>) -> DfaState<P> {
    DfaState {
      empty: empty,
      red: red,
      black: black,
      bad: bad,
      is_final: is_final,
      patterns: patterns
    }
  }
}

#[derive(Clone, Debug)]
pub struct Dfa<P: Eq + Hash + Clone> {
  states: Vec<DfaState<P>>
}

impl<P: Eq + Hash + Clone> Dfa<P> {
  pub fn empty() -> Dfa<P> {
    let state = DfaState::new(0, 0, 0, 0, true, HashSet::with_capacity(0));
    Dfa {
      states: vec![state]
    }
  }

  pub fn new(states: Vec<DfaState<P>>) -> Dfa<P> {
    Dfa {
      states: states
    }
  }

  pub fn is_empty(&self) -> bool {
    self.states[0].is_final && self.states[0].patterns.is_empty()
  }

  pub fn states_count(&self) -> usize {
    self.states.len()
  }

  pub fn product(&self, other: &Dfa<P>) -> Dfa<P> {
    fn build_state<P: Eq + Hash + Clone>(other_len: usize, left: &DfaState<P>, right: &DfaState<P>) -> DfaState<P> {
      DfaState {
        empty: left.empty * other_len + right.empty,
        red: left.red * other_len + right.red,
        black: left.black * other_len + right.black,
        bad: left.bad * other_len + right.bad,
        is_final: left.is_final && right.is_final,
        patterns: left.patterns.union(&right.patterns).cloned().collect()
      }
    }
    if self.is_empty() {
      return other.clone();
    }
    if other.is_empty() {
      return self.clone();
    }
    let other_len = other.states.len();
    let mut states = Vec::new();
    states.push(build_state(other_len, &self.states[0], &other.states[0]));
    let mut map = HashMap::new();
    map.insert(0, 0);
    let mut q = VecDeque::new();
    q.push_back(0);
    while let Some(cur_idx) = q.pop_front() {
      let self_idx = cur_idx / other_len;
      let other_idx = cur_idx % other_len;
      let self_state = &self.states[self_idx];
      let other_state = &other.states[other_idx];
      let cur_map_idx = map.get(&cur_idx).cloned().unwrap();
      let empty_next = states[cur_map_idx].empty;
      let red_next = states[cur_map_idx].red;
      let black_next = states[cur_map_idx].black;
      let bad_next = states[cur_map_idx].bad;
      let empty_map_next = map.get(&empty_next).cloned().unwrap_or_else(|| {
        q.push_back(empty_next);
        let empty_map_next = states.len();
        map.insert(empty_next, empty_map_next);
        states.push(build_state(other_len, &self.states[self_state.empty], &other.states[other_state.empty]));
        empty_map_next
      });
      let red_map_next = map.get(&red_next).cloned().unwrap_or_else(|| {
        q.push_back(red_next);
        let red_map_next = states.len();
        map.insert(red_next, red_map_next);
        states.push(build_state(other_len, &self.states[self_state.red], &other.states[other_state.red]));
        red_map_next
      });
      let black_map_next = map.get(&black_next).cloned().unwrap_or_else(|| {
        q.push_back(black_next);
        let black_map_next = states.len();
        map.insert(black_next, black_map_next);
        states.push(build_state(other_len, &self.states[self_state.black], &other.states[other_state.black]));
        black_map_next
      });
      let bad_map_next = map.get(&bad_next).cloned().unwrap_or_else(|| {
        q.push_back(bad_next);
        let bad_map_next = states.len();
        map.insert(bad_next, bad_map_next);
        states.push(build_state(other_len, &self.states[self_state.bad], &other.states[other_state.bad]));
        bad_map_next
      });
      states[cur_map_idx].empty = empty_map_next;
      states[cur_map_idx].red = red_map_next;
      states[cur_map_idx].black = black_map_next;
      states[cur_map_idx].bad = bad_map_next;
    }
    Dfa {
      states: states
    }
  }

  pub fn run<T: Iterator<Item = Cell>>(&self, iter: &mut T, inv_color: bool, first_match: bool) -> &HashSet<P> {
    if self.is_empty() {
      return &self.states[0].patterns;
    }
    let mut state_idx = 0usize;
    loop {
      let state = &self.states[state_idx];
      if state.is_final || first_match && !state.patterns.is_empty() {
        return &state.patterns;
      }
      if let Some(cell) = iter.next() {
        if cell.is_bad() {
          state_idx = state.bad;
        } else if let Some(player) = cell.get_owner() {
          match player {
            Player::Red => state_idx = if inv_color { state.black } else { state.red },
            Player::Black => state_idx = if inv_color { state.red } else { state.black }
          }
        } else {
          state_idx = state.empty;
        }
      } else {
        return &state.patterns;
      }
    }
  }
}
