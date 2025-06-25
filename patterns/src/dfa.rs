use oppai_field::cell::Cell;
use oppai_field::player::Player;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, vec_deque::VecDeque};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DfaState<P: Clone> {
  pub empty: usize,
  pub red: usize,
  pub black: usize,
  pub bad: usize,
  pub is_final: bool,
  pub patterns: Vec<P>,
}

impl<P: Clone> DfaState<P> {
  pub fn new(empty: usize, red: usize, black: usize, bad: usize, is_final: bool, patterns: Vec<P>) -> DfaState<P> {
    DfaState {
      empty,
      red,
      black,
      bad,
      is_final,
      patterns,
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Dfa<P: Clone> {
  states: Vec<DfaState<P>>,
}

impl<P: Clone> Default for Dfa<P> {
  fn default() -> Self {
    let state = DfaState::new(0, 0, 0, 0, true, Vec::new());
    Self { states: vec![state] }
  }
}

impl<P: Clone> Dfa<P> {
  pub fn new(states: Vec<DfaState<P>>) -> Dfa<P> {
    Dfa { states }
  }

  pub fn is_empty(&self) -> bool {
    self.states[0].is_final && self.states[0].patterns.is_empty()
  }

  pub fn states_count(&self) -> usize {
    self.states.len()
  }

  pub fn product(&self, other: &Dfa<P>) -> Dfa<P> {
    fn build_state<P: Clone>(other_len: usize, left: &DfaState<P>, right: &DfaState<P>) -> DfaState<P> {
      DfaState {
        empty: left.empty * other_len + right.empty,
        red: left.red * other_len + right.red,
        black: left.black * other_len + right.black,
        bad: left.bad * other_len + right.bad,
        is_final: left.is_final && right.is_final,
        patterns: left.patterns.iter().chain(right.patterns.iter()).cloned().collect(),
      }
    }
    if self.is_empty() {
      return other.clone();
    }
    if other.is_empty() {
      return self.clone();
    }
    let other_len = other.states.len();
    let mut states = vec![build_state(other_len, &self.states[0], &other.states[0])];
    let mut map = HashMap::new();
    map.insert(0, 0);
    // TODO: compare performance with using a stack to minimize jumps in memory
    // empty value is supposed to appear often, it should have a small jump
    // bad value is supposed to be rare, it should imply a big jump
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
        states.push(build_state(
          other_len,
          &self.states[self_state.empty],
          &other.states[other_state.empty],
        ));
        empty_map_next
      });
      let red_map_next = map.get(&red_next).cloned().unwrap_or_else(|| {
        q.push_back(red_next);
        let red_map_next = states.len();
        map.insert(red_next, red_map_next);
        states.push(build_state(
          other_len,
          &self.states[self_state.red],
          &other.states[other_state.red],
        ));
        red_map_next
      });
      let black_map_next = map.get(&black_next).cloned().unwrap_or_else(|| {
        q.push_back(black_next);
        let black_map_next = states.len();
        map.insert(black_next, black_map_next);
        states.push(build_state(
          other_len,
          &self.states[self_state.black],
          &other.states[other_state.black],
        ));
        black_map_next
      });
      let bad_map_next = map.get(&bad_next).cloned().unwrap_or_else(|| {
        q.push_back(bad_next);
        let bad_map_next = states.len();
        map.insert(bad_next, bad_map_next);
        states.push(build_state(
          other_len,
          &self.states[self_state.bad],
          &other.states[other_state.bad],
        ));
        bad_map_next
      });
      states[cur_map_idx].empty = empty_map_next;
      states[cur_map_idx].red = red_map_next;
      states[cur_map_idx].black = black_map_next;
      states[cur_map_idx].bad = bad_map_next;
    }
    Dfa { states }
  }

  pub fn run<T: Iterator<Item = Cell>>(&self, iter: &mut T, inv_color: bool, first_match: bool) -> &Vec<P> {
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
            Player::Black => state_idx = if inv_color { state.red } else { state.black },
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
