use std::{iter, mem};
use std::collections::vec_deque::VecDeque;
use std::collections::{HashSet, HashMap};
use player::Player;
use cell::Cell;

#[derive(Clone, Debug)]
pub struct DfaState {
  empty: usize,
  red: usize,
  black: usize,
  bad: usize,
  is_final: bool,
  patterns: HashSet<usize>
}

impl DfaState {
  pub fn new(empty: usize, red: usize, black: usize, bad: usize, is_final: bool, patterns: HashSet<usize>) -> DfaState {
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
pub struct Dfa {
  states: Vec<DfaState>
}

impl Dfa {
  pub fn empty() -> Dfa {
    let state = DfaState::new(0, 0, 0, 0, true, HashSet::with_capacity(0));
    Dfa {
      states: vec![state]
    }
  }

  pub fn new(states: Vec<DfaState>) -> Dfa {
    Dfa {
      states: states
    }
  }

  pub fn is_empty(&self) -> bool {
    self.states[0].is_final == true && self.states[0].patterns.is_empty()
  }

  pub fn product(&self, other: &Dfa) -> Dfa {
    fn build_state(other_len: usize, left: &DfaState, right: &DfaState) -> DfaState {
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

  pub fn run<T: Iterator<Item = Cell>>(&self, iter: &mut T, inv_color: bool, first_match: bool) -> &HashSet<usize> {
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

  fn delete_states(&mut self, states_for_delete: Vec<usize>) {
    let mut shifts = Vec::with_capacity(self.states.len());
    for sd in &states_for_delete {
      let shift = shifts.last().map_or(0, |&d| d) + sd;
      shifts.push(shift);
    }
    let deletions = shifts.last().map_or(0, |&d| d);
    if deletions == 0 {
      return;
    }
    let mut states = Vec::with_capacity(self.states.len() - deletions);
    mem::swap(&mut self.states, &mut states);
    for state in states.into_iter().zip(states_for_delete.into_iter()).filter_map(|(state, sd)| if sd == 0 { Some(state) } else { None }) {
      let new_state = DfaState {
        empty: state.empty - shifts[state.empty],
        red: state.red - shifts[state.red],
        black: state.black - shifts[state.black],
        bad: state.bad - shifts[state.bad],
        is_final: state.is_final,
        patterns: state.patterns
      };
      self.states.push(new_state);
    }
  }

  pub fn delete_non_reachable(&mut self) {
    if self.is_empty() {
      return;
    }
    let mut non_reachable = iter::repeat(1).take(self.states.len()).collect::<Vec<usize>>();
    non_reachable[0] = 0;
    let mut q = VecDeque::with_capacity(self.states.len());
    q.push_back(0);
    while let Some(idx) = q.pop_front() {
      let state = &self.states[idx];
      if non_reachable[state.empty] == 1 {
        non_reachable[state.empty] = 0;
        q.push_back(state.empty);
      }
      if non_reachable[state.red] == 1 {
        non_reachable[state.red] = 0;
        q.push_back(state.red);
      }
      if non_reachable[state.black] == 1 {
        non_reachable[state.black] = 0;
        q.push_back(state.black);
      }
      if non_reachable[state.bad] == 1 {
        non_reachable[state.bad] = 0;
        q.push_back(state.bad);
      }
    }
    self.delete_states(non_reachable);
  }

  fn pyramid_idx_base(i: usize) -> usize {
    i * (i - 1) / 2
  }

  fn pyramid_idx(i: usize, j: usize) -> usize {
    Dfa::pyramid_idx_base(i) + j
  }

  pub fn minimize(&mut self) { //TODO: delete unnecesarry states at the end.
    if self.is_empty() {
      return;
    }
    let len = self.states.len();
    let mut not_equal = iter::repeat(0).take(len * (len - 1) / 2 + len - 1).collect::<Vec<u32>>();
    for (i, pattern_i) in self.states.iter().enumerate().skip(1) {
      let base = Dfa::pyramid_idx_base(i);
      for (j, pattern_j) in self.states[.. i].iter().enumerate() {
        if pattern_i.is_final != pattern_j.is_final || pattern_i.patterns != pattern_j.patterns {
          not_equal[base + j] = 1;
        }
      }
    }
    let mut another_iter = true;
    while another_iter {
      another_iter = false;
      for (i, pattern_i) in self.states.iter().enumerate().skip(1) {
        let base = Dfa::pyramid_idx_base(i);
        for (j, pattern_j) in self.states[.. i].iter().enumerate() {
          let idx = base + j;
          if not_equal[idx] == 0 {
            if pattern_i.empty != pattern_j.empty && not_equal[Dfa::pyramid_idx(pattern_i.empty, pattern_j.empty)] == 1 ||
               pattern_i.red != pattern_j.red && not_equal[Dfa::pyramid_idx(pattern_i.red, pattern_j.red)] == 1 ||
               pattern_i.black != pattern_j.black && not_equal[Dfa::pyramid_idx(pattern_i.black, pattern_j.black)] == 1 ||
               pattern_i.bad != pattern_j.bad && not_equal[Dfa::pyramid_idx(pattern_i.bad, pattern_j.bad)] == 1 {
              not_equal[idx] = 1;
              another_iter = true;
            }
          }
        }
      }
    }
    let mut deleted = iter::repeat(0).take(self.states.len()).collect::<Vec<usize>>();
    for i in 1 .. self.states.len() {
      let base = Dfa::pyramid_idx_base(i);
      for j in 0 .. i {
        let idx = base + j;
        if not_equal[idx] == 0 && deleted[j] == 0 {
          for state in &mut self.states {
            if state.empty == i {
              state.empty = j;
            }
            if state.red == i {
              state.red = j;
            }
            if state.black == i {
              state.black = j;
            }
            if state.bad == i {
              state.bad = j;
            }
          }
          deleted[i] = 1;
        }
      }
    }
    self.delete_states(deleted);
  }
}
