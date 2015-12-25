use std::iter;
use std::collections::vec_deque::VecDeque;
use player::Player;
use cell::Cell;

pub struct DfaState {
  empty: i32,
  red: i32,
  black: i32,
  bad: i32,
  pattern: i32
}

impl DfaState {
  pub fn new(empty: i32, red: i32, black: i32, bad: i32, pattern: i32) -> DfaState {
    DfaState {
      empty: empty,
      red: red,
      black: black,
      bad: bad,
      pattern: pattern
    }
  }
}

pub struct Dfa {
  states: Vec<DfaState>
}

impl Dfa {
  pub fn new(states: Vec<DfaState>) -> Dfa {
    Dfa {
      states: states
    }
  }

  pub fn product(&self, other: &Dfa) -> Dfa {
    let other_len = other.states.len();
    let other_len_i32 = other_len as i32;
    let mut new_states = Vec::with_capacity(self.states.len() * other_len);
    for self_state in &self.states {
      let base_empty = (self_state.empty + 1) * other_len_i32;
      let base_red = (self_state.red + 1) * other_len_i32;
      let base_black = (self_state.black + 1) * other_len_i32;
      let base_bad = (self_state.bad + 1) * other_len_i32;
      for other_state in &other.states {
        let new_state = DfaState {
          empty: base_empty + other_state.empty,
          red: base_red + other_state.red,
          black: base_black + other_state.black,
          bad: base_bad + other_state.bad,
          pattern: if self_state.pattern != -1 { self_state.pattern } else { other_state.pattern }
        };
        new_states.push(new_state);
      }
    }
    Dfa {
      states: new_states
    }
  }

  pub fn run(&self, iter: &mut Iterator<Item = Cell>) -> Option<u32> {
    let mut state_idx = 0i32;
    loop {
      let state = &self.states[state_idx as usize];
      if state.pattern != -1 {
        return Some(state.pattern as u32);
      }
      if let Some(cell) = iter.next() {
        if cell.is_bad() {
          state_idx = state.bad;
        } else if let Some(player) = cell.get_owner() {
          match player {
            Player::Red => state_idx = state.red,
            Player::Black => state_idx = state.black
          }
        } else {
          state_idx = state.empty;
        }
      } else {
        return None;
      }
      if state_idx == -1 {
        return None;
      }
    }
  }

  fn delete_states(&mut self, states_for_delete: Vec<u32>) {
    let mut shifts = Vec::with_capacity(self.states.len());
    for sd in &states_for_delete {
      let shift = shifts.last().map_or(0, |&d| d) + sd;
      shifts.push(shift);
    }
    let deletions = shifts.last().map_or(0, |&d| d);
    if deletions == 0 {
      return;
    }
    let mut new_states = Vec::with_capacity(self.states.len() - deletions as usize);
    for state in self.states.iter().zip(states_for_delete.into_iter()).filter_map(|(state, sd)| if sd == 0 { Some(state) } else { None }) {
      let new_state = DfaState {
        empty: if state.empty == -1 { -1 } else { state.empty - shifts[state.empty as usize] as i32 },
        red: if state.red == -1 { -1 } else { state.red - shifts[state.red as usize] as i32 },
        black: if state.black == -1 { -1 } else { state.black - shifts[state.black as usize] as i32 },
        bad: if state.bad == -1 { -1 } else { state.bad - shifts[state.bad as usize] as i32 },
        pattern: state.pattern
      };
      new_states.push(new_state);
    }
    self.states = new_states;
  }

  pub fn delete_non_reachable(&mut self) {
    let mut non_reachable = iter::repeat(1).take(self.states.len()).collect::<Vec<u32>>();
    let mut q = VecDeque::with_capacity(self.states.len());
    q.push_back(1);
    while let Some(idx) = q.pop_front() {
      let state = &self.states[idx as usize];
      if state.empty != -1 && non_reachable[state.empty as usize] == 1 {
        non_reachable[state.empty as usize] = 0;
        q.push_back(state.empty);
      }
      if state.red != -1 && non_reachable[state.red as usize] == 1 {
        non_reachable[state.red as usize] = 0;
        q.push_back(state.red);
      }
      if state.black != -1 && non_reachable[state.black as usize] == 1 {
        non_reachable[state.black as usize] = 0;
        q.push_back(state.black);
      }
      if state.bad != -1 && non_reachable[state.bad as usize] == 1 {
        non_reachable[state.bad as usize] = 0;
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

  pub fn minimize(&mut self) {
    let len = self.states.len();
    let mut not_equal = iter::repeat(0).take(len * (len - 1) / 2 + len - 1).collect::<Vec<u32>>();
    for (i, pattern_i) in self.states.iter().enumerate().skip(1) {
      let base = Dfa::pyramid_idx_base(i);
      for (j, pattern_j) in self.states[.. i - 1].iter().enumerate() {
        if pattern_i.pattern != pattern_j.pattern {
          not_equal[base + j] = 1;
        }
      }
    }
    'outer: loop {
      for (i, pattern_i) in self.states.iter().enumerate().skip(1) {
        let base = Dfa::pyramid_idx_base(i);
        for (j, pattern_j) in self.states[.. i - 1].iter().enumerate() {
          let idx = base + j;
          if not_equal[idx] == 0 {
            if pattern_i.empty != pattern_j.empty && (pattern_i.empty == -1 || pattern_j.empty == -1 || not_equal[Dfa::pyramid_idx(pattern_i.empty as usize, pattern_j.empty as usize)] == 1) ||
               pattern_i.red != pattern_j.red && (pattern_i.red == -1 || pattern_j.red == -1 || not_equal[Dfa::pyramid_idx(pattern_i.red as usize, pattern_j.red as usize)] == 1) ||
               pattern_i.black != pattern_j.black && (pattern_i.black == -1 || pattern_j.black == -1 || not_equal[Dfa::pyramid_idx(pattern_i.black as usize, pattern_j.black as usize)] == 1) ||
               pattern_i.bad != pattern_j.bad && (pattern_i.bad == -1 || pattern_j.bad == -1 || not_equal[Dfa::pyramid_idx(pattern_i.bad as usize, pattern_j.black as usize)] == 1) {
              not_equal[idx] = 1;
              continue 'outer;
            }
          }
        }
      }
      break;
    }
    let mut deleted = iter::repeat(0).take(self.states.len()).collect::<Vec<u32>>();
    for i in 1 .. self.states.len() {
      let base = Dfa::pyramid_idx_base(i);
      let i_i32 = i as i32;
      for j in 0 .. i - 1 {
        let idx = base + j;
        if not_equal[idx] == 0 && deleted[j] == 0 {
          let j_i32 = j as i32;
          for state in &mut self.states {
            if state.empty == i_i32 {
              state.empty = j_i32;
            }
            if state.red == i_i32 {
              state.red = j_i32;
            }
            if state.black == i_i32 {
              state.black = j_i32;
            }
            if state.bad == i_i32 {
              state.bad = j_i32;
            }
          }
          deleted[i] = 1;
        }
      }
    }
    self.delete_states(deleted);
  }
}
