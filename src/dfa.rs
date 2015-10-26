use std::iter;
use std::collections::vec_deque::VecDeque;
use player::Player;
use cell::Cell;

struct DfaState {
    empty: i32,
    red: i32,
    black: i32,
    bad: i32,
    pattern: i32
}

struct Dfa {
    states: Vec<DfaState>
}

impl Dfa {
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

    pub fn delete_non_reachable(&mut self) {
        let mut reachable = iter::repeat(0).take(self.states.len()).collect::<Vec<u32>>();
        let mut q = VecDeque::with_capacity(self.states.len());
        q.push_back(1);
        while let Some(idx) = q.pop_front() {
            let state = &self.states[idx as usize];
            if state.empty != -1 && reachable[state.empty as usize] == 0 {
                reachable[state.empty as usize] = 1;
                q.push_back(state.empty);
            }
            if state.red != -1 && reachable[state.red as usize] == 0 {
                reachable[state.red as usize] = 1;
                q.push_back(state.red);
            }
            if state.black != -1 && reachable[state.black as usize] == 0 {
                reachable[state.black as usize] = 1;
                q.push_back(state.black);
            }
            if state.bad != -1 && reachable[state.bad as usize] == 0 {
                reachable[state.bad as usize] = 1;
                q.push_back(state.bad);
            }
        }
        let mut shifts = Vec::with_capacity(self.states.len());
        for r in &reachable {
            let shift = shifts.last().map_or(0, |&d| d) + 1 - r;
            shifts.push(shift);
        }
        let deletions = shifts.last().map_or(0, |&d| d);
        if deletions == 0 {
            return;
        }
        let mut new_states = Vec::with_capacity(self.states.len() - deletions as usize);
        for state in self.states.iter().zip(reachable.into_iter()).filter_map(|(state, r)| if r == 1 { Some(state) } else { None }) {
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
}
