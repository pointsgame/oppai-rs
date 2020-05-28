use decorum::R64;
use oppai_field::field::Pos;

pub struct MctsNode {
  /// Current move.
  pos: Pos,
  /// Visits count.
  n: u64,
  /// Prior probability.
  p: f64,
  /// Total action value.
  w: u64,
  /// Children moves.
  children: Vec<MctsNode>,
}

const C_PUCT: f64 = 1f64;

const TEMPERATURE: f64 = 1f64;

impl MctsNode {
  pub fn new(pos: Pos, p: f64) -> Self {
    Self {
      pos,
      n: 0,
      p,
      w: 0, // TODO: initialize from parent value?
      children: Vec::new(),
    }
  }

  /// Mean action value.
  pub fn q(&self) -> f64 {
    self.w as f64 / (self.n + 1) as f64
  }

  pub fn probability(&self) -> f64 {
    (self.n as f64).powf(1f64 / TEMPERATURE)
  }

  pub fn mcts_value(&self, parent_n: u64) -> f64 {
    // TODO: moinigo uses a more complex formula
    // max(1, parent_n - 1) instead of parent_n
    // 2.0 * (log((1.0 + parent_n + c_puct_base) / c_puct_base) + c_puct_init) instead of C_PUCT
    self.q() + C_PUCT * self.p * (parent_n as f64).sqrt() / (1 + self.n) as f64
  }

  pub fn select(&mut self, moves: &mut Vec<Pos>) {
    // TODO: noise?
    let n = self.n;
    if let Some(child) = self
      .children
      .iter_mut()
      .max_by_key(|child| R64::from(child.mcts_value(n)))
    {
      // virtual loss
      child.w -= 1;
      moves.push(child.pos);
      child.select(moves)
    }
  }
}
