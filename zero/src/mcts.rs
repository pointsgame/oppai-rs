use decorum::R64;
use ndarray::Array2;
use oppai_field::field::{to_x, to_y, Pos};
use oppai_rotate::rotate::{rotate, rotate_sizes};

pub struct MctsNode {
  /// Current move.
  pub pos: Pos,
  /// Visits count.
  pub n: u64,
  /// Prior probability.
  pub p: f64,
  /// Total action value.
  pub w: f64,
  /// Children moves.
  pub children: Vec<MctsNode>,
}

const C_PUCT: f64 = 1f64;

const TEMPERATURE: f64 = 1f64;

impl MctsNode {
  pub fn new(pos: Pos, p: f64, w: f64) -> Self {
    Self {
      pos,
      n: 0,
      p,
      w,
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

  fn select_child(&mut self) -> Option<&mut MctsNode> {
    let n = self.n;
    self
      .children
      .iter_mut()
      .max_by_key(|child| R64::from(child.mcts_value(n)))
  }

  pub fn select(&mut self) -> Vec<Pos> {
    // TODO: noise?
    let mut moves = Vec::new();
    let mut node = self;

    while let Some(child) = node.select_child() {
      moves.push(child.pos);
      // virtual loss
      child.w -= 1.0;
      node = child;
    }

    moves
  }

  pub fn revert_virtual_loss(&mut self, moves: &[Pos]) {
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.w += 1.0;
    }
  }

  pub fn add_result(&mut self, moves: &[Pos], mut result: f64, children: Vec<MctsNode>) {
    self.n += 1;
    self.w -= result;
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.n += 1;
      node.w += result;
      result = -result;
    }
    if !children.is_empty() {
      node.children = children;
    }
  }

  /// Improved stochastic policy values.
  pub fn policies(&self, width: u32, height: u32, rotation: u8) -> Array2<f64> {
    let (width, height) = rotate_sizes(width, height, rotation);
    let mut policies = Array2::zeros((height as usize, width as usize));

    for child in &self.children {
      let x = to_x(width, child.pos);
      let y = to_y(width, child.pos);
      let (x, y) = rotate(width, height, x, y, rotation);
      policies[(y as usize, x as usize)] = child.n as f64 / self.n as f64;
    }

    policies
  }
}
