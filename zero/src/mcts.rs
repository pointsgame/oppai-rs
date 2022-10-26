use ndarray::Array2;
use num_traits::{Float, Zero};
use oppai_field::field::{to_x, to_y, NonZeroPos, Pos};
use oppai_rotate::rotate::{rotate, rotate_sizes};

pub struct MctsNode<N> {
  /// Current move.
  pub pos: Pos,
  /// Visits count.
  pub n: u64,
  /// Prior probability.
  pub p: N,
  /// Total action value.
  pub w: N,
  /// Children moves.
  pub children: Vec<MctsNode<N>>,
}

impl<N: Zero> Default for MctsNode<N> {
  fn default() -> Self {
    MctsNode::new(0, N::zero(), N::zero())
  }
}

const C_PUCT: f64 = 1f64;

const TEMPERATURE: f64 = 1f64;

impl<N: Zero> MctsNode<N> {
  pub fn new(pos: Pos, p: N, w: N) -> Self {
    Self {
      pos,
      n: 0,
      p,
      w,
      children: Vec::new(),
    }
  }
}

impl<N: Float> MctsNode<N> {
  /// Mean action value.
  pub fn q(&self) -> N {
    self.w / N::from(self.n + 1).unwrap()
  }

  pub fn probability(&self) -> N {
    N::from(self.n).unwrap().powf(N::one() / N::from(TEMPERATURE).unwrap())
  }

  pub fn mcts_value(&self, parent_n: u64) -> N {
    // TODO: moinigo uses a more complex formula
    // max(1, parent_n - 1) instead of parent_n
    // 2.0 * (log((1.0 + parent_n + c_puct_base) / c_puct_base) + c_puct_init) instead of C_PUCT
    self.q() + N::from(C_PUCT).unwrap() * self.p * N::from(parent_n).unwrap().sqrt() / N::from(1 + self.n).unwrap()
  }

  fn select_child(&mut self) -> Option<&mut MctsNode<N>> {
    let n = self.n;
    self
      .children
      .iter_mut()
      .max_by(|child1, child2| child1.mcts_value(n).partial_cmp(&child2.mcts_value(n)).unwrap())
  }

  pub fn select(&mut self) -> Vec<Pos> {
    // TODO: noise?
    let mut moves = Vec::new();
    let mut node = self;

    while let Some(child) = node.select_child() {
      moves.push(child.pos);
      // virtual loss
      child.w = child.w - N::one();
      node = child;
    }

    moves
  }

  pub fn revert_virtual_loss(&mut self, moves: &[Pos]) {
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.w = node.w + N::one();
    }
  }

  pub fn add_result(&mut self, moves: &[Pos], mut result: N, children: Vec<MctsNode<N>>) {
    self.n += 1;
    self.w = self.w - result;
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.n += 1;
      node.w = node.w + result;
      result = -result;
    }
    node.children = children;
  }

  /// Improved stochastic policy values.
  pub fn policies(&self, width: u32, height: u32, rotation: u8) -> Array2<N> {
    let (width, height) = rotate_sizes(width, height, rotation);
    let mut policies = Array2::zeros((height as usize, width as usize));

    for child in &self.children {
      let x = to_x(width, child.pos);
      let y = to_y(width, child.pos);
      let (x, y) = rotate(width, height, x, y, rotation);
      policies[(y as usize, x as usize)] = N::from(child.n).unwrap() / N::from(self.n - 1).unwrap();
    }

    policies
  }

  pub fn best_child(self) -> Option<MctsNode<N>> {
    // TODO: option to use winrate?
    self.children.into_iter().max_by_key(|child| child.n)
  }

  pub fn best_move(&self) -> Option<NonZeroPos> {
    // TODO: option to use winrate?
    self
      .children
      .iter()
      .max_by_key(|child| child.n)
      .and_then(|child| NonZeroPos::new(child.pos))
  }
}
