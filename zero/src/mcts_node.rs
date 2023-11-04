use num_traits::{Float, Zero};
use oppai_field::field::{NonZeroPos, Pos};
use rand::Rng;
use rand_distr::{Distribution, Exp1, Gamma, Open01, StandardNormal};
use std::iter::Sum;

#[derive(Clone)]
pub struct MctsNode<N> {
  /// Current move.
  pub pos: Pos,
  /// Visits count.
  pub visits: u64,
  /// Prior probability.
  pub policy: N,
  /// Total action value.
  pub wins: N,
  /// Children moves.
  pub children: Vec<MctsNode<N>>,
}

impl<N: Zero> Default for MctsNode<N> {
  #[inline]
  fn default() -> Self {
    MctsNode::new(0, N::zero(), N::zero())
  }
}

const C_PUCT: f64 = 2f64;

impl<N: Zero> MctsNode<N> {
  #[inline]
  pub fn new(pos: Pos, policy: N, wins: N) -> Self {
    Self {
      pos,
      visits: 0,
      policy,
      wins,
      children: Vec::new(),
    }
  }
}

impl<N: Float> MctsNode<N> {
  /// Mean action value.
  #[inline]
  pub fn win_rate(&self) -> N {
    self.wins / N::from(self.visits + 1).unwrap()
  }

  #[inline]
  fn uct(&self, parent_visits_sqrt: N) -> N {
    // There ara different variants of this formula:
    // 1. C_PUCT * p * sqrt(parent_visits) / (1 + visits)
    //    proposed by `Mastering the game of Go without human knowledge`
    // 2. C_PUCT * p * sqrt(ln(parent_visits) / (1 + visits))
    //    proposed by `Integrating Factorization Ranked Features in MCTS: an Experimental Study`
    // 3. sqrt(3 * ln(parent_visits) / (2 * visits)) + 2 / p * sqrt(ln(parent_visits) / parent_visits)
    //    proposed by `Multi-armed Bandits with Episode Context`
    N::from(C_PUCT).unwrap() * self.policy * parent_visits_sqrt / N::from(1 + self.visits).unwrap()
  }

  #[inline]
  pub fn mcts_value(&self, parent_visits_sqrt: N) -> N {
    self.win_rate() + self.uct(parent_visits_sqrt)
  }

  fn select_child(&mut self) -> Option<&mut MctsNode<N>> {
    let n_sqrt = N::from(self.visits).unwrap().sqrt();
    self.children.iter_mut().max_by(|child1, child2| {
      child1
        .mcts_value(n_sqrt)
        .partial_cmp(&child2.mcts_value(n_sqrt))
        .unwrap()
    })
  }

  pub fn select(&mut self) -> Vec<Pos> {
    let mut moves = Vec::new();
    let mut node = self;

    while let Some(child) = node.select_child() {
      moves.push(child.pos);
      // virtual loss
      child.wins = child.wins - N::one();
      node = child;
    }

    moves
  }

  pub fn revert_virtual_loss(&mut self, moves: &[Pos]) {
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.wins = node.wins + N::one();
    }
  }

  pub fn add_result(&mut self, moves: &[Pos], mut result: N, children: Vec<MctsNode<N>>) {
    self.visits += 1;
    self.wins = self.wins - result;
    let mut node = self;
    for &pos in moves {
      node = node.children.iter_mut().find(|child| child.pos == pos).unwrap();
      node.visits += 1;
      node.wins = node.wins + result;
      result = -result;
    }
    node.children = children;
  }

  pub fn best_child(self) -> Option<MctsNode<N>> {
    self.children.into_iter().max_by_key(|child| child.visits)
  }

  pub fn best_move(&self) -> Option<NonZeroPos> {
    self
      .children
      .iter()
      .max_by_key(|child| child.visits)
      .and_then(|child| NonZeroPos::new(child.pos))
  }
}

impl<N> MctsNode<N>
where
  N: Float + Sum,
  StandardNormal: Distribution<N>,
  Exp1: Distribution<N>,
  Open01: Distribution<N>,
{
  pub fn add_dirichlet_noise<R: Rng>(&mut self, rng: &mut R, epsilon: N, shape: N) {
    let gamma = Gamma::<N>::new(N::from(shape).unwrap(), N::one()).unwrap();
    let mut dirichlet = gamma.sample_iter(rng).take(self.children.len()).collect::<Vec<_>>();
    let sum = dirichlet.iter().cloned().sum::<N>();
    if sum == N::zero() {
      return;
    }
    for eta in dirichlet.iter_mut() {
      *eta = *eta / sum;
    }
    for (child, eta) in self.children.iter_mut().zip(dirichlet.into_iter()) {
      child.policy = child.policy * (N::one() - epsilon) + epsilon * eta;
    }
  }
}
