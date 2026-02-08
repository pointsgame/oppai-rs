use crate::field_features::{CHANNELS, field_features_len, field_features_to_vec};
use crate::model::Model;
use ndarray::{Array, ArrayView2, s};
use num_traits::Float;
use oppai_field::field::{to_x, to_y};
use oppai_field::{
  field::{Field, NonZeroPos, Pos},
  player::Player,
};
use std::{collections::HashMap, iter, iter::Sum};

/// Represents an edge from a parent to a child in the graph.
struct Edge<N: Float> {
  pos: Pos,
  /// Index into the global node storage
  node_idx: usize,
  /// Number of times this specific edge was traversed (N(n, a))
  visits: u32,
  /// The raw policy prediction P(a)
  prior: N,
  /// Virtual losses to reduce parallelization conflicts
  virtual_losses: u32,
}

/// Represents a single state in the game graph.
struct Node<N: Float> {
  /// N(n): Total visits to this node.
  /// Note: In MCGS, this is effectively 1 + sum(edge.visits).
  visits: u32,
  /// Q(n): Expected utility.
  /// Calculated recursively: (U(n) + sum(edge.visits * child.Q)) / N(n)
  value: N,
  /// U(n): Raw utility from the neural net for this state.
  raw_value: N,
  /// Edges to children.
  children: Vec<Edge<N>>,
}

impl<N: Float> Node<N> {
  fn new() -> Self {
    Node {
      visits: 0,
      value: N::zero(),
      raw_value: N::zero(),
      children: Vec::new(),
    }
  }
}

impl<N: Float> Default for Node<N> {
  fn default() -> Self {
    Self::new()
  }
}

pub struct Search<N: Float> {
  /// Index of the root node in `nodes`
  root_idx: usize,
  /// Arena allocation for nodes
  nodes: Vec<Node<N>>,
  /// Maps Zobrist hash -> index in `nodes`
  map: HashMap<u64, usize>,
}

impl<N: Float + Sum> Search<N> {
  pub fn new(field: &Field) -> Self {
    let mut search = Search {
      root_idx: 0,
      nodes: Vec::new(),
      map: HashMap::new(),
    };

    // Initialize root
    search.add_node(field.hash);
    search
  }

  fn game_result(field: &Field, player: Player) -> N {
    N::from(field.score(player).signum()).unwrap()
  }

  fn add_node(&mut self, hash: u64) -> usize {
    if let Some(&idx) = self.map.get(&hash) {
      return idx;
    }

    let idx = self.nodes.len();
    let node = Node::new();

    self.nodes.push(node);
    self.map.insert(hash, idx);
    idx
  }

  fn update_node(nodes: &mut [Node<N>], node_idx: usize) {
    let mut sum_values = N::zero();
    let mut sum_visits = 0;

    for edge in nodes[node_idx].children.iter() {
      let child = &nodes[edge.node_idx];
      sum_values = sum_values + N::from(edge.visits).unwrap() * (-child.value);
      sum_visits += edge.visits;
    }

    nodes[node_idx].visits = 1 + sum_visits;
    nodes[node_idx].value = (nodes[node_idx].raw_value + sum_values) / N::from(nodes[node_idx].visits).unwrap();
  }

  fn select_edge(&self, node_idx: usize) -> Option<usize> {
    let node = &self.nodes[node_idx];
    if node.children.is_empty() {
      return None;
    }

    let total_n_sqrt = N::from(node.visits).unwrap().sqrt();

    let mut best_score = -N::infinity();
    let mut best = 0;

    for (idx, edge) in node.children.iter().enumerate() {
      let child = &self.nodes[edge.node_idx];

      // Hyperparameter for PUCT
      let c_puct = N::from(2.5).unwrap();

      // Child value is from child's perspective.
      // Parent wants to maximize own value, which is -child.value
      let q = -child.value;
      let n = edge.visits + edge.virtual_losses;
      let p = edge.prior;

      // PUCT formula
      // Score = Q(a) + C * P(a) * sqrt(sum(N)) / (N(a) + 1)
      let score = q + c_puct * p * total_n_sqrt / N::from(n + 1).unwrap();

      if score > best_score {
        best_score = score;
        best = idx;
      }
    }

    Some(best)
  }

  fn select_path(&mut self) -> Vec<Pos> {
    let mut moves = Vec::new();
    let mut idx = self.root_idx;

    while let Some(edge_idx) = self.select_edge(idx) {
      self.nodes[idx].children[edge_idx].virtual_losses += 1;
      moves.push(self.nodes[idx].children[edge_idx].pos);
      idx = self.nodes[idx].children[edge_idx].node_idx;
    }

    moves
  }

  fn revert_virtual_loss(&mut self, moves: &[Pos]) {
    let mut idx = self.root_idx;
    for &pos in moves {
      let edge_idx = self.nodes[idx]
        .children
        .iter()
        .position(|edge| edge.pos == pos)
        .unwrap();
      self.nodes[idx].children[edge_idx].virtual_losses -= 1;
      idx = self.nodes[idx].children[edge_idx].node_idx;
    }
  }

  fn add_result(&mut self, moves: &[Pos], result: N, children: Vec<Edge<N>>) {
    let mut indices = Vec::new();
    let mut idx = self.root_idx;
    for &pos in moves {
      indices.push(idx);
      let edge_idx = self.nodes[idx]
        .children
        .iter()
        .position(|edge| edge.pos == pos)
        .unwrap();
      self.nodes[idx].children[edge_idx].visits += 1;
      idx = self.nodes[idx].children[edge_idx].node_idx;
    }
    self.nodes[idx].value = result;
    self.nodes[idx].raw_value = result;
    self.nodes[idx].visits = 1;
    self.nodes[idx].children = children;
    while let Some(idx) = indices.pop() {
      Self::update_node(&mut self.nodes, idx);
    }
  }

  const PARALLEL_READOUTS: usize = 8;

  fn make_moves(initial: &Field, moves: &[Pos], mut player: Player) -> Field {
    let mut field = initial.clone();
    for &pos in moves {
      field.put_point(pos, player);
      player = player.next();
    }
    field
  }

  fn create_children(&mut self, field: &mut Field, policy: &ArrayView2<N>) -> Vec<Edge<N>> {
    let stride = field.stride;
    let mut children = Vec::new();

    for pos in field.min_pos()..=field.max_pos() {
      if !field.is_putting_allowed(pos) || field.is_corner(pos) {
        continue;
      }

      field.put_point(pos, field.cur_player());
      let hash = field.hash;
      field.undo();

      let x = to_x(stride, pos);
      let y = to_y(stride, pos);
      let p = policy[(y as usize, x as usize)];

      children.push(Edge {
        pos,
        node_idx: self.add_node(hash),
        visits: 0,
        prior: p,
        virtual_losses: 0,
      });
    }

    // renormalize
    let sum: N = children.iter().map(|child| child.prior).sum();
    for child in children.iter_mut() {
      child.prior = child.prior / sum;
    }

    children
  }

  pub fn mcgs<M: Model<N>>(&mut self, field: &mut Field, player: Player, model: &mut M) -> Result<(), M::E> {
    let mut leafs = iter::repeat_with(|| self.select_path())
      .take(Self::PARALLEL_READOUTS)
      .collect::<Vec<_>>();
    for moves in &leafs {
      self.revert_virtual_loss(moves);
    }

    leafs.sort_unstable();
    leafs.dedup();

    let mut fields = leafs
      .iter()
      .map(|leaf| Self::make_moves(field, leaf, player))
      .collect::<Vec<_>>();

    fields.retain_mut(|cur_field| {
      if cur_field.is_game_over() {
        self.add_result(
          &cur_field.moves[field.moves_count()..],
          Self::game_result(
            cur_field,
            if (cur_field.moves_count() - field.moves_count()).is_multiple_of(2) {
              player
            } else {
              player.next()
            },
          ),
          Vec::new(),
        );
        false
      } else {
        true
      }
    });

    if fields.is_empty() {
      return Ok(());
    }

    let mut features = Vec::with_capacity(field_features_len(field.width(), field.height()) * fields.len());
    for cur_field in &fields {
      field_features_to_vec::<N>(
        cur_field,
        if (cur_field.moves_count() - field.moves_count()).is_multiple_of(2) {
          player
        } else {
          player.next()
        },
        field.width(),
        field.height(),
        0,
        &mut features,
      )
    }
    let features = Array::from_shape_vec(
      (fields.len(), CHANNELS, field.height() as usize, field.width() as usize),
      features,
    )
    .unwrap();

    let (policies, values) = model.predict(features)?;

    for (i, mut cur_field) in fields.into_iter().enumerate() {
      let policy = policies.slice(s![i, .., ..]);
      let value = values[i];
      let children = self.create_children(&mut cur_field, &policy);
      self.add_result(&cur_field.moves[field.moves_count()..], value, children);
    }

    Ok(())
  }

  /// Get the best move based on visit counts
  pub fn best_move(&self) -> Option<NonZeroPos> {
    self.nodes[0]
      .children
      .iter()
      .max_by_key(|edge| edge.visits)
      .and_then(|edge| NonZeroPos::new(edge.pos))
  }
}
