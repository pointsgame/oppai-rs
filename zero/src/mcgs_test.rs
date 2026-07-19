use either::Either;
use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, array};
use oppai_field::construct_field::construct_field;
use oppai_field::field::{Hash, Pos};
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::mcgs::{Edge, Node, Search};

const SEED: u64 = 7;

pub fn uniform_policies(inputs: &Array4<f64>) -> Array3<f64> {
  let batch_size = inputs.len_of(Axis(0));
  let height = inputs.len_of(Axis(2));
  let width = inputs.len_of(Axis(3));
  let policy = 1f64 / (width * height) as f64;
  Array::from_elem((batch_size, height, width), policy)
}

pub fn const_value(inputs: &Array4<f64>, value: Array1<f64>) -> Array2<f64> {
  let batch_size = inputs.len_of(Axis(0));
  value.broadcast((batch_size, value.len())).unwrap().to_owned()
}

#[test]
fn mcts_first_iterations() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ......
    ..aA..
    ......
    ",
  );
  let mut search = Search::<f64>::new(false);

  futures::executor::block_on(search.mcgs(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![1.0, 0.0])));
      result
    },
    0,
    &mut rng,
  ))
  .unwrap();
  assert_eq!(search.nodes[0].visits, 1);
  assert_eq!(search.nodes[0].value, 1.0);
  // corner moves are not considered
  assert_eq!(
    search.nodes[0].children.len(),
    (field.width() * field.height()) as usize - 2
  );
  assert!(
    search.nodes[0]
      .children
      .iter()
      .all(|edge| !search.map.contains_key(&edge.hash))
  );

  futures::executor::block_on(search.mcgs(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.0, 1.0])));
      result
    },
    0,
    &mut rng,
  ))
  .unwrap();
  assert_eq!(search.nodes[0].visits, 9);
  assert_eq!(search.nodes[0].value, 1.0);
  assert_eq!(search.nodes[0].children.iter().map(|edge| edge.visits).sum::<u64>(), 8);
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .flat_map(|edge| search.map.get(&edge.hash))
      .copied()
      .filter(|&edge_idx| search.nodes[edge_idx].children.len() == (field.width() * field.height()) as usize - 3)
      .count(),
    8
  );
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .flat_map(|edge| search.map.get(&edge.hash))
      .copied()
      .filter(|&edge_idx| search.nodes[edge_idx].raw_value == -1.0)
      .count(),
    8
  );
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .flat_map(|edge| search.map.get(&edge.hash))
      .copied()
      .filter(|&edge_idx| search.nodes[edge_idx].value == -1.0)
      .count(),
    8
  );
  // All values backed up through the root equal 1 from its perspective, so
  // the propagated second moment matches the squared value exactly.
  assert_eq!(search.nodes[0].value_sq, 1.0);
}

/// Adds a root child whose node accumulated `visits` observations with mean
/// value `value` and mean squared value `value_sq`.
fn add_root_child(search: &mut Search<f64>, pos: Pos, edge_visits: u64, visits: u64, value: f64, value_sq: f64) {
  let hash = pos as Hash;
  let node_idx = search.nodes.len();
  search.nodes.push(Node {
    visits,
    value,
    raw_value: value,
    value_sq,
    ..Node::new()
  });
  search.map.insert(hash, node_idx);
  search.nodes[search.root_idx].children.push(Edge {
    pos,
    hash,
    visits: edge_visits,
    prior: 0.1,
    virtual_losses: 0,
  });
}

#[test]
fn lcb_prefers_stable_value_over_visits() {
  let mut search = Search::<f64>::new(false);

  // The most visited child is slightly better on average but its backed up
  // values are noisy (all observations are +-1): the 5-stdev confidence radius
  // is ~0.48, giving an LCB of ~-0.18.
  add_root_child(&mut search, 10, 100, 100, -0.3, 1.0);
  // The runner-up has a slightly worse average but zero variance, so its LCB
  // is almost the full 0.25, shaved only by the max-variance prior.
  add_root_child(&mut search, 11, 60, 60, -0.25, 0.0625);
  // A child with a great value but too few visits (below 15% of the leader's
  // 100) is not eligible for LCB selection.
  add_root_child(&mut search, 12, 5, 5, -0.9, 0.81);
  search.nodes[0].visits = 166;

  assert_eq!(search.best_move().map(|pos| pos.get()), Some(11));

  // Every LCB-eligible child is weighted by its LCB, which orders above the
  // visit counts of the ineligible ones.
  let selection = search.play_selection();
  assert!(matches!(selection[0], (10, Either::Right(lcb)) if (-0.18..-0.17).contains(&lcb)));
  assert!(matches!(selection[1], (11, Either::Right(lcb)) if (0.24..0.25).contains(&lcb)));
  assert_eq!(selection[2], (12, Either::Left((5, 0.1))));
}

#[test]
fn lcb_variance_prior_dominates_low_counts() {
  let mut search = Search::<f64>::new(false);

  // With a single observation per child the variance is dominated by the
  // max-variance prior: both confidence radii are equally huge, so the ranking
  // reduces to the values themselves despite the visit difference.
  add_root_child(&mut search, 10, 3, 1, -0.1, 0.01);
  add_root_child(&mut search, 11, 2, 1, -0.9, 0.81);
  search.nodes[0].visits = 6;

  assert_eq!(search.best_move().map(|pos| pos.get()), Some(11));

  let selection = search.play_selection();
  assert!(matches!(selection[0], (10, Either::Right(lcb)) if (-2.41..-2.39).contains(&lcb)));
  assert!(matches!(selection[1], (11, Either::Right(lcb)) if (-1.61..-1.59).contains(&lcb)));
}

#[test]
fn lcb_falls_back_to_most_visited() {
  let mut search = Search::<f64>::new(false);

  // Children that were never expanded into nodes have no value estimate at
  // all, so play selection keeps the visit counts and the most visited child
  // wins with the prior as the tie-breaker.
  for (pos, prior) in [(10, 0.2), (11, 0.3)] {
    search.nodes[0].children.push(Edge {
      pos,
      hash: pos as Hash,
      visits: 1,
      prior,
      virtual_losses: 0,
    });
  }
  search.nodes[0].visits = 3;

  assert_eq!(search.best_move().map(|pos| pos.get()), Some(11));
  assert_eq!(
    search.play_selection(),
    vec![(10, Either::Left((1, 0.2))), (11, Either::Left((1, 0.3)))]
  );
}

#[test]
fn mcts_last_iterations() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .a.
    aAa
    .a.
    ",
  );
  let mut search = Search::<f64>::new(false);

  futures::executor::block_on(search.mcgs(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.5, 0.5])));
      result
    },
    0,
    &mut rng,
  ))
  .unwrap();
  assert_eq!(search.nodes[0].visits, 1);
  assert_eq!(search.nodes[0].value, 1.0);
  assert!(search.nodes[0].children.is_empty());
}

/// A value head whose output grows with the number of stones on the board (the
/// total feature mass), so that a node's raw value systematically disagrees with
/// the deeper (higher stone count) values in its subtree. This drives a nonzero
/// observed bias for the subtree value bias correction to pick up.
fn depth_value(inputs: &Array4<f64>) -> Array2<f64> {
  let batch_size = inputs.len_of(Axis(0));
  let mut value = Array::zeros((batch_size, 2));
  for i in 0..batch_size {
    let mass: f64 = inputs.index_axis(Axis(0), i).sum();
    let p = (mass * 0.02).tanh();
    value[(i, 0)] = (1.0 + p) / 2.0;
    value[(i, 1)] = (1.0 - p) / 2.0;
  }
  value
}

#[test]
fn subtree_value_bias_correction() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .....
    ..aA.
    .Aa..
    .....
    ",
  );
  let mut search = Search::<f64>::new(false);

  for _ in 0..40 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _| {
        let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_value(&inputs)));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }

  // The search built buckets and recorded a genuine observed error in at least
  // one of them (delta_sum is the visit-weighted sum of children-minus-net
  // utility, which is nonzero because the value head is depth dependent).
  assert!(!search.bias.is_empty());
  assert!(search.bias.values().any(|entry| entry.weight_sum > 0.0));
  assert!(search.bias.values().any(|entry| entry.delta_sum.abs() > 1e-6));

  // Every internal node is bucketed and its tracked contribution is consistent
  // with its visit count: ChildVisits(n) = Visits(n) - 1, and the contribution
  // weight is ChildVisits(n)^alpha.
  let alpha = 0.8;
  for node_idx in 0..search.nodes.len() {
    let node = &search.nodes[node_idx];
    if node.children.is_empty() || node.visits <= 1 {
      continue;
    }
    let sum_visits: u64 = node.children.iter().map(|edge| edge.visits).sum();
    assert_eq!(node.visits, 1 + sum_visits, "Visits(n) = 1 + sum of child visits");
    let weight = (sum_visits as f64).powf(alpha);
    assert!(
      (node.last_bias_weight - weight).abs() < 1e-9,
      "tracked bucket weight should be ChildVisits(n)^alpha"
    );
    assert!(node.bias_key.is_some(), "internal node should be bucketed");
  }

  // The incremental bookkeeping is exact: each bucket's accumulated sums equal
  // the sum of its members' currently tracked contributions. This holds
  // regardless of how stale individual node values are.
  let mut delta_by_key: std::collections::HashMap<_, f64> = std::collections::HashMap::new();
  let mut weight_by_key: std::collections::HashMap<_, f64> = std::collections::HashMap::new();
  for node in &search.nodes {
    if let Some(key) = node.bias_key {
      *delta_by_key.entry(key).or_default() += node.last_bias_delta;
      *weight_by_key.entry(key).or_default() += node.last_bias_weight;
    }
  }
  for (key, entry) in &search.bias {
    let delta = delta_by_key.get(key).copied().unwrap_or(0.0);
    let weight = weight_by_key.get(key).copied().unwrap_or(0.0);
    assert!(
      (entry.delta_sum - delta).abs() < 1e-9,
      "bucket delta_sum should equal the sum of member contributions"
    );
    assert!(
      (entry.weight_sum - weight).abs() < 1e-9,
      "bucket weight_sum should equal the sum of member weights"
    );
  }

  // At least one node ends up measurably corrected away from its raw net value.
  let lambda = 0.3;
  let corrected_any = search.nodes.iter().any(|node| {
    node.bias_key.is_some_and(|key| {
      let entry = &search.bias[&key];
      entry.weight_sum > 1e-3 && (lambda * entry.delta_sum / entry.weight_sum).abs() > 1e-6
    })
  });
  assert!(corrected_any, "at least one node should be measurably bias corrected");
}

#[test]
fn subtree_value_bias_survives_compaction() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .....
    ..aA.
    .Aa..
    .....
    ",
  );
  let mut search = Search::<f64>::new(false);
  let mut model = |inputs: Array4<f64>, _| {
    let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_value(&inputs)));
    result
  };

  for _ in 0..30 {
    futures::executor::block_on(search.mcgs(&mut field.clone(), Player::Red, &mut model, 0, &mut rng)).unwrap();
  }
  assert!(!search.bias.is_empty());

  // Move the root to the best child and drop the rest of the tree.
  let pos = search.next_best_root().expect("a move should be available");
  assert!(field.put_point(pos.get(), Player::Red));
  field.update_grounded();
  search.compact();

  // The carried-over buckets must remain finite and non-negative in weight.
  for entry in search.bias.values() {
    assert!(entry.weight_sum >= -1e-9, "bucket weight should not go negative");
    assert!(entry.delta_sum.is_finite() && entry.weight_sum.is_finite());
  }

  // Buckets that no surviving node belongs to are garbage collected: nothing
  // can ever read or update them again.
  let live_keys = search
    .nodes
    .iter()
    .filter_map(|node| node.bias_key)
    .collect::<std::collections::HashSet<_>>();
  for key in search.bias.keys() {
    assert!(live_keys.contains(key), "compaction should drop unreferenced buckets");
  }

  // After compaction the bucket bookkeeping still matches the surviving nodes,
  // plus the residual (1 - free_prop) left behind by dropped nodes. The exact
  // invariant relaxes to: every surviving node's contribution is still present,
  // so the search keeps running consistently.
  for _ in 0..30 {
    futures::executor::block_on(search.mcgs(&mut field.clone(), Player::Black, &mut model, 0, &mut rng)).unwrap();
  }
  for entry in search.bias.values() {
    assert!(entry.delta_sum.is_finite() && entry.weight_sum.is_finite());
  }
}
