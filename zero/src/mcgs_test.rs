use either::Either;
use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, array};
use oppai_field::construct_field::construct_field;
use oppai_field::field::{Hash, Pos};
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::mcgs::{BiasEntry, Edge, Node, Params, Search, t_cdf, value_weight_cdf};
use crate::model::Model;

/// Play's parameters, but without forbidding apriori bad moves: the tests use
/// tiny boards where pruning corners would change the trees they assert on.
const PARAMS: Params = Params {
  forbid_bad: false,
  ..Params::PLAY
};

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
  let mut search = Search::<f64>::new(PARAMS);

  futures::executor::block_on(search.mcgs(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>, _, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![1.0, 0.0, 0.0])));
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
    &mut |inputs: Array4<f64>, _, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.0, 1.0, 0.0])));
      result
    },
    0,
    &mut rng,
  ))
  .unwrap();
  assert_eq!(search.nodes[0].visits, 9);
  // Every child agrees, so the value is 1 up to the rounding that value
  // downweighting's renormalization introduces even when it is a no-op.
  assert!((search.nodes[0].value - 1.0).abs() < 1e-12);
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
  // the propagated second moment matches the squared value.
  assert!((search.nodes[0].value_sq - 1.0).abs() < 1e-12);
}

/// Adds a root child whose node accumulated `visits` observations with mean
/// value `value` and mean squared value `value_sq`. Every observation is given
/// unit weight, so the child's weight and squared-weight totals both equal its
/// visit count and its effective sample size is exactly that count.
fn add_root_child(search: &mut Search<f64>, pos: Pos, edge_visits: u64, visits: u64, value: f64, value_sq: f64) {
  let hash = pos as Hash;
  let node_idx = search.nodes.len();
  search.nodes.push(Node {
    visits,
    own_visits: 1,
    value,
    raw_value: value,
    value_sq,
    weight: 1.0,
    weight_sq: 1.0,
    weight_sum: visits as f64,
    weight_sq_sum: visits as f64,
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
  let mut search = Search::<f64>::new(PARAMS);

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
  // search weights of the ineligible ones.
  let selection = search.play_selection();
  assert!(matches!(selection[0], (10, Either::Right(lcb)) if (-0.18..-0.17).contains(&lcb)));
  assert!(matches!(selection[1], (11, Either::Right(lcb)) if (0.24..0.25).contains(&lcb)));
  assert_eq!(selection[2], (12, Either::Left((5.0, 0.1))));
}

#[test]
fn lcb_variance_prior_dominates_low_counts() {
  let mut search = Search::<f64>::new(PARAMS);

  // Both children's observations agree with themselves, so their entire
  // confidence radius comes from the max-variance prior, which at these counts
  // still outweighs the evidence. The ranking therefore reduces to the values
  // themselves despite the visit difference.
  add_root_child(&mut search, 10, 3, 3, -0.1, 0.01);
  add_root_child(&mut search, 11, 2, 2, -0.9, 0.81);
  search.nodes[0].visits = 6;

  assert_eq!(search.best_move().map(|pos| pos.get()), Some(11));

  let selection = search.play_selection();
  assert!(matches!(selection[0], (10, Either::Right(lcb)) if (-0.43..-0.42).contains(&lcb)));
  assert!(matches!(selection[1], (11, Either::Right(lcb)) if (-0.17..-0.16).contains(&lcb)));
}

#[test]
fn lcb_falls_back_to_the_prior() {
  let mut search = Search::<f64>::new(PARAMS);

  // Children that were never expanded into nodes have no value estimate and no
  // weight at all, so play selection falls back to the priors and the child with
  // the highest one wins.
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
    vec![(10, Either::Left((0.0, 0.2))), (11, Either::Left((0.0, 0.3)))]
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
  let mut search = Search::<f64>::new(PARAMS);

  futures::executor::block_on(search.mcgs(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>, _, _| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.5, 0.5, 0.0])));
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
  let mut value = Array::zeros((batch_size, 3));
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
  let mut search = Search::<f64>::new(PARAMS);

  for _ in 0..40 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _, _| {
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

  // The root is deliberately left out of the table: its bucket could only ever
  // hold itself, so the correction would be its own error fed back to it.
  assert!(
    search.nodes[search.root_idx].bias_key.is_none(),
    "the root should not be bucketed"
  );
  assert_eq!(search.nodes[search.root_idx].last_bias_delta, 0.0);
  assert_eq!(search.nodes[search.root_idx].last_bias_weight, 0.0);

  // Every other internal node is bucketed, and its tracked contribution is
  // consistent with the weight of its children: ChildWeight(n) = W(n) - weight(n),
  // and the contribution weight is ChildWeight(n)^alpha. Both are compared against
  // the node's own stored totals rather than recomputed from its children, since a
  // transposition can update a child after this node was last recomputed.
  //
  // Reading ChildWeight(n) back off `weight_sum` only works while nothing has been
  // pruned from it, which holds here because the policy is uniform: every child's
  // lenient share is then twice the average weight of the children before it, and
  // this search never concentrates that hard. See
  // `the_bias_contribution_ignores_the_pruned_weight` for the general case.
  let alpha = 0.8;
  let mut bucketed = 0;
  for node_idx in 0..search.nodes.len() {
    let node = &search.nodes[node_idx];
    if node.children.is_empty() || node.visits <= 1 {
      continue;
    }
    let sum_visits: u64 = node.children.iter().map(|edge| edge.visits).sum();
    assert_eq!(
      node.visits,
      node.own_visits + sum_visits,
      "Visits(n) = own evaluations + sum of child visits"
    );
    if node_idx == search.root_idx {
      continue;
    }
    let weight = (node.weight_sum - node.weight).powf(alpha);
    assert!(
      (node.last_bias_weight - weight).abs() < 1e-9,
      "tracked bucket weight should be ChildWeight(n)^alpha"
    );
    assert!(node.bias_key.is_some(), "internal node should be bucketed");
    bucketed += 1;
  }
  assert!(bucketed > 0, "the search should have bucketed some internal node");

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
      // A key with no bucket is simply uncorrected: only nodes with children ever
      // create one, so a leaf whose tactic no internal node shares has none.
      search
        .bias
        .get(&key)
        .is_some_and(|entry| entry.weight_sum > 1e-3 && (lambda * entry.delta_sum / entry.weight_sum).abs() > 1e-6)
    })
  });
  assert!(corrected_any, "at least one node should be measurably bias corrected");
}

// The node promoted to root leaves the bias table. Its bucket is keyed on its own
// last move, which nothing in its subtree can replay, so the bucket would hold
// only itself and the correction it read back would be its own observed error fed
// straight into the value the search reports and into its own exploration.
#[test]
fn the_promoted_root_leaves_the_bias_table() {
  let (mut search, _) = run_search(depth_value, 200);
  let pos = search.best_move().expect("the search should have found a move").get();
  let hash = search.nodes[search.root_idx]
    .children
    .iter()
    .find(|edge| edge.pos == pos)
    .expect("the played move should be a root child")
    .hash;
  let idx = search.map[&hash];
  let promoted = &search.nodes[idx];
  let key = promoted
    .bias_key
    .expect("the child about to be promoted should be bucketed");
  let (delta, weight) = (promoted.last_bias_delta, promoted.last_bias_weight);
  assert!(weight > 0.0, "it should have contributed to its bucket first");
  let before = search.bias.get(&key).expect("its bucket should exist").clone();

  assert!(search.next_root(pos));
  assert_eq!(search.root_idx, idx, "the root should be that very node");

  let root = &search.nodes[search.root_idx];
  assert!(root.bias_key.is_none(), "the new root should not be bucketed");
  assert_eq!(root.last_bias_delta, 0.0);
  assert_eq!(root.last_bias_weight, 0.0);

  // Its contribution is released the same way a dropped node's is: decayed by
  // `BIAS_FREE_PROP`, so the bucket keeps the remaining fifth of it rather than
  // either the whole thing or nothing.
  let free_prop = 0.8;
  let after = search.bias.get(&key).expect("the bucket should survive");
  assert!(
    (after.weight_sum - (before.weight_sum - weight * free_prop)).abs() < 1e-9,
    "bucket weight went from {} to {}, releasing {weight}",
    before.weight_sum,
    after.weight_sum
  );
  assert!((after.delta_sum - (before.delta_sum - delta * free_prop)).abs() < 1e-9);

  // And it stays out: further search recomputes the root over and over without it
  // ever rejoining a bucket or contributing to one again.
  search.recompute_stats();
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
  assert!(field.put_point(pos, Player::Red));
  field.update_grounded();
  for _ in 0..20 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Black,
      &mut |inputs: Array4<f64>, _, _| {
        let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_value(&inputs)));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
    let root = &search.nodes[search.root_idx];
    assert!(root.bias_key.is_none(), "the root should stay out of the table");
    assert_eq!(root.last_bias_weight, 0.0, "and keep contributing nothing to one");
  }
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
  let mut search = Search::<f64>::new(PARAMS);
  let mut model = |inputs: Array4<f64>, _, _| {
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

// The PUCT exploration coefficient scales with the node's observed utility
// stdev: volatile nodes explore more, quiet ones less, and an unvisited node
// sits exactly at the neutral factor of 1.
#[test]
fn utility_stdev_scales_exploration() {
  let search = Search::<f64>::new(PARAMS);
  let mut node = Node::<f64>::new();
  assert_eq!(search.utility_stdev_factor(&node), 1.0);

  // A quiet node: every playout agrees on the value, so the observed variance
  // is zero and only the prior keeps the factor above its minimum.
  node.visits = 100;
  node.weight_sum = 100.0;
  node.value = 0.3;
  node.value_sq = 0.3 * 0.3;
  let quiet = search.utility_stdev_factor(&node);
  assert!(quiet < 1.0, "quiet node should explore less, got {}", quiet);

  // A volatile node: values swing between -1 and 1, so the observed stdev is
  // far above the prior.
  node.value = 0.0;
  node.value_sq = 1.0;
  let volatile = search.utility_stdev_factor(&node);
  assert!(volatile > 1.0, "volatile node should explore more, got {}", volatile);
  assert!(volatile > quiet);

  // Self-play leaves the scaling off, pinning the factor at 1 regardless of how
  // volatile the node is.
  let self_play = Search::<f64>::new(Params::SELF_PLAY);
  assert_eq!(self_play.utility_stdev_factor(&node), 1.0);

  // A single confident evaluation already carries more than one unit of weight,
  // so its variance is defined and the factor must be computed rather than
  // falling back to the prior. Guards against reintroducing a `visits <= 1`
  // condition.
  let mut fresh = Node::<f64>::new();
  fresh.visits = 1;
  fresh.weight = 8.0;
  fresh.weight_sum = 8.0;
  fresh.weight_sq_sum = 64.0;
  fresh.value = 0.0;
  fresh.value_sq = 0.0;
  let factor = search.utility_stdev_factor(&fresh);
  // sqrt(((0 + 0.3²) * 2 + 0 * 8) / (2 + 8 - 1)) = 0.1414, so
  // 1 + 0.85 * (0.1414 / 0.3 - 1). With no observed variance the ratio is
  // sqrt(2/9) whatever the prior, so this number does not pin the prior down.
  assert!(
    (factor - 0.550694).abs() < 1e-6,
    "a single weighty evaluation should still scale exploration, got {}",
    factor
  );

  // One unit of weight or less genuinely has no variance to measure.
  fresh.weight_sum = 1.0;
  assert_eq!(search.utility_stdev_factor(&fresh), 1.0);
}

// The exploration coefficient grows with the logarithm of the weight already
// spent below a node. Without that growth the `sqrt(W)` factor falls behind the
// `1 / (1 + W(a))` denominator as a node is searched harder, and the node narrows
// onto its current best child sooner than it should - the deeper the search, the
// worse the shortfall.
#[test]
fn exploration_coefficient_grows_with_the_search() {
  let search = Search::<f64>::new(PARAMS);
  // A fresh node has no variance to measure, so the utility-stdev factor is 1 and
  // dividing out the sqrt leaves exactly the cpuct coefficient.
  let node = Node::<f64>::new();
  let cpuct = |weight: f64| search.explore_scaling(weight, &node) / (weight + 0.01).sqrt();

  assert!((cpuct(0.0) - PARAMS.cpuct_exploration).abs() < 1e-12);
  // The base is 500, so that much child weight grows the log term by ln 2.
  let at_base = PARAMS.cpuct_exploration + PARAMS.cpuct_exploration_log * 2f64.ln();
  assert!((cpuct(500.0) - at_base).abs() < 1e-12, "got {}", cpuct(500.0));
  assert!(
    cpuct(5000.0) > cpuct(500.0) && cpuct(500.0) > cpuct(0.0),
    "the coefficient should keep growing, got {} then {} then {}",
    cpuct(0.0),
    cpuct(500.0),
    cpuct(5000.0)
  );

  // Both profiles grow it; self-play just does so more slowly.
  let self_play = Search::<f64>::new(Params::SELF_PLAY);
  let self_play_cpuct = |weight: f64| self_play.explore_scaling(weight, &node) / (weight + 0.01).sqrt();
  assert!(self_play_cpuct(5000.0) > self_play_cpuct(0.0));
}

// Certain evaluations weigh up to the maximum, uncertain ones weigh less, and
// the weight decreases monotonically with the predicted error.
#[test]
fn uncertainty_weight_scales_with_error() {
  let coeff = Search::<f64>::UNCERTAINTY_COEFF;
  let max_weight = Search::<f64>::UNCERTAINTY_MAX_WEIGHT;

  let max = Search::<f64>::uncertainty_weight(0.0);
  assert!(
    (max - max_weight).abs() < 1e-9,
    "zero error should give the maximum weight"
  );

  // Where the curve sits is set by the coefficient, which the maximum above
  // cannot pin - `coeff / (0 + coeff / 8)` is 8 for any coefficient. An
  // evaluation whose predicted error equals the coefficient weighs
  // `1 / (1 + 1/8)`, and that is the point of the coefficient: it is meant to
  // be the error a typical evaluation predicts, so a typical playout weighs
  // about one and the weight behind a node tracks its playout count.
  let at_coeff = Search::<f64>::uncertainty_weight(coeff);
  assert!(
    (at_coeff - max_weight / (max_weight + 1.0)).abs() < 1e-9,
    "an error equal to the coefficient should weigh 8/9, got {}",
    at_coeff
  );

  let confident = Search::<f64>::uncertainty_weight(0.2 * coeff);
  let uncertain = Search::<f64>::uncertainty_weight(2.0 * coeff);
  assert!(max > confident && confident > uncertain);
  assert!(confident > 1.0 && uncertain < 1.0);
}

/// A value head that reports a predicted error shrinking with the number of
/// stones on the board, so that evaluations in the same tree carry genuinely
/// different weights.
fn depth_uncertainty_value(inputs: &Array4<f64>) -> Array2<f64> {
  let mut value = depth_value(inputs);
  for i in 0..inputs.len_of(Axis(0)) {
    let mass: f64 = inputs.index_axis(Axis(0), i).sum();
    value[(i, 2)] = 0.5 / (1.0 + mass * 0.05);
  }
  value
}

// Every evaluation's uncertainty weight reaches the tree: when the net reports
// the same error everywhere, each node's own weight is that error's weight and
// the total weight is the exact weighted analog of the visit count.
#[test]
fn uncertainty_weights_propagate_through_the_tree() {
  for error in [0.0, 0.1, 0.6] {
    let expected = Search::<f64>::uncertainty_weight(error);
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
    let mut search = Search::<f64>::new(PARAMS);

    for _ in 0..30 {
      futures::executor::block_on(search.mcgs(
        &mut field,
        Player::Red,
        &mut |inputs: Array4<f64>, _, _| {
          let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.6, 0.4, error])));
          result
        },
        0,
        &mut rng,
      ))
      .unwrap();
    }

    for node in &search.nodes {
      if node.visits == 0 {
        continue;
      }
      // A terminal node would carry the maximum weight instead; the board is
      // far from full, so the tree should not contain any.
      assert!(!node.children.is_empty(), "no terminal nodes expected in this tree");
      assert!(
        (node.weight - expected).abs() < 1e-9,
        "node weight should be the reported error's weight, got {} instead of {}",
        node.weight,
        expected
      );
      // W(n) = weight(n) + sum_c W(c) * Visits(edge c) / Visits(c), which with a
      // uniform weight per evaluation collapses to weight * Visits(n).
      assert!(
        (node.weight_sum - expected * node.visits as f64).abs() < 1e-9,
        "total weight should be the weighted analog of the visit count, got {} instead of {}",
        node.weight_sum,
        expected * node.visits as f64
      );
    }
  }
}

// With a net whose confidence varies across the board the weights genuinely
// differ between nodes, and they all stay within the range the formula allows.
#[test]
fn uncertainty_weights_vary_with_the_predicted_error() {
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
  let mut search = Search::<f64>::new(PARAMS);

  for _ in 0..30 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _, _| {
        let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_uncertainty_value(&inputs)));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }

  let weights = search
    .nodes
    .iter()
    .filter(|node| node.visits > 0)
    .map(|node| node.weight)
    .collect::<Vec<_>>();
  assert!(weights.len() > 1);
  let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
  let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
  assert!(max - min > 1e-6, "weights should differ between nodes");
  // The weight of any evaluation stays between the most uncertain one the model
  // can report here and the maximum for a perfectly certain one.
  assert!(min >= Search::<f64>::uncertainty_weight(0.5) - 1e-9);
  assert!(max <= Search::<f64>::uncertainty_weight(0.0) + 1e-9);

  // The root's total weight is its own weight plus its children's, distributed
  // across the edges by visits. Only the root is checked: it lies on every
  // playout, so it is recomputed last and is always consistent with its
  // children, whereas a deeper node goes stale as soon as a transposition
  // updates one of its children through another path.
  let root = &search.nodes[search.root_idx];
  let children_weight = root
    .children
    .iter()
    .filter(|edge| edge.visits > 0)
    .filter_map(|edge| search.map.get(&edge.hash).map(|&idx| (edge, &search.nodes[idx])))
    .map(|(edge, child)| child.weight_sum * edge.visits as f64 / child.visits as f64)
    .sum::<f64>();
  assert!(
    (root.weight_sum - root.weight - children_weight).abs() < 1e-9,
    "total weight should be the node's own weight plus its children's"
  );
}

/// Runs a search to completion on a fixed position with the given value head.
fn run_search(value: impl Fn(&Array4<f64>) -> Array2<f64>, iterations: usize) -> (Search<f64>, Xoshiro256PlusPlus) {
  run_search_with(PARAMS, value, iterations)
}

fn run_search_with(
  params: Params,
  value: impl Fn(&Array4<f64>) -> Array2<f64>,
  iterations: usize,
) -> (Search<f64>, Xoshiro256PlusPlus) {
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
  let mut search = Search::<f64>::new(params);

  for _ in 0..iterations {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _, _| {
        let result: Result<_, ()> = Ok((uniform_policies(&inputs), value(&inputs)));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }

  (search, rng)
}

// The policy target records the search weight behind each move rather than the
// number of playouts. Uncertainty weighting makes PUCT equalize weight, so
// visits pile up behind the moves the net is least sure about; recording weights
// is what keeps that from becoming the training signal.
#[test]
fn policy_target_is_weight_based() {
  let (search, _) = run_search(depth_uncertainty_value, 40);

  let root = &search.nodes[search.root_idx];
  let visits = root
    .children
    .iter()
    .filter(|edge| edge.visits > 0)
    .map(|edge| (edge.pos, edge.visits))
    .collect::<std::collections::HashMap<_, _>>();
  let weights = search.weights().collect::<Vec<_>>();
  assert!(weights.len() > 1, "expected several searched moves");

  // Every recorded weight is the child's weight, which with a confidence that
  // varies across the board is not proportional to the edge's visits.
  let total_weight = weights.iter().map(|&(_, w)| w).sum::<f64>();
  let total_visits = visits.values().sum::<u64>() as f64;
  let diverges = weights.iter().any(|&(pos, weight)| {
    let visit_share = visits[&pos] as f64 / total_visits;
    (weight / total_weight - visit_share).abs() > 1e-6
  });
  assert!(
    diverges,
    "weight and visit distributions should differ when confidence varies"
  );

  // A uniformly confident net weighs every playout the same, so then the two
  // distributions must coincide.
  let (search, _) = run_search(depth_value, 40);
  let weights = search.weights().collect::<Vec<_>>();
  let total_weight = weights.iter().map(|&(_, w)| w).sum::<f64>();
  let root = &search.nodes[search.root_idx];
  let total_visits = root.children.iter().map(|edge| edge.visits).sum::<u64>() as f64;
  for &(pos, weight) in &weights {
    let edge = root.children.iter().find(|edge| edge.pos == pos).unwrap();
    assert!(
      (weight / total_weight - edge.visits as f64 / total_visits).abs() < 1e-9,
      "with equal weights the target should match the visit distribution"
    );
  }
}

// The squared-weight sum feeding the LCB's effective sample size accumulates
// correctly. The edge's share of its child enters this recurrence *squared*,
// unlike the linear share in `child_weight_sq`: taking a fraction of a child's
// playouts scales each of their weights by that fraction, so the squares scale by
// its square. Getting the two confused would silently mis-scale every confidence
// radius, so the recurrence is checked against an independent recomputation.
//
// Only the root is checked: it lies on every playout and so is recomputed last,
// whereas a deeper node's totals go stale as soon as a transposition updates one
// of its children through another path.
//
// Note that the resulting effective sample size is not bounded by the visit
// count. Transpositions make an edge account for only part of its child's
// visits, and the squared-weight sum shrinks with the square of that share while
// the weight sum shrinks linearly, so shared evaluations inflate the ratio.
#[test]
fn squared_weight_sum_accumulates_over_the_children() {
  // A parent reaching a child through an edge that covers half of the child's
  // visits: the child's four unit-weight playouts are halved to a weight of 2,
  // and their squared weights of 4 are quartered to 1. A linear share would
  // leave 2 instead and understate every confidence radius built on it.
  let mut search = Search::<f64>::new(PARAMS);
  let child_idx = search.nodes.len();
  search.nodes.push(Node {
    visits: 4,
    value: 0.5,
    raw_value: 0.5,
    value_sq: 0.25,
    weight: 1.0,
    weight_sum: 4.0,
    weight_sq_sum: 4.0,
    ..Node::new()
  });
  search.map.insert(1 as Hash, child_idx);
  let root = &mut search.nodes[search.root_idx];
  root.own_visits = 1;
  root.weight = 3.0;
  root.weight_sq = 9.0;
  root.raw_value = 0.0;
  root.children.push(Edge {
    pos: 10,
    hash: 1 as Hash,
    visits: 2,
    prior: 1.0,
    virtual_losses: 0,
  });

  let Search { map, nodes, bias, .. } = &mut search;
  // A single child means both reweightings are no-ops by construction, so this
  // isolates the weight bookkeeping.
  Search::update_node(map, nodes, bias, 0, PARAMS, &mut Vec::new());

  let root = &search.nodes[search.root_idx];
  assert_eq!(root.visits, 3, "1 own visit plus the edge's 2");
  assert_eq!(root.weight_sum, 3.0 + 2.0, "own weight plus half the child's");
  assert_eq!(
    root.weight_sq_sum,
    9.0 + 1.0,
    "own squared weight plus a quarter of the child's"
  );

  // Unequal weights across the tree change how much independent evidence the
  // playouts carry, so the effective sample size parts ways with the
  // uniform-weight case.
  let root_ess = |value: &dyn Fn(&Array4<f64>) -> Array2<f64>| {
    let (search, _) = run_search(value, 40);
    let root = &search.nodes[search.root_idx];
    root.weight_sum * root.weight_sum / root.weight_sq_sum
  };
  assert!((root_ess(&depth_uncertainty_value) - root_ess(&depth_value)).abs() > 1e-6);
}

/// Runs a search on a board crowded enough that playouts keep running into
/// finished positions, so the tree holds childless nodes that every playout
/// reaching them evaluates again.
fn run_search_with_terminals(params: Params, iterations: usize) -> Search<f64> {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    aAa
    Aa.
    .A.
    ",
  );
  let mut search = Search::<f64>::new(params);
  for _ in 0..iterations {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _, _| {
        let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_uncertainty_value(&inputs)));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }
  search
}

// A node can be evaluated more than once - a childless one is reached as a leaf
// by every playout that descends to it, and two paths of one batch can transpose
// onto the same fresh node - and each of those is a playout's worth of evidence
// that has to accumulate. Overwriting instead pins the node at one evaluation
// while its parent edges keep counting, which does not show up in `weight_sum`
// (the edge's share scales linearly, so it cancels) but squares away in
// `weight_sq_sum`, collapsing the effective sample size the LCB divides its
// variance by and blowing up every confidence radius in the region.
#[test]
fn re_evaluated_nodes_accumulate_their_playouts() {
  for params in [Params::SELF_PLAY, PARAMS] {
    let search = run_search_with_terminals(
      Params {
        forbid_bad: false,
        ..params
      },
      200,
    );

    // Some node really was evaluated more than once, or this proves nothing.
    assert!(
      search.nodes.iter().any(|node| node.own_visits > 1),
      "the crowded board should have left a node to re-evaluate"
    );

    for (idx, node) in search.nodes.iter().enumerate() {
      if node.visits == 0 {
        continue;
      }
      let edge_visits = node.children.iter().map(|edge| edge.visits).sum::<u64>();
      assert_eq!(
        node.visits,
        node.own_visits + edge_visits,
        "node {idx} should count its own evaluations plus its edges'"
      );
      // No edge may claim more visits than the node it points at has, or the
      // edge's `edge_visits / child.visits` share of the child's totals exceeds
      // the whole and the weight gets counted more than once.
      for edge in &node.children {
        if let Some(&child_idx) = search.map.get(&edge.hash) {
          let child = &search.nodes[child_idx];
          if child.visits > 0 {
            assert!(
              edge.visits <= child.visits,
              "edge of node {idx} claims {} of the child's {} visits",
              edge.visits,
              child.visits
            );
          }
        }
      }
    }
  }
}

// With every playout weighing the same, the root's effective sample size has to
// come back out as the number of playouts. It is the sharpest statement of the
// bookkeeping above: a single re-evaluated terminal is enough to drag it from
// the full playout count down to nearly one.
#[test]
fn effective_sample_size_tracks_the_playout_count() {
  let search = run_search_with_terminals(
    Params {
      forbid_bad: false,
      ..Params::SELF_PLAY
    },
    200,
  );
  let root = &search.nodes[search.root_idx];
  let ess = root.weight_sum * root.weight_sum / root.weight_sq_sum;
  assert_eq!(root.weight_sum, root.visits as f64);
  // Value downweighting redistributes weight between siblings, which costs a
  // little effective sample size, so this comes out close to the playout count
  // rather than exactly at it.
  assert!(
    ess > 0.9 * root.visits as f64,
    "effective sample size {ess} should be within a tenth of the {} playouts",
    root.visits
  );
}

// A freshly evaluated leaf records the square of its own weight. This is easy
// to miss because only ancestors are recomputed on the way back up, and nothing
// else would ever fill it in: a leaf keeps its totals until a playout goes
// through it. Left at zero the whole frontier contributes nothing to its
// ancestors' squared-weight sums, which inflates every effective sample size
// and so shrinks every LCB radius - the opposite of what the bound is for.
#[test]
fn leaf_records_its_squared_weight() {
  // One playout leaves the root as a leaf: no child has backed up into it, so its
  // totals are exactly its own evaluation's.
  let (search, _) = run_search(depth_uncertainty_value, 1);
  let root = &search.nodes[search.root_idx];
  assert_eq!(root.visits, 1);
  assert!(root.weight > 1.0, "the net should report a weight above one playout's");
  assert_eq!(root.weight_sum, root.weight);
  assert_eq!(root.weight_sq_sum, root.weight * root.weight);

  // The same holds for every node still on the frontier of a finished search, and
  // no evaluated node anywhere is left without a squared-weight sum.
  let (search, _) = run_search(depth_uncertainty_value, 40);
  let mut leafs = 0;
  for node in &search.nodes {
    if node.visits == 0 {
      continue;
    }
    assert!(
      node.weight_sq_sum > 0.0,
      "an evaluated node needs a squared-weight sum to size its confidence radius"
    );
    if node.visits == 1 {
      assert_eq!(node.weight_sq_sum, node.weight * node.weight);
      leafs += 1;
    }
  }
  assert!(leafs > 0, "the search should have left a frontier to check");

  // With uncertainty weighting off every playout weighs one, so no node's
  // squared-weight sum can fall below that single square.
  let (search, _) = run_search_with(Params::SELF_PLAY, depth_uncertainty_value, 40);
  for node in &search.nodes {
    if node.visits == 0 {
      continue;
    }
    assert!(
      node.weight_sq_sum >= 1.0,
      "unit-weight playouts should each contribute a square of one, got {}",
      node.weight_sq_sum
    );
  }
}

// Self-play leaves uncertainty weighting off: every playout counts once no
// matter how unsure the net says it is. The weight-based policy target then
// collapses back onto the visit distribution, so recording weights costs
// self-play nothing while staying correct for the profiles that do weigh
// evaluations.
#[test]
fn self_play_counts_every_playout_once() {
  // A net whose reported error varies across the board - which self-play must
  // ignore entirely.
  let (search, _) = run_search_with(Params::SELF_PLAY, depth_uncertainty_value, 40);

  for node in &search.nodes {
    if node.visits == 0 {
      continue;
    }
    assert_eq!(
      node.weight, node.own_visits as f64,
      "every evaluation should count once, however many the node had"
    );
    // W(n) = 1 * Visits(n) once every playout weighs the same.
    assert!(
      (node.weight_sum - node.visits as f64).abs() < 1e-9,
      "total weight should equal the visit count, got {} vs {}",
      node.weight_sum,
      node.visits
    );
  }

  let root = &search.nodes[search.root_idx];
  let total_visits = root.children.iter().map(|edge| edge.visits).sum::<u64>() as f64;
  let weights = search.weights().collect::<Vec<_>>();
  let total_weight = weights.iter().map(|&(_, w)| w).sum::<f64>();
  assert!(weights.len() > 1);
  for &(pos, weight) in &weights {
    let edge = root.children.iter().find(|edge| edge.pos == pos).unwrap();
    assert!(
      (weight / total_weight - edge.visits as f64 / total_visits).abs() < 1e-9,
      "the target should match the visit distribution when playouts weigh the same"
    );
  }

  // The same net under the play profile does weigh evaluations unequally, so the
  // two profiles genuinely differ rather than both being off.
  let (play, _) = run_search_with(PARAMS, depth_uncertainty_value, 40);
  assert!(
    play.nodes.iter().any(|node| node.visits > 0 && node.weight != 1.0),
    "the play profile should still weigh evaluations by their predicted error"
  );
}

/// Adds a root child with an arbitrary weight, decoupled from its visit count so
/// that the heaviest child and the most stably explored one can differ.
fn add_weighted_root_child(search: &mut Search<f64>, pos: Pos, visits: u64, weight: f64, prior: f64) {
  let hash = pos as Hash;
  let node_idx = search.nodes.len();
  search.nodes.push(Node {
    visits,
    own_visits: 1,
    value: -0.5,
    raw_value: -0.5,
    value_sq: 0.25,
    weight: weight / visits as f64,
    weight_sq: weight * weight / (visits * visits) as f64,
    weight_sum: weight,
    weight_sq_sum: weight * weight / visits as f64,
    ..Node::new()
  });
  search.map.insert(hash, node_idx);
  search.nodes[search.root_idx].children.push(Edge {
    pos,
    hash,
    visits,
    prior,
    virtual_losses: 0,
  });
}

// Policy target pruning and LCB eligibility measure against the most stably
// explored child, not simply the heaviest one. A child with a single visit
// contributes none of its weight to its own stability, because that one visit
// could have been overweighted.
#[test]
fn reference_child_is_the_stably_explored_one() {
  let mut search = Search::<f64>::new(PARAMS);
  // Heaviest, but on the strength of a single playout.
  add_weighted_root_child(&mut search, 10, 1, 10.0, 0.01);
  // Lighter, but explored steadily.
  add_weighted_root_child(&mut search, 11, 9, 9.0, 0.01);
  search.nodes[0].visits = 11;

  let (idx, weight) = search.reference_child().unwrap();
  assert_eq!(idx, 1, "the steadily explored child should be the reference");
  assert_eq!(weight, 9.0);

  // Without the one-visit discount the heaviest child would win, so this really
  // is the adjustment being exercised rather than a tie.
  let heaviest = search.nodes[0]
    .children
    .iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| {
      let wa = search.nodes[search.map[&a.hash]].weight_sum;
      let wb = search.nodes[search.map[&b.hash]].weight_sum;
      wa.partial_cmp(&wb).unwrap()
    })
    .map(|(idx, _)| idx);
  assert_eq!(heaviest, Some(0), "the heaviest child is a different one");
}

// Only explored children may be the reference. The root holds an edge per legal
// move from the moment it is expanded, and a child with a single visit has its
// whole weight discounted away, so an unexplored edge with a higher prior can
// out-score it. Letting one win would set the reference weight to zero, which
// empties the policy target and leaves self-play training on a uniform
// distribution over the whole board.
#[test]
fn reference_child_ignores_unexplored_edges() {
  let mut search = Search::<f64>::new(PARAMS);
  // Explored, but on the strength of a single playout, so the one-visit discount
  // leaves its stability resting on its low prior alone.
  add_weighted_root_child(&mut search, 10, 1, 10.0, 0.01);
  // Never explored: no weight at all, but enough prior to out-score that.
  search.nodes[0].children.push(Edge {
    pos: 11,
    hash: 11 as Hash,
    visits: 0,
    prior: 0.5,
    virtual_losses: 0,
  });
  search.nodes[0].visits = 2;

  let (idx, weight) = search.reference_child().unwrap();
  assert_eq!(idx, 0, "an edge carrying no weight cannot be the reference child");
  assert_eq!(weight, 10.0);

  // So the policy target keeps the one move the search actually explored.
  assert_eq!(search.pruned_weights().collect::<Vec<_>>(), vec![(10, 10.0)]);

  // With nothing explored there is no reference at all, and no target to record.
  let mut search = Search::<f64>::new(PARAMS);
  search.nodes[0].children.push(Edge {
    pos: 11,
    hash: 11 as Hash,
    visits: 0,
    prior: 0.5,
    virtual_losses: 0,
  });
  assert_eq!(search.reference_child(), None);
  assert!(search.pruned_weights().next().is_none());
}

// What leaves the policy target is decided by the reduction, not by a floor on
// the weight: rounding up means a child PUCT would have visited at all keeps a
// whole playout's worth of target however little the search actually left it,
// and only the children PUCT would not have touched reach zero and are dropped.
//
// Without the rounding, `CHOSEN_MOVE_PRUNE` would instead read as "less than one
// playout's worth of weight", which deletes the entire thin tail of the target -
// the moves the search looked at once and rejected, which is exactly the part
// the net has nothing else to learn a nonzero probability from.
#[test]
fn pruning_drops_only_what_puct_would_not_have_visited() {
  let mut search = Search::<f64>::new(PARAMS);
  // The leader, which is also the reference the reduction measures against and
  // so keeps its weight exactly.
  add_weighted_root_child(&mut search, 10, 200, 200.0, 0.05);
  // Close enough behind that PUCT would still have spent a little on it, but far
  // less than the eight it was given.
  add_weighted_root_child(&mut search, 11, 8, 8.0, 0.05);
  search.nodes[2].value = -0.35;
  search.nodes[2].raw_value = -0.35;
  // Hopeless, and with a prior too small for PUCT to have looked at it at all.
  add_weighted_root_child(&mut search, 12, 8, 8.0, 0.001);
  search.nodes[3].value = 0.4;
  search.nodes[3].raw_value = 0.4;
  search.nodes[0].visits = 217;
  search.nodes[0].weight = 1.0;
  search.nodes[0].weight_sum = 217.0;

  let target = search.pruned_weights().collect::<Vec<_>>();
  let weight_of = |pos: Pos| target.iter().find(|&&(p, _)| p == pos).map(|&(_, w)| w);

  assert!(
    weight_of(10).is_some_and(|weight| weight >= 200.0),
    "the leader must stay in the target, got {target:?}"
  );
  assert_eq!(
    weight_of(11),
    Some(1.0),
    "a child PUCT wanted a fraction of should keep a whole playout, got {target:?}"
  );
  assert_eq!(
    weight_of(12),
    None,
    "a child PUCT would not have visited should be dropped, got {target:?}"
  );
}

// The children's total weight is summed from the children, never read from the
// parent's cached `weight_sum - weight`. That cache is only correct as of the
// last time this node was recomputed, and a transposition can update one of its
// children through another path without touching it.
#[test]
fn total_child_weight_is_summed_fresh() {
  let mut search = Search::<f64>::new(PARAMS);
  add_weighted_root_child(&mut search, 10, 40, 40.0, 0.5);
  add_weighted_root_child(&mut search, 11, 8, 8.0, 0.3);
  // A deliberately stale cache, as if a transposition had grown a child since.
  search.nodes[0].visits = 49;
  search.nodes[0].weight = 1.0;
  search.nodes[0].weight_sum = 500.0;

  let root = &search.nodes[search.root_idx];
  assert_eq!(
    search.total_child_weight(root),
    48.0,
    "the cached total claims 499 of child weight; the children hold 48"
  );
}

// A child whose value sits well below its siblings is downweighted before being
// averaged into the parent, so the parent's value stays closer to the good lines
// than a plain weighted mean would put it. The weights are renormalized, so the
// node's total weight is unchanged - only the distribution across children.
#[test]
fn bad_children_are_downweighted_before_averaging() {
  let build = |exponent: f64| {
    let params = Params {
      value_weight_exponent: exponent,
      // Isolated from the noise pruning, which is a no-op here anyway: the three
      // children share a prior, so none of them holds more than the lenient
      // share of the weight of the ones before it.
      noise_prune_utility_scale: 0.0,
      ..PARAMS
    };
    let mut search = Search::<f64>::new(params);
    // Three equally searched children: two agree the position is fine for the
    // parent, one is a disaster. Child values are from the child's perspective,
    // so -0.5 is +0.5 for the parent.
    add_weighted_root_child(&mut search, 10, 20, 20.0, 0.3);
    add_weighted_root_child(&mut search, 11, 20, 20.0, 0.3);
    add_weighted_root_child(&mut search, 12, 20, 20.0, 0.3);
    search.nodes[3].value = 0.9;
    search.nodes[3].raw_value = 0.9;
    search.nodes[3].value_sq = 0.81;
    search.nodes[search.root_idx].weight = 1.0;
    search.nodes[search.root_idx].raw_value = 0.0;
    let Search { map, nodes, bias, .. } = &mut search;
    Search::update_node(map, nodes, bias, 0, params, &mut Vec::new());
    (search.nodes[0].value, search.nodes[0].weight_sum)
  };

  // The plain weighted mean: (0 * 1 + 0.5 * 20 + 0.5 * 20 - 0.9 * 20) / 61.
  let (plain, plain_total) = build(0.0);
  assert!((plain - (0.5 * 20.0 + 0.5 * 20.0 - 0.9 * 20.0) / 61.0).abs() < 1e-9);

  let (downweighted, total) = build(0.5);
  assert!(
    downweighted > plain + 1e-6,
    "the bad child should pull the value down less, got {} vs {}",
    downweighted,
    plain
  );
  // Renormalization keeps the node's total weight identical.
  assert!(
    (total - plain_total).abs() < 1e-9,
    "total weight should be unchanged, got {} vs {}",
    total,
    plain_total
  );
  assert!((total - 61.0).abs() < 1e-9);
}

// Weight that exploration piled onto a move the policy ranked low, and that the
// search then found to be worse than the moves it ranked high, is discarded
// rather than redistributed: it says what that move is worth, not what the
// position is. So unlike the value downweighting, this lowers the node's total.
#[test]
fn refuted_low_policy_children_lose_their_excess_weight() {
  // Two equally searched children. The one the policy likes is good for the
  // parent - child values are from the child's perspective, so -0.5 is +0.5 -
  // and the other is bad. `swap` inserts them the other way round: the pruning
  // has to judge them in policy order, not in the order the edges happen to sit
  // in, which is a shuffle of the board.
  let build = |scale: f64, refuted_prior: f64, swap: bool| {
    let params = Params {
      // Isolated from the value downweighting, which reshuffles the same weights
      // among the same children rather than dropping any.
      value_weight_exponent: 0.0,
      noise_prune_utility_scale: scale,
      ..PARAMS
    };
    let mut search = Search::<f64>::new(params);
    let add = |search: &mut Search<f64>, pos: Pos, value: f64, prior: f64| {
      add_weighted_root_child(search, pos, 20, 20.0, prior);
      let idx = search.nodes.len() - 1;
      search.nodes[idx].value = value;
      search.nodes[idx].raw_value = value;
      search.nodes[idx].value_sq = value * value;
    };
    let good = (10, -0.5, 0.9);
    let bad = (11, 0.4, refuted_prior);
    for (pos, value, prior) in if swap { [bad, good] } else { [good, bad] } {
      add(&mut search, pos, value, prior);
    }
    search.nodes[search.root_idx].weight = 1.0;
    search.nodes[search.root_idx].raw_value = 0.0;
    let Search { map, nodes, bias, .. } = &mut search;
    Search::update_node(map, nodes, bias, 0, params, &mut Vec::new());
    (search.nodes[0].value, search.nodes[0].weight_sum)
  };

  // The plain weighted mean over the root's own evaluation and both children.
  let (plain, plain_total) = build(0.0, 0.1, false);
  assert!((plain - (0.5 * 20.0 - 0.4 * 20.0) / 41.0).abs() < 1e-9);
  assert!((plain_total - 41.0).abs() < 1e-9);

  // A ninth of the policy but half of the weight, and refuted by 0.9 of utility:
  // all but 2 * 20 * (0.1 / 0.9) of its weight goes, up to the `exp(-0.9 / 0.15)`
  // of the excess that six scale lengths of doubt leave behind.
  let excess = 20.0 - 2.0 * 20.0 * (0.1 / 0.9);
  let expected = 41.0 - excess * (1.0 - (-0.9f64 / 0.15).exp());
  let (pruned, total) = build(0.15, 0.1, false);
  assert!(
    (total - expected).abs() < 1e-9,
    "the excess weight should be gone, got {total} vs {expected}"
  );
  assert!(
    pruned > plain + 1e-6,
    "dropping the refuted child's weight should lift the value, got {pruned} vs {plain}"
  );

  // Same two children in the other order in the edge list.
  let (swapped, swapped_total) = build(0.15, 0.1, true);
  assert!((swapped - pruned).abs() < 1e-9, "policy order decides, not edge order");
  assert!((swapped_total - total).abs() < 1e-9);

  // A child that is just as refuted but holds no more than its policy share of
  // the weight keeps all of it: there is nothing exploration overspent here.
  let (_, unpruned_total) = build(0.15, 0.9, false);
  assert!(
    (unpruned_total - 41.0).abs() < 1e-9,
    "a child within its policy share should keep its weight, got {unpruned_total}"
  );
}

// The weight a node adds to its bias bucket says how much search stands behind
// the error that node observed, and the playouts the pruning discards were still
// spent looking at the position. So the bucket weighs the node by the search it
// did, not by the part of it the value ends up trusting - the pruning moves the
// observed error, not the confidence in it.
#[test]
fn the_bias_contribution_ignores_the_pruned_weight() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    .....
    ..aA.
    .Aa..
    .....
    ",
  );
  // Any real bucket will do; the node is bucketed by hand because the root, which
  // this exercises `update_node` on, is deliberately left out of the table.
  let key = Search::<f64>::bias_key(&field).unwrap();

  // The same two children as above: the one the policy likes is good for the
  // parent, the low-prior one holds half the weight and is refuted.
  let build = |scale: f64| {
    let params = Params {
      value_weight_exponent: 0.0,
      noise_prune_utility_scale: scale,
      ..PARAMS
    };
    let mut search = Search::<f64>::new(params);
    for (pos, value, prior) in [(10, -0.5, 0.9), (11, 0.4, 0.1)] {
      add_weighted_root_child(&mut search, pos, 20, 20.0, prior);
      let idx = search.nodes.len() - 1;
      search.nodes[idx].value = value;
      search.nodes[idx].raw_value = value;
      search.nodes[idx].value_sq = value * value;
    }
    let root = &mut search.nodes[search.root_idx];
    root.weight = 1.0;
    root.raw_value = 0.0;
    root.bias_key = Some(key);
    let Search { map, nodes, bias, .. } = &mut search;
    Search::update_node(map, nodes, bias, 0, params, &mut Vec::new());
    let root = &search.nodes[0];
    (
      root.weight_sum,
      root.last_bias_weight,
      root.last_bias_delta,
      search.bias[&key].clone(),
    )
  };

  let (plain_total, plain_weight, plain_delta, plain_entry) = build(0.0);
  let (total, weight, delta, entry) = build(0.15);

  // The pruning did fire, so the two runs really do differ in the node's total.
  assert!(
    total < plain_total - 1e-6,
    "the refuted child's excess weight should be gone, got {total} vs {plain_total}"
  );

  // Both runs weigh the node by the 40 of child weight the search actually spent.
  let expected = 40f64.powf(0.8);
  assert!(
    (plain_weight - expected).abs() < 1e-9 && (weight - expected).abs() < 1e-9,
    "the bucket weight should be ChildWeight(n)^alpha before pruning, got {weight} vs {plain_weight}"
  );
  assert!((entry.weight_sum - expected).abs() < 1e-9);
  assert!((plain_entry.weight_sum - expected).abs() < 1e-9);

  // The observed error, on the other hand, is the one the pruning arrived at: it
  // is the children's utility that the node's value is built from.
  assert!(
    delta > plain_delta + 1e-6,
    "dropping the refuted child's weight should lift the observed error, got {delta} vs {plain_delta}"
  );
  assert!((entry.delta_sum - delta).abs() < 1e-9);
  let children_utility = (0.5 * 20.0 - 0.4 * (total - 21.0)) / (total - 1.0);
  assert!((delta - children_utility * expected).abs() < 1e-9);
}

// The recorded policy target promotes the child with the best lower confidence
// bound, so the move the search would play is also the one the target points at.
#[test]
fn policy_target_promotes_the_best_lcb_child() {
  let mut search = Search::<f64>::new(PARAMS);
  // The heavier child is slightly better on average but its values are noisy, so
  // its lower bound is worse than the steadier runner-up's. The gap between them
  // has to be small enough that the reduction still credits the runner-up with
  // enough weight to be eligible for the bound at all: a child PUCT would not
  // have explored in hindsight does not get promoted on the strength of an
  // estimate that hindsight says was never worth gathering.
  add_root_child(&mut search, 10, 100, 100, -0.3, 1.0);
  add_root_child(&mut search, 11, 60, 60, -0.29, 0.0841);
  search.nodes[0].visits = 161;
  search.nodes[0].weight = 1.0;
  search.nodes[0].weight_sum = 161.0;
  search.nodes[0].weight_sq_sum = 161.0;

  // The runner-up is the one LCB selection plays.
  assert_eq!(search.best_move().map(|pos| pos.get()), Some(11));

  let target = search.pruned_weights().collect::<Vec<_>>();
  let weight_of = |pos: Pos| target.iter().find(|&&(p, _)| p == pos).map(|&(_, w)| w);
  let promoted = weight_of(11).expect("the best-LCB child must be in the target");
  let other = weight_of(10).expect("the heavier child must be in the target");
  assert!(
    promoted > other,
    "the target should point at the move that gets played, got {} vs {}",
    promoted,
    other
  );
}

// A child can carry enough raw weight to clear the LCB eligibility bar while its
// reduced weight - what PUCT would retrospectively have spent on it - falls short.
// Eligibility has to be judged on the reduced weight, because that is what the
// policy target's own LCB promotion is gated on. Judged on the raw weight, this
// child would be played on the strength of a bound the target does not trust, and
// the target would peak on a different move altogether.
#[test]
fn lcb_eligibility_is_judged_on_the_reduced_weight() {
  let mut search = Search::<f64>::new(PARAMS);
  // The reference child: heavy, and the better of the two on average, but its
  // values are noisy enough that its bound is the poorer one.
  add_root_child(&mut search, 10, 100, 100, -0.5, 1.0);
  // A far lighter child whose value never wavers, so its bound is the better one.
  add_root_child(&mut search, 12, 16, 16, -0.2, 0.04);
  search.nodes[0].visits = 117;
  search.nodes[0].own_visits = 1;
  search.nodes[0].weight = 1.0;
  search.nodes[0].weight_sq = 1.0;
  search.nodes[0].weight_sum = 117.0;
  search.nodes[0].weight_sq_sum = 117.0;
  search.nodes[0].value = 0.0;
  search.nodes[0].value_sq = 0.09;

  // The bar is 15% of the reference child's weight. The light child's raw weight
  // of 16 clears it; the weight left after the reduction does not.
  let min_weight = 0.15 * 100.0;
  let target = search.pruned_weights().collect::<Vec<_>>();
  let reduced = target
    .iter()
    .find(|&&(pos, _)| pos == 12)
    .map(|&(_, weight)| weight)
    .expect("the light child should still be in the target");
  assert!(16.0 > min_weight, "its raw weight has to clear the bar");
  assert!(
    reduced < min_weight,
    "the reduced weight should fall short of it, got {reduced}"
  );

  // So it is ranked on that weight rather than on its bound, and the move played
  // is the reference child - which is where the target peaks too.
  let selection = search.play_selection();
  assert!(
    matches!(selection[1], (12, Either::Left((weight, _))) if weight == reduced),
    "the light child should not be judged on its bound, got {:?}",
    selection[1]
  );
  assert!(matches!(selection[0], (10, Either::Right(_))));
  assert_eq!(search.best_move().map(|pos| pos.get()), Some(10));
  assert_eq!(best_by_weight(target), 10);
}

// The same agreement has to hold over real searches, at every size. Ties are the
// awkward case there: two children can hold the very same bound, and the move to
// play and the target read that tie from different places.
#[test]
fn the_played_move_is_the_one_the_target_points_at() {
  for value in [
    &depth_value as &dyn Fn(&Array4<f64>) -> Array2<f64>,
    &depth_uncertainty_value,
  ] {
    for iterations in [3, 7, 20, 60, 200] {
      let (search, _) = run_search(value, iterations);
      let Some(played) = search.best_move().map(|pos| pos.get()) else {
        continue;
      };
      let target = search.pruned_weights().collect::<Vec<_>>();
      if target.is_empty() {
        continue;
      }
      let max_weight = target.iter().map(|&(_, weight)| weight).fold(0.0, f64::max);
      let played_weight = target
        .iter()
        .find(|&&(pos, _)| pos == played)
        .map(|&(_, weight)| weight)
        .unwrap_or_else(|| panic!("after {iterations} playouts the move played is not in the target"));
      // A peak, not necessarily the unique one: at low playout counts several
      // children can be left with the same weight, and the two paths break such
      // ties differently - on the prior when choosing the move, on child order
      // when reading the target.
      assert_eq!(
        played_weight, max_weight,
        "after {iterations} playouts the move played is not a peak of the target"
      );
    }
  }
}

// The interpolated CDF table stands in for the exact Student's t CDF, so it has
// to agree with it everywhere the downweighting can land, and be a genuine
// distribution at the edges.
#[test]
fn value_weight_cdf_table_matches_the_exact_form() {
  let mut worst = 0.0f64;
  let mut at = 0.0f64;
  // Step off the bucket edges so the interpolation error is actually sampled.
  let mut x = -12.0;
  while x <= 12.0 {
    let diff = (value_weight_cdf(x) - t_cdf(x)).abs();
    if diff > worst {
      worst = diff;
      at = x;
    }
    x += 0.017;
  }
  assert!(
    worst < 1e-4,
    "interpolated CDF should track the exact one, worst {} at {}",
    worst,
    at
  );

  // Monotone, and a probability throughout.
  let mut prev = 0.0;
  let mut x = -60.0;
  while x <= 60.0 {
    let p = value_weight_cdf(x);
    assert!((0.0..=1.0).contains(&p), "cdf out of range at {}: {}", x, p);
    assert!(p >= prev - 1e-12, "cdf should be monotone, dipped at {}", x);
    prev = p;
    x += 0.31;
  }
  // Clamped outside the tabulated range rather than wrapping or panicking.
  assert_eq!(value_weight_cdf(-1e9), value_weight_cdf(-50.0));
  assert_eq!(value_weight_cdf(1e9), value_weight_cdf(50.0));
  assert!((value_weight_cdf(0.0) - 0.5).abs() < 1e-12);
}

// The move self-play plays is drawn from the play selection values, not from the
// raw search weights. Forced playouts widen the search on purpose and the policy
// target takes them back out again; letting them steer the game as well would play
// low-prior moves more often than the search ever judged them worth.
#[test]
fn self_play_samples_the_pruned_target() {
  let mut search = Search::<f64>::new(PARAMS);
  // The heavier child is slightly better on average but its values are noisy, so
  // the steadier runner-up wins on its lower bound and the target promotes it
  // above the heavier one - the setup from `policy_target_promotes_the_best_lcb_child`.
  add_root_child(&mut search, 10, 100, 100, -0.3, 1.0);
  add_root_child(&mut search, 11, 60, 60, -0.29, 0.0841);
  search.nodes[0].visits = 161;
  search.nodes[0].weight = 1.0;
  search.nodes[0].weight_sum = 161.0;
  search.nodes[0].weight_sq_sum = 161.0;

  let heaviest = |search: &Search<f64>| best_by_weight(search.weights().collect());
  let target = |search: &Search<f64>| best_by_weight(search.pruned_weights().collect());
  // The two disagree, which is what makes the sampling source observable at all.
  assert_eq!(heaviest(&search), 10);
  assert_eq!(target(&search), 11);

  // At a temperature this low the draw is the argmax of whatever it samples from,
  // and the two candidates differ by more than a factor of two.
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let played = search.clone().next_root_with_temperature(0.1, &mut rng);
  assert_eq!(played.map(|pos| pos.get()), Some(11));

  // At a realistic temperature the whole distribution has to follow the pruned
  // values: 273 against 100 means the promoted child is played about 73% of the
  // time, where the raw weights of 60 against 100 would give it 38%.
  let pruned = search.pruned_weights().collect::<Vec<_>>();
  let total = pruned.iter().map(|&(_, weight)| weight).sum::<f64>();
  let expected = pruned.iter().find(|&&(pos, _)| pos == 11).unwrap().1 / total;
  const SAMPLES: usize = 200;
  let promoted = (0..SAMPLES)
    .filter(|_| {
      search
        .clone()
        .next_root_with_temperature(1.0, &mut rng)
        .map(|pos| pos.get())
        == Some(11)
    })
    .count() as f64
    / SAMPLES as f64;
  assert!(
    (promoted - expected).abs() < 0.08,
    "sampling should follow the pruned target, played {promoted} of the time against an expected {expected}"
  );
}

/// The position with the highest weight, for comparing what the two targets rank
/// first.
fn best_by_weight(weights: Vec<(Pos, f64)>) -> Pos {
  weights
    .into_iter()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .expect("expected a searched child")
    .0
}

// A reused tree holds values computed against the bias buckets as they stood when
// each node was last visited, and those buckets keep moving as the rest of the
// search feeds them.
#[test]
fn tree_reuse_refreshes_the_bias_corrections() {
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
  let mut search = Search::<f64>::new(PARAMS);
  // A depth-dependent value head is what gives the buckets a nonzero observed
  // bias to drift with in the first place.
  let mut model = |inputs: Array4<f64>, _, _| {
    let result: Result<_, ()> = Ok((uniform_policies(&inputs), depth_value(&inputs)));
    result
  };
  // Enough search that the subtree the root moves into still holds internal
  // nodes sharing buckets with each other. That sharing is what makes a retained
  // value go stale, so too small a surviving tree leaves nothing to observe.
  for _ in 0..200 {
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
  }

  let pos = search.best_move().expect("the search should have found a move");
  assert!(search.next_root(pos.get()));
  search.compact();
  assert!(search.stats_stale, "reusing the tree should mark its stats stale");

  // `Search::BIAS_LAMBDA`, private to the mcgs module. A node no playout has gone
  // past yet - one visit, so nothing of its children in its value - is exactly its
  // raw network value corrected by this much of its bucket's observed bias. That is
  // the relation `add_result` establishes and reuse breaks. (Nodes here always hold
  // an edge per legal move from the moment they are evaluated, so a node with no
  // children at all is a terminal one, which carries no bias key.)
  const LAMBDA: f64 = 0.3;
  // Nodes whose value came straight from `add_result`: nothing has backed up into
  // them, so their value should be exactly their raw value plus their bucket's
  // share, and it is those that reuse leaves behind.
  let is_leaf = |node: &Node<f64>| node.visits > 0 && node.children.iter().all(|edge| edge.visits == 0);
  let stale_leafs = |search: &Search<f64>| {
    search
      .nodes
      .iter()
      .filter(|node| is_leaf(node))
      .filter_map(|node| node.bias_key.map(|key| (node, key)))
      .filter(|(node, key)| {
        let bias = search
          .bias
          .get(key)
          .filter(|entry| entry.weight_sum > 1e-3)
          .map_or(0.0, |entry| entry.delta_sum / entry.weight_sum);
        (node.value - (node.raw_value + LAMBDA * bias)).abs() > 1e-9
      })
      .count()
  };
  let before = stale_leafs(&search);
  assert!(
    before > 0,
    "reuse should have left values behind the buckets that have moved since"
  );

  // One bottom-up pass is deliberately not a fixed point: rebuilding a node feeds
  // its bucket, which shifts every other node sharing that bucket, including ones
  // the pass already went past. So the exact guarantee is pinned on a bucket that
  // only one node belongs to, where nothing can move it again afterwards, and that
  // bucket is shifted by hand rather than by whatever shape the search happened to
  // leave - which is what made this test brittle before.
  let mut members = std::collections::HashMap::new();
  for node in &search.nodes {
    if let Some(key) = node.bias_key {
      *members.entry(key).or_insert(0usize) += 1;
    }
  }
  let (idx, key) = search
    .nodes
    .iter()
    .enumerate()
    .filter(|(_, node)| is_leaf(node))
    .filter_map(|(idx, node)| node.bias_key.map(|key| (idx, key)))
    .find(|(_, key)| members[key] == 1)
    .expect("some retained leaf should be the only member of its bucket");
  search.bias.insert(
    key,
    BiasEntry {
      delta_sum: 0.5,
      weight_sum: 1.0,
    },
  );
  let expected = search.nodes[idx].raw_value + LAMBDA * 0.5;
  assert!(
    (search.nodes[idx].value - expected).abs() > 1e-9,
    "moving the bucket should have left this node's value behind"
  );

  let mut refreshed = search.clone();
  refreshed.recompute_stats();
  assert!(!refreshed.stats_stale);
  let node = &refreshed.nodes[idx];
  assert!(
    (node.value - expected).abs() < 1e-9,
    "the node should have been rebuilt against its bucket, got {} want {expected}",
    node.value
  );
  // And the pass leaves the tree at large less stale than it found it.
  assert!(
    stale_leafs(&refreshed) < before,
    "recomputing should have refreshed retained nodes, still {} of {before} stale",
    stale_leafs(&refreshed)
  );
  // Non-vacuous: at least one of those corrections is a real one.
  assert!(
    refreshed
      .nodes
      .iter()
      .any(|node| node.visits > 0 && (node.value - node.raw_value).abs() > 1e-9),
    "the buckets should be correcting something"
  );

  // And the next search does it without being asked, once, before it descends.
  assert!(field.put_point(pos.get(), Player::Red));
  field.update_grounded();
  futures::executor::block_on(search.mcgs(&mut field, Player::Black, &mut model, 0, &mut rng)).unwrap();
  assert!(!search.stats_stale, "a search should consume the staleness");
}

/// A net with no error head: it fills the third value column with zeros and says
/// so, the way the dummy models do.
struct NoErrorHead;

impl Model<f64> for NoErrorHead {
  type E = ();

  fn predicts_uncertainty(&self) -> bool {
    false
  }

  async fn predict(
    &mut self,
    inputs: Array4<f64>,
    _: Array2<f64>,
    _: Array1<f64>,
  ) -> Result<(Array3<f64>, Array2<f64>), Self::E> {
    Ok((uniform_policies(&inputs), const_value(&inputs, array![0.6, 0.4, 0.0])))
  }
}

// A net that leaves the predicted error at zero must not have that read as a
// perfectly certain evaluation, which would earn the largest weight of all
// rather than an average one and scale the whole search's weights by it. Such a
// net counts every playout once instead, so the weight behind a node is its
// playout count.
#[test]
fn a_net_without_an_error_head_counts_playouts_equally() {
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
  let mut search = Search::<f64>::new(PARAMS);
  // The point of the test is that the net overrides this, so it has to be set.
  const { assert!(PARAMS.uncertainty) };

  for _ in 0..30 {
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut NoErrorHead, 0, &mut rng)).unwrap();
  }

  for node in &search.nodes {
    if node.visits == 0 {
      continue;
    }
    assert!(
      (node.weight - 1.0).abs() < 1e-9,
      "an evaluation should weigh one, got {}",
      node.weight
    );
    assert!(
      (node.weight_sum - node.visits as f64).abs() < 1e-9,
      "total weight should be the playout count, got {} instead of {}",
      node.weight_sum,
      node.visits
    );
  }
}

/// Every position the search submits asks the net for the optimism its
/// parameters call for, one weight per position of the batch: the root asks
/// for the root's own weight, everything below it for the search-wide one. A
/// search that asked for none would silently get the plain policy however the
/// parameters were set.
#[test]
fn the_optimism_of_the_parameters_reaches_the_net() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ......
    ..aA..
    ......
    ",
  );
  let asked = std::cell::RefCell::new(Vec::new());
  let mut model = |inputs: Array4<f64>, _: Array2<f64>, optimism: Array1<f64>| {
    asked.borrow_mut().push(optimism.to_vec());
    let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.6, 0.4, 0.1])));
    result
  };

  let params = Params {
    policy_optimism: 0.75,
    root_policy_optimism: 0.25,
    ..PARAMS
  };
  let mut search = Search::<f64>::new(params);
  for _ in 0..4 {
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
  }

  let asked = asked.borrow();
  // The first batch expands the root alone, at the root's own optimism; the
  // later ones evaluate several leaves each, all below the root, and every one
  // of them carries the search-wide weight.
  assert_eq!(asked[0], vec![0.25]);
  assert!(asked.len() > 1);
  for batch in asked.iter().skip(1) {
    assert!(!batch.is_empty());
    assert!(batch.iter().all(|&optimism| optimism == 0.75), "got {batch:?}");
  }
}

/// Moving the root into a node the previous search expanded as an ordinary
/// leaf leaves it with priors interpolated at the search-wide optimism. The
/// next search has to ask the net about that one position again at the root's
/// own optimism and swap the priors in place, keeping the children and their
/// visits.
#[test]
fn a_reused_root_gets_its_priors_re_predicted_at_root_optimism() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ......
    ..aA..
    ......
    ",
  );
  let asked = std::cell::RefCell::new(Vec::new());
  // The net's policy depends on the optimism, so the swap is observable: the
  // optimistic policy is uniform, the root one leans on the leftmost column.
  let mut model = |inputs: Array4<f64>, _: Array2<f64>, optimism: Array1<f64>| {
    asked.borrow_mut().push(optimism.to_vec());
    let mut policies = uniform_policies(&inputs);
    if optimism[0] == 0.25 {
      let width = policies.len_of(Axis(2));
      for (i, p) in policies.iter_mut().enumerate() {
        if i % width == 0 {
          *p *= 16.0;
        }
      }
    }
    let result: Result<_, ()> = Ok((policies, const_value(&inputs, array![0.6, 0.4, 0.1])));
    result
  };

  let params = Params {
    policy_optimism: 1.0,
    root_policy_optimism: 0.25,
    ..PARAMS
  };
  let mut search = Search::<f64>::new(params);
  for _ in 0..8 {
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
  }

  let pos = search.next_best_root().unwrap();
  assert!(field.put_point(pos.get(), Player::Red));
  let root = &search.nodes[search.root_idx];
  assert!(!root.children.is_empty(), "the new root should come expanded");
  let inherited: Vec<u64> = root.children.iter().map(|edge| edge.visits).collect();
  // Expanded as an inner node, so its priors are the uniform optimistic policy.
  let priors: Vec<f64> = root.children.iter().map(|edge| edge.prior).collect();
  assert!(priors.iter().all(|&prior| (prior - priors[0]).abs() < 1e-9));

  asked.borrow_mut().clear();
  futures::executor::block_on(search.mcgs(&mut field, Player::Black, &mut model, 0, &mut rng)).unwrap();

  // The refresh is a batch of exactly the root position at the root's
  // optimism, before the playout batch at the search-wide one.
  let batches = asked.borrow();
  assert_eq!(batches[0], vec![0.25]);
  assert!(batches[1..].iter().flatten().all(|&optimism| optimism == 1.0));

  // The priors are the root policy now - no longer uniform, renormalized - and
  // the children still carry everything the reused subtree earned: the call's
  // own playouts can only have added to the inherited visits.
  let root = &search.nodes[search.root_idx];
  let priors: Vec<f64> = root.children.iter().map(|edge| edge.prior).collect();
  assert!((priors.iter().sum::<f64>() - 1.0).abs() < 1e-9);
  assert!(priors.iter().any(|&prior| (prior - priors[0]).abs() > 1e-9));
  for (edge, &visits) in root.children.iter().zip(&inherited) {
    assert!(edge.visits >= visits);
  }

  // The refresh happens once: the root the search now owns is no longer
  // stale, so the next call submits no second single-position batch.
  assert!(!search.root_priors_stale);
}

// Two paths of one batch can land on the same position, and only one of them may
// be evaluated: a node's visits are what its edges divide up to take their share
// of its weight, so they have to stay equal to the playouts that reached it, and
// its own evaluation may not be counted twice against its subtree.
#[test]
fn transposing_paths_of_one_batch_are_evaluated_once() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...aA...
    ...Bb...
    ........
    ........
    ",
  );
  let mut search = Search::<f64>::new(PARAMS);

  for _ in 0..80 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      // Concentrating the policy on a few points is what makes the readouts of
      // one batch converge and transpose onto each other.
      &mut |inputs: Array4<f64>, _, _| {
        let batch = inputs.len_of(Axis(0));
        let height = inputs.len_of(Axis(2));
        let width = inputs.len_of(Axis(3));
        let mut policies = Array::from_elem((batch, height, width), 1e-4);
        for b in 0..batch {
          for y in 0..height {
            for x in 0..width {
              if x + y < 3 {
                policies[(b, y, x)] = 0.3;
              }
            }
          }
        }
        let result: Result<_, ()> = Ok((policies, const_value(&inputs, array![0.6, 0.4, 0.1])));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }

  let mut expanded = 0;
  for (idx, node) in search.nodes.iter().enumerate() {
    let edge_visits = node.children.iter().map(|edge| edge.visits).sum::<u64>();
    assert_eq!(
      node.visits,
      node.own_visits + edge_visits,
      "node {idx} should account for every playout that reached it exactly once"
    );
    if !node.children.is_empty() {
      expanded += 1;
      assert_eq!(
        node.own_visits, 1,
        "node {idx} has children, so it was evaluated once and may not weigh more than that"
      );
    }
  }
  assert!(
    expanded > 10,
    "expected a tree of some size, got {expanded} expanded nodes"
  );
}

// Forcing a root child ignores its PUCT score entirely, so the only thing that
// can send the next readout of a batch elsewhere is the virtual loss counting
// towards the child's own forced-playout threshold. Without that the whole batch
// descends into the same root child, which is the diversity the forced playouts
// and the noise exist to create.
#[test]
fn forced_playouts_do_not_absorb_a_whole_batch() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...aA...
    ...Bb...
    ........
    ........
    ",
  );
  let mut search = Search::<f64>::new(Params::SELF_PLAY);
  let mut model = |inputs: Array4<f64>, _, _| {
    let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.6, 0.4, 0.1])));
    result
  };
  // One batch to expand the root, then noise, as self-play does.
  futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
  search.add_dirichlet_noise(&mut rng, 0.25, 10.83, 1.0);

  for batch in 0..50 {
    let before = search.nodes[search.root_idx]
      .children
      .iter()
      .map(|edge| edge.visits)
      .collect::<Vec<_>>();
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
    let touched = search.nodes[search.root_idx]
      .children
      .iter()
      .zip(&before)
      .filter(|(edge, was)| edge.visits > **was)
      .count();
    // The board is wide and the policy uniform, so no single child can honestly
    // want a whole batch to itself here.
    assert!(
      touched > 1,
      "batch {batch} descended into {touched} root children, so the batch was absorbed by one"
    );
  }
}

// A node with no children - terminal, or with every move forbidden - has no edge
// to descend, so every playout that reaches it evaluates it again and its own
// evaluations are the whole of its visit count. `own_visits` is what lets
// `update_node` rebuild that count when a reused tree is recomputed. Rebuilding
// it as "one evaluation plus the children's" instead would reset such a node to a
// single playout while the parent edge went on counting, and the edge's share of
// its weight - which divides by the child's visits - would blow up by the number
// of times the node had been re-entered.
#[test]
fn childless_nodes_keep_their_playout_count_through_a_recompute() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  // Nearly full, so the game ends a ply below the root and the childless nodes
  // sit inside the tree rather than at its root.
  let mut field = construct_field(
    &mut rng,
    "
    aAa
    Aa.
    .A.
    ",
  );
  let mut model = |inputs: Array4<f64>, _, _| {
    let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, array![0.6, 0.4, 0.1])));
    result
  };
  let mut search = Search::<f64>::new(PARAMS);
  for _ in 0..40 {
    futures::executor::block_on(search.mcgs(&mut field, Player::Red, &mut model, 0, &mut rng)).unwrap();
  }

  let re_entered = search
    .nodes
    .iter()
    .filter(|node| node.children.is_empty() && node.own_visits > 1)
    .count();
  assert!(
    re_entered > 0,
    "the position should produce a childless node that playouts re-enter, or this proves nothing"
  );

  // Promoting the root marks the retained values stale, and the next search
  // rebuilds every reachable node through `update_node`.
  let pos = search.best_move().expect("a move to play");
  assert!(field.put_point(pos.get(), Player::Red));
  field.update_grounded();
  assert!(search.next_root(pos.get()));
  assert!(search.stats_stale);
  futures::executor::block_on(search.mcgs(&mut field, Player::Black, &mut model, 0, &mut rng)).unwrap();

  for (idx, node) in search.nodes.iter().enumerate() {
    let edge_visits = node.children.iter().map(|edge| edge.visits).sum::<u64>();
    assert_eq!(
      node.visits,
      node.own_visits + edge_visits,
      "node {idx} lost track of the playouts that reached it"
    );
  }

  // What the count feeds: an edge takes its share of the child's weight by its
  // share of the child's visits, so a reset count multiplies that share instead.
  let total_weight = search.nodes[search.root_idx].weight_sum;
  for node in &search.nodes {
    for edge in &node.children {
      if edge.visits == 0 {
        continue;
      }
      let Some(&child_idx) = search.map.get(&edge.hash) else {
        continue;
      };
      let child = &search.nodes[child_idx];
      if child.visits == 0 {
        continue;
      }
      let share = child.weight_sum * edge.visits as f64 / child.visits as f64;
      assert!(
        share <= child.weight_sum + 1e-9 && share <= total_weight + 1e-9,
        "an edge claims {share} of a child holding {} in a search of {total_weight}",
        child.weight_sum
      );
    }
  }
}

/// A node's value carries its subtree bias correction, which can push the
/// average past the `[-1, 1]` a value is defined on, and the training loss
/// reads the q target as a probability that has to stay in `[0, 1]` - so the
/// q is clamped back into range. The score has no such range and passes
/// through as the search left it.
#[test]
fn q_values_are_clamped_to_the_value_range() {
  let mut search = Search::<f64>::new(PARAMS);
  add_root_child(&mut search, 10, 4, 4, -1.25, 1.0);
  let child_idx = search.map[&10];
  search.nodes[child_idx].score = -3.5;

  let q_values: Vec<_> = search.q_values().collect();
  assert_eq!(q_values.len(), 1);
  let (pos, weight, q, score) = q_values[0];
  assert_eq!(pos, 10);
  assert_eq!(weight, 4.0);
  assert_eq!(q, 1.0);
  assert_eq!(score, 3.5);
}

/// The net's score estimate - the fourth value column - rides the backups into
/// the tree, so the q targets can say what score the search settled on for
/// each reply. A root child evaluated once carries exactly the net's estimate,
/// negated into the root player's perspective.
#[test]
fn q_scores_follow_the_net_estimate() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ......
    ..aA..
    ......
    ",
  );
  let mut search = Search::<f64>::new(PARAMS);

  // The first batch only expands the root; the following ones descend into
  // its children.
  for _ in 0..3 {
    futures::executor::block_on(search.mcgs(
      &mut field,
      Player::Red,
      &mut |inputs: Array4<f64>, _, _| {
        let result: Result<_, ()> = Ok((
          uniform_policies(&inputs),
          const_value(&inputs, array![1.0, 0.0, 0.0, 2.5]),
        ));
        result
      },
      0,
      &mut rng,
    ))
    .unwrap();
  }

  let q_values: Vec<_> = search.q_values().collect();
  assert!(!q_values.is_empty());
  // Deeper subtrees mix the estimate across perspectives, but a child whose
  // only evidence is its own evaluation reports it exactly.
  assert!(q_values.iter().any(|&(_, _, _, score)| score == -2.5));
  // And nothing in the tree can exceed an estimate every evaluation agrees on.
  assert!(q_values.iter().all(|&(_, _, _, score)| score.abs() <= 2.5 + 1e-9));
}
