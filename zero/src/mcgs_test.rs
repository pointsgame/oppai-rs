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
    children: Vec::new(),
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
