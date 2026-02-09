use ndarray::{Array, Array1, Array3, Array4, Axis};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::mcgs::Search;

const SEED: u64 = 7;

pub fn uniform_policies(inputs: &Array4<f64>) -> Array3<f64> {
  let batch_size = inputs.len_of(Axis(0));
  let height = inputs.len_of(Axis(2));
  let width = inputs.len_of(Axis(3));
  let policy = 1f64 / (width * height) as f64;
  Array::from_elem((batch_size, height, width), policy)
}

pub fn const_value(inputs: &Array4<f64>, value: f64) -> Array1<f64> {
  let batch_size = inputs.len_of(Axis(0));
  Array::from_elem(batch_size, value)
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
  let mut search = Search::<f64>::new();

  search
    .mcgs(&mut field, Player::Red, &mut |inputs: Array4<f64>| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, 1.0)));
      result
    })
    .unwrap();
  assert_eq!(search.nodes[0].visits, 1);
  assert_eq!(search.nodes[0].value, 1.0);
  // corner moves are not considered
  assert_eq!(
    search.nodes[0].children.len(),
    (field.width() * field.height()) as usize - 6
  );
  assert!(
    search.nodes[0]
      .children
      .iter()
      .all(|edge| search.nodes[search.map[&edge.hash]].raw_value == 0.0)
  );
  assert!(
    search.nodes[0]
      .children
      .iter()
      .all(|edge| search.nodes[search.map[&edge.hash]].value == 0.0)
  );
  assert!(
    search.nodes[0]
      .children
      .iter()
      .all(|edge| search.nodes[search.map[&edge.hash]].children.is_empty())
  );

  search
    .mcgs(&mut field, Player::Red, &mut |inputs: Array4<f64>| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, -1.0)));
      result
    })
    .unwrap();
  assert_eq!(search.nodes[0].visits, 9);
  assert_eq!(search.nodes[0].value, 1.0);
  assert_eq!(search.nodes[0].children.iter().map(|edge| edge.visits).sum::<u64>(), 8);
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .filter(
        |edge| search.nodes[search.map[&edge.hash]].children.len() == (field.width() * field.height()) as usize - 7
      )
      .count(),
    8
  );
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .filter(|edge| search.nodes[search.map[&edge.hash]].raw_value == -1.0)
      .count(),
    8
  );
  assert_eq!(
    search.nodes[0]
      .children
      .iter()
      .filter(|edge| search.nodes[search.map[&edge.hash]].value == -1.0)
      .count(),
    8
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
  let mut search = Search::<f64>::new();

  search
    .mcgs(&mut field, Player::Red, &mut |inputs: Array4<f64>| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, 0.0)));
      result
    })
    .unwrap();
  assert_eq!(search.nodes[0].visits, 1);
  assert_eq!(search.nodes[0].value, 1.0);
  assert!(search.nodes[0].children.is_empty());
}
