use ndarray::{Array, Array1, Array3, Array4, Axis};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::iter;

use crate::episode::mcts;
use crate::mcts::MctsNode;
use crate::model::Model;

const SEED: u64 = 7;

struct StubModel {
  width: u32,
  height: u32,
  value: f64,
}

impl Model for StubModel {
  type E = ();

  fn predict(&self, inputs: Array4<f64>) -> Result<(Array3<f64>, Array1<f64>), Self::E> {
    let batch_size = inputs.len_of(Axis(0));
    let height = inputs.len_of(Axis(2));
    let width = inputs.len_of(Axis(3));

    assert_eq!(width, self.width as usize);
    assert_eq!(height, self.height as usize);

    let policy = 1f64 / (width * height) as f64;

    let mut policies = Vec::with_capacity(batch_size * width * height);
    let mut values = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
      policies.extend(iter::repeat(policy).take(width * height));
      values.push(self.value);
    }

    let policies = Array::from_shape_vec((batch_size, height, width), policies).unwrap();
    let values = Array::from(values);
    Ok((policies, values))
  }
}

#[test]
fn mcts_first_iterations() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..aA..
    ......
    ",
  );
  let mut node = MctsNode::new(0, 0f64, 0f64);
  let mut model = StubModel {
    width: field.width(),
    height: field.height(),
    value: 1.0,
  };

  mcts(&mut field, Player::Red, &mut node, &model).unwrap();
  assert_eq!(node.n, 1);
  assert_eq!(node.w, -1.0);
  assert_eq!(node.children.len(), (field.width() * field.height()) as usize - 2);
  assert!(node.children.iter().all(|child| child.w == 1.0));
  assert!(node.children.iter().all(|child| child.children.is_empty()));

  model.value = -1.0;
  mcts(&mut field, Player::Red, &mut node, &model).unwrap();
  assert_eq!(node.n, 9);
  assert_eq!(node.w, -9.0);
  assert_eq!(node.children.iter().map(|child| child.n).sum::<u64>(), 8);
  assert_eq!(
    node
      .children
      .iter()
      .filter(|child| child.children.len() == (field.width() * field.height()) as usize - 3)
      .count(),
    8
  );
}

#[test]
fn mcts_last_iterations() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aAa
    .a.
    ",
  );
  let mut node = MctsNode::new(0, 0f64, 0f64);
  let model = StubModel {
    width: field.width(),
    height: field.height(),
    value: 0.0,
  };

  mcts(&mut field, Player::Red, &mut node, &model).unwrap();
  assert_eq!(node.n, 1);
  assert_eq!(node.w, -1.0);
  assert!(node.children.is_empty());
}