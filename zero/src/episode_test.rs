use ndarray::{Array, Array1, Array3, Array4, Axis};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_rotate::rotate::{rotate, ROTATIONS};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cell::RefCell;

use crate::episode::{episode, logistic, mcts};
use crate::field_features::{field_features, CHANNELS};
use crate::mcts::MctsNode;
use crate::model::Model;

const SEED: u64 = 7;

fn uniform_policies(inputs: &Array4<f64>) -> Array3<f64> {
  let batch_size = inputs.len_of(Axis(0));
  let height = inputs.len_of(Axis(2));
  let width = inputs.len_of(Axis(3));
  let policy = 1f64 / (width * height) as f64;
  Array::from_elem((batch_size, height, width), policy)
}

fn const_value(inputs: &Array4<f64>, value: f64) -> Array1<f64> {
  let batch_size = inputs.len_of(Axis(0));
  Array::from_elem(batch_size, value)
}

impl<T> Model<f64> for T
where
  T: Fn(Array4<f64>) -> (Array3<f64>, Array1<f64>),
{
  type E = ();

  fn predict(&self, inputs: Array4<f64>) -> Result<(Array3<f64>, Array1<f64>), Self::E> {
    Ok(self(inputs))
  }
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
  let mut node = MctsNode::default();

  mcts(
    &mut field,
    Player::Red,
    &mut node,
    &|inputs: Array4<f64>| (uniform_policies(&inputs), const_value(&inputs, 1.0)),
    &mut rng,
  )
  .unwrap();
  assert_eq!(node.visits, 1);
  assert_eq!(node.wins, -1.0);
  // corner moves are not considered
  assert_eq!(node.children.len(), (field.width() * field.height()) as usize - 6);
  assert!(node.children.iter().all(|child| child.wins == 1.0));
  assert!(node.children.iter().all(|child| child.children.is_empty()));

  mcts(
    &mut field,
    Player::Red,
    &mut node,
    &|inputs: Array4<f64>| (uniform_policies(&inputs), const_value(&inputs, -1.0)),
    &mut rng,
  )
  .unwrap();
  assert_eq!(node.visits, 9);
  assert_eq!(node.wins, -9.0);
  assert_eq!(node.children.iter().map(|child| child.visits).sum::<u64>(), 8);
  assert_eq!(
    node
      .children
      .iter()
      .filter(|child| child.children.len() == (field.width() * field.height()) as usize - 7)
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
  let mut node = MctsNode::default();

  mcts(
    &mut field,
    Player::Red,
    &mut node,
    &|inputs: Array4<f64>| (uniform_policies(&inputs), const_value(&inputs, 0.0)),
    &mut rng,
  )
  .unwrap();
  assert_eq!(node.visits, 1);
  assert_eq!(node.wins, logistic(-1.0));
  assert!(node.children.is_empty());
}

#[test]
fn episode_simple_surrounding() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .a.
    .Aa
    .a.
    ",
  );

  let model_inputs: RefCell<Vec<Array4<f64>>> = Default::default();

  let mut examples = Default::default();
  episode(
    &mut field,
    Player::Red,
    &|inputs: Array4<f64>| {
      let result = (uniform_policies(&inputs), const_value(&inputs, 0.0));
      model_inputs.borrow_mut().push(inputs);
      result
    },
    &mut rng,
    &mut examples,
  )
  .unwrap();

  assert_eq!(field.moves_count(), 5);

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 0, 1, rotation);
    assert_eq!(examples.policies[rotation as usize][(y as usize, x as usize)], 1.0);
    for channel in 0..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[rotation as usize],
      field_features(&field, Player::Red, rotation)
    );
  }

  assert_eq!(model_inputs.borrow().len(), 1);
  assert_eq!(
    model_inputs.borrow()[0],
    field_features(&field, Player::Red, 0)
      .into_shape((1, CHANNELS, field.height() as usize, field.width() as usize))
      .unwrap()
  );

  assert_eq!(examples.values, vec![logistic(1.0); 8]);
}

#[test]
fn episode_trap() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    .A.
    ..A
    .A.
    ",
  );

  let model_inputs: RefCell<Vec<Array4<f64>>> = Default::default();

  let mut examples = Default::default();
  episode(
    &mut field,
    Player::Red,
    &|inputs: Array4<f64>| {
      let result = (uniform_policies(&inputs), const_value(&inputs, 0.0));
      model_inputs.borrow_mut().push(inputs);
      result
    },
    &mut rng,
    &mut examples,
  )
  .unwrap();

  assert_eq!(field.moves_count(), 5);

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 1, 1, rotation);
    assert_eq!(
      examples.policies[(ROTATIONS + rotation) as usize][(y as usize, x as usize)],
      1.0
    );
    for channel in 0..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[(ROTATIONS + rotation) as usize],
      field_features(&field, Player::Black, rotation)
    );
  }

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 0, 1, rotation);
    assert!(
      examples.policies[rotation as usize][(y as usize, x as usize)] > examples.policies[rotation as usize][(1, 1)]
    );
    for channel in 0..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[rotation as usize],
      field_features(&field, Player::Red, rotation)
    );
  }

  assert_eq!(model_inputs.borrow().len(), 2);

  let features = field_features(&field, Player::Red, 0)
    .into_shape((1, CHANNELS, field.height() as usize, field.width() as usize))
    .unwrap();
  assert_eq!(model_inputs.borrow()[0], features);

  field.put_point(field.to_pos(0, 1), Player::Red);
  let features1 = field_features::<f64>(&field, Player::Black, 0);
  field.undo();
  field.put_point(field.to_pos(1, 1), Player::Red);
  let features2 = field_features::<f64>(&field, Player::Black, 0);
  // order depends on rng
  assert_eq!(features1, model_inputs.borrow()[1].index_axis(Axis(0), 0));
  assert_eq!(features2, model_inputs.borrow()[1].index_axis(Axis(0), 1));

  assert_eq!(examples.values, vec![0.0; 16]);
}

#[test]
fn episode_winning_game() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let mut field = construct_field(
    &mut rng,
    "
    ..........
    ..........
    ..aaaaaa..
    ..aAAAAa..
    ..aAAAAa..
    ..aAAAAa..
    ..aAAAAa..
    ..aaaaaa..
    ..........
    ..........
    ",
  );

  let center_x = (field.width() / 2) as usize;
  let center_y = (field.height() / 2) as usize;

  let mut examples = Default::default();
  episode(
    &mut field,
    Player::Red,
    &|inputs: Array4<f64>| {
      let batch_size = inputs.len_of(Axis(0));
      let values = Array::from_iter((0..batch_size).map(|i| {
        if inputs[(i, 0, center_y, center_x)] > 0.0 {
          1.0
        } else {
          0.0
        }
      }));
      (uniform_policies(&inputs), values)
    },
    &mut rng,
    &mut examples,
  )
  .unwrap();

  for (value, input) in examples.values.into_iter().zip(examples.inputs.into_iter()) {
    assert!(if input[(0, center_y, center_x)] > 0.0 {
      value > 0.0
    } else {
      value < 0.0
    });
  }
}
