use crate::episode::{Visits, episode, examples};
use crate::field_features::{CHANNELS, field_features};
use crate::mcgs_test::{const_value, uniform_policies};
use ndarray::{Array, Array4, Axis, array};
use oppai_field::construct_field::construct_field;
use oppai_field::player::Player;
use oppai_rotate::rotate::{ROTATIONS, rotate, rotate_back};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cell::RefCell;

const SEED: u64 = 7;

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

  let mut visits = episode(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, 0.0)));
      model_inputs.borrow_mut().push(inputs);
      result
    },
    &mut rng,
  )
  .unwrap();
  for visits in &mut visits {
    visits.1 = true;
  }
  let examples = examples::<f64>(
    field.width(),
    field.height(),
    field.zobrist_arc(),
    &visits,
    &field.colored_moves().collect::<Vec<_>>(),
  );

  assert_eq!(field.moves_count(), 5);
  assert!(examples.policies.iter().all(|p| (p.sum() - 1.0).abs() < 0.001));

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 0, 1, rotation);
    assert_eq!(examples.policies[rotation as usize][(y as usize, x as usize)], 1.0);
    for channel in 1..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[rotation as usize],
      field_features(&field, Player::Red, field.width(), field.height(), rotation)
    );
  }

  assert_eq!(model_inputs.borrow().len(), 1);
  assert_eq!(
    model_inputs.borrow()[0],
    field_features(&field, Player::Red, field.width(), field.height(), 0)
      .to_shape((1, CHANNELS, field.height() as usize, field.width() as usize))
      .unwrap()
  );

  assert_eq!(examples.values, vec![1.0; 8]);
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

  let mut visits = episode(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>| {
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), const_value(&inputs, 0.0)));
      model_inputs.borrow_mut().push(inputs);
      result
    },
    &mut rng,
  )
  .unwrap();
  for visits in &mut visits {
    visits.1 = true;
  }
  let examples = examples::<f64>(
    field.width(),
    field.height(),
    field.zobrist_arc(),
    &visits,
    &field.colored_moves().collect::<Vec<_>>(),
  );

  assert_eq!(field.moves_count(), 5);
  assert!(examples.policies.iter().all(|p| (p.sum() - 1.0).abs() < 0.001));

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 1, 1, rotation);
    assert_eq!(
      examples.policies[(ROTATIONS + rotation) as usize][(y as usize, x as usize)],
      1.0
    );
    for channel in 1..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[(ROTATIONS + rotation) as usize],
      field_features(&field, Player::Black, field.width(), field.height(), rotation)
    );
  }

  field.undo();
  for rotation in 0..ROTATIONS {
    let (x, y) = rotate(field.width(), field.height(), 0, 1, rotation);
    assert!(
      examples.policies[rotation as usize][(y as usize, x as usize)] > examples.policies[rotation as usize][(1, 1)]
    );
    for channel in 1..CHANNELS {
      assert_eq!(
        examples.inputs[rotation as usize][(channel, y as usize, x as usize)],
        0.0
      );
    }
  }
  for rotation in 0..ROTATIONS {
    assert_eq!(
      examples.inputs[rotation as usize],
      field_features(&field, Player::Red, field.width(), field.height(), rotation)
    );
  }

  assert_eq!(model_inputs.borrow().len(), 2);

  let features = field_features(&field, Player::Red, field.width(), field.height(), 0);
  let features = features
    .to_shape((1, CHANNELS, field.height() as usize, field.width() as usize))
    .unwrap();
  assert_eq!(model_inputs.borrow()[0], features);

  field.put_point(field.to_pos(0, 1), Player::Red);
  field.update_grounded();
  let features1 = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
  field.undo();
  field.put_point(field.to_pos(1, 1), Player::Red);
  field.update_grounded();
  let features2 = field_features::<f64>(&field, Player::Black, field.width(), field.height(), 0);
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

  let visits = episode(
    &mut field,
    Player::Red,
    &mut |inputs: Array4<f64>| {
      let batch_size = inputs.len_of(Axis(0));
      let values = Array::from_iter((0..batch_size).map(|i| {
        if inputs[(i, 0, center_y, center_x)] > 0.0 {
          1.0
        } else {
          0.0
        }
      }));
      let result: Result<_, ()> = Ok((uniform_policies(&inputs), values));
      result
    },
    &mut rng,
  )
  .unwrap();
  let examples = examples::<f64>(
    field.width(),
    field.height(),
    field.zobrist_arc(),
    &visits,
    &field.colored_moves().collect::<Vec<_>>(),
  );

  assert!(examples.policies.iter().all(|p| (p.sum() - 1.0).abs() < 0.001));

  for (value, input) in examples.values.into_iter().zip(examples.inputs.into_iter()) {
    assert!(if input[(0, center_y, center_x)] > 0.0 {
      value > 0.0
    } else {
      value < 0.0
    });
  }
}

#[test]
fn visits_to_examples() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let field = construct_field(
    &mut rng,
    "
    iBc
    HaD
    gFe
    ",
  );
  let visits = vec![
    Visits(vec![(field.to_pos(0, 0), 2), (field.to_pos(0, 1), 6)], true),
    Visits(vec![(field.to_pos(0, 0), 8)], true),
  ];
  let examples = examples::<f32>(
    field.width(),
    field.height(),
    field.zobrist_arc(),
    &visits,
    &field.colored_moves().collect::<Vec<_>>(),
  );

  #[rustfmt::skip]
  let inputs_0 = array![
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]],

    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],

    [[0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0]],

    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],

    [[0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0]],

    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],

    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],

    [[0.0, 1.0, 1.0],
     [0.0, 0.0, 1.0],
     [1.0, 1.0, 1.0]],
  ];
  assert_eq!(examples.inputs[0], inputs_0);
  for rotation in 0..ROTATIONS {
    for c in 0..CHANNELS {
      for y in 0..field.height() {
        for x in 0..field.width() {
          let (x_rotated, y_rotated) = rotate_back(field.width(), field.height(), x, y, rotation);
          assert_eq!(
            examples.inputs[rotation as usize][[c, y as usize, x as usize]],
            inputs_0[[c, y_rotated as usize, x_rotated as usize]],
          );
        }
      }
    }
  }

  #[rustfmt::skip]
  let policies_0 = array![
    [0.25, 0.0, 0.0],
    [0.75, 0.0, 0.0],
    [0.00, 0.0, 0.0],
  ];
  assert_eq!(examples.policies[0], policies_0);
  for rotation in 0..ROTATIONS {
    for y in 0..field.height() {
      for x in 0..field.width() {
        let (x_rotated, y_rotated) = rotate_back(field.width(), field.height(), x, y, rotation);
        assert_eq!(
          examples.policies[rotation as usize][[y as usize, x as usize]],
          policies_0[[y_rotated as usize, x_rotated as usize]],
        );
      }
    }
  }

  assert!(examples.values[0] > 0.0);

  #[rustfmt::skip]
  let inputs_1 = array![
    [[1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]],

    [[0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0]],

    [[0.0, 1.0, 0.0],
     [1.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]],

    [[0.0, 0.0, 1.0],
     [0.0, 0.0, 0.0],
     [1.0, 0.0, 1.0]],

    [[0.0, 1.0, 0.0],
     [1.0, 1.0, 1.0],
     [0.0, 1.0, 0.0]],

    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],

    [[0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0]],

    [[0.0, 1.0, 1.0],
     [1.0, 1.0, 1.0],
     [1.0, 1.0, 1.0]],
  ];
  assert_eq!(examples.inputs[8], inputs_1);

  #[rustfmt::skip]
  let policies_1 = array![
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
  ];
  assert_eq!(examples.policies[8], policies_1);

  assert!(examples.values[8] < 0.0);
}
