use crate::episode::Visits;
use crate::examples::{ExampleGame, Examples, TD_VALUE_COEFFS, TD_VALUES};
use oppai_field::player::Player;

fn game(values: [f64; 2]) -> ExampleGame {
  ExampleGame {
    width: 4,
    height: 4,
    moves: vec![(0, Player::Red), (1, Player::Black)],
    komi_x_2: 0,
    score: 1,
    has_result: true,
    visits: vec![
      Visits(Vec::new(), true, 0.0, values[0], 0.0),
      Visits(Vec::new(), true, 0.0, values[1], 0.0),
    ],
  }
}

// Each TD horizon blends the future turns' search values geometrically, with
// the remaining weight on the final result. The stored values are in the
// mover's perspective, so the second turn's value flips sign for Red.
#[test]
fn td_values_blend_search_values() {
  let game = game([0.5, -0.25]);
  let mut td_values = Vec::<f64>::new();
  Examples::td_values_to_vec(&game, 0, Player::Red, 1.0, &mut td_values);
  assert_eq!(td_values.len(), TD_VALUES * 2);

  let area = 16.0;
  for (i, c) in TD_VALUE_COEFFS.into_iter().enumerate() {
    let now_factor = 1.0 / (1.0 + area * c);
    let expected = now_factor * 0.5 + (1.0 - now_factor) * now_factor * 0.25
      + (1.0 - now_factor) * (1.0 - now_factor) * 1.0;
    let expected_win = (1.0 + expected) / 2.0;
    assert!((td_values[2 * i] - expected_win).abs() < 1e-12);
    assert!((td_values[2 * i] + td_values[2 * i + 1] - 1.0).abs() < 1e-12);
  }

  // Shorter horizons weigh the near-term search value more, so with a current
  // value below the final result they sit further from the final result.
  assert!(td_values[0] > td_values[2 * (TD_VALUES - 1)]);
}

// Games without recorded search values (old data) fall back to the final
// result for every horizon.
#[test]
fn td_values_fall_back_to_final_result() {
  let game = game([0.0, 0.0]);
  let mut td_values = Vec::<f64>::new();
  Examples::td_values_to_vec(&game, 0, Player::Red, 1.0, &mut td_values);
  for i in 0..TD_VALUES {
    assert_eq!(td_values[2 * i], 1.0);
    assert_eq!(td_values[2 * i + 1], 0.0);
  }
}

// The replay window keeps the most recent rows, drops the games that no
// remaining example references, and reindexes the survivors.
#[test]
fn window_keeps_recent_examples() {
  use crate::examples::Example;

  let mut examples = Examples::default();
  for _ in 0..10 {
    let index = examples.games.len();
    examples.games.push(game([0.1, -0.1]));
    for _ in 0..10 {
      examples.examples.push(Example {
        game: index,
        position: 0,
        rotation: 0,
        history: 5,
      });
    }
  }

  // total 100, min 20: window = 20 + 80 * 0.25 = 40 most recent rows.
  examples.window(20, 0.25);
  assert_eq!(examples.examples.len(), 40);
  assert_eq!(examples.games.len(), 4);
  assert!(examples.examples.iter().all(|example| example.game < examples.games.len()));

  // Below the minimum nothing is dropped.
  let len = examples.examples.len();
  examples.window(1000, 0.25);
  assert_eq!(examples.examples.len(), len);
}

// Sampling keeps about the requested number of rows.
#[test]
fn sample_bounds_examples() {
  use crate::examples::Example;
  use rand::SeedableRng;

  let mut examples = Examples::default();
  examples.games.push(game([0.1, -0.1]));
  for _ in 0..10000 {
    examples.examples.push(Example {
      game: 0,
      position: 0,
      rotation: 0,
      history: 5,
    });
  }

  let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
  examples.sample(1000, &mut rng);
  let len = examples.examples.len();
  assert!((900..1100).contains(&len), "expected about 1000 examples, got {}", len);
}

// A side game (no result) trains the value towards its own recorded search
// value and gets zero weight on the outcome-derived targets.
#[test]
fn side_games_use_search_values() {
  use oppai_field::construct_field::construct_field;
  use oppai_field::field::length;
  use oppai_field::zobrist::Zobrist;
  use rand::SeedableRng;
  use std::sync::Arc;

  let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
  let field = construct_field(
    &mut rng,
    "
    ....
    .aA.
    ....
    ",
  );
  let pos = field.to_pos(0, 0);
  // One full search for the last move, with a search value of 0.5 for the
  // mover.
  let visits = vec![Visits(vec![(pos, 10)], true, 0.0, 0.5, 0.25)];
  let mut examples = Examples::default();
  examples.add(0, visits, &field, false, false, false, &mut rng);
  assert!(!examples.is_empty());

  let zobrist = Arc::new(Zobrist::new(length(4, 3) * 3, &mut rng));
  let batch = examples.batches::<f64>(4, 3, zobrist, examples.len()).next().unwrap();
  for i in 0..batch.values.dim().0 {
    assert!((batch.values[(i, 0)] - 0.75).abs() < 1e-9);
    assert!((batch.values[(i, 1)] - 0.25).abs() < 1e-9);
    // All TD horizons of a single-position game collapse to the search value.
    for td in 0..TD_VALUES {
      assert!((batch.td_values[(i, td, 0)] - 0.75).abs() < 1e-9);
    }
    assert_eq!(batch.outcome_weights[i], 0.0);
  }
}
