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
    let expected =
      now_factor * 0.5 + (1.0 - now_factor) * now_factor * 0.25 + (1.0 - now_factor) * (1.0 - now_factor) * 1.0;
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
