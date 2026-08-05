use crate::episode::Visits;
use crate::examples::{ExampleGame, Examples, TD_VALUE_COEFFS, TD_VALUES};
use ndarray::Axis;
use oppai_field::{
  construct_field::construct_field,
  field::{Field, length},
  player::Player,
  zobrist::Zobrist,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::Arc;

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

// The TD score horizons blend the score the game stood at on the coming turns
// the same way, with the remaining weight on the final score - so the shortest
// horizon says where the score is heading next while the longest is nearly the
// score the game ended at.
#[test]
fn td_scores_blend_the_coming_turns() {
  let area = 16.0;
  // Two points behind now, level next turn, and the game ends four points up.
  let turn_scores = [-2.0, 0.0];
  let final_score = 4.0;
  let mut td_scores = Vec::<f64>::new();
  Examples::td_scores_to_vec(area, &turn_scores, final_score, &mut td_scores);
  assert_eq!(td_scores.len(), TD_VALUES);

  for (i, c) in TD_VALUE_COEFFS.into_iter().enumerate() {
    let now_factor = 1.0 / (1.0 + area * c);
    let expected =
      now_factor * -2.0 + (1.0 - now_factor) * now_factor * 0.0 + (1.0 - now_factor) * (1.0 - now_factor) * final_score;
    assert!((td_scores[i] - expected).abs() < 1e-12);
  }

  // The shortest horizon leans on the turns at hand, so it sits furthest from
  // the final score; every horizon stays between the two.
  assert!(td_scores[TD_VALUES - 1] < td_scores[0]);
  assert!(td_scores.iter().all(|&score| (-2.0..=final_score).contains(&score)));
}

// The TD score target of a training row has to describe that row's own game,
// from the perspective of the player to move there and with the komi that player
// gets: a row of the winning side must see the score it is about to win by, and
// its opponent's rows the same score negated.
#[test]
fn batch_td_scores_follow_the_game_score() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  // Red closes a diamond around the black dot with its last move, so the score
  // stands at zero all game and only moves at the very end.
  let field = construct_field(
    &mut rng,
    "
    .a..
    bAcB
    .d.C
    ",
  );
  assert_eq!(field.score(Player::Red), 1, "the fixture should end in a capture");
  let (width, height) = (field.width(), field.height());
  // Red is a point up and gets one and a half more from the komi.
  let komi_x_2 = 3;
  let final_red_score = 2.5;

  let mut examples = Examples::default();
  let visits = (0..field.moves_count())
    .map(|_| Visits(Vec::new(), true, 0.0, 0.0, 0.0))
    .collect();
  examples.add(komi_x_2, visits, &field, false, false, &mut rng);

  let zobrist = Arc::new(Zobrist::new(length(width, height) * 3, &mut rng));
  let rows = examples.len();
  let batch = examples
    .batches::<f64>(width, height, zobrist.clone(), rows)
    .next()
    .unwrap();
  assert_eq!(batch.td_scores.dim(), (rows, TD_VALUES));

  for (row, example) in examples.examples.iter().enumerate() {
    let game = &examples.games[example.game];
    let player = game.moves[example.position].1;
    let komi = f64::from(if player == Player::Red { komi_x_2 } else { -komi_x_2 }) / 2.0;

    // The score the game stood at on this row's turn and on every turn after
    // it, in this row's perspective, derived from the moves rather than from
    // what the batch did with them.
    let mut replay = Field::new(game.width, game.height, zobrist.clone());
    let mut turn_scores = Vec::new();
    for (i, &(pos, move_player)) in game.moves.iter().enumerate() {
      if i >= example.position {
        turn_scores.push(f64::from(replay.score(player)) + komi);
      }
      assert!(replay.put_point(pos, move_player));
    }
    let final_score = f64::from(replay.score(player)) + komi;

    let mut expected = Vec::<f64>::new();
    Examples::td_scores_to_vec(
      f64::from(game.width * game.height),
      &turn_scores,
      final_score,
      &mut expected,
    );
    for (horizon, expected) in expected.into_iter().enumerate() {
      assert!((batch.td_scores[(row, horizon)] - expected).abs() < 1e-12);
      // The score never dips below the komi and ends at the final score, so
      // every horizon of every row sits between the two - positive for Red,
      // negated for Black.
      let score = batch.td_scores[(row, horizon)] * if player == Player::Red { 1.0 } else { -1.0 };
      assert!(
        score > 1.5 && score < final_red_score,
        "row {row} horizon {horizon}: got {score}"
      );
    }
  }
}

// A position at the very end of a game has no coming turns to blend, so every
// horizon is the final score.
#[test]
fn td_scores_of_the_last_position_are_the_final_score() {
  let mut td_scores = Vec::<f64>::new();
  Examples::td_scores_to_vec(16.0, &[], -1.5, &mut td_scores);
  assert_eq!(td_scores, vec![-1.5; TD_VALUES]);
}

// The last position of a game has no reply, so its opponent policy target is
// all zeros - a zero target contributes no cross-entropy loss. Every other
// row's opponent target is the next turn's search distribution. A uniform
// fallback on the last row would train the opponent heads towards noise once
// per game.
#[test]
fn opponent_policy_of_the_last_position_is_zero() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let field = construct_field(
    &mut rng,
    "
    aA..
    ....
    ",
  );
  let (width, height) = (field.width(), field.height());

  let mut examples = Examples::default();
  // Every recorded search put all of its weight on the move that was played.
  let visits = field
    .moves
    .iter()
    .map(|&pos| Visits(vec![(pos, 1.0)], true, 0.0, 0.0, 0.0))
    .collect();
  examples.add(0, visits, &field, false, false, &mut rng);

  let zobrist = Arc::new(Zobrist::new(length(width, height) * 3, &mut rng));
  let rows = examples.len();
  let batch = examples.batches::<f64>(width, height, zobrist, rows).next().unwrap();

  for (row, example) in examples.examples.iter().enumerate() {
    let sum = batch.opponent_policies.index_axis(Axis(0), row).sum();
    if example.position == examples.games[example.game].moves.len() - 1 {
      assert_eq!(sum, 0.0, "row {row}: the last position must have a zero target");
    } else {
      assert!((sum - 1.0).abs() < 1e-12, "row {row}: got {sum}");
    }
  }
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
