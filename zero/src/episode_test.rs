use crate::episode::{
  MCTS_FULL_VISITS, MCTS_VISITS, REDUCED_VISITS_WEIGHT, SideStart, reduced_search, search_side_positions, side_move,
};
use crate::random_model::RandomModel;
use oppai_field::construct_field::construct_field;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// A game whose recent search values sit inside the threshold keeps the full
/// budget and weight, however lopsided single turns were before the window.
#[test]
fn no_reduction_while_the_game_is_open() {
  assert_eq!(reduced_search(&[]), (MCTS_FULL_VISITS, 1.0));
  assert_eq!(reduced_search(&[0.95, 0.0, 0.5, 0.89]), (MCTS_FULL_VISITS, 1.0));
}

/// Too few turns to look back over cannot trigger the reduction, no matter how
/// extreme they are.
#[test]
fn no_reduction_before_the_lookback_fills() {
  assert_eq!(reduced_search(&[1.0, 1.0]), (MCTS_FULL_VISITS, 1.0));
}

/// Every turn of the window has to point at the same winner: one turn leaning
/// the other way keeps the budget intact.
#[test]
fn no_reduction_when_the_window_disagrees() {
  assert_eq!(reduced_search(&[1.0, -1.0, 1.0]), (MCTS_FULL_VISITS, 1.0));
  // Which player is winning does not matter, only the agreement.
  assert_eq!(reduced_search(&[-1.0, -1.0, -1.0]), reduced_search(&[1.0, 1.0, 1.0]));
}

/// A window fully committed to one winner bottoms out at the cheap search
/// budget and the reduced training weight.
#[test]
fn full_reduction_when_the_game_is_decided() {
  let (visits, weight) = reduced_search(&[0.0, 1.0, 1.0, 1.0]);
  assert_eq!(visits, MCTS_VISITS);
  assert!((weight - REDUCED_VISITS_WEIGHT).abs() < 1e-12);
}

/// Between the threshold and certainty the reduction ramps with the square of
/// how far past the threshold the least extreme turn sits.
#[test]
fn reduction_ramps_quadratically() {
  // Halfway through the ramp: a quarter of the way down.
  let (visits, weight) = reduced_search(&[0.95, 1.0, 1.0]);
  let expected_visits = f64::from(MCTS_FULL_VISITS) + 0.25 * (f64::from(MCTS_VISITS) - f64::from(MCTS_FULL_VISITS));
  assert_eq!(visits, expected_visits.round() as u32);
  assert!((weight - (1.0 + 0.25 * (REDUCED_VISITS_WEIGHT - 1.0))).abs() < 1e-12);
}

/// A side move is a legal move other than the one the game played, and with
/// no policy mass on any such move it is still drawn uniformly from them.
#[test]
fn side_moves_avoid_the_played_move() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let field = construct_field(
    &mut rng,
    "
    ....
    .aA.
    ....
    ",
  );
  let played = field.to_pos(0, 0);
  let legal = |pos| field.is_putting_allowed(pos) && pos != played;
  // All the policy on the played move and one other.
  let mut priors = vec![0.0f64; field.length()];
  priors[played] = 0.9;
  priors[field.to_pos(3, 2)] = 0.1;
  let zero = vec![0.0f64; field.length()];
  for _ in 0..200 {
    let pos = side_move(&field, &priors, Some(played), &mut rng).unwrap();
    assert!(legal(pos));
    assert!(legal(side_move(&field, &zero, Some(played), &mut rng).unwrap()));
  }
}

/// Searching a side position records a full search of it, with the move that
/// search chose as the last move of the side line; a side line that went on
/// from it starts with that same line.
#[test]
fn side_positions_record_their_search() {
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
  let mut field = construct_field(
    &mut rng,
    "
    ........
    ........
    ...aA...
    ...Ba...
    ........
    ........
    ",
  );
  let player = field.colored_moves().last().unwrap().1.next();
  let side = field.to_pos(1, 1);
  let branch = field.moves_count();
  assert!(field.put_point(side, player));
  field.update_grounded();
  let model = RandomModel::new(Xoshiro256PlusPlus::seed_from_u64(9));
  let mut lines = 0;
  for _ in 0..32 {
    let start = SideStart {
      field: field.clone(),
      player: player.next(),
      komi_x_2: 0,
      branch,
      first_player: player,
      moves: vec![side],
    };
    let side_positions =
      futures::executor::block_on(search_side_positions::<f64, _, _>(vec![start], &model, 1, &mut rng)).unwrap();
    let first = &side_positions[0];
    assert_eq!((first.branch, first.player), (branch, player));
    assert_eq!(first.moves.len(), 2);
    assert_eq!(first.moves[0], side);
    assert_eq!(first.visits.1, 1.0);
    assert!(first.visits.total() > 0.0);
    // The recorded move is one the search explored.
    assert!(first.visits.0.iter().any(|&(pos, _)| pos == first.moves[1]));
    assert!(field.is_putting_allowed(first.moves[1]));
    for line in &side_positions[1..] {
      assert!(line.moves.starts_with(&first.moves));
      assert_eq!(line.moves.len() % 2, 0);
      lines += 1;
    }
  }
  // A quarter of the side positions go on, so some of 32 do.
  assert!(lines > 0);
}
