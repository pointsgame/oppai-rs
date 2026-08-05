use crate::episode::{MCTS_FULL_VISITS, MCTS_VISITS, REDUCED_VISITS_WEIGHT, reduced_search};

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
