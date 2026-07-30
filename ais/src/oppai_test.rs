use crate::oppai::zero_weight_to_f64;
use either::Either;

/// Flattening Zero's play selection weight must preserve its two tier ordering:
/// every child with an LCB outranks every child ranked by visits alone, LCBs
/// order among themselves, and so do visit counts. Otherwise the caller that
/// sorts by the flattened weight plays a barely visited move.
#[test]
fn zero_weight_keeps_play_selection_order() {
  let visited = |visits, prior| zero_weight_to_f64::<f64>(Either::Left((visits, prior)));
  let lcb = |lcb| zero_weight_to_f64::<f64>(Either::Right(lcb));

  // The worst LCB still beats the highest visit count without one.
  assert!(lcb(-5.0) > visited(u64::MAX, 1.0));

  // LCBs keep their relative order.
  assert!(lcb(1.0) > lcb(0.0));
  assert!(lcb(0.0) > lcb(-0.25));
  assert!(lcb(-0.25) > lcb(-5.0));

  // So do visit counts, with the prior as the tie breaker.
  assert!(visited(60, 0.1) > visited(5, 0.9));
  assert!(visited(5, 0.9) > visited(5, 0.1));
  assert!(visited(0, 0.0) > f64::NEG_INFINITY);
}
