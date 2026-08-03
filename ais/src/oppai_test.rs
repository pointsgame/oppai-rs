use crate::oppai::zero_weight_to_f64;
use either::Either;

/// Flattening Zero's play selection weight must preserve its two tier ordering:
/// every child with an LCB outranks every child ranked by search weight alone,
/// LCBs order among themselves, and so do search weights. Otherwise the caller
/// that sorts by the flattened weight plays a barely searched move.
#[test]
fn zero_weight_keeps_play_selection_order() {
  let searched = |weight, prior| zero_weight_to_f64::<f64>(Either::Left((weight, prior)));
  let lcb = |lcb| zero_weight_to_f64::<f64>(Either::Right(lcb));

  // The worst LCB still beats the highest search weight without one.
  assert!(lcb(-5.0) > searched(f64::MAX, 1.0));

  // LCBs keep their relative order.
  assert!(lcb(1.0) > lcb(0.0));
  assert!(lcb(0.0) > lcb(-0.25));
  assert!(lcb(-0.25) > lcb(-5.0));

  // So do search weights, with the prior as the tie breaker.
  assert!(searched(60.0, 0.1) > searched(5.0, 0.9));
  assert!(searched(5.0, 0.9) > searched(5.0, 0.1));
  assert!(searched(0.0, 0.0) > f64::NEG_INFINITY);
}
