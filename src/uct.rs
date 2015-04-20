use std::sync::atomic::*;
use types::*;

struct UctNode {
  wins: AtomicUsize,
  draws: AtomicUsize,
  visits: AtomicUsize,
  pos: Pos,
  child: AtomicPtr<UctNode>,
  sibling: Option<Box<UctNode>>
}
