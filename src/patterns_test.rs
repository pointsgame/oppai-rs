use patterns::Patterns;

fn construct_patterns(s: &str) -> Patterns {
  let mut pattern_s = String::new();
  for line in s.split('\n').map(|line| line.trim_matches(' ')).filter(|line| !line.is_empty()) {
    pattern_s.push_str(line);
    pattern_s.push('\n');
  }
  let mut p = Patterns::empty();
  p.add_str(pattern_s.as_str());
  p
}

#[test]
#[should_panic]
fn pattern_moves_discrepancy() {
  construct_patterns(
    "
    4 4 1.0
    ....
    .RB.
    .BR.
    ..+.
    3 3 1.0
    "
  );
}
