use player::Player;
use patterns::Patterns;
use construct_field::construct_field;

fn construct_patterns(strings: &[&str]) -> Patterns {
  let mut pattern_s = String::new();
  let mut p = Patterns::empty();
  for s in strings {
    pattern_s.clear();
    for line in s.split('\n').map(|line| line.trim_matches(' ')).filter(|line| !line.is_empty()) {
      pattern_s.push_str(line);
      pattern_s.push('\n');
    }
    p.add_str(pattern_s.as_str());
  }
  p
}

#[test]
#[should_panic]
fn pattern_moves_discrepancy() {
  construct_patterns(&[
    "
    4 4 1.0
    ....
    .RB.
    .BR.
    ..+.
    3 3 1.0
    "
  ]);
}

#[test]
#[should_panic]
fn pattern_without_moves_on_image() {
  construct_patterns(&[
    "
    4 4 1.0
    ....
    .RB.
    .BR.
    ....
    3 3 1.0
    "
  ]);
}

#[test]
fn pattern_borders_matches() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #BR.
    #..+
    3 3 1.0
    "
  ]);
  let field = construct_field(
    "
    ...
    aA.
    Aa.
    ...
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(2, 3), 1f64)]);
}

#[test]
fn pattern_borders_doesnt_match() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #BR.
    #..+
    3 3 1.0
    "
  ]);
  let field = construct_field(
    "
    ....
    .aA.
    .Aa.
    ....
    "
  );
  assert!(p.find(&field, Player::Red, false).is_empty());
}
