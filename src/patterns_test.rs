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
    .+..
    2 3 1.0
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
    2 3 1.0
    "
  ]);
}

#[test]
#[should_panic]
fn pattern_with_less_moves_than_on_image() {
  construct_patterns(&[
    "
    4 4 1.0
    ....
    .RB.
    .BR.
    .++.
    2 3 1.0
    "
  ]);
}

#[test]
fn pattern_empty_doesnt_match() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #B..
    #.+.
    2 3 1.0
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
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_borders_matches() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #BR.
    #.+.
    2 3 1.0
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
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_borders_doesnt_match() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #BR.
    #.+.
    2 3 1.0
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

#[test]
fn pattern_any_matches() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    ****
    #.+.
    2 3 1.0
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
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_any_except_border_matches() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    #???
    #.+.
    2 3 1.0
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
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_any_except_border_doesnt_match() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #RB.
    ????
    #.+.
    2 3 1.0
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
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_red_black_or_none_matches() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #Rbb
    #Brr
    #.+.
    2 3 1.0
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
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 3), 1f64)]);
}

#[test]
fn pattern_red_black_or_none_doesnt_match() {
  let p = construct_patterns(&[
    "
    4 4 1.0
    #...
    #bbb
    #rrr
    #.+.
    2 3 1.0
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
  assert!(p.find(&field, Player::Red, false).is_empty());
}

#[test]
fn pattern_rotation_0() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(0, 1), 1f64)]);
}

#[test]
fn pattern_rotation_1() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    ...a.
    .Aa..
    .aAA.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 4), 1f64)]);
}

#[test]
fn pattern_rotation_2() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .a.A.
    ..aA.
    ..Aa.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(4, 3), 1f64)]);
}

#[test]
fn pattern_rotation_3() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .AAa.
    ..aA.
    .a...
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(3, 0), 1f64)]);
}

#[test]
fn pattern_rotation_4() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    ..Aa.
    ..aA.
    .a.A.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(4, 1), 1f64)]);
}

#[test]
fn pattern_rotation_5() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .aAA.
    .Aa..
    ...a.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(1, 0), 1f64)]);
}

#[test]
fn pattern_rotation_6() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .A.a.
    .Aa..
    .aA..
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(0, 3), 1f64)]);
}

#[test]
fn pattern_rotation_7() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .a...
    ..aA.
    .AAa.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Red, false), vec![(field.to_pos(3, 4), 1f64)]);
}

#[test]
fn pattern_inversion_doesnt_match() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    "
  );
  assert!(p.find(&field, Player::Black, false).is_empty());
}

#[test]
fn pattern_inversion_matches() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .....
    +RB..
    .BR..
    .B.R.
    .....
    0 1 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .Aa..
    .aA..
    .a.A.
    .....
    "
  );
  assert_eq!(p.find(&field, Player::Black, false), vec![(field.to_pos(0, 1), 1f64)]);
}

#[test]
fn pattern_multiple_moves() {
  let p = construct_patterns(&[
    "
    5 5 5.0
    .+...
    +RB..
    .BR..
    .B.R.
    .....
    0 1 3.0
    1 0 1.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    "
  );
  assert_eq!(p.find_sorted(&field, Player::Red, false), vec![(field.to_pos(0, 1), 0.75f64), (field.to_pos(1, 0), 0.25f64)]);
  assert_eq!(p.find_foreground(&field, Player::Red, false), Some(field.to_pos(0, 1)));
}

#[test]
fn multiple_patterns() {
  let p = construct_patterns(&[
    "
    5 5 1.0
    .+...
    +RB..
    .BR..
    .B.R.
    .....
    0 1 3.0
    1 0 1.0
    ",
    "
    5 5 4.0
    ???..
    ?rb..
    .br..
    .B.R+
    ...+.
    4 3 1.0
    3 4 3.0
    "
  ]);
  let field = construct_field(
    "
    .....
    .aA..
    .Aa..
    .A.a.
    .....
    "
  );
  assert_eq!(p.find_sorted(&field, Player::Red, false), vec![(field.to_pos(3, 4), 0.6f64), (field.to_pos(4, 3), 0.2f64), (field.to_pos(0, 1), 0.15f64), (field.to_pos(1, 0), 0.05f64)]);
  assert_eq!(p.find_foreground(&field, Player::Red, false), Some(field.to_pos(3, 4)));
}
