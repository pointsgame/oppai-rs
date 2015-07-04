use std::ascii::AsciiExt;
use std::sync::Arc;
use rand::{Rng, XorShiftRng, SeedableRng};
use quickcheck;
use quickcheck::TestResult;
use types::{Coord, Pos};
use player::Player;
use zobrist::Zobrist;
use field;
use field::Field;

fn construct_field(image: &str) -> Field {
  let lines = image.split('\n').map(|line| line.trim_matches(' ')).filter(|line| !line.is_empty()).collect::<Vec<&str>>();
  let height = lines.len() as Coord;
  assert!(height > 0);
  let width = lines.first().unwrap().len() as Coord;
  assert!(lines.iter().all(|line| line.len() as Coord == width));
  let mut moves = lines.into_iter().enumerate().flat_map(|(y, line)|
    line.chars().enumerate().filter(|&(_, c)| c.to_ascii_lowercase() != c.to_ascii_uppercase()).map(move |(x, c)| (c, x as Coord, y as Coord))
  ).collect::<Vec<(char, Coord, Coord)>>();
  moves.sort_by(|&(c1, _, _), &(c2, _, _)| (c1.to_ascii_lowercase(), c1.is_lowercase()).cmp(&(c2.to_ascii_lowercase(), c2.is_lowercase())));
  let mut rng = XorShiftRng::new_unseeded();
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
  let mut field = Field::new(width, height, zobrist);
  for (c, x, y) in moves.into_iter() {
    let player = Player::from_bool(c.is_uppercase());
    let pos = field.to_pos(x, y);
    field.put_point(pos, player);
  }
  field
}

#[test]
fn simple_surround() {
  let field = construct_field(
    "
    .a.
    cBa
    .a.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.is_putting_allowed(field.to_pos(1, 1)));
  assert!(!field.is_putting_allowed(field.to_pos(0, 1)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 0)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 2)));
  assert!(!field.is_putting_allowed(field.to_pos(2, 1)));
}

#[test]
fn surround_empty_territory() {
  let field = construct_field(
    "
    .a.
    a.a
    .a.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.is_putting_allowed(field.to_pos(1, 1)));
  assert!(field.is_empty_base(field.to_pos(1, 1)));
  assert_eq!(field.get_empty_base_player(field.to_pos(1, 1)), Some(Player::Red));
  assert!(!field.is_putting_allowed(field.to_pos(0, 1)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 0)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 2)));
  assert!(!field.is_putting_allowed(field.to_pos(2, 1)));
}

#[test]
fn move_priority() {
  let field = construct_field(
    "
    .aB.
    aCaB
    .aB.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 1);
}

#[test]
fn move_priority_big() {
  let field = construct_field(
    "
    .B..
    BaB.
    aCaB
    .aB.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 2);
}

#[test]
fn onion_surroundings() {
  let field = construct_field(
    "
    ...c...
    ..cBc..
    .cBaBc.
    ..cBc..
    ...c...
    "
  );
  assert_eq!(field.captured_count(Player::Red), 4);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn apply_control_surrounding_in_same_turn() {
  let field = construct_field(
    "
    .a.
    aBa
    .a.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn double_surround() {
  let field = construct_field(
    "
    .b.b..
    bAzAb.
    .b.b..
    "
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn double_surround_with_empty_part() {
  let field = construct_field(
    "
    .b.b..
    b.zAb.
    .b.b..
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.is_putting_allowed(field.to_pos(1, 1)));
  assert!(!field.is_putting_allowed(field.to_pos(3, 1)));
}

#[test]
fn should_not_leave_empty_inside() {
  let field = construct_field(
    "
    .aaaa..
    a....a.
    a.b...a
    .z.bC.a
    a.b...a
    a....a.
    .aaaa..
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.is_putting_allowed(field.to_pos(2, 3)));
  assert!(!field.is_putting_allowed(field.to_pos(2, 4)));
  assert!(!field.is_putting_allowed(field.to_pos(2, 2)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 3)));
  assert!(!field.is_putting_allowed(field.to_pos(3, 3)));
  assert!(!field.is_putting_allowed(field.to_pos(1, 1)));
}

#[test]
fn a_hole_inside_a_surrounding() {
  let field = construct_field(
    "
    ....c....
    ...c.c...
    ..c...c..
    .c..a..c.
    c..a.a..c
    .c..a..c.
    ..c...c..
    ...cBc...
    ....d....
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.is_putting_allowed(field.to_pos(4, 4)));
  assert!(!field.is_putting_allowed(field.to_pos(4, 1)));
}

#[test]
fn a_hole_inside_a_surrounding_after_control_surrounding() {
  let field = construct_field(
    "
    ....b....
    ...b.b...
    ..b...b..
    .b..a..b.
    b..a.a..b
    .b..a..b.
    ..b...b..
    ...bCb...
    ....b....
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.is_putting_allowed(field.to_pos(4, 4)));
  assert!(!field.is_putting_allowed(field.to_pos(4, 1)));
}

#[test]
fn surrounding_does_not_expand() {
  let field = construct_field(
    "
    ....a....
    ...a.a...
    ..a.a.a..
    .a.a.a.a.
    a.a.aBa.a
    .a.a.a.a.
    ..a.a.a..
    ...a.a...
    ....a....
    "
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.is_putting_allowed(field.to_pos(6, 3)));
  assert!(field.is_putting_allowed(field.to_pos(4, 3)));
  assert!(field.is_putting_allowed(field.to_pos(4, 5)));
  assert!(field.is_putting_allowed(field.to_pos(6, 5)));
  assert!(!field.is_putting_allowed(field.to_pos(5, 4)));
}

#[test]
fn two_surroundings_with_common_border() {
  let field = construct_field(
    "
    .a..
    aAa.
    .bAa
    ..a.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn two_surroundings_with_common_dot() {
  let field = construct_field(
    "
    .a.a.
    aAbAa
    .a.a.
    "
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn three_surroundings_with_common_borders() {
  let field = construct_field(
    "
    ..a..
    .aAa.
    ..bAa
    .aAa.
    ..a..
    "
  );
  assert_eq!(field.captured_count(Player::Red), 3);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn undo_check() {
  fn prop(width_seed: Coord, height_seed: Coord, seed: u64) -> TestResult {
    let width = width_seed % 30;
    let height = height_seed % 30;
    if width < 3 || height < 3 {
      return TestResult::discard();
    }
    let seed_array = [3, seed as u32, 7, (seed >> 32) as u32];
    let mut rng = XorShiftRng::from_seed(seed_array);
    let mut moves = (field::min_pos(width) .. field::max_pos(width, height)).collect::<Vec<Pos>>();
    rng.shuffle(&mut moves);
    let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, &mut rng));
    let mut field = Field::new(width, height, zobrist);
    let mut player = Player::Red;
    for pos in moves {
      if field.is_putting_allowed(pos) {
        player = player.next();
        let field_before = field.clone();
        field.put_point(pos, player);
        field.undo();
        if field_before != field {
          return TestResult::failed();
        }
        field.put_point(pos, player);
      }
    }
    TestResult::passed()
  }
  quickcheck::quickcheck(prop as fn(Coord, Coord, u64) -> TestResult);
}
