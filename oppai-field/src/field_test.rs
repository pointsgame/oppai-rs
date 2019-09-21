use crate::construct_field::construct_field;
use crate::field::{self, Field, Pos};
use crate::player::Player;
use crate::zobrist::Zobrist;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;

const SEED: [u8; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

#[test]
fn simple_surround() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a.
    cBa
    .a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(0, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 0)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 2)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 1)).is_putting_allowed());
}

#[test]
fn surround_empty_territory() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a.
    a.a
    .a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(field.cell(field.to_pos(1, 1)).is_empty_base());
  assert_eq!(
    field.cell(field.to_pos(1, 1)).get_empty_base_player(),
    Some(Player::Red)
  );
  assert!(!field.cell(field.to_pos(0, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 0)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 2)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 1)).is_putting_allowed());
}

#[test]
fn move_priority() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .aB.
    aCaB
    .aB.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 1);
}

#[test]
fn move_priority_big() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .B..
    BaB.
    aCaB
    .aB.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 2);
}

#[test]
fn onion_surroundings() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ...c...
    ..cBc..
    .cBaBc.
    ..cBc..
    ...c...
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 4);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn apply_control_surrounding_in_same_turn() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a.
    aBa
    .a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn double_surround() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a.a.
    aAbAa
    .a.a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn double_surround_with_empty_part() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .b.b..
    b.zAb.
    .b.b..
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(3, 1)).is_putting_allowed());
}

#[test]
fn should_not_leave_empty_inside() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .aaaa..
    a....a.
    a.b...a
    .z.bC.a
    a.b...a
    a....a.
    .aaaa..
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.cell(field.to_pos(2, 3)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 4)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 2)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 3)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(3, 3)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 1)).is_putting_allowed());
}

#[test]
fn a_hole_inside_a_surrounding() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
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
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.cell(field.to_pos(4, 4)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(4, 1)).is_putting_allowed());
}

#[test]
fn a_hole_inside_a_surrounding_after_control_surrounding() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
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
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(!field.cell(field.to_pos(4, 4)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(4, 1)).is_putting_allowed());
}

#[test]
fn surrounding_does_not_expand() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
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
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.cell(field.to_pos(6, 3)).is_putting_allowed());
  assert!(field.cell(field.to_pos(4, 3)).is_putting_allowed());
  assert!(field.cell(field.to_pos(4, 5)).is_putting_allowed());
  assert!(field.cell(field.to_pos(6, 5)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(5, 4)).is_putting_allowed());
}

#[test]
fn two_surroundings_with_common_border() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    .a..
    aAa.
    .bAa
    ..a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn three_surroundings_with_common_borders() {
  let field = construct_field(
    &mut XorShiftRng::from_seed(SEED),
    "
    ..a..
    .aAa.
    ..bAa
    .aAa.
    ..a..
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 3);
  assert_eq!(field.captured_count(Player::Black), 0);
}

#[test]
fn undo_check() {
  let width = 20;
  let height = 20;
  let mut rng = XorShiftRng::from_seed(SEED);
  let zobrist = Zobrist::new(field::length(width, height) * 2, &mut rng);
  let mut field = Field::new(width, height, Arc::new(zobrist));
  let mut moves = (field::min_pos(width)..=field::max_pos(width, height)).collect::<Vec<Pos>>();
  moves.shuffle(&mut rng);
  let mut player = Player::Red;
  for &pos in &moves {
    if field.is_putting_allowed(pos) {
      let field_before = field.clone();
      field.put_point(pos, player);
      field.undo();
      assert!(field_before == field);
      field.put_point(pos, player);
      player = player.next();
    }
  }
}
