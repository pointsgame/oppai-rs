use std::sync::Arc;
use quickcheck;
use quickcheck::{Arbitrary, Gen, TestResult};
use zobrist::Zobrist;
use player::Player;
use field;
use field::{Pos, Field};
use construct_field::construct_field;

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
  assert!(!field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(0, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 0)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 2)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 1)).is_putting_allowed());
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
  assert!(field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(field.cell(field.to_pos(1, 1)).is_empty_base());
  assert_eq!(field.cell(field.to_pos(1, 1)).get_empty_base_player(), Some(Player::Red));
  assert!(!field.cell(field.to_pos(0, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 0)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(1, 2)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(2, 1)).is_putting_allowed());
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
    .a.a.
    aAbAa
    .a.a.
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
  assert!(field.cell(field.to_pos(1, 1)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(3, 1)).is_putting_allowed());
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
  assert!(!field.cell(field.to_pos(4, 4)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(4, 1)).is_putting_allowed());
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
  assert!(!field.cell(field.to_pos(4, 4)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(4, 1)).is_putting_allowed());
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
  assert!(field.cell(field.to_pos(6, 3)).is_putting_allowed());
  assert!(field.cell(field.to_pos(4, 3)).is_putting_allowed());
  assert!(field.cell(field.to_pos(4, 5)).is_putting_allowed());
  assert!(field.cell(field.to_pos(6, 5)).is_putting_allowed());
  assert!(!field.cell(field.to_pos(5, 4)).is_putting_allowed());
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

#[derive(Clone, PartialEq, Debug)]
struct FieldArbitrary {
  width: u32,
  height: u32,
  moves: Vec<Pos>,
  zobrist: Arc<Zobrist>
}

impl Iterator for FieldArbitrary {
  type Item = FieldArbitrary;
  fn next(&mut self) -> Option<FieldArbitrary> {
    if self.moves.is_empty() {
      None
    } else {
      let result = self.clone();
      self.moves.pop();
      Some(result)
    }
  }
  fn count(self) -> usize {
    self.moves.len()
  }
}

impl Arbitrary for FieldArbitrary {
  fn arbitrary<G: Gen>(gen: &mut G) -> FieldArbitrary {
    let width = gen.next_u32() % 27 + 3;
    let height = gen.next_u32() % 27 + 3;
    let mut moves = (field::min_pos(width) .. field::max_pos(width, height) + 1).collect::<Vec<Pos>>();
    gen.shuffle(&mut moves);
    let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 2, gen));
    FieldArbitrary {
      width: width,
      height: height,
      moves: moves,
      zobrist: zobrist
    }
  }
  fn shrink(&self) -> Box<Iterator<Item = FieldArbitrary>> {
    let mut result = self.clone();
    result.moves.pop();
    Box::new(result)
  }
}

#[test]
fn undo_check() {
  #[cfg_attr(feature="clippy", allow(needless_pass_by_value))]
  fn prop(field_arbitrary: FieldArbitrary) -> TestResult {
    let mut field = Field::new(field_arbitrary.width, field_arbitrary.height, field_arbitrary.zobrist.clone());
    let mut player = Player::Red;
    for &pos in &field_arbitrary.moves {
      if field.is_putting_allowed(pos) {
        let field_before = field.clone();
        field.put_point(pos, player);
        field.undo();
        if field_before != field {
          return TestResult::failed();
        }
        field.put_point(pos, player);
        player = player.next();
      }
    }
    TestResult::passed()
  }
  quickcheck::quickcheck(prop as fn(FieldArbitrary) -> TestResult);
}
