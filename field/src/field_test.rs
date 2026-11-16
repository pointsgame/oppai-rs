use crate::construct_field::{construct_field, construct_field_with_zobrist};
use crate::field::{self, Field, Pos};
use crate::player::Player;
use crate::zobrist::Zobrist;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::Arc;

const SEED: u64 = 7;

#[test]
fn simple_surround() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn surround_empty_territory() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert!(field.get_last_chain().is_empty());
}

#[test]
fn move_priority() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .aB.
    aCaB
    .aB.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 1);
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn move_priority_big() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .B..
    BaB.
    aCaB
    .aB.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 2);
  assert_eq!(field.get_last_chain().len(), 8);
}

#[test]
fn onion_surroundings() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..c..
    .cBc.
    cBaBc
    .cBc.
    ..c..
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 4);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert_eq!(field.get_last_chain().len(), 8);
}

#[test]
fn deep_onion_surroundings() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ...D...
    ..DcD..
    .DcBcD.
    DcBaBcD
    .DcBcD.
    ..DcD..
    ...D...
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 0);
  assert_eq!(field.captured_count(Player::Black), 9);
  assert_eq!(field.get_last_chain().len(), 12);
}

#[test]
fn apply_control_surrounding_in_same_turn() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aBa
    .a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn captured_point_is_not_empty_base() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aBa
    .a.
    ",
  );
  let cell = field.cell(field.to_pos(1, 1));
  assert!(cell.is_captured());
  assert!(!cell.is_empty_base());
}

#[test]
fn double_surround() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.a.
    aAbAa
    .a.a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert_eq!(field.get_last_chain().len(), 8);
}

#[test]
fn double_surround_with_empty_part() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn should_not_leave_empty_inside() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 18);
}

#[test]
fn a_hole_inside_a_surrounding() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 16);
}

#[test]
fn a_hole_inside_a_surrounding_after_control_surrounding() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 16);
}

#[test]
fn surrounding_does_not_expand() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn two_surroundings_with_common_border() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a..
    aAa.
    .bAa
    ..a.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 2);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert_eq!(field.get_last_chain().len(), 8);
}

#[test]
fn three_surroundings_with_common_borders() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
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
  assert_eq!(field.get_last_chain().len(), 12);
}

#[test]
fn ambiguous_surrounding_1() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .aa.aa.
    a..b..a
    a.aAa.a
    a..a..a
    .a...a.
    ..aaa..
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.cell(field.to_pos(3, 4)).is_putting_allowed());
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn ambiguous_surrounding_2() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..aaa..
    .a...a.
    a..a..a
    a.aAa.a
    a..b..a
    .aa.aa.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  assert_eq!(field.captured_count(Player::Black), 0);
  assert!(field.cell(field.to_pos(3, 1)).is_putting_allowed());
  assert_eq!(field.get_last_chain().len(), 4);
}

#[test]
fn game_over_1() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .....
    .aa..
    aAAb.
    .aa..
    ...a.
    .....
    ",
  );
  assert_eq!(field.score(Player::Red), 2);
  assert_eq!(field.non_grounded_red, 1);
  assert_eq!(field.non_grounded_black, 0);
  assert!(field.is_game_over(0));
}

#[test]
fn game_over_2() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .....
    .AA..
    AaaB.
    .AA..
    ...A.
    .....
    ",
  );
  assert_eq!(field.score(Player::Black), 2);
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 1);
  assert!(field.is_game_over(0));
}

#[test]
fn game_over_3() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..........
    .......a..
    ..a...aAa.
    .aAb.aAAAb
    ..a...aAa.
    .......a..
    ..........
    ",
  );
  assert_eq!(field.score(Player::Red), 6);
  assert_eq!(field.non_grounded_red, 5);
  assert_eq!(field.non_grounded_black, 0);
  assert!(field.is_game_over(0));
}

#[test]
fn game_over_4() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..........
    .......A..
    ..A...AaA.
    .AaB.AaaaB
    ..A...AaA.
    .......A..
    ..........
    ",
  );
  assert_eq!(field.score(Player::Black), 6);
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 5);
  assert!(field.is_game_over(0));
}

#[test]
fn game_over_5() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aaa
    .a.
    ",
  );
  assert_eq!(field.score(Player::Red), 0);
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 0);
  assert!(field.is_game_over(0));
}

#[test]
fn game_not_over_1() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ....
    .a..
    aAb.
    .a..
    ..a.
    ....
    ",
  );
  assert_eq!(field.score(Player::Red), 1);
  assert_eq!(field.non_grounded_red, 1);
  assert_eq!(field.non_grounded_black, 0);
  assert!(!field.is_game_over(0));
}

#[test]
fn game_not_over_2() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ....
    .A..
    AaB.
    .A..
    ..A.
    ....
    ",
  );
  assert_eq!(field.score(Player::Black), 1);
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 1);
  assert!(!field.is_game_over(0));
}

#[test]
fn game_not_over_3() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..........
    .......a..
    ..a...aAa.
    .aAb.aAA.b
    ..a...aAa.
    .......a..
    ..........
    ",
  );
  assert_eq!(field.score(Player::Red), 5);
  assert_eq!(field.non_grounded_red, 5);
  assert_eq!(field.non_grounded_black, 0);
  assert!(!field.is_game_over(0));
}

#[test]
fn game_not_over_4() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..........
    .......A..
    ..A...AaA.
    .AaB.Aaa.B
    ..A...AaA.
    .......A..
    ..........
    ",
  );
  assert_eq!(field.score(Player::Black), 5);
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 5);
  assert!(!field.is_game_over(0));
}

#[test]
fn grounded() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ....
    .aba
    ..C.
    ....
    ",
  );
  assert!(field.cell(field.to_pos(1, 1)).is_grounded());
  assert!(field.cell(field.to_pos(2, 1)).is_grounded());
  assert!(field.cell(field.to_pos(3, 1)).is_grounded());
  assert!(!field.cell(field.to_pos(2, 2)).is_grounded());
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 1);
}

#[test]
fn grounded_surrounding() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..aa..
    .bA.aa
    ..aa..
    ......
    ",
  );
  for (x, y) in [(2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3)] {
    assert!(field.cell(field.to_pos(x, y)).is_grounded());
  }
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 0);
}

#[test]
fn grounded_control_surrounding() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..aa..
    .aB.aa
    ..aa..
    ......
    ",
  );
  for (x, y) in [(2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (2, 3), (3, 3)] {
    assert!(field.cell(field.to_pos(x, y)).is_grounded());
  }
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 0);
}

#[test]
fn grounded_empty_surrounding() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ......
    ..aa..
    .a..ba
    ..aa..
    ......
    ",
  );
  for (x, y) in [(4, 2), (5, 2)] {
    assert!(field.cell(field.to_pos(x, y)).is_grounded());
  }
  for (x, y) in [(2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3)] {
    assert!(!field.cell(field.to_pos(x, y)).is_grounded());
  }
}

#[test]
fn grounded_hole() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ..........
    .aaaaaaaaa
    .a.......a
    .a.aaaaa.a
    .a.a...aAa
    .b.a.a.aaa
    .a.a...a.a
    .a.aaaaa.a
    .a.......a
    .aaaaaaaaa
    ..........
    ",
  );
  assert_eq!(
    (field.min_pos()..=field.max_pos())
      .filter(|&pos| field.cell(pos).is_grounded())
      .count(),
    81
  );
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 0);
}

#[test]
fn grounded_after_capture_from_empty_base() {
  let mut field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .......
    .HEf...
    A.lBgjd
    .KIc...
    .......
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  let pos = field.to_pos(1, 2);
  assert!(!field.cell(pos).is_grounded());
  assert!(field.put_point(pos, Player::Red));
  field.update_grounded();
  assert!(field.cell(pos).is_grounded());
  assert_eq!(field.non_grounded_red, 0);
  assert_eq!(field.non_grounded_black, 4);
}

#[test]
fn zobrist() {
  let field_1 = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    ab
    CD
    ",
  );
  let field_2 = construct_field_with_zobrist(
    field_1.zobrist_arc(),
    "
    dc
    BA
    ",
  );
  assert_eq!(field_1.hash(), field_2.hash());
}

#[test]
fn zobrist_surrounding() {
  let field_1 = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .a.
    aBa
    .a.
    ",
  );
  let field_2 = construct_field_with_zobrist(
    field_1.zobrist_arc(),
    "
    .a.
    aAb
    .a.
    ",
  );
  assert_eq!(field_1.hash(), field_2.hash());
}

#[test]
fn zobrist_own_point_inside_surrounding() {
  let field = construct_field(
    &mut Xoshiro256PlusPlus::seed_from_u64(SEED),
    "
    .bbbbb.
    b.....b
    b.A.a.b
    b.....b
    .bbbbb.
    ",
  );
  assert_eq!(field.captured_count(Player::Red), 1);
  let length = field.length();
  let hashes = &field.zobrist().hashes;
  let mut expected = 0;
  for pos in 0..length {
    let cell = field.cell(pos);
    if cell.is_put() {
      expected ^= hashes[length * cell.get_player() as usize + pos];
      if cell.is_captured() {
        expected ^= hashes[length * 2 + pos];
      }
    }
  }
  assert_eq!(field.hash(), expected);
}

#[test]
fn undo_check() {
  let width = 20;
  let height = 20;
  let checks = 100;
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 3, &mut rng));
  let mut moves = (0..field::length(width, height)).collect::<Vec<Pos>>();
  for _ in 0..checks {
    let mut field = Field::new(width, height, zobrist.clone());
    moves.shuffle(&mut rng);
    let mut player = Player::Red;
    for &pos in &moves {
      if field.is_putting_allowed(pos) {
        let field_before = field.clone();
        field.put_point(pos, player);
        field.update_grounded();
        field.undo();
        assert_eq!(field_before.points, field.points);
        field.put_point(pos, player);
        field.update_grounded();
        player = player.next();
      }
    }
  }
}

/// The rings of a cell hold exactly its live and its captured neighbours in
/// the documented order, and the neighbour counts read from them are the
/// ones counted cell by cell, on random games with captures, for both
/// players. With [`ring_groups_of_every_ring`] it settles
/// [`Field::number_near_groups`], which counts the runs of the ring.
#[test]
fn live_ring_of_random_games() {
  let width = 12;
  let height = 12;
  let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
  let zobrist = Arc::new(Zobrist::new(field::length(width, height) * 3, &mut rng));
  let mut moves = (0..field::length(width, height)).collect::<Vec<Pos>>();
  let mut checked = 0;
  for _ in 0..20 {
    let mut field = Field::new(width, height, zobrist.clone());
    moves.shuffle(&mut rng);
    let mut player = Player::Red;
    for &pos in &moves {
      if !field.is_putting_allowed(pos) {
        continue;
      }
      field.put_point(pos, player);
      field.update_grounded();
      player = player.next();
      for x in 0..width {
        for y in 0..height {
          let center = field.to_pos(x, y);
          let around = [
            field.n(center),
            field.ne(center),
            field.e(center),
            field.se(center),
            field.s(center),
            field.sw(center),
            field.w(center),
            field.nw(center),
          ];
          for p in [Player::Red, Player::Black] {
            let ring = field.live_ring(center, p);
            let captured = field.captured_ring(center, p);
            for (direction, &neighbor) in around.iter().enumerate() {
              let cell = field.cell(neighbor);
              assert_eq!(field.ring_pos(center, direction), neighbor);
              assert_eq!(ring >> direction & 1 == 1, cell.is_live_players_point(p));
              assert_eq!(
                captured >> direction & 1 == 1,
                cell.is_players_point(p) && !cell.is_live_players_point(p)
              );
            }
            let live = |pos: &Pos| field.cell(*pos).is_live_players_point(p);
            let orthogonal = field.directions(center).iter().filter(|pos| live(pos)).count() as u32;
            let all = around.iter().filter(|pos| live(pos)).count() as u32;
            assert_eq!(field.number_near_points(center, p), orthogonal);
            assert_eq!(field.number_near_points_diag(center, p), all);
            assert_eq!(field.has_near_points(center, p), orthogonal > 0);
            assert_eq!(field.has_near_points_diag(center, p), all > 0);
            checked += 1;
          }
        }
      }
    }
  }
  assert!(checked > 100_000);
}

/// Every one of the 256 rings against the rule spelled out: a run starts at
/// every orthogonal neighbour that is not a dot and is followed clockwise by a
/// dot.
#[test]
fn ring_groups_of_every_ring() {
  for ring in 0..=u8::MAX {
    let dot = |direction: usize| ring >> (direction % 8) & 1 == 1;
    let expected = [0, 2, 4, 6]
      .into_iter()
      .filter(|&direction| !dot(direction) && (dot(direction + 1) || dot(direction + 2)))
      .count() as u32;
    assert_eq!(field::ring_groups(ring), expected, "{ring:08b}");
  }
}
