use config::{set_minimax_type, MinimaxType};
use construct_field::construct_field;
use hash_table::HashTable;
use minimax::minimax;
use player::Player;
use rand::XorShiftRng;
use test::Bencher;

#[bench]
fn negascout_find_best_move(bencher: &mut Bencher) {
  let field = construct_field(
    "
    ........
    ........
    ...a....
    ..AaA...
    ...Aaa..
    ..A.A...
    ........
    ........
    ",
  );
  set_minimax_type(MinimaxType::NegaScout);
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let hash_table = HashTable::new(1000);
    let mut local_field = field.clone();
    minimax(&mut local_field, Player::Red, &hash_table, &mut rng, 8)
  });
}

#[bench]
fn mtdf_find_best_move(bencher: &mut Bencher) {
  let field = construct_field(
    "
    ........
    ........
    ...a....
    ..AaA...
    ...Aaa..
    ..A.A...
    ........
    ........
    ",
  );
  set_minimax_type(MinimaxType::MTDF);
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let hash_table = HashTable::new(1000);
    let mut local_field = field.clone();
    minimax(&mut local_field, Player::Red, &hash_table, &mut rng, 8)
  });
}
