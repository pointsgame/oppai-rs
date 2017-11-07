use rand::XorShiftRng;
use test::Bencher;
use player::Player;
use hash_table::HashTable;
use minimax::minimax;
use construct_field::construct_field;

#[bench]
fn find_best_move_1(bencher: &mut Bencher) {
  let field = construct_field(
    "
    .............
    .............
    ...aAa.......
    ..AAa...A....
    ..Aa...A..a..
    ..Aaa.AA.a...
    ..AaaaaAa....
    ..AAa.Aaa....
    ..aaAA.A.....
    .............
    .............
    "
  );
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let hash_table = HashTable::new(1000);
    let mut local_field = field.clone();
    minimax(&mut local_field, Player::Black, &hash_table, &mut rng, 6)
  });
}

#[bench]
fn find_best_move_2(bencher: &mut Bencher) {
  let field = construct_field(
    "
    .......
    ...a...
    .......
    ..Aa.A.
    .A...A.
    .AaaaA.
    ..AAAa.
    .....a.
    .......
    "
  );
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let hash_table = HashTable::new(1000);
    let mut local_field = field.clone();
    minimax(&mut local_field, Player::Red, &hash_table, &mut rng, 6)
  });
}
