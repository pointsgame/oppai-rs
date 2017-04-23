use rand::XorShiftRng;
use test::Bencher;
use player::Player;
use field;
use uct::UctRoot;
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
  let length = field::length(field.width(), field.height());
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let mut uct = UctRoot::new(length);
    uct.best_move_with_iterations_count(&field, Player::Black, &mut rng, 10000)
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
  let length = field::length(field.width(), field.height());
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let mut uct = UctRoot::new(length);
    uct.best_move_with_iterations_count(&field, Player::Red, &mut rng, 10000)
  });
}
