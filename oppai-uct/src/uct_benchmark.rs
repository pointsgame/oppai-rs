use construct_field::construct_field;
use field;
use player::Player;
use rand::XorShiftRng;
use test::Bencher;
use uct::UctRoot;

#[bench]
fn find_best_move(bencher: &mut Bencher) {
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
  let length = field::length(field.width(), field.height());
  bencher.iter(|| {
    let mut rng = XorShiftRng::new_unseeded();
    let mut uct = UctRoot::new(length);
    uct.best_move_with_iterations_count(&field, Player::Red, &mut rng, 500000)
  });
}
