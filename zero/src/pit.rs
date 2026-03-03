use crate::mcgs::Search;
use crate::model::Model;
use num_traits::Float;
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::Rng;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem;

const MCTS_SIMS: u32 = 32;

pub fn play<'a, N, M, R>(
  field: &mut Field,
  mut player: Player,
  mut model1: &'a mut M,
  mut model2: &'a mut M,
  mut komi_x_2: i32,
  rng: &mut R,
) -> Result<i32, M::E>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
  R: Rng,
{
  let mut moves_count = 0;
  let mut search1 = Search::new();
  let mut search2 = Search::new();

  while !field.is_game_over() {
    for _ in 0..MCTS_SIMS {
      search1.mcgs(field, player, model1, komi_x_2, rng)?;
    }

    let pos = if let Some(pos) = search1.next_best_root() {
      pos
    } else {
      break;
    };

    search1.compact();
    search2.next_root(pos.get());
    search2.compact();
    assert!(field.put_point(pos.get(), player));
    field.update_grounded();

    mem::swap(&mut model1, &mut model2);
    mem::swap(&mut search1, &mut search2);
    player = player.next();
    komi_x_2 = -komi_x_2;
    moves_count += 1;
  }

  Ok(field.score(if moves_count % 2 == 0 { player } else { player.next() }))
}
