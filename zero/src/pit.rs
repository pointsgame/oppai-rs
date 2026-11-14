use crate::mcgs::{Params, Search};
use crate::model::Model;
use num_traits::Float;
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::Rng;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem;

const MCTS_SIMS: u32 = 32;

/// Plays a game from the position in `field` between `model1`, moving first as
/// `player`, and `model2`, and returns its score from `model1`'s side, or `None`
/// if `should_stop` turned true before the game ended - it's checked before every
/// move, so that games whose outcome no longer matters can be abandoned.
///
/// Every move is searched with a clone of the model on move, dropped once the
/// move is made. A [`crate::batch_model::BatchModel`] counts its clones as the
/// games its evaluator waits for, and that way a game counts only for the model
/// it's waiting on: holding both models for the whole game would make each
/// evaluator wait for games that are submitting to the other one.
pub async fn play<'a, N, M, R, S>(
  field: &mut Field,
  mut player: Player,
  mut model1: &'a M,
  mut model2: &'a M,
  mut komi_x_2: i32,
  rng: &mut R,
  should_stop: S,
) -> Result<Option<i32>, M::E>
where
  M: Model<N> + Clone,
  N: Float + Sum + Display + Debug,
  R: Rng,
  S: Fn() -> bool,
{
  let mut moves_count = 0;
  let mut search1 = Search::new(Params::PLAY);
  let mut search2 = Search::new(Params::PLAY);

  while !field.is_game_over(if player == Player::Red { komi_x_2 } else { -komi_x_2 }) {
    if should_stop() {
      return Ok(None);
    }
    let model = model1.clone();
    for _ in 0..MCTS_SIMS {
      search1.mcgs(field, player, &model, komi_x_2, rng).await?;
    }
    drop(model);

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

  let score = field.score(player) * 2 + komi_x_2;
  Ok(Some(if moves_count % 2 == 0 { score } else { -score }))
}
