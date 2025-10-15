use crate::mcts::mcts;
use crate::mcts_node::MctsNode;
use crate::model::Model;
use num_traits::Float;
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::Rng;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem;

const PIT_GAMES: u64 = 100;
const WIN_RATE_THRESHOLD: f64 = 0.55;
const MCTS_SIMS: u32 = 32;

fn play<'a, N, M, R>(
  field: &mut Field,
  mut player: Player,
  mut model1: &'a mut M,
  mut model2: &'a mut M,
  rng: &mut R,
) -> Result<i32, M::E>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
  R: Rng,
{
  let mut moves_count = 0;
  let mut node1 = MctsNode::default();
  let mut node2 = MctsNode::default();

  while !field.is_game_over() {
    for _ in 0..MCTS_SIMS {
      mcts(field, player, &mut node1, model1, rng)?;
    }

    node1 = node1.best_child().unwrap();
    node2 = node2
      .children
      .into_iter()
      .find(|child| child.pos == node1.pos)
      .unwrap_or_default();
    field.put_point(node1.pos, player);

    log::debug!(
      "Score: {}, visits: {}, policy: {}, wins: {}\n{:?}",
      field.score(Player::Red),
      node1.visits,
      node1.policy,
      node1.wins,
      field
    );

    mem::swap(&mut model1, &mut model2);
    mem::swap(&mut node1, &mut node2);
    player = player.next();
    moves_count += 1;
  }

  Ok(field.score(if moves_count % 2 == 0 { player } else { player.next() }))
}

#[inline]
fn win_rate(wins: u64, losses: u64, games: u64) -> f64 {
  if games == 0 {
    0.0
  } else {
    let draws = games - wins - losses;
    (wins as f64 + draws as f64 / 2.0) / games as f64
  }
}

pub fn pit<N, M, R>(
  field: &Field,
  player: Player,
  new_model: &mut M,
  old_model: &mut M,
  rng: &mut R,
) -> Result<bool, M::E>
where
  M: Model<N>,
  N: Float + Sum + Display + Debug,
  R: Rng,
{
  let mut wins = 0;
  let mut losses = 0;

  for i in 0..PIT_GAMES {
    log::info!("Game {}, win rate {}", i, win_rate(wins, losses, i));

    let result = if i % 2 == 0 {
      play(&mut field.clone(), player, new_model, old_model, rng)?
    } else {
      -play(&mut field.clone(), player, old_model, new_model, rng)?
    };

    match result.cmp(&0) {
      Ordering::Less => losses += 1,
      Ordering::Greater => wins += 1,
      Ordering::Equal => {}
    };
  }

  let win_rate = win_rate(wins, losses, PIT_GAMES);
  log::info!("Win rate is {}", win_rate);

  Ok(win_rate > WIN_RATE_THRESHOLD)
}
