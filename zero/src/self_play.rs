use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::mem;

use crate::episode::{episode, mcts};
use crate::mcts::MctsNode;
use crate::model::{Model, TrainableModel};
use ndarray::{Array, Axis};
use num_traits::Float;
use oppai_field::field::Field;
use oppai_field::player::Player;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;

const ITERATIONS_NUMBER: u32 = 10000;
const PIT_GAMES: u64 = 100;
const WIN_RATE_THRESHOLD: f64 = 0.55;
const EPISODES: u32 = 20;
const MCTS_SIMS: u32 = 32;

fn play<'a, N, M, R>(
  field: &mut Field,
  mut player: Player,
  mut model1: &'a M,
  mut model2: &'a M,
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
      "Score: {}, n: {}, p: {}, w: {}\n{:?}",
      field.score(Player::Red),
      node1.n,
      node1.p,
      node1.w,
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

fn pit<N, M, R>(field: &Field, player: Player, new_model: &M, old_model: &M, rng: &mut R) -> Result<bool, M::E>
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

pub fn self_play<N, M, R>(field: &Field, player: Player, mut model: M, rng: &mut R) -> Result<(), M::TE>
where
  M: TrainableModel<N> + Clone,
  N: Float + Sum + SampleUniform + Display + Debug,
  R: Rng,
{
  let mut inputs = Vec::new();
  let mut policies = Vec::new();
  for i in 0..ITERATIONS_NUMBER {
    log::info!("Iteration {}", i);

    let copy = model.clone();
    for j in 0..EPISODES {
      log::info!("Episode {}", j);

      inputs.clear();
      policies.clear();
      let mut values = Vec::new();
      episode(
        &mut field.clone(),
        player,
        &model,
        rng,
        &mut inputs,
        &mut policies,
        &mut values,
      )?;

      log::info!("Train the model");
      let inputs = ndarray::stack(Axis(0), inputs.iter().map(|i| i.view()).collect::<Vec<_>>().as_slice()).unwrap();
      let policies = ndarray::stack(
        Axis(0),
        policies.iter().map(|p| p.view()).collect::<Vec<_>>().as_slice(),
      )
      .unwrap();
      let values = Array::from(values);
      model.train(inputs, policies, values)?;
    }

    log::info!("Pit the new model");
    if pit(field, player, &model, &copy, rng)? {
      model.save()?;
    } else {
      log::warn!("Rejecting new model");
      model = copy;
    }
  }

  Ok(())
}
