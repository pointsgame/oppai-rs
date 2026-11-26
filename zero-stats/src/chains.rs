//! How the live dots of each player split into 8-connected chains over the
//! course of games: whether pooling features per chain would differ from
//! pooling them per player.

use crate::common::{Table, fmt, read_games};
use anyhow::Result;
use oppai_field::{
  field::{Field, directions_diag, to_xy},
  player::Player,
};
use std::{collections::VecDeque, path::PathBuf};

const PHASES: usize = 5;

#[derive(Default, Clone, Copy)]
struct Acc {
  /// Player-sides of positions with at least one live dot.
  sides: f64,
  dots: f64,
  chains: f64,
  /// Σ over sides of Σs²/(Σs)²: the chance two random dots of a side share a
  /// chain.
  same: f64,
  largest: f64,
  /// Dots, weighted by their own count, in chains spanning more than 4 and
  /// more than 8 cells.
  span4: f64,
  span8: f64,
  ungrounded: f64,
  ungrounded_chains: f64,
  ungrounded_span8: f64,
  ungrounded_same: f64,
  ungrounded_sides: f64,
  /// Empty cells next to 2 or more distinct chains of the side.
  joining: f64,
  /// Joining cells whose chains are not all within 4 cells of the cell.
  joining_far: f64,
}

struct Chain {
  size: u32,
  grounded: bool,
  min: (u32, u32),
  max: (u32, u32),
}

impl Chain {
  fn span(&self) -> u32 {
    (self.max.0 - self.min.0).max(self.max.1 - self.min.1) + 1
  }
}

fn analyse(field: &Field, player: Player, labels: &mut [u32], acc: &mut Acc) {
  let stride = field.stride;
  labels.fill(u32::MAX);
  let mut chains: Vec<Chain> = Vec::new();
  let mut q = VecDeque::new();
  for pos in field.min_pos()..=field.max_pos() {
    if labels[pos] != u32::MAX || !field.cell(pos).is_live_players_point(player) {
      continue;
    }
    let id = chains.len() as u32;
    let xy = to_xy(stride, pos);
    let mut chain = Chain {
      size: 0,
      grounded: false,
      min: xy,
      max: xy,
    };
    labels[pos] = id;
    q.push_back(pos);
    while let Some(p) = q.pop_front() {
      let (x, y) = to_xy(stride, p);
      chain.size += 1;
      chain.grounded |= field.cell(p).is_grounded();
      chain.min = (chain.min.0.min(x), chain.min.1.min(y));
      chain.max = (chain.max.0.max(x), chain.max.1.max(y));
      for n in directions_diag(stride, p) {
        if labels[n] == u32::MAX && field.cell(n).is_live_players_point(player) {
          labels[n] = id;
          q.push_back(n);
        }
      }
    }
    chains.push(chain);
  }
  let dots: u32 = chains.iter().map(|c| c.size).sum();
  if dots == 0 {
    return;
  }
  let dots_f = dots as f64;
  acc.sides += 1.0;
  acc.dots += dots_f;
  acc.chains += chains.len() as f64;
  acc.same += chains.iter().map(|c| (c.size as f64).powi(2)).sum::<f64>() / dots_f.powi(2);
  acc.largest += chains.iter().map(|c| c.size).max().unwrap() as f64 / dots_f;
  acc.span4 += chains
    .iter()
    .filter(|c| c.span() > 4)
    .map(|c| c.size as f64)
    .sum::<f64>();
  acc.span8 += chains
    .iter()
    .filter(|c| c.span() > 8)
    .map(|c| c.size as f64)
    .sum::<f64>();
  let ungrounded: Vec<&Chain> = chains.iter().filter(|c| !c.grounded).collect();
  let ungrounded_dots: u32 = ungrounded.iter().map(|c| c.size).sum();
  acc.ungrounded += ungrounded_dots as f64;
  acc.ungrounded_chains += ungrounded.len() as f64;
  acc.ungrounded_span8 += ungrounded
    .iter()
    .filter(|c| c.span() > 8)
    .map(|c| c.size as f64)
    .sum::<f64>();
  if ungrounded_dots > 0 {
    acc.ungrounded_sides += 1.0;
    acc.ungrounded_same +=
      ungrounded.iter().map(|c| (c.size as f64).powi(2)).sum::<f64>() / (ungrounded_dots as f64).powi(2);
  }
  for pos in field.min_pos()..=field.max_pos() {
    if !field.cell(pos).is_putting_allowed() {
      continue;
    }
    let mut ids = [u32::MAX; 8];
    let mut count = 0;
    for n in directions_diag(stride, pos) {
      let id = labels[n];
      if id != u32::MAX && !ids[..count].contains(&id) {
        ids[count] = id;
        count += 1;
      }
    }
    if count >= 2 {
      acc.joining += 1.0;
      if ids[..count].iter().any(|&id| chains[id as usize].span() > 4) {
        acc.joining_far += 1.0;
      }
    }
  }
}

pub fn run(files: &[PathBuf], count: Option<usize>) -> Result<()> {
  let mut accs = [Acc::default(); PHASES + 1];
  let mut lengths = Vec::new();
  for path in files {
    for game in read_games(path, count)? {
      let mut field = game.field();
      let mut labels = vec![u32::MAX; field.length()];
      let len = game.moves.len();
      lengths.push(len as f64);
      for (i, &(pos, player)) in game.moves.iter().enumerate() {
        assert!(field.put_point(pos, player));
        field.update_grounded();
        let phase = (i * PHASES / len).min(PHASES - 1);
        for p in [Player::Red, Player::Black] {
          analyse(&field, p, &mut labels, &mut accs[phase]);
        }
      }
    }
  }
  for phase in 0..PHASES {
    let a = accs[phase];
    let all = &mut accs[PHASES];
    macro_rules! add { ($($f:ident),*) => { $(all.$f += a.$f;)* } }
    add!(
      sides,
      dots,
      chains,
      same,
      largest,
      span4,
      span8,
      ungrounded,
      ungrounded_chains,
      ungrounded_span8,
      ungrounded_same,
      ungrounded_sides,
      joining,
      joining_far
    );
  }
  lengths.sort_by(f64::total_cmp);
  println!(
    "{} games, median length {} moves\n",
    lengths.len(),
    lengths.get(lengths.len() / 2).copied().unwrap_or(0.0)
  );
  let mut table = Table::new(
    "Chains of live dots per side (8-connected), by fifth of the game",
    &[
      "phase",
      "dots",
      "chains",
      "P(same)",
      "largest",
      "span>4",
      "span>8",
      "ungr. dots",
      "ungr. chains",
      "ungr. P(same)",
      "ungr. span>8",
      "joining cells",
      "joining far",
    ],
  );
  for (phase, a) in accs.iter().enumerate() {
    let label = if phase < PHASES {
      format!("{}-{}%", phase * 100 / PHASES, (phase + 1) * 100 / PHASES)
    } else {
      "all".to_string()
    };
    let s = a.sides;
    table.row(vec![
      label,
      fmt(Some(a.dots / s), 1),
      fmt(Some(a.chains / s), 1),
      fmt(Some(a.same / s), 2),
      fmt(Some(a.largest / s), 2),
      fmt(Some(a.span4 / a.dots), 2),
      fmt(Some(a.span8 / a.dots), 2),
      fmt(Some(a.ungrounded / s), 1),
      fmt(Some(a.ungrounded_chains / s), 1),
      fmt(Some(a.ungrounded_same / a.ungrounded_sides), 2),
      fmt(Some(a.ungrounded_span8 / a.ungrounded.max(1.0)), 2),
      fmt(Some(a.joining / s), 1),
      fmt(Some(a.joining_far / s), 1),
    ]);
  }
  table.print();
  Ok(())
}
