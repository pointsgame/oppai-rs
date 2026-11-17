//! Statistics of self-play or pit games: outcomes, lengths, how fast decided
//! games end and what drawn games spend their moves on.

use crate::common::{Game, Table, fmt, label, mean, median, percent, quantile, read_games};
use anyhow::Result;
use oppai_field::player::Player;
use std::{collections::BTreeMap, path::PathBuf, thread};

/// The search value from which the winner of a game counts as having it won:
/// the moves after it are the winner finishing the game.
const WON_VALUE: f64 = 0.95;

/// The search value within which a drawn game counts as settled: the moves
/// after the last one outside it are spent in a draw both sides expect.
const LEVEL_VALUE: f64 = 0.2;

/// The settled phase of a drawn game: the moves from the last one the search
/// valued outside [`LEVEL_VALUE`], by what each did to the mover's own
/// ungrounded dots.
struct Settled {
  length: usize,
  grounding: usize,
  same: usize,
  adding: usize,
  /// Both sides' ungrounded dots at the start of the phase.
  ungrounded_at_start: u32,
}

struct GameStats {
  width: u32,
  height: u32,
  komi_x_2: i32,
  length: usize,
  /// Red's lead at the end, komi included, twice.
  result_x_2: i32,
  /// Moves that captured something.
  captures: usize,
  first_capture: Option<usize>,
  mean_abs_value: Option<f64>,
  /// Decided games: the moves after the winner's search value last rose to
  /// [`WON_VALUE`].
  winner_tail: Option<usize>,
  /// Decided games: the winner's ungrounded dots at the end.
  winner_ungrounded: Option<u32>,
  settled: Option<Settled>,
}

impl GameStats {
  fn new(game: &Game) -> Self {
    let values = &game.red_values;
    // How many moves at the end of the game had search values for which `f`
    // holds, or `None` if there is none.
    let tail = |f: &dyn Fn(f64) -> bool| {
      let k = values.iter().rposition(|&v| !f(v)).map_or(0, |k| k + 1);
      (k < values.len()).then(|| values.len() - k)
    };

    let mut field = game.field();
    let mut captures = 0;
    let mut first_capture = None;
    let result_x_2 = game.final_field().score(Player::Red) * 2 + game.komi_x_2;
    let sign = result_x_2.signum() as f64;
    let winner_tail = (result_x_2 != 0).then(|| tail(&|v| v * sign >= WON_VALUE).unwrap_or(0));
    let settled_tail = (result_x_2 == 0).then(|| tail(&|v| v.abs() < LEVEL_VALUE).unwrap_or(0));
    let settled_start = settled_tail.map(|t| game.moves.len() - t);
    let mut settled = settled_tail.map(|length| Settled {
      length,
      grounding: 0,
      same: 0,
      adding: 0,
      ungrounded_at_start: 0,
    });

    for (i, &(pos, player)) in game.moves.iter().enumerate() {
      let own = |field: &oppai_field::field::Field| match player {
        Player::Red => field.non_grounded_red,
        Player::Black => field.non_grounded_black,
      };
      if Some(i) == settled_start
        && let Some(settled) = settled.as_mut()
      {
        settled.ungrounded_at_start = field.non_grounded_red + field.non_grounded_black;
      }
      let before = own(&field);
      assert!(field.put_point(pos, player));
      field.update_grounded();
      if field.get_delta_score(player) != 0 {
        captures += 1;
        first_capture.get_or_insert(i);
      }
      if settled_start.is_some_and(|start| i >= start)
        && let Some(settled) = settled.as_mut()
      {
        match own(&field).cmp(&before) {
          std::cmp::Ordering::Less => settled.grounding += 1,
          std::cmp::Ordering::Equal => settled.same += 1,
          std::cmp::Ordering::Greater => settled.adding += 1,
        }
      }
    }

    GameStats {
      width: game.width,
      height: game.height,
      komi_x_2: game.komi_x_2,
      length: game.moves.len(),
      result_x_2,
      captures,
      first_capture,
      mean_abs_value: mean(&values.iter().map(|v| v.abs()).collect::<Vec<_>>()),
      winner_tail,
      winner_ungrounded: (result_x_2 != 0).then_some(if result_x_2 > 0 {
        field.non_grounded_red
      } else {
        field.non_grounded_black
      }),
      settled,
    }
  }

  fn draw(&self) -> bool {
    self.result_x_2 == 0
  }

  fn whole_komi(&self) -> bool {
    self.komi_x_2 % 2 == 0
  }
}

fn lengths(games: &[&GameStats]) -> Vec<f64> {
  games.iter().map(|g| g.length as f64).collect()
}

fn mean_median(values: &[f64]) -> String {
  format!("{} / {}", fmt(mean(values), 0), fmt(median(values), 0))
}

/// The rows of every table for one group of games.
struct Rows {
  outcomes: Vec<String>,
  lengths: Vec<String>,
  decided: Vec<String>,
  draws: Vec<String>,
}

fn rows(name: &str, games: &[&GameStats]) -> Rows {
  let all: Vec<&GameStats> = games.to_vec();
  let whole: Vec<&GameStats> = all.iter().copied().filter(|g| g.whole_komi()).collect();
  let komi_0: Vec<&GameStats> = all.iter().copied().filter(|g| g.komi_x_2 == 0).collect();
  let half: Vec<&GameStats> = all.iter().copied().filter(|g| !g.whole_komi()).collect();
  let draws: Vec<&GameStats> = all.iter().copied().filter(|g| g.draw()).collect();
  let decided: Vec<&GameStats> = all.iter().copied().filter(|g| !g.draw()).collect();
  let decided_whole: Vec<&GameStats> = whole.iter().copied().filter(|g| !g.draw()).collect();
  let red_wins = decided.iter().filter(|g| g.result_x_2 > 0).count();
  let side = mean(
    &all
      .iter()
      .map(|g| ((g.width * g.height) as f64).sqrt())
      .collect::<Vec<_>>(),
  );
  let mean_abs_value = mean(&all.iter().filter_map(|g| g.mean_abs_value).collect::<Vec<_>>());

  let outcomes = vec![
    name.to_string(),
    all.len().to_string(),
    fmt(side, 1),
    format!(
      "{} / {}",
      whole.len(),
      percent(whole.iter().filter(|g| g.draw()).count(), whole.len())
    ),
    format!(
      "{} / {}",
      komi_0.len(),
      percent(komi_0.iter().filter(|g| g.draw()).count(), komi_0.len())
    ),
    half.len().to_string(),
    percent(red_wins, decided.len()),
    fmt(mean_abs_value, 3),
  ];

  let captures: Vec<f64> = all.iter().map(|g| g.captures as f64).collect();
  let first_capture: Vec<f64> = all.iter().filter_map(|g| g.first_capture.map(|c| c as f64)).collect();
  let lengths_row = vec![
    name.to_string(),
    mean_median(&lengths(&all)),
    mean_median(&lengths(&draws)),
    mean_median(&lengths(&decided_whole)),
    mean_median(&lengths(&half)),
    fmt(mean(&captures), 1),
    fmt(median(&first_capture), 0),
  ];

  let tails: Vec<f64> = decided.iter().filter_map(|g| g.winner_tail.map(|t| t as f64)).collect();
  let winner_ungrounded: Vec<f64> = decided
    .iter()
    .filter_map(|g| g.winner_ungrounded.map(|u| u as f64))
    .collect();
  let decided_row = vec![
    name.to_string(),
    decided.len().to_string(),
    fmt(median(&tails), 0),
    fmt(quantile(&tails, 0.9), 0),
    fmt(mean(&tails), 1),
    fmt(mean(&winner_ungrounded), 1),
  ];

  let settled: Vec<&Settled> = draws.iter().filter_map(|g| g.settled.as_ref()).collect();
  let per_draw = |f: &dyn Fn(&Settled) -> usize| mean(&settled.iter().map(|s| f(s) as f64).collect::<Vec<_>>());
  let settled_moves: usize = settled.iter().map(|s| s.length).sum();
  let adding: usize = settled.iter().map(|s| s.adding).sum();
  let draw_moves: usize = draws.iter().map(|g| g.length).sum();
  let draws_row = vec![
    name.to_string(),
    draws.len().to_string(),
    fmt(per_draw(&|s| s.length), 1),
    percent(settled_moves, draw_moves),
    format!(
      "{} / {} / {}",
      fmt(per_draw(&|s| s.grounding), 1),
      fmt(per_draw(&|s| s.same), 1),
      fmt(per_draw(&|s| s.adding), 1)
    ),
    percent(adding, settled_moves),
    fmt(per_draw(&|s| s.ungrounded_at_start as usize), 1),
  ];

  Rows {
    outcomes,
    lengths: lengths_row,
    decided: decided_row,
    draws: draws_row,
  }
}

/// The group of board sizes a game falls in when the games are split by size:
/// the mean of its sides, in bins of three.
fn size_bin(game: &GameStats) -> (u32, String) {
  let side = (game.width + game.height) / 2;
  let low = side / 3 * 3;
  (low, format!("side {}-{}", low, low + 2))
}

pub fn run(files: &[PathBuf], limit: Option<usize>, by_size: bool) -> Result<()> {
  let stats: Vec<Result<Vec<GameStats>>> = thread::scope(|scope| {
    let handles: Vec<_> = files
      .iter()
      .map(|path| {
        scope.spawn(move || -> Result<Vec<GameStats>> {
          Ok(read_games(path, limit)?.iter().map(GameStats::new).collect())
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
  });

  let mut outcomes = Table::new(
    "Outcomes",
    &[
      "games",
      "n",
      "mean side",
      "whole komi: n / draws",
      "komi 0: n / draws",
      "half komi",
      "red wins of decided",
      "mean |search value|",
    ],
  );
  let mut lengths = Table::new(
    "Game length in moves, mean / median",
    &[
      "games",
      "all",
      "draws",
      "decided, whole komi",
      "half komi",
      "captures per game",
      "first capture (median move)",
    ],
  );
  let mut decided = Table::new(
    &format!("Decided games: moves after the winner's search value last reached {WON_VALUE}"),
    &[
      "games",
      "n",
      "median",
      "p90",
      "mean",
      "winner's ungrounded dots at the end",
    ],
  );
  let mut draws = Table::new(
    &format!(
      "Drawn games: the settled phase, the moves after the search value last left ±{LEVEL_VALUE}, by what each did to the mover's own ungrounded dots"
    ),
    &[
      "games",
      "draws",
      "settled moves per draw",
      "share of draw moves",
      "grounding / same / adding",
      "adding",
      "ungrounded at start",
    ],
  );

  for (path, stats) in files.iter().zip(stats) {
    let stats = stats?;
    let name = label(path);
    let mut groups = vec![(name.clone(), stats.iter().collect::<Vec<_>>())];
    if by_size {
      let mut bins: BTreeMap<u32, (String, Vec<&GameStats>)> = BTreeMap::new();
      for game in &stats {
        let (key, bin) = size_bin(game);
        bins.entry(key).or_insert_with(|| (bin, Vec::new())).1.push(game);
      }
      groups.extend(bins.into_values().map(|(bin, games)| (format!("{name} {bin}"), games)));
    }
    for (group, games) in groups {
      let rows = rows(&group, &games);
      outcomes.row(rows.outcomes);
      lengths.row(rows.lengths);
      decided.row(rows.decided);
      draws.row(rows.draws);
    }
  }

  outcomes.print();
  lengths.print();
  decided.print();
  draws.print();
  Ok(())
}
