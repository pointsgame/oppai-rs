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

/// The search value from which the mover counts as winning for
/// [`MoveKinds`]: the early-finish term of the utility grows with the square
/// of the value, so it is the clearly winning side it steers.
const WINNING_VALUE: f64 = 0.9;

/// What the moves of a game were, by where the mover stood before them: behind,
/// level and ahead on the board, komi included, or winning by the search's
/// value, whatever the board.
#[derive(Default, Clone, Copy)]
struct MoveKinds {
  total: [usize; 4],
  /// Dots near the edge of the board - on its two outer lines - that capture
  /// nothing and leave the mover's ungrounded dots as they were: grounded from
  /// the start, by the edge or a grounded chain they join, and grounding
  /// nothing else. Such a move changes nothing the utility reads, so a search
  /// that penalises new ungrounded dots finds it free.
  idle_edge: [usize; 4],
  /// Moves after which the mover has fewer ungrounded dots.
  grounding: [usize; 4],
}

impl MoveKinds {
  fn add(&mut self, other: &MoveKinds) {
    for i in 0..4 {
      self.total[i] += other.total[i];
      self.idle_edge[i] += other.idle_edge[i];
      self.grounding[i] += other.grounding[i];
    }
  }
}

/// How many game moves after its placement a dot counts as a recent stone of
/// the move that grounds it.
const RECENT_MOVES: usize = 10;

/// How the mover handles its ungrounded dots, split by whether it was winning
/// by the search's value ([`WINNING_VALUE`]) or not: index 0 not winning, 1
/// winning. A side that grounds its groups well places ungrounded stones
/// between a group and the edge or a grounded chain and then connects them,
/// rather than crawling from the edge one grounded dot at a time.
#[derive(Default, Clone, Copy)]
struct Grounding {
  moves: [usize; 2],
  captures: [usize; 2],
  /// Moves after which the mover has fewer ungrounded dots.
  grounding: [usize; 2],
  /// Moves that capture nothing and leave the mover's ungrounded dots as they
  /// were: the new dot is grounded and grounds nothing else.
  same: [usize; 2],
  /// Moves that add an ungrounded dot.
  adding: [usize; 2],
  /// The dots the grounding moves grounded, the new dot not counted; moves
  /// that capture are left out, though they can ground dots too.
  grounded_dots: [usize; 2],
  /// Of them, those placed ungrounded by the mover in the last
  /// [`RECENT_MOVES`] moves.
  recent_stones: [usize; 2],
  /// Grounding moves that grounded at least one recent stone.
  grounding_recent: [usize; 2],
  /// The ungrounded dots placed, by what became of them: grounded within
  /// [`RECENT_MOVES`], grounded later, captured, ungrounded at the end.
  stones: [[usize; 4]; 2],
}

impl Grounding {
  fn add(&mut self, other: &Grounding) {
    for i in 0..2 {
      self.moves[i] += other.moves[i];
      self.captures[i] += other.captures[i];
      self.grounding[i] += other.grounding[i];
      self.same[i] += other.same[i];
      self.adding[i] += other.adding[i];
      self.grounded_dots[i] += other.grounded_dots[i];
      self.recent_stones[i] += other.recent_stones[i];
      self.grounding_recent[i] += other.grounding_recent[i];
      for k in 0..4 {
        self.stones[i][k] += other.stones[i][k];
      }
    }
  }
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
  /// Decided games won on the board: the moves after the winner's lead in
  /// captured dots, komi left out, last rose above zero - how long it took to
  /// end the game once ahead for good, whatever the net thought of it. The
  /// komi is left out because a half komi puts one side ahead from the start.
  lead_tail: Option<usize>,
  /// The moves after the last capture, if there was one.
  capture_tail: Option<usize>,
  /// Decided games: the winner's ungrounded dots at the end.
  winner_ungrounded: Option<u32>,
  settled: Option<Settled>,
  moves: MoveKinds,
  grounding: Grounding,
  /// Points captured by the winner and the loser - red and black for a draw.
  captured: (i32, i32),
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
    let mut last_capture = None;
    // The last move after which the eventual winner was not ahead on the board.
    let mut last_not_ahead = None;
    let mut ever_ahead = false;
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

    let mut moves = MoveKinds::default();
    let mut grounding = Grounding::default();
    // The ungrounded dots of both sides not yet grounded or captured: where,
    // whose, the move that placed them and whether its mover was winning.
    let mut stones: Vec<(oppai_field::field::Pos, Player, usize, usize)> = Vec::new();
    for (i, &(pos, player)) in game.moves.iter().enumerate() {
      let own = |field: &oppai_field::field::Field| match player {
        Player::Red => field.non_grounded_red,
        Player::Black => field.non_grounded_black,
      };
      let komi_x_2 = if player == Player::Red {
        game.komi_x_2
      } else {
        -game.komi_x_2
      };
      let value = (i + values.len())
        .checked_sub(game.moves.len())
        .map(|k| if player == Player::Red { values[k] } else { -values[k] });
      let lead = match (field.score(player) * 2 + komi_x_2).signum() {
        _ if value.is_some_and(|v| v >= WINNING_VALUE) => 3,
        -1 => 0,
        0 => 1,
        _ => 2,
      };
      let (x, y) = field.to_xy(pos);
      let near_edge = x.min(y).min(game.width - 1 - x).min(game.height - 1 - y) <= 1;
      if Some(i) == settled_start
        && let Some(settled) = settled.as_mut()
      {
        settled.ungrounded_at_start = field.non_grounded_red + field.non_grounded_black;
      }
      let before = own(&field);
      assert!(field.put_point(pos, player));
      field.update_grounded();
      moves.total[lead] += 1;
      if own(&field) < before {
        moves.grounding[lead] += 1;
      }
      if near_edge && own(&field) == before && field.get_delta_score(player) == 0 {
        moves.idle_edge[lead] += 1;
      }
      let winning = usize::from(lead == 3);
      let after = own(&field);
      grounding.moves[winning] += 1;
      if field.get_delta_score(player) != 0 {
        grounding.captures[winning] += 1;
      } else {
        match after.cmp(&before) {
          std::cmp::Ordering::Less => grounding.grounding[winning] += 1,
          std::cmp::Ordering::Equal => grounding.same[winning] += 1,
          std::cmp::Ordering::Greater => grounding.adding[winning] += 1,
        }
      }
      let grounding_move = field.get_delta_score(player) == 0 && after < before;
      let mut recent = 0;
      stones.retain(|&(stone, owner, placed, stone_winning)| {
        let cell = field.cell(stone);
        let fate = if cell.is_captured() {
          2
        } else if cell.is_grounded() {
          if owner == player && grounding_move {
            grounding.grounded_dots[winning] += 1;
            if i - placed <= RECENT_MOVES {
              recent += 1;
            }
          }
          if i - placed <= RECENT_MOVES { 0 } else { 1 }
        } else {
          return true;
        };
        grounding.stones[stone_winning][fate] += 1;
        false
      });
      grounding.recent_stones[winning] += recent;
      if recent > 0 {
        grounding.grounding_recent[winning] += 1;
      }
      let cell = field.cell(pos);
      if !cell.is_captured() && !cell.is_grounded() {
        stones.push((pos, player, i, winning));
      }
      if field.get_delta_score(player) != 0 {
        captures += 1;
        first_capture.get_or_insert(i);
        last_capture = Some(i);
      }
      if field.score(Player::Red) * result_x_2.signum() <= 0 {
        last_not_ahead = Some(i);
      } else {
        ever_ahead = true;
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

    for &(_, _, _, stone_winning) in &stones {
      grounding.stones[stone_winning][3] += 1;
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
      lead_tail: (result_x_2 != 0 && ever_ahead && last_not_ahead != Some(game.moves.len() - 1))
        .then(|| last_not_ahead.map_or(game.moves.len(), |i| game.moves.len() - i - 1)),
      capture_tail: last_capture.map(|i| game.moves.len() - i - 1),
      winner_ungrounded: (result_x_2 != 0).then_some(if result_x_2 > 0 {
        field.non_grounded_red
      } else {
        field.non_grounded_black
      }),
      settled,
      moves,
      grounding,
      captured: if result_x_2 < 0 {
        (field.score_black, field.score_red)
      } else {
        (field.score_red, field.score_black)
      },
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
  moves: Vec<String>,
  grounding: Vec<Vec<String>>,
  scores: Vec<String>,
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
  let lead_tails: Vec<f64> = decided.iter().filter_map(|g| g.lead_tail.map(|t| t as f64)).collect();
  let capture_tails: Vec<f64> = decided
    .iter()
    .filter_map(|g| g.capture_tail.map(|t| t as f64))
    .collect();
  let median_p90 = |values: &[f64]| format!("{} / {}", fmt(median(values), 0), fmt(quantile(values, 0.9), 0));
  let decided_row = vec![
    name.to_string(),
    decided.len().to_string(),
    median_p90(&tails),
    fmt(mean(&tails), 1),
    median_p90(&lead_tails),
    median_p90(&capture_tails),
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

  let mut kinds = MoveKinds::default();
  for game in &all {
    kinds.add(&game.moves);
  }
  let mut moves_row = vec![name.to_string()];
  for lead in 0..4 {
    moves_row.push(format!(
      "{} / {} / {}",
      kinds.total[lead],
      percent(kinds.idle_edge[lead], kinds.total[lead]),
      percent(kinds.grounding[lead], kinds.total[lead])
    ));
  }

  let mut g = Grounding::default();
  for game in &all {
    g.add(&game.grounding);
  }
  let grounding_rows = (0..2)
    .map(|w| {
      let stones = g.stones[w].iter().sum();
      vec![
        name.to_string(),
        (if w == 1 { "winning" } else { "not winning" }).to_string(),
        g.moves[w].to_string(),
        format!(
          "{} / {} / {} / {}",
          percent(g.captures[w], g.moves[w]),
          percent(g.grounding[w], g.moves[w]),
          percent(g.same[w], g.moves[w]),
          percent(g.adding[w], g.moves[w])
        ),
        fmt(
          (g.grounding[w] > 0).then(|| g.grounded_dots[w] as f64 / g.grounding[w] as f64),
          1,
        ),
        percent(g.grounding_recent[w], g.grounding[w]),
        percent(g.recent_stones[w], g.grounded_dots[w]),
        format!(
          "{} / {} / {} / {}",
          percent(g.stones[w][0], stones),
          percent(g.stones[w][1], stones),
          percent(g.stones[w][2], stones),
          percent(g.stones[w][3], stones)
        ),
      ]
    })
    .collect();

  // The winner's margin in points, komi included.
  let margins: Vec<f64> = all.iter().map(|g| (g.result_x_2.abs() as f64) / 2.0).collect();
  let bucket = |low: f64, high: f64| {
    percent(
      margins.iter().filter(|&&m| m >= low && m <= high).count(),
      margins.len(),
    )
  };
  let max = margins
    .iter()
    .copied()
    .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
  let winner_captured: Vec<f64> = decided.iter().map(|g| g.captured.0 as f64).collect();
  let loser_captured: Vec<f64> = decided.iter().map(|g| g.captured.1 as f64).collect();
  let scores_row = vec![
    name.to_string(),
    margins.len().to_string(),
    format!(
      "{} / {} / {} / {} / {} / {}",
      bucket(0.0, 0.0),
      bucket(0.5, 2.0),
      bucket(2.5, 5.0),
      bucket(5.5, 10.0),
      bucket(10.5, 20.0),
      bucket(20.5, f64::INFINITY)
    ),
    format!(
      "{} / {} / {} / {}",
      fmt(median(&margins), 1),
      fmt(quantile(&margins, 0.9), 1),
      fmt(quantile(&margins, 0.99), 1),
      fmt(max, 1)
    ),
    format!("{} / {}", fmt(mean(&winner_captured), 1), fmt(mean(&loser_captured), 1)),
  ];

  Rows {
    outcomes,
    lengths: lengths_row,
    decided: decided_row,
    draws: draws_row,
    moves: moves_row,
    grounding: grounding_rows,
    scores: scores_row,
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
    "Decided games: moves to the end after each event, median / p90",
    &[
      "games",
      "n",
      &format!("winner's search value >= {WON_VALUE}"),
      "mean",
      "winner ahead on the board for good",
      "last capture",
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

  let mut moves = Table::new(
    &format!(
      "Moves by where the mover stood: moves / idle edge moves (on the two outer lines, no capture, mover's ungrounded dots unchanged) / grounding the mover's dots"
    ),
    &[
      "games",
      "behind",
      "level",
      "ahead",
      &format!("winning (value >= {WINNING_VALUE})"),
    ],
  );

  let mut scores = Table::new(
    "Final scores: the winner's margin in points, komi included, 0 for a draw",
    &[
      "games",
      "n",
      "margin 0 / 0.5-2 / 2.5-5 / 5.5-10 / 10.5-20 / >20",
      "margin median / p90 / p99 / max",
      "decided: points captured by winner / loser",
    ],
  );
  let mut grounding = Table::new(
    &format!(
      "Ungrounded dots by whether the mover was winning (value >= {WINNING_VALUE}); recent: placed ungrounded in the last {RECENT_MOVES} moves"
    ),
    &[
      "games",
      "mover",
      "moves",
      "capture / grounding / grounded, no change / new ungrounded",
      "dots grounded per grounding move",
      "grounding moves with recent stones",
      "recent stones of grounded dots",
      "new ungrounded dots: grounded in 10 / later / captured / never",
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
      moves.row(rows.moves);
      scores.row(rows.scores);
      for row in rows.grounding {
        grounding.row(row);
      }
    }
  }

  outcomes.print();
  lengths.print();
  decided.print();
  draws.print();
  moves.print();
  grounding.print();
  scores.print();
  Ok(())
}
