//! Reading games files and printing the tables of the reports.

use anyhow::{Context, Result};
use flate2::read::MultiGzDecoder;
use oppai_field::{
  any_field::AnyField,
  extended_field::ExtendedField,
  field::{Field, Pos},
  player::Player,
  zobrist::Zobrist,
};
use oppai_sgf::from_sgf;
use oppai_zero_sgf::sgf_to_visits;
use rand::{SeedableRng, rngs::SmallRng};
use sgf_parse::{GameTree, SgfNode, unknown_game::Prop};
use std::{
  fs::File,
  io::{BufRead, BufReader},
  path::Path,
  sync::Arc,
};

/// A recorded game: its board, komi and moves, and the value the search gave
/// every searched position.
pub struct Game {
  pub width: u32,
  pub height: u32,
  /// Twice the komi, in favour of Red.
  pub komi_x_2: i32,
  pub moves: Vec<(Pos, Player)>,
  /// The value of the search for every searched move, the last ones of
  /// `moves` - the ones before them are the initial position - from the view
  /// of Red, from -1 to 1.
  pub red_values: Vec<f64>,
  pub zobrist: Arc<Zobrist<u64>>,
}

impl Game {
  /// A fresh field of the game's size to replay its moves on.
  pub fn field(&self) -> Field {
    Field::new(self.width, self.height, self.zobrist.clone())
  }

  /// The final position of the game.
  pub fn final_field(&self) -> Field {
    let mut field = self.field();
    for &(pos, player) in &self.moves {
      assert!(field.put_point(pos, player));
      field.update_grounded();
    }
    field
  }

  /// Whether a draw is possible: a whole komi.
  pub fn whole_komi(&self) -> bool {
    self.komi_x_2 % 2 == 0
  }
}

/// The komi recorded in a game, twice, or 0 when there is none.
fn komi_x_2(node: &SgfNode<Prop>) -> i32 {
  node
    .properties
    .iter()
    .find_map(|prop| match prop {
      Prop::Unknown(name, values) if name == "KM" => values.first().and_then(|v| v.parse::<f32>().ok()),
      _ => None,
    })
    .map_or(0, |komi| (komi * 2.0).round() as i32)
}

fn to_game(node: &SgfNode<Prop>, rng: &mut SmallRng) -> Option<Game> {
  let extended = from_sgf::<ExtendedField, _>(node, rng)?;
  let field = extended.field();
  let moves: Vec<_> = field.colored_moves().collect();
  let visits = sgf_to_visits(node, field.stride);
  let initial = moves.len().checked_sub(visits.len())?;
  let red_values = visits
    .iter()
    .zip(&moves[initial..])
    .map(|(visits, &(_, player))| if player == Player::Red { visits.3 } else { -visits.3 })
    .collect();
  Some(Game {
    width: field.width(),
    height: field.height(),
    komi_x_2: komi_x_2(node),
    moves,
    red_values,
    zobrist: field.zobrist_arc(),
  })
}

/// Reads up to `limit` games of a games file: concatenated gzip members, one
/// game per line. Files that are not compressed are read as they are.
pub fn read_games(path: &Path, limit: Option<usize>) -> Result<Vec<Game>> {
  let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
  let gzip = path.extension().is_some_and(|e| e == "gz") || {
    let mut magic = [0u8; 2];
    use std::io::Read;
    File::open(path)?.read_exact(&mut magic).is_ok() && magic == [0x1f, 0x8b]
  };
  let reader: Box<dyn BufRead> = if gzip {
    Box::new(BufReader::new(MultiGzDecoder::new(BufReader::new(file))))
  } else {
    Box::new(BufReader::new(file))
  };
  let mut rng = SmallRng::seed_from_u64(1);
  let mut games = Vec::new();
  for line in reader.lines() {
    let line = line?;
    if line.trim().is_empty() {
      continue;
    }
    for tree in sgf_parse::parse(&line).with_context(|| format!("parsing {}", path.display()))? {
      if limit.is_some_and(|limit| games.len() >= limit) {
        return Ok(games);
      }
      if let GameTree::Unknown(node) = tree
        && let Some(game) = to_game(&node, &mut rng)
      {
        games.push(game);
      }
    }
  }
  Ok(games)
}

/// The name of a file for a table row: the file name without its extension,
/// and without `.gz` before that.
pub fn label(path: &Path) -> String {
  let name = path
    .file_name()
    .map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
  let name = name.strip_suffix(".gz").unwrap_or(&name);
  name.rsplit_once('.').map_or(name, |(stem, _)| stem).to_string()
}

pub fn mean(values: &[f64]) -> Option<f64> {
  (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// The `q` quantile of `values`, the nearest rank.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
  if values.is_empty() {
    return None;
  }
  let mut sorted = values.to_vec();
  sorted.sort_by(f64::total_cmp);
  let index = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
  Some(sorted[index])
}

pub fn median(values: &[f64]) -> Option<f64> {
  quantile(values, 0.5)
}

/// Formats a number with `decimals` digits after the point, or `-` for none.
pub fn fmt(value: Option<f64>, decimals: usize) -> String {
  value.map_or_else(|| "-".to_string(), |v| format!("{v:.decimals$}"))
}

/// Formats a share as a percentage, or `-` when there is nothing to share.
pub fn percent(part: usize, total: usize) -> String {
  if total == 0 {
    "-".to_string()
  } else {
    format!("{:.1}%", 100.0 * part as f64 / total as f64)
  }
}

/// A table printed with its columns aligned: the first column to the left, the
/// rest to the right.
pub struct Table {
  title: String,
  header: Vec<String>,
  rows: Vec<Vec<String>>,
}

impl Table {
  pub fn new(title: &str, header: &[&str]) -> Self {
    Table {
      title: title.to_string(),
      header: header.iter().map(|h| h.to_string()).collect(),
      rows: Vec::new(),
    }
  }

  pub fn row(&mut self, row: Vec<String>) {
    self.rows.push(row);
  }

  pub fn print(&self) {
    let columns = self.header.len();
    let widths: Vec<usize> = (0..columns)
      .map(|c| {
        self
          .rows
          .iter()
          .chain(std::iter::once(&self.header))
          .map(|row| row.get(c).map_or(0, |cell| cell.chars().count()))
          .max()
          .unwrap_or(0)
      })
      .collect();
    println!("{}", self.title);
    for row in std::iter::once(&self.header).chain(&self.rows) {
      let cells: Vec<String> = (0..columns)
        .map(|c| {
          let cell = row.get(c).map_or("", String::as_str);
          if c == 0 {
            format!("{cell:<w$}", w = widths[c])
          } else {
            format!("{cell:>w$}", w = widths[c])
          }
        })
        .collect();
      println!("  {}", cells.join("  "));
    }
    println!();
  }
}
