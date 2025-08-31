mod config;

use anyhow::Result;
use config::cli_parse;
use oppai_field::field::{Field, Pos, to_pos, to_xy};
use oppai_field::player::Player;
use oppai_sgf::to_sgf_str;
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn all_moves(width: u32, height: u32) -> Vec<Pos> {
  (0..width)
    .flat_map(|x| (0..height).map(move |y| to_pos(width + 1, x, y)))
    .collect()
}

fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let config = cli_parse();
  let mut process = Command::new(config.worker)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .args(config.worker_args)
    .spawn()?;
  let mut stdin = process.stdin.take().ok_or_else(|| anyhow::anyhow!("no stdin"))?;
  let mut stdout = BufReader::new(process.stdout.take().ok_or_else(|| anyhow::anyhow!("no stdout"))?);
  let mut rng = config.seed.map_or_else(SmallRng::from_os_rng, SmallRng::seed_from_u64);
  let mut field = Field::new_from_rng(20, 20, &mut rng);
  let mut moves = all_moves(20, 20);
  let mut s = String::new();

  for i in 0..config.games {
    if i % (config.games / 100) == 0 {
      println!("{}%", i * 100 / config.games);
    }
    field.clear();
    moves.shuffle(&mut rng);
    let mut player = Player::Red;
    for &pos in &moves {
      if !field.is_putting_allowed(pos) {
        continue;
      }
      let (x, y) = to_xy(field.stride, pos);
      if !field.put_point(pos, player) {
        anyhow::bail!("failed to put point");
      }
      writeln!(stdin, "{} {}", x, y)?;
      s.clear();
      stdout.read_line(&mut s)?;
      let mut i = s.trim().split(" ");
      let captured_red = i.next().ok_or_else(|| anyhow::anyhow!("no red"))?.parse()?;
      let captured_black = i.next().ok_or_else(|| anyhow::anyhow!("no black"))?.parse()?;
      if field.score_red != captured_red {
        anyhow::bail!("captured red mismatch:\n{:?}", to_sgf_str(&field.into()));
      }
      if field.score_black != captured_black {
        anyhow::bail!("captured black mismatch:\n{:?}", to_sgf_str(&field.into()));
      }
      player = player.next();
    }
    writeln!(stdin)?;
  }

  process.kill()?;

  Ok(())
}
