use clap::{Arg, Command, value_parser};
use oppai_field::{
  field::{Field, Pos, to_pos},
  player::Player,
};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

struct Args {
  width: u32,
  height: u32,
  games_number: u32,
  seed: u64,
}

fn cli_parse() -> Args {
  let matches = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .args([
      Arg::new("width")
        .long("width")
        .short('w')
        .help("Field width")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .required(true),
      Arg::new("height")
        .long("height")
        .short('h')
        .help("Field height")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .required(true),
      Arg::new("games-number")
        .long("games-number")
        .short('n')
        .help("Games number")
        .num_args(1)
        .value_parser(value_parser!(u32))
        .required(true),
      Arg::new("seed")
        .long("seed")
        .short('s')
        .help("RNG seed")
        .num_args(1)
        .value_parser(value_parser!(u64))
        .required(true),
    ])
    .get_matches();

  Args {
    width: matches.get_one("width").cloned().unwrap(),
    height: matches.get_one("height").cloned().unwrap(),
    games_number: matches.get_one("games-number").cloned().unwrap(),
    seed: matches.get_one("seed").cloned().unwrap(),
  }
}

struct GamesResult {
  red: u32,
  black: u32,
}

fn all_moves(width: u32, height: u32) -> Vec<Pos> {
  (0..width)
    .flat_map(|x| (0..height).map(move |y| to_pos(width, x, y)))
    .collect()
}

fn main() {
  let args = cli_parse();
  let mut rng = SmallRng::seed_from_u64(args.seed);
  let mut moves = all_moves(args.width, args.height);
  let mut field = Field::new_from_rng(args.width, args.height, &mut rng);
  let mut result = GamesResult { red: 0, black: 0 };
  for _ in 0..args.games_number {
    moves.shuffle(&mut rng);
    for &pos in &moves {
      field.put_point(pos, field.cur_player());
    }
    match field.winner() {
      Some(Player::Red) => result.red += 1,
      Some(Player::Black) => result.black += 1,
      None => {}
    }
    field.clear();
  }
  println!("{}:{}", result.red, result.black);
}
