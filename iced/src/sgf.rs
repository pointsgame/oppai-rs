use oppai_field::player::Player;
use sgf_parser::{Action, Color, Game, GameNode, GameTree, SgfToken};

pub struct SgfGame {
  pub width: u32,
  pub height: u32,
  pub moves: Vec<(Player, u32, u32)>,
}

fn color_to_player(color: Color) -> Player {
  match color {
    Color::White => Player::Red,
    Color::Black => Player::Black,
  }
}

fn player_to_color(player: Player) -> Color {
  match player {
    Player::Red => Color::White,
    Player::Black => Color::Black,
  }
}

fn to_coordinate(c: u8) -> u8 {
  if c > 96 {
    c - 97
  } else {
    c - 39
  }
}

fn parse_root(root: &GameNode) -> Option<(u32, u32)> {
  let mut game = false;
  let mut size = None;
  for token in &root.tokens {
    match token {
      SgfToken::Game(g) => game = g == &Game::Other(40),
      SgfToken::Size(w, h) => size = Some((*w, *h)),
      _ => {}
    }
  }
  if game {
    size
  } else {
    None
  }
}

pub fn from_sgf(game_tree: GameTree) -> Option<SgfGame> {
  let (width, height) = game_tree.nodes.first().and_then(parse_root)?;

  let mut moves = Vec::new();

  for node in game_tree.nodes {
    for token in &node.tokens {
      match token {
        SgfToken::Move { color, action } => match action {
          Action::Pass => {}
          Action::Move(x, y) => moves.push((color_to_player(*color), (x - 1) as u32, (y - 1) as u32)),
        },
        SgfToken::Add {
          color,
          coordinate: (x, y),
        } => moves.push((color_to_player(*color), (x - 1) as u32, (y - 1) as u32)),
        SgfToken::Invalid((ident, value)) => {
          let player = if ident == "W" {
            Player::Red
          } else if ident == "B" {
            Player::Black
          } else {
            continue;
          };
          if value.len() < 2 {
            warn!("Ignoring too short move {} {}", ident, value);
            continue;
          }
          let x = to_coordinate(value.as_bytes()[0]);
          let y = to_coordinate(value.as_bytes()[1]);
          moves.push((player, x as u32, y as u32));
        }
        _ => {}
      }
    }
  }

  Some(SgfGame { width, height, moves })
}

pub fn to_sgf(game: SgfGame) -> GameTree {
  let root = GameNode {
    tokens: vec![SgfToken::Game(Game::Other(40)), SgfToken::Size(game.width, game.height)],
  };

  let mut nodes = Vec::with_capacity(game.moves.len() + 1);
  nodes.push(root);

  for (player, x, y) in game.moves {
    nodes.push(GameNode {
      tokens: vec![SgfToken::Move {
        color: player_to_color(player),
        action: Action::Move((x + 1) as u8, (y + 1) as u8),
      }],
    });
  }

  GameTree {
    nodes,
    variations: Vec::new(),
  }
}
