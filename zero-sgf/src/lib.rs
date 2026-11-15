use std::iter;

use oppai_field::{
  field::{Pos, to_pos, to_x, to_y},
  player::Player,
};
use oppai_sgf::{from_coordinate, to_coordinate};
use oppai_zero::episode::{SidePosition, Visits};
use sgf_parse::{SgfNode, unknown_game::Prop};

/// The two letters of a position.
fn coordinates(stride: u32, pos: Pos) -> String {
  format!(
    "{}{}",
    from_coordinate(to_x(stride, pos) as u8) as char,
    from_coordinate(to_y(stride, pos) as u8) as char,
  )
}

/// The position the first two letters of `s` name.
fn parse_coordinates(stride: u32, s: &str) -> Pos {
  let x = to_coordinate(s.as_bytes()[0]) as u32;
  let y = to_coordinate(s.as_bytes()[1]) as u32;
  to_pos(stride, x, y)
}

/// The properties recording a search: `ZR` with the training weight, the
/// surprise, the values and the search weights, and `ZQ` with the per-move q
/// values when there are any.
fn visits_properties(visits: &Visits, stride: u32) -> Vec<Prop> {
  let Visits(weights, target_weight, surprise, value, raw_value, q_values) = visits;
  let mut properties = vec![Prop::Unknown(
    "ZR".into(),
    iter::once(target_weight.to_string())
      .chain([surprise, value, raw_value].map(|value| value.to_string()))
      .chain(
        weights
          .iter()
          .map(|&(pos, weight)| format!("{}{}", coordinates(stride, pos), weight)),
      )
      .collect(),
  )];

  // The per-move q values go into a property of their own rather than the ZR
  // entries, so data recorded without them keeps parsing and the absence
  // stays distinguishable from a search with no explored children.
  if !q_values.is_empty() {
    properties.push(Prop::Unknown(
      "ZQ".into(),
      q_values
        .iter()
        .map(|&(pos, weight, q, score)| format!("{}{}:{}:{}", coordinates(stride, pos), weight, q, score))
        .collect(),
    ));
  }

  properties
}

/// The search recorded on a node, if there is one.
fn node_visits(node: &SgfNode<Prop>, stride: u32) -> Option<Visits> {
  let visits = match node.get_property("ZR") {
    Some(Prop::Unknown(_, visits)) => visits,
    _ => return None,
  };
  // The training weight of the row. Data recorded when this was a boolean
  // full-search flag loads as weight 1 or 0, which is exactly what those
  // searches weighed.
  let target_weight = match visits[0].as_str() {
    "true" => 1.0,
    "false" => 0.0,
    weight => weight.parse().unwrap(),
  };
  // The policy surprise, search value and raw network value are stored
  // after the training weight. Older self-play data predates some or all
  // of them, so parse greedily and fall back to 0 - weight entries always
  // start with a coordinate letter and so never parse as a float.
  let mut numbers = [0.0f64; 3];
  let mut rest = &visits[1..];
  for number in &mut numbers {
    if let Some(Ok(value)) = rest.first().map(|s| s.parse::<f64>()) {
      *number = value;
      rest = &rest[1..];
    } else {
      break;
    }
  }
  let [surprise, value, raw_value] = numbers;
  // Data recorded before per-move q values were stored has no ZQ property
  // and loads with an empty list, which the training loss masks out.
  let q_values = match node.get_property("ZQ") {
    Some(Prop::Unknown(_, entries)) => entries
      .iter()
      .filter_map(|s| {
        let (weight, rest) = s[2..].split_once(':')?;
        // Entries recorded before the per-move score was stored have no
        // second separator. They are dropped rather than given a made-up
        // score: the loss has a single weight for both targets, so a zero
        // score would be trained towards as if the search had settled on
        // it.
        let (q, score) = rest.split_once(':')?;
        Some((
          parse_coordinates(stride, s),
          weight.parse().unwrap(),
          q.parse().unwrap(),
          score.parse().unwrap(),
        ))
      })
      .collect(),
    _ => Vec::new(),
  };
  Some(Visits(
    rest
      .iter()
      // Search weights are fractional. Data predating uncertainty weighting
      // stores integer visit counts, which parse as the weights they are -
      // one per playout.
      .map(|s| (parse_coordinates(stride, s), s[2..].parse().unwrap()))
      .collect(),
    target_weight,
    surprise,
    value,
    raw_value,
    q_values,
  ))
}

/// The property of a move.
fn move_property(stride: u32, player: Player, pos: Pos) -> Prop {
  match player {
    Player::Red => Prop::W(coordinates(stride, pos)),
    Player::Black => Prop::B(coordinates(stride, pos)),
  }
}

/// The move of a node, if it has one. A move may carry the chains it
/// surrounds after its coordinates, which say nothing about the move itself.
fn node_move(node: &SgfNode<Prop>, stride: u32) -> Option<(Player, Pos)> {
  node.properties.iter().find_map(|property| match property {
    Prop::B(s) if s.len() >= 2 => Some((Player::Black, parse_coordinates(stride, s))),
    Prop::W(s) if s.len() >= 2 => Some((Player::Red, parse_coordinates(stride, s))),
    _ => None,
  })
}

pub fn visits_to_sgf(mut node: &mut SgfNode<Prop>, visits: &[Visits], stride: u32, moves_count: usize) {
  for _ in 0..moves_count - visits.len() {
    node = &mut node.children[0];
  }

  for visits in visits {
    node = &mut node.children[0];
    node.properties.extend(visits_properties(visits, stride));
  }
}

pub fn sgf_to_visits(node: &SgfNode<Prop>, stride: u32) -> Vec<Visits> {
  node
    .main_variation()
    .filter_map(|node| node_visits(node, stride))
    .collect()
}

/// Records side positions as variations of the game `root` holds as its main
/// line: each side line branches off the main line where it does in the game,
/// and its search goes onto the node of its last move - the move that search
/// chose, the way a search of the main line goes onto the move it chose.
///
/// Side lines starting the same way share their nodes, so a side line that went
/// on after its first search runs through the node holding that search. Two
/// side lines ending at the very same node would be two searches of one
/// position, and only the first is kept.
pub fn side_positions_to_sgf(root: &mut SgfNode<Prop>, side_positions: &[SidePosition], stride: u32) {
  for side in side_positions {
    let mut node = &mut *root;
    for _ in 0..side.branch {
      node = &mut node.children[0];
    }
    let mut player = side.player;
    for &pos in &side.moves {
      // The main line stays the first child, which is what reading the game
      // follows.
      let index = match node
        .children
        .iter()
        .position(|child| node_move(child, stride) == Some((player, pos)))
      {
        Some(index) => index,
        None => {
          node.children.push(SgfNode::new(
            vec![move_property(stride, player, pos)],
            Vec::new(),
            false,
          ));
          node.children.len() - 1
        }
      };
      node = &mut node.children[index];
      player = player.next();
    }
    if node.get_property("ZR").is_none() {
      node.properties.extend(visits_properties(&side.visits, stride));
    }
  }
}

/// The side positions recorded as variations of the game `root` holds as its
/// main line, see [`side_positions_to_sgf`].
pub fn sgf_to_side_positions(root: &SgfNode<Prop>, stride: u32) -> Vec<SidePosition> {
  fn collect(
    node: &SgfNode<Prop>,
    stride: u32,
    branch: usize,
    moves: &mut Vec<(Player, Pos)>,
    side_positions: &mut Vec<SidePosition>,
  ) {
    let Some(colored_move) = node_move(node, stride) else {
      return;
    };
    moves.push(colored_move);
    if let Some(visits) = node_visits(node, stride) {
      side_positions.push(SidePosition {
        branch,
        player: moves[0].0,
        moves: moves.iter().map(|&(_, pos)| pos).collect(),
        visits,
      });
    }
    for child in &node.children {
      collect(child, stride, branch, moves, side_positions);
    }
    moves.pop();
  }

  let mut side_positions = Vec::new();
  let mut node = root;
  // The root holds no move, so the node `branch` steps down the main line
  // follows the game's first `branch` moves.
  let mut branch = 0;
  loop {
    for child in node.children.iter().skip(1) {
      collect(child, stride, branch, &mut Vec::new(), &mut side_positions);
    }
    match node.children.first() {
      Some(child) => node = child,
      None => break,
    }
    branch += 1;
  }
  side_positions
}

#[cfg(test)]
mod tests {
  use oppai_field::{any_field::AnyField, construct_field::construct_field, extended_field::ExtendedField};
  use oppai_sgf::to_sgf;
  use oppai_zero::episode::{SidePosition, Visits};
  use rand::SeedableRng;
  use rand_xoshiro::Xoshiro256PlusPlus;
  use sgf_parse::unknown_game::Prop;

  use super::{sgf_to_side_positions, sgf_to_visits, side_positions_to_sgf, visits_to_sgf};

  const SEED: u64 = 7;

  #[test]
  fn save_load_visits() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let visits = vec![Visits(
      vec![
        (field.field().to_pos(0, 0), 1.5),
        (field.field().to_pos(0, 1), 2.25),
        (field.field().to_pos(2, 0), 24.0),
      ],
      1.0,
      0.625,
      0.25,
      -0.125,
      vec![
        (field.field().to_pos(0, 0), 1.5, 0.5, 3.5),
        (field.field().to_pos(2, 0), 24.0, -0.75, -1.5),
      ],
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, field.field().stride, field.field().moves_count());
    let sgf_visits = sgf_to_visits(&node, field.field().stride);
    assert_eq!(sgf_visits, visits);
  }

  /// Data recorded when the q entries carried only the value has no per-move
  /// score to load. Such entries are dropped rather than given a made-up score,
  /// since one weight covers both targets.
  #[test]
  fn load_q_values_recorded_without_scores() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    let visits = vec![Visits(
      vec![(field.field().to_pos(0, 0), 1.5)],
      1.0,
      0.625,
      0.25,
      -0.125,
      vec![(field.field().to_pos(0, 0), 1.5, 0.5, 3.5)],
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, stride, field.field().moves_count());
    let mut sgf_node = &mut node;
    loop {
      for property in &mut sgf_node.properties {
        if let Prop::Unknown(name, values) = property
          && name == "ZQ"
        {
          // The old encoding: coordinates, weight and q value, no score.
          *values = vec!["aa1.5:0.5".to_string()];
        }
      }
      match sgf_node.children.first_mut() {
        Some(child) => sgf_node = child,
        None => break,
      }
    }
    let mut old = visits;
    old[0].5.clear();
    assert_eq!(sgf_to_visits(&node, stride), old);
  }

  /// Self-play data recorded before uncertainty weighting stores integer visit
  /// counts where the weights now go. Those are weights with every playout
  /// weighing one, so they must still load - and normalize to the same policy
  /// target they always did.
  #[test]
  fn load_visits_recorded_as_integers() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    // Legacy data also predates the per-move q values, so they stay empty -
    // and an empty list must not write the property at all.
    let integer = vec![Visits(
      vec![
        (field.field().to_pos(0, 0), 1.0),
        (field.field().to_pos(0, 1), 2.0),
        (field.field().to_pos(2, 0), 3.0),
      ],
      1.0,
      0.625,
      0.25,
      -0.125,
      Vec::new(),
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &integer, stride, field.field().moves_count());
    // Whole weights serialize without a decimal point, exactly as the old
    // integer format did, so this is byte-for-byte the legacy encoding.
    let property = node
      .main_variation()
      .flat_map(|node| node.get_property("ZR"))
      .next()
      .unwrap();
    let Prop::Unknown(_, values) = property else {
      panic!("expected an unknown ZR property");
    };
    assert_eq!(&values[4..], &["aa1", "ab2", "ca3"]);
    assert!(node.main_variation().all(|node| node.get_property("ZQ").is_none()));
    assert_eq!(sgf_to_visits(&node, stride), integer);
  }

  /// Data recorded when the training weight was a boolean full-search flag
  /// stores `true`/`false` where the weight now goes. A full search weighed 1
  /// and a cheap one 0, so that is what the flags load as.
  #[test]
  fn load_visits_recorded_with_full_flag() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let stride = field.field().stride;
    let visits = vec![Visits(
      vec![(field.field().to_pos(0, 0), 1.5)],
      1.0,
      0.625,
      0.25,
      -0.125,
      Vec::new(),
    )];
    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, stride, field.field().moves_count());
    let mut sgf_node = &mut node;
    loop {
      for property in &mut sgf_node.properties {
        if let Prop::Unknown(name, values) = property
          && name == "ZR"
        {
          // The old encoding: a boolean flag where the weight now goes.
          values[0] = "true".to_string();
        }
      }
      match sgf_node.children.first_mut() {
        Some(child) => sgf_node = child,
        None => break,
      }
    }
    assert_eq!(sgf_to_visits(&node, stride), visits);
  }

  /// Side positions are variations off the main line: they load back as they
  /// were saved, a side line that went on shares the nodes of the one it went
  /// on from, a second search of a node already holding one is dropped, and
  /// the main line - the game and its searches - reads as if they weren't
  /// there.
  #[test]
  fn save_load_side_positions() {
    env_logger::try_init().ok();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
    let field: ExtendedField = construct_field(
      &mut rng,
      "
      ....
      .aB.
      .Dc.
      ....
      ",
    )
    .into();
    let f = field.field();
    let stride = f.stride;
    let moves = f.colored_moves().collect::<Vec<_>>();
    let visits = moves
      .iter()
      .map(|&(pos, _)| Visits(vec![(pos, 2.0)], 1.0, 0.5, 0.25, -0.125, Vec::new()))
      .collect::<Vec<_>>();
    let search = |pos, value| {
      Visits(
        vec![(pos, 3.0), (f.to_pos(3, 3), 1.0)],
        1.0,
        0.75,
        value,
        0.0,
        vec![(pos, 3.0, value, 1.5)],
      )
    };
    let player = moves[1].1;
    let first = SidePosition {
      branch: 1,
      player,
      moves: vec![f.to_pos(0, 0), f.to_pos(1, 0)],
      visits: search(f.to_pos(1, 0), 0.5),
    };
    // The same line, gone on with another odd move and a reply.
    let continued = SidePosition {
      branch: 1,
      player,
      moves: vec![f.to_pos(0, 0), f.to_pos(1, 0), f.to_pos(2, 0), f.to_pos(3, 0)],
      visits: search(f.to_pos(3, 0), -0.25),
    };
    // Branching off right at the start.
    let other = SidePosition {
      branch: 0,
      player: moves[0].1,
      moves: vec![f.to_pos(0, 3)],
      visits: search(f.to_pos(0, 3), 0.0),
    };
    let duplicate = SidePosition {
      visits: search(f.to_pos(1, 0), 1.0),
      ..first.clone()
    };
    assert_eq!(player.next(), moves[2].1);

    let mut node = to_sgf(&field).unwrap();
    visits_to_sgf(&mut node, &visits, stride, f.moves_count());
    let main_line = node.clone();
    side_positions_to_sgf(
      &mut node,
      &[first.clone(), continued.clone(), other.clone(), duplicate],
      stride,
    );

    assert_eq!(sgf_to_visits(&node, stride), visits);
    let mut loaded = sgf_to_side_positions(&node, stride);
    loaded.sort_by_key(|side| (side.branch, side.moves.len()));
    assert_eq!(loaded, vec![other, first, continued]);
    // Only the branching nodes were added: the main line is untouched.
    assert_eq!(sgf_to_side_positions(&main_line, stride), Vec::new());
    assert_eq!(node.main_variation().count(), main_line.main_variation().count());
    assert_eq!(node.children.len(), 2);
    assert_eq!(node.children[0].children.len(), 2);
  }
}
