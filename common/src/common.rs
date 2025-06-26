use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;

fn is_trap(field: &Field, player: Player, pos: Pos) -> bool {
  let enemy = player.next();
  let directions = field.directions(pos);
  let enemies_around = directions
    .iter()
    .filter(|&&pos| field.cell(pos).is_players_point(enemy))
    .count();
  enemies_around == 3 && directions.iter().any(|&pos| field.cell(pos).is_putting_allowed())
}

pub fn is_last_move_stupid(field: &Field, pos: Pos, player: Player) -> bool {
  let delta_score = field.get_delta_score(player);
  delta_score < 0 || delta_score == 0 && is_trap(field, player, pos) || field.is_corner(pos)
}

pub fn is_penult_move_stupid(field: &Field) -> bool {
  let moves_count = field.moves_count();
  moves_count > 1 && field.cell(field.moves[moves_count - 2]).is_captured()
}
