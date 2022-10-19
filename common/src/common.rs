use oppai_field::field::{to_x, to_y, Field, Pos};
use oppai_field::player::Player;

fn is_corner(width: u32, height: u32, pos: Pos) -> bool {
  let x = to_x(width, pos);
  let y = to_y(width, pos);
  (x == 0 || x == width - 1) && (y == 0 || y == height - 1)
}

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
  delta_score < 0 || delta_score == 0 && is_trap(field, player, pos) || is_corner(field.width(), field.height(), pos)
}

pub fn is_penult_move_stupid(field: &Field) -> bool {
  let moves_count = field.moves_count();
  moves_count > 1 && field.cell(field.points_seq()[moves_count - 2]).is_captured()
}
