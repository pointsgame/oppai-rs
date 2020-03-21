use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;

pub fn is_last_move_stupid(field: &Field, pos: Pos, player: Player) -> bool {
  let delta_score = field.get_delta_score(player);
  delta_score < 0
    || delta_score == 0
      && {
        let enemy = player.next();
        let enemies_around = field
          .directions(pos)
          .iter()
          .filter(|&&pos| field.cell(pos).is_players_point(enemy))
          .count();
        enemies_around == 3
      }
      && field
        .directions(pos)
        .iter()
        .any(|&pos| field.cell(pos).is_putting_allowed())
}

pub fn is_penult_move_stupid(field: &Field) -> bool {
  let moves_count = field.moves_count();
  moves_count > 1 && field.cell(field.points_seq()[moves_count - 2]).is_captured()
}
