use oppai_field::field::{Field, Pos};
use oppai_field::player::Player;

pub fn is_last_move_stupid(field: &Field, pos: Pos, player: Player) -> bool {
  let delta_score = field.get_delta_score(player);
  delta_score < 0
    || delta_score == 0
      && {
        let enemy = player.next();
        let mut enemies_around = 0u32;
        if field.cell(field.n(pos)).is_players_point(enemy) {
          enemies_around += 1;
        }
        if field.cell(field.s(pos)).is_players_point(enemy) {
          enemies_around += 1;
        }
        if field.cell(field.w(pos)).is_players_point(enemy) {
          enemies_around += 1;
        }
        if field.cell(field.e(pos)).is_players_point(enemy) {
          enemies_around += 1;
        }
        enemies_around == 3
      }
      && {
        field.cell(field.n(pos)).is_putting_allowed()
          || field.cell(field.s(pos)).is_putting_allowed()
          || field.cell(field.w(pos)).is_putting_allowed()
          || field.cell(field.e(pos)).is_putting_allowed()
      }
}

pub fn is_penult_move_stupid(field: &Field) -> bool {
  let moves_count = field.moves_count();
  moves_count > 1 && field.cell(field.points_seq()[moves_count - 2]).is_captured()
}
