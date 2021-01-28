use iced_futures::futures::stream;
use iced_futures::futures::stream::StreamExt;
use iced_futures::subscription::Recipe;
use iced_futures::BoxStream;
use oppai_bot::bot::Bot;
use oppai_field::field::NonZeroPos;
use oppai_field::player::Player;
use rand::rngs::SmallRng;
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

pub struct FindMove {
  pub bot: Arc<Mutex<Bot<SmallRng>>>,
  pub player: Player,
}

impl<H: Hasher, I> Recipe<H, I> for FindMove {
  type Output = Option<NonZeroPos>;

  fn hash(&self, state: &mut H) {
    use std::hash::Hash;

    std::any::TypeId::of::<Self>().hash(state);
  }

  fn stream(self: Box<Self>, _input: BoxStream<I>) -> BoxStream<Self::Output> {
    Box::pin(stream::once(async move { self.bot.lock().unwrap().best_move(self.player) }).chain(stream::pending()))
  }
}
