use crate::batch_model::{BatchModel, Closed, batch_model, run_evaluator};
use crate::model::Model;
use futures::join;
use ndarray::{Array1, Array2, Array3, Array4};
use std::cell::{Cell, RefCell};

#[test]
fn batches_across_games_and_pads_sizes() {
  let calls = Cell::new(0usize);
  let model = |inputs: Array4<f64>, global: Array2<f64>, optimism: Array1<f64>| {
    calls.set(calls.get() + 1);
    let (batch, channels, height, width) = inputs.dim();
    assert_eq!((batch, channels, height, width), (3, 3, 5, 4));
    assert_eq!(global.dim(), (3, 1));
    // The two games fill their own regions with ones; the padding must be zero.
    assert_eq!(inputs.sum(), (2 * 3 * 3 * 4 + 3 * 5 * 2) as f64);
    assert_eq!(global.sum(), 1.5);
    // Each game's optimism stays with its own rows, so games searching with
    // different parameters can share one evaluator.
    assert_eq!(optimism.to_vec(), vec![0.25, 0.25, 1.0]);
    // Encode (row, y, x) in the policy so splitting and cropping can be verified.
    let policies = Array3::from_shape_fn((batch, height, width), |(i, y, x)| (i * 10000 + y * 100 + x) as f64);
    let values = Array2::from_shape_fn((batch, 2), |(i, j)| (i * 2 + j) as f64);
    Ok::<_, ()>((policies, values))
  };

  let (handle, requests) = batch_model::<f64>(true);

  let game = |n: usize, h: usize, w: usize, optimism: f64| {
    let model = handle.clone();
    async move {
      let features = Array4::from_elem((n, 3, h, w), 1.0);
      let global = Array2::from_elem((n, 1), 0.5);
      model
        .predict(features, global, Array1::from_elem(n, optimism))
        .await
        .unwrap()
    }
  };

  let game1 = game(2, 3, 4, 0.25);
  let game2 = game(1, 5, 2, 1.0);
  drop(handle);

  let evaluator = async { run_evaluator(&model, requests, usize::MAX).await.unwrap() };
  let ((policy1, value1), (policy2, value2), ()) =
    futures::executor::block_on(async { join!(game1, game2, evaluator) });

  // Both games were served by a single merged forward pass.
  assert_eq!(calls.get(), 1);

  // The first game got rows 0..2 cropped back to its own board size.
  assert_eq!(policy1.dim(), (2, 3, 4));
  assert_eq!(policy1[(0, 0, 0)], 0.0);
  assert_eq!(policy1[(1, 2, 3)], 10203.0);
  assert_eq!(
    value1,
    Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap()
  );

  // The second game got row 2 cropped to its taller, narrower board.
  assert_eq!(policy2.dim(), (1, 5, 2));
  assert_eq!(policy2[(0, 0, 0)], 20000.0);
  assert_eq!(policy2[(0, 4, 1)], 20401.0);
  assert_eq!(value2, Array2::from_shape_vec((1, 2), vec![4.0, 5.0]).unwrap());
}

#[test]
fn finished_games_are_not_waited_for() {
  // Batch sizes of the consecutive forward passes.
  let batches = RefCell::new(Vec::new());
  let model = |inputs: Array4<f64>, _: Array2<f64>, _: Array1<f64>| {
    let (batch, _, height, width) = inputs.dim();
    batches.borrow_mut().push(batch);
    let policies = Array3::from_elem((batch, height, width), 0.25);
    let values = Array2::from_elem((batch, 2), 0.5);
    Ok::<_, ()>((policies, values))
  };

  let (handle, messages) = batch_model::<f64>(true);

  let game = |predictions: usize| {
    let model = handle.clone();
    async move {
      for _ in 0..predictions {
        let features = Array4::from_elem((1, 3, 2, 2), 1.0);
        let global = Array2::from_elem((1, 1), 0.5);
        model.predict(features, global, Array1::zeros(1)).await.unwrap();
      }
    }
  };

  // One game finishes after the first round; the evaluator must keep serving
  // the remaining two without waiting for it.
  let game1 = game(1);
  let game2 = game(2);
  let game3 = game(2);
  drop(handle);

  let evaluator = async { run_evaluator(&model, messages, usize::MAX).await.unwrap() };
  futures::executor::block_on(async { join!(game1, game2, game3, evaluator) });

  assert_eq!(*batches.borrow(), vec![3, 2]);
}

#[test]
fn predict_fails_when_evaluator_is_gone() {
  let (handle, requests) = batch_model::<f64>(true);
  drop(requests);

  let model = handle;
  let features = Array4::from_elem((1, 3, 2, 2), 1.0);
  let global = Array2::from_elem((1, 1), 0.0);
  let result = futures::executor::block_on(model.predict(features, global, Array1::zeros(1)));
  assert_eq!(result.unwrap_err(), Closed);
}

/// The evaluator dispatches once `batch_games` games are waiting instead of all
/// of them, which is what leaves the remaining games free to keep selecting while
/// the device works. Four games with a target of two make two forward passes,
/// where waiting for all of them would have made one.
#[test]
fn a_batch_dispatches_before_every_game_has_submitted() {
  let batches = RefCell::new(Vec::new());
  let model = |inputs: Array4<f64>, _: Array2<f64>, _: Array1<f64>| {
    let (batch, _, height, width) = inputs.dim();
    batches.borrow_mut().push(batch);
    Ok::<_, ()>((
      Array3::from_elem((batch, height, width), 0.25),
      Array2::from_elem((batch, 2), 0.5),
    ))
  };

  let (handle, messages) = batch_model::<f64>(true);
  let game = || {
    let model = handle.clone();
    async move {
      let features = Array4::from_elem((1, 3, 2, 2), 1.0);
      let global = Array2::from_elem((1, 1), 0.5);
      model.predict(features, global, Array1::zeros(1)).await.unwrap();
    }
  };

  let (game1, game2, game3, game4) = (game(), game(), game(), game());
  drop(handle);

  let evaluator = async { run_evaluator(&model, messages, 2).await.unwrap() };
  futures::executor::block_on(async { join!(game1, game2, game3, game4, evaluator) });

  assert_eq!(*batches.borrow(), vec![2, 2]);
}

/// The target is capped at the number of games actually in flight, so asking for
/// bigger batches than there are games must not leave the evaluator waiting for
/// submissions that can never arrive.
#[test]
fn a_target_above_the_game_count_still_dispatches() {
  let batches = RefCell::new(Vec::new());
  let model = |inputs: Array4<f64>, _: Array2<f64>, _: Array1<f64>| {
    let (batch, _, height, width) = inputs.dim();
    batches.borrow_mut().push(batch);
    Ok::<_, ()>((
      Array3::from_elem((batch, height, width), 0.25),
      Array2::from_elem((batch, 2), 0.5),
    ))
  };

  let (handle, messages) = batch_model::<f64>(true);
  let game = || {
    let model = handle.clone();
    async move {
      let features = Array4::from_elem((1, 3, 2, 2), 1.0);
      let global = Array2::from_elem((1, 1), 0.5);
      model.predict(features, global, Array1::zeros(1)).await.unwrap();
    }
  };

  let (game1, game2) = (game(), game());
  drop(handle);

  let evaluator = async { run_evaluator(&model, messages, 10).await.unwrap() };
  futures::executor::block_on(async { join!(game1, game2, evaluator) });

  assert_eq!(*batches.borrow(), vec![2]);
}

/// A per-thread clone source must not be counted as a game of its own: were it
/// counted, the evaluator would wait for a submission from every thread as well
/// as every game, and the threads never submit anything themselves.
#[test]
fn a_clone_source_is_not_counted_as_a_game() {
  let batches = RefCell::new(Vec::new());
  let model = |inputs: Array4<f64>, _: Array2<f64>, _: Array1<f64>| {
    let (batch, _, height, width) = inputs.dim();
    batches.borrow_mut().push(batch);
    Ok::<_, ()>((
      Array3::from_elem((batch, height, width), 0.25),
      Array2::from_elem((batch, 2), 0.5),
    ))
  };

  let (handle, messages) = batch_model::<f64>(true);
  // Two "threads", one game each.
  let sources = [handle.source(), handle.source()];
  drop(handle);
  let game = |source: &BatchModel<f64>| {
    let model = source.clone();
    async move {
      let features = Array4::from_elem((1, 3, 2, 2), 1.0);
      let global = Array2::from_elem((1, 1), 0.5);
      model.predict(features, global, Array1::zeros(1)).await.unwrap();
    }
  };

  let game1 = game(&sources[0]);
  let game2 = game(&sources[1]);
  drop(sources);

  let evaluator = async { run_evaluator(&model, messages, usize::MAX).await.unwrap() };
  futures::executor::block_on(async { join!(game1, game2, evaluator) });

  // Both games were served together, so the evaluator counted two of them - not
  // the two sources on top.
  assert_eq!(*batches.borrow(), vec![2]);
}
