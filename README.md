# Oppai-rs

![Logo](/resources/Logo.svg)

Oppai-rs (acronym for "OPen Points Artificial Intelligence") is an artificial
intelligence for the game of points.

You can play with it using [iced module](iced).

## Features

* Two algorithms for searching the optimal move: UCT, Minimax.
* Two implementations of Minimax search: PVS (a.k.a. NegaScout), MTD(f).
* UCT tree reuse between moves.
* Trajectories for moves pruning in the Minimax search tree.
* Lock-free multi-threading for both Minimax and UCT.
* Transposition table using Zobrist hashing for Minimax.
* DFA-based patterns searching.
* DSU to optimize capturing (behind a feature flag since it's good only for UCT).
* Time-based and complexity-based calculations.
* Generic ladders solver.

## Running

Once you have rust installed on your system, compile with:

```sh
cargo build --release
```

Run with:

```sh
cargo run --release
```

or with:

```sh
./target/release/oppai-rs
```

If you are running the produced binary on the same CPU it was built on you might
want to specify `target-cpu` flag:

```sh
RUSTFLAGS="-C target-cpu=native" \
  cargo build --release
```

Depending on your hardware it might increase the performance by up to 10%.

## Testing

You can run test with:

```sh
cargo test
```

If you want to see log output during tests running you can use RUST_LOG
environment variable:

```sh
RUST_LOG=debug cargo test
```

Also if you have nightly rust you can run benchmarks with:

```sh
cargo bench --features bench
```

## Ideas

* Best Node Search algorithm (see [link](https://dspace.lu.lv/dspace/bitstream/handle/7/4903/38550-Dmitrijs_Rutko_2013.pdf)).
* Fill debuts database.
* Fill heuristics database.
* Use patterns for UCT random games (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Use patterns for Minimax best move prediction.
* Complex estimating function for Minimax (see [link](https://www.gnu.org/software/gnugo/gnugo_13.html#SEC167))
* Smart time control for UCT (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Smart time control for Minimax.
* Think on enemy's move.
* Forbid typical losing ladders.
* Split trajectories by groups for Minimax and solve them independently.

## License

This project is licensed under AGPL version 3 or (at your option) any later
version. See LICENSE.txt for details.

Copyright (C) 2015-2024 Kurnevsky Evgeny, Vasya Novikov
