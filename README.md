Oppai-rs
====

[![Travis Build Status](https://travis-ci.org/kurnevsky/oppai-rs.svg?branch=master)](https://travis-ci.org/kurnevsky/oppai-rs)

Oppai-rs is an artificial intelligence for the game of points.

It's written in rust language and implements "points console AI protocol v6". (See /doc/PointsAIProtocol6.txt for details.)

You can play with it using "[missile](https://github.com/kurnevsky/missile)".

Features
====

* Two algorithms for searching the optimal move: UCT, Minimax.
* Two implementations of Minimax search: NegaScout, MTD(f).
* UCT caching that persists between moves.
* Trajectories for moves pruning in the Minimax search tree.
* Lock-free multi-threading for both Minimax and UCT.
* Transposition table using Zobrist hashing for Minimax.
* DFA-based patterns searching.
* DSU to optimize capturing (optional).
* Time-based (`gen_move_with_time`) and complexity-based (`gen_move_with_complexity`) calculations.

Running
====

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

If you have nightly rust and want to see clippy warnings, compile with:

```sh
cargo build --features clippy
```

Testing
====

You can run test with:

```sh
cargo test
```

If you want to see log output during tests running you can use RUST_LOG environment variable:

```sh
RUST_LOG=debug cargo test
```

Also if you have nightly rust you can run benchmarks with:

```sh
cargo bench --features bench
```

TODO
====
* Parallel trajectories building.
* Best Node Search algorithm (see [link](https://dspace.lu.lv/dspace/bitstream/handle/7/4903/38550-Dmitrijs_Rutko_2013.pdf)).
* Cache built DFA for fast patterns loading.
* Fill debuts database.
* Fill heuristics database.
* Use patterns for UCT random games (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Use patterns for Minimax best move prediction.
* Complex estimating function for Minimax (see [link](https://www.gnu.org/software/gnugo/gnugo_13.html#SEC167))
* Smart time control for UCT (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Smart time control for Minimax.
* Think on enemy's move.
* Ladders solver.
* Fractional komi support.
* Split trajectories by groups for Minimax (see [link](https://www.icsi.berkeley.edu/ftp/global/pub/techreports/1996/tr-96-030.pdf)).

License
====

This project is licensed under AGPL version 3 or (at your option) any later version. See LICENSE.txt for details.

Copyright (C) 2015 Kurnevsky Evgeny, Vasya Novikov
