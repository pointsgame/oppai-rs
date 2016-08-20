Opai-rs
====

Opai-rs an artificial intelligence for the game of points.

It's written in rust language and implements "points console AI protocol v6". (See /doc/PointsAIProtocol6.txt for details.)

You can play with it using "[missile](https://github.com/kurnevsky/missile)".

Features
====

* Two algorithms for searching the optimal move: UCT, NegaScout (principal variation search).
* UCT caching that persists between moves.
* Trajectories for moves pruning in the NegaScout search tree.
* Lock-free multi-threading for both NegaScout and UCT.
* DFA-based patterns searching.
* DSU to optimize capturing.
* Time-based (`gen_move_with_time`) and complexity-based (`gen_move_with_complexity`) calculations.

Running
====

In order to build opai-rs you need a _nightly_ rust installed on your system.

Rustc version 1.12.0-nightly (bbfcb471d 2016-07-18) is known to be able to build opai-rs. You can use "[multirust](https://github.com/brson/multirust)" to specify and update nightly versions of rust.

Once you have rust installed on your system, compile with

```sh
    cargo build --release
```

Run with

```sh
    cargo run --release
```

or with

```sh
    ./target/release/opai-rs
```

TODO
====
* Parallel patterns loading.
* Cache built DFA for fast patterns loading.
* Fill debuts database.
* Fill heuristics database.
* Use patterns for UCT random games (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Use patterns for NegaScout best move prediction.
* Use Zobrist hashes for NegaScout (wait for concurrent hashmap implementation in rust).
* Complex estimating function for NegaScout (see [link](https://www.gnu.org/software/gnugo/gnugo_13.html#SEC167))
* MTD(f).
* Smart time control for UCT (see [link](http://pasky.or.cz/go/pachi-tr.pdf)).
* Smart time control for NegaScout.
* Think on enemy's move.
* Ladders solver.
* Fractional komi support.
* Split trajectories by groups for NegaScout (see [link](https://www.icsi.berkeley.edu/ftp/global/pub/techreports/1996/tr-96-030.pdf)).

License
====

This project is licensed under AGPL version 3 or (at your option) any later version. See LICENSE.txt for details.

Copyright (C) 2015 Kurnevsky Evgeny, Vasya Novikov
