Opai-rs
====

Opai-rs an artificial intelligence for the game of points.

It's written in rust language and implements "points console AI protocol v6". (See /doc/PointsAIProtocol6.txt for details.)

Running
====

In order to build opai-rs you need a _nightly_ rust installed on your system.

Rustc version "nightly-2015-11-01" is known to be able to build opai-rs. You can use "multirust" to specify and update nightly versions of rust. [https://github.com/brson/multirust](https://github.com/brson/multirust)

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

License
====

This project is licensed under AGPL version 3 or (at your option) any later version. See LICENSE.txt for details.

Copyright (C) Kurnevsky Evgeny 2015
