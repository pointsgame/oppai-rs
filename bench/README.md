# Benchmark

## Backends

### Native

Build with:

```sh
cargo build --release --package=oppai-bench
```

Run with:

```sh
time ../target/release/oppai-bench -w 39 -h 32 -n 100000 -s 7
```

### Wasmtime

Build with:

```sh
cargo build --release --package=oppai-bench --target=wasm32-wasip1
```

Run with:

```sh
time wasmtime ../target/wasm32-wasip1/release/oppai-bench.wasm -w 39 -h 32 -n 100000 -s 7
```
