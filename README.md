# rust-qldpc-decoders

Rust implementations of a series of decoders meant for Quantum Low-Density Parity-Check Codes.

## Run unit tests

```bash
$ uv venv --python 3.12
$ . .venv/bin/activate
$ uv pip install quits
$ export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
$ cargo test
```

## Install python package

```bash
$ maturin develop --release
```
