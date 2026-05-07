# Rust components

`sevenz-rs/` is a memory-safe Rust port of the algorithms in the 7-Zip /
LZMA SDK C codebase. Built with `#![forbid(unsafe_code)]`.

See `sevenz-rs/Cargo.toml` and the per-module sources for what is
currently implemented. The intention is for this crate to eventually
back the codec plugins in `../plugins/`, replacing the C++ wrappers.
