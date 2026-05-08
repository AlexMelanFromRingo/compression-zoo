//! `cmix-rs` — memory-safe Rust port of [CMIX] (Byron Knoll).
//!
//! Upstream is `plugins/cmix/upstream/src/` — GPL-3.0+. Note the
//! upstream is huge (~21 kLOC across PAQ8, FXCM, PPMD, LSTM mixer,
//! context manager, etc.) and full of file-scope mutable globals; a
//! straight port has to refactor those into properly-scoped state.
//!
//! Status: skeleton only.
//!
//! [CMIX]: https://github.com/byronknoll/cmix

#![forbid(unsafe_code)]

// pub mod arith;       // arithmetic coder (encoder.cpp / decoder.cpp)
// pub mod predictor;   // top-level mixer
// pub mod context_manager;
// pub mod contexts;    // bit/bracket/combined/context-hash/...
// pub mod mixer;       // byte-mixer, lstm, mixer, sigmoid, sse
// pub mod models;      // bracket, byte-model, direct, indirect, fxcmv1,
//                      // match, paq8, ppmd
// pub mod states;      // nonstationary, run-map
