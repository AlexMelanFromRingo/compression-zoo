//! `cmix-rs` — memory-safe Rust port of [CMIX] (Byron Knoll).
//!
//! Upstream is `plugins/cmix/upstream/src/` — GPL-3.0+. Note the
//! upstream is huge (~21 kLOC across PAQ8, FXCM, PPMD, LSTM mixer,
//! context manager, etc.) and full of file-scope mutable globals; a
//! straight port has to refactor those into properly-scoped state.
//!
//! Status: foundation in place — arith coder, sigmoid table, two
//! state machines (Nonstationary / RunMap). The mixer, predictor,
//! context manager, and per-component models are out of scope for
//! this crate's first cut and tracked in `HANDOFF.md`.
//!
//! [CMIX]: https://github.com/byronknoll/cmix

#![forbid(unsafe_code)]

pub mod coder;     // arithmetic coder (encoder.cpp / decoder.cpp)
pub mod mixer;     // mixer + mixer-input
pub mod sigmoid;   // mixer/sigmoid table
pub mod state;     // State trait
pub mod states;    // nonstationary, run-map

// TODO (multi-week scope): predictor, context_manager, contexts/*,
// mixer/{byte-mixer,lstm,sse}, models/*. See HANDOFF.
