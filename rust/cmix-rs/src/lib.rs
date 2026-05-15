//! `cmix-rs` — memory-safe Rust port of [CMIX] (Byron Knoll).
//!
//! Upstream lives at `plugins/cmix/upstream/src/` (GPL-3.0+). The
//! port refactors upstream's file-scope mutable globals into
//! properly-scoped state in safe Rust (no `unsafe`).
//!
//! [CMIX]: https://github.com/byronknoll/cmix

#![forbid(unsafe_code)]

pub mod coder;            // arithmetic coder (encoder.cpp / decoder.cpp)
pub mod context_manager;  // shared per-byte state (history, words, …)
pub mod contexts;         // bit/bracket/combined/context-hash/...
pub mod mixer;            // mixer + mixer-input + LSTM stack + SSE
pub mod models;           // direct/direct-hash/indirect/match/byte-model/
                          // bracket/ppmd/paq8/fxcmv1
pub mod orchestrator;     // full CmixPredictor — 3-layer tree + SSE + LSTM
pub mod predictor;        // top-level Predictor orchestrator
pub mod preprocess;       // dictionary-based word substitution +
                          // file-type-aware transformations
pub mod runner;           // encode / decode entry points
pub mod sigmoid;          // mixer/sigmoid table
pub mod state;            // State trait
pub mod states;           // nonstationary, run-map
