//! `zpaq-rs` — memory-safe Rust port of [libzpaq] (Matt Mahoney).
//!
//! Upstream is `plugins/zpaq/upstream/libzpaq.{h,cpp}` (public domain).
//!
//! Status: foundations only.
//!
//! What's done:
//!   * `io` — `Reader` / `Writer` traits + `SliceReader`/`VecWriter`
//!     adapters mirroring libzpaq's abstract base classes.
//!   * `arith` — 32-bit binary arithmetic coder (Encoder + Decoder),
//!     bit-for-bit round-trip with self.
//!
//! TODO:
//!   * Predictor (CM/ICM/MATCH/AVG/MIX2/MIX/ISSE/SSE components).
//!   * ZPAQL VM — interpreter for the small bytecode that drives the
//!     predictor and post-processor.
//!   * Block-format reader / writer (magic, header, segments, SHA-1).
//!   * Top-level `compress` / `decompress` against `libzpaq` test vectors.
//!
//! [libzpaq]: http://mattmahoney.net/dc/zpaq.html

#![forbid(unsafe_code)]

pub mod arith;
pub mod decompress;
pub mod format;
pub mod io;
pub mod predictor;
pub mod predictor_tables;
pub mod sha1;
pub mod state_table;
pub mod zpaql;
