//! `zpaq-rs` — memory-safe Rust port of [libzpaq] (Matt Mahoney).
//!
//! Upstream is `plugins/zpaq/upstream/libzpaq.{h,cpp}` — public domain.
//!
//! Status: skeleton only. Future work: arithmetic coder, ZPAQL VM,
//! Predictor (model mixing), block-format reader/writer.
//!
//! [libzpaq]: http://mattmahoney.net/dc/zpaq.html

#![forbid(unsafe_code)]

// pub mod arith;       // TODO: arithmetic coder
// pub mod zpaql;       // TODO: ZPAQL virtual machine
// pub mod predictor;   // TODO: model mixer
// pub mod format;      // TODO: block-format reader/writer
// pub mod compress;    // TODO: top-level compress()
// pub mod decompress;  // TODO: top-level decompress()
