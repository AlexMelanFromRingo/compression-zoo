//! `bsc-rs` — memory-safe Rust port of [libbsc] (BWT + ST + LZP + QLFC).
//!
//! Upstream is C/C++ at `plugins/bsc/upstream/libbsc/`. This crate is a
//! line-by-line port with `#![forbid(unsafe_code)]`.
//!
//! Status: skeleton + adler32 only. The block sorters, LZP, QLFC coder,
//! and libsais suffix array construction are TODO.
//!
//! [libbsc]: https://github.com/IlyaGrebnov/libbsc

#![forbid(unsafe_code)]

pub mod adler32;
pub mod bwt;
pub mod coder_tables;
pub mod format;
pub mod libbsc;
pub mod lzp;
pub mod predictor;
pub mod qlfc;
pub mod qlfc_model;
pub mod rangecoder;
pub mod sais;
pub mod st;
