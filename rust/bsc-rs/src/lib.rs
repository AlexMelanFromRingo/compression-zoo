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
pub mod format;
pub mod lzp;
// pub mod bwt;        // TODO
// pub mod st;         // TODO
// pub mod lzp;        // TODO
// pub mod qlfc;       // TODO
// pub mod libsais;    // TODO
// pub mod libbsc;     // top-level public API (compress/decompress)
