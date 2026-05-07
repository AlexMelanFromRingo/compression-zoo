//! Memory-safe Rust port of the 7-Zip / LZMA SDK algorithms.
//!
//! Each module is a port of one C source unit from `7zip/C/`. Modules are
//! self-contained so they can be tested independently against the original
//! reference implementation.

#![forbid(unsafe_code)]

pub mod aes;
pub mod bcj2_dec;
pub mod bcj2_enc;
pub mod bra;
pub mod crc32;
pub mod crc64;
pub mod delta;
pub mod hashes;
pub mod lzma_dec;
pub mod lzma_enc;
pub mod lzma2_dec;
pub mod lzma2_enc;
pub mod ppmd7;
pub mod ppmd8;
