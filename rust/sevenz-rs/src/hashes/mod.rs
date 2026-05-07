//! Cryptographic and non-cryptographic hash functions ported from the
//! `7zip/C/` SDK. Each sub-module is a faithful port; standard test vectors
//! are included to confirm bit-exact output.

pub mod blake2s;
pub mod md5;
pub mod sha1;
pub mod sha256;
pub mod sha512;
pub mod xxh64;
