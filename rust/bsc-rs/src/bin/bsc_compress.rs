//! End-to-end libbsc compressor written against the Rust port.
//!
//! Reads stdin, encodes a single libbsc block at the given level
//! (BWT + QLFC_STATIC + optional LZP), writes to stdout.
//!
//! Usage:  bsc_compress [level=5]
//!
//! Levels follow `plugins/bsc/tests/bsc_cli`:
//!     1, 3 → QLFC_FAST    (NOT supported by the Rust encoder yet)
//!     5    → QLFC_STATIC, lzp_hash=15, lzp_min_len=72
//!     7    → QLFC_STATIC, lzp_hash=16, lzp_min_len=96
//!     9    → QLFC_ADAPTIVE  (NOT supported by the Rust encoder yet)
//!
//! Anything outside of {5, 7} falls back to the level-5 settings.

use std::io::{Read, Write};

fn main() {
    let level: i32 = std::env::args().nth(1)
        .map(|s| s.parse().unwrap_or(5)).unwrap_or(5);

    use bsc_rs::format::{
        LIBBSC_CODER_QLFC_ADAPTIVE, LIBBSC_CODER_QLFC_FAST, LIBBSC_CODER_QLFC_STATIC,
    };
    let (lzp_hash, lzp_min_len, coder) = match level {
        9 => (16, 128, LIBBSC_CODER_QLFC_ADAPTIVE),
        7 => (16,  96, LIBBSC_CODER_QLFC_STATIC),
        5 => (15,  72, LIBBSC_CODER_QLFC_STATIC),
        3 => (14,  64, LIBBSC_CODER_QLFC_FAST),
        _ => ( 0,   0, LIBBSC_CODER_QLFC_FAST), // level 1
    };

    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();

    let mut out = Vec::new();
    match bsc_rs::libbsc::compress_with_coder(&input, &mut out, lzp_hash, lzp_min_len, coder) {
        Ok(_) => {
            std::io::stdout().write_all(&out).unwrap();
        }
        Err(e) => {
            eprintln!("compress error: {:?}", e);
            std::process::exit(1);
        }
    }
}
