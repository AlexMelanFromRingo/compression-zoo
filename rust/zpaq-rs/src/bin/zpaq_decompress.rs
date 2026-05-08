//! End-to-end ZPAQ decompressor (subset). Reads a ZPAQ archive
//! from stdin, writes decompressed bytes to stdout.
//!
//! Currently only handles method "0" archives. For modeled
//! methods ("1".."5", "x...") the program exits with an error.

use std::io::{Read, Write};
use zpaq_rs::decompress;
use zpaq_rs::io::{SliceReader, VecWriter};

fn main() {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    let mut r = SliceReader::new(&input);
    let mut w = VecWriter::new();

    let result = decompress::decompress(&mut r, &mut w);
    // Always flush whatever bytes we accumulated, even on error,
    // so callers can inspect partial output (useful for debugging
    // a divergent predictor / VM).
    let _ = std::io::stdout().write_all(&w.buf);
    match result {
        Ok(segs) => {
            for s in &segs {
                eprintln!("segment: filename=\"{}\" bytes={} sha1_verified={}",
                          String::from_utf8_lossy(&s.filename),
                          s.bytes_written,
                          s.sha1_verified);
            }
        }
        Err(e) => {
            eprintln!("decompress error: {:?}", e);
            std::process::exit(1);
        }
    }
}
