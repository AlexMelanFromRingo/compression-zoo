//! Cross-check harness for the bsc-rs range coder.
//!
//! Modes:
//!   `e`     — read stdin bytes; encode with `RangeEncoder`; write the
//!             wire bytes to stdout.
//!   `d N`   — read N bytes of decoded output from a libbsc-compatible
//!             range-coded wire on stdin; write decoded bytes to stdout.

use std::io::{Read, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("mode e|d");
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();

    match mode.as_str() {
        "e" => {
            let mut out = Vec::with_capacity(buf.len() * 2 + 4096);
            let cap = buf.len() * 2 + 4096;
            let mut enc = bsc_rs::rangecoder::RangeEncoder::new(&mut out, cap);
            for &b in &buf {
                enc.encode_byte(b as u32);
            }
            let _ = enc.finish();
            std::io::stdout().write_all(&out).unwrap();
        }
        "d" => {
            let n: usize = args
                .next()
                .expect("decoded byte count")
                .parse()
                .expect("integer");
            let mut dec = bsc_rs::rangecoder::RangeDecoder::new(&buf);
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(dec.decode_byte() as u8);
            }
            std::io::stdout().write_all(&out).unwrap();
        }
        _ => {
            eprintln!("usage: rc_xcheck e|d [N]");
            std::process::exit(2);
        }
    }
}
