//! End-to-end libbsc decompressor written against the Rust port.
//!
//! Reads a libbsc-format byte stream (concatenation of one or more
//! independently-encoded blocks, the same framing
//! `plugins/bsc/tests/bsc_cli e` produces) and writes the original
//! bytes to stdout.

use std::io::{Read, Write};

fn main() {
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();

    let mut pos = 0;
    let mut out = Vec::new();
    while pos < input.len() {
        let mut block_out: Vec<u8> = Vec::new();
        match bsc_rs::libbsc::decompress(&input[pos..], &mut block_out) {
            Ok(_) => {
                let info = bsc_rs::format::block_info(&input[pos..]).unwrap();
                pos += info.block_size as usize;
                out.extend_from_slice(&block_out);
            }
            Err(e) => {
                eprintln!("decompress at pos={pos}: {:?}", e);
                std::process::exit(1);
            }
        }
    }
    std::io::stdout().write_all(&out).unwrap();
}
