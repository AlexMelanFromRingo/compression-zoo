//! Decode an LZMA-Alone stream (5-byte properties + 8-byte size + payload).
//! Mirrors `7lzma d` from the SDK.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    match sevenz::lzma_dec::decode_lzma_alone(&buf) {
        Ok(out) => {
            std::io::stdout().write_all(&out).unwrap();
        }
        Err(e) => {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    }
}
