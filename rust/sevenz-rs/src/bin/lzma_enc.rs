//! LZMA encoder cross-check binary (encodes stdin to LZMA-Alone format).

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let cfg = sevenz::lzma_enc::EncoderConfig::default();
    match sevenz::lzma_enc::encode_lzma_alone(&buf, cfg) {
        Ok(out) => std::io::stdout().write_all(&out).unwrap(),
        Err(e) => {
            eprintln!("encode error: {e}");
            std::process::exit(1);
        }
    }
}
