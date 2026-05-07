//! Encode stdin to LZMA2 in the same wire format as `lzma2_xcheck.c`:
//! prop byte (1) || size_le_u64 (8) || payload.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let cfg = sevenz::lzma_enc::EncoderConfig::default();
    let (prop, framed) = match sevenz::lzma2_enc::encode_one_shot(&buf, cfg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("encode error: {e}");
            std::process::exit(1);
        }
    };
    let mut out = Vec::with_capacity(9 + framed.len());
    out.push(prop);
    out.extend_from_slice(&(buf.len() as u64).to_le_bytes());
    out.extend_from_slice(&framed);
    std::io::stdout().write_all(&out).unwrap();
}
