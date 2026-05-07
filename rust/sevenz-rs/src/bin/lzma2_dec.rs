//! Decode an LZMA2 stream framed as: prop-byte (1) || size_le_u64 (8) || payload.
//! Mirrors the layout produced by the `lzma2_xcheck e` harness.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 9 {
        eprintln!("input too short");
        std::process::exit(1);
    }
    let prop = buf[0];
    let unpacked = u64::from_le_bytes(buf[1..9].try_into().unwrap()) as usize;
    let payload = &buf[9..];
    match sevenz::lzma2_dec::decode_one_shot(prop, payload) {
        Ok(out) => {
            if out.len() != unpacked {
                eprintln!(
                    "size mismatch: expected {} got {}",
                    unpacked,
                    out.len()
                );
                std::process::exit(1);
            }
            std::io::stdout().write_all(&out).unwrap();
        }
        Err(e) => {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    }
}
