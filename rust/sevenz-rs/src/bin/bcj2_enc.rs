//! BCJ2 encoder cross-check binary.
//! Output layout matches `bcj2_xcheck.c e`: 4 LE u32 sizes, 4 streams, 4-byte
//! little-endian original size.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let out = sevenz::bcj2_enc::encode_one_shot(&buf, 0);
    let mut o = Vec::new();
    o.extend_from_slice(&(out.main.len() as u32).to_le_bytes());
    o.extend_from_slice(&(out.call.len() as u32).to_le_bytes());
    o.extend_from_slice(&(out.jump.len() as u32).to_le_bytes());
    o.extend_from_slice(&(out.rc.len() as u32).to_le_bytes());
    o.extend_from_slice(&out.main);
    o.extend_from_slice(&out.call);
    o.extend_from_slice(&out.jump);
    o.extend_from_slice(&out.rc);
    o.extend_from_slice(&(buf.len() as u32).to_le_bytes());
    std::io::stdout().write_all(&o).unwrap();
}
