//! Encode stdin to PPMd7 in the same wire format as `ppmd7_xcheck e`:
//! order(1) || mem_size(4 LE) || len(4 LE) || payload.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let order = 6u8;
    let mem = 1u32 << 20; // 1 MiB
    let payload = sevenz::ppmd7::encode_one_shot(&buf, mem, order as u32);

    let mut out = Vec::with_capacity(9 + payload.len());
    out.push(order);
    out.extend_from_slice(&mem.to_le_bytes());
    out.extend_from_slice(&(buf.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    std::io::stdout().write_all(&out).unwrap();
}
