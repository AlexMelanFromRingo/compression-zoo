//! Encode stdin to PPMd8 in same format as `ppmd8_xcheck e`:
//! order(1) || mem_size(4 LE) || method(1) || len(4 LE) || payload.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let order = 6u8;
    let mem = 1u32 << 20;
    let method = sevenz::ppmd8::RestoreMethod::Restart;
    let payload = sevenz::ppmd8::encode_one_shot(&buf, mem, order as u32, method);

    let mut out = Vec::with_capacity(10 + payload.len());
    out.push(order);
    out.extend_from_slice(&mem.to_le_bytes());
    out.push(0); // method
    out.extend_from_slice(&(buf.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    std::io::stdout().write_all(&out).unwrap();
}
