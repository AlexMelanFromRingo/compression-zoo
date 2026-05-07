//! Decode a PPMd8 stream packaged as: order(1) || mem_size(4 LE) || method(1) || len(4 LE) || payload.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 10 {
        eprintln!("too short");
        std::process::exit(1);
    }
    let order = buf[0] as u32;
    let mem_size = u32::from_le_bytes(buf[1..5].try_into().unwrap());
    let method = match buf[5] {
        0 => sevenz::ppmd8::RestoreMethod::Restart,
        _ => sevenz::ppmd8::RestoreMethod::CutOff,
    };
    let unpack = u32::from_le_bytes(buf[6..10].try_into().unwrap());
    let payload = &buf[10..];

    let mut dec = match sevenz::ppmd8::Ppmd8Decoder::new(mem_size, order, method, payload) {
        Ok(d) => d,
        Err(e) => { eprintln!("init error: {e}"); std::process::exit(1); }
    };
    let mut out = Vec::with_capacity(unpack as usize);
    for i in 0..unpack {
        match dec.decode_symbol() {
            Ok(b) => out.push(b),
            Err(c) => { eprintln!("decode err at {i}: code {c}"); std::process::exit(1); }
        }
    }
    std::io::stdout().write_all(&out).unwrap();
}
