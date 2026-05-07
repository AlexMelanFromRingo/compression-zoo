//! Decode a PPMd7 stream packaged as: order(1) || mem_size(4 LE) || len(4 LE) || payload.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 9 {
        eprintln!("too short");
        std::process::exit(1);
    }
    let order = buf[0] as u32;
    let mem_size = u32::from_le_bytes(buf[1..5].try_into().unwrap());
    let unpack = u32::from_le_bytes(buf[5..9].try_into().unwrap());
    let payload = &buf[9..];

    let mut dec = match sevenz::ppmd7::Ppmd7Decoder::new(mem_size, order, payload) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("init error: {e}");
            std::process::exit(1);
        }
    };
    let mut out = Vec::with_capacity(unpack as usize);
    for i in 0..unpack {
        match dec.decode_symbol() {
            Ok(b) => out.push(b),
            Err(c) => {
                eprintln!("decode err at {i}: code {c}");
                std::process::exit(1);
            }
        }
        if std::env::var("PPMD_TRACE").is_ok() {
            let st = dec.debug_state_full();
            let mut line = format!("[{}] sym={:02x} mc={}(ns={} sf={} suf={}) ff={}@{} of={} root_sf={}",
                i, out[i as usize], st.0, st.1, st.2, st.3, st.4, st.5, st.6, st.7);
            if st.1 > 1 {
                let mcs = dec.debug_mc_states(st.1.min(4) as usize);
                line.push_str(" mcs:");
                for (s, f, _) in mcs {
                    line.push_str(&format!(" {:02x}:{}", s, f));
                }
            }
            eprintln!("{}", line);
        }
    }
    std::io::stdout().write_all(&out).unwrap();
}
