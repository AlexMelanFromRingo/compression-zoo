//! QLFC static-decoder cross-check harness.
//!
//! Reads `[expected_decoded_size_LE32 || qlfc_wire_bytes]` from stdin
//! and writes the decoded bytes to stdout. The C harness emits just
//! `qlfc_wire_bytes`; the Bash test glues the size in front so this
//! binary doesn't have to read it from argv.
//!
//! Mode "d N" matches the C harness directly: read N decoded bytes
//! out of the wire stream on stdin.

use std::io::{Read, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("mode e|d");
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    match mode.as_str() {
        "e" => {
            let cap = buf.len() * 2 + 4096;
            let mut out = Vec::with_capacity(cap);
            match bsc_rs::qlfc::static_encode(&buf, &mut out, cap) {
                Ok(_) => {
                    std::io::stdout().write_all(&out).unwrap();
                }
                Err(e) => {
                    eprintln!("qlfc encode error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        "d" => {
            let n: usize = args.next().expect("decoded byte count").parse().expect("N");
            let mut out = vec![0u8; n];
            match bsc_rs::qlfc::static_decode(&buf, &mut out) {
                Ok(written) => {
                    std::io::stdout().write_all(&out[..written]).unwrap();
                }
                Err(e) => {
                    eprintln!("qlfc decode error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("usage: qlfc_xcheck e|d [N]");
            std::process::exit(2);
        }
    }
}
