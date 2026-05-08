//! ST inverse cross-check harness: read [LE32 idx || ST_bytes] from
//! stdin, run unst with the given k, write decoded bytes to stdout.

use std::io::{Read, Write};

fn main() {
    let k: i32 = std::env::args().nth(1).expect("k").parse().unwrap();
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 4 {
        eprintln!("input too short ({} bytes)", buf.len());
        std::process::exit(1);
    }
    let idx = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let mut t = buf[4..].to_vec();
    bsc_rs::st::unst(&mut t, idx, k).unwrap_or_else(|e| {
        eprintln!("unst error: {:?}", e);
        std::process::exit(1);
    });
    std::io::stdout().write_all(&t).unwrap();
}
