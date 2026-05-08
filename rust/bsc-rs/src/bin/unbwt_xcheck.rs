//! Read [index_LE32 || bwt_bytes] from stdin, run inverse BWT, write
//! original bytes to stdout. Used for cross-language round-trip with
//! libsais's `bsc_bwt_encode`.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 4 {
        eprintln!("input too short ({} bytes)", buf.len());
        std::process::exit(1);
    }
    let index = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let mut t = buf[4..].to_vec();
    bsc_rs::bwt::unbwt(&mut t, index).unwrap_or_else(|e| {
        eprintln!("unbwt error: {:?}", e);
        std::process::exit(1);
    });
    std::io::stdout().write_all(&t).unwrap();
}
