//! ZPAQ stored/modeled compresser CLI. Reads stdin, writes a ZPAQ
//! archive to stdout. Method:
//!   * `0` — stored (no model, libzpaq-compatible level 0).
//!   * `1` / `2` / `3` — modeled, using the canned `min.cfg` /
//!     `mid.cfg` / `max.cfg` headers from upstream.

use std::io::{Read, Write};
use zpaq_rs::compress::{Compresser, ZPAQ_TAG};
use zpaq_rs::io::Writer;
use zpaq_rs::models::{MIN_CFG, MID_CFG, MAX_CFG};

struct StdoutWriter;
impl Writer for StdoutWriter {
    fn put(&mut self, c: u8) {
        let _ = std::io::stdout().write_all(&[c]);
    }
    fn write(&mut self, buf: &[u8]) {
        let _ = std::io::stdout().write_all(buf);
    }
}

fn main() {
    let _ = ZPAQ_TAG; // ensure import is exercised
    let method: u8 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).expect("stdin");

    let mut c = Compresser::new(StdoutWriter);
    c.write_tag().expect("tag");
    if method == 0 {
        c.start_block_stored().expect("start");
    } else {
        let cfg = match method {
            1 => MIN_CFG,
            2 => MID_CFG,
            3 => MAX_CFG,
            _ => {
                eprintln!("zpaq_compress: method must be 0..3");
                std::process::exit(2);
            }
        };
        c.start_block_modeled(cfg).expect("start");
    }
    c.start_segment(b"", b"").expect("seg");
    c.post_process_pass().expect("pp");
    c.write_bytes(&input).expect("data");
    c.end_segment(None).expect("eseg");
    c.end_block().expect("eblk");
    let _ = std::io::stdout().flush();
}
