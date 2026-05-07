//! BCJ2 decoder cross-check binary.
//!
//! Input layout (matches `bcj2_xcheck.c`):
//! 4 little-endian u32 sub-stream sizes (main, call, jump, rc),
//! then concatenated streams,
//! then 4-byte little-endian original size for convenience.

use std::io::{Read, Write};

fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    if buf.len() < 16 {
        eprintln!("input too short");
        std::process::exit(1);
    }
    let s0 = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let s1 = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let s2 = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
    let s3 = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;

    let mut off = 16;
    let main = &buf[off..off + s0]; off += s0;
    let call = &buf[off..off + s1]; off += s1;
    let jump = &buf[off..off + s2]; off += s2;
    let rc = &buf[off..off + s3]; off += s3;

    let _orig_size = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    match sevenz::bcj2_dec::decode_one_shot(main, call, jump, rc) {
        Ok(out) => std::io::stdout().write_all(&out).unwrap(),
        Err(e) => {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    }
}
