//! Preprocessing helpers — currently just E8E9. Mirrors
//! `e8e9()` in `plugins/zpaq/upstream/libzpaq.cpp:6450`.
//!
//! Future additions (per `docs/gpu-acceleration.md`):
//!   * LZ77 search front-end (the `LZBuffer` class).
//!   * `makeConfig` — expand `"x4,3,..."` method strings into full
//!     ZPAQL config strings consumable by [`crate::compiler::compile`].

#![allow(dead_code)]

/// E8E9 forward transform: rewrite Intel/AMD `CALL`/`JMP rel32`
/// (opcodes `0xE8` / `0xE9`) so the 32-bit relative offset becomes
/// absolute. Improves compressibility on x86/x64 machine code.
///
/// The function is structurally idempotent on byte streams that
/// don't contain matching opcode patterns (so calling it on text
/// data is a no-op for almost every byte).
///
/// Mirrors `void e8e9(unsigned char* buf, int n)` upstream. Operates
/// in place.
pub fn e8e9_forward(buf: &mut [u8]) {
    let n = buf.len();
    if n < 5 { return; }
    // Iterate from the end backwards so that earlier rewrites can't
    // walk over later candidate positions.
    for i in (0..=n - 5).rev() {
        let opcode_match = (buf[i] & 0xFE) == 0xE8;
        // The +1 wrap mirrors upstream: `((buf[i+4]+1)&254)==0`.
        let tail = buf[i + 4].wrapping_add(1);
        let tail_match = (tail & 0xFE) == 0;
        if opcode_match && tail_match {
            let a = (buf[i + 1] as u32)
                  | ((buf[i + 2] as u32) << 8)
                  | ((buf[i + 3] as u32) << 16);
            let abs = a.wrapping_add(i as u32);
            buf[i + 1] = (abs & 0xFF) as u8;
            buf[i + 2] = ((abs >> 8) & 0xFF) as u8;
            buf[i + 3] = ((abs >> 16) & 0xFF) as u8;
        }
    }
}

/// E8E9 inverse transform: undo [`e8e9_forward`]. Iterates forwards
/// (mirror of the encoder's reverse iteration) and subtracts the
/// position to recover the relative offset.
pub fn e8e9_inverse(buf: &mut [u8]) {
    let n = buf.len();
    if n < 5 { return; }
    for i in 0..=n - 5 {
        let opcode_match = (buf[i] & 0xFE) == 0xE8;
        let tail = buf[i + 4].wrapping_add(1);
        let tail_match = (tail & 0xFE) == 0;
        if opcode_match && tail_match {
            let a = (buf[i + 1] as u32)
                  | ((buf[i + 2] as u32) << 8)
                  | ((buf[i + 3] as u32) << 16);
            let rel = a.wrapping_sub(i as u32);
            buf[i + 1] = (rel & 0xFF) as u8;
            buf[i + 2] = ((rel >> 8) & 0xFF) as u8;
            buf[i + 3] = ((rel >> 16) & 0xFF) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_random() {
        let mut x: u32 = 0xC0FFEE;
        let mut buf = vec![0u8; 4096];
        for b in buf.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        let orig = buf.clone();
        e8e9_forward(&mut buf);
        e8e9_inverse(&mut buf);
        assert_eq!(buf, orig);
    }

    #[test]
    fn no_op_on_short_buffer() {
        let mut tiny = vec![0xE8, 0x00, 0x00, 0x00]; // only 4 bytes
        let orig = tiny.clone();
        e8e9_forward(&mut tiny);
        assert_eq!(tiny, orig);
    }

    #[test]
    fn rewrites_matching_pattern() {
        // 0xE8 (CALL), rel32=0x00000010, terminator 0x00 → abs = 0x10 + i
        let mut buf = vec![0xE8, 0x10, 0x00, 0x00, 0x00];
        e8e9_forward(&mut buf);
        // i=0, so abs=0x10+0=0x10. No change in this case.
        assert_eq!(buf, vec![0xE8, 0x10, 0x00, 0x00, 0x00]);

        // Place the pattern at offset 5: the rewrite should shift.
        let mut buf2 = vec![0u8; 5];
        buf2.extend_from_slice(&[0xE9, 0x10, 0x00, 0x00, 0x00]);
        e8e9_forward(&mut buf2);
        assert_eq!(&buf2[5..], &[0xE9, 0x15, 0x00, 0x00, 0x00]);
        e8e9_inverse(&mut buf2);
        assert_eq!(&buf2[5..], &[0xE9, 0x10, 0x00, 0x00, 0x00]);
    }
}
