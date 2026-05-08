//! Standard SHA-1 (FIPS 180-4) — used by libzpaq for per-segment
//! integrity. We don't pull in a `sha1` crate dependency; the
//! algorithm is small and a hand-rolled scalar implementation
//! sufficies for our test/cross-check needs.
//!
//! Output is a 20-byte big-endian digest.

#![allow(dead_code)]

const H_INIT: [u32; 5] = [
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0,
];

pub struct Sha1 {
    h: [u32; 5],
    buf: [u8; 64],
    buflen: usize,
    /// Total bytes hashed so far (for the length pad).
    total_bytes: u64,
}

impl Sha1 {
    pub fn new() -> Self {
        Self {
            h: H_INIT,
            buf: [0u8; 64],
            buflen: 0,
            total_bytes: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total_bytes += data.len() as u64;
        if self.buflen > 0 {
            let take = (64 - self.buflen).min(data.len());
            self.buf[self.buflen..self.buflen + take]
                .copy_from_slice(&data[..take]);
            self.buflen += take;
            data = &data[take..];
            if self.buflen == 64 {
                let block = self.buf;
                Self::compress(&mut self.h, &block);
                self.buflen = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            Self::compress(&mut self.h, &block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buflen = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 20] {
        let total_bits = self.total_bytes * 8;

        // 0x80 followed by zeros, then 8-byte length BE.
        let mut pad = [0u8; 64 + 64];
        pad[0] = 0x80;
        // We need the message ending at a 64-byte boundary with last 8 bytes = length.
        let pad_len = if self.buflen < 56 { 56 - self.buflen } else { 56 + 64 - self.buflen };
        let mut tail = Vec::with_capacity(pad_len + 8);
        tail.extend_from_slice(&pad[..pad_len]);
        tail.extend_from_slice(&total_bits.to_be_bytes());
        self.update(&tail);

        let mut out = [0u8; 20];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(h: &mut [u32; 5], block: &[u8; 64]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4 + 0], block[i * 4 + 1],
                block[i * 4 + 2], block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let t = a.rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
}

impl Default for Sha1 {
    fn default() -> Self { Self::new() }
}

/// Convenience: hash a byte slice in one call.
pub fn sha1_of(data: &[u8]) -> [u8; 20] {
    let mut h = Sha1::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes { s.push_str(&format!("{:02x}", b)); }
        s
    }

    #[test]
    fn empty() {
        // SHA-1("") = da39a3ee5e6b4b0d3255bfef95601890afd80709
        assert_eq!(hex(&sha1_of(b"")),
                   "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn abc() {
        assert_eq!(hex(&sha1_of(b"abc")),
                   "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn long_message() {
        // FIPS 180-2 test vector (56 bytes).
        let m = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(hex(&sha1_of(m)),
                   "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    }

    #[test]
    fn million_a() {
        // 1,000,000 bytes of 'a'.
        let m = vec![b'a'; 1_000_000];
        assert_eq!(hex(&sha1_of(&m)),
                   "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }

    #[test]
    fn streaming_matches_oneshot() {
        let m: Vec<u8> = (0u8..200).cycle().take(10000).collect();
        let mut h = Sha1::new();
        for chunk in m.chunks(73) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha1_of(&m));
    }
}
