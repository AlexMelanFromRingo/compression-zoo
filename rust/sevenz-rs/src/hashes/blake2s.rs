//! BLAKE2s — RFC 7693 scalar implementation. Output: 32-byte digest.
//!
//! 7-Zip ships a parallel variant (BLAKE2sp) with heavy SSE/AVX
//! vectorisation; this scalar port covers the underlying compression
//! function and the standard single-tree BLAKE2s mode used by most code.

const BLOCK_SIZE: usize = 64;
const OUT_SIZE: usize = 32;

const IV: [u32; 8] = [
    0x6A09_E667, 0xBB67_AE85, 0x3C6E_F372, 0xA54F_F53A,
    0x510E_527F, 0x9B05_688C, 0x1F83_D9AB, 0x5BE0_CD19,
];

const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

#[derive(Clone, Debug)]
pub struct Blake2s {
    h: [u32; 8],
    t: u64,
    buf: [u8; BLOCK_SIZE],
    buf_len: usize,
}

impl Default for Blake2s {
    fn default() -> Self {
        Self::new()
    }
}

impl Blake2s {
    pub fn new() -> Self {
        let mut h = IV;
        // Parameter block: digest_length=32, key_length=0, fanout=1, depth=1.
        h[0] ^= 0x0101_0000 | (OUT_SIZE as u32);
        Self {
            h,
            t: 0,
            buf: [0; BLOCK_SIZE],
            buf_len: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if data.is_empty() {
            return;
        }
        if self.buf_len > 0 {
            let take = (BLOCK_SIZE - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == BLOCK_SIZE && !data.is_empty() {
                let buf = self.buf;
                self.t = self.t.wrapping_add(BLOCK_SIZE as u64);
                self.compress(&buf, false);
                self.buf_len = 0;
            }
        }
        while data.len() > BLOCK_SIZE {
            let chunk: [u8; 64] = data[..BLOCK_SIZE].try_into().unwrap();
            self.t = self.t.wrapping_add(BLOCK_SIZE as u64);
            self.compress(&chunk, false);
            data = &data[BLOCK_SIZE..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        // Pad final block with zeros.
        for i in self.buf_len..BLOCK_SIZE {
            self.buf[i] = 0;
        }
        self.t = self.t.wrapping_add(self.buf_len as u64);
        let buf = self.buf;
        self.compress(&buf, true);
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * (i + 1)].copy_from_slice(&self.h[i].to_le_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8; 64], last: bool) {
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(block[4 * i..4 * (i + 1)].try_into().unwrap());
        }
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= self.t as u32;
        v[13] ^= (self.t >> 32) as u32;
        if last {
            v[14] = !v[14];
        }

        macro_rules! g {
            ($a:expr, $b:expr, $c:expr, $d:expr, $x:expr, $y:expr) => {{
                v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add($x);
                v[$d] = (v[$d] ^ v[$a]).rotate_right(16);
                v[$c] = v[$c].wrapping_add(v[$d]);
                v[$b] = (v[$b] ^ v[$c]).rotate_right(12);
                v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add($y);
                v[$d] = (v[$d] ^ v[$a]).rotate_right(8);
                v[$c] = v[$c].wrapping_add(v[$d]);
                v[$b] = (v[$b] ^ v[$c]).rotate_right(7);
            }};
        }

        for i in 0..10 {
            let s = &SIGMA[i];
            g!(0, 4, 8, 12, m[s[0]], m[s[1]]);
            g!(1, 5, 9, 13, m[s[2]], m[s[3]]);
            g!(2, 6, 10, 14, m[s[4]], m[s[5]]);
            g!(3, 7, 11, 15, m[s[6]], m[s[7]]);
            g!(0, 5, 10, 15, m[s[8]], m[s[9]]);
            g!(1, 6, 11, 12, m[s[10]], m[s[11]]);
            g!(2, 7, 8, 13, m[s[12]], m[s[13]]);
            g!(3, 4, 9, 14, m[s[14]], m[s[15]]);
        }

        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }
}

pub fn digest(data: &[u8]) -> [u8; 32] {
    let mut h = Blake2s::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(d: &[u8]) -> String {
        d.iter().map(|b| format!("{:02x}", b)).collect()
    }

    #[test]
    fn rfc_7693_vectors() {
        // RFC 7693 BLAKE2s-256 test vector.
        assert_eq!(
            hex(&digest(b"abc")),
            "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"
        );
        assert_eq!(
            hex(&digest(b"")),
            "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"
        );
    }
}
