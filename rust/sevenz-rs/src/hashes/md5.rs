//! MD5 — port of `7zip/C/Md5.c`.
//!
//! RFC 1321 message-digest algorithm. Output: 16-byte digest.
//! ⚠ Cryptographically broken — included for compatibility with legacy
//! formats only.

const BLOCK_SIZE: usize = 64;

#[inline(always)]
fn f1(x: u32, y: u32, z: u32) -> u32 { z ^ (x & (y ^ z)) }
#[inline(always)]
fn f2(x: u32, y: u32, z: u32) -> u32 { f1(z, x, y) }
#[inline(always)]
fn f3(x: u32, y: u32, z: u32) -> u32 { x ^ y ^ z }
#[inline(always)]
fn f4(x: u32, y: u32, z: u32) -> u32 { y ^ (x | !z) }

#[derive(Clone, Debug)]
pub struct Md5 {
    state: [u32; 4],
    buffer: [u8; BLOCK_SIZE],
    count: u64,
}

impl Default for Md5 {
    fn default() -> Self {
        Self::new()
    }
}

impl Md5 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
            buffer: [0; BLOCK_SIZE],
            count: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let pos = (self.count as usize) & (BLOCK_SIZE - 1);
        let num = BLOCK_SIZE - pos;
        self.count = self.count.wrapping_add(data.len() as u64);

        if num > data.len() {
            self.buffer[pos..pos + data.len()].copy_from_slice(data);
            return;
        }
        if pos != 0 {
            self.buffer[pos..].copy_from_slice(&data[..num]);
            data = &data[num..];
            let block = self.buffer;
            self.process_block(&block);
        }
        while data.len() >= BLOCK_SIZE {
            self.process_block(data[..BLOCK_SIZE].try_into().unwrap());
            data = &data[BLOCK_SIZE..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
        }
    }

    pub fn finalize(mut self) -> [u8; 16] {
        let bit_count = self.count << 3;
        let mut pos = (self.count as usize) & (BLOCK_SIZE - 1);
        self.buffer[pos] = 0x80;
        pos += 1;
        if pos > BLOCK_SIZE - 8 {
            while pos < BLOCK_SIZE {
                self.buffer[pos] = 0;
                pos += 1;
            }
            let block = self.buffer;
            self.process_block(&block);
            pos = 0;
        }
        for i in pos..(BLOCK_SIZE - 8) {
            self.buffer[i] = 0;
        }
        self.buffer[BLOCK_SIZE - 8..].copy_from_slice(&bit_count.to_le_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut out = [0u8; 16];
        for i in 0..4 {
            out[4 * i..4 * (i + 1)].copy_from_slice(&self.state[i].to_le_bytes());
        }
        out
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 16];
        for i in 0..16 {
            w[i] = u32::from_le_bytes(block[4 * i..4 * (i + 1)].try_into().unwrap());
        }

        let [mut a, mut b, mut c, mut d] = self.state;

        // Round 1: f1, start=0, step=1, shifts (7,12,17,22)
        const K1: [u32; 16] = [
            0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
            0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
            0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
            0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
        ];
        const K2: [u32; 16] = [
            0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
            0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
            0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
            0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
        ];
        const K3: [u32; 16] = [
            0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
            0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
            0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
            0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
        ];
        const K4: [u32; 16] = [
            0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
            0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
            0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
            0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
        ];
        const S1: [u32; 4] = [7, 12, 17, 22];
        const S2: [u32; 4] = [5, 9, 14, 20];
        const S3: [u32; 4] = [4, 11, 16, 23];
        const S4: [u32; 4] = [6, 10, 15, 21];

        macro_rules! r {
            ($f:ident, $a:ident, $b:ident, $c:ident, $d:ident, $idx:expr, $s:expr, $k:expr) => {{
                let v = $a.wrapping_add(w[$idx]).wrapping_add($k).wrapping_add($f($b, $c, $d));
                $a = v.rotate_left($s).wrapping_add($b);
            }};
        }

        // Round 1
        for i in 0..4 {
            let i4 = i * 4;
            r!(f1, a, b, c, d, i4 + 0, S1[0], K1[i4 + 0]);
            r!(f1, d, a, b, c, i4 + 1, S1[1], K1[i4 + 1]);
            r!(f1, c, d, a, b, i4 + 2, S1[2], K1[i4 + 2]);
            r!(f1, b, c, d, a, i4 + 3, S1[3], K1[i4 + 3]);
        }
        // Round 2: start=1, step=5
        for i in 0..4 {
            let base = (1 + 5 * (i * 4)) & 15;
            let i4 = i * 4;
            r!(f2, a, b, c, d, (base + 0 * 5) & 15, S2[0], K2[i4 + 0]);
            r!(f2, d, a, b, c, (base + 1 * 5) & 15, S2[1], K2[i4 + 1]);
            r!(f2, c, d, a, b, (base + 2 * 5) & 15, S2[2], K2[i4 + 2]);
            r!(f2, b, c, d, a, (base + 3 * 5) & 15, S2[3], K2[i4 + 3]);
        }
        // Round 3: start=5, step=3
        for i in 0..4 {
            let i4 = i * 4;
            r!(f3, a, b, c, d, (5 + 3 * (i4 + 0)) & 15, S3[0], K3[i4 + 0]);
            r!(f3, d, a, b, c, (5 + 3 * (i4 + 1)) & 15, S3[1], K3[i4 + 1]);
            r!(f3, c, d, a, b, (5 + 3 * (i4 + 2)) & 15, S3[2], K3[i4 + 2]);
            r!(f3, b, c, d, a, (5 + 3 * (i4 + 3)) & 15, S3[3], K3[i4 + 3]);
        }
        // Round 4: start=0, step=7
        for i in 0..4 {
            let i4 = i * 4;
            r!(f4, a, b, c, d, (7 * (i4 + 0)) & 15, S4[0], K4[i4 + 0]);
            r!(f4, d, a, b, c, (7 * (i4 + 1)) & 15, S4[1], K4[i4 + 1]);
            r!(f4, c, d, a, b, (7 * (i4 + 2)) & 15, S4[2], K4[i4 + 2]);
            r!(f4, b, c, d, a, (7 * (i4 + 3)) & 15, S4[3], K4[i4 + 3]);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
    }
}

pub fn digest(data: &[u8]) -> [u8; 16] {
    let mut h = Md5::new();
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
    fn rfc1321_vectors() {
        // From RFC 1321 appendix A.5.
        assert_eq!(hex(&digest(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&digest(b"a")), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(hex(&digest(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            hex(&digest(b"message digest")),
            "f96b697d7cb7938d525a2f31aaf161d0"
        );
        assert_eq!(
            hex(&digest(b"abcdefghijklmnopqrstuvwxyz")),
            "c3fcd3d76192e4007dfb496cca67e13b"
        );
        assert_eq!(
            hex(&digest(
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            )),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );
    }

    #[test]
    fn streaming() {
        let data: Vec<u8> = (0..2000u32).map(|i| (i * 97 + 31) as u8).collect();
        let one = digest(&data);
        for split in [0, 1, 63, 64, 65, 100, 1000, 1999, 2000] {
            let mut h = Md5::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), one);
        }
    }
}
