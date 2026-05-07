//! SHA-1 — port of `7zip/C/Sha1.c`.
//!
//! FIPS 180-4. Output: 20-byte digest.

const BLOCK_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct Sha1 {
    state: [u32; 5],
    buffer: [u8; BLOCK_SIZE],
    count: u64,
}

impl Default for Sha1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha1 {
    pub fn new() -> Self {
        Self {
            state: [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0],
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

    pub fn finalize(mut self) -> [u8; 20] {
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
        self.buffer[BLOCK_SIZE - 8..].copy_from_slice(&bit_count.to_be_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut out = [0u8; 20];
        for i in 0..5 {
            out[4 * i..4 * (i + 1)].copy_from_slice(&self.state[i].to_be_bytes());
        }
        out
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * (i + 1)].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let [mut a, mut b, mut c, mut d, mut e] = self.state;

        for i in 0..20 {
            let f = (b & c) | (!b & d);
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(0x5A82_7999)
                .wrapping_add(w[i]);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }
        for i in 20..40 {
            let f = b ^ c ^ d;
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(0x6ED9_EBA1)
                .wrapping_add(w[i]);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }
        for i in 40..60 {
            let f = (b & c) | (b & d) | (c & d);
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(0x8F1B_BCDC)
                .wrapping_add(w[i]);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }
        for i in 60..80 {
            let f = b ^ c ^ d;
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(0xCA62_C1D6)
                .wrapping_add(w[i]);
            e = d; d = c; c = b.rotate_left(30); b = a; a = temp;
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

pub fn digest(data: &[u8]) -> [u8; 20] {
    let mut h = Sha1::new();
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
    fn fips_180_vectors() {
        assert_eq!(hex(&digest(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(hex(&digest(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            hex(&digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
        // 1M 'a' → 34aa973cd4c4daa4f61eeb2bdbad27316534016f
        let one_mb: Vec<u8> = std::iter::repeat(b'a').take(1_000_000).collect();
        assert_eq!(
            hex(&digest(&one_mb)),
            "34aa973cd4c4daa4f61eeb2bdbad27316534016f"
        );
    }

    #[test]
    fn streaming() {
        let data: Vec<u8> = (0..3000u32).map(|i| (i * 53 + 7) as u8).collect();
        let one = digest(&data);
        for split in [0, 1, 63, 64, 65, 1000, 2999, 3000] {
            let mut h = Sha1::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), one);
        }
    }
}
