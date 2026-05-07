//! XXH64 — port of `7zip/C/Xxh64.c`. 64-bit non-cryptographic hash.

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline(always)]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

#[inline(always)]
fn merge(mut acc: u64, val: u64) -> u64 {
    acc ^= round(0, val);
    acc.wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
}

#[derive(Clone, Debug)]
pub struct Xxh64 {
    v: [u64; 4],
    buf: [u8; 32],
    count: u64,
}

impl Default for Xxh64 {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Xxh64 {
    pub fn new(seed: u64) -> Self {
        Self {
            v: [
                seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2),
                seed.wrapping_add(PRIME64_2),
                seed,
                seed.wrapping_sub(PRIME64_1),
            ],
            buf: [0; 32],
            count: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let cnt = (self.count as usize) & 31;
        self.count = self.count.wrapping_add(data.len() as u64);

        if cnt != 0 {
            let rem = (32 - cnt).min(data.len());
            self.buf[cnt..cnt + rem].copy_from_slice(&data[..rem]);
            data = &data[rem..];
            if cnt + rem != 32 {
                return;
            }
            self.absorb_block();
        }
        while data.len() >= 32 {
            let block: [u8; 32] = data[..32].try_into().unwrap();
            self.absorb_arr(&block);
            data = &data[32..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
        }
    }

    fn absorb_block(&mut self) {
        let block = self.buf;
        self.absorb_arr(&block);
    }
    fn absorb_arr(&mut self, b: &[u8; 32]) {
        let v0 = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let v1 = u64::from_le_bytes(b[8..16].try_into().unwrap());
        let v2 = u64::from_le_bytes(b[16..24].try_into().unwrap());
        let v3 = u64::from_le_bytes(b[24..32].try_into().unwrap());
        self.v[0] = round(self.v[0], v0);
        self.v[1] = round(self.v[1], v1);
        self.v[2] = round(self.v[2], v2);
        self.v[3] = round(self.v[3], v3);
    }

    pub fn digest(&self) -> u64 {
        let mut h;
        if self.count >= 32 {
            h = self.v[0].rotate_left(1)
                .wrapping_add(self.v[1].rotate_left(7))
                .wrapping_add(self.v[2].rotate_left(12))
                .wrapping_add(self.v[3].rotate_left(18));
            h = merge(h, self.v[0]);
            h = merge(h, self.v[1]);
            h = merge(h, self.v[2]);
            h = merge(h, self.v[3]);
        } else {
            h = self.v[2].wrapping_add(PRIME64_5);
        }
        h = h.wrapping_add(self.count);

        let mut cnt = (self.count as usize) & 31;
        let buf = &self.buf[..cnt];
        let mut p = 0;
        while cnt >= 8 {
            let v = u64::from_le_bytes(buf[p..p + 8].try_into().unwrap());
            p += 8;
            h ^= round(0, v);
            h = h.rotate_left(27);
            h = h.wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
            cnt -= 8;
        }
        if cnt >= 4 {
            let v = u32::from_le_bytes(buf[p..p + 4].try_into().unwrap()) as u64;
            p += 4;
            h ^= v.wrapping_mul(PRIME64_1);
            h = h.rotate_left(23);
            h = h.wrapping_mul(PRIME64_2).wrapping_add(PRIME64_3);
            cnt -= 4;
        }
        while cnt > 0 {
            let v = buf[p] as u64;
            p += 1;
            h ^= v.wrapping_mul(PRIME64_5);
            h = h.rotate_left(11).wrapping_mul(PRIME64_1);
            cnt -= 1;
        }
        h ^= h >> 33; h = h.wrapping_mul(PRIME64_2);
        h ^= h >> 29; h = h.wrapping_mul(PRIME64_3);
        h ^= h >> 32;
        h
    }
}

pub fn checksum(data: &[u8]) -> u64 {
    let mut h = Xxh64::new(0);
    h.update(data);
    h.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // Reference XXH64 with seed=0:
        assert_eq!(checksum(b""), 0xef46_db37_51d8_e999);
        assert_eq!(checksum(b"a"), 0xd24e_c4f1_a98c_6e5b);
        // Standard "Nobody inspects the spammish repetition" vector.
        assert_eq!(
            checksum(b"Nobody inspects the spammish repetition"),
            0xfbcea83c8a378bf1
        );
    }

    #[test]
    fn streaming() {
        let data: Vec<u8> = (0..2000u32).map(|i| (i * 31 + 1) as u8).collect();
        let one = checksum(&data);
        for split in [0usize, 1, 7, 8, 31, 32, 33, 100, 1999, 2000] {
            let mut h = Xxh64::new(0);
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.digest(), one, "split={split}");
        }
    }
}
