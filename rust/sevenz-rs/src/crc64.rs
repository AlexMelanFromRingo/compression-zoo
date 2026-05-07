//! CRC-64 (XZ / ECMA-182, reflected, polynomial `0xC96C5795D7870F42`,
//! init/xor `0xFFFFFFFFFFFFFFFF`). Port of `7zip/C/XzCrc64.c`.

pub const POLY: u64 = 0xC96C_5795_D787_0F42;
pub const INIT: u64 = 0xFFFF_FFFF_FFFF_FFFF;

#[derive(Clone, Debug)]
struct Tables([[u64; 256]; 4]);

impl Tables {
    const fn new() -> Self {
        let mut t = [[0u64; 256]; 4];
        let mut i = 0;
        while i < 256 {
            let mut r = i as u64;
            let mut j = 0;
            while j < 8 {
                r = (r >> 1) ^ (POLY & 0u64.wrapping_sub(r & 1));
                j += 1;
            }
            t[0][i] = r;
            i += 1;
        }
        let mut k = 1;
        while k < 4 {
            let mut i = 0;
            while i < 256 {
                let v = t[k - 1][i];
                t[k][i] = t[0][(v & 0xFF) as usize] ^ (v >> 8);
                i += 1;
            }
            k += 1;
        }
        Tables(t)
    }
}

static TABLES: Tables = Tables::new();

#[derive(Clone, Debug)]
pub struct Crc64 {
    state: u64,
}

impl Default for Crc64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc64 {
    #[inline]
    pub fn new() -> Self {
        Self { state: INIT }
    }

    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.state = update_raw(self.state, data);
    }

    #[inline]
    pub fn finalize(self) -> u64 {
        self.state ^ INIT
    }

    #[inline]
    pub fn current(&self) -> u64 {
        self.state ^ INIT
    }

    #[inline]
    pub fn reset(&mut self) {
        self.state = INIT;
    }
}

#[inline]
pub fn checksum(data: &[u8]) -> u64 {
    update_raw(INIT, data) ^ INIT
}

pub fn update_raw(mut v: u64, mut data: &[u8]) -> u64 {
    let t = &TABLES.0;
    while data.len() >= 4 {
        let chunk: [u8; 4] = data[..4].try_into().unwrap();
        let w = u32::from_le_bytes(chunk) as u64 ^ (v & 0xFFFF_FFFF);
        v = t[3][(w & 0xFF) as usize]
            ^ t[2][((w >> 8) & 0xFF) as usize]
            ^ t[1][((w >> 16) & 0xFF) as usize]
            ^ t[0][((w >> 24) & 0xFF) as usize]
            ^ (v >> 32);
        data = &data[4..];
    }
    for &b in data {
        v = t[0][((v ^ b as u64) & 0xFF) as usize] ^ (v >> 8);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        assert_eq!(checksum(b""), 0);
    }

    #[test]
    fn known_vectors() {
        // CRC-64/XZ check vectors.
        assert_eq!(checksum(b"123456789"), 0x995D_C9BB_DF19_39FA);
        assert_eq!(checksum(b""), 0);
    }

    #[test]
    fn streaming() {
        let data: Vec<u8> = (0..2000u32).map(|i| (i * 41 + 3) as u8).collect();
        let one = checksum(&data);
        for split in [0usize, 1, 3, 4, 5, 100, 999, 1000, 1999, 2000] {
            let mut h = Crc64::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), one, "split={split}");
        }
    }
}
