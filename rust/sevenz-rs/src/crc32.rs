//! CRC-32 (reflected, polynomial `0xEDB88320`, init/xor `0xFFFFFFFF`).
//! Port of `7zip/C/7zCrc.c` and `7zip/C/7zCrcOpt.c`.
//!
//! Bit-for-bit compatible with the reference: the same algorithm as zlib /
//! IEEE 802.3 CRC-32. Implementation uses slicing-by-8 over a 256·8 table
//! generated at startup; this matches the C code's `CrcUpdateT8`.

pub const POLY: u32 = 0xEDB8_8320;
pub const INIT: u32 = 0xFFFF_FFFF;

/// Eight 256-entry tables for slicing-by-8.
#[derive(Clone, Debug)]
struct Tables([[u32; 256]; 8]);

impl Tables {
    const fn new() -> Self {
        let mut t = [[0u32; 256]; 8];
        let mut i = 0;
        while i < 256 {
            let mut r = i as u32;
            let mut j = 0;
            while j < 8 {
                r = (r >> 1) ^ (POLY & 0u32.wrapping_sub(r & 1));
                j += 1;
            }
            t[0][i] = r;
            i += 1;
        }
        let mut k = 1;
        while k < 8 {
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

/// Streaming CRC-32 hasher.
#[derive(Clone, Debug)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Create a fresh hasher initialised with `INIT`.
    #[inline]
    pub fn new() -> Self {
        Self { state: INIT }
    }

    /// Feed more bytes.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        self.state = update_raw(self.state, data);
    }

    /// Finalise — applies the trailing XOR.
    #[inline]
    pub fn finalize(self) -> u32 {
        self.state ^ INIT
    }

    /// Same as [`finalize`] but doesn't consume `self`.
    #[inline]
    pub fn current(&self) -> u32 {
        self.state ^ INIT
    }

    /// Reset to the initial state.
    #[inline]
    pub fn reset(&mut self) {
        self.state = INIT;
    }
}

/// One-shot CRC-32 of `data` (matches `CrcCalc`).
#[inline]
pub fn checksum(data: &[u8]) -> u32 {
    update_raw(INIT, data) ^ INIT
}

/// Update a raw (un-XORed) CRC value, matching the C `CrcUpdate` function.
pub fn update_raw(mut v: u32, mut data: &[u8]) -> u32 {
    let t = &TABLES.0;

    while data.len() >= 8 {
        // Slicing-by-8: process 8 bytes per iteration.
        let chunk: [u8; 8] = data[..8].try_into().unwrap();
        let lo = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) ^ v;
        let hi = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        v = t[7][(lo & 0xFF) as usize]
            ^ t[6][((lo >> 8) & 0xFF) as usize]
            ^ t[5][((lo >> 16) & 0xFF) as usize]
            ^ t[4][((lo >> 24) & 0xFF) as usize]
            ^ t[3][(hi & 0xFF) as usize]
            ^ t[2][((hi >> 8) & 0xFF) as usize]
            ^ t[1][((hi >> 16) & 0xFF) as usize]
            ^ t[0][((hi >> 24) & 0xFF) as usize];
        data = &data[8..];
    }
    for &b in data {
        v = t[0][((v ^ b as u32) & 0xFF) as usize] ^ (v >> 8);
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
        // Standard CRC-32 (IEEE 802.3) test vectors.
        assert_eq!(checksum(b"a"), 0xE8B7BE43);
        assert_eq!(checksum(b"abc"), 0x352441C2);
        assert_eq!(
            checksum(b"The quick brown fox jumps over the lazy dog"),
            0x414FA339
        );
        assert_eq!(checksum(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn streaming_equals_one_shot() {
        let data: Vec<u8> = (0..1024u32).map(|i| (i * 37 + 11) as u8).collect();
        let one = checksum(&data);
        for split in [0usize, 1, 7, 8, 9, 100, 511, 512, 513, 1023, 1024] {
            let mut h = Crc32::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), one, "mismatch at split={split}");
        }
    }
}
