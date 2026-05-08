//! Canned ZPAQ models — bytewise copies of the three headers
//! upstream ships in `Compressor::startBlock(int level)`
//! (`plugins/zpaq/upstream/libzpaq.cpp:2796`):
//!
//!   * [`MIN_CFG`] — level 1 model (min.cfg, 28 bytes).
//!   * [`MID_CFG`] — level 2 model (mid.cfg, 71 bytes).
//!   * [`MAX_CFG`] — level 3 model (max.cfg, 217 bytes).
//!
//! Each blob is the wire format starting at the LE u16 `hsize`
//! prefix, exactly the bytes [`crate::compress::Compresser::start_block_modeled`]
//! expects.

#![allow(dead_code)]

/// Level 1 model — `min.cfg`. ICM context order 2, hashed via HCOMP.
pub const MIN_CFG: &[u8] = &[
    26, 0, 1, 2, 0, 0, 2, 3, 16, 8, 19, 0, 0, 96, 4, 28,
    59, 10, 59, 112, 25, 10, 59, 10, 59, 112, 56, 0,
];

/// Level 2 model — `mid.cfg`. ICM + 5×ISSE + MIX over 7 components.
pub const MID_CFG: &[u8] = &[
    69, 0, 3, 3, 0, 0, 8, 3, 5, 8, 13, 0, 8, 17, 1, 8,
    18, 2, 8, 18, 3, 8, 19, 4, 4, 22, 24, 7, 16, 0, 7, 24,
    255, 0, 17, 104, 74, 4, 95, 1, 59, 112, 10, 25, 59, 112, 10, 25,
    59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 10, 59, 112,
    25, 69, 207, 8, 112, 56, 0,
];

/// Level 3 model — `max.cfg`. 22 components incl. ICM/ISSE/MATCH/MIX/SSE.
pub const MAX_CFG: &[u8] = &[
    196, 0, 5, 9, 0, 0, 22, 1, 160, 3, 5, 8, 13, 1, 8, 16,
    2, 8, 18, 3, 8, 19, 4, 8, 19, 5, 8, 20, 6, 4, 22, 24,
    3, 17, 8, 19, 9, 3, 13, 3, 13, 3, 13, 3, 14, 7, 16, 0,
    15, 24, 255, 7, 8, 0, 16, 10, 255, 6, 0, 15, 16, 24, 0, 9,
    8, 17, 32, 255, 6, 8, 17, 18, 16, 255, 9, 16, 19, 32, 255, 6,
    0, 19, 20, 16, 0, 0, 17, 104, 74, 4, 95, 2, 59, 112, 10, 25,
    59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25,
    59, 10, 59, 112, 10, 25, 59, 112, 10, 25, 69, 183, 32, 239, 64, 47,
    14, 231, 91, 47, 10, 25, 60, 26, 48, 134, 151, 20, 112, 63, 9, 70,
    223, 0, 39, 3, 25, 112, 26, 52, 25, 25, 74, 10, 4, 59, 112, 25,
    10, 4, 59, 112, 25, 10, 4, 59, 112, 25, 65, 143, 212, 72, 4, 59,
    112, 8, 143, 216, 8, 68, 175, 60, 60, 25, 69, 207, 9, 112, 25, 25,
    25, 25, 25, 112, 56, 0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_sizes_match_hsize_prefix() {
        for cfg in [MIN_CFG, MID_CFG, MAX_CFG] {
            let hsize = (cfg[0] as usize) | ((cfg[1] as usize) << 8);
            // The blob is `hsize + 2` bytes total (hsize prefix +
            // hsize bytes of payload). Verify upstream's invariant.
            assert_eq!(cfg.len(), hsize + 2);
        }
    }
}
