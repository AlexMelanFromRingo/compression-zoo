//! Block header format port: `bsc_block_info` and the constants it
//! relies on. Mirrors `plugins/bsc/upstream/libbsc/libbsc/libbsc.cpp`.
//!
//! A libbsc block on the wire is:
//!
//!   bytes  0..4   block_size (i32 LE) — total compressed length, including header
//!   bytes  4..8   data_size  (i32 LE) — original input length
//!   bytes  8..12  mode       (i32 LE) — packed (block_sorter, coder, lzp_min_len, lzp_hash_size)
//!   bytes 12..16  index      (i32 LE) — BWT primary index (or sentinel)
//!   bytes 16..20  data_adler (i32 LE) — Adler-32 of the original data
//!   bytes 20..24  body_adler (i32 LE) — Adler-32 of the post-coder bytes
//!   bytes 24..28  hdr_adler  (i32 LE) — Adler-32 of bytes 0..24
//!
//! followed by `block_size - 28` bytes of body.

use crate::adler32;

pub const LIBBSC_HEADER_SIZE: usize = 28;

pub const LIBBSC_BLOCKSORTER_NONE: i32 = 0;
pub const LIBBSC_BLOCKSORTER_BWT:  i32 = 1;
pub const LIBBSC_BLOCKSORTER_ST3:  i32 = 3;
pub const LIBBSC_BLOCKSORTER_ST4:  i32 = 4;
pub const LIBBSC_BLOCKSORTER_ST5:  i32 = 5;
pub const LIBBSC_BLOCKSORTER_ST6:  i32 = 6;
pub const LIBBSC_BLOCKSORTER_ST7:  i32 = 7;
pub const LIBBSC_BLOCKSORTER_ST8:  i32 = 8;

pub const LIBBSC_CODER_NONE:          i32 = 0;
pub const LIBBSC_CODER_QLFC_STATIC:   i32 = 1;
pub const LIBBSC_CODER_QLFC_ADAPTIVE: i32 = 2;
pub const LIBBSC_CODER_QLFC_FAST:     i32 = 3;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum BlockInfoError {
    /// Header is shorter than `LIBBSC_HEADER_SIZE`.
    UnexpectedEob,
    /// One of the header checksums is wrong, or the (sorter, coder,
    /// lzp_*) fields are inconsistent with the encoded `mode` value.
    DataCorrupt,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct BlockInfo {
    /// Total bytes on the wire for this block, including the 28-byte
    /// header.
    pub block_size: i32,
    /// Bytes of the original input encoded into this block.
    pub data_size: i32,
    /// BWT primary index (only meaningful for BWT blocks).
    pub index: i32,
    pub block_sorter: i32,
    pub coder: i32,
    pub lzp_min_len: i32,
    pub lzp_hash_size: i32,
}

#[inline]
fn read_i32_le(b: &[u8], at: usize) -> i32 {
    let bytes = [b[at], b[at + 1], b[at + 2], b[at + 3]];
    i32::from_le_bytes(bytes)
}

/// Port of `bsc_block_info` from libbsc.cpp.
///
/// Verifies the header's own Adler-32 (`bytes 24..28` should equal
/// `adler32(bytes 0..24)`), then unpacks the `mode` field and returns
/// the decoded sorter/coder/LZP parameters along with `block_size` and
/// `data_size`.
pub fn block_info(header: &[u8]) -> Result<BlockInfo, BlockInfoError> {
    if header.len() < LIBBSC_HEADER_SIZE {
        return Err(BlockInfoError::UnexpectedEob);
    }

    let recorded = read_i32_le(header, 24) as u32;
    if recorded != adler32::adler32(&header[..24]) {
        return Err(BlockInfoError::DataCorrupt);
    }

    let block_size = read_i32_le(header, 0);
    let data_size  = read_i32_le(header, 4);
    let mode       = read_i32_le(header, 8) as u32;
    let index      = read_i32_le(header, 12);

    let lzp_hash_size = ((mode >> 16) & 0xff) as i32;
    let lzp_min_len   = ((mode >>  8) & 0xff) as i32;
    let coder         = ((mode >>  5) & 0x7)  as i32;
    let block_sorter  = ((mode >>  0) & 0x1f) as i32;

    // Re-derive the canonical packed `mode` and compare. This mirrors
    // upstream's invariant check.
    let mut test_mode: i32 = 0;
    match block_sorter {
        LIBBSC_BLOCKSORTER_NONE => {}
        LIBBSC_BLOCKSORTER_BWT
        | LIBBSC_BLOCKSORTER_ST3
        | LIBBSC_BLOCKSORTER_ST4
        | LIBBSC_BLOCKSORTER_ST5
        | LIBBSC_BLOCKSORTER_ST6
        | LIBBSC_BLOCKSORTER_ST7
        | LIBBSC_BLOCKSORTER_ST8 => test_mode = block_sorter,
        _ => return Err(BlockInfoError::DataCorrupt),
    }

    match coder {
        LIBBSC_CODER_NONE => {}
        LIBBSC_CODER_QLFC_STATIC
        | LIBBSC_CODER_QLFC_ADAPTIVE
        | LIBBSC_CODER_QLFC_FAST => test_mode += coder << 5,
        _ => return Err(BlockInfoError::DataCorrupt),
    }

    if lzp_min_len != 0 || lzp_hash_size != 0 {
        if lzp_min_len < 4 || lzp_min_len > 255 {
            return Err(BlockInfoError::DataCorrupt);
        }
        if lzp_hash_size < 10 || lzp_hash_size > 28 {
            return Err(BlockInfoError::DataCorrupt);
        }
        test_mode += lzp_min_len << 8;
        test_mode += lzp_hash_size << 16;
    }

    if test_mode as u32 != mode {
        return Err(BlockInfoError::DataCorrupt);
    }

    if block_size <= LIBBSC_HEADER_SIZE as i32 || data_size < 0 {
        return Err(BlockInfoError::DataCorrupt);
    }

    Ok(BlockInfo {
        block_size,
        data_size,
        index,
        block_sorter,
        coder,
        lzp_min_len,
        lzp_hash_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_header(
        block_size: i32, data_size: i32, mode: u32, index: i32,
        data_adler: u32, body_adler: u32,
    ) -> Vec<u8> {
        let mut h = Vec::with_capacity(LIBBSC_HEADER_SIZE);
        h.extend_from_slice(&block_size.to_le_bytes());
        h.extend_from_slice(&data_size.to_le_bytes());
        h.extend_from_slice(&mode.to_le_bytes());
        h.extend_from_slice(&index.to_le_bytes());
        h.extend_from_slice(&data_adler.to_le_bytes());
        h.extend_from_slice(&body_adler.to_le_bytes());
        let hdr_adler = adler32::adler32(&h);
        h.extend_from_slice(&hdr_adler.to_le_bytes());
        debug_assert_eq!(h.len(), LIBBSC_HEADER_SIZE);
        h
    }

    #[test]
    fn rejects_short_header() {
        assert_eq!(block_info(&[]), Err(BlockInfoError::UnexpectedEob));
        assert_eq!(block_info(&[0u8; 27]), Err(BlockInfoError::UnexpectedEob));
    }

    #[test]
    fn rejects_bad_header_adler() {
        let mut h = pack_header(100, 50, LIBBSC_BLOCKSORTER_BWT as u32, 0, 0, 0);
        h[24] ^= 0xFF; // corrupt the header adler
        assert_eq!(block_info(&h), Err(BlockInfoError::DataCorrupt));
    }

    #[test]
    fn parses_bwt_qlfc_static_no_lzp() {
        // mode = block_sorter (BWT=1) + (coder=QLFC_STATIC=1) << 5 = 1 + 32 = 33
        let mode = (LIBBSC_BLOCKSORTER_BWT as u32) | ((LIBBSC_CODER_QLFC_STATIC as u32) << 5);
        let h = pack_header(200, 100, mode, 7, 0xdead, 0xbeef);
        let info = block_info(&h).unwrap();
        assert_eq!(info.block_size, 200);
        assert_eq!(info.data_size, 100);
        assert_eq!(info.index, 7);
        assert_eq!(info.block_sorter, LIBBSC_BLOCKSORTER_BWT);
        assert_eq!(info.coder, LIBBSC_CODER_QLFC_STATIC);
        assert_eq!(info.lzp_min_len, 0);
        assert_eq!(info.lzp_hash_size, 0);
    }

    #[test]
    fn parses_bwt_qlfc_with_lzp() {
        // BWT (1) + QLFC_STATIC (1) << 5 + min_len (72) << 8 + hash (15) << 16
        let mode = (LIBBSC_BLOCKSORTER_BWT as u32)
            | ((LIBBSC_CODER_QLFC_STATIC as u32) << 5)
            | (72u32 << 8)
            | (15u32 << 16);
        let h = pack_header(500, 200, mode, 0, 0, 0);
        let info = block_info(&h).unwrap();
        assert_eq!(info.lzp_min_len, 72);
        assert_eq!(info.lzp_hash_size, 15);
    }

    #[test]
    fn rejects_unknown_sorter() {
        // Use sorter id 2 (LIBBSC_BLOCKSORTER_NONE=0, BWT=1, ST3..ST8=3..8 — 2 unused)
        let mode = 2u32;
        let h = pack_header(100, 50, mode, 0, 0, 0);
        assert_eq!(block_info(&h), Err(BlockInfoError::DataCorrupt));
    }

    #[test]
    fn rejects_unknown_coder() {
        // coder id 5 (only 1..3 valid)
        let mode = (LIBBSC_BLOCKSORTER_BWT as u32) | (5u32 << 5);
        let h = pack_header(100, 50, mode, 0, 0, 0);
        assert_eq!(block_info(&h), Err(BlockInfoError::DataCorrupt));
    }

    #[test]
    fn rejects_invalid_lzp_min_len() {
        // min_len = 3 (must be >= 4 when LZP enabled)
        let mode = (LIBBSC_BLOCKSORTER_BWT as u32)
            | ((LIBBSC_CODER_QLFC_STATIC as u32) << 5)
            | (3u32 << 8)
            | (15u32 << 16);
        let h = pack_header(100, 50, mode, 0, 0, 0);
        assert_eq!(block_info(&h), Err(BlockInfoError::DataCorrupt));
    }
}
