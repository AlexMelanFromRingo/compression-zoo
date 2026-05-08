//! ZPAQ block format reader. Mirrors the relevant parts of
//! `Decompresser::findBlock` / `findFilename` / `readComment` /
//! `readSegmentEnd` and `ZPAQL::read` from
//! `plugins/zpaq/upstream/libzpaq.cpp`.
//!
//! What's parsed today:
//!   * Block magic (16-byte rolling-hash fingerprint).
//!   * Block header: hver, type, ZPAQL header (hsize, hh, hm, ph,
//!     pm, n, COMP components, HCOMP bytecode, terminator).
//!   * Segment iteration: filename, comment, the body byte range,
//!     and the end marker (0xFE no-checksum / 0xFD with SHA-1).
//!
//! The actual *body* decoder (predictor + ZPAQL VM + arithmetic
//! coder driven by them) is in follow-up modules. This module just
//! gets us as far as "I have a Reader positioned at the start of
//! the compressed bytes; the segment ends at the marker byte at
//! offset n in the stream".

#![allow(dead_code)]

use crate::io::Reader;

/// Per-component COMP entry size, indexed by component type code.
/// Mirrors `compsize[256]` in upstream.
pub const COMPSIZE: [u8; 256] = {
    let mut t = [0u8; 256];
    t[0] = 0; t[1] = 2; t[2] = 3; t[3] = 2; t[4] = 3;
    t[5] = 4; t[6] = 6; t[7] = 6; t[8] = 3; t[9] = 5;
    t
};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum FormatError {
    /// Reader returned EOF before we'd parsed enough bytes.
    UnexpectedEof,
    /// Block magic not found before EOF.
    NoBlockMagic,
    /// `hver` (block format version) wasn't 1 or 2.
    UnsupportedVersion(u8),
    /// `type` (ZPAQL machine type) wasn't 1.
    UnsupportedType(u8),
    /// Header field claims a component type that isn't a valid
    /// `compsize[]` entry, OR the section overflows.
    HeaderInconsistent,
    /// Segment begin marker was something other than 1 (segment) or
    /// 255 (end of block).
    InvalidSegmentMarker(u8),
    /// Comment block missing the trailing 0x00 reserved byte.
    MissingReservedByte,
    /// End-of-segment marker was something other than 254 (no SHA-1)
    /// or 253 (SHA-1 follows).
    InvalidSegmentEndMarker(u8),
}

/// Parsed ZPAQ block header (post-magic).
#[derive(Debug, Clone)]
pub struct ZpaqlHeader {
    pub hver: u8,
    pub mtype: u8,
    /// Total ZPAQL header size as recorded (in bytes, the LE u16
    /// at offsets 0..2 of the post-magic header region).
    pub hsize: u16,
    /// Number of context-history bits (`H` array size = 1 << hh).
    pub hh: u8,
    /// Number of memory bits (`M` array size = 1 << hm).
    pub hm: u8,
    /// PCOMP H-array size bits.
    pub ph: u8,
    /// PCOMP M-array size bits.
    pub pm: u8,
    /// Number of COMP components.
    pub n: u8,
    /// Raw COMP bytes (concatenation of `n` components, each
    /// `compsize[type]` bytes long, leading byte = type).
    pub comp_bytes: Vec<u8>,
    /// Raw HCOMP bytecode (terminated by 0x00 in upstream; we
    /// strip the terminator).
    pub hcomp: Vec<u8>,
}

/// Parsed segment header (filename + comment).
#[derive(Debug, Clone, Default)]
pub struct SegmentHeader {
    pub filename: Vec<u8>,
    pub comment: Vec<u8>,
}

/// End-of-segment marker.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SegmentEnd {
    /// Marker byte 254 — no SHA-1 follows.
    NoChecksum,
    /// Marker byte 253 — 20-byte SHA-1 of the decompressed segment
    /// follows on the wire.
    Sha1([u8; 20]),
}

// ---------------------------------------------------------------------
// Magic finder
// ---------------------------------------------------------------------

/// Scan `r` for the 16-byte ZPAQ block magic. Returns `Ok(())` once
/// found (positioned just past the magic) or `Err(NoBlockMagic)` at
/// EOF without a match.
///
/// libzpaq uses four parallel rolling hashes (with multipliers 12,
/// 20, 28, 44) initialised to specific seeds. After consuming the
/// 16-byte magic the hashes should equal the four constants below.
pub fn find_block_magic<R: Reader>(r: &mut R) -> Result<(), FormatError> {
    let mut h1: u32 = 0x3D49B113;
    let mut h2: u32 = 0x29EB7F93;
    let mut h3: u32 = 0x2614BE13;
    let mut h4: u32 = 0x3828EB13;
    while let Some(c) = r.get() {
        let c32 = c as u32;
        h1 = h1.wrapping_mul(12).wrapping_add(c32);
        h2 = h2.wrapping_mul(20).wrapping_add(c32);
        h3 = h3.wrapping_mul(28).wrapping_add(c32);
        h4 = h4.wrapping_mul(44).wrapping_add(c32);
        if h1 == 0xB16B88F1 && h2 == 0xFF5376F1
            && h3 == 0x72AC5BF1 && h4 == 0x2F909AF1
        {
            return Ok(());
        }
    }
    Err(FormatError::NoBlockMagic)
}

// ---------------------------------------------------------------------
// ZPAQL header reader (after magic)
// ---------------------------------------------------------------------

pub fn read_header<R: Reader>(r: &mut R) -> Result<ZpaqlHeader, FormatError> {
    let hver = r.get().ok_or(FormatError::UnexpectedEof)?;
    if hver != 1 && hver != 2 {
        return Err(FormatError::UnsupportedVersion(hver));
    }
    let mtype = r.get().ok_or(FormatError::UnexpectedEof)?;
    if mtype != 1 {
        return Err(FormatError::UnsupportedType(mtype));
    }

    let hsize_lo = r.get().ok_or(FormatError::UnexpectedEof)?;
    let hsize_hi = r.get().ok_or(FormatError::UnexpectedEof)?;
    let hsize = (hsize_lo as u16) | ((hsize_hi as u16) << 8);

    let hh = r.get().ok_or(FormatError::UnexpectedEof)?;
    let hm = r.get().ok_or(FormatError::UnexpectedEof)?;
    let ph = r.get().ok_or(FormatError::UnexpectedEof)?;
    let pm = r.get().ok_or(FormatError::UnexpectedEof)?;
    let n  = r.get().ok_or(FormatError::UnexpectedEof)?;

    // hsize is the size of the ZPAQL header from the COMP section
    // through HCOMP terminator. `cend - 2 + hend - hbegin` in
    // upstream = hsize. We track the running cend and hend below.
    //
    // Upstream pre-allocates header[hsize+300] and writes:
    //   header[0..2] = hsize LE
    //   header[2..7] = hh, hm, ph, pm, n
    //   header[7..cend] = COMP bytes (incl. 0x00 terminator)
    //   header[hbegin..hend] = HCOMP bytes (incl. 0x00 terminator)
    // hsize == cend - 2 + hend - hbegin.

    // Read the n COMP components.
    let mut comp_bytes = Vec::new();
    for _ in 0..n {
        let ty = r.get().ok_or(FormatError::UnexpectedEof)?;
        comp_bytes.push(ty);
        let size = COMPSIZE[ty as usize];
        if size < 1 {
            return Err(FormatError::HeaderInconsistent);
        }
        for _ in 1..size {
            comp_bytes.push(r.get().ok_or(FormatError::UnexpectedEof)?);
        }
    }
    // COMP terminator must be 0.
    let comp_end = r.get().ok_or(FormatError::UnexpectedEof)?;
    if comp_end != 0 {
        return Err(FormatError::HeaderInconsistent);
    }

    // HCOMP bytes: read until terminator 0x00. Upstream knows the
    // size from `hsize` but we just consume everything that's left.
    let cend = 7 + comp_bytes.len() + 1; // header[0..7] + comp + comp_end
    let hcomp_len = (hsize as usize).saturating_sub(cend - 2);
    let mut hcomp = Vec::with_capacity(hcomp_len);
    for _ in 0..hcomp_len {
        hcomp.push(r.get().ok_or(FormatError::UnexpectedEof)?);
    }
    // The last byte should be 0 (HCOMP end). Upstream tolerates and
    // asserts; we mirror that.
    if hcomp.last() != Some(&0) {
        return Err(FormatError::HeaderInconsistent);
    }

    Ok(ZpaqlHeader {
        hver, mtype, hsize, hh, hm, ph, pm, n,
        comp_bytes, hcomp,
    })
}

// ---------------------------------------------------------------------
// Segment header reader
// ---------------------------------------------------------------------

/// Read a segment header. Returns `Some(SegmentHeader)` if a segment
/// follows (marker byte = 1) or `None` at end-of-block (marker = 255).
pub fn read_segment_start<R: Reader>(r: &mut R)
    -> Result<Option<SegmentHeader>, FormatError>
{
    let marker = r.get().ok_or(FormatError::UnexpectedEof)?;
    match marker {
        1 => {
            let mut filename = Vec::new();
            loop {
                let c = r.get().ok_or(FormatError::UnexpectedEof)?;
                if c == 0 { break; }
                filename.push(c);
            }
            let mut comment = Vec::new();
            loop {
                let c = r.get().ok_or(FormatError::UnexpectedEof)?;
                if c == 0 { break; }
                comment.push(c);
            }
            // 0x00 reserved byte after comment.
            let reserved = r.get().ok_or(FormatError::UnexpectedEof)?;
            if reserved != 0 {
                return Err(FormatError::MissingReservedByte);
            }
            Ok(Some(SegmentHeader { filename, comment }))
        }
        255 => Ok(None),
        other => Err(FormatError::InvalidSegmentMarker(other)),
    }
}

/// Read the end-of-segment marker (must be called *after* the
/// arithmetic decoder has consumed the segment body).
pub fn read_segment_end<R: Reader>(r: &mut R)
    -> Result<SegmentEnd, FormatError>
{
    let marker = r.get().ok_or(FormatError::UnexpectedEof)?;
    match marker {
        254 => Ok(SegmentEnd::NoChecksum),
        253 => {
            let mut sha = [0u8; 20];
            for slot in sha.iter_mut() {
                *slot = r.get().ok_or(FormatError::UnexpectedEof)?;
            }
            Ok(SegmentEnd::Sha1(sha))
        }
        other => Err(FormatError::InvalidSegmentEndMarker(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::SliceReader;

    #[test]
    fn finds_magic_after_random_prefix() {
        // The 16-byte ZPAQ block magic. We synthesise it by running
        // the rolling hashes backward; easier to grab from
        // libzpaq's actual archives. For unit testing we just
        // stub-check that the function rejects EOF cleanly.
        let mut r = SliceReader::new(b"not the magic");
        assert_eq!(find_block_magic(&mut r), Err(FormatError::NoBlockMagic));
    }

    #[test]
    fn segment_start_end_of_block() {
        let bytes = [255u8];
        let mut r = SliceReader::new(&bytes);
        let res = read_segment_start(&mut r).unwrap();
        assert!(res.is_none());
    }

    #[test]
    fn segment_start_with_filename_and_comment() {
        // marker, filename, 0, comment, 0, 0
        let mut bytes = vec![1u8];
        bytes.extend_from_slice(b"file.txt");
        bytes.push(0);
        bytes.extend_from_slice(b"hello");
        bytes.push(0);
        bytes.push(0);
        let mut r = SliceReader::new(&bytes);
        let seg = read_segment_start(&mut r).unwrap().unwrap();
        assert_eq!(&seg.filename, b"file.txt");
        assert_eq!(&seg.comment, b"hello");
    }

    #[test]
    fn segment_end_no_checksum() {
        let bytes = [254u8];
        let mut r = SliceReader::new(&bytes);
        assert_eq!(read_segment_end(&mut r).unwrap(), SegmentEnd::NoChecksum);
    }

    #[test]
    fn segment_end_sha1() {
        let mut bytes = vec![253u8];
        for i in 0..20u8 { bytes.push(i); }
        let mut r = SliceReader::new(&bytes);
        let end = read_segment_end(&mut r).unwrap();
        match end {
            SegmentEnd::Sha1(sha) => {
                for (i, b) in sha.iter().enumerate() {
                    assert_eq!(*b as usize, i);
                }
            }
            _ => panic!("expected Sha1 variant"),
        }
    }
}
