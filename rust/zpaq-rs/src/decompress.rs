//! Top-level ZPAQ decompressor (subset).
//!
//! Currently supports:
//!   * Method "0" archives (`n == 0` components): the body is a
//!     sequence of 4-byte BE length-prefixed chunks terminated by 4
//!     zero bytes; no arithmetic coding, no predictor, no ZPAQL VM.
//!   * PostProcessor PASS mode (mode marker = 0): the decompressed
//!     bytes pass straight through to the output.
//!
//! What's missing (TODO, multi-day work):
//!   * Predictor with all 8 component types (CONS/CM/ICM/MATCH/AVG/
//!     MIX2/MIX/ISSE/SSE).
//!   * ZPAQL VM (interpreter for HCOMP/PCOMP bytecode).
//!   * PostProcessor PROG mode (PCOMP bytecode → arbitrary
//!     post-processor, used by LZ77/BWT/E8E9 methods).
//!
//! When asked to decompress an archive that needs the missing
//! pieces we return [`DecompressError::ModeledNotSupported`] /
//! [`DecompressError::PcompNotSupported`].

#![allow(dead_code)]

use crate::arith::Decoder;
use crate::format::{self, FormatError, SegmentEnd, ZpaqlHeader};
use crate::io::{Reader, Writer};
use crate::predictor::Predictor;
use crate::sha1::Sha1;
use crate::zpaql::ZpaqlVm;

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum DecompressError {
    Format(FormatError),
    /// Archive uses a modeled (`n > 0`) block, but the predictor +
    /// ZPAQL VM port hasn't landed yet.
    ModeledNotSupported,
    /// PostProcessor PROG mode (PCOMP-driven post-processing) isn't
    /// supported yet.
    PcompNotSupported,
    /// Stored-mode body ended unexpectedly.
    UnexpectedEof,
    /// Recorded SHA-1 didn't match the bytes we emitted.
    Sha1Mismatch { expected: [u8; 20], actual: [u8; 20] },
}

impl From<FormatError> for DecompressError {
    fn from(e: FormatError) -> Self { DecompressError::Format(e) }
}

/// Information about one decompressed segment.
#[derive(Debug, Clone)]
pub struct DecompressedSegment {
    pub filename: Vec<u8>,
    pub comment: Vec<u8>,
    pub bytes_written: usize,
    pub end: SegmentEnd,
    /// `true` when end == Sha1 and our recompute matched.
    pub sha1_verified: bool,
}

/// Decompress all blocks/segments from `r`, writing the
/// concatenated output to `w`. Returns one entry per segment.
pub fn decompress<R: Reader, W: Writer>(
    r: &mut R, w: &mut W,
) -> Result<Vec<DecompressedSegment>, DecompressError> {
    let mut segments: Vec<DecompressedSegment> = Vec::new();

    loop {
        match format::find_block_magic(r) {
            Ok(()) => {}
            Err(FormatError::NoBlockMagic) => break,
            Err(e) => return Err(e.into()),
        }
        let header = format::read_header(r)?;
        decompress_block(r, w, &header, &mut segments)?;
    }
    Ok(segments)
}

fn decompress_block<R: Reader, W: Writer>(
    r: &mut R,
    w: &mut W,
    header: &ZpaqlHeader,
    segments: &mut Vec<DecompressedSegment>,
) -> Result<(), DecompressError> {
    if header.n != 0 {
        return decompress_modeled_block(r, w, header, segments);
    }

    loop {
        let seg = match format::read_segment_start(r)? {
            Some(s) => s,
            None => return Ok(()), // end of block
        };

        let mut sha = Sha1::new();
        let mut written = 0usize;

        // Stored-mode "Decoder::decompress" produces bytes by
        // running 4-byte BE length-prefixed chunks. The first byte
        // of the decoded stream is the PostProcessor mode marker
        // (0 = PASS, 1 = PROG). We only support PASS today.
        let mut pp_state = PpState::Init;
        // Stored-mode body still goes through the PostProcessor;
        // the PCOMP that follows uses ph/pm from the block header to
        // size its H and M arrays. Passing 0/0 here was a bug — it
        // collapsed M to size 1 so every write to M[B] aliased to
        // M[0], which silently corrupts LZ77 back-references.
        let ph = header.ph;
        let pm = header.pm;
        decode_stored_body(r, |byte_or_eof| -> Result<(), DecompressError> {
            pp_write(byte_or_eof, &mut pp_state, w, &mut sha, &mut written, ph, pm)
        })?;

        let end = format::read_segment_end(r)?;
        let mut sha1_verified = false;
        if let SegmentEnd::Sha1(expected) = end {
            let actual = sha.finalize();
            if actual == expected {
                sha1_verified = true;
            } else {
                return Err(DecompressError::Sha1Mismatch { expected, actual });
            }
        }

        segments.push(DecompressedSegment {
            filename: seg.filename,
            comment: seg.comment,
            bytes_written: written,
            end,
            sha1_verified,
        });
    }
}

enum PpState {
    Init,
    Pass,
    /// PROG: reading low byte of PCOMP size.
    ProgSizeLo,
    /// PROG: reading high byte of PCOMP size, with the low half saved.
    ProgSizeHi { lo: u32 },
    /// PROG: streaming PCOMP bytecode into a fresh ZpaqlVm. Once
    /// `remaining` reaches 0 the VM is ready for `Post`.
    ProgLoading { vm_header: Vec<u8>, write_idx: usize, remaining: u32, ph: u8, pm: u8 },
    /// POST: run the PCOMP VM per input byte; OUT writes output.
    Post { vm: Box<crate::zpaql::ZpaqlVm> },
}

/// Reconstruct the full upstream-compatible header buffer from a
/// parsed `ZpaqlHeader`. Required because the predictor / ZPAQL VM
/// indexes into the header at well-known offsets (cend, hbegin, hend).
fn rebuild_header_buffer(header: &ZpaqlHeader)
    -> (Vec<u8>, usize /*cend*/, usize /*hbegin*/, usize /*hend*/)
{
    // Layout:
    //   [0..2]   hsize LE
    //   [2..7]   hh, hm, ph, pm, n
    //   [7..]    COMP bytes
    //   [cend-1] = 0 (COMP terminator)
    //   [cend..cend+128] = padding
    //   [hbegin..hend] = HCOMP bytes (last byte = 0)
    let hsize = header.hsize as usize;
    let cend = 7 + header.comp_bytes.len() + 1;
    let hbegin = cend + 128;
    let hend = hbegin + header.hcomp.len();
    let mut buf = vec![0u8; hend + 16]; // small slack for safety
    buf[0] = (hsize & 0xFF) as u8;
    buf[1] = (hsize >> 8) as u8;
    buf[2] = header.hh;
    buf[3] = header.hm;
    buf[4] = header.ph;
    buf[5] = header.pm;
    buf[6] = header.n;
    buf[7..7 + header.comp_bytes.len()].copy_from_slice(&header.comp_bytes);
    // buf[cend - 1] is the COMP terminator, already 0.
    buf[hbegin..hend].copy_from_slice(&header.hcomp);
    (buf, cend, hbegin, hend)
}

/// Modeled-mode block decompress: instantiate the predictor, the
/// arithmetic decoder, and the post-processor, then loop emitting
/// bytes per segment.
fn decompress_modeled_block<R: Reader, W: Writer>(
    r: &mut R,
    w: &mut W,
    header: &ZpaqlHeader,
    segments: &mut Vec<DecompressedSegment>,
) -> Result<(), DecompressError> {
    let (header_buf, cend, hbegin, hend) = rebuild_header_buffer(header);
    let mut vm = ZpaqlVm::new(header_buf, hbegin, hend, cend);
    let mut predictor = Predictor::new();
    predictor.init(&mut vm).map_err(|_| {
        DecompressError::Format(FormatError::HeaderInconsistent)
    })?;

    let mut first_segment = true;

    loop {
        let seg = match format::read_segment_start(r)? {
            Some(s) => s,
            None => return Ok(()),
        };

        let mut sha = Sha1::new();
        let mut written = 0usize;

        // Re-init coder per segment if first; subsequent segments
        // share the predictor state.
        let mut dec = Decoder::new();
        if first_segment {
            dec.init_modeled();
            // pp.init: just reset the post-processor state.
        }
        // Read 4 bytes for curr.
        dec.fill_curr(r).map_err(|_| DecompressError::UnexpectedEof)?;

        // Inner decoder.decompress loop until -1. Decoded bytes
        // flow through the PostProcessor (which may be PASS or
        // PROG/PCOMP). This mirrors upstream's
        // `while (n) { pp.write(dec.decompress()); ... }` loop —
        // initially we don't know whether we're in PASS or PROG and
        // need pp_write to handle both paths.
        let mut pp_state = PpState::Init;
        let ph = header.ph;
        let pm = header.pm;
        loop {
            let byte_or_eof = decode_byte(&mut dec, &mut predictor, &mut vm, r)?;
            match byte_or_eof {
                None => {
                    if let PpState::Post { vm: ppvm } = &mut pp_state {
                        let mut sink = TrackedWriter { inner: w, sha: &mut sha, count: &mut written };
                        let _ = ppvm.run(0xFFFF_FFFF, Some(&mut sink), None);
                    }
                    break;
                }
                Some(b) => {
                    pp_write(Some(b), &mut pp_state, w, &mut sha, &mut written, ph, pm)?;
                }
            }
        }

        let end = format::read_segment_end(r)?;
        let mut sha1_verified = false;
        if let SegmentEnd::Sha1(expected) = end {
            let actual = sha.finalize();
            if actual == expected {
                sha1_verified = true;
            } else {
                return Err(DecompressError::Sha1Mismatch { expected, actual });
            }
        }

        segments.push(DecompressedSegment {
            filename: seg.filename,
            comment: seg.comment,
            bytes_written: written,
            end,
            sha1_verified,
        });
        first_segment = false;
    }
}

fn decode_byte<R: Reader>(
    dec: &mut Decoder,
    pr: &mut Predictor,
    vm: &mut ZpaqlVm,
    r: &mut R,
) -> Result<Option<u8>, DecompressError> {
    if dec.decode(r, 0).map_err(|_| DecompressError::Format(FormatError::HeaderInconsistent))? != 0 {
        return Ok(None);
    }
    let mut c: u32 = 1;
    while c < 256 {
        let p = pr.predict(vm) as u32;
        let prob = (p * 2 + 1) & 0xFFFF;
        let bit = dec.decode(r, prob).map_err(|_| DecompressError::Format(FormatError::HeaderInconsistent))?;
        c = c + c + bit;
        pr.update(c & 1, vm);
    }
    Ok(Some((c - 256) as u8))
}

/// Helper writer that forwards through to a Writer, updates a SHA-1,
/// and tracks bytes written.
struct TrackedWriter<'a, W: Writer> {
    inner: &'a mut W,
    sha: &'a mut Sha1,
    count: &'a mut usize,
}

impl<'a, W: Writer> Writer for TrackedWriter<'a, W> {
    fn put(&mut self, c: u8) {
        self.inner.put(c);
        self.sha.update(&[c]);
        *self.count += 1;
    }
    fn write(&mut self, buf: &[u8]) {
        self.inner.write(buf);
        self.sha.update(buf);
        *self.count += buf.len();
    }
}

/// PostProcessor write: feed one decoded byte through the
/// post-processor state machine. Mirrors `PostProcessor::write`.
/// `b == None` is treated as end-of-segment (only used by callers
/// to signal "you're done"; the upstream version would call it via
/// `pp.write(-1)` but we handle that at the outer loop).
fn pp_write<W: Writer>(
    b: Option<u8>,
    state: &mut PpState,
    w: &mut W,
    sha: &mut Sha1,
    written: &mut usize,
    ph: u8,
    pm: u8,
) -> Result<(), DecompressError> {
    let c = match b { Some(x) => x, None => return Ok(()) };
    let new_state: Option<PpState> = match state {
        PpState::Init => {
            match c {
                0 => Some(PpState::Pass),
                1 => Some(PpState::ProgSizeLo),
                _ => return Err(DecompressError::Format(FormatError::HeaderInconsistent)),
            }
        }
        PpState::Pass => {
            w.put(c);
            sha.update(&[c]);
            *written += 1;
            None
        }
        PpState::ProgSizeLo => {
            Some(PpState::ProgSizeHi { lo: c as u32 })
        }
        PpState::ProgSizeHi { lo } => {
            let hsize = *lo + (c as u32) * 256;
            // Allocate the PCOMP header. Layout:
            //   [0..2]   hsize LE (low byte = hsize_lo, high = hsize_hi)
            //   [2..7]   hh, hm, ph, pm, n=0
            //   [7]      0 (COMP terminator)
            //   [8..136] 128-byte gap
            //   [136..136+hsize] PCOMP bytecode
            let mut buf = vec![0u8; (hsize as usize) + 256];
            buf[0] = (hsize & 0xFF) as u8;
            buf[1] = ((hsize >> 8) & 0xFF) as u8;
            buf[2] = 0; buf[3] = 0; // hh, hm — PCOMP doesn't use them
            buf[4] = ph; buf[5] = pm; buf[6] = 0;
            // buf[7] already 0 (COMP terminator).
            let write_idx = 7 + 1 + 128;
            Some(PpState::ProgLoading {
                vm_header: buf,
                write_idx,
                remaining: hsize,
                ph, pm,
            })
        }
        PpState::ProgLoading { vm_header, write_idx, remaining, ph: _, pm: _ } => {
            vm_header[*write_idx] = c;
            *write_idx += 1;
            *remaining -= 1;
            if *remaining == 0 {
                let cend = 9; // 7 fields + COMP terminator + (cend = 8?)
                let _ = cend;
                let cend = 8usize;
                let hbegin = cend + 128;
                let hend = *write_idx;
                let header_buf = std::mem::take(vm_header);
                let mut vm = crate::zpaql::ZpaqlVm::new(header_buf, hbegin, hend, cend);
                vm.init_pcomp();
                Some(PpState::Post { vm: Box::new(vm) })
            } else {
                None
            }
        }
        PpState::Post { vm } => {
            let mut sink = TrackedWriter { inner: w, sha, count: written };
            vm.run(c as u32, Some(&mut sink), None)
                .map_err(|_| DecompressError::Format(FormatError::HeaderInconsistent))?;
            None
        }
    };
    if let Some(s) = new_state { *state = s; }
    Ok(())
}

/// Pulls bytes off the stored-mode body via 4-byte BE length-prefixed
/// chunks (terminated by a length of 0). Calls `sink(Some(b))` for
/// each decoded byte and `sink(None)` once at end-of-stream.
fn decode_stored_body<R: Reader, F>(
    r: &mut R,
    mut sink: F,
) -> Result<(), DecompressError>
where
    F: FnMut(Option<u8>) -> Result<(), DecompressError>,
{
    loop {
        let mut len_bytes = [0u8; 4];
        for slot in len_bytes.iter_mut() {
            *slot = r.get().ok_or(DecompressError::UnexpectedEof)?;
        }
        let n = u32::from_be_bytes(len_bytes);
        if n == 0 {
            // End-of-stream sentinel.
            sink(None)?;
            return Ok(());
        }
        for _ in 0..n {
            let b = r.get().ok_or(DecompressError::UnexpectedEof)?;
            sink(Some(b))?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceReader, VecWriter};

    /// Unit: a well-formed stored body decodes via decode_stored_body.
    #[test]
    fn stored_body_round_trip() {
        // length=3 "abc" length=2 "de" length=0 -> EOF
        let mut wire = Vec::new();
        wire.extend_from_slice(&3u32.to_be_bytes());
        wire.extend_from_slice(b"abc");
        wire.extend_from_slice(&2u32.to_be_bytes());
        wire.extend_from_slice(b"de");
        wire.extend_from_slice(&0u32.to_be_bytes());

        let mut r = SliceReader::new(&wire);
        let mut got: Vec<u8> = Vec::new();
        let mut hit_eof = false;
        decode_stored_body(&mut r, |b| {
            match b {
                Some(byte) => got.push(byte),
                None => hit_eof = true,
            }
            Ok(())
        }).unwrap();
        assert!(hit_eof);
        assert_eq!(got, b"abcde");
    }
}
