//! Top-level ZPAQ compressor (subset).
//!
//! Mirrors `Compressor::*` in `plugins/zpaq/upstream/libzpaq.cpp`,
//! with two encode paths:
//!   * `start_block_modeled(hcomp_bytes)` runs the Predictor + ZPAQL
//!     VM under arithmetic coding (the same components that
//!     `decompress.rs` already understands).
//!   * `start_block_stored()` selects libzpaq's `n=0` store mode,
//!     where the body is just length-prefixed raw chunks — no model.
//!
//! What's missing on purpose (out of scope for the v1 port):
//!   * `Compiler` — we don't parse libzpaq config strings yet.
//!     Callers must hand us the COMP+HCOMP bytecode directly. The
//!     bundled [`models`] module exposes upstream's three canned
//!     models (min.cfg, mid.cfg, max.cfg).
//!   * Preprocessing (LZ77, BWT, E8E9). Levels 1–5 of upstream's
//!     `compress(method)` API rely on these and are intentionally
//!     unsupported here.

#![allow(dead_code)]

use crate::arith::Encoder;
use crate::format::COMPSIZE;
use crate::io::Writer;
use crate::predictor::Predictor;
use crate::zpaql::ZpaqlVm;

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum CompressError {
    /// Caller-provided header didn't parse: COMP terminator missing,
    /// component type unknown, or HCOMP missing its trailing 0x00.
    InvalidHeader,
    /// Method calls used out of order (start_segment before
    /// start_block, etc.). State machine bug on the caller's side.
    InvalidState,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
enum State {
    /// Before any block — only `write_tag` and `start_block_*` valid.
    Init,
    /// Inside a block but no segment yet.
    Block,
    /// Inside a segment, before any data byte (post_process pending).
    SegPre,
    /// Inside a segment, post-processor declared, body in progress.
    SegBody,
}

/// 13-byte ZPAQ locator tag. Combined with the 'z' 'P' 'Q' written by
/// `start_block_*` it forms the 16-byte block magic that
/// `format::find_block_magic` scans for.
pub const ZPAQ_TAG: [u8; 13] = [
    0x37, 0x6b, 0x53, 0x74, 0xa0, 0x31, 0x83, 0xd3, 0x8c, 0xb2, 0x28, 0xb0, 0xd3,
];

/// Streaming ZPAQ compresser. Mirrors the upstream `Compressor`
/// state machine: write_tag → start_block_* → (start_segment →
/// post_process_* → write_byte* → end_segment)+ → end_block.
pub struct Compresser<W: Writer> {
    state: State,
    /// `n == 0` selects the `Encoder`-bypass stored-mode path.
    n: u8,
    out: W,
    enc: Encoder,
    predictor: Predictor,
    vm: Option<ZpaqlVm>,
    /// Stored-mode buffer (used only when `n == 0`). Capacity 64 KB
    /// matches upstream's `1<<16`.
    stored_buf: Vec<u8>,
}

impl<W: Writer> Compresser<W> {
    pub fn new(out: W) -> Self {
        Self {
            state: State::Init,
            n: 0,
            out,
            enc: Encoder::new(),
            predictor: Predictor::new(),
            vm: None,
            stored_buf: Vec::new(),
        }
    }

    pub fn into_inner(self) -> W { self.out }

    /// Write the 13-byte locator tag. Optional but cheap: it lets
    /// `find_block_magic` resync after corrupted data, and it's what
    /// `compressBlock` does in upstream.
    pub fn write_tag(&mut self) -> Result<(), CompressError> {
        if self.state != State::Init { return Err(CompressError::InvalidState); }
        for b in ZPAQ_TAG { self.out.put(b); }
        Ok(())
    }

    /// Start a stored-mode (n == 0) block. The body bypasses the
    /// predictor and arithmetic coder; bytes are emitted in 64 KB
    /// length-prefixed chunks terminated by a 0-length sentinel.
    pub fn start_block_stored(&mut self) -> Result<(), CompressError> {
        if self.state != State::Init { return Err(CompressError::InvalidState); }
        // Header: hsize=0, hh=hm=ph=pm=0, n=0, COMP terminator (0),
        // HCOMP terminator (0). Total = 9 bytes including the LE
        // hsize prefix.
        self.out.put(b'z');
        self.out.put(b'P');
        self.out.put(b'Q');
        // Level byte: upstream emits `1 + (n == 0)` ⇒ 2 for stored.
        self.out.put(2);
        self.out.put(1); // mtype
        // hsize = 7: hh hm ph pm n (5) + COMP terminator (1) + HCOMP
        // terminator (1). Mirrors `/tmp/zpaq_make 0` wire output.
        self.out.put(7); // hsize_lo
        self.out.put(0); // hsize_hi
        self.out.put(0); // hh
        self.out.put(0); // hm
        self.out.put(0); // ph
        self.out.put(0); // pm
        self.out.put(0); // n
        self.out.put(0); // COMP terminator
        // No HCOMP body, but upstream still emits the trailing 0
        // terminator. Verified by inspecting m0 archives produced by
        // /tmp/zpaq_make.
        self.out.put(0);
        self.n = 0;
        self.state = State::Block;
        Ok(())
    }

    /// Start a modeled-mode block from a pre-built header.
    ///
    /// `hcomp_bytes` is the full upstream "z.read" payload starting
    /// at the LE u16 hsize: `[hsize_lo, hsize_hi, hh, hm, ph, pm, n,
    /// COMP_bytes..., 0, HCOMP_bytes..., 0]`.
    pub fn start_block_modeled(&mut self, header: &[u8]) -> Result<(), CompressError> {
        if self.state != State::Init { return Err(CompressError::InvalidState); }
        if header.len() < 8 { return Err(CompressError::InvalidHeader); }

        let hsize = (header[0] as usize) | ((header[1] as usize) << 8);
        // hsize counts COMP + COMP terminator(1) + HCOMP. Header layout
        // bytes: [hsize_lo, hsize_hi, hh, hm, ph, pm, n, ...].
        let n = header[6];
        if n == 0 { return Err(CompressError::InvalidHeader); }

        // Sanity-walk the COMP entries.
        let mut cp = 7usize;
        for _ in 0..n {
            if cp >= header.len() { return Err(CompressError::InvalidHeader); }
            let ty = header[cp] as usize;
            if ty == 0 || ty > 9 { return Err(CompressError::InvalidHeader); }
            cp += COMPSIZE[ty] as usize;
        }
        // COMP terminator.
        if cp >= header.len() || header[cp] != 0 {
            return Err(CompressError::InvalidHeader);
        }
        cp += 1;
        // hsize must end at the HCOMP terminator. Layout:
        //   2 (hsize prefix) + hh hm ph pm n | comp | 0 | hcomp | 0
        //   hsize == cend-2 + (hend-hbegin), where cend-2 == 5+comp_len+1 (n..=COMP_term).
        // So hsize == comp_len + 1 + hcomp_len(+terminator).
        let total_after_prefix = 5 + (cp - 7) + (header.len() - cp);
        // Forgive `total_after_prefix == hsize + 5` (header includes
        // the post-COMP HCOMP body).
        let _ = (hsize, total_after_prefix);

        // Build the cend/hbegin/hend triple needed by ZpaqlVm.
        let comp_len = cp - 8; // bytes including COMP-terminator? cp-7-1 = comp_len; cp now points past terminator.
        let cend = 7 + comp_len + 1; // hh..n=5, +comp, +0
        let hcomp_start = cp;
        let hcomp_len_with_term = header.len() - hcomp_start;
        if hcomp_len_with_term == 0 || header[header.len() - 1] != 0 {
            return Err(CompressError::InvalidHeader);
        }

        // Materialise the upstream-style header buffer: prefix the
        // user-supplied bytes with 0 padding so cend..hbegin contains
        // 128 zero bytes (matches the layout the VM/predictor expect).
        // Upstream layout:
        //   [0..2]   hsize LE
        //   [2..7]   hh hm ph pm n
        //   [7..cend] COMP + 0
        //   [cend..hbegin] 128 padding zeros
        //   [hbegin..hend] HCOMP + 0
        let mut buf = vec![0u8; cend + 128 + hcomp_len_with_term + 16];
        buf[..cend].copy_from_slice(&header[..cend]);
        let hbegin = cend + 128;
        buf[hbegin..hbegin + hcomp_len_with_term]
            .copy_from_slice(&header[hcomp_start..]);
        let hend = hbegin + hcomp_len_with_term;

        // Wire write: 'zPQ' + level + mtype + hsize + ... + HCOMP + 0
        self.out.put(b'z');
        self.out.put(b'P');
        self.out.put(b'Q');
        self.out.put(1); // level 1 (modeled)
        self.out.put(1); // mtype
        // Write the rest of the header (everything after the magic
        // and level/mtype): hsize_lo .. HCOMP terminator.
        self.out.write(header);

        // Initialise predictor + VM.
        let mut vm = ZpaqlVm::new(buf, hbegin, hend, cend);
        self.predictor.init(&mut vm)
            .map_err(|_| CompressError::InvalidHeader)?;
        self.vm = Some(vm);
        self.enc.init();
        self.n = n;
        self.state = State::Block;
        Ok(())
    }

    /// Begin a segment with the given filename and comment. Both may
    /// be empty (`b""`).
    pub fn start_segment(&mut self, filename: &[u8], comment: &[u8])
        -> Result<(), CompressError>
    {
        if self.state != State::Block { return Err(CompressError::InvalidState); }
        self.out.put(1); // segment marker
        self.out.write(filename);
        self.out.put(0);
        self.out.write(comment);
        self.out.put(0);
        self.out.put(0); // reserved
        self.state = State::SegPre;
        Ok(())
    }

    /// Declare the post-processor as PASS (mode marker 0): the
    /// decoded bytes flow to the output unchanged. This is the
    /// equivalent of upstream's `postProcess(0, 0)`.
    pub fn post_process_pass(&mut self) -> Result<(), CompressError> {
        if self.state != State::SegPre { return Err(CompressError::InvalidState); }
        if self.n != 0 {
            // Modeled mode: the PostProcessor mode marker is the
            // first byte of the encoded stream.
            self.encode_byte(0);
        } else {
            // Stored mode: the marker is the first byte of the
            // length-prefixed body stream.
            self.write_stored(0);
        }
        self.state = State::SegBody;
        Ok(())
    }

    /// Declare the post-processor as PROG (mode marker 1) with the
    /// given PCOMP bytecode. After decoding, every byte is fed
    /// through the PCOMP ZPAQL VM, whose OUT bytes flow to the
    /// archive consumer. Mirror of upstream's `postProcess(pcomp, len)`.
    ///
    /// `pcomp` should be the raw bytecode from
    /// [`crate::compiler::CompiledConfig::pcomp`] (without any
    /// trailing 0 framing — we add that here).
    pub fn post_process_prog(&mut self, pcomp: &[u8]) -> Result<(), CompressError> {
        if self.state != State::SegPre { return Err(CompressError::InvalidState); }
        if self.n == 0 {
            // upstream forbids stored-mode + PCOMP.
            return Err(CompressError::InvalidState);
        }
        // Wire format: 1 (PROG marker), len_lo, len_hi, pcomp[0..len].
        // upstream's pcomp_len excludes the trailing 0 we added in the
        // Compiler — strip it back off here so the decoder gets the
        // exact byte count it expects.
        let body: &[u8] = if pcomp.last() == Some(&0) {
            &pcomp[..pcomp.len() - 1]
        } else {
            pcomp
        };
        let len = body.len();
        if len > 0xFFFF { return Err(CompressError::InvalidState); }
        self.encode_byte(1);
        self.encode_byte((len & 0xFF) as u8);
        self.encode_byte(((len >> 8) & 0xFF) as u8);
        for &b in body { self.encode_byte(b); }
        self.state = State::SegBody;
        Ok(())
    }

    /// Append one input byte to the current segment.
    pub fn write_byte(&mut self, b: u8) -> Result<(), CompressError> {
        if self.state != State::SegBody { return Err(CompressError::InvalidState); }
        if self.n != 0 {
            self.encode_byte(b);
        } else {
            self.write_stored(b);
        }
        Ok(())
    }

    /// Append a chunk of input bytes.
    pub fn write_bytes(&mut self, buf: &[u8]) -> Result<(), CompressError> {
        for &b in buf { self.write_byte(b)?; }
        Ok(())
    }

    /// Close the current segment, optionally recording a SHA-1.
    pub fn end_segment(&mut self, sha1: Option<&[u8; 20]>) -> Result<(), CompressError> {
        if self.state != State::SegBody { return Err(CompressError::InvalidState); }
        if self.n != 0 {
            // Modeled: encode the EOF marker bit (1 at p=0).
            self.enc.encode(&mut self.out, 1, 0);
            // The 4 trailing zero bytes are explicit framing; upstream
            // writes them after enc.compress(-1) returns.
            for _ in 0..4 { self.out.put(0); }
        } else {
            // Stored: flush remaining buffer with its 4-byte BE
            // length, then write 4-byte zero sentinel for "no more
            // chunks".
            self.flush_stored();
            for _ in 0..4 { self.out.put(0); }
        }
        match sha1 {
            Some(h) => { self.out.put(253); self.out.write(h); }
            None    => { self.out.put(254); }
        }
        self.state = State::Block;
        Ok(())
    }

    /// Close the block (writes the terminator).
    pub fn end_block(&mut self) -> Result<(), CompressError> {
        if self.state != State::Block { return Err(CompressError::InvalidState); }
        self.out.put(255);
        self.state = State::Init;
        Ok(())
    }

    // ---- internals ------------------------------------------------

    fn encode_byte(&mut self, c: u8) {
        // 0 marker bit + 8 data bits MSB-first, mirroring upstream.
        self.enc.encode(&mut self.out, 0, 0);
        let vm = self.vm.as_mut().expect("vm");
        let cu = c as u32;
        for i in (0..8).rev() {
            let p = self.predictor.predict(vm) as u32 * 2 + 1;
            let y = (cu >> i) & 1;
            self.enc.encode(&mut self.out, y, p);
            self.predictor.update(y, vm);
        }
    }

    fn write_stored(&mut self, b: u8) {
        if self.stored_buf.len() == 0x1_0000 {
            self.flush_stored();
        }
        self.stored_buf.push(b);
    }

    fn flush_stored(&mut self) {
        if self.stored_buf.is_empty() { return; }
        let len = self.stored_buf.len() as u32;
        self.out.put(((len >> 24) & 0xFF) as u8);
        self.out.put(((len >> 16) & 0xFF) as u8);
        self.out.put(((len >> 8) & 0xFF) as u8);
        self.out.put((len & 0xFF) as u8);
        let buf = std::mem::take(&mut self.stored_buf);
        self.out.write(&buf);
    }
}

/// Convenience: wrap a stored-mode round-trip into one call.
pub fn compress_stored<W: Writer>(
    out: W,
    data: &[u8],
    filename: &[u8],
    comment: &[u8],
) -> Result<W, CompressError> {
    let mut c = Compresser::new(out);
    c.write_tag()?;
    c.start_block_stored()?;
    c.start_segment(filename, comment)?;
    c.post_process_pass()?;
    c.write_bytes(data)?;
    c.end_segment(None)?;
    c.end_block()?;
    Ok(c.into_inner())
}

/// Convenience: compile a config string and encode `data` against
/// the resulting header in one call. PASS post-processor only —
/// callers that need PCOMP should drive `Compresser` manually.
pub fn compress_with_config<W: Writer>(
    out: W,
    data: &[u8],
    config: &str,
    filename: &[u8],
    comment: &[u8],
) -> Result<W, CompressError> {
    let cc = crate::compiler::compile(config)
        .map_err(|_| CompressError::InvalidHeader)?;
    compress_modeled(out, data, &cc.header, filename, comment)
}

/// Convenience: wrap a modeled-mode round-trip into one call.
pub fn compress_modeled<W: Writer>(
    out: W,
    data: &[u8],
    header: &[u8],
    filename: &[u8],
    comment: &[u8],
) -> Result<W, CompressError> {
    let mut c = Compresser::new(out);
    c.write_tag()?;
    c.start_block_modeled(header)?;
    c.start_segment(filename, comment)?;
    c.post_process_pass()?;
    c.write_bytes(data)?;
    c.end_segment(None)?;
    c.end_block()?;
    Ok(c.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decompress::decompress;
    use crate::io::{SliceReader, VecWriter};

    #[test]
    fn stored_round_trip_short() {
        let inp = b"Hello, ZPAQ!".to_vec();
        let out = compress_stored(VecWriter::new(), &inp, b"", b"").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn stored_round_trip_empty() {
        let out = compress_stored(VecWriter::new(), b"", b"", b"").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), b"");
    }

    /// End-to-end: compile a custom config, encode, decode (Rust),
    /// verify round-trip.
    #[test]
    fn config_string_round_trip_min_cfg() {
        let cfg = r#"
            comp 1 2 0 0 2
              0 icm 16
              1 isse 19 0
            hcomp
              *b=a a=0 d=0 hash b-- hash *d=a d++ b-- hash b-- hash *d=a halt
            post 0 end
        "#;
        let inp = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
        let out = compress_with_config(VecWriter::new(), &inp, cfg, b"", b"").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn stored_round_trip_64k_plus() {
        // Crosses the 1<<16 buffer boundary so we exercise the
        // mid-segment flush path.
        let mut inp = Vec::with_capacity(1 << 17);
        for i in 0..(1 << 17) {
            inp.push((i & 0xFF) as u8);
        }
        let out = compress_stored(VecWriter::new(), &inp, b"", b"").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }
}
