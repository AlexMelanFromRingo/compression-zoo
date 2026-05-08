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

    /// Start a block from a pre-built header.
    ///
    /// `header` is the full upstream "z.read" payload starting at
    /// the LE u16 hsize: `[hsize_lo, hsize_hi, hh, hm, ph, pm, n,
    /// COMP_bytes..., 0, HCOMP_bytes..., 0]`. `n == 0` is valid —
    /// it means "stored body with optional PCOMP postprocessor",
    /// the layout used by upstream's BWT and LZ77 method paths.
    pub fn start_block_modeled(&mut self, header: &[u8]) -> Result<(), CompressError> {
        if self.state != State::Init { return Err(CompressError::InvalidState); }
        if header.len() < 8 { return Err(CompressError::InvalidHeader); }

        let n = header[6];

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

        // Build the cend/hbegin/hend triple needed by ZpaqlVm.
        let comp_len = cp - 8; // bytes between the n field and the 0 terminator.
        let cend = 7 + comp_len + 1; // hh..n = 5, + comp_bytes, + 0 terminator.
        let hcomp_start = cp;
        let hcomp_len_with_term = header.len() - hcomp_start;
        if hcomp_len_with_term == 0 || header[header.len() - 1] != 0 {
            return Err(CompressError::InvalidHeader);
        }

        let mut buf = vec![0u8; cend + 128 + hcomp_len_with_term + 16];
        buf[..cend].copy_from_slice(&header[..cend]);
        let hbegin = cend + 128;
        buf[hbegin..hbegin + hcomp_len_with_term]
            .copy_from_slice(&header[hcomp_start..]);
        let hend = hbegin + hcomp_len_with_term;

        // Wire: 'zPQ' + level (1 if modeled, 2 if stored body) + mtype + header.
        self.out.put(b'z');
        self.out.put(b'P');
        self.out.put(b'Q');
        self.out.put(1 + (n == 0) as u8);
        self.out.put(1);
        self.out.write(header);

        // Initialise predictor + VM only when there's a model to drive.
        let mut vm = ZpaqlVm::new(buf, hbegin, hend, cend);
        if n != 0 {
            self.predictor.init(&mut vm)
                .map_err(|_| CompressError::InvalidHeader)?;
            self.enc.init();
        }
        self.vm = Some(vm);
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
        // Same dispatch as the data path: arith-encoded for n>0,
        // stored-buffer for n=0 (the BWT/LZ77 method layout).
        if self.n != 0 {
            self.encode_byte(1);
            self.encode_byte((len & 0xFF) as u8);
            self.encode_byte(((len >> 8) & 0xFF) as u8);
            for &b in body { self.encode_byte(b); }
        } else {
            self.write_stored(1);
            self.write_stored((len & 0xFF) as u8);
            self.write_stored(((len >> 8) & 0xFF) as u8);
            for &b in body { self.write_stored(b); }
        }
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

/// Expand a digit-style method ("0", "5", "5,128,1,2", …) into the
/// canonical `"x..."` (or `"0..."`) form upstream's `compressBlock`
/// would produce. Mirrors libzpaq.cpp:7556-7689 verbatim.
///
/// `data` is the input being compressed — used both to derive the
/// block-size argument and (for level 5..=9) to detect periodic
/// patterns and add matching context-map components.
pub fn expand_digit_method(method: &str, data: &[u8]) -> Option<String> {
    let bytes = method.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() { return None; }
    let level: i32 = (bytes[0] - b'0') as i32;
    if !(0..=9).contains(&level) { return None; }

    // Parse trailing comma-separated args (up to 4): arg[0..=3].
    let mut commas = 0usize;
    let mut arg = [0i32; 4];
    let mut i = 1;
    while i < bytes.len() && commas < 4 {
        let c = bytes[i];
        if c == b',' || c == b'.' { commas += 1; }
        else if c.is_ascii_digit() { arg[commas] = arg[commas] * 10 + (c - b'0') as i32; }
        i += 1;
    }

    let n = data.len();
    let arg0 = lg_block_size(n);
    // type=512 if no commas; otherwise R*4 + t (where R=arg[1], t=arg[2]).
    let typ: i32 = if commas == 0 { 512 } else { arg[1] * 4 + arg[2] };

    // Stored: "0<arg0>,0".
    if level == 0 { return Some(format!("0{},0", arg0)); }

    let doe8: i32 = (typ & 2) * 2;       // E8E9 enable bit (0 or 4).
    let mut out = format!("x{}", arg0);
    let htsz = format!(",{}", 19 + arg0 + (if arg0 <= 6 { 1 } else { 0 }));
    let sasz = format!(",{}", 21 + arg0);

    match level {
        1 => {
            if typ < 40 { out.push_str(",0"); }
            else {
                out.push_str(&format!(",{},", 1 + doe8));
                if      typ < 80  { out.push_str("4,0,1,15"); }
                else if typ < 128 { out.push_str("4,0,2,16"); }
                else if typ < 256 { out.push_str(&format!("4,0,2{}", htsz)); }
                else if typ < 960 { out.push_str(&format!("5,0,3{}", htsz)); }
                else              { out.push_str(&format!("6,0,3{}", htsz)); }
            }
        }
        2 => {
            if typ < 32 { out.push_str(",0"); }
            else {
                out.push_str(&format!(",{},", 1 + doe8));
                if typ < 64 { out.push_str(&format!("4,0,3{}", htsz)); }
                else        { out.push_str(&format!("4,0,7{},1", sasz)); }
            }
        }
        3 => {
            if typ < 20 { out.push_str(",0"); }
            else if typ < 48 {
                out.push_str(&format!(",{},4,0,3{}", 1 + doe8, htsz));
            } else if typ >= 640 || (typ & 1) != 0 {
                out.push_str(&format!(",{}ci1", 3 + doe8));
            } else {
                out.push_str(&format!(",{},12,0,7{},1c0,0,511i2", 2 + doe8, sasz));
            }
        }
        4 => {
            if typ < 12 { out.push_str(",0"); }
            else if typ < 24 {
                out.push_str(&format!(",{},4,0,3{}", 1 + doe8, htsz));
            } else if typ < 48 {
                out.push_str(&format!(",{},5,0,7{}1c0,0,511", 2 + doe8, sasz));
            } else if typ < 900 {
                out.push_str(&format!(",{}ci1,1,1,1,2a", doe8));
                if (typ & 1) != 0 { out.push('w'); }
                out.push('m');
            } else {
                out.push_str(&format!(",{}ci1", 3 + doe8));
            }
        }
        5..=9 => {
            // Slow CM with lots of models.
            out.push_str(&format!(",{}", doe8));
            if (typ & 1) != 0 { out.push_str("w2c0,1010,255i1"); }
            else              { out.push_str("w1i1"); }
            out.push_str("c256ci1,1,1,1,1,1,2a");

            // Periodic-model analysis: for each byte, count gap to its
            // previous occurrence; pick the two strongest gaps and add
            // matching context-map components.
            const NR: usize = 1 << 12;
            let mut pt = [0i32; 256];
            let mut r = vec![0i32; NR];
            for (i, &b) in data.iter().enumerate() {
                let k = i as i32 - pt[b as usize];
                if k > 0 && (k as usize) < NR { r[k as usize] += 1; }
                pt[b as usize] = i as i32;
            }
            let mut n1 = n as i32 - r[1] - r[2] - r[3];
            for _ in 0..2 {
                let mut period = 0i32;
                let mut score = 0.0f64;
                let mut t = 0i32;
                for j in 5..NR {
                    if t >= n1 { break; }
                    let s = r[j] as f64 / (256.0 + (n1 - t) as f64);
                    if s > score { score = s; period = j as i32; }
                    t += r[j];
                }
                if period > 4 && score > 0.1 {
                    out.push_str(&format!("c0,0,{},255i1", 999 + period));
                    if period <= 255 {
                        out.push_str(&format!("c0,{}i1", period));
                    }
                    n1 -= r[period as usize];
                    r[period as usize] = 0;
                } else { break; }
            }
            out.push_str("c0,2,0,255i1c0,3,0,0,255i1c0,4,0,0,0,255i1mm16ts19t0");
        }
        _ => return None,
    }

    Some(out)
}

/// `MAX(lg(n+4095) - 20, 0)` — upstream's block-size argument
/// computation. `lg` is the upstream "round-up log2" so e.g.
/// `lg(1) = 1`, `lg(2) = 1`, `lg(3) = 2`, `lg(4) = 2`, `lg(5) = 3`.
fn lg_block_size(n: usize) -> i32 {
    let v = (n as u64).saturating_add(4095);
    if v <= 1 { return 0; }
    let mut log = 0i32;
    let mut x = v - 1;
    while x > 0 { log += 1; x >>= 1; }
    (log - 20).max(0)
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

/// High-level entry point — mirrors upstream's `compress(in, out,
/// method, ...)` for a curated subset of methods.
///
/// Supported method strings:
///   * `"0"` — stored mode (no model, no preprocessing).
///   * `"1"` — alias for `"x4,1,4"` (variable-bit LZ77, min match 4).
///   * `"2"` — alias for `"x4,2,4"` (byte-aligned LZ77, min match 4).
///   * `"3"` — alias for `"x4,3"`   (BWT, 16 MiB block).
///   * `"x4,1,M"` — variable-bit LZ77 with min match `M ∈ [4..64]`.
///   * `"x4,2,M"` — byte-aligned LZ77 with min match `M ∈ [1..64]`.
///   * `"x4,3"`   — BWT.
///   * `"x4,5,M"` — variable-bit LZ77 + E8E9 prefilter.
///   * `"x4,6,M"` — byte-aligned LZ77 + E8E9 prefilter.
///   * `"x4,7"`   — BWT + E8E9 prefilter.
///
/// Unsupported strings return [`CompressError::InvalidHeader`].
pub fn compress_method<W: Writer>(
    out: W,
    data: &[u8],
    method: &str,
) -> Result<W, CompressError> {
    // Expand digit-method strings (e.g. "5", "5,B,R,t") into the
    // canonical "x..." form, mirroring upstream's `compressBlock`
    // type-inference at libzpaq.cpp:7556-7689. Plain "0" keeps the
    // dedicated stored-mode fast path below.
    let expanded;
    let method: &str = if !method.is_empty()
        && method.as_bytes()[0].is_ascii_digit()
        && method != "0"
    {
        expanded = expand_digit_method(method, data)
            .ok_or(CompressError::InvalidHeader)?;
        &expanded
    } else {
        method
    };

    let mut c = Compresser::new(out);
    c.write_tag()?;

    if method == "0" {
        c.start_block_stored()?;
        c.start_segment(b"", b"")?;
        c.post_process_pass()?;
        c.write_bytes(data)?;
        c.end_segment(None)?;
        c.end_block()?;
        return Ok(c.into_inner());
    }

    // Any "x..." method goes through the upstream-compatible
    // `make_config` builder. Component specs (`ci1`, `i`, `m`, `t`,
    // `s`, `a`, `w`) are honoured here.
    if method.starts_with('x') {
        let (cfg, args) = crate::make_config::make_config(method)
            .map_err(|_| CompressError::InvalidHeader)?;
        let cc = crate::compiler::compile_with_args(&cfg, args)
            .map_err(|_| CompressError::InvalidHeader)?;
        let level = args[1] & 3;
        let pcomp_bytes = cc.pcomp.clone();

        c.start_block_modeled(&cc.header)?;
        c.start_segment(b"", b"")?;
        match (level, pcomp_bytes) {
            (0, _) => c.post_process_pass()?,
            (_, Some(pc)) => c.post_process_prog(&pc)?,
            _ => return Err(CompressError::InvalidHeader),
        }
        // Pre-process input per level (level 0 = no preproc).
        let preprocessed: Vec<u8>;
        let body: &[u8] = match level {
            0 => data,
            1 | 2 | 3 => {
                let lvl_args = crate::lzbuffer::LzArgs {
                    log_block_mib: args[0] as u32,
                    level_flag: args[1] as u32,
                    min_match: args[2].max(if level == 1 { 4 } else { 1 }) as u32,
                    min_match2: args[3] as u32,
                    log_bucket: args[4].max(0) as u32,
                    log_ht_size: args[5].max(16) as u32,
                };
                preprocessed = crate::lzbuffer::preprocess(data, lvl_args);
                &preprocessed
            }
            _ => return Err(CompressError::InvalidHeader),
        };
        c.write_bytes(body)?;
        c.end_segment(None)?;
        c.end_block()?;
        return Ok(c.into_inner());
    }

    Err(CompressError::InvalidHeader)
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

    /// End-to-end via `compress_method("x4,3")`: BWT preproc +
    /// canned IBWT PCOMP. Validates the LZBuffer-level-3 →
    /// Compresser → Decompresser → IBWT pipeline.
    #[test]
    fn compress_method_bwt_round_trip() {
        let inp = b"banana mining banana band ".repeat(40);
        let out = compress_method(VecWriter::new(), &inp, "x4,3").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn compress_method_zero_round_trip() {
        let inp = b"Hello stored".to_vec();
        let out = compress_method(VecWriter::new(), &inp, "0").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    /// E8E9 LZ77 / BWT round-trip with input that contains
    /// `0xE8` / `0xE9` opcode bytes followed by valid offsets.
    /// Without the inverse-E8E9 step in PCOMP, these positions
    /// would be corrupted by the encoder-side prefilter.
    #[test]
    fn compress_method_e8e9_variants() {
        let mut inp: Vec<u8> = Vec::new();
        inp.extend_from_slice(b"prefix bytes ");
        inp.extend_from_slice(&[0xE8, 0x10, 0x00, 0x00, 0x00]);
        inp.extend_from_slice(b" middle ");
        inp.extend_from_slice(&[0xE9, 0x20, 0x00, 0x00, 0xFF]);
        inp.extend_from_slice(b" tail bytes for padding ".repeat(20).as_slice());
        for method in ["x4,5,4", "x4,6,4", "x4,7"] {
            let out = compress_method(VecWriter::new(), &inp, method).unwrap();
            let wire = out.into_inner();
            let mut r = SliceReader::new(&wire);
            let mut w = VecWriter::new();
            decompress(&mut r, &mut w).unwrap();
            assert_eq!(w.into_inner(), inp,
                "round-trip failed for method='{}'", method);
        }
    }

    /// Component-spec methods round-trip through both the Rust
    /// decoder and the upstream wire format. Adds a context model
    /// on top of the LZ77 / BWT preprocessor.
    #[test]
    fn compress_method_with_component_specs() {
        let inp = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
        for method in [
            "x4,3,ci1",                 // BWT + ICM + ISSE
            "x4,3,c0,0,0,255i1",        // BWT + masked CM + ISSE
            "x4,2,4ci1",                // byte-LZ77 + ICM + ISSE
            "x4,1,4ci1",                // var-bit LZ77 + ICM + ISSE
        ] {
            let out = compress_method(VecWriter::new(), &inp, method).unwrap();
            let wire = out.into_inner();
            let mut r = SliceReader::new(&wire);
            let mut w = VecWriter::new();
            decompress(&mut r, &mut w).unwrap();
            assert_eq!(w.into_inner(), inp,
                "round-trip failed for method='{}'", method);
        }
    }

    /// Digit-method aliases ("1", "2", "3") should resolve to the
    /// same `"x4,..."` paths and round-trip identically.
    #[test]
    #[test]
    fn expand_digit_method_zero_stored() {
        // Level 0 → "0<arg0>,0" with arg0 derived from n.
        let exp = expand_digit_method("0", &vec![0u8; 100]).unwrap();
        assert_eq!(exp, "00,0");
    }

    #[test]
    fn expand_digit_method_no_commas_uses_type_512() {
        // "1" with no commas → type=512 → falls into the `else if
        // type<960` arm at level 1: "x<arg0>,1,5,0,3<htsz>".
        let exp = expand_digit_method("1", &vec![0u8; 100]).unwrap();
        assert_eq!(exp, "x0,1,5,0,3,20");
    }

    #[test]
    fn expand_digit_method_level2_with_explicit_args() {
        // Upstream's digit format is "LB,R,t": L=level digit, then B
        // is one or more digits glued onto the level (arg[0]),
        // followed by R (arg[1]) and t (arg[2]) after commas. Type
        // is `R*4 + t`.
        //
        // "20,0,0" → arg=[0,0,0,0], type=0 → type<32 → just ",0".
        let exp = expand_digit_method("20,0,0", &vec![0u8; 100]).unwrap();
        assert_eq!(exp, "x0,0");
        // "20,128,1" → arg=[0,128,1,0], type=128*4+1=513.
        // type≥64 → "4,0,7<sasz>,1".
        let exp = expand_digit_method("20,128,1", &vec![0u8; 100]).unwrap();
        // doe8 = (513&2)*2 = 0; sasz = ",21" for arg0=0.
        assert_eq!(exp, "x0,1,4,0,7,21,1");
    }

    #[test]
    fn expand_digit_method_level3_default_text_branch() {
        // "3" with no commas → type=512. Level 3 with type=512 ≥ 256
        // takes the `else` branch (LZ77+CM), not the BWT branch.
        let exp = expand_digit_method("3", &vec![0u8; 100]).unwrap();
        // 1+doe8 where doe8 = (512&2)*2 = 0. Followed by ",2,12,0,7,21,1c0,0,511i2".
        assert!(exp.contains("c0,0,511i2"),
            "expected level-3 LZ77+CM expansion, got {}", exp);
    }

    #[test]
    fn expand_digit_method_level5_includes_periodic_models() {
        // Build a periodic input (period 7, 1024 bytes).
        let data: Vec<u8> = (0..1024).map(|i| (i % 7) as u8).collect();
        let exp = expand_digit_method("5", &data).unwrap();
        // Should include the periodic model component
        // `c0,0,<999+period>,255i1`. With period=7 → "c0,0,1006,255i1".
        assert!(exp.contains("c0,0,1006,255i1") || exp.contains("c0,0,100"),
            "expected periodic model in: {}", exp);
        // Trailing tail is fixed.
        assert!(exp.ends_with("c0,2,0,255i1c0,3,0,0,255i1c0,4,0,0,0,255i1mm16ts19t0"),
            "missing fixed tail in: {}", exp);
    }

    #[test]
    fn compress_method_level1_round_trip() {
        // Level 1 with default type=512 → BWT-less LZ77 path.
        let inp: Vec<u8> = (0..2000).map(|i| (i % 251) as u8).collect();
        let out = compress_method(VecWriter::new(), &inp, "1").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn compress_method_level3_round_trip() {
        // Level 3 with default type=512 takes the LZ77+CM branch
        // (`,2,12,0,7<sasz>,1c0,0,511i2`).
        let inp: Vec<u8> = (0..2000).map(|i| (i % 251) as u8).collect();
        let out = compress_method(VecWriter::new(), &inp, "3").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn compress_method_level5_round_trip() {
        // Level 5..9 share the slow-CM branch — periodic-model
        // analysis + a long fixed component tail. Round-trip a small
        // periodic input to exercise the path.
        let inp: Vec<u8> = (0..1024).map(|i| (i % 7) as u8).collect();
        let out = compress_method(VecWriter::new(), &inp, "5").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn compress_method_level4_round_trip() {
        // Level 4 with default type=512 (≥900) → ",3ci1" — BWT with
        // CM+ISSE component spec, exercised end-to-end.
        let inp: Vec<u8> = (0..2000).map(|i| (i % 251) as u8).collect();
        let out = compress_method(VecWriter::new(), &inp, "4").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
    }

    #[test]
    fn expand_digit_method_lg_block_size_scales_with_input() {
        // n < 1MB → arg0 = 0; n in [1MB, 2MB) → arg0 = 1; etc.
        let small  = expand_digit_method("0", &vec![0u8; 100]).unwrap();
        let medium = expand_digit_method("0", &vec![0u8; 1 << 20]).unwrap();
        assert_eq!(small,  "00,0");
        // 1 MiB → block-size argument = 1 (or higher).
        assert!(medium.starts_with("01,") || medium.starts_with("02,"),
            "expected medium arg0 ≥ 1, got {}", medium);
    }

    #[test]
    fn compress_method_digit_aliases() {
        let inp = b"the quick brown fox jumps over the lazy dog. ".repeat(20);
        for method in ["1", "2", "3"] {
            let out = compress_method(VecWriter::new(), &inp, method).unwrap();
            let wire = out.into_inner();
            let mut r = SliceReader::new(&wire);
            let mut w = VecWriter::new();
            decompress(&mut r, &mut w).unwrap();
            assert_eq!(w.into_inner(), inp,
                "round-trip failed for method='{}'", method);
        }
    }

    /// `x4,1,M` — variable-bit Elias-gamma LZ77. Validates the
    /// bit-packed encoder + canned PCOMP integration.
    #[test]
    fn compress_method_lz77_var_round_trip() {
        let inp = b"the quick brown fox jumps over the lazy dog. ".repeat(40);
        let out = compress_method(VecWriter::new(), &inp, "x4,1,4").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
        assert!(wire.len() < inp.len());
    }

    /// `x4,2,M` — byte-aligned LZ77. Repetitive input should
    /// compress measurably while still round-tripping.
    #[test]
    fn compress_method_lz77_round_trip() {
        let inp = b"the quick brown fox jumps over the lazy dog. ".repeat(40);
        let out = compress_method(VecWriter::new(), &inp, "x4,2,4").unwrap();
        let wire = out.into_inner();
        let mut r = SliceReader::new(&wire);
        let mut w = VecWriter::new();
        decompress(&mut r, &mut w).unwrap();
        assert_eq!(w.into_inner(), inp);
        // Smoke check: compressed should be smaller than input for
        // this very repetitive fixture.
        assert!(wire.len() < inp.len(),
            "lz77 didn't compress repetitive input: {} → {}",
            inp.len(), wire.len());
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
