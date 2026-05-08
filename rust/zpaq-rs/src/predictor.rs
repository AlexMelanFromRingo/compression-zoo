//! ZPAQ Predictor — context-mixing model. Port of `Predictor` from
//! `plugins/zpaq/upstream/libzpaq.cpp` (~500 lines of original).
//!
//! 8 component types are recognised in upstream. We port them all
//! using the scalar (`predict0` / `update0`) reference path; the
//! JIT (`assemble_p`) is intentionally skipped.
//!
//! Component opcodes (from the COMP section of a ZPAQ header):
//!   1 = CONS   c
//!   2 = CM     sizebits limit
//!   3 = ICM    sizebits
//!   4 = MATCH  sizebits bufbits
//!   5 = AVG    j k wt
//!   6 = MIX2   sizebits j k rate mask
//!   7 = MIX    sizebits j m rate mask
//!   8 = ISSE   sizebits j
//!   9 = SSE    sizebits j start limit

#![allow(dead_code)]

use crate::predictor_tables::{SDT, SDT2K, SSQUASHT, STDT};
use crate::state_table::SNS;
use crate::zpaql::ZpaqlVm;

// Component type codes.
pub const NONE: u8 = 0;
pub const CONS: u8 = 1;
pub const CM:   u8 = 2;
pub const ICM:  u8 = 3;
pub const MATCH:u8 = 4;
pub const AVG:  u8 = 5;
pub const MIX2: u8 = 6;
pub const MIX:  u8 = 7;
pub const ISSE: u8 = 8;
pub const SSE:  u8 = 9;

/// `compsize[type]` — byte length of each component's COMP entry.
pub const COMPSIZE: [u8; 16] = [0, 2, 3, 2, 3, 4, 6, 6, 3, 5, 0, 0, 0, 0, 0, 0];

#[derive(Default, Clone)]
pub struct Component {
    pub limit: usize,
    pub cxt: usize,
    pub a: u32,
    pub b: u32,
    pub c: usize,
    pub cm: Vec<u32>,
    pub ht: Vec<u8>,
    pub a16: Vec<u16>,
    /// Mask used to wrap `cm`/`ht` indices (`size - 1`); cached for speed.
    cm_mask: usize,
    ht_mask: usize,
    a16_mask: usize,
}

impl Component {
    fn new() -> Self { Self::default() }
    fn cm_get(&self, i: usize) -> u32 {
        self.cm[i & self.cm_mask]
    }
    fn cm_at(&mut self, i: usize) -> &mut u32 {
        let idx = i & self.cm_mask;
        &mut self.cm[idx]
    }
    fn ht_get(&self, i: usize) -> u8 {
        self.ht[i & self.ht_mask]
    }
    fn ht_at(&mut self, i: usize) -> &mut u8 {
        let idx = i & self.ht_mask;
        &mut self.ht[idx]
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum PredictorError {
    /// Predictor state out of sync with the bytestream.
    StateInvalid,
    /// Header references a component type we don't support.
    UnknownComponent(u8),
    /// Component dimension exceeds libzpaq's documented cap.
    OutOfRange,
}

pub struct Predictor {
    /// Predictor sees the current byte being decoded as a 9-bit
    /// register: 1 followed by the bits decoded so far, so values
    /// range from 1 (no bits) up to 511 (8 bits, a full byte).
    /// Per upstream this stays in [1..=255] mid-byte.
    c8: u32,
    /// Tracks bit position split into nibbles for ICM/ISSE.
    hmap4: i32,
    /// Per-component prediction (12-bit signed, -2048..=2047).
    p: [i32; 256],
    /// Per-component context hash (computed by HCOMP).
    h: [u32; 256],
    pub comp: Vec<Component>, // size = n
    /// Cached `squash` lookup: `[i+2048]` for i in -2048..=2047.
    squasht: Vec<u16>,
    /// Cached `stretch` lookup: `[i]` for i in 0..=32767.
    stretcht: Vec<i16>,
    init_tables: bool,
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            c8: 1,
            hmap4: 1,
            p: [0; 256],
            h: [0; 256],
            comp: Vec::new(),
            squasht: vec![0u16; 4096],
            stretcht: vec![0i16; 32768],
            init_tables: false,
        }
    }

    fn ensure_tables(&mut self) {
        if self.init_tables { return; }
        self.init_tables = true;

        // squasht: copy 1344 entries starting at offset 1376; clamp
        // upper range to 32767.
        for i in 0..1376 { self.squasht[i] = 0; }
        for i in 0..1344 { self.squasht[1376 + i] = SSQUASHT[i]; }
        for i in 2720..4096 { self.squasht[i] = 32767; }

        // stretcht: derived from STDT.
        let mut k: usize = 16384;
        for i in 0..712 {
            for _ in 0..STDT[i] {
                if k >= 32768 { break; }
                self.stretcht[k] = i as i16;
                k += 1;
            }
        }
        for i in 0..16384 {
            self.stretcht[i] = -self.stretcht[32767 - i];
        }
    }

    fn squash(&self, x: i32) -> i32 {
        debug_assert!(self.init_tables);
        debug_assert!(x >= -2048 && x <= 2047);
        self.squasht[(x + 2048) as usize] as i32
    }

    fn stretch(&self, x: i32) -> i32 {
        debug_assert!(self.init_tables);
        debug_assert!(x >= 0 && x <= 32767);
        self.stretcht[x as usize] as i32
    }

    fn clamp2k(x: i32) -> i32 {
        x.clamp(-2048, 2047)
    }
    fn clamp512k(x: i32) -> i32 {
        x.clamp(-(1 << 19), (1 << 19) - 1)
    }

    /// Initialise model from the COMP section of a header.
    /// `cp_start` is the offset in `header` where COMP begins
    /// (always 7 in the libzpaq layout).
    pub fn init(&mut self, vm: &mut ZpaqlVm) -> Result<(), PredictorError> {
        self.ensure_tables();
        vm.init_hcomp();

        for i in 0..256 { self.h[i] = 0; self.p[i] = 0; }

        let n = vm.header[6] as usize;
        self.comp = Vec::with_capacity(n);
        for _ in 0..n { self.comp.push(Component::new()); }

        let mut cp = 7usize;
        for i in 0..n {
            let ty = vm.header[cp];
            match ty {
                CONS => {
                    self.p[i] = (vm.header[cp + 1] as i32 - 128) * 4;
                }
                CM => {
                    if vm.header[cp + 1] > 32 { return Err(PredictorError::OutOfRange); }
                    let sb = vm.header[cp + 1] as usize;
                    let size = 1usize << sb;
                    let comp = &mut self.comp[i];
                    comp.cm = vec![0x8000_0000u32; size];
                    comp.cm_mask = size - 1;
                    comp.limit = vm.header[cp + 2] as usize * 4;
                }
                ICM => {
                    if vm.header[cp + 1] > 26 { return Err(PredictorError::OutOfRange); }
                    let sb = vm.header[cp + 1] as usize;
                    let size = 1usize << sb;
                    let comp = &mut self.comp[i];
                    comp.limit = 1023;
                    comp.cm = vec![0u32; 256];
                    comp.cm_mask = 255;
                    comp.ht = vec![0u8; 64 * size];
                    comp.ht_mask = comp.ht.len() - 1;
                    for j in 0..256 {
                        comp.cm[j] = cminit(j);
                    }
                }
                MATCH => {
                    if vm.header[cp + 1] > 32 || vm.header[cp + 2] > 32 {
                        return Err(PredictorError::OutOfRange);
                    }
                    let sb1 = vm.header[cp + 1] as usize;
                    let sb2 = vm.header[cp + 2] as usize;
                    let sz1 = 1usize << sb1;
                    let sz2 = 1usize << sb2;
                    let comp = &mut self.comp[i];
                    comp.cm = vec![0u32; sz1];
                    comp.cm_mask = sz1 - 1;
                    comp.ht = vec![0u8; sz2];
                    comp.ht_mask = sz2 - 1;
                    comp.ht[0] = 1;
                }
                AVG => {
                    if vm.header[cp + 1] as usize >= i || vm.header[cp + 2] as usize >= i {
                        return Err(PredictorError::OutOfRange);
                    }
                }
                MIX2 => {
                    if vm.header[cp + 1] > 32 { return Err(PredictorError::OutOfRange); }
                    let sb = vm.header[cp + 1] as usize;
                    let size = 1usize << sb;
                    if vm.header[cp + 2] as usize >= i || vm.header[cp + 3] as usize >= i {
                        return Err(PredictorError::OutOfRange);
                    }
                    let comp = &mut self.comp[i];
                    comp.c = size;
                    comp.a16 = vec![32768u16; size];
                    comp.a16_mask = size - 1;
                }
                MIX => {
                    if vm.header[cp + 1] > 32 { return Err(PredictorError::OutOfRange); }
                    let sb = vm.header[cp + 1] as usize;
                    let size = 1usize << sb;
                    let m = vm.header[cp + 3] as usize;
                    if vm.header[cp + 2] as usize >= i { return Err(PredictorError::OutOfRange); }
                    if m < 1 || m > i.saturating_sub(vm.header[cp + 2] as usize) {
                        return Err(PredictorError::OutOfRange);
                    }
                    let comp = &mut self.comp[i];
                    comp.c = size;
                    let total = m * size;
                    let init = (65536 / m as u32) as u32;
                    comp.cm = vec![init; total];
                    comp.cm_mask = total - 1;
                }
                ISSE => {
                    if vm.header[cp + 1] > 32 { return Err(PredictorError::OutOfRange); }
                    if vm.header[cp + 2] as usize >= i { return Err(PredictorError::OutOfRange); }
                    let sb = vm.header[cp + 1] as usize;
                    // Stretch lookups before borrowing comp.
                    let mut cm = vec![0u32; 512];
                    for j in 0..256 {
                        cm[j * 2] = 1 << 15;
                        let s = self.stretch(((cminit(j) >> 8) as i32) & 0x7FFF) * 1024;
                        cm[j * 2 + 1] = Self::clamp512k(s) as u32;
                    }
                    let comp = &mut self.comp[i];
                    comp.ht = vec![0u8; 64 * (1 << sb)];
                    comp.ht_mask = comp.ht.len() - 1;
                    comp.cm = cm;
                    comp.cm_mask = 511;
                }
                SSE => {
                    if vm.header[cp + 1] > 32 { return Err(PredictorError::OutOfRange); }
                    if vm.header[cp + 2] as usize >= i { return Err(PredictorError::OutOfRange); }
                    if vm.header[cp + 3] > vm.header[cp + 4].wrapping_mul(4) {
                        return Err(PredictorError::OutOfRange);
                    }
                    let sb = vm.header[cp + 1] as usize;
                    let size = 32 * (1 << sb);
                    let limit = vm.header[cp + 4] as usize * 4;
                    let start = vm.header[cp + 3] as u32;
                    // Materialise the SSE init table outside the
                    // mutable borrow of self.comp[i].
                    let mut init = vec![0u32; size];
                    for j in 0..size {
                        let q = (j & 31) as i32 * 64 - 992;
                        init[j] = ((self.squash(q) as u32) << 17) | start;
                    }
                    let comp = &mut self.comp[i];
                    comp.cm = init;
                    comp.cm_mask = size - 1;
                    comp.limit = limit;
                }
                _ => return Err(PredictorError::UnknownComponent(ty)),
            }
            cp += COMPSIZE[ty as usize] as usize;
        }
        self.c8 = 1;
        self.hmap4 = 1;
        Ok(())
    }

    /// Returns whether the model has any components (modeled vs
    /// stored).
    pub fn is_modeled(&self) -> bool { !self.comp.is_empty() }

    /// `predict()` — return P(next bit == 1) in 0..=4095.
    pub fn predict(&mut self, vm: &mut ZpaqlVm) -> i32 {
        let n = self.comp.len();
        let mut cp = 7usize;
        for i in 0..n {
            let ty = vm.header[cp];
            match ty {
                CONS => {}
                CM => {
                    let h = self.h[i];
                    let cxt = (h ^ self.hmap4 as u32) as usize;
                    self.comp[i].cxt = cxt;
                    let v = self.comp[i].cm_get(cxt);
                    // libzpaq: `stretch(cr.cm(cr.cxt) >> 17)` — unsigned
                    // shift, value bounded to [0, 32767] by the model
                    // invariants. Clamp defensively rather than panicking
                    // if a bug causes drift.
                    self.p[i] = self.stretch(((v >> 17) as i32).min(32767));
                }
                ICM => {
                    if self.c8 == 1 || (self.c8 & 0xF0) == 16 {
                        let cxt = self.h[i].wrapping_add(16 * self.c8) as usize;
                        let sb = vm.header[cp + 1] as usize + 2;
                        let cidx = find(&mut self.comp[i].ht, sb, cxt);
                        self.comp[i].c = cidx;
                    }
                    let cidx = self.comp[i].c + (self.hmap4 as usize & 15);
                    let cxt = self.comp[i].ht[cidx & self.comp[i].ht_mask] as usize;
                    self.comp[i].cxt = cxt;
                    let v = self.comp[i].cm_get(cxt);
                    self.p[i] = self.stretch(((v >> 8) as i32).min(32767));
                }
                MATCH => {
                    let comp = &mut self.comp[i];
                    if comp.a == 0 {
                        self.p[i] = 0;
                    } else {
                        let off = (comp.limit as i32 - comp.b as i32) as usize;
                        let bit = (comp.ht[off & comp.ht_mask] >> (7 - comp.cxt as u8)) & 1;
                        comp.c = bit as usize;
                        let dt = SDT2K[comp.a as usize];
                        let s = (dt * (bit as i32 * -2 + 1)) & 32767;
                        self.p[i] = self.stretch(s);
                    }
                }
                AVG => {
                    let j = vm.header[cp + 1] as usize;
                    let k = vm.header[cp + 2] as usize;
                    let wt = vm.header[cp + 3] as i32;
                    self.p[i] = (self.p[j] * wt + self.p[k] * (256 - wt)) >> 8;
                }
                MIX2 => {
                    let mask = vm.header[cp + 5] as u32;
                    let cxt = (self.h[i].wrapping_add(self.c8 & mask) as usize)
                        & (self.comp[i].c - 1);
                    self.comp[i].cxt = cxt;
                    let w = self.comp[i].a16[cxt & self.comp[i].a16_mask] as i32;
                    let j = vm.header[cp + 2] as usize;
                    let k = vm.header[cp + 3] as usize;
                    self.p[i] = (w * self.p[j] + (65536 - w) * self.p[k]) >> 16;
                }
                MIX => {
                    let m = vm.header[cp + 3] as usize;
                    let mask = vm.header[cp + 5] as u32;
                    let j = vm.header[cp + 2] as usize;
                    let cxt0 = self.h[i].wrapping_add(self.c8 & mask) as usize;
                    // Upstream layout: cm has m*c entries; cxt = (h&(c-1))*m
                    // is a *row pointer*, so cxt+jj is bounded by cm.size().
                    // cm.size() may not be a power of two (m can be e.g. 7),
                    // so do NOT AND with cm_mask here — just use the raw index.
                    let cxt = (cxt0 & (self.comp[i].c - 1)) * m;
                    self.comp[i].cxt = cxt;
                    let mut sum: i64 = 0;
                    for jj in 0..m {
                        let w = self.comp[i].cm[cxt + jj] as i32;
                        sum += (w >> 8) as i64 * self.p[j + jj] as i64;
                    }
                    self.p[i] = Self::clamp2k((sum >> 8) as i32);
                }
                ISSE => {
                    if self.c8 == 1 || (self.c8 & 0xF0) == 16 {
                        let cxt = self.h[i].wrapping_add(16 * self.c8) as usize;
                        let sb = vm.header[cp + 1] as usize + 2;
                        let cidx = find(&mut self.comp[i].ht, sb, cxt);
                        self.comp[i].c = cidx;
                    }
                    let cidx = self.comp[i].c + (self.hmap4 as usize & 15);
                    let cxt = self.comp[i].ht[cidx & self.comp[i].ht_mask] as usize;
                    self.comp[i].cxt = cxt;
                    let w0 = self.comp[i].cm[(cxt * 2) & self.comp[i].cm_mask] as i32;
                    let w1 = self.comp[i].cm[(cxt * 2 + 1) & self.comp[i].cm_mask] as i32;
                    let j = vm.header[cp + 2] as usize;
                    self.p[i] = Self::clamp2k(((w0 * self.p[j] + w1 * 64) >> 16) as i32);
                }
                SSE => {
                    let j = vm.header[cp + 2] as usize;
                    let cxt0 = (self.h[i].wrapping_add(self.c8) as usize) * 32;
                    let mut pq = self.p[j] + 992;
                    if pq < 0 { pq = 0; }
                    if pq > 1983 { pq = 1983; }
                    let wt = pq & 63;
                    let pq_hi = pq >> 6;
                    self.comp[i].cxt = cxt0 + pq_hi as usize;
                    let cm0 = self.comp[i].cm_get(self.comp[i].cxt);
                    let cm1 = self.comp[i].cm_get(self.comp[i].cxt + 1);
                    let s = (((cm0 >> 10) as i32) * (64 - wt)
                        + ((cm1 >> 10) as i32) * wt) >> 13;
                    self.p[i] = self.stretch(s & 0x7FFF);
                    self.comp[i].cxt += (wt >> 5) as usize;
                }
                _ => panic!("unknown component"),
            }
            cp += COMPSIZE[ty as usize] as usize;
        }
        self.squash(self.p[n - 1])
    }

    /// `update(y)` — train the model on the just-decoded bit `y`.
    pub fn update(&mut self, y: u32, vm: &mut ZpaqlVm) {
        let n = self.comp.len();
        let mut cp = 7usize;
        for i in 0..n {
            let ty = vm.header[cp];
            match ty {
                CONS => {}
                CM => {
                    train(&mut self.comp[i], y);
                }
                ICM => {
                    let cidx = self.comp[i].c + (self.hmap4 as usize & 15);
                    let mask = self.comp[i].ht_mask;
                    let bh = self.comp[i].ht[cidx & mask];
                    let next = SNS[(bh as usize) * 4 + y as usize];
                    self.comp[i].ht[cidx & mask] = next;
                    let cxt = self.comp[i].cxt;
                    let cm_idx = cxt & self.comp[i].cm_mask;
                    let mut pn = self.comp[i].cm[cm_idx];
                    // Bug fix: libzpaq does `pn >> 8` as an *unsigned*
                    // shift and then casts to int. `pn as i32 >> 8`
                    // would do an arithmetic shift on the signed
                    // re-interpretation, which diverges for pn >= 2^31.
                    let delta = ((y as i32 * 32767) - ((pn >> 8) as i32)) >> 2;
                    pn = pn.wrapping_add(delta as u32);
                    self.comp[i].cm[cm_idx] = pn;
                }
                MATCH => {
                    let bufbits = vm.header[cp + 2] as usize;
                    let comp = &mut self.comp[i];
                    if comp.c as u32 != y { comp.a = 0; }
                    {
                        let p = comp.ht_at(comp.limit);
                        *p = p.wrapping_add(p.wrapping_add(y as u8));
                    }
                    comp.cxt += 1;
                    if comp.cxt == 8 {
                        comp.cxt = 0;
                        comp.limit += 1;
                        comp.limit &= (1usize << bufbits) - 1;
                        if comp.a == 0 {
                            let h_i = self.h[i] as usize;
                            comp.b = (comp.limit as u32).wrapping_sub(comp.cm_get(h_i));
                            if (comp.b as usize) & comp.ht_mask != 0 {
                                while comp.a < 255 {
                                    let l = comp.limit as i64 - comp.a as i64 - 1;
                                    let r = l - comp.b as i64;
                                    let lb = comp.ht[(l as usize) & comp.ht_mask];
                                    let rb = comp.ht[(r as usize) & comp.ht_mask];
                                    if lb != rb { break; }
                                    comp.a += 1;
                                }
                            }
                        } else if comp.a < 255 {
                            comp.a += 1;
                        }
                        let h_i = self.h[i] as usize;
                        let lim = comp.limit as u32;
                        *comp.cm_at(h_i) = lim;
                    }
                }
                AVG => {}
                MIX2 => {
                    let j = vm.header[cp + 2] as usize;
                    let k = vm.header[cp + 3] as usize;
                    let rate = vm.header[cp + 4] as i32;
                    let err = ((y as i32 * 32767) - self.squash(self.p[i])) * rate >> 5;
                    let cxt = self.comp[i].cxt;
                    let mask = self.comp[i].a16_mask;
                    let mut w = self.comp[i].a16[cxt & mask] as i32;
                    w += (err * (self.p[j] - self.p[k]) + (1 << 12)) >> 13;
                    if w < 0 { w = 0; } else if w > 65535 { w = 65535; }
                    self.comp[i].a16[cxt & mask] = w as u16;
                }
                MIX => {
                    let j = vm.header[cp + 2] as usize;
                    let m = vm.header[cp + 3] as usize;
                    let rate = vm.header[cp + 4] as i32;
                    let err = ((y as i32 * 32767) - self.squash(self.p[i])) * rate >> 4;
                    let cxt = self.comp[i].cxt;
                    // Upstream: cm has m*c entries; cxt+m <= cm.size() by
                    // construction. Don't mask — cm.size() isn't always a
                    // power of two (e.g. m=7 ⇒ total = 7*256 = 1792).
                    for jj in 0..m {
                        let mut w = self.comp[i].cm[cxt + jj] as i32;
                        w = Self::clamp512k(w + ((err * self.p[j + jj] + (1 << 12)) >> 13));
                        self.comp[i].cm[cxt + jj] = w as u32;
                    }
                }
                ISSE => {
                    let j = vm.header[cp + 2] as usize;
                    let err = (y as i32 * 32767) - self.squash(self.p[i]);
                    let cxt = self.comp[i].cxt;
                    let mask = self.comp[i].cm_mask;
                    let i0 = (cxt * 2) & mask;
                    let i1 = (cxt * 2 + 1) & mask;
                    let mut w0 = self.comp[i].cm[i0] as i32;
                    let mut w1 = self.comp[i].cm[i1] as i32;
                    w0 = Self::clamp512k(w0 + ((err * self.p[j] + (1 << 12)) >> 13));
                    w1 = Self::clamp512k(w1 + ((err + 16) >> 5));
                    self.comp[i].cm[i0] = w0 as u32;
                    self.comp[i].cm[i1] = w1 as u32;
                    let cidx = self.comp[i].c + (self.hmap4 as usize & 15);
                    let ht_mask = self.comp[i].ht_mask;
                    let cur = self.comp[i].ht[cidx & ht_mask];
                    self.comp[i].ht[cidx & ht_mask] =
                        SNS[(cur as usize) * 4 + y as usize];
                }
                SSE => {
                    train(&mut self.comp[i], y);
                }
                _ => {}
            }
            cp += COMPSIZE[ty as usize] as usize;
        }

        // Save bit y in c8, hmap4. Every 8 bits, run HCOMP and refresh
        // h[].
        self.c8 = self.c8 + self.c8 + y;
        if self.c8 >= 256 {
            let _ = vm.run::<crate::io::VecWriter>(self.c8 - 256, None, None);
            self.hmap4 = 1;
            self.c8 = 1;
            for i in 0..n { self.h[i] = vm.get_h(i as u32); }
        } else if self.c8 >= 16 && self.c8 < 32 {
            self.hmap4 = ((self.hmap4 & 0xF) << 5) | ((y as i32) << 4) | 1;
        } else {
            self.hmap4 = (self.hmap4 & 0x1F0)
                | (((self.hmap4 & 0xF) * 2 + y as i32) & 0xF);
        }
    }
}

impl Default for Predictor {
    fn default() -> Self { Self::new() }
}

/// `train(cr, y)` — CM/SSE training (mirrors libzpaq's
/// `Predictor::train`).
fn train(cr: &mut Component, y: u32) {
    let cxt = cr.cxt;
    let mask = cr.cm_mask;
    let pn = &mut cr.cm[cxt & mask];
    let count = (*pn) & 0x3FF;
    let error = (y as i32 * 32767) - ((*pn >> 17) as i32);
    let dt = SDT[count.min(1023) as usize];
    let delta = (error * dt) & -1024;
    *pn = pn.wrapping_add(delta as u32);
    if (count as usize) < cr.limit { *pn = pn.wrapping_add(1); }
}

/// `Predictor::find` — locate a 16-byte row in `ht` keyed by `cxt`,
/// or evict the LRU candidate among the three nearest rows.
fn find(ht: &mut [u8], sizebits: usize, cxt: usize) -> usize {
    let chk = ((cxt >> sizebits) & 255) as u8;
    let mask = ht.len() - 16;
    let h0 = (cxt * 16) & mask;
    if ht[h0] == chk { return h0; }
    let h1 = h0 ^ 16;
    if ht[h1] == chk { return h1; }
    let h2 = h0 ^ 32;
    if ht[h2] == chk { return h2; }
    // Evict the row with the smallest priority (ht[hN+1]).
    // Upstream uses `<` not `<=` in the second comparison. The
    // distinction matters when h1 and h2 have equal priority — they
    // get different rows back. Mismatching this against libzpaq's
    // archives makes the predictor diverge from the encoder.
    let p0 = ht[h0 + 1];
    let p1 = ht[h1 + 1];
    let p2 = ht[h2 + 1];
    let target = if p0 <= p1 && p0 <= p2 { h0 }
                 else if p1 < p2 { h1 }
                 else { h2 };
    for k in 0..16 { ht[target + k] = 0; }
    ht[target] = chk;
    target
}

/// StateTable.cminit: `((sns[s*4+3]*2 + 1) << 22) / (sns[s*4+2] +
/// sns[s*4+3] + 1)`.
fn cminit(state: usize) -> u32 {
    let n0 = SNS[state * 4 + 2] as u32;
    let n1 = SNS[state * 4 + 3] as u32;
    ((n1 * 2 + 1) << 22) / (n0 + n1 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_stretch_roundtrip_inits() {
        let mut p = Predictor::new();
        p.ensure_tables();
        // squash(stretch(s)) ≈ s for s in middle range.
        for s in [4096, 8192, 16384, 24576, 28672] {
            let st = p.stretch(s);
            let sq = p.squash(st);
            // Allow generous slack — the tables are coarse-grained.
            assert!((sq - s).abs() < 2048,
                    "squash(stretch({})) = {} (stretch={})", s, sq, st);
        }
    }

    #[test]
    fn cminit_matches_upstream_anchor_values() {
        // Anchor values verified against upstream `Predictor::init`'s
        // `cminit` formula. State 0: sns[0..4] = [1, 2, 0, 0] →
        // ((0*2 + 1) << 22) / (0 + 0 + 1) = 1 << 22.
        assert_eq!(cminit(0), 1 << 22);
        // State 1: sns[4..8] = [3, 5, 1, 0] → ((0*2 + 1) << 22) / 2.
        assert_eq!(cminit(1), (1 << 22) / 2);
        // State 2: sns[8..12] = [4, 6, 0, 1] → ((1*2 + 1) << 22) / 2.
        assert_eq!(cminit(2), (3 << 22) / 2);
    }
}
