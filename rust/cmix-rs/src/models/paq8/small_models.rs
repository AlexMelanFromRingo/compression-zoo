//! Small paq8 sub-models — paq8.cpp:3845-4613.
//!
//! Each upstream model is a free function with `static` locals; here
//! they become structs that hold what upstream put in `static`,
//! taking the shared paq8 globals through [`Paq8State`].
//!
//! Ported in this module:
//! * [`PicModel`]      — paq8.cpp:3845-3865 (1-bit-image surrounding pixels)
//! * [`DistanceModel`] — paq8.cpp:4599-4613 (distance to last 0x00/0x20/NL)
//! * [`SparseModel`]   — paq8.cpp:4505-4537
//! * [`SparseModel1`]  — paq8.cpp:4540-4597
//! * [`NestModel`]     — paq8.cpp:4108-4182 (bracket/quote nesting)
//!
//! Larger entangled models (wordModel, recordModel, recordModel1,
//! indirectModel, linearPredictionModel) are tracked separately;
//! they share the same `&mut Paq8State` calling convention.

#![allow(dead_code)]

use super::apm::StateMap;
use super::context_map::{
    ContextMap, IndirectMap, SmallStationaryContextMap, StationaryMap,
};
use super::mixer::Mixer;
use super::state::Paq8State;
use super::substrate::{finalize64, hash2, hash3, hash4, ilog2, nex};

// =============================================================
// PicModel — paq8.cpp:3845-3865.
// =============================================================

pub struct PicModel {
    r0: u32, r1: u32, r2: u32, r3: u32,
    t:   Vec<u8>,
    cxt: [usize; 3],
    sm:  Vec<StateMap>,
}

impl PicModel {
    pub fn new(_dt: [i32; 1024]) -> Self {
        Self {
            r0: 0, r1: 0, r2: 0, r3: 0,
            t:   vec![0u8; 0x10200],
            cxt: [0; 3],
            sm:  (0..3).map(|_| StateMap::new()).collect(),
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        let y = s.y;
        let bpos = s.bpos;
        for i in 0..3 {
            self.t[self.cxt[i]] = nex(self.t[self.cxt[i]], y as usize);
        }
        self.r0 = self.r0.wrapping_mul(2).wrapping_add(y as u32);
        self.r1 = self.r1.wrapping_mul(2)
            .wrapping_add(((s.buf.at(215) >> (7 - bpos)) & 1) as u32);
        self.r2 = self.r2.wrapping_mul(2)
            .wrapping_add(((s.buf.at(431) >> (7 - bpos)) & 1) as u32);
        self.r3 = self.r3.wrapping_mul(2)
            .wrapping_add(((s.buf.at(647) >> (7 - bpos)) & 1) as u32);
        self.cxt[0] = ((self.r0 & 0x7)
            | ((self.r1 >> 4) & 0x38)
            | ((self.r2 >> 3) & 0xc0)) as usize;
        self.cxt[1] = (0x100
            + ((self.r0 & 1)
                | ((self.r1 >> 4) & 0x3e)
                | ((self.r2 >> 2) & 0x40)
                | ((self.r3 >> 1) & 0x80))) as usize;
        self.cxt[2] = (0x200
            + ((self.r0 & 0x3f)
                ^ (self.r1 & 0x3ffe)
                ^ ((self.r2 << 2) & 0x7f00)
                ^ ((self.r3 << 5) & 0xf800))) as usize;
        for i in 0..3 {
            let st = self.t[self.cxt[i]];
            let p = self.sm[i].p(st as u32, y);
            m.add(s.stretch.get(p) as i16);
        }
    }
}

// =============================================================
// DistanceModel — paq8.cpp:4599-4613.
// =============================================================

pub struct DistanceModel {
    cm:    ContextMap,
    pos00: u32, pos20: u32, posnl: u32,
}

impl DistanceModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self { cm: ContextMap::new(mem, 3, dt), pos00: 0, pos20: 0, posnl: 0 }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let c = s.c4 & 0xff;
            let pos = s.buf.pos;
            if c == 0x00 { self.pos00 = pos; }
            if c == 0x20 { self.pos20 = pos; }
            if c == 0xff || c == b'\r' as u32 || c == b'\n' as u32 {
                self.posnl = pos;
            }
            self.cm.set(hash2(1,
                (255.min(pos - self.pos00) | (c << 8)) as u64));
            self.cm.set(hash2(2,
                (255.min(pos - self.pos20) | (c << 8)) as u64));
            self.cm.set(hash2(3,
                (255.min(pos - self.posnl) | (c << 8)) as u64));
        }
        cm_mix(&mut self.cm, s, m);
    }
}

// =============================================================
// SparseModel — paq8.cpp:4505-4537.
// =============================================================

pub struct SparseModel {
    cm: ContextMap,
}

impl SparseModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self { cm: ContextMap::new(mem * 2, 40 + 2, dt) }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer,
                seenbefore: i32, howmany: i32) {
        if s.bpos == 0 {
            let b = |k: u32| s.buf.at(k) as u64;
            let c4 = s.c4 as u64;
            let f4 = s.f4 as u64;
            let mut i: u64 = 0;
            macro_rules! cs { ($v:expr) => {{ i += 1; self.cm.set(hash2(i, $v)); }} }
            cs!(seenbefore as u64);
            cs!(howmany as u64);
            cs!(b(1) | (b(5) << 8));
            cs!(b(1) | (b(6) << 8));
            cs!(b(3) | (b(6) << 8));
            cs!(b(4) | (b(8) << 8));
            cs!(b(1) | (b(3) << 8) | (b(5) << 16));
            cs!(b(2) | (b(4) << 8) | (b(6) << 16));
            cs!(c4 & 0x00f0f0ff);
            cs!(c4 & 0x00ff00ff);
            cs!(c4 & 0xff0000ff);
            cs!(c4 & 0x00f8f8f8);
            cs!(c4 & 0xf8f8f8f8);
            cs!(f4 & 0x00000fff);
            cs!(f4);
            cs!(c4 & 0x00e0e0e0);
            cs!(c4 & 0xe0e0e0e0);
            cs!(c4 & 0x810000c1);
            cs!(c4 & 0xC3CCC38C);
            cs!(c4 & 0x0081CC81);
            cs!(c4 & 0x00c10081);
            for j in 1u32..8 {
                cs!(seenbefore as u64 | (b(j) << 8));
                cs!((b(j + 2) << 8) | b(j + 1));
                cs!((b(j + 3) << 8) | b(j + 1));
            }
        }
        cm_mix(&mut self.cm, s, m);
    }
}

// =============================================================
// SparseModel1 — paq8.cpp:4540-4597.
// =============================================================

pub struct SparseModel1 {
    cm:   ContextMap,
    scm1: SmallStationaryContextMap,
    scm2: SmallStationaryContextMap,
    scm3: SmallStationaryContextMap,
    scm4: SmallStationaryContextMap,
    scm5: SmallStationaryContextMap,
    scm6: SmallStationaryContextMap,
    scma: SmallStationaryContextMap,
}

impl SparseModel1 {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            cm:   ContextMap::new(mem * 4, 31, dt),
            scm1: SmallStationaryContextMap::new(7, 8),
            scm2: SmallStationaryContextMap::new(8, 8),
            scm3: SmallStationaryContextMap::new(4, 8),
            scm4: SmallStationaryContextMap::new(6, 8),
            scm5: SmallStationaryContextMap::new(4, 8),
            scm6: SmallStationaryContextMap::new(4, 8),
            scma: SmallStationaryContextMap::new(7, 8),
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer,
                seenbefore: i32, howmany: i32) {
        if s.bpos == 0 {
            self.scm5.set(seenbefore as u32);
            self.scm6.set(howmany as u32);
            let b1 = s.buf.at(1) as u32;
            let mut h: u32 = s.x4 << 6;
            self.cm.set((b1 + (h & 0xffffff00)) as u64);
            self.cm.set((b1 + (h & 0x00ffff00)) as u64);
            self.cm.set((b1 + (h & 0x0000ff00)) as u64);
            let mut d: u32 = s.c4 & 0xffff;
            h <<= 6;
            self.cm.set((d + (h & 0xffff0000)) as u64);
            self.cm.set((d + (h & 0x00ff0000)) as u64);
            h <<= 6;
            d = s.c4 & 0xffffff;
            self.cm.set((d + (h & 0xff000000)) as u64);
            for k in 1u32..5 {
                self.cm.set((seenbefore as u32 | (s.buf.at(k) as u32) << 8) as u64);
                self.cm.set((((s.buf.at(k + 3) as u32) << 8)
                    | s.buf.at(k + 1) as u32) as u64);
            }
            self.cm.set((s.words & 0x7fff) as u64);
            self.cm.set((s.words & 0xff) as u64);
            self.cm.set((s.words & 0x1ffff) as u64);
            self.cm.set((s.f4 & 0x000fffff) as u64);
            self.cm.set((s.tt & 0x00000fff) as u64);
            h = s.w4 << 6;
            self.cm.set((b1 + (h & 0xffffff00)) as u64);
            self.cm.set((b1 + (h & 0x00ffff00)) as u64);
            self.cm.set((b1 + (h & 0x0000ff00)) as u64);
            d = s.c4 & 0xffff;
            h <<= 6;
            self.cm.set((d + (h & 0xffff0000)) as u64);
            self.cm.set((d + (h & 0x00ff0000)) as u64);
            h <<= 6;
            d = s.c4 & 0xffffff;
            self.cm.set((d + (h & 0xff000000)) as u64);
            self.cm.set((s.w4 & 0xf0f0f0ff) as u64);
            self.cm.set(((s.w4 & 63) * 128 + (5 << 17)) as u64);
            self.cm.set((((s.f4 & 0xffff) << 11) | s.frstchar) as u64);
            self.cm.set((s.spafdo * 8 * ((s.w4 & 3 == 1) as u32)) as u64);
            self.scm1.set(s.words & 127);
            self.scm2.set((s.words & 12) * 16 + (s.w4 & 12) * 4
                + ((s.buf.at(1) as u32) >> 4));
            self.scm3.set(s.w4 & 15);
            self.scm4.set(s.spafdo * ((s.w4 & 3 == 1) as u32));
            self.scma.set(s.frstchar);
        }
        cm_mix(&mut self.cm, s, m);
        let y = s.y;
        for scm in [&mut self.scm1, &mut self.scm2, &mut self.scm3,
                    &mut self.scm4, &mut self.scm5, &mut self.scm6,
                    &mut self.scma] {
            scm.mix(m, y, 7, 1, 4, &s.squash, &s.stretch);
        }
    }
}

// =============================================================
// NestModel — paq8.cpp:4108-4182.
// =============================================================

pub struct NestModel {
    cm: ContextMap,
    ic: i32, bc: i32, pc: i32, qc: i32, lvc: i32,
    ac: i32, ec: i32, uc: i32,
    sense1: i32, sense2: i32, w: i32,
    vc: u32, wc: u32,
}

impl NestModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            cm: ContextMap::new(mem / 2, 12, dt),
            ic: 0, bc: 0, pc: 0, qc: 0, lvc: 0,
            ac: 0, ec: 0, uc: 0,
            sense1: 0, sense2: 0, w: 0,
            vc: 0, wc: 0,
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let c = (s.c4 & 0xff) as i32;
            let mut matched = 1;
            self.w *= ((self.vc & 7) > 0 && (self.vc & 7) < 3) as i32;
            if c & 0x80 != 0 {
                self.w = self.w.wrapping_mul(11 * 32).wrapping_add(c);
            }
            let lc = if (b'A' as i32..=b'Z' as i32).contains(&c) {
                c + b'a' as i32 - b'A' as i32
            } else { c };
            let vv;
            if lc == b'a' as i32 || lc == b'e' as i32 || lc == b'i' as i32
                || lc == b'o' as i32 || lc == b'u' as i32
            {
                vv = 1;
                self.w = self.w.wrapping_mul(997 * 8)
                    .wrapping_add(lc / 4 - 22);
            } else if (b'a' as i32..=b'z' as i32).contains(&lc) {
                vv = 2;
                self.w = self.w.wrapping_mul(271 * 32)
                    .wrapping_add(lc - 97);
            } else if lc == b' ' as i32 || lc == b'.' as i32
                || lc == b',' as i32 || lc == b'!' as i32
                || lc == b'?' as i32 || lc == b'\n' as i32
            {
                vv = 3;
            } else if (b'0' as i32..=b'9' as i32).contains(&lc) {
                vv = 4;
            } else if lc == b'y' as i32 {
                vv = 5;
            } else if lc == b'\'' as i32 {
                vv = 6;
            } else {
                vv = if c & 32 != 0 { 7 } else { 0 };
            }
            self.vc = (self.vc << 3) | vv as u32;
            if vv != self.lvc {
                self.wc = (self.wc << 3) | vv as u32;
                self.lvc = vv;
            }
            match c as u8 {
                b' ' => self.qc = 0,
                b'(' => self.ic += 31,
                b')' => self.ic -= 31,
                b'[' => self.ic += 11,
                b']' => self.ic -= 11,
                b'<' => { self.ic += 23; self.qc += 34; }
                b'>' => { self.ic -= 23; self.qc /= 5; }
                b':' => self.pc = 20,
                b'{' => self.ic += 17,
                b'}' => self.ic -= 17,
                b'|' => self.pc += 223,
                b'"' => self.pc += 0x40,
                b'\'' => {
                    self.pc += 0x42;
                    if c as u8 != ((s.c4 >> 8) & 0xff) as u8 {
                        self.sense2 ^= 1;
                    } else {
                        self.ac += 2 * self.sense2 - 1;
                    }
                }
                b'\n' => { self.pc = 0; self.qc = 0; }
                b'.' | b'!' | b'?' => self.pc = 0,
                b'#' => self.pc += 0x08,
                b'%' => self.pc += 0x76,
                b'$' => self.pc += 0x45,
                b'*' => self.pc += 0x35,
                b'-' => self.pc += 0x3,
                b'@' => self.pc += 0x72,
                b'&' => self.qc += 0x12,
                b';' => self.qc /= 3,
                b'\\' => self.pc += 0x29,
                b'/' => {
                    self.pc += 0x11;
                    if s.buf.size() > 1 && s.buf.at(1) == b'<' {
                        self.qc += 74;
                    }
                }
                b'=' => {
                    self.pc += 87;
                    if c as u8 != ((s.c4 >> 8) & 0xff) as u8 {
                        self.sense1 ^= 1;
                    } else {
                        self.ec += 2 * self.sense1 - 1;
                    }
                }
                _ => matched = 0,
            }
            if s.c4 == 0x266C743B {
                self.uc = 7.min(self.uc + 1);
            } else if s.c4 == 0x2667743B {
                self.uc -= (self.uc > 0) as i32;
            }
            if matched != 0 { self.bc = 0; } else { self.bc += 1; }
            if self.bc > 300 {
                self.bc = 0; self.ic = 0; self.pc = 0;
                self.qc = 0; self.uc = 0;
            }
            let vc = self.vc; let pc = self.pc; let ic = self.ic;
            let qc = self.qc; let bc = self.bc; let wc = self.wc;
            let f4 = s.f4;
            let mut i: u64 = 0;
            macro_rules! cs { ($v:expr) => {{ i += 1; self.cm.set(hash2(i, $v)); }} }
            cs!(hash5(
                if vv > 0 && vv < 3 { 0 } else { (lc | 0x100) as u64 },
                (ic & 0x3FF) as u64, (self.ec & 0x7) as u64,
                (self.ac & 0x7) as u64, self.uc as u64));
            cs!(hash3(ic as u64, self.w as u64,
                ilog2(bc as u32 + 1) as u64));
            cs!((3u32.wrapping_mul(vc)
                .wrapping_add(77u32.wrapping_mul(pc as u32))
                .wrapping_add(373u32.wrapping_mul(ic as u32))
                .wrapping_add(qc as u32) & 0xffff) as u64);
            cs!((31u32.wrapping_mul(vc)
                .wrapping_add(27u32.wrapping_mul(pc as u32))
                .wrapping_add(281u32.wrapping_mul(qc as u32)) & 0xffff) as u64);
            cs!((13u32.wrapping_mul(vc)
                .wrapping_add(271u32.wrapping_mul(ic as u32))
                .wrapping_add(qc as u32).wrapping_add(bc as u32)
                & 0xffff) as u64);
            cs!((17u32.wrapping_mul(pc as u32)
                .wrapping_add(7u32.wrapping_mul(ic as u32)) & 0xffff) as u64);
            cs!((13u32.wrapping_mul(vc).wrapping_add(ic as u32)
                & 0xffff) as u64);
            cs!(((vc / 3).wrapping_add(pc as u32) & 0xffff) as u64);
            cs!((7u32.wrapping_mul(wc).wrapping_add(qc as u32)
                & 0xffff) as u64);
            cs!(((vc & 0xffff) as u64) | (((f4 & 0xf) as u64) << 16));
            cs!((((3u32.wrapping_mul(pc as u32)) & 0xffff) as u64)
                | (((f4 & 0xf) as u64) << 16));
            cs!((((ic as u32) & 0xffff) as u64)
                | (((f4 & 0xf) as u64) << 16));
        }
        cm_mix(&mut self.cm, s, m);
    }
}

use super::substrate::{hash5, llog};
use super::util::{IndirectContext, Ols};

// =============================================================
// RecordModel1 — paq8.cpp:4436-4475.
// =============================================================

pub struct RecordModel1 {
    cm: ContextMap, cn: ContextMap, co: ContextMap,
    cp: ContextMap, cq: ContextMap,
    cpos1: [u32; 256],
    wpos1: Vec<u32>,
}

impl RecordModel1 {
    pub fn new(dt: [i32; 1024]) -> Self {
        Self {
            cm: ContextMap::new(32768, 2, dt),
            cn: ContextMap::new(32768 / 2, 5, dt),
            co: ContextMap::new(32768 * 4, 4, dt),
            cp: ContextMap::new(32768 * 2, 3, dt),
            cq: ContextMap::new(32768 * 2, 3, dt),
            cpos1: [0; 256],
            wpos1: vec![0u32; 0x10000],
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let w = (s.c4 & 0xffff) as usize;
            let c = w & 255;
            let d = w & 0xf0ff;
            let e = (s.c4 & 0xffffff) as u64;
            let pos = s.buf.pos;
            self.cm.set(((c << 8) as u32
                + (255.min(pos - self.cpos1[c]) / 4)) as u64);
            self.cm.set(((w << 9) as u32
                + (llog(&s.ilog, pos - self.wpos1[w]) >> 2)) as u64);
            self.cn.set(w as u64);
            self.cn.set((d << 8) as u64);
            self.cn.set((c << 16) as u64);
            self.cn.set((s.f4 & 0xfffff) as u64);
            let col = pos & 3;
            self.cn.set((col | (2 << 12)) as u64);
            self.co.set(c as u64);
            self.co.set((w << 8) as u64);
            self.co.set((s.w5 & 0x3ffff) as u64);
            self.co.set(e << 3);
            self.cp.set(d as u64);
            self.cp.set((c << 8) as u64);
            self.cp.set((w as u64) << 16);
            self.cq.set((w << 3) as u64);
            self.cq.set((c as u64) << 19);
            self.cq.set(e);
            self.cpos1[c] = pos;
            self.wpos1[w] = pos;
        }
        cm_mix(&mut self.cm, s, m);
        cm_mix(&mut self.cn, s, m);
        cm_mix(&mut self.co, s, m);
        cm_mix(&mut self.cq, s, m);
        cm_mix(&mut self.cp, s, m);
    }
}

// =============================================================
// LinearPredictionModel — paq8.cpp:4477-4503.
// =============================================================

pub struct LinearPredictionModel {
    s_map: Vec<SmallStationaryContextMap>,
    ols:   Vec<Ols>,
    prd:   [u8; 5],
}

impl LinearPredictionModel {
    pub fn new() -> Self {
        Self {
            s_map: (0..5).map(|_| SmallStationaryContextMap::new(11, 1)).collect(),
            // OLS<double, U8> {32, 4, 0.995} — hasZeroMean=true ⇒ sub=0.
            ols: (0..3).map(|_| Ols::new(32, 4, 0.995, 0.001, 0.0)).collect(),
            prd: [0; 5],
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let w = s.buf.at(1);
            let ww = s.buf.at(2);
            let www = s.buf.at(3);
            for o in self.ols.iter_mut() { o.update(w as f64); }
            for i in 1..=32u32 {
                self.ols[0].add(s.buf.at(i) as f64);
                self.ols[1].add(s.buf.at(i * 2 - 1) as f64);
                self.ols[2].add(s.buf.at(i * 2) as f64);
            }
            for i in 0..3 {
                let p = self.ols[i].predict().floor();
                self.prd[i] = p.clamp(0.0, 255.0) as u8;
            }
            self.prd[3] = clip(w as i32 * 2 - ww as i32);
            self.prd[4] = clip(w as i32 * 3 - ww as i32 * 3 + www as i32);
        }
        let b = (s.c0 << (8 - s.bpos)) as u8;
        let y = s.y;
        for i in 0..5 {
            let ctx = ((self.prd[i] as i32 - b as i32) * 8 + s.bpos) as u32;
            self.s_map[i].set(ctx);
            self.s_map[i].mix(m, y, 6, 1, 2, &s.squash, &s.stretch);
        }
    }
}

impl Default for LinearPredictionModel {
    fn default() -> Self { Self::new() }
}

#[inline]
fn clip(px: i32) -> u8 { px.clamp(0, 0xFF) as u8 }

// =============================================================
// RecordModel — paq8.cpp:4205-4434.
// =============================================================

#[derive(Default, Clone, Copy)]
struct DBase {
    version:       u8,
    n_records:     u32,
    record_length: u16,
    header_length: u16,
    start:         i32,
    end:           i32,
}

pub struct RecordModel {
    cpos1: [u32; 256], cpos2: [u32; 256],
    cpos3: [u32; 256], cpos4: [u32; 256],
    wpos1: Vec<u32>,
    rlen:  [i32; 3],
    rcount: [i32; 2],
    padding: u8,
    n: u8, nn: u8, nnn: u8, nnnn: u8, wx_nw: u8,
    prev_transition: i32, n_transition: i32,
    col: i32, mx_ctx: i32, x: i32,
    cm: ContextMap, cn: ContextMap, co: ContextMap, cp: ContextMap,
    maps:  Vec<StationaryMap>,
    s_map: Vec<SmallStationaryContextMap>,
    i_map: Vec<IndirectMap>,
    may_be_img24b: bool,
    dbase: DBase,
    i_ctx: Vec<IndirectContext>,
}

impl RecordModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        let maps = vec![
            StationaryMap::new(10, 8, 0), StationaryMap::new(10, 8, 0),
            StationaryMap::new(8, 8, 0),  StationaryMap::new(8, 8, 0),
            StationaryMap::new(8, 8, 0),  StationaryMap::new(11, 1, 0),
        ];
        let s_map = vec![
            SmallStationaryContextMap::new(11, 1),
            SmallStationaryContextMap::new(3, 1),
            SmallStationaryContextMap::new(19, 1),
        ];
        let i_map = vec![
            IndirectMap::new(8, 8, dt),
            IndirectMap::new(8, 8, dt),
            IndirectMap::new(8, 8, dt),
        ];
        let i_ctx = vec![
            IndirectContext::new(16, 8, 16),
            IndirectContext::new(16, 8, 16),
            IndirectContext::new(16, 8, 16),
            IndirectContext::new(20, 8, 16),
            IndirectContext::new(11, 1, 16),
        ];
        Self {
            cpos1: [0; 256], cpos2: [0; 256],
            cpos3: [0; 256], cpos4: [0; 256],
            wpos1: vec![0u32; 0x10000],
            rlen: [2, 3, 4], rcount: [0, 0],
            padding: 0,
            n: 0, nn: 0, nnn: 0, nnnn: 0, wx_nw: 0,
            prev_transition: 0, n_transition: 0,
            col: 0, mx_ctx: 0, x: 0,
            cm: ContextMap::new(32768, 3, dt),
            cn: ContextMap::new(32768 / 2, 3, dt),
            co: ContextMap::new(32768 * 2, 3, dt),
            cp: ContextMap::new(mem, 16, dt),
            maps, s_map, i_map,
            may_be_img24b: false,
            dbase: DBase::default(),
            i_ctx,
        }
    }

    /// `mix` — paq8.cpp:4205-4434. `is_text` ⇒ filetype is DEFAULT or
    /// TEXT (dBASE detection only runs for those).
    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, is_text: bool) {
        let dt_slice = s.dt;
        if s.bpos == 0 {
            let w = (s.c4 & 0xffff) as usize;
            let c = w & 255;
            let d = w >> 8;
            let pos = s.buf.pos;
            let blpos = s.blpos;

            // ModelStats.Record-driven override.
            if s.stats.record != 0
                && (s.stats.record >> 16) != self.rlen[0] as u32
            {
                self.rlen[0] = (s.stats.record >> 16) as i32;
                self.rcount = [0, 0];
            } else {
                // dBASE table detection.
                if blpos == 0
                    || (self.dbase.version > 0 && blpos >= self.dbase.end)
                {
                    self.dbase.version = 0;
                } else if self.dbase.version == 0 && is_text && blpos >= 31 {
                    let b32 = s.buf.at(32);
                    let b30 = s.buf.at(30);
                    let b29 = s.buf.at(29);
                    if ((b32 & 7) == 3 || (b32 & 7) == 4
                        || (b32 >> 4) == 3 || b32 == 0xF5)
                        && (b30 > 0 && b30 < 13)
                        && (b29 > 0 && b29 < 32)
                    {
                        let n_records = s.buf.at(28) as u32
                            | ((s.buf.at(27) as u32) << 8)
                            | ((s.buf.at(26) as u32) << 16)
                            | ((s.buf.at(25) as u32) << 24);
                        let mut header_length = s.buf.at(24) as u32
                            | ((s.buf.at(23) as u32) << 8);
                        let record_length = s.buf.at(22) as u32
                            | ((s.buf.at(21) as u32) << 8);
                        let hdr_ok = header_length > 32
                            && (((header_length - 32 - 1) % 32) == 0
                                || (header_length > 255 + 8 && {
                                    header_length -= 255 + 8;
                                    ((header_length - 32 - 1) % 32) == 0
                                }));
                        if n_records > 0 && n_records < 0xFFFFF
                            && hdr_ok
                            && record_length > 8
                            && s.buf.at(20) == 0 && s.buf.at(19) == 0
                            && s.buf.at(17) <= 1 && s.buf.at(16) <= 1
                        {
                            self.dbase.version =
                                if (b32 >> 4) == 3 { 3 } else { b32 & 7 };
                            self.dbase.n_records = n_records;
                            self.dbase.header_length = header_length as u16;
                            self.dbase.record_length = record_length as u16;
                            self.dbase.start =
                                blpos - 32 + header_length as i32;
                            self.dbase.end = self.dbase.start
                                + (n_records * record_length) as i32;
                            if self.dbase.version == 3 {
                                self.rlen[0] = 32;
                                self.rcount = [0, 0];
                            }
                        }
                    }
                } else if self.dbase.version > 0
                    && blpos == self.dbase.start
                {
                    self.rlen[0] = self.dbase.record_length as i32;
                    self.rcount = [0, 0];
                }

                // Run-length detection.
                let r = pos as i32 - self.cpos1[c] as i32;
                if r > 1
                    && r == self.cpos1[c] as i32 - self.cpos2[c] as i32
                    && r == self.cpos2[c] as i32 - self.cpos3[c] as i32
                    && (r > 32
                        || r == self.cpos3[c] as i32 - self.cpos4[c] as i32)
                    && (r > 10
                        || (c as u8 == s.buf.at((r * 5 + 1) as u32)
                            && c as u8 == s.buf.at((r * 6 + 1) as u32)))
                {
                    if r == self.rlen[1] {
                        self.rcount[0] += 1;
                    } else if r == self.rlen[2] {
                        self.rcount[1] += 1;
                    } else if self.rcount[0] > self.rcount[1] {
                        self.rlen[2] = r;
                        self.rcount[1] = 1;
                    } else {
                        self.rlen[1] = r;
                        self.rcount[0] = 1;
                    }
                }
                // Candidate-length check.
                for i in 0..2 {
                    if self.rcount[i]
                        > 0.max(12 - ilog2(self.rlen[i + 1] as u32) as i32)
                    {
                        if self.rlen[0] != self.rlen[i + 1] {
                            if self.may_be_img24b && self.rlen[i + 1] == 3 {
                                self.rcount[0] >>= 1;
                                self.rcount[1] >>= 1;
                                continue;
                            } else if self.rlen[i + 1] > self.rlen[0]
                                && self.rlen[i + 1] % self.rlen[0] == 0
                            {
                                if self.rlen[0] > 32
                                    && self.rlen[i + 1] == self.rlen[0] * 2
                                {
                                    self.rcount[0] >>= 1;
                                    self.rcount[1] >>= 1;
                                    continue;
                                }
                            }
                            self.rlen[0] = self.rlen[i + 1];
                            self.rcount[i] = 0;
                            self.may_be_img24b = self.rlen[0] > 30
                                && (self.rlen[0] % 3) == 0;
                            self.n_transition = 0;
                        } else {
                            self.rcount[i] >>= 2;
                        }
                        if (self.rlen[i + 1] << 4) > self.rlen[1 + (i ^ 1)] {
                            self.rcount[i ^ 1] = 0;
                        }
                    }
                }
            }

            self.col = (pos % self.rlen[0].max(1) as u32) as i32;
            self.x = 0x1F.min(self.col
                / 1.max(self.rlen[0] / 32));
            let rl0 = self.rlen[0].max(1) as u32;
            self.n = s.buf.at(rl0);
            self.nn = s.buf.at(rl0 * 2);
            self.nnn = s.buf.at(rl0 * 3);
            self.nnnn = s.buf.at(rl0 * 4);
            for k in 0..4 {
                self.i_ctx[k].add(c as u32);
            }
            self.i_ctx[0].set(((c << 8) | self.n as usize) as u32);
            self.i_ctx[1].set(
                ((s.buf.at(rl0 - 1) as u32) << 8) | self.n as u32);
            self.i_ctx[2].set(
                ((c as u32) << 8) | s.buf.at(rl0 - 1) as u32);
            self.i_ctx[3].set(finalize64(
                hash3(c as u64, self.n as u64,
                      s.buf.at(rl0 + 1) as u64), 20));

            if self.col == 0 { self.n_transition = 0; }
            let c4 = s.c4;
            if (((c4 >> 8) == (SPACE_U32 * 0x010101)) && (c as u32 != SPACE_U32))
                || ((c4 >> 8) == 0 && c != 0
                    && (self.padding != SPACE_U32 as u8
                        || pos as i32 - self.prev_transition > self.rlen[0]))
            {
                self.prev_transition = pos as i32;
                self.n_transition += (self.n_transition < 31) as i32;
                self.padding = d as u8;
            }

            let mut i: u64 = 0;
            macro_rules! cs { ($cm:expr, $v:expr) => {{ i += 1; $cm.set(hash2(i, $v)); }} }
            let w_idx = (s.c4 & 0xffff) as u32;
            cs!(self.cm, ((c << 8) as u32
                | 255.min(pos - self.cpos1[c]) >> 2) as u64);
            cs!(self.cm, ((w_idx << 9)
                | (llog(&s.ilog, pos - self.wpos1[w]) >> 2)) as u64);
            cs!(self.cm, (self.rlen[0] as u64)
                | ((self.n as u64) << 10) | ((self.nn as u64) << 18));
            cs!(self.cn, (w_idx as u64) | ((self.rlen[0] as u64) << 16));
            cs!(self.cn, (d as u64) | ((self.rlen[0] as u64) << 8));
            cs!(self.cn, (c as u64) | ((self.rlen[0] as u64) << 8));
            cs!(self.co, ((c << 8) as u64)
                | 255.min(pos - self.cpos1[c]) as u64);
            cs!(self.co, ((c as u64) << 17) | ((d as u64) << 9)
                | ((llog(&s.ilog, pos - self.wpos1[w]) >> 2) as u64));
            cs!(self.co, ((c << 8) as u64) | self.n as u64);
            cs!(self.cp, (self.rlen[0] as u64)
                | ((self.n as u64) << 10) | ((self.col as u64) << 18));
            cs!(self.cp, (self.rlen[0] as u64)
                | ((c as u64) << 10) | ((self.col as u64) << 18));
            cs!(self.cp, (self.col as u64)
                | ((self.rlen[0] as u64) << 12));
            if self.rlen[0] > 8 {
                cs!(self.cp, hash4(
                    (0xFF.min(self.rlen[0]) as u32)
                        .min(pos.wrapping_sub(self.prev_transition as u32)) as u64,
                    0x3FF.min(self.col as u32) as u64,
                    ((w_idx & 0xF0F0)
                        | ((w_idx == ((self.padding as u32) << 8)
                            | self.padding as u32) as u32)) as u64,
                    self.n_transition as u64));
                cs!(self.cp, hash3(w_idx as u64,
                    (s.buf.at(rl0 + 1) == self.padding
                        && self.n == self.padding) as u64,
                    (self.col / 1.max(self.rlen[0] / 32)) as u64));
            } else {
                self.cp.set(0); self.cp.set(0);
                i += 2;
            }
            cs!(self.cp, (self.n as u64)
                | ((self.nn as u64 & 0xF0) << 4)
                | ((self.nnn as u64 & 0xE0) << 7)
                | ((self.nnnn as u64 & 0xE0) << 10)
                | (((self.col / 1.max(self.rlen[0] / 16)) as u64) << 18));
            cs!(self.cp, (self.n as u64 & 0xF8)
                | ((self.nn as u64 & 0xF8) << 8)
                | ((self.col as u64) << 16));
            cs!(self.cp, hash2(self.n as u64, self.nn as u64));
            cs!(self.cp, hash2(self.col as u64, self.i_ctx[0].get() as u64));
            cs!(self.cp, hash2(self.col as u64, self.i_ctx[1].get() as u64));
            cs!(self.cp, hash3(self.col as u64,
                (self.i_ctx[0].get() & 0xFF) as u64,
                (self.i_ctx[1].get() & 0xFF) as u64));
            cs!(self.cp, self.i_ctx[2].get() as u64);
            cs!(self.cp, self.i_ctx[3].get() as u64);
            cs!(self.cp, hash2((self.i_ctx[1].get() & 0xFF) as u64,
                (self.i_ctx[3].get() & 0xFF) as u64));
            self.wx_nw = c as u8 ^ s.buf.at(rl0 + 1);
            cs!(self.cp, hash2(self.n as u64, self.wx_nw as u64));
            let exp_byte = if s.stats.r#match.length > 0 {
                s.stats.r#match.expected_byte as u64
            } else {
                0x100 | (self.i_ctx[1].get() as u8) as u64
            };
            cs!(self.cp, hash3(exp_byte, self.n as u64, self.wx_nw as u64));

            // Direct maps.
            let k = if self.may_be_img24b {
                ((self.col % 3) << 8) as u32
            } else { 0x300 };
            if self.may_be_img24b {
                self.maps[0].set_direct(clip(
                    ((c4 >> 16) & 0xff) as i32 + c as i32
                        - ((c4 >> 24) & 0xff) as i32) as u32 | k);
            } else {
                self.maps[0].set_direct(
                    clip(c as i32 * 2 - d as i32) as u32 | k);
            }
            self.maps[1].set_direct(clip(
                c as i32 + self.n as i32 - s.buf.at(rl0 + 1) as i32) as u32 | k);
            self.maps[2].set_direct(clip(
                self.n as i32 + self.nn as i32 - self.nnn as i32) as u32);
            self.maps[3].set_direct(clip(
                self.n as i32 * 2 - self.nn as i32) as u32);
            self.maps[4].set_direct(clip(
                self.n as i32 * 3 - self.nn as i32 * 3 + self.nnn as i32) as u32);
            self.i_map[0].set_direct(
                (self.n as i32 + self.nn as i32 - self.nnn as i32) as u32);
            self.i_map[1].set_direct(
                (self.n as i32 * 2 - self.nn as i32) as u32);
            self.i_map[2].set_direct((self.n as i32 * 3
                - self.nn as i32 * 3 + self.nnn as i32) as u32);

            self.cpos4[c] = self.cpos3[c];
            self.cpos3[c] = self.cpos2[c];
            self.cpos2[c] = self.cpos1[c];
            self.cpos1[c] = pos;
            self.wpos1[w] = pos;
            self.mx_ctx = if self.rlen[0] > 128 {
                0x7F.min(self.col / 1.max(self.rlen[0] / 128))
            } else {
                self.col
            };
            let _ = i;
        }

        let b = (s.c0 << (8 - s.bpos)) as u8;
        let ctx = (self.n as u32 ^ b as u32) | ((s.bpos as u32) << 8);
        let li = self.i_ctx.len() - 1;
        self.i_ctx[li].add(s.y as u32);
        self.i_ctx[li].set(ctx);
        let nm = self.maps.len() - 1;
        self.maps[nm].set_direct(ctx);
        self.s_map[0].set(ctx);
        self.s_map[1].set(self.i_ctx[li].get());
        self.s_map[2].set((ctx << 8) | self.wx_nw as u32);

        cm_mix(&mut self.cm, s, m);
        cm_mix(&mut self.cn, s, m);
        cm_mix(&mut self.co, s, m);
        cm_mix(&mut self.cp, s, m);
        let y = s.y;
        for mp in self.maps.iter_mut() {
            mp.mix(m, y, 1, 3, 1023, &dt_slice, &s.squash, &s.stretch);
        }
        for im in self.i_map.iter_mut() {
            im.mix(m, y, 1, 3, 255, &s.squash, &s.stretch);
        }
        self.s_map[0].mix(m, y, 6, 1, 3, &s.squash, &s.stretch);
        self.s_map[1].mix(m, y, 6, 1, 3, &s.squash, &s.stretch);
        self.s_map[2].mix(m, y, 5, 1, 2, &s.squash, &s.stretch);

        m.set((self.rlen[0] > 2) as u32
            * (((s.bpos as u32) << 7) | self.mx_ctx as u32), 1024);
        m.set(((self.n as u32 ^ b as u32) >> 4)
            | ((self.x as u32) << 4), 512);
        m.set(((s.grp0 as u32) << 5) | self.x as u32, 11 * 32);
        s.stats.record = ((0xFFFF.min(self.rlen[0]) as u32) << 16)
            | 0xFFFF.min(self.col as u32);
    }
}

const SPACE_U32: u32 = 0x20;

// =============================================================
// WordModel — paq8.cpp:3874-4106.
// =============================================================

use super::word::Word;
use super::stemmer::EnglishStemmer;
use super::substrate::combine64;

pub struct WordModel {
    word0: u64, word1: u64, word2: u64, word3: u64, word4: u64, word5: u64,
    wrdhsh: u32,
    xword0: u64, xword1: u64, xword2: u64, cword0: u64, ccword: u64,
    number0: u64, number1: u64,
    text0: u32, data0: u32, type0: u32,
    last_letter: u32, first_letter: u32, last_upper: u32,
    last_digit: u32, word_gap: u32,
    cm: ContextMap,
    nl1: i32, nl: i32,
    mask: u32, mask2: u32,
    wpos: Vec<i32>,
    w: u32,
    stem_words: [Word; 4],
    c_word: usize, p_word: usize,
    stem_index: usize,
}

impl WordModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            word0: 0, word1: 0, word2: 0, word3: 0, word4: 0, word5: 0,
            wrdhsh: 0,
            xword0: 0, xword1: 0, xword2: 0, cword0: 0, ccword: 0,
            number0: 0, number1: 0,
            text0: 0, data0: 0, type0: 0,
            last_letter: 0, first_letter: 0, last_upper: 0,
            last_digit: 0, word_gap: 0,
            cm: ContextMap::new(mem * 16, 61, dt),
            nl1: -3, nl: -2,
            mask: 0, mask2: 0,
            wpos: vec![0i32; 0x10000],
            w: 0,
            stem_words: [Word::new(), Word::new(), Word::new(), Word::new()],
            c_word: 0, p_word: 3,
            stem_index: 0,
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let mut c = (s.c4 & 255) as i32;
            let p_c = ((s.c4 >> 8) & 0xff) as u8;
            let mut f = 0u32;
            if s.spaces & 0x80000000 != 0 { s.spacecount -= 1; }
            if s.words & 0x80000000 != 0 { s.wordcount -= 1; }
            s.spaces = s.spaces.wrapping_mul(2);
            s.words = s.words.wrapping_mul(2);
            self.last_upper = (self.last_upper + 1).min(255);
            self.last_letter = (self.last_letter + 1).min(255);
            self.mask2 <<= 2;

            if (b'A' as i32..=b'Z' as i32).contains(&c) {
                c += b'a' as i32 - b'A' as i32;
                self.last_upper = 0;
            }
            let ci = c as u8;
            if (b'a'..=b'z').contains(&ci) || ci == b'\'' || ci == b'-' {
                self.stem_words[self.c_word].push(ci);
            } else if self.stem_words[self.c_word].length() > 0 {
                EnglishStemmer::stem(&mut self.stem_words[self.c_word]);
                self.stem_words[self.c_word].get_hashes();
                self.stem_index = (self.stem_index + 1) & 3;
                self.p_word = self.c_word;
                self.c_word = self.stem_index;
                self.stem_words[self.c_word] = Word::new();
            }

            let is_word = (b'a' as i32..=b'z' as i32).contains(&c)
                || (c >= 128 && s.b3 != 3)
                || (c > 0 && c < 4);
            if is_word {
                if s.wordlen == 0 {
                    self.word_gap = self.last_letter;
                    self.first_letter = c as u32;
                    self.wrdhsh = 0;
                }
                self.last_letter = 0;
                s.words += 1;
                s.wordcount += 1;
                if c > 4 {
                    self.word0 = combine64(self.word0, c as u64);
                }
                self.text0 = self.text0.wrapping_mul(997 * 16)
                    .wrapping_add(c as u32);
                s.wordlen += 1;
                s.wordlen = s.wordlen.min(45);
                f = 0;
                self.w = (self.word0 as u32) & (self.wpos.len() as u32 - 1);
                if c == b'a' as i32 || c == b'e' as i32 || c == b'i' as i32
                    || c == b'o' as i32 || c == b'u' as i32
                    || (c == b'y' as i32 && s.wordlen > 0
                        && p_c != b'a' && p_c != b'e' && p_c != b'i'
                        && p_c != b'o' && p_c != b'u')
                {
                    self.mask2 += 1;
                    self.wrdhsh = self.wrdhsh.wrapping_mul(997 * 8)
                        .wrapping_add((c / 4 - 22) as u32);
                } else if (b'b' as i32..=b'z' as i32).contains(&c) {
                    self.mask2 += 2;
                    self.wrdhsh = self.wrdhsh.wrapping_mul(271 * 32)
                        .wrapping_add((c - 97) as u32);
                } else {
                    self.wrdhsh = self.wrdhsh.wrapping_mul(11 * 32)
                        .wrapping_add(c as u32);
                }
            } else {
                if self.word0 != 0 {
                    self.type0 = (self.type0 << 2) | 1;
                    self.word5 = self.word4;
                    self.word4 = self.word3;
                    self.word3 = self.word2;
                    self.word2 = self.word1;
                    self.word1 = self.word0;
                    s.wordlen1 = s.wordlen;
                    self.wpos[self.w as usize] = s.blpos;
                    if c == b':' as i32 || c == b'=' as i32 {
                        self.cword0 = self.word0;
                    }
                    if c == b']' as i32 && s.frstchar != b':' as u32 {
                        self.xword0 = self.word0;
                    }
                    self.ccword = 0;
                    self.word0 = 0;
                    s.wordlen = 0;
                    if (c == b'.' as i32 || c == b'!' as i32
                        || c == b'?' as i32 || c == b'}' as i32
                        || c == b')' as i32)
                        && s.buf.at(2) != 10
                    {
                        f = 1;
                    }
                }
                if c == SPACE_U32 as i32 || c == 10 || c == 5 {
                    s.spaces += 1;
                    s.spacecount += 1;
                    if c == 10 || c == 5 {
                        self.nl1 = self.nl;
                        self.nl = s.buf.pos as i32 - 1;
                    }
                } else if c == b'.' as i32 || c == b'!' as i32
                    || c == b'?' as i32 || c == b',' as i32
                    || c == b';' as i32 || c == b':' as i32
                {
                    s.spafdo = 0;
                    self.ccword = c as u64;
                    self.mask2 += 3;
                } else {
                    s.spafdo += 1;
                    s.spafdo = s.spafdo.min(63);
                }
            }
            if (s.c4 & 0xFFFF) == 0x3D3D && s.frstchar == 0x3d {
                self.xword1 = self.word1;
            }
            if (s.c4 & 0xFFFF) == 0x2727 {
                self.xword2 = self.word1;
            }
            self.last_digit = self.last_digit.saturating_add(1).min(0xFF);
            if (b'0' as i32..=b'9' as i32).contains(&c) {
                if (b'0'..=b'9').contains(&s.buf.at(3))
                    && s.buf.at(2) == b'.' && self.number0 == 0
                {
                    self.number0 = self.number1;
                    self.number1 = 0;
                }
                self.number0 = combine64(self.number0, c as u64);
                self.last_digit = 0;
            } else if self.number0 != 0 {
                self.type0 = (self.type0 << 2) | 2;
                self.number1 = self.number0;
                self.number0 = 0;
                self.ccword = 0;
            }
            if !((b'a' as i32..=b'z' as i32).contains(&c)
                || (b'0' as i32..=b'9' as i32).contains(&c)
                || c >= 128)
            {
                self.data0 ^= combine64(self.data0 as u64, c as u64) as u32;
            } else if self.data0 != 0 {
                self.type0 = (self.type0 << 2) | 3;
                self.data0 = 0;
            }
            s.col = (s.buf.pos as i32 - self.nl).min(255).max(0) as u32;
            let above = s.buf.abs((self.nl1 + s.col as i32) as u32);
            if s.col <= 2 {
                s.frstchar = if s.col == 2 { (c as u32).min(96) } else { 0 };
            }
            if s.frstchar == b'[' as u32 && c == 32 {
                if s.buf.at(3) == b']' || s.buf.at(4) == b']' {
                    s.frstchar = 96;
                    self.xword0 = 0;
                }
            }

            // 61 ContextMap.set calls.
            let mut idx: u64 = 0;
            macro_rules! cs { ($v:expr) => {{ idx += 1; let _ = idx; self.cm.set($v); }} }
            let h0 = |a, b| hash3(a, b, 0);
            cs!(hash4(513, s.spafdo as u64, s.spaces as u64, self.ccword));
            cs!(hash3(514, s.frstchar as u64, c as u64));
            cs!(hash4(515, s.col as u64, s.frstchar as u64,
                ((self.last_upper < s.col) as u64) * 4 + (self.mask2 & 3) as u64));
            cs!(hash3(516, s.spaces as u64, (s.words & 255) as u64));
            cs!((s.spaces & 0x7fff) as u64);
            cs!((s.spaces & 0xff) as u64);
            cs!(hash4(257, self.number0, self.word1, self.word_gap as u64));
            cs!(hash4(258, self.number1, c as u64, self.ccword));
            cs!(hash4(259, self.number0, self.number1, self.word_gap as u64));
            cs!(hash4(260, self.word0, self.number1,
                (self.last_digit < self.word_gap + s.wordlen) as u64));
            cs!(hash3(274, self.number0, self.cword0));
            cs!(hash3(518, s.wordlen1 as u64, s.col as u64));
            cs!(hash4(519, c as u64, (s.spacecount / 2) as u64,
                self.word_gap as u64));
            let hh = (s.wordcount * 64 + s.spacecount) as u64;
            cs!(hash4(520, c as u64, hh, self.ccword));
            cs!(hash4(517, s.frstchar as u64, hh, self.last_letter as u64));
            cs!(hash4(self.data0 as u64, self.word1, self.number1,
                (self.type0 & 0xFFF) as u64));
            cs!(hash3(521, hh, s.spafdo as u64));
            let dd = (s.c4 & 0xf0ff) as u64;
            cs!(hash4(522, dd, s.frstchar as u64, self.ccword));
            let mut hword = (self.word0.wrapping_mul(271))
                .wrapping_add(s.buf.at(1) as u64);
            cs!(h0(262, hword));
            cs!(h0(self.number0.wrapping_mul(271)
                .wrapping_add(s.buf.at(1) as u64), 0));
            cs!(h0(263, self.word0));
            if self.wrdhsh != 0 {
                cs!(hash2(self.wrdhsh as u64,
                    s.buf.at(self.wpos[(self.word1 as usize)
                        & (self.wpos.len() - 1)] as u32) as u64));
            } else { cs!(0u64); }
            cs!(hash3(264, hword, self.word1));
            cs!(hash3(265, self.word0, self.word1));
            cs!(hash5(266, hword, self.word1, self.word2,
                (self.last_upper < s.wordlen) as u64));
            cs!(h0(267, (self.text0 & 0xffffff) as u64));
            cs!((self.text0 & 0xfffff) as u64);
            cs!(hash3(269, self.word0, self.xword0));
            cs!(hash3(270, hword, self.xword1));
            cs!(hash3(271, hword, self.xword2));
            cs!(hash3(272, s.frstchar as u64, self.xword2));
            cs!(hash3(273, self.word0, self.cword0));
            cs!(hash3(275, hword, self.word2));
            cs!(hash3(276, hword, self.word3));
            cs!(hash3(277, hword, self.word4));
            cs!(hash3(278, hword, self.word5));
            cs!(hash4(279, hword, self.word1, self.word3));
            cs!(hash4(280, hword, self.word2, self.word3));
            cs!((s.buf.at(1) as u64) | ((s.buf.at(3) as u64) << 8)
                | ((s.buf.at(5) as u64) << 16));
            cs!((s.buf.at(2) as u64) | ((s.buf.at(4) as u64) << 8)
                | ((s.buf.at(6) as u64) << 16));
            cs!((s.buf.at(1) as u64) | ((s.buf.at(4) as u64) << 8)
                | ((s.buf.at(7) as u64) << 16));
            if f != 0 {
                self.word5 = self.word4;
                self.word4 = self.word3;
                self.word3 = self.word2;
                self.word2 = self.word1;
                self.word1 = b'.' as u64;
            }
            if s.col < 255 {
                cs!(hash4(523, s.col as u64, s.buf.at(1) as u64,
                    above as u64));
                cs!(hash3(524, s.buf.at(1) as u64, above as u64));
                cs!(hash3(525, s.col as u64, s.buf.at(1) as u64));
                cs!(hash3(526, s.col as u64, (c == 32) as u64));
            } else {
                cs!(0u64); cs!(0u64); cs!(0u64); cs!(0u64);
            }
            if s.wordlen != 0 {
                cs!(hash3(281, self.word0,
                    (llog(&s.ilog, (s.blpos as u32).wrapping_sub(
                        self.wpos[(self.word1 as usize)
                            & (self.wpos.len() - 1)] as u32)) >> 4) as u64));
            } else { cs!(0u64); }
            cs!(hash3(282, s.buf.at(1) as u64,
                (llog(&s.ilog, (s.blpos as u32).wrapping_sub(
                    self.wpos[(self.word1 as usize)
                        & (self.wpos.len() - 1)] as u32)) >> 2) as u64));
            cs!(hash4(283, s.buf.at(1) as u64, self.word0,
                (llog(&s.ilog, (s.blpos as u32).wrapping_sub(
                    self.wpos[(self.word2 as usize)
                        & (self.wpos.len() - 1)] as u32)) >> 2) as u64));

            let mut fl = 0u32;
            let cb = s.c4 & 0xff;
            if cb != 0 {
                fl = if (cb as u8).is_ascii_alphabetic() { 1 }
                    else if (cb as u8).is_ascii_punctuation() { 2 }
                    else if (cb as u8).is_ascii_whitespace() { 3 }
                    else if cb == 0xff { 4 }
                    else if cb < 16 { 5 }
                    else if cb < 64 { 6 }
                    else { 7 };
            }
            self.mask = (self.mask << 3) | fl;
            cs!(hash3(528, self.mask as u64, 0));
            cs!(hash3(529, self.mask as u64, s.buf.at(1) as u64));
            cs!(hash3(530, (self.mask & 0xff) as u64, s.col as u64));
            cs!(hash4(531, self.mask as u64, s.buf.at(2) as u64,
                s.buf.at(3) as u64));
            cs!(hash4(532, (self.mask & 0x1ff) as u64,
                (s.f4 & 0x00fff0) as u64, 0));
            cs!(hash4(hword,
                llog(&s.ilog, self.word_gap) as u64,
                (self.mask & 0x1FF) as u64,
                (((s.wordlen1 > 3) as u64) << 6)
                | (((s.wordlen > 0) as u64) << 5)
                | (((s.spafdo == s.wordlen + 2) as u64) << 4)
                | (((s.spafdo == s.wordlen + s.wordlen1 + 3) as u64) << 3)
                | (((s.spafdo >= self.last_letter + s.wordlen1
                    + self.word_gap) as u64) << 2)
                | (((self.last_upper < self.last_letter + s.wordlen1) as u64) << 1)
                | ((self.last_upper < s.wordlen + s.wordlen1
                    + self.word_gap) as u64)));
            if s.wordlen1 != 0 {
                cs!(hash4(s.col as u64, s.wordlen1 as u64,
                    (above & 0x5F) as u64, (s.c4 & 0x5F) as u64));
            } else { cs!(0u64); }
            if self.wrdhsh != 0 {
                cs!(hash4((self.mask2 & 0x3F) as u64,
                    (self.wrdhsh & 0xFFF) as u64,
                    ((0x100 | self.first_letter)
                        * (s.wordlen < 6) as u32) as u64,
                    ((self.word_gap > 4) as u64) * 2
                        + (s.wordlen1 > 5) as u64));
            } else { cs!(0u64); }
            if self.last_letter < 16 {
                let pw_h2 = self.stem_words[self.p_word].hash[2];
                cs!(hash2(pw_h2, hword));
            } else { cs!(0u64); }
            let _ = &mut hword;
        }
        cm_mix(&mut self.cm, s, m);
    }
}

// =============================================================
// IndirectModel — paq8.cpp:7549-7596.
// =============================================================

pub struct IndirectModel {
    cm:    ContextMap,
    t1:    Vec<u32>,
    t2:    Vec<u16>,
    t3:    Vec<u16>,
    t4:    Vec<u16>,
    i_ctx: IndirectContext,
}

impl IndirectModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            cm:    ContextMap::new(mem, 15, dt),
            t1:    vec![0u32; 256],
            t2:    vec![0u16; 0x10000],
            t3:    vec![0u16; 0x8000],
            t4:    vec![0u16; 0x8000],
            i_ctx: IndirectContext::new(16, 8, 32),
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let c4 = s.c4;
            let b = |k: u32| s.buf.at(k) as u32;
            let d = c4 & 0xffff;
            let c = d & 255;
            let d2 = (b(1) & 31) + 32 * (b(2) & 31) + 1024 * (b(3) & 31);
            let d3 = (b(1) >> 3 & 31) + 32 * (b(3) >> 3 & 31)
                + 1024 * (b(4) >> 3 & 31);

            let r1i = (d >> 8) as usize;
            self.t1[r1i] = (self.t1[r1i] << 8) | c;
            let r2i = ((c4 >> 8) & 0xffff) as usize;
            self.t2[r2i] = (self.t2[r2i] << 8) | (c as u16);
            let r3i = ((b(2) & 31) + 32 * (b(3) & 31)
                + 1024 * (b(4) & 31)) as usize;
            self.t3[r3i] = (self.t3[r3i] << 8) | (c as u16);
            let r4i = ((b(2) >> 3 & 31) + 32 * (b(4) >> 3 & 31)
                + 1024 * (b(5) >> 3 & 31)) as usize;
            self.t4[r4i] = (self.t4[r4i] << 8) | (c as u16);

            let t  = c | (self.t1[c as usize] << 8);
            let t0 = d | ((self.t2[d as usize] as u32) << 16);
            let ta = d2 | ((self.t3[d2 as usize] as u32) << 16);
            let tc = d3 | ((self.t4[d3 as usize] as u32) << 16);
            let pc = ((c4 >> 8) as u8).to_ascii_lowercase();
            let c_lower = (c as u8).to_ascii_lowercase() as u32;
            self.i_ctx.add(c_lower);
            self.i_ctx.set(((pc as u32) << 8) | c_lower);
            let ctx0 = self.i_ctx.get();
            let mask = ((self.t1[c as usize] as u8 == self.t2[d as usize] as u8) as u32)
                | (((self.t1[c as usize] as u8 == self.t3[d2 as usize] as u8) as u32) << 1)
                | (((self.t1[c as usize] as u8 == self.t4[d3 as usize] as u8) as u32) << 2)
                | (((self.t1[c as usize] as u8 == ctx0 as u8) as u32) << 3);

            let mut i: u64 = 0;
            macro_rules! cs { ($v:expr) => {{ i += 1; self.cm.set(hash2(i, $v)); }} }
            cs!(t as u64);
            cs!(t0 as u64);
            cs!(ta as u64);
            cs!(tc as u64);
            cs!(((t & 0xff00) as u64) | ((mask as u64) << 16));
            cs!((t0 & 0xff0000) as u64);
            cs!((ta & 0xff0000) as u64);
            cs!((tc & 0xff0000) as u64);
            cs!((t & 0xffff) as u64);
            cs!((t0 & 0xffffff) as u64);
            cs!((ta & 0xffffff) as u64);
            cs!((tc & 0xffffff) as u64);
            cs!(hash2((ctx0 & 0xff) as u64, c as u64));
            cs!((ctx0 & 0xffff) as u64);
            cs!((ctx0 & 0x7f7fff) as u64);
        }
        cm_mix(&mut self.cm, s, m);
    }
}

/// Shared `ContextMap::mix` driver — forwards the paq8 globals a
/// `ContextMap` needs each bit.
pub(crate) fn cm_mix(cm: &mut ContextMap, s: &mut Paq8State, m: &mut Mixer) {
    let cc = s.c0;
    let bp = s.bpos;
    let c1 = s.buf.at(1);
    let y = s.y;
    cm.mix1(m, cc, bp, c1, y, &s.ilog, &s.squash, &s.stretch);
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    fn drive<F: FnMut(&mut Paq8State, &mut Mixer)>(mut f: F) {
        let mut st = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        for &byte in b"hello (world) {nested} text. 12345" {
            for bp in 0..8 {
                st.bpos = bp;
                st.c0 = if bp == 0 { 1 }
                    else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                st.y = ((byte >> (7 - bp)) & 1) as i32;
                f(&mut st, &mut mixer);
            }
            st.c4 = (st.c4 << 8) | byte as u32;
            st.buf.push(byte);
        }
    }

    #[test]
    fn pic_model_runs_without_panic() {
        let mut pm = PicModel::new(build_dt());
        drive(|s, m| pm.mix(s, m));
    }

    #[test]
    fn distance_model_runs_without_panic() {
        let mut dm = DistanceModel::new(64 * 1024, build_dt());
        drive(|s, m| dm.mix(s, m));
    }

    #[test]
    fn sparse_models_run_without_panic() {
        let mut sm = SparseModel::new(64 * 1024, build_dt());
        let mut sm1 = SparseModel1::new(16 * 1024, build_dt());
        drive(|s, m| { sm.mix(s, m, 1, 2); sm1.mix(s, m, 1, 2); });
    }

    #[test]
    fn nest_model_runs_without_panic() {
        let mut nm = NestModel::new(64 * 1024, build_dt());
        drive(|s, m| nm.mix(s, m));
    }

    #[test]
    fn indirect_model_runs_without_panic() {
        let mut im = IndirectModel::new(64 * 1024, build_dt());
        drive(|s, m| im.mix(s, m));
    }

    #[test]
    fn record_model1_runs_without_panic() {
        let mut rm = RecordModel1::new(build_dt());
        drive(|s, m| rm.mix(s, m));
    }

    #[test]
    fn linear_prediction_model_runs_without_panic() {
        let mut lpm = LinearPredictionModel::new();
        drive(|s, m| lpm.mix(s, m));
    }

    #[test]
    fn record_model_runs_without_panic() {
        let mut rm = RecordModel::new(64 * 1024, build_dt());
        drive(|s, m| rm.mix(s, m, true));
    }

    #[test]
    fn word_model_runs_without_panic() {
        let mut wm = WordModel::new(4 * 1024, build_dt());
        drive(|s, m| wm.mix(s, m));
    }
}
