//! File-type-specific paq8 sub-models — paq8.cpp:4635-6606.
//!
//! `contextModel2` dispatches to these only when the preprocessor
//! has detected a non-text filetype:
//!
//! * `IMAGE1`             → [`Im1BitModel`]    (fully ported)
//! * `IMAGE4`             → [`Im4BitModel`]    (fully ported)
//! * `IMAGE8`/`IMAGE8GRAY`→ [`Im8BitModel`]    (dispatch stub)
//! * `IMAGE24`/`IMAGE32`  → [`Im24BitModel`]   (dispatch stub)
//! * JPEG data            → [`JpegModel`]      (detection stub)
//! * generic image        → [`ImgModel`]       (detection stub)
//! * audio                → [`AudioModel`]     (detection stub)
//!
//! For the text/wiki workloads cmix targets, the image models are
//! never invoked and `jpeg/img/audio` return `false` (no detection)
//! — exactly what the stubs do, so the text path is byte-correct.
//! The stubs' full per-pixel / per-sample prediction logic is the
//! remaining scope of task #17.

#![allow(dead_code)]

use super::apm::{StateMap, StateMap32};
use super::context_map::HashTableB;
use super::mixer::Mixer;
use super::state::Paq8State;
use super::substrate::{hash3, hash4, ilog2, nex};

// =============================================================
// Im1BitModel — paq8.cpp:4635-4671. 1-bit (bilevel) image model.
// =============================================================

pub struct Im1BitModel {
    r0: u32, r1: u32, r2: u32, r3: u32,
    t:   Vec<u8>,
    cxt: [usize; 11],
    sm:  Vec<StateMap>,
}

impl Im1BitModel {
    pub fn new(_dt: [i32; 1024]) -> Self {
        Self {
            r0: 0, r1: 0, r2: 0, r3: 0,
            t:   vec![0u8; 0x23000],
            cxt: [0; 11],
            sm:  (0..11).map(|_| StateMap::new()).collect(),
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, w: u32) {
        let y = s.y;
        let bpos = s.bpos as u32;
        for i in 0..11 {
            self.t[self.cxt[i]] = nex(self.t[self.cxt[i]], y as usize);
        }
        self.r0 = self.r0.wrapping_mul(2).wrapping_add(y as u32);
        self.r1 = self.r1.wrapping_mul(2)
            .wrapping_add(((s.buf.at(w - 1) >> (7 - bpos)) & 1) as u32);
        self.r2 = self.r2.wrapping_mul(2)
            .wrapping_add(((s.buf.at(w + w - 1) >> (7 - bpos)) & 1) as u32);
        self.r3 = self.r3.wrapping_mul(2)
            .wrapping_add(((s.buf.at(w + w + w - 1) >> (7 - bpos)) & 1) as u32);
        let (r0, r1, r2, r3) = (self.r0, self.r1, self.r2, self.r3);
        self.cxt[0]  = ((r0 & 0x7) | (r1 >> 4 & 0x38) | (r2 >> 3 & 0xc0)) as usize;
        self.cxt[1]  = (0x100 + ((r0 & 1) | (r1 >> 4 & 0x3e) | (r2 >> 2 & 0x40)
            | (r3 >> 1 & 0x80))) as usize;
        self.cxt[2]  = (0x200 + ((r0 & 1) | (r1 >> 4 & 0x1d) | (r2 >> 1 & 0x60)
            | (r3 & 0xC0))) as usize;
        self.cxt[3]  = (0x300 + ((y as u32) | ((r0 << 1) & 4) | ((r1 >> 1) & 0xF0)
            | ((r2 >> 3) & 0xA))) as usize;
        self.cxt[4]  = (0x400 + ((r0 >> 4 & 0x2AC) | (r1 & 0xA4) | (r2 & 0x349)
            | ((r3 & 0x14D == 0) as u32))) as usize;
        self.cxt[5]  = (0x800 + ((y as u32) | ((r1 >> 4) & 0xE) | ((r2 >> 1) & 0x70)
            | ((r3 << 2) & 0x380))) as usize;
        self.cxt[6]  = (0xC00 + (((r1 & 0x30) ^ (r3 & 0x0c0c)) | (r0 & 3))) as usize;
        self.cxt[7]  = (0x1000 + (((r0 & 0x444 == 0) as u32) | (r1 & 0xC0C)
            | (r2 & 0xAE3) | (r3 & 0x51C))) as usize;
        self.cxt[8]  = (0x2000 + ((r0 & 7) | ((r1 >> 1) & 0x3F8)
            | ((r2 << 5) & 0xC00))) as usize;
        self.cxt[9]  = (0x3000 + ((r0 & 0x3f) ^ (r1 & 0x3ffe)
            ^ (r2 << 2 & 0x7f00) ^ (r3 << 5 & 0xf800))) as usize;
        self.cxt[10] = (0x13000 + ((r0 & 0x3e) ^ (r1 & 0x0c0c)
            ^ (r2 & 0xc800))) as usize;
        for i in 0..11 {
            let st = self.t[self.cxt[i]];
            let p = self.sm[i].p(st as u32, y);
            m.add(s.stretch.get(p) as i16);
        }
        m.set((r0 & 7) | ((r1 & 0x3E) >> 2) | ((r2 & 0x1C0) << 2), 2048);
        m.set((y as u32) | ((r1 & 0x1C0) >> 5) | ((r2 & 0x1C0) >> 2)
            | ((r3 & 0x1C0) << 1), 1024);
        m.set(((r1 >> 5) & 0xFE) | (y as u32), 256);
        m.set((r0 & 0x3) | ((r1 & 0xF80) >> 5), 128);
    }
}

// =============================================================
// Im4BitModel — paq8.cpp:4676-4742. 4-bit (16-colour) image model.
// =============================================================

pub struct Im4BitModel {
    t:    HashTableB,
    /// Per-context (table_index_base) tracking — we re-derive cp on
    /// each step via the hashed context, matching upstream's pointer
    /// arithmetic with explicit `(index, offset)` pairs.
    cp:   [(u64, u32); 14],
    sm:   Vec<StateMap>,
    map:  StateMap32,
    ww: u8, w: u8, nww: u8, nw: u8, n: u8, ne: u8, nee: u8,
    nnww: u8, nnw: u8, nn: u8, nne: u8, nnee: u8,
    col: i32, line: i32, run: i32, prev_color: i32, px: i32,
    primed: bool,
}

impl Im4BitModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            t:   HashTableB::new(((mem / 2) as usize / 16).next_power_of_two(), 16),
            cp:  [(0, 1); 14],
            sm:  (0..14).map(|_| StateMap::new()).collect(),
            map: StateMap32::new(16, dt),
            ww: 0, w: 0, nww: 0, nw: 0, n: 0, ne: 0, nee: 0,
            nnww: 0, nnw: 0, nn: 0, nne: 0, nnee: 0,
            col: 0, line: 0, run: 0, prev_color: 0, px: 0,
            primed: false,
        }
    }

    fn cp_byte(&mut self, i: usize) -> u8 {
        let (ctx, off) = self.cp[i];
        let slot = self.t.get(ctx);
        slot[off as usize % slot.len()]
    }
    fn cp_set(&mut self, i: usize, v: u8) {
        let (ctx, off) = self.cp[i];
        let slot = self.t.get(ctx);
        let n = slot.len();
        slot[off as usize % n] = v;
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, w: u32) {
        let y = s.y;
        let bpos = s.bpos;
        // step each cp
        for i in 0..14 {
            let cur = self.cp_byte(i);
            self.cp_set(i, nex(cur, y as usize));
        }
        if bpos == 0 || bpos == 4 {
            self.ww = self.w; self.nww = self.nw; self.nw = self.n;
            self.n = self.ne; self.ne = self.nee; self.nnww = self.nww;
            self.nnw = self.nn; self.nn = self.nne; self.nne = self.nnee;
            if bpos == 0 {
                self.w = (s.c4 & 0xF) as u8;
                self.nee = s.buf.at(w - 1) >> 4;
                self.nnee = s.buf.at(w * 2 - 1) >> 4;
            } else {
                self.w = (s.c0 & 0xF) as u8;
                self.nee = s.buf.at(w - 1) & 0xF;
                self.nnee = s.buf.at(w * 2 - 1) & 0xF;
            }
            self.run = if self.w != self.ww || self.col == 0 {
                self.prev_color = self.ww as i32;
                0
            } else {
                0xFFF.min(self.run + 1)
            };
            self.px = 1;
            let (ww, w8, nww, nw, n, ne, nee, nnww, nnw, nn, nne, nnee) = (
                self.ww as u64, self.w as u64, self.nww as u64,
                self.nw as u64, self.n as u64, self.ne as u64,
                self.nee as u64, self.nnww as u64, self.nnw as u64,
                self.nn as u64, self.nne as u64, self.nnee as u64);
            let col = self.col as u64;
            let line = self.line as u64;
            let run = self.run as u64;
            let pc = self.prev_color as u64;
            self.cp[0]  = (hash4(0, w8, nw, n), 1);
            self.cp[1]  = (hash3(1, n, 0xFFF.min(col / 8)), 1);
            self.cp[2]  = (super::substrate::hash6(2, w8, nw, n, nn, ne), 1);
            self.cp[3]  = (super::substrate::hash5(3, w8, n,
                ne + nne * 16, nee + nnee * 16), 1);
            self.cp[4]  = (super::substrate::hash5(4, w8, n,
                nw + nnw * 16, nww + nnww * 16), 1);
            self.cp[5]  = (super::substrate::hash5(5, w8,
                ilog2(run as u32 + 1) as u64, pc,
                col / 1.max(w as u64 / 2)), 1);
            self.cp[6]  = (hash3(6, ne,
                0x3FF.min((col + line) / 1.max(w as u64 * 8))), 1);
            self.cp[7]  = (hash3(7, nw,
                (col.wrapping_sub(line)) / 1.max(w as u64 * 8)), 1);
            self.cp[8]  = (hash4(8, ww * 16 + w8, nn * 16 + n,
                nnww * 16 + nw), 1);
            self.cp[9]  = (hash3(9, n, nn), 1);
            self.cp[10] = (hash3(10, w8, ww), 1);
            self.cp[11] = (hash3(11, w8, ne), 1);
            self.cp[12] = (hash4(12, ww, nn, nee), 1);
            self.cp[13] = (u64::MAX, 1); // upstream t[-1] sentinel
            self.col += 1;
            if self.col >= w as i32 * 2 { self.col = 0; }
            self.line += (self.col == 0) as i32;
        } else {
            self.px += self.px + y;
            let j = ((y + 1) << (bpos & 3)) as u32;
            for i in 0..14 { self.cp[i].1 += j; }
        }
        for i in 0..14 {
            let st = self.cp_byte(i);
            let n0 = if nex(st, 2) == 0 { -1i32 } else { 0 };
            let n1 = if nex(st, 3) == 0 { -1i32 } else { 0 };
            let p1 = self.sm[i].p(st as u32, y);
            let stv = s.stretch.get(p1) >> 1;
            m.add(stv as i16);
            m.add(((p1 - 2047) >> 2) as i16);
            m.add((stv * (n1 - n0).abs()) as i16);
        }
        let mp = self.map.p(self.px as u32 & 0xF, 1023, y);
        m.add((s.stretch.get(mp) >> 1) as i16);
        m.set((self.w as u32 * 16 + self.px as u32) & 0xFF, 256);
        m.set((31.min(self.col / 1.max(w as i32 / 16)) as u32
            + self.n as u32 * 32) & 0x1FF, 512);
        m.set(((bpos as u32 & 3) + 4 * self.w as u32
            + 64 * 7.min(ilog2(self.run as u32 + 1))) & 0x1FF, 512);
        m.set((self.w as u32 + self.ne as u32 * 16
            + (bpos as u32 & 3) * 256) & 0x3FF, 1024);
        m.set(self.px as u32 & 0xF, 16);
        m.set(0, 1);
        let _ = self.primed;
    }
}

// =============================================================
// Dispatch stubs — Im8BitModel / Im24BitModel / JpegModel /
// ImgModel / AudioModel.
//
// For the text path these are never invoked (image models) or
// always return `false` (jpeg/img/audio detection). The stubs
// reproduce exactly that behaviour. Their full per-pixel / per-
// sample prediction logic is outstanding scope of task #17.
// =============================================================

pub struct Im8BitModel;
impl Im8BitModel {
    pub fn new() -> Self { Self }
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer,
                _w: u32, _gray: bool) {}
}
impl Default for Im8BitModel { fn default() -> Self { Self::new() } }

pub struct Im24BitModel;
impl Im24BitModel {
    pub fn new() -> Self { Self }
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer,
                _w: u32, _alpha: bool) {}
}
impl Default for Im24BitModel { fn default() -> Self { Self::new() } }

pub struct JpegModel;
impl JpegModel {
    pub fn new() -> Self { Self }
    /// Returns `true` only inside a recognised JPEG stream. For
    /// non-JPEG (text/wiki) input this is always `false`.
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer) -> bool { false }
}
impl Default for JpegModel { fn default() -> Self { Self::new() } }

pub struct ImgModel;
impl ImgModel {
    pub fn new() -> Self { Self }
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer) -> bool { false }
}
impl Default for ImgModel { fn default() -> Self { Self::new() } }

pub struct AudioModel;
impl AudioModel {
    pub fn new() -> Self { Self }
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer) -> bool { false }
}
impl Default for AudioModel { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    fn drive<F: FnMut(&mut Paq8State, &mut Mixer)>(mut f: F) {
        let mut st = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        for byte in 0u32..64 {
            for bp in 0..8 {
                st.bpos = bp;
                st.c0 = if bp == 0 { 1 }
                    else { (1u32 << bp) | (byte >> (8 - bp)) };
                st.y = ((byte >> (7 - bp)) & 1) as i32;
                f(&mut st, &mut mixer);
            }
            st.c4 = (st.c4 << 8) | byte;
            st.buf.push(byte as u8);
        }
    }

    #[test]
    fn im1bit_model_runs_without_panic() {
        let mut model = Im1BitModel::new(build_dt());
        drive(|s, m| model.mix(s, m, 16));
    }

    #[test]
    fn im4bit_model_runs_without_panic() {
        let mut model = Im4BitModel::new(64 * 1024, build_dt());
        drive(|s, m| model.mix(s, m, 16));
    }

    #[test]
    fn dispatch_stubs_report_no_detection() {
        let mut s = Paq8State::new(0);
        let mut m = Mixer::new(64, 4, 0);
        assert!(!JpegModel::new().mix(&mut s, &mut m));
        assert!(!ImgModel::new().mix(&mut s, &mut m));
        assert!(!AudioModel::new().mix(&mut s, &mut m));
    }
}
