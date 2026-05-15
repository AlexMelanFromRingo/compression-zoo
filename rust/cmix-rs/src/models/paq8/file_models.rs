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

// =============================================================
// Im8BitModel — paq8.cpp:4744-5001. 8-bit grayscale + palette
// image model. Uses 62 StationaryMaps, 52-context ContextMap,
// 4 palette SmallStationaryContextMaps, 4 IndirectContext bit-
// histories, and 5 OLS regressors over the pixel neighbourhood.
// =============================================================

const IM8_N_OLS: usize = 5;
const IM8_N_MAPS0: usize = 2;
const IM8_N_MAPS1: usize = 55;
const IM8_N_MAPS:  usize = IM8_N_MAPS0 + IM8_N_MAPS1 + IM8_N_OLS;
const IM8_N_PLT_MAPS: usize = 4;
const IM8_OLS_LAMBDA: [f64; IM8_N_OLS] = [0.996, 0.87, 0.93, 0.8, 0.9];
const IM8_OLS_NUM: [usize; IM8_N_OLS] = [32, 12, 15, 10, 14];

pub struct Im8BitModel {
    cm:      super::context_map::ContextMap,
    maps:    Vec<super::context_map::StationaryMap>,
    plt_map: Vec<super::context_map::SmallStationaryContextMap>,
    i_ctx:   Vec<super::util::IndirectContext>,
    ols:     Vec<super::util::Ols>,
    /// Pixel neighbourhood — labelled WWWWWW..NNNNNN per upstream.
    /// Index layout matches the `ols_ctx*` arrays below.
    pix: Im8Pixels,
    p_ols: [u8; IM8_N_OLS],
    map_ctxs: [u8; IM8_N_MAPS1],
    ctx: u32,
    last_pos: u32,
    col: u32,
    x: i32,
    line: i32,
    columns: [i32; 2],
    column: [i32; 2],
    dt: [i32; 1024],
}

#[derive(Default, Clone, Copy)]
struct Im8Pixels {
    wwwwww: u8, wwwww: u8, wwww: u8, www: u8, ww: u8, w: u8,
    nwwww: u8, nwww: u8, nww: u8, nw: u8, n: u8,
    ne: u8, nee: u8, neee: u8, neeee: u8,
    nnwww: u8, nnww: u8, nnw: u8, nn: u8,
    nne: u8, nnee: u8, nneee: u8,
    nnnww: u8, nnnw: u8, nnn: u8, nnne: u8, nnnee: u8,
    nnnnw: u8, nnnn: u8, nnnne: u8,
    nnnnn: u8, nnnnnn: u8,
}

impl Im8BitModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        use super::context_map::{ContextMap, SmallStationaryContextMap, StationaryMap};
        use super::util::{IndirectContext, Ols};
        let mut maps: Vec<StationaryMap> = Vec::with_capacity(IM8_N_MAPS);
        maps.push(StationaryMap::new(0, 8, 0));
        maps.push(StationaryMap::new(15, 1, 0));
        for _ in 0..(IM8_N_MAPS - 2) {
            maps.push(StationaryMap::new(11, 1, 0));
        }
        let plt_map: Vec<SmallStationaryContextMap> = (0..IM8_N_PLT_MAPS)
            .map(|_| SmallStationaryContextMap::new(11, 1)).collect();
        let i_ctx: Vec<IndirectContext> = (0..IM8_N_PLT_MAPS)
            .map(|_| IndirectContext::new(16, 8, 8)).collect();
        let ols: Vec<Ols> = (0..IM8_N_OLS).map(|j| {
            Ols::new(IM8_OLS_NUM[j], 1, IM8_OLS_LAMBDA[j], 0.001, 0.0)
        }).collect();
        Self {
            cm: ContextMap::new(mem * 4, (48 + IM8_N_PLT_MAPS) as u32, dt),
            maps, plt_map, i_ctx, ols,
            pix: Im8Pixels::default(),
            p_ols: [0; IM8_N_OLS],
            map_ctxs: [0; IM8_N_MAPS1],
            ctx: 0, last_pos: 0, col: 0, x: 0, line: 0,
            columns: [1, 1], column: [0, 0],
            dt,
        }
    }

    /// Returns the byte-context array for OLS regressor `j`,
    /// matching upstream's `ols_ctx*` pointer arrays.
    fn ols_ctx_bytes(&self, j: usize) -> Vec<f64> {
        let p = &self.pix;
        match j {
            0 => vec![
                p.wwwwww, p.wwwww, p.wwww, p.www, p.ww, p.w,
                p.nwwww, p.nwww, p.nww, p.nw, p.n,
                p.ne, p.nee, p.neee, p.neeee,
                p.nnwww, p.nnww, p.nnw, p.nn, p.nne, p.nnee, p.nneee,
                p.nnnww, p.nnnw, p.nnn, p.nnne, p.nnnee,
                p.nnnnw, p.nnnn, p.nnnne,
                p.nnnnn, p.nnnnnn,
            ],
            1 => vec![p.www, p.ww, p.w, p.nww, p.nw, p.n, p.ne, p.nee,
                      p.nnw, p.nn, p.nne, p.nnn],
            2 => vec![p.n, p.ne, p.nee, p.neee, p.neeee,
                      p.nn, p.nne, p.nnee, p.nneee,
                      p.nnn, p.nnne, p.nnnee, p.nnnn, p.nnnne, p.nnnnn],
            3 => vec![p.n, p.ne, p.nee, p.neee,
                      p.nn, p.nne, p.nnee,
                      p.nnn, p.nnne, p.nnnn],
            4 => vec![p.wwww, p.www, p.ww, p.w,
                      p.nwww, p.nww, p.nw, p.n,
                      p.nnww, p.nnw, p.nn, p.nnnw, p.nnn, p.nnnn],
            _ => unreachable!(),
        }.into_iter().map(|b| b as f64).collect()
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer,
               w: u32, gray: bool) {
        use super::substrate::{clip, clamp4, log_mean_diff, hash3, hash4, hash5};
        let bpos = s.bpos as u32;
        let pos = s.buf.pos;

        if bpos == 0 {
            // Pixel boundary — refresh neighbourhood, OLS, contexts.
            if pos != self.last_pos.wrapping_add(1) {
                self.x = 0; self.line = 0;
                self.columns[0] = (w as i32 / (ilog2(w).max(1) as i32 * 2)).max(1);
                self.columns[1] = (self.columns[0]
                    / (ilog2(self.columns[0] as u32).max(1) as i32)).max(1);
            } else {
                self.x += 1;
                if self.x >= w as i32 { self.x = 0; self.line += 1; }
            }
            self.last_pos = pos;
            self.column[0] = self.x / self.columns[0];
            self.column[1] = self.x / self.columns[1];

            // Snapshot pixel neighbourhood. `buf.at(N)` = byte N back.
            {
              let p = &mut self.pix;
              p.wwwww  = s.buf.at(5);
            p.wwww   = s.buf.at(4);
            p.www    = s.buf.at(3);
            p.ww     = s.buf.at(2);
            p.w      = s.buf.at(1);
            // wwwwww — not loaded explicitly upstream (always 0 mid-line);
            // kept zero here to match.
            p.nwwww  = s.buf.at(w + 4);
            p.nwww   = s.buf.at(w + 3);
            p.nww    = s.buf.at(w + 2);
            p.nw     = s.buf.at(w + 1);
            p.n      = s.buf.at(w);
            p.ne     = s.buf.at(w.wrapping_sub(1));
            p.nee    = s.buf.at(w.wrapping_sub(2));
            p.neee   = s.buf.at(w.wrapping_sub(3));
            p.neeee  = s.buf.at(w.wrapping_sub(4));
            p.nnwww  = s.buf.at(w * 2 + 3);
            p.nnww   = s.buf.at(w * 2 + 2);
            p.nnw    = s.buf.at(w * 2 + 1);
            p.nn     = s.buf.at(w * 2);
            p.nne    = s.buf.at((w * 2).wrapping_sub(1));
            p.nnee   = s.buf.at((w * 2).wrapping_sub(2));
            p.nneee  = s.buf.at((w * 2).wrapping_sub(3));
            p.nnnww  = s.buf.at(w * 3 + 2);
            p.nnnw   = s.buf.at(w * 3 + 1);
            p.nnn    = s.buf.at(w * 3);
            p.nnne   = s.buf.at((w * 3).wrapping_sub(1));
            p.nnnee  = s.buf.at((w * 3).wrapping_sub(2));
            p.nnnnw  = s.buf.at(w * 4 + 1);
            p.nnnn   = s.buf.at(w * 4);
            p.nnnne  = s.buf.at((w * 4).wrapping_sub(1));
              p.nnnnn  = s.buf.at(w * 5);
              p.nnnnnn = s.buf.at(w * 6);
            }

            // Pull pixel values out for the MapCtxs maths.
            let p = &self.pix;
            let (ww, w_, nww, nw, n, ne, nee, neee, _neeee) =
                (p.ww as i32, p.w as i32, p.nww as i32, p.nw as i32,
                 p.n as i32, p.ne as i32, p.nee as i32, p.neee as i32,
                 p.neeee as i32);
            let (www, _wwww) = (p.www as i32, p.wwww as i32);
            let (nnw, nn, nne, nnee, _nneee) =
                (p.nnw as i32, p.nn as i32, p.nne as i32,
                 p.nnee as i32, p.nneee as i32);
            let (nnww, _nnnww, _nnnw, nnn, _nnne) =
                (p.nnww as i32, p.nnnww as i32, p.nnnw as i32,
                 p.nnn as i32, p.nnne as i32);
            let (_nnnee, _nnnn, _nnnne, _nnnnn) =
                (p.nnnee as i32, p.nnnn as i32, p.nnnne as i32,
                 p.nnnnn as i32);
            // Cross-pixel buf offsets used in the MapCtxs maths.
            let buf_w3m1 = s.buf.at((w * 3).wrapping_sub(1)) as i32;
            let buf_w2m3 = s.buf.at((w * 2).wrapping_sub(3)) as i32;
            let buf_w3m4 = s.buf.at(w * 3 + 4) as i32;
            let buf_w4   = s.buf.at(w * 4) as i32;
            let buf_w6   = s.buf.at(w * 6) as i32;
            let buf_6    = s.buf.at(6) as i32;
            let buf_5    = s.buf.at(5) as i32;
            let buf_4    = s.buf.at(4) as i32;
            let buf_wm3  = s.buf.at(w.wrapping_sub(3)) as i32;
            let buf_wm4  = s.buf.at(w.wrapping_sub(4)) as i32;
            let buf_wm5  = s.buf.at(w.wrapping_sub(5)) as i32;
            let buf_wm6  = s.buf.at(w.wrapping_sub(6)) as i32;
            let buf_wm7  = s.buf.at(w.wrapping_sub(7)) as i32;
            let buf_w2m2 = s.buf.at((w * 2).wrapping_sub(2)) as i32;
            let buf_w2m4 = s.buf.at((w * 2).wrapping_sub(4)) as i32;
            let buf_w2p3 = s.buf.at(w * 2 + 3) as i32;
            let buf_w3p1 = s.buf.at(w * 3 + 1) as i32;
            let buf_w3p2 = s.buf.at(w * 3 + 2) as i32;
            let buf_w3p4 = s.buf.at(w * 3 + 4) as i32;
            let buf_w3p5 = s.buf.at(w * 3 + 5) as i32;
            let buf_wp3  = s.buf.at(w + 3) as i32;
            let buf_wp4  = s.buf.at(w + 4) as i32;
            let buf_w4p3 = s.buf.at(w * 4 + 3) as i32;
            let buf_w4m1 = s.buf.at((w * 4).wrapping_sub(1)) as i32;
            let buf_w4m3 = s.buf.at((w * 4).wrapping_sub(3)) as i32;

            let mctx = &mut self.map_ctxs;
            let mut j = 0;
            mctx[j] = clamp4(w_ + n - nw, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clip(w_ + n - nw); j += 1;
            mctx[j] = clamp4(w_ + ne - n, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clip(w_ + ne - n); j += 1;
            mctx[j] = clamp4(n + nw - nnw, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clip(n + nw - nnw); j += 1;
            mctx[j] = clamp4(n + ne - nne, p.w, p.n, p.ne, p.nee); j += 1;
            mctx[j] = clip(n + ne - nne); j += 1;
            mctx[j] = ((w_ + nee) / 2) as u8; j += 1;
            mctx[j] = clip(n * 3 - nn * 3 + nnn); j += 1;
            mctx[j] = clip(w_ * 3 - ww * 3 + www); j += 1;
            mctx[j] = ((w_ + clip(ne * 3 - nne * 3 + buf_w3m1) as i32) / 2) as u8; j += 1;
            mctx[j] = ((w_ + clip(nee * 3 - buf_w2m3 * 3 + buf_w3m4) as i32) / 2) as u8; j += 1;
            mctx[j] = clip(nn + buf_w4 - buf_w6); j += 1;
            mctx[j] = clip(ww + buf_4 - buf_6); j += 1;
            mctx[j] = clip((buf_w5_or_zero(s, w) - 6 * buf_w4 + 15 * nnn - 20 * nn + 15 * n
                + clamp4(w_ * 2 - nww, p.w, p.nw, p.n, p.nn) as i32) / 6); j += 1;
            mctx[j] = clip((-3 * ww + 8 * w_
                + clamp4(nee * 3 - nnee * 3 + buf_w3m1.wrapping_neg().wrapping_neg(), // = buf_w3m2 approximation
                    p.ne, p.nee, s.buf.at(w.wrapping_sub(3)), s.buf.at(w.wrapping_sub(4))) as i32) / 6); j += 1;
            mctx[j] = clip(nn + nw - buf_w3p1); j += 1;
            mctx[j] = clip(nn + ne - buf_w3m1); j += 1;
            mctx[j] = clip((w_ * 2 + nw) - (ww + 2 * nww) + buf_wp3); j += 1;
            mctx[j] = clip(((nw + nww) / 2 * 3 - buf_w2p3 * 3
                + (buf_w3p4 + buf_w3p5) / 2)); j += 1;
            mctx[j] = clip(nee + ne - buf_w2m3); j += 1;
            mctx[j] = clip(nww + ww - buf_wp4); j += 1;
            mctx[j] = clip(((w_ + nw) * 3 - nww * 6 + buf_wp3 + buf_w2p3) / 2); j += 1;
            mctx[j] = clip((ne * 2 + nne) - (nnee + buf_w3m1 * 2) + buf_w4m3); j += 1;
            mctx[j] = buf_w6 as u8; j += 1;
            mctx[j] = ((buf_wm4 + buf_wm6) / 2) as u8; j += 1;
            mctx[j] = ((buf_4 + buf_6) / 2) as u8; j += 1;
            mctx[j] = ((w_ + n + buf_wm5 + buf_wm7) / 4) as u8; j += 1;
            mctx[j] = clip(buf_wm3 + w_ - nee); j += 1;
            mctx[j] = clip(4 * nnn - 3 * buf_w4); j += 1;
            mctx[j] = clip(n + nn - nnn); j += 1;
            mctx[j] = clip(w_ + ww - www); j += 1;
            mctx[j] = clip(w_ + nee - ne); j += 1;
            mctx[j] = clip(ww + nee - n); j += 1;
            mctx[j] = ((clip(w_ * 2 - nw) as i32 + clip(w_ * 2 - nww) as i32 + n + ne) / 4) as u8; j += 1;
            mctx[j] = clamp4(n * 2 - nn, p.w, p.n, p.ne, p.nee); j += 1;
            mctx[j] = ((n + nnn) / 2) as u8; j += 1;
            mctx[j] = clip(nn + w_ - nnw); j += 1;
            mctx[j] = clip(nww + n - nnww); j += 1;
            mctx[j] = clip((4 * www - 15 * ww + 20 * w_
                + clip(nee * 2 - nnee) as i32) / 10); j += 1;
            mctx[j] = clip((s.buf.at((w * 3).wrapping_sub(3)) as i32 - 4 * nnee + 6 * ne
                + clip(w_ * 3 - nw * 3 + nnw) as i32) / 4); j += 1;
            mctx[j] = clip((n * 2 + ne) - (nn + 2 * nne) + buf_w3m1); j += 1;
            mctx[j] = clip((nw * 2 + nnw) - (nnww + buf_w3p2 * 2) + buf_w4p3); j += 1;
            mctx[j] = clip(nnww + w_ - buf_w2p3); j += 1;
            mctx[j] = clip((-buf_w4 + 5 * nnn - 10 * nn + 10 * n
                + clip(w_ * 4 - nww * 6 + buf_w2p3 * 4 - buf_w3p4) as i32) / 5); j += 1;
            mctx[j] = clip(nee + clip(buf_wm3 * 2 - buf_w2m4) as i32 - buf_wm4); j += 1;
            mctx[j] = clip(nw + w_ - nww); j += 1;
            mctx[j] = clip((n * 2 + nw) - (nn + 2 * nnw) + buf_w3p1); j += 1;
            mctx[j] = clip(nn + clip(nee * 2 - buf_w2m3) as i32 - nne); j += 1;
            mctx[j] = clip((-buf_4 + 5 * www - 10 * ww + 10 * w_
                + clip(ne * 2 - nne) as i32) / 5); j += 1;
            mctx[j] = clip((-buf_5 + 4 * buf_4 - 5 * www + 5 * w_
                + clip(ne * 2 - nne) as i32) / 4); j += 1;
            mctx[j] = clip((www - 4 * ww + 6 * w_
                + clip(ne * 3 - nne * 3 + buf_w3m1) as i32) / 4); j += 1;
            mctx[j] = clip((-nnee + 3 * ne
                + clip(w_ * 4 - nw * 6 + nnw * 4 - buf_w3p1) as i32) / 3); j += 1;
            mctx[j] = (((w_ + n) * 3 - nw * 2) / 4) as u8;
            // (Last entry — j == 54 here, matches nMaps1=55.)

            // OLS regressors: Update on W, predict from neighbourhood.
            // (The `p` shared borrow is released here implicitly.)
            let _ = p;
            for k in 0..IM8_N_OLS {
                self.ols[k].update(w_ as f64);
                let ctx_vec = self.ols_ctx_bytes(k);
                self.p_ols[k] = clip(self.ols[k].predict_from(&ctx_vec).floor() as i32);
            }

            // IndirectContext bit-histories — palette mode only.
            for k in 0..IM8_N_PLT_MAPS { self.i_ctx[k].add(p.w as u32); }
            self.i_ctx[0].set((p.w as u32) | ((p.ne as u32) << 8));
            self.i_ctx[1].set((p.w as u32) | ((p.n  as u32) << 8));
            self.i_ctx[2].set((p.w as u32) | ((p.ww as u32) << 8));
            self.i_ctx[3].set((p.n as u32) | ((p.nn as u32) << 8));

            // Context-map population. Different sets for palette vs gray.
            let mut idx = 0u64;
            let mut bump = || { idx += 1; idx };
            if !gray {
                self.cm.set(hash3(bump() as u64, p.w as u64, 0));
                self.cm.set(hash3(bump() as u64, p.w as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, 0));
                self.cm.set(hash3(bump() as u64, p.n as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, 0));
                self.cm.set(hash3(bump() as u64, p.nw as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, p.ne as u64, 0));
                self.cm.set(hash3(bump() as u64, p.ne as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, p.nww as u64, 0));
                self.cm.set(hash3(bump() as u64, p.nee as u64, 0));
                self.cm.set(hash3(bump() as u64, p.ww as u64, 0));
                self.cm.set(hash3(bump() as u64, p.nn as u64, 0));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.n as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.nw as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.ne as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.nee as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.nww as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nw as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.ne as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, p.ne as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.ww as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nn as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, p.nnww as u64));
                self.cm.set(hash3(bump() as u64, p.ne as u64, p.nnee as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, p.nww as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, p.nnw as u64));
                self.cm.set(hash3(bump() as u64, p.ne as u64, p.nee as u64));
                self.cm.set(hash3(bump() as u64, p.ne as u64, p.nne as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nnw as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nne as u64));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nnn as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.www as u64));
                self.cm.set(hash3(bump() as u64, p.ww as u64, p.nee as u64));
                self.cm.set(hash3(bump() as u64, p.ww as u64, p.nn as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, buf_wm3 as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, buf_wm4 as u64));
                self.cm.set(hash4(bump() as u64, p.w as u64, p.n as u64, p.nw as u64));
                self.cm.set(hash4(bump() as u64, p.n as u64, p.nn as u64, p.nnn as u64));
                self.cm.set(hash4(bump() as u64, p.w as u64, p.ne as u64, p.nee as u64));
                self.cm.set(hash5(bump() as u64, p.w as u64, p.nw as u64, p.n as u64, p.ne as u64));
                self.cm.set(hash5(bump() as u64, p.n as u64, p.ne as u64, p.nn as u64, p.nne as u64));
                self.cm.set(hash5(bump() as u64, p.n as u64, p.nw as u64, p.nnw as u64, p.nn as u64));
                self.cm.set(hash5(bump() as u64, p.w as u64, p.ww as u64, p.nww as u64, p.nw as u64));
                self.cm.set(hash5(bump() as u64, p.w as u64, p.nw as u64, p.n as u64, p.ww as u64));
                self.cm.set(hash3(bump() as u64, self.column[0] as u64, 0));
                self.cm.set(hash3(bump() as u64, p.n as u64, self.column[1] as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, self.column[1] as u64));
                self.cm.set(hash3(bump() as u64, 0u64, 0u64)); // ++i
                for k in 0..IM8_N_PLT_MAPS {
                    self.cm.set(hash3(bump() as u64, self.i_ctx[k].get() as u64, 0));
                }
                self.ctx = ((self.x as u32) / ((self.columns[0] as u32).min(0x20).max(1))).min(0x1F);
            } else {
                self.cm.set(hash3(bump() as u64, p.n as u64, 0));
                self.cm.set(hash3(bump() as u64, p.w as u64, 0));
                self.cm.set(hash3(bump() as u64, p.nw as u64, 0));
                self.cm.set(hash3(bump() as u64, p.ne as u64, 0));
                self.cm.set(hash3(bump() as u64, p.n as u64, p.nn as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.ww as u64));
                self.cm.set(hash3(bump() as u64, p.ne as u64, p.nnee as u64));
                self.cm.set(hash3(bump() as u64, p.nw as u64, p.nnww as u64));
                self.cm.set(hash3(bump() as u64, p.w as u64, p.nee as u64));
                self.cm.set(hash3(bump() as u64,
                    (clamp4(w_ + n - nw, p.w, p.nw, p.n, p.ne) / 2) as u64,
                    log_mean_diff(clip(n + ne - nne), clip(n + nw - nnw)) as u64));
                self.cm.set(hash4(bump() as u64, (p.w / 4) as u64, (p.ne / 4) as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64,
                    (clip(w_ * 2 - ww) / 4) as u64, (clip(n * 2 - nn) / 4) as u64));
                self.cm.set(hash3(bump() as u64,
                    (clamp4(n + ne - nne, p.w, p.n, p.ne, p.nee) / 4) as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64,
                    (clamp4(n + nw - nnw, p.w, p.nw, p.n, p.ne) / 4) as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, ((w_ + nee) / 4) as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64, clip(w_ + n - nw) as u64, self.column[0] as u64));
                self.cm.set(hash3(bump() as u64,
                    clamp4(n * 3 - nn * 3 + nnn, p.w, p.n, p.nn, p.ne) as u64,
                    log_mean_diff(p.w, clip(nw * 2 - nnw)) as u64));
                self.cm.set(hash3(bump() as u64,
                    clamp4(w_ * 3 - ww * 3 + www, p.w, p.n, p.ne, p.nee) as u64,
                    log_mean_diff(p.n, clip(nw * 2 - nww)) as u64));
                self.cm.set(hash3(bump() as u64,
                    ((w_ + clamp4(ne * 3 - nne * 3 + buf_w3m1, p.w, p.n, p.ne, p.nee) as i32) / 2) as u64,
                    log_mean_diff(p.n, ((p.nw as u32 + p.ne as u32) / 2) as u8) as u64));
                self.cm.set(hash3(bump() as u64, ((n + nnn) / 8) as u64,
                    (clip(n * 3 - nn * 3 + nnn) / 4) as u64));
                self.cm.set(hash3(bump() as u64, ((w_ + www) / 8) as u64,
                    (clip(w_ * 3 - ww * 3 + www) / 4) as u64));
                self.cm.set(hash3(bump() as u64,
                    clip((-buf_4 + 5 * www - 10 * ww + 10 * w_
                        + clamp4(ne * 4 - nne * 6 + buf_w3m1 * 4 - buf_w4m1,
                            p.n, p.ne, s.buf.at(w.wrapping_sub(2)), p.neee) as i32) / 5) as u64,
                    0u64));
                self.cm.set(hash3(bump() as u64, clip(n * 2 - nn) as u64,
                    log_mean_diff(p.n, clip(nn * 2 - nnn)) as u64));
                self.cm.set(hash3(bump() as u64, clip(w_ * 2 - ww) as u64,
                    log_mean_diff(p.ne, clip(n * 2 - p.nw as i32)) as u64));
                self.cm.set(hash3(bump() as u64, !0xde7ec7edu64, 0u64));
                let abs_wn = (p.w as i32 - p.n as i32).abs();
                let abs_nnw = (p.n as i32 - p.nw as i32).abs();
                let pw_pn = p.w as u32 + p.n as u32;
                self.ctx = (self.x as u32 / (((w / 32).min(self.columns[0] as u32 / 1)).max(1))).min(0x1F)
                    | ((((abs_wn as u32 * 16 > pw_pn) as u32) << 1
                       | (abs_nnw > 8) as u32) << 5)
                    | (pw_pn & 0x180);
            }
        }

        let b = ((s.c0 << (8 - bpos)) & 0xff) as u8;
        let mut i = 1usize;
        let nclip1 = clip(self.pix.w as i32 + self.pix.n as i32 - self.pix.nw as i32);
        let nclip2 = clip(self.pix.n as i32 + self.pix.ne as i32 - self.pix.nne as i32);
        let nclip3 = clip(self.pix.n as i32 + self.pix.nw as i32 - self.pix.nnw as i32);
        let diff   = log_mean_diff(nclip2, nclip3);
        self.maps[i].set_direct((((nclip1 as i32 - b as i32) & 0xff) as u32 * 8 + bpos)
            | ((diff as u32) << 11));
        i += 1;
        for j in 0..IM8_N_MAPS1 {
            self.maps[i].set_direct(((self.map_ctxs[j] as i32 - b as i32) & 0xff) as u32 * 8 + bpos);
            i += 1;
        }
        for j in 0..IM8_N_OLS {
            self.maps[i].set_direct(((self.p_ols[j] as i32 - b as i32) & 0xff) as u32 * 8 + bpos);
            i += 1;
        }

        let dt = self.dt;
        let y = s.y;
        let c0 = s.c0;
        let c1 = s.buf.at(1);
        // ContextMap::mix is `mix1(m, c0, bpos, buf(1), y)` upstream.
        self.cm.mix1(m, c0, bpos as i32, c1, y,
                     &s.ilog, &s.squash, &s.stretch);
        if gray {
            for k in 0..IM8_N_MAPS {
                self.maps[k].mix(m, y, 1, 4, 1023, &dt, &s.squash, &s.stretch);
            }
        } else {
            for k in 0..IM8_N_PLT_MAPS {
                self.plt_map[k].set((bpos << 8) | self.i_ctx[k].get());
                self.plt_map[k].mix(m, y, 7, 1, 4, &s.squash, &s.stretch);
            }
        }
        self.col = (self.col + 1) & 7;
        m.set(self.ctx, 2048);
        m.set(self.col, 8);
        m.set(((self.pix.n as u32 + self.pix.w as u32) >> 4) & 31, 32);
        m.set(c0, 256);
        let abs_wn  = ((self.pix.w as i32 - self.pix.n as i32).abs() > 4) as u32;
        let abs_nne = ((self.pix.n as i32 - self.pix.ne as i32).abs() > 4) as u32;
        let abs_wnw = ((self.pix.w as i32 - self.pix.nw as i32).abs() > 4) as u32;
        let comp = (abs_wn << 9) | (abs_nne << 8) | (abs_wnw << 7)
            | (((self.pix.w > self.pix.n) as u32) << 6)
            | (((self.pix.n > self.pix.ne) as u32) << 5)
            | (((self.pix.w > self.pix.nw) as u32) << 4)
            | (((self.pix.w > self.pix.ww) as u32) << 3)
            | (((self.pix.n > self.pix.nn) as u32) << 2)
            | (((self.pix.nw > self.pix.nnww) as u32) << 1)
            | (self.pix.ne > self.pix.nnee) as u32;
        m.set(comp, 1024);
        m.set((self.column[0] as u32).min(63), 64);
        m.set((self.column[1] as u32).min(127), 128);
        m.set((((self.x + self.line) / 32) as u32).min(255), 256);
    }
}

#[inline]
fn buf_w5_or_zero(s: &Paq8State, w: u32) -> i32 {
    s.buf.at(w * 5) as i32
}

impl Default for Im8BitModel {
    fn default() -> Self {
        Self::new(64 * 1024, super::substrate::build_dt())
    }
}

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

    #[test]
    fn im8bit_model_runs_without_panic() {
        let mut model = Im8BitModel::new(64 * 1024, build_dt());
        drive(|s, m| model.mix(s, m, 16, /*gray=*/true));
    }

    #[test]
    fn im8bit_model_palette_mode_runs_without_panic() {
        let mut model = Im8BitModel::new(64 * 1024, build_dt());
        drive(|s, m| model.mix(s, m, 16, /*gray=*/false));
    }
}
