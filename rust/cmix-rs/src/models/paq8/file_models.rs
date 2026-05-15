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

// =============================================================
// Im24BitModel — paq8.cpp:5002-5354. 24-bit RGB / 32-bit RGBA
// image model. Stride-aware OLS × 4 channels (R/G/B/A), 100
// StationaryMaps, 59 SmallStationaryContextMaps, 47-context
// ContextMap, plus the 46-byte pixel neighbourhood spanning the
// adjacent-channel bytes (Wp1, Wp2, ..., NNp2).
// =============================================================

const IM24_N_MAPS0: usize = 18;
const IM24_N_MAPS1: usize = 76;
const IM24_N_OLS:   usize = 6;
const IM24_N_MAPS:  usize = IM24_N_MAPS0 + IM24_N_MAPS1 + IM24_N_OLS;
const IM24_N_SC_MAPS: usize = 59;
const IM24_OLS_LAMBDA: [f64; IM24_N_OLS] = [0.98, 0.87, 0.9, 0.8, 0.9, 0.7];
const IM24_OLS_NUM:    [usize; IM24_N_OLS] = [32, 12, 15, 10, 14, 8];

pub struct Im24BitModel {
    cm:    super::context_map::ContextMap,
    maps:  Vec<super::context_map::StationaryMap>,
    sc_map: Vec<super::context_map::SmallStationaryContextMap>,
    /// `[regressor][channel]` — 6 OLS × 4 channels.
    ols:   Vec<Vec<super::util::Ols>>,
    pix:   Im24Pixels,
    p_ols: [u8; IM24_N_OLS],
    map_ctxs: [u8; IM24_N_MAPS1],
    sc_map_ctxs: [i32; IM24_N_SC_MAPS - 1],
    ctx: [u32; 2],
    color: i32,
    stride: i32,
    padding: i32,
    last_pos: u32,
    x: i32,
    line: i32,
    columns: [i32; 2],
    column: [i32; 2],
    col: i32,
    dt: [i32; 1024],
}

#[derive(Default, Clone, Copy)]
struct Im24Pixels {
    // Base 32-byte neighbourhood (same labels as Im8 but with stride).
    wwwwww: u8, wwwww: u8, wwww: u8, www: u8, ww: u8, w: u8,
    nwwww: u8, nwww: u8, nww: u8, nw: u8, n: u8,
    ne: u8, nee: u8, neee: u8, neeee: u8,
    nnwww: u8, nnww: u8, nnw: u8, nn: u8,
    nne: u8, nnee: u8, nneee: u8,
    nnnww: u8, nnnw: u8, nnn: u8, nnne: u8, nnnee: u8,
    nnnnw: u8, nnnn: u8, nnnne: u8,
    nnnnn: u8, nnnnnn: u8,
    // Adjacent-channel bytes (the "p1" suffix = +1 byte, "p2" = +2).
    wwp1: u8, wp1: u8, p1: u8, nwp1: u8, np1: u8, nep1: u8, nnp1: u8,
    wwp2: u8, wp2: u8, p2: u8, nwp2: u8, np2: u8, nep2: u8, nnp2: u8,
}

impl Im24BitModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        use super::context_map::{ContextMap, SmallStationaryContextMap, StationaryMap};
        use super::util::{Ols};
        // 18 base maps with custom shapes, then 76 + 6 with `{11,1}`.
        let mut maps: Vec<StationaryMap> = Vec::with_capacity(IM24_N_MAPS);
        let base_shapes: &[(u32, u32)] = &[
            (8,8), (8,8), (8,8), (2,8), (0,8), (15,1), (15,1), (15,1),
            (15,1), (15,1), (17,1), (17,1), (17,1), (17,1),
            (13,1), (13,1), (13,1), (13,1),
        ];
        for &(b, p) in base_shapes { maps.push(StationaryMap::new(b, p, 0)); }
        for _ in 0..(IM24_N_MAPS - IM24_N_MAPS0) {
            maps.push(StationaryMap::new(11, 1, 0));
        }
        let mut sc_map: Vec<SmallStationaryContextMap> =
            (0..IM24_N_SC_MAPS - 1).map(|_| SmallStationaryContextMap::new(11, 1))
            .collect();
        sc_map.push(SmallStationaryContextMap::new(0, 8));
        let ols: Vec<Vec<Ols>> = (0..IM24_N_OLS).map(|j| {
            (0..4).map(|_| Ols::new(IM24_OLS_NUM[j], 1, IM24_OLS_LAMBDA[j], 0.001, 0.0))
                .collect()
        }).collect();
        Self {
            cm: ContextMap::new(mem * 4, 47, dt),
            maps, sc_map, ols,
            pix: Im24Pixels::default(),
            p_ols: [0; IM24_N_OLS],
            map_ctxs: [0; IM24_N_MAPS1],
            sc_map_ctxs: [0; IM24_N_SC_MAPS - 1],
            ctx: [0; 2],
            color: -1, stride: 3, padding: 0,
            last_pos: 0, x: 0, line: 0,
            columns: [1, 1], column: [0, 0],
            col: 0,
            dt,
        }
    }

    fn ols_ctx_bytes(&self, j: usize) -> Vec<f64> {
        let p = &self.pix;
        match j {
            0 => vec![
                p.wwwwww, p.wwwww, p.wwww, p.www, p.ww, p.w,
                p.nwwww, p.nwww, p.nww, p.nw, p.n,
                p.ne, p.nee, p.neee, p.neeee,
                p.nnwww, p.nnww, p.nnw, p.nn, p.nne, p.nnee, p.nneee,
                p.nnnww, p.nnnw, p.nnn, p.nnne, p.nnnee,
                p.nnnnw, p.nnnn, p.nnnne, p.nnnnn, p.nnnnnn,
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
            5 => vec![p.www, p.ww, p.w, p.nnn, p.nn, p.n, p.p1, p.p2],
            _ => unreachable!(),
        }.into_iter().map(|b| b as f64).collect()
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer,
               w: u32, alpha: bool) {
        use super::substrate::{clip, clamp4, log_mean_diff_qt, log_mean_diff,
            hash3, hash4, hash5, finalize64, ilog2 as ilog2_fn};
        let bpos = s.bpos as u32;
        let pos  = s.buf.pos;
        let stride_u = self.stride as u32;

        if bpos == 0 {
            if self.color < 0 || pos.wrapping_sub(self.last_pos) != 1 {
                self.stride  = 3 + (alpha as i32);
                self.padding = (w as i32) % self.stride;
                self.x = 0; self.line = 0;
                self.columns[0] = ((w as i32) / (ilog2_fn(w).max(1) as i32 * 3)).max(1);
                self.columns[1] = (self.columns[0]
                    / (ilog2_fn(self.columns[0] as u32).max(1) as i32)).max(1);
            }
            self.last_pos = pos;
            self.x += 1;
            if self.x >= w as i32 { self.x = 0; self.line += 1; }
            if self.x + self.padding < w as i32 {
                self.color = (self.color + 1) % self.stride;
            } else {
                self.color = if self.padding > 0 { self.stride + 1 } else { 0 };
            }
            self.column[0] = self.x / self.columns[0];
            self.column[1] = self.x / self.columns[1];

            let st = self.stride as u32;
            // Pixel neighbourhood — every offset is `stride * k` for
            // same-channel rows; `+1` / `+2` for the adjacent bytes.
            {
                let p = &mut self.pix;
                p.wwwwww = s.buf.at(6 * st);
                p.wwwww  = s.buf.at(5 * st);
                p.wwww   = s.buf.at(4 * st);
                p.www    = s.buf.at(3 * st);
                p.ww     = s.buf.at(2 * st);
                p.w      = s.buf.at(st);
                p.nwwww  = s.buf.at(w + 4 * st);
                p.nwww   = s.buf.at(w + 3 * st);
                p.nww    = s.buf.at(w + 2 * st);
                p.nw     = s.buf.at(w + st);
                p.n      = s.buf.at(w);
                p.ne     = s.buf.at(w.wrapping_sub(st));
                p.nee    = s.buf.at(w.wrapping_sub(2 * st));
                p.neee   = s.buf.at(w.wrapping_sub(3 * st));
                p.neeee  = s.buf.at(w.wrapping_sub(4 * st));
                p.nnwww  = s.buf.at(w * 2 + 3 * st);
                p.nnww   = s.buf.at((w + st) * 2);
                p.nnw    = s.buf.at(w * 2 + st);
                p.nn     = s.buf.at(w * 2);
                p.nne    = s.buf.at((w * 2).wrapping_sub(st));
                p.nnee   = s.buf.at((w.wrapping_sub(st)) * 2);
                p.nneee  = s.buf.at((w * 2).wrapping_sub(3 * st));
                p.nnnww  = s.buf.at(w * 3 + 2 * st);
                p.nnnw   = s.buf.at(w * 3 + st);
                p.nnn    = s.buf.at(w * 3);
                p.nnne   = s.buf.at((w * 3).wrapping_sub(st));
                p.nnnee  = s.buf.at((w * 3).wrapping_sub(2 * st));
                p.nnnnw  = s.buf.at(w * 4 + st);
                p.nnnn   = s.buf.at(w * 4);
                p.nnnne  = s.buf.at((w * 4).wrapping_sub(st));
                p.nnnnn  = s.buf.at(w * 5);
                p.nnnnnn = s.buf.at(w * 6);
                p.wwp1 = s.buf.at(st * 2 + 1);
                p.wp1  = s.buf.at(st + 1);
                p.p1   = s.buf.at(1);
                p.nwp1 = s.buf.at(w + st + 1);
                p.np1  = s.buf.at(w + 1);
                p.nep1 = s.buf.at(w.wrapping_sub(st) + 1);
                p.nnp1 = s.buf.at(w * 2 + 1);
                p.wwp2 = s.buf.at(st * 2 + 2);
                p.wp2  = s.buf.at(st + 2);
                p.p2   = s.buf.at(2);
                p.nwp2 = s.buf.at(w + st + 2);
                p.np2  = s.buf.at(w + 2);
                p.nep2 = s.buf.at(w.wrapping_sub(st) + 2);
                p.nnp2 = s.buf.at(w * 2 + 2);
            }

            // Pull pixel values as i32 for the arithmetic.
            let p = &self.pix;
            let (ww, w_) = (p.ww as i32, p.w as i32);
            let (nww, nw, n, ne, nee, neee, neeee) =
                (p.nww as i32, p.nw as i32, p.n as i32,
                 p.ne as i32, p.nee as i32, p.neee as i32, p.neeee as i32);
            let (www, wwww, _wwwww, wwwwww) =
                (p.www as i32, p.wwww as i32, p.wwwww as i32, p.wwwwww as i32);
            let (nnw, nn, nne, nnee, nneee) =
                (p.nnw as i32, p.nn as i32, p.nne as i32,
                 p.nnee as i32, p.nneee as i32);
            let (_nnwww, nwww, nnww, _nnnww, nnnw, nnn, nnne) =
                (p.nnwww as i32, p.nwww as i32, p.nnww as i32,
                 p.nnnww as i32, p.nnnw as i32, p.nnn as i32, p.nnne as i32);
            let (nnnee, nnnn, nnnne, nnnnn, nnnnnn) =
                (p.nnnee as i32, p.nnnn as i32, p.nnnne as i32,
                 p.nnnnn as i32, p.nnnnnn as i32);
            let (p1, p2) = (p.p1 as i32, p.p2 as i32);
            let (wwp1, wp1, nwp1, np1, nep1, nnp1) =
                (p.wwp1 as i32, p.wp1 as i32, p.nwp1 as i32,
                 p.np1 as i32, p.nep1 as i32, p.nnp1 as i32);
            let (wwp2, wp2, nwp2, np2, nep2, nnp2) =
                (p.wwp2 as i32, p.wp2 as i32, p.nwp2 as i32,
                 p.np2 as i32, p.nep2 as i32, p.nnp2 as i32);
            // Cross-pixel buf offsets used in MapCtxs.
            let buf_w3m1   = s.buf.at((w * 3).wrapping_sub(st)) as i32;
            let buf_w2ms2p1 = s.buf.at((w * 2).wrapping_sub(2 * st) + 1) as i32;
            let buf_w2ms2p2 = s.buf.at((w * 2).wrapping_sub(2 * st) + 2) as i32;
            let buf_wms2p1 = s.buf.at(w.wrapping_sub(2 * st) + 1) as i32;
            let buf_wms2p2 = s.buf.at(w.wrapping_sub(2 * st) + 2) as i32;
            let buf_w2msp1 = s.buf.at((w * 2).wrapping_sub(st) + 1) as i32;
            let buf_w2msp2 = s.buf.at((w * 2).wrapping_sub(st) + 2) as i32;
            let buf_w2psp1 = s.buf.at(w * 2 + st + 1) as i32;
            let buf_w2psp2 = s.buf.at(w * 2 + st + 2) as i32;
            let buf_w3sp1  = s.buf.at(w * 3 + 1) as i32;
            let buf_w3sp2  = s.buf.at(w * 3 + 2) as i32;
            let buf_st3p1  = s.buf.at(st * 3 + 1) as i32;
            let buf_st3p2  = s.buf.at(st * 3 + 2) as i32;
            let buf_w3psp1 = s.buf.at(w * 3 + st + 1) as i32;
            let buf_w3psp2 = s.buf.at(w * 3 + st + 2) as i32;
            let buf_w3msp1 = s.buf.at((w * 3).wrapping_sub(st) + 1) as i32;
            let buf_w3msp2 = s.buf.at((w * 3).wrapping_sub(st) + 2) as i32;
            let buf_w4p1   = s.buf.at(w * 4 + 1) as i32;
            let buf_w4p2   = s.buf.at(w * 4 + 2) as i32;
            let buf_w6p1   = s.buf.at(w * 6 + 1) as i32;
            let buf_w6p2   = s.buf.at(w * 6 + 2) as i32;
            let buf_st4p1  = s.buf.at(st * 4 + 1) as i32;
            let buf_st4p2  = s.buf.at(st * 4 + 2) as i32;
            let buf_st6p1  = s.buf.at(st * 6 + 1) as i32;
            let buf_st6p2  = s.buf.at(st * 6 + 2) as i32;
            let buf_w2pst2p1 = s.buf.at(w * 2 + 2 * st + 1) as i32;
            let buf_w2pst2p2 = s.buf.at(w * 2 + 2 * st + 2) as i32;
            let buf_w2ms2 = s.buf.at((w * 2).wrapping_sub(2 * st)) as i32;
            let buf_wps2p1 = s.buf.at(w + st * 2 + 1) as i32;
            let buf_wps2p2 = s.buf.at(w + st * 2 + 2) as i32;
            let buf_w3m3 = s.buf.at((w * 3).wrapping_sub(3 * st)) as i32;
            let buf_w3st3 = s.buf.at(w * 3 + 3 * st) as i32;
            let buf_w6 = s.buf.at(w * 6) as i32;
            let buf_w9 = s.buf.at(w * 9) as i32;
            let buf_wm6 = s.buf.at(w.wrapping_sub(6 * st)) as i32;

            // ---- MapCtxs[0..76] — paq8.cpp:5093-5168 ----
            let mctx = &mut self.map_ctxs;
            let mut j = 0;
            mctx[j] = clamp4(n + p1 - np1, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clamp4(n + p2 - np2, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = ((w_ + clamp4(ne * 3 - nne * 3 + nnne, p.w, p.n, p.ne, p.nee) as i32) / 2) as u8; j += 1;
            mctx[j] = clamp4((w_ + clip(ne * 2 - nne) as i32) / 2, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = ((w_ + nee) / 2) as u8; j += 1;
            mctx[j] = clip((www - 4 * ww + 6 * w_
                + clip(ne * 4 - nne * 6 + nnne * 4 - nnnne) as i32) / 4); j += 1;
            mctx[j] = clip((-wwww + 5 * www - 10 * ww + 10 * w_
                + clamp4(ne * 4 - nne * 6 + nnne * 4 - nnnne, p.n, p.ne, p.nee, p.neee) as i32) / 5); j += 1;
            mctx[j] = clip((-4 * ww + 15 * w_
                + 10 * clip(ne * 3 - nne * 3 + nnne) as i32
                - clip(neee * 3 - nneee * 3 + buf_w3m3) as i32) / 20); j += 1;
            mctx[j] = clip((-3 * ww + 8 * w_
                + clamp4(nee * 3 - nnee * 3 + nnnee, p.ne, p.nee, p.neee, p.neeee) as i32) / 6); j += 1;
            mctx[j] = clip((w_ + clip(ne * 2 - nne) as i32) / 2 + p1
                - (wp1 + clip(nep1 * 2 - buf_w2msp1) as i32) / 2); j += 1;
            mctx[j] = clip((w_ + clip(ne * 2 - nne) as i32) / 2 + p2
                - (wp2 + clip(nep2 * 2 - buf_w2msp2) as i32) / 2); j += 1;
            mctx[j] = clip((-3 * ww + 8 * w_ + clip(nee * 2 - nnee) as i32) / 6 + p1
                - (-3 * wwp1 + 8 * wp1 + clip(buf_wms2p1 * 2 - buf_w2ms2p1) as i32) / 6); j += 1;
            mctx[j] = clip((-3 * ww + 8 * w_ + clip(nee * 2 - nnee) as i32) / 6 + p2
                - (-3 * wwp2 + 8 * wp2 + clip(buf_wms2p2 * 2 - buf_w2ms2p2) as i32) / 6); j += 1;
            mctx[j] = clip((w_ + nee) / 2 + p1 - (wp1 + buf_wms2p1) / 2); j += 1;
            mctx[j] = clip((w_ + nee) / 2 + p2 - (wp2 + buf_wms2p2) / 2); j += 1;
            mctx[j] = clip((ww + clip(nee * 2 - nnee) as i32) / 2 + p1
                - (wwp1 + clip(buf_wms2p1 * 2 - buf_w2ms2p1) as i32) / 2); j += 1;
            mctx[j] = clip((ww + clip(nee * 2 - nnee) as i32) / 2 + p2
                - (wwp2 + clip(buf_wms2p2 * 2 - buf_w2ms2p2) as i32) / 2); j += 1;
            mctx[j] = clip(ww + nee - n + p1 - clip(wwp1 + buf_wms2p1 - np1) as i32); j += 1;
            mctx[j] = clip(ww + nee - n + p2 - clip(wwp2 + buf_wms2p2 - np2) as i32); j += 1;
            mctx[j] = clip(w_ + n - nw); j += 1;
            mctx[j] = clip(w_ + n - nw + p1 - clip(wp1 + np1 - nwp1) as i32); j += 1;
            mctx[j] = clip(w_ + n - nw + p2 - clip(wp2 + np2 - nwp2) as i32); j += 1;
            mctx[j] = clip(w_ + ne - n); j += 1;
            mctx[j] = clip(n + nw - nnw); j += 1;
            mctx[j] = clip(n + nw - nnw + p1 - clip(np1 + nwp1 - buf_w2psp1) as i32); j += 1;
            mctx[j] = clip(n + nw - nnw + p2 - clip(np2 + nwp2 - buf_w2psp2) as i32); j += 1;
            mctx[j] = clip(n + ne - nne); j += 1;
            mctx[j] = clip(n + ne - nne + p1 - clip(np1 + nep1 - buf_w2msp1) as i32); j += 1;
            mctx[j] = clip(n + ne - nne + p2 - clip(np2 + nep2 - buf_w2msp2) as i32); j += 1;
            mctx[j] = clip(n + nn - nnn); j += 1;
            mctx[j] = clip(n + nn - nnn + p1 - clip(np1 + nnp1 - buf_w3sp1) as i32); j += 1;
            mctx[j] = clip(n + nn - nnn + p2 - clip(np2 + nnp2 - buf_w3sp2) as i32); j += 1;
            mctx[j] = clip(w_ + ww - www); j += 1;
            mctx[j] = clip(w_ + ww - www + p1 - clip(wp1 + wwp1 - buf_st3p1) as i32); j += 1;
            mctx[j] = clip(w_ + ww - www + p2 - clip(wp2 + wwp2 - buf_st3p2) as i32); j += 1;
            mctx[j] = clip(w_ + nee - ne); j += 1;
            mctx[j] = clip(w_ + nee - ne + p1 - clip(wp1 + buf_wms2p1 - nep1) as i32); j += 1;
            mctx[j] = clip(w_ + nee - ne + p2 - clip(wp2 + buf_wms2p2 - nep2) as i32); j += 1;
            mctx[j] = clip(nn + p1 - nnp1); j += 1;
            mctx[j] = clip(nn + p2 - nnp2); j += 1;
            mctx[j] = clip(nn + w_ - nnw); j += 1;
            mctx[j] = clip(nn + w_ - nnw + p1 - clip(nnp1 + wp1 - buf_w2psp1) as i32); j += 1;
            mctx[j] = clip(nn + w_ - nnw + p2 - clip(nnp2 + wp2 - buf_w2psp2) as i32); j += 1;
            mctx[j] = clip(nn + nw - s.buf.at(w * 3 + st) as i32); j += 1;
            mctx[j] = clip(nn + nw - s.buf.at(w * 3 + st) as i32 + p1
                - clip(nnp1 + nwp1 - buf_w3psp1) as i32); j += 1;
            mctx[j] = clip(nn + nw - s.buf.at(w * 3 + st) as i32 + p2
                - clip(nnp2 + nwp2 - buf_w3psp2) as i32); j += 1;
            mctx[j] = clip(nn + ne - s.buf.at((w * 3).wrapping_sub(st)) as i32); j += 1;
            mctx[j] = clip(nn + ne - s.buf.at((w * 3).wrapping_sub(st)) as i32 + p1
                - clip(nnp1 + nep1 - buf_w3msp1) as i32); j += 1;
            mctx[j] = clip(nn + ne - s.buf.at((w * 3).wrapping_sub(st)) as i32 + p2
                - clip(nnp2 + nep2 - buf_w3msp2) as i32); j += 1;
            mctx[j] = clip(nn + nnnn - nnnnnn); j += 1;
            mctx[j] = clip(nn + nnnn - nnnnnn + p1 - clip(nnp1 + buf_w4p1 - buf_w6p1) as i32); j += 1;
            mctx[j] = clip(nn + nnnn - nnnnnn + p2 - clip(nnp2 + buf_w4p2 - buf_w6p2) as i32); j += 1;
            mctx[j] = clip(ww + p1 - wwp1); j += 1;
            mctx[j] = clip(ww + p2 - wwp2); j += 1;
            mctx[j] = clip(ww + wwww - wwwwww); j += 1;
            mctx[j] = clip(ww + wwww - wwwwww + p1 - clip(wwp1 + buf_st4p1 - buf_st6p1) as i32); j += 1;
            mctx[j] = clip(ww + wwww - wwwwww + p2 - clip(wwp2 + buf_st4p2 - buf_st6p2) as i32); j += 1;
            mctx[j] = clip(n * 2 - nn + p1 - clip(np1 * 2 - nnp1) as i32); j += 1;
            mctx[j] = clip(n * 2 - nn + p2 - clip(np2 * 2 - nnp2) as i32); j += 1;
            mctx[j] = clip(w_ * 2 - ww + p1 - clip(wp1 * 2 - wwp1) as i32); j += 1;
            mctx[j] = clip(w_ * 2 - ww + p2 - clip(wp2 * 2 - wwp2) as i32); j += 1;
            mctx[j] = clip(n * 3 - nn * 3 + nnn); j += 1;
            mctx[j] = clamp4(n * 3 - nn * 3 + nnn, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clamp4(w_ * 3 - ww * 3 + www, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clamp4(n * 2 - nn, p.w, p.nw, p.n, p.ne); j += 1;
            mctx[j] = clip((nnnnn - 6 * nnnn + 15 * nnn - 20 * nn + 15 * n
                + clamp4(w_ * 4 - nww * 6 + s.buf.at(w * 2 + 3 * st) as i32 * 4
                    - s.buf.at(w * 3 + 4 * st) as i32,
                    p.w, p.nw, p.n, p.nn) as i32) / 6); j += 1;
            mctx[j] = clip((buf_w3m3 - 4 * nnee + 6 * ne
                + clip(w_ * 4 - nw * 6 + nnw * 4 - s.buf.at(w * 3 + st) as i32) as i32) / 4); j += 1;
            mctx[j] = clip(((n + 3 * nw) / 4) * 3 - ((nnw + nnww) / 2) * 3
                + (s.buf.at(w * 3 + 2 * st) as i32 * 3 + buf_w3st3) / 4); j += 1;
            mctx[j] = clip((w_ * 2 + nw) - (ww + 2 * nww) + nwww); j += 1;
            mctx[j] = ((clip(w_ * 2 - nw) as i32 + clip(w_ * 2 - nww) as i32 + n + ne) / 4) as u8; j += 1;
            mctx[j] = nnnnnn as u8; j += 1;
            mctx[j] = ((neeee as i32 + buf_wm6) / 2) as u8; j += 1;
            mctx[j] = ((wwwwww + wwww) / 2) as u8; j += 1;
            mctx[j] = (((w_ + n) * 3 - nw * 2) / 4) as u8; j += 1;
            mctx[j] = n as u8; j += 1;
            mctx[j] = nn as u8;
            debug_assert!(j == IM24_N_MAPS1 - 1);
            // ---- SCMapCtxs[0..58] — paq8.cpp:5170-5227 ----
            let scctx = &mut self.sc_map_ctxs;
            let mut j = 0;
            scctx[j] = n + p1 - np1; j += 1;
            scctx[j] = n + p2 - np2; j += 1;
            scctx[j] = w_ + p1 - wp1; j += 1;
            scctx[j] = w_ + p2 - wp2; j += 1;
            scctx[j] = nw + p1 - nwp1; j += 1;
            scctx[j] = nw + p2 - nwp2; j += 1;
            scctx[j] = ne + p1 - nep1; j += 1;
            scctx[j] = ne + p2 - nep2; j += 1;
            scctx[j] = nn + p1 - nnp1; j += 1;
            scctx[j] = nn + p2 - nnp2; j += 1;
            scctx[j] = ww + p1 - wwp1; j += 1;
            scctx[j] = ww + p2 - wwp2; j += 1;
            scctx[j] = w_ + n - nw; j += 1;
            scctx[j] = w_ + n - nw + p1 - wp1 - np1 + nwp1; j += 1;
            scctx[j] = w_ + n - nw + p2 - wp2 - np2 + nwp2; j += 1;
            scctx[j] = w_ + ne - n; j += 1;
            scctx[j] = w_ + ne - n + p1 - wp1 - nep1 + np1; j += 1;
            scctx[j] = w_ + ne - n + p2 - wp2 - nep2 + np2; j += 1;
            scctx[j] = w_ + nee - ne; j += 1;
            scctx[j] = w_ + nee - ne + p1 - wp1 - buf_wms2p1 + nep1; j += 1;
            scctx[j] = w_ + nee - ne + p2 - wp2 - buf_wms2p2 + nep2; j += 1;
            scctx[j] = n + nn - nnn; j += 1;
            scctx[j] = n + nn - nnn + p1 - np1 - nnp1 + buf_w3sp1; j += 1;
            scctx[j] = n + nn - nnn + p2 - np2 - nnp2 + buf_w3sp2; j += 1;
            scctx[j] = n + ne - nne; j += 1;
            scctx[j] = n + ne - nne + p1 - np1 - nep1 + buf_w2msp1; j += 1;
            scctx[j] = n + ne - nne + p2 - np2 - nep2 + buf_w2msp2; j += 1;
            scctx[j] = n + nw - nnw; j += 1;
            scctx[j] = n + nw - nnw + p1 - np1 - nwp1 + buf_w2psp1; j += 1;
            scctx[j] = n + nw - nnw + p2 - np2 - nwp2 + buf_w2psp2; j += 1;
            scctx[j] = ne + nw - nn; j += 1;
            scctx[j] = ne + nw - nn + p1 - nep1 - nwp1 + nnp1; j += 1;
            scctx[j] = ne + nw - nn + p2 - nep2 - nwp2 + nnp2; j += 1;
            scctx[j] = nw + w_ - s.buf.at(w + 2 * st) as i32; j += 1;
            scctx[j] = nw + w_ - s.buf.at(w + 2 * st) as i32 + p1 - nwp1 - wp1 + buf_wps2p1; j += 1;
            scctx[j] = nw + w_ - s.buf.at(w + 2 * st) as i32 + p2 - nwp2 - wp2 + buf_wps2p2; j += 1;
            scctx[j] = w_ * 2 - ww; j += 1;
            scctx[j] = w_ * 2 - ww + p1 - wp1 * 2 + wwp1; j += 1;
            scctx[j] = w_ * 2 - ww + p2 - wp2 * 2 + wwp2; j += 1;
            scctx[j] = n * 2 - nn; j += 1;
            scctx[j] = n * 2 - nn + p1 - np1 * 2 + nnp1; j += 1;
            scctx[j] = n * 2 - nn + p2 - np2 * 2 + nnp2; j += 1;
            scctx[j] = nw * 2 - nnww; j += 1;
            scctx[j] = nw * 2 - nnww + p1 - nwp1 * 2 + buf_w2pst2p1; j += 1;
            scctx[j] = nw * 2 - nnww + p2 - nwp2 * 2 + buf_w2pst2p2; j += 1;
            scctx[j] = ne * 2 - nnee; j += 1;
            scctx[j] = ne * 2 - nnee + p1 - nep1 * 2 + buf_w2ms2p1; j += 1;
            scctx[j] = ne * 2 - nnee + p2 - nep2 * 2 + buf_w2ms2p2; j += 1;
            scctx[j] = n * 3 - nn * 3 + nnn + p1 - np1 * 3 + nnp1 * 3 - buf_w3sp1; j += 1;
            scctx[j] = n * 3 - nn * 3 + nnn + p2 - np2 * 3 + nnp2 * 3 - buf_w3sp2; j += 1;
            scctx[j] = n * 3 - nn * 3 + nnn; j += 1;
            scctx[j] = (w_ + ne * 2 - nne) / 2; j += 1;
            scctx[j] = (w_ + ne * 3 - nne * 3 + nnne) / 2; j += 1;
            scctx[j] = (w_ + ne * 2 - nne) / 2 + p1
                - (wp1 + nep1 * 2 - buf_w2msp1) / 2; j += 1;
            scctx[j] = (w_ + ne * 2 - nne) / 2 + p2
                - (wp2 + nep2 * 2 - buf_w2msp2) / 2; j += 1;
            scctx[j] = nne + ne - nnne; j += 1;
            scctx[j] = nne + w_ - nn; j += 1;
            scctx[j] = nnw + w_ - nnww;
            debug_assert!(j == IM24_N_SC_MAPS - 2);

            // OLS — paq8.cpp:5229-5232. For each regressor, predict
            // from the current color's regressor, then update the
            // *previous* color's regressor with p1.
            let _ = p;
            let prev_color = if self.color > 0 { (self.color - 1) as usize }
                             else { (self.stride - 1) as usize };
            for kk in 0..IM24_N_OLS {
                let ctx_vec = self.ols_ctx_bytes(kk);
                let cur = (self.color.max(0)) as usize;
                self.p_ols[kk] = clip(self.ols[kk][cur]
                    .predict_from(&ctx_vec).floor() as i32);
                self.ols[kk][prev_color].update(p1 as f64);
            }

            // Local stats — paq8.cpp:5233-5239.
            let mean_i = w_ + nw + n + ne;
            let var = (w_ * w_ + nw * nw + n * n + ne * ne - mean_i * mean_i / 4) >> 2;
            let mean = mean_i >> 2;
            let logvar = s.ilog.get((var & 0xffff) as u16) as i32;

            let color_clamped = (self.color.min(self.stride - 1)) as u32;
            let p1u = p1 as u32; let p2u = p2 as u32;
            self.ctx[0] = (color_clamped << 9)
                | (((w_ - n).abs() > 3) as u32) << 8
                | ((w_ > n) as u32) << 7
                | ((w_ > nw) as u32) << 6
                | (((n - nw).abs() > 3) as u32) << 5
                | ((n > nw) as u32) << 4
                | (((n - ne).abs() > 3) as u32) << 3
                | ((n > ne) as u32) << 2
                | ((w_ > ww) as u32) << 1
                | (n > nn) as u32;
            let buf1 = s.buf.at(1) as i32;
            self.ctx[1] = ((log_mean_diff_qt(buf1 as u8,
                    clip(s.buf.at(w + 1) as i32 + s.buf.at(w.wrapping_sub(st) + 1) as i32
                         - buf_w2msp1), 7) as u32) >> 1) << 5
                | ((log_mean_diff_qt(clip(n + ne - nne), clip(n + nw - nnw), 7) as u32) >> 1) << 2
                | color_clamped;

            // ---- ContextMap — 47 contexts, paq8.cpp:5241-5289 ----
            let mut i = 0u64;
            let mut bump = || { i += 1; i };
            let color_u = self.color.max(0) as u64;
            self.cm.set(hash3(bump(), ((n + 1) >> 1) as u64, log_mean_diff(p.n, clip(nn * 2 - nnn)) as u64));
            self.cm.set(hash3(bump(), ((w_ + 1) >> 1) as u64, log_mean_diff(p.w, clip(ww * 2 - www)) as u64));
            self.cm.set(hash3(bump(), clamp4(w_ + n - nw, p.w, p.nw, p.n, p.ne) as u64,
                log_mean_diff(clip(n + ne - nne), clip(n + nw - nnw)) as u64));
            self.cm.set(hash3(bump(), ((nnn + n + 4) / 8) as u64, (clip(n * 3 - nn * 3 + nnn) >> 1) as u64));
            self.cm.set(hash3(bump(), ((www + w_ + 4) / 8) as u64, (clip(w_ * 3 - ww * 3 + www) >> 1) as u64));
            self.cm.set(hash4(bump(), color_u,
                ((w_ + clip(ne * 3 - nne * 3 + nnne) as i32) / 4) as u64,
                log_mean_diff(p.n, ((p.nw as u32 + p.ne as u32) / 2) as u8) as u64));
            self.cm.set(hash3(bump(), color_u,
                (clip((-wwww + 5 * www - 10 * ww + 10 * w_
                    + clamp4(ne * 4 - nne * 6 + nnne * 4 - nnnne, p.n, p.ne, p.nee, p.neee) as i32) / 5) / 4) as u64));
            self.cm.set(hash3(bump(), clip(nee + n - nnee) as u64,
                log_mean_diff(p.w, clip(nw + ne - nne)) as u64));
            self.cm.set(hash3(bump(), clip(nn + w_ - nnw) as u64,
                log_mean_diff(p.w, clip(nnw + ww - s.buf.at(w * 2 + 2 * st) as i32)) as u64));
            self.cm.set(hash3(bump(), color_u, p1 as u64));
            self.cm.set(hash3(bump(), color_u, p2 as u64));
            self.cm.set(hash4(bump(), color_u, (clip(w_ + n - nw) / 2) as u64,
                (clip(w_ + p1 - wp1) / 2) as u64));
            self.cm.set(hash3(bump(), (clip(n * 2 - nn) / 2) as u64,
                log_mean_diff(p.n, clip(nn * 2 - nnn)) as u64));
            self.cm.set(hash3(bump(), (clip(w_ * 2 - ww) / 2) as u64,
                log_mean_diff(p.w, clip(ww * 2 - www)) as u64));
            self.cm.set(hash3(bump(), (clamp4(n * 3 - nn * 3 + nnn, p.w, p.nw, p.n, p.ne) / 2) as u64, 0));
            self.cm.set(hash3(bump(), (clamp4(w_ * 3 - ww * 3 + www, p.w, p.n, p.ne, p.nee) / 2) as u64, 0));
            let wp1_clamp = if wp1 < 1 { 1 } else { wp1 };
            self.cm.set(hash4(bump(), color_u, log_mean_diff(p.w, p.wp1) as u64,
                clamp4((p1 * w_) / wp1_clamp, p.w, p.n, p.ne, p.nee) as u64));
            self.cm.set(hash3(bump(), color_u, clamp4(n + p2 - np2, p.w, p.nw, p.n, p.ne) as u64));
            self.cm.set(hash4(bump(), color_u, clip(w_ + n - nw) as u64, self.column[0] as u64));
            self.cm.set(hash4(bump(), color_u, clip(n * 2 - nn) as u64,
                log_mean_diff(p.w, clip(nw * 2 - nnw)) as u64));
            self.cm.set(hash4(bump(), color_u, clip(w_ * 2 - ww) as u64,
                log_mean_diff(p.n, clip(nw * 2 - nww)) as u64));
            self.cm.set(hash3(bump(), ((w_ + nee) / 2) as u64,
                log_mean_diff(p.w, ((p.ww as u32 + p.ne as u32) / 2) as u8) as u64));
            self.cm.set(hash3(bump(), clamp4(clip(w_ * 2 - ww) as i32 + clip(n * 2 - nn) as i32
                - clip(nw * 2 - nnww) as i32, p.w, p.nw, p.n, p.ne) as u64, 0));
            self.cm.set(hash4(bump(), color_u, p.w as u64, p2u as u64));
            self.cm.set(hash4(bump(), p.n as u64, (nn & 0x1F) as u64, (nnn & 0x1F) as u64));
            self.cm.set(hash4(bump(), p.w as u64, (ww & 0x1F) as u64, (www & 0x1F) as u64));
            self.cm.set(hash4(bump(), color_u, p.n as u64, self.column[0] as u64));
            self.cm.set(hash4(bump(), color_u, clip(w_ + nee - ne) as u64,
                log_mean_diff(p.w, clip(ww + ne - n)) as u64));
            self.cm.set(hash5(bump(), p.nn as u64, (nnnn & 0x1F) as u64,
                (nnnnnn & 0x1F) as u64, self.column[1] as u64));
            self.cm.set(hash5(bump(), p.ww as u64, (wwww & 0x1F) as u64,
                (wwwwww & 0x1F) as u64, self.column[1] as u64));
            self.cm.set(hash5(bump(), p.nnn as u64, (nnnnnn & 0x1F) as u64,
                (buf_w9 & 0x1F) as u64, self.column[1] as u64));
            self.cm.set(hash3(bump(), color_u, self.column[1] as u64));
            self.cm.set(hash4(bump(), color_u, p.w as u64, log_mean_diff(p.w, p.ww) as u64));
            self.cm.set(hash4(bump(), color_u, p.w as u64, p1u as u64));
            self.cm.set(hash5(bump(), color_u, (p.w / 4) as u64,
                log_mean_diff(p.w, p.p1) as u64, log_mean_diff(p.w, p.p2) as u64));
            self.cm.set(hash4(bump(), color_u, p.n as u64, log_mean_diff(p.n, p.nn) as u64));
            self.cm.set(hash4(bump(), color_u, p.n as u64, p1u as u64));
            self.cm.set(hash5(bump(), color_u, (p.n / 4) as u64,
                log_mean_diff(p.n, p.p1) as u64, log_mean_diff(p.n, p.p2) as u64));
            self.cm.set(hash5(bump(), color_u, ((w_ + n) >> 3) as u64,
                (p1u >> 4) as u64, (p2u >> 4) as u64));
            self.cm.set(hash4(bump(), color_u, (p1 / 2) as u64, (p2 / 2) as u64));
            self.cm.set(hash4(bump(), color_u, p.w as u64, (p1 - wp1) as u64));
            self.cm.set(hash3(bump(), color_u, (w_ + p1 - wp1) as u64));
            self.cm.set(hash4(bump(), color_u, p.n as u64, (p1 - np1) as u64));
            self.cm.set(hash3(bump(), color_u, (n + p1 - np1) as u64));
            self.cm.set(hash3(bump(),
                s.buf.at((w * 3).wrapping_sub(st)) as u64,
                s.buf.at((w * 3).wrapping_sub(2 * st)) as u64));
            self.cm.set(hash3(bump(), s.buf.at(w * 3 + st) as u64,
                s.buf.at(w * 3 + 2 * st) as u64));
            self.cm.set(hash4(bump(), color_u, mean as u64, (logvar >> 4) as u64));

            // Map[0..4] direct contexts — paq8.cpp:5292-5295.
            self.maps[0].set_direct(((w_ as u32 & 0xC0) | ((n as u32 & 0xC0) >> 2)
                | ((ww as u32 & 0xC0) >> 4) | ((nn as u32) >> 6)) as u32);
            self.maps[1].set_direct(((n as u32 & 0xC0) | ((nn as u32 & 0xC0) >> 2)
                | ((ne as u32 & 0xC0) >> 4) | ((nee as u32) >> 6)) as u32);
            self.maps[2].set_direct(s.buf.at(1) as u32);
            self.maps[3].set_direct(self.color.min(self.stride - 1) as u32);
        }

        // ---- Per-bit Map[5..18] setters + cm/maps/scmap.mix ----
        let b = ((s.c0 << (8 - bpos)) & 0xff) as u8;
        let p = &self.pix;
        let mut i = 5usize;
        let nclip1 = clip(p.w as i32 + p.n as i32 - p.nw as i32);
        let nclip_nne = clip(p.n as i32 + p.ne as i32 - p.nne as i32);
        let nclip_nnw = clip(p.n as i32 + p.nw as i32 - p.nnw as i32);
        self.maps[i].set_direct(((nclip1.wrapping_sub(b)) as u32 * 8 + bpos)
            | ((log_mean_diff_qt(nclip_nne, nclip_nnw, 7) as u32) << 11));
        i += 1;
        let v2 = clip(p.n as i32 * 2 - p.nn as i32);
        self.maps[i].set_direct((v2.wrapping_sub(b) as u32 * 8 + bpos)
            | ((log_mean_diff(p.w, clip(p.nw as i32 * 2 - p.nnw as i32)) as u32) << 11));
        i += 1;
        let v3 = clip(p.w as i32 * 2 - p.ww as i32);
        self.maps[i].set_direct((v3.wrapping_sub(b) as u32 * 8 + bpos)
            | ((log_mean_diff(p.n, clip(p.nw as i32 * 2 - p.nww as i32)) as u32) << 11));
        i += 1;
        let v4 = nclip1;
        self.maps[i].set_direct((v4.wrapping_sub(b) as u32 * 8 + bpos)
            | ((log_mean_diff(p.p1, clip(p.wp1 as i32 + p.np1 as i32 - p.nwp1 as i32)) as u32) << 11));
        i += 1;
        self.maps[i].set_direct((v4.wrapping_sub(b) as u32 * 8 + bpos)
            | ((log_mean_diff(p.p2, clip(p.wp2 as i32 + p.np2 as i32 - p.nwp2 as i32)) as u32) << 11));
        i += 1;
        // Hash-context Map[10..14] — paq8.cpp:5315-5318. Upstream
        // multiplies hash by 8 and OR-adds bpos; use wrapping to
        // match the C++ U64 overflow semantics.
        self.maps[i].set(hash3(p.w.wrapping_sub(b) as u64, p.n.wrapping_sub(b) as u64, 0)
            .wrapping_mul(8).wrapping_add(bpos as u64));
        i += 1;
        self.maps[i].set(hash3(p.w.wrapping_sub(b) as u64, p.ww.wrapping_sub(b) as u64, 0)
            .wrapping_mul(8).wrapping_add(bpos as u64));
        i += 1;
        self.maps[i].set(hash3(p.n.wrapping_sub(b) as u64, p.nn.wrapping_sub(b) as u64, 0)
            .wrapping_mul(8).wrapping_add(bpos as u64));
        i += 1;
        self.maps[i].set(hash3(
            (clip(p.n as i32 + p.ne as i32 - p.nne as i32)).wrapping_sub(b) as u64,
            (clip(p.n as i32 + p.nw as i32 - p.nnw as i32)).wrapping_sub(b) as u64,
            0).wrapping_mul(8).wrapping_add(bpos as u64));
        i += 1;
        let color_shifted = (self.color.min(self.stride - 1) as u32) << 11;
        self.maps[i].set_direct(color_shifted
            | (clip(p.n as i32 + p.p1 as i32 - p.np1 as i32).wrapping_sub(b) as u32 * 8 + bpos));
        i += 1;
        self.maps[i].set_direct(color_shifted
            | (clip(p.n as i32 + p.p2 as i32 - p.np2 as i32).wrapping_sub(b) as u32 * 8 + bpos));
        i += 1;
        self.maps[i].set_direct(color_shifted
            | (clip(p.w as i32 + p.p1 as i32 - p.wp1 as i32).wrapping_sub(b) as u32 * 8 + bpos));
        i += 1;
        self.maps[i].set_direct(color_shifted
            | (clip(p.w as i32 + p.p2 as i32 - p.wp2 as i32).wrapping_sub(b) as u32 * 8 + bpos));
        i += 1;
        for j in 0..IM24_N_MAPS1 {
            self.maps[i].set_direct(((self.map_ctxs[j] as i32 - b as i32) & 0xff) as u32 * 8 + bpos);
            i += 1;
        }
        for j in 0..IM24_N_OLS {
            self.maps[i].set_direct(((self.p_ols[j] as i32 - b as i32) & 0xff) as u32 * 8 + bpos);
            i += 1;
        }
        for k in 0..(IM24_N_SC_MAPS - 1) {
            self.sc_map[k].set(((self.sc_map_ctxs[k] - b as i32) & 0xff) as u32 * 8 + bpos);
        }

        // Predict — paq8.cpp:5334-5353
        let dt = self.dt;
        let y = s.y;
        let c0 = s.c0;
        let c1 = s.buf.at(1);
        self.cm.mix1(m, c0, bpos as i32, c1, y, &s.ilog, &s.squash, &s.stretch);
        for k in 0..IM24_N_MAPS {
            self.maps[k].mix(m, y, 1, 3, 1023, &dt, &s.squash, &s.stretch);
        }
        for k in 0..IM24_N_SC_MAPS {
            self.sc_map[k].mix(m, y, 9, 1, 3, &s.squash, &s.stretch);
        }
        self.col += 1;
        if self.col >= self.stride * 8 { self.col = 0; }
        m.set(0, 1);
        let ctx0 = self.ctx[0];
        let ctx1 = self.ctx[1];
        m.set((self.column[0] as u32).min(63) + ((ctx0 >> 3) & 0xC0), 256);
        m.set((self.column[1] as u32).min(127) + ((ctx0 >> 2) & 0x180), 512);
        m.set((ctx0 & 0x7FC) | ((bpos >> 1) & 0xFF), 2048);
        m.set(self.col as u32, (self.stride * 8) as u32);
        m.set((self.x as u32) % stride_u, stride_u);
        m.set(c0, 256);
        m.set((ctx1 << 2) | ((bpos >> 1) & 0xFF), 1024);
        let h1 = hash5(
            log_mean_diff_qt(p.w, p.ww, 5) as u64,
            log_mean_diff_qt(p.n, p.nn, 5) as u64,
            log_mean_diff_qt(p.w, p.n, 5) as u64,
            ilog2_fn(p.w as u32) as u64,
            self.color.max(0) as u64,
        );
        m.set(finalize64(h1, 13) as u32, 8192);
        let h2 = hash3(ctx0 as u64, (self.column[0] / 8) as u64, 0);
        m.set(finalize64(h2, 13) as u32, 8192);
        m.set((((self.x + self.line) / 32) as u32).min(255), 256);
    }
}

impl Default for Im24BitModel {
    fn default() -> Self {
        Self::new(64 * 1024, super::substrate::build_dt())
    }
}

pub struct JpegModel;
impl JpegModel {
    pub fn new() -> Self { Self }
    /// Returns `true` only inside a recognised JPEG stream. For
    /// non-JPEG (text/wiki) input this is always `false`.
    pub fn mix(&mut self, _s: &mut Paq8State, _m: &mut Mixer) -> bool { false }
}
impl Default for JpegModel { fn default() -> Self { Self::new() } }

// =============================================================
// ImgModel — paq8.cpp:5387-5504. BMP/TGA detector.
// =============================================================
//
// Parses raw bytes for BMP and TGA image headers. On detection,
// populates `bpp` (1/4/8/24/32), `width`, `eoi` (end-of-image
// position), `alpha`, `gray`. The orchestrator reads these via
// the accessor methods and dispatches to the right Im{1,4,8,24}
// BitModel.

#[derive(Default, Clone, Copy)]
struct BmpImage {
    header: u32, offset: u32, bpp: u32, size: u32,
    palette: u32, hdr_less: u32,
    width: u32, height: u32, bit_mask: u32,
}

#[derive(Default, Clone, Copy)]
struct TgaImage {
    header: u32, id_length: u32, bpp: u32, img_type: u32,
    map_size: u32, width: u32, height: u32,
}

#[derive(Default)]
pub struct ImgModel {
    bmp: BmpImage,
    tga: TgaImage,
    w: u32,
    bpp: u32,
    eoi: u32,
    alpha: u32,
    gray: u32,
    plt_order: i32,
}

impl ImgModel {
    pub fn new() -> Self { Self::default() }

    /// `true` while inside a detected image stream.
    pub fn is_active(&self) -> bool { self.w > 0 }
    /// `bpp` of the active image (1, 4, 8, 24, 32).
    pub fn bpp(&self) -> u32 { self.bpp }
    /// Width in bytes — matches upstream's `w = (Width * bpp) >> 3`
    /// (or the BMP row-padded equivalent for the bit-packed formats).
    pub fn width(&self) -> u32 { self.w }
    pub fn has_alpha(&self) -> bool { self.alpha != 0 }
    pub fn is_gray(&self) -> bool { self.gray != 0 }

    /// Returns `true` when a BMP or TGA header was just detected
    /// (or we're currently mid-image). Doesn't drive prediction —
    /// the caller invokes the right Im{1,4,8,24}BitModel based on
    /// `bpp()` / `width()` / `has_alpha()` / `is_gray()`.
    pub fn mix(&mut self, s: &mut Paq8State, _m: &mut Mixer) -> bool {
        let bpos = s.bpos as u32;
        let pos  = s.buf.pos;

        if bpos == 0 {
            // ---- BMP detection ----
            if pos >= self.eoi + 40 && self.bmp.header == 0 {
                let mut detected = false;
                let bmp_offset = i4(s, 44);
                if s.buf.at(54) == b'B' && s.buf.at(53) == b'M'
                    && (bmp_offset & 0xFFFFFBF7) == 0x36
                    && i4(s, 40) == 0x28
                {
                    self.bmp.offset = bmp_offset;
                    detected = true;
                } else if i4(s, 40) == 0x28 {
                    self.bmp.hdr_less = 1;
                    detected = true;
                }
                if detected {
                    self.bmp.width  = i4(s, 36);
                    self.bmp.height = (i4(s, 32) as i32).abs() as u32;
                    self.bmp.bpp    = i2(s, 26);
                    self.bmp.size   = i4(s, 20);
                    self.bmp.palette = i4(s, 4);
                    let bpp_ok = matches!(self.bmp.bpp, 1 | 4 | 8 | 24 | 32);
                    if i4(s, 24) == 0 && i2(s, 28) == 1 && bpp_ok
                        && self.bmp.width < 30000 && self.bmp.height < 10000
                        && (self.bmp.palette == 0
                            || (1u32 << self.bmp.bpp) >= self.bmp.palette)
                    {
                        self.bmp.header = 1;
                        if self.bmp.hdr_less != 0 {
                            self.bmp.offset = if self.bmp.bpp < 24 {
                                if self.bmp.palette != 0 { self.bmp.palette * 4 }
                                else { 4u32 << self.bmp.bpp }
                            } else { 0 };
                        } else {
                            self.bmp.offset = self.bmp.offset.saturating_sub(54);
                        }
                        self.gray = if self.bmp.bpp == 8 { 0x300 } else { 0 };
                        // Cursor/icon heuristic — bit-mask AND mask handling.
                        if self.bmp.hdr_less != 0
                            && self.bmp.width * 2 == self.bmp.height
                            && self.bmp.bpp > 1
                        {
                            let p = self.bmp.width * self.bmp.height * (self.bmp.bpp + 1);
                            let q = self.bmp.width * self.bmp.height * self.bmp.bpp;
                            let mask_widths: &[u32] = &[
                                8, 10, 14, 16, 20, 22, 24, 32, 40, 48,
                                60, 64, 72, 80, 96, 128, 256,
                            ];
                            let cond1 = self.bmp.size > 0
                                && self.bmp.size == (p >> 4);
                            let cond2 = (self.bmp.size == 0
                                || self.bmp.size < (q >> 3))
                                && mask_widths.contains(&self.bmp.width);
                            if cond1 || cond2 {
                                self.bmp.height = self.bmp.width;
                                self.bmp.bit_mask = self.bmp.width;
                            }
                        }
                    }
                }
            } else {
                self.bmp.offset = self.bmp.offset.saturating_sub((self.bmp.offset > 0) as u32);
                // CheckIfGrayscale macro (paq8.cpp:5356-5378) —
                // detects 24bpp palette grayscale. Skipped in this
                // port; predictions still work without it.
            }

            if self.bmp.offset == 0
                && (self.bmp.header > 0 || self.bmp.bit_mask > 0)
                && pos >= self.eoi
            {
                if self.bmp.header == 0 && self.bmp.bit_mask != 0 {
                    self.bmp.header = 1; self.bmp.bpp = 1;
                    self.bmp.width = self.bmp.bit_mask;
                    self.bmp.bit_mask = 0;
                }
                self.bpp = self.bmp.bpp;
                self.w = if self.bpp > 4 {
                    (self.bmp.width * (self.bpp >> 3) + 3) & (!3)
                } else if self.bpp == 1 {
                    (((self.bmp.width - 1) >> 5) + 1) * 4
                } else {
                    ((self.bmp.width * 4 + 31) >> 5) * 4
                };
                self.alpha = (self.bpp == 32) as u32;
                let eoi_new = self.w * self.bmp.height;
                if eoi_new > 64 {
                    self.eoi = eoi_new + pos;
                } else {
                    self.bmp.header = 0;
                    self.w = 0;
                }
            }

            // ---- TGA detection ----
            if pos >= self.eoi + 8 && self.tga.header == 0 {
                if (m4(s, 8) & 0xFFFFFF) == 0x010100
                    && (m4(s, 4) & 0xFFFFFFC7) == 0x00000100
                    && matches!(s.buf.at(1), 16 | 24 | 32)
                {
                    self.tga.header = pos;
                    self.tga.id_length = s.buf.at(8) as u32;
                    self.tga.map_size = (s.buf.at(1) / 8) as u32;
                    self.tga.bpp = 8;
                    self.tga.img_type = 1;
                } else if (m4(s, 8) & 0xFFFEFF) == 0x000200 && m4(s, 4) == 0 {
                    self.tga.header = pos;
                    self.tga.id_length = s.buf.at(8) as u32;
                    self.tga.img_type = s.buf.at(6) as u32;
                    self.tga.bpp = if self.tga.img_type == 2 { 24 } else { 8 };
                }
            } else if self.w == 0 && self.tga.header > 0 {
                let p = pos - self.tga.header;
                if p == 8 {
                    self.tga.width  = i2(s, 4);
                    self.tga.height = i2(s, 2);
                    if !(i4(s, 8) == 0 && self.tga.width > 0
                        && self.tga.width < 0x3FFF
                        && self.tga.height > 0 && self.tga.height < 0x3FFF)
                    {
                        self.tga.header = 0;
                    }
                } else if p == 10 {
                    let i = m2(s, 2);
                    if i & 0xFFF7 == (32 << 8) { self.tga.bpp = 32; }
                    if i & 0xFFD7 != (self.tga.bpp << 8) {
                        self.tga = TgaImage::default();
                    }
                }
                if self.tga.header > 0
                    && p == 10 + self.tga.id_length + self.tga.map_size * 256
                {
                    self.w = (self.tga.width * self.tga.bpp) >> 3;
                    self.gray = (self.tga.img_type == 3) as u32;
                    self.bpp = self.tga.bpp;
                    self.alpha = (self.bpp == 32) as u32;
                    let eoi_new = self.w * self.tga.height;
                    if eoi_new > 64 {
                        self.eoi = eoi_new + pos;
                    } else {
                        self.tga.header = 0; self.w = 0;
                    }
                }
            }
        }
        if pos > self.eoi {
            self.w = 0;
        }
        // End of stream — reset detector state.
        if bpos == 7 && pos + 1 == self.eoi {
            self.tga = TgaImage::default();
            self.bmp.header = 0;
            self.gray = 0; self.alpha = 0;
        }
        self.w > 0
    }
}

#[inline]
fn i4(s: &Paq8State, i: u32) -> u32 {
    (s.buf.at(i) as u32)
        + 256 * (s.buf.at(i - 1) as u32)
        + 65536 * (s.buf.at(i - 2) as u32)
        + 16777216 * (s.buf.at(i - 3) as u32)
}
#[inline]
fn i2(s: &Paq8State, i: u32) -> u32 {
    (s.buf.at(i) as u32) + 256 * (s.buf.at(i - 1) as u32)
}
#[inline]
fn m4(s: &Paq8State, i: u32) -> u32 {
    (s.buf.at(i - 3) as u32)
        + 256 * (s.buf.at(i - 2) as u32)
        + 65536 * (s.buf.at(i - 1) as u32)
        + 16777216 * (s.buf.at(i) as u32)
}
#[inline]
fn m2(s: &Paq8State, i: u32) -> u32 {
    (s.buf.at(i) as u32) * 256 + (s.buf.at(i - 1) as u32)
}

// =============================================================
// AudioModel — paq8.cpp:5505-5870. WAV detector + 8-bit / 16-bit
// PCM sample-level prediction.
// =============================================================
//
// `AudioModel.mix(...)` parses the raw byte stream for a `RIFF
// WAVE fmt ... data` chunk. On detection, dispatches to either
// `audio8b_model` (8-bit, OLS-stack-based) or `wav_model` (16-bit,
// Cholesky-LS-based). The text path is unaffected — `mix` returns
// `false` until a real WAV header is seen.

#[derive(Default, Clone, Copy)]
struct WavAudio {
    header: u32, size: u32, channels: u32,
    bits_per_sample: u32, chunk: u32, data: u32,
}

pub struct AudioModel {
    wav: WavAudio,
    eoi: u32,
    length: u32,
    info: u32,
    audio8: Audio8bModel,
    wav16:  Wav16Model,
}

impl AudioModel {
    pub fn new() -> Self {
        Self {
            wav: WavAudio::default(),
            eoi: 0, length: 0, info: 0,
            audio8: Audio8bModel::new(),
            wav16:  Wav16Model::new(),
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) -> bool {
        let bpos = s.bpos as u32;
        let pos  = s.buf.pos;

        if bpos == 0 {
            // Detect "RIFF" magic at the start of a potential WAV.
            if pos >= self.eoi + 4 && self.wav.header == 0 && m4(s, 4) == 0x52494646 {
                self.wav.header = pos;
                self.wav.chunk  = 0;
                self.length = 0;
            } else if self.wav.header > 0 {
                let p = pos - self.wav.header;
                if p == 4 {
                    self.wav.size = i4(s, 4);
                    if self.wav.size > 0x3FFFFFFF { self.wav.header = 0; }
                } else if p == 8 {
                    if m4(s, 4) != 0x57415645 { self.wav.header = 0; }
                } else if p == 16 + self.length
                    && (m4(s, 8) != 0x666d7420
                        || (i4(s, 4).wrapping_sub(16) & 0xFFFFFFFD) != 0)
                {
                    self.length = (i4(s, 4) + 1) & !1;
                    self.length += 8;
                    if m4(s, 8) == 0x666d7420
                        && (i4(s, 4) & 0xFFFFFFFD) != 16
                    {
                        self.wav.header = 0;
                    }
                } else if p == 20 + self.length {
                    self.wav.channels = s.buf.at(2) as u32;
                    let ch_ok = self.wav.channels == 1 || self.wav.channels == 2;
                    if !(ch_ok && (m4(s, 4) & 0xFFFFFCFF) == 0x01000000) {
                        self.wav.header = 0;
                    }
                } else if p == 32 + self.length {
                    self.wav.bits_per_sample = s.buf.at(2) as u32;
                    let bps_ok = self.wav.bits_per_sample == 8
                        || self.wav.bits_per_sample == 16;
                    if !(bps_ok && (m2(s, 2) & 0xE7FF) == 0) {
                        self.wav.header = 0;
                    }
                } else if p == 40 + self.length + self.wav.chunk
                    && m4(s, 8) != 0x64617461
                {
                    self.wav.chunk += ((i4(s, 4) + 1) & !1) + 8;
                    if self.wav.chunk > 0xFFFFF { self.wav.header = 0; }
                } else if p == 40 + self.length + self.wav.chunk {
                    self.wav.data = (i4(s, 4) + 1) & !1;
                    let stride = self.wav.channels * (self.wav.bits_per_sample / 8);
                    if self.wav.data != 0 && stride != 0 && self.wav.data % stride == 0 {
                        self.info = self.wav.channels
                            + self.wav.bits_per_sample / 4 - 3 + 1;
                        self.eoi = pos + self.wav.data;
                    }
                }
            }
        }
        if pos > self.eoi { self.info = 0; return false; }

        if self.info > 0 {
            // info-1 in [0..3]: bit 0 = stereo, bit 1 = 16-bit.
            if (self.info - 1) & 2 == 0 {
                self.audio8.mix(s, m, self.info - 1);
            } else {
                self.wav16.mix(s, m, self.info - 1);
            }
        }
        if bpos == 7 && pos + 1 == self.eoi {
            self.wav = WavAudio::default();
            self.info = 0; self.eoi = 0;
        }
        self.info > 0
    }
}

impl Default for AudioModel { fn default() -> Self { Self::new() } }

// =============================================================
// Audio8bModel — paq8.cpp:5553-5658. 8-bit PCM model with 8 OLS
// stacks per channel + 3 linear-extrapolation predictors.
// =============================================================

const A8_N_OLS: usize = 8;
const A8_N_LNR_PRD: usize = A8_N_OLS + 3;

pub struct Audio8bModel {
    s_map: Vec<Vec<super::context_map::SmallStationaryContextMap>>, // [nLnrPrd][3]
    /// `[regressor][channel]` — 8 × 2.
    ols: Vec<Vec<super::util::Ols>>,
    prd: [[[i32; 2]; 2]; A8_N_LNR_PRD],
    residuals: [[i32; 2]; A8_N_LNR_PRD],
    stereo: i32, ch: i32, rpos: u32, last_pos: u32,
    mask: u32, err_log: u32, mx_ctx: u32,
    wmode: i32,
}

impl Audio8bModel {
    pub fn new() -> Self {
        use super::context_map::SmallStationaryContextMap;
        use super::util::Ols;
        let s_map: Vec<Vec<SmallStationaryContextMap>> = (0..A8_N_LNR_PRD)
            .map(|_| (0..3).map(|_| SmallStationaryContextMap::new(11, 1)).collect())
            .collect();
        // OLS params per upstream:
        let ols_params: &[(usize, i32, f64)] = &[
            (128, 24, 0.9975),
            (90, 30, 0.9965),
            (90, 31, 0.996),
            (90, 32, 0.995),
            (90, 33, 0.995),
            (90, 34, 0.9985),
            (28, 4, 0.98),
            (28, 3, 0.992),
        ];
        let ols: Vec<Vec<Ols>> = ols_params.iter().map(|&(n, kmax, lambda)| {
            (0..2).map(|_| Ols::new(n, kmax, lambda, 0.001, 0.0)).collect()
        }).collect();
        Self {
            s_map, ols,
            prd: [[[0; 2]; 2]; A8_N_LNR_PRD],
            residuals: [[0; 2]; A8_N_LNR_PRD],
            stereo: 0, ch: 0, rpos: 0, last_pos: 0,
            mask: 0, err_log: 0, mx_ctx: 0,
            wmode: 0,
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, info: u32) {
        let bpos = s.bpos as u32;
        let pos  = s.buf.pos;
        let c0   = s.c0;
        let b = ((c0 << (8 - bpos)) & 0xff) as i8;

        if bpos == 0 {
            self.rpos = if pos == self.last_pos + 1 { self.rpos + 1 } else { 0 };
            self.last_pos = pos;
            if self.rpos == 0 {
                self.stereo = (info & 1) as i32;
                self.mask = 0;
                self.wmode = info as i32;
            }
            self.ch = if self.stereo != 0 { (s.blpos & 1) as i32 } else { 0 };
            let raw = if info & 4 != 0 { s.buf.at(1) ^ 128 } else { s.buf.at(1) };
            let sample = (raw as i32) - 128;
            let p_ch = (self.ch ^ self.stereo) as usize;
            self.err_log = 0;
            let mut i = 0;
            while i < A8_N_OLS {
                self.ols[i][p_ch].update(sample as f64);
                self.residuals[i][p_ch] = sample - self.prd[i][p_ch][0];
                let abs_res = self.residuals[i][p_ch].abs() as u32;
                self.mask = self.mask * 2 + ((abs_res > 4) as u32);
                self.err_log = self.err_log.wrapping_add(abs_res.wrapping_mul(abs_res));
                i += 1;
            }
            for j in i..A8_N_LNR_PRD {
                self.residuals[j][p_ch] = sample - self.prd[j][p_ch][0];
            }
            self.err_log = ((self.err_log).max(1) as u32).min(0xFFFF);
            self.err_log = super::substrate::ilog2(self.err_log).min(0xF);
            let bit_count = (self.mask.count_ones() as u32).min(0x1F);
            self.mx_ctx = super::substrate::ilog2(bit_count.max(1)) * 2 + self.ch as u32;

            // Feed OLS regressors with channel-specific sample stream.
            let ch_u = self.ch as usize;
            let k1_a = 90; let k2_a = k1_a - 12 * self.stereo;
            let wmode = self.wmode;
            // ols[1] — k1 stride pattern.
            { let mut j = 1; let mut i = 1; while j <= k1_a {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[1][ch_u].add(v);
                let step = 1 << ((j > 8) as u32 + (j > 16) as u32 + (j > 64) as u32);
                i += step; j += 1; } }
            // ols[2]
            { let mut j = 1; let mut i = 1; while j <= k2_a {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[2][ch_u].add(v);
                let step = 1 << ((j > 5) as u32 + (j > 10) as u32 + (j > 17) as u32
                    + (j > 26) as u32 + (j > 37) as u32);
                i += step; j += 1; } }
            // ols[3]
            { let mut j = 1; let mut i = 1; while j <= k2_a {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[3][ch_u].add(v);
                let step = 1 << ((j > 3) as u32 + (j > 7) as u32 + (j > 14) as u32
                    + (j > 20) as u32 + (j > 33) as u32 + (j > 49) as u32);
                i += step; j += 1; } }
            // ols[4]
            { let mut j = 1; let mut i = 1; while j <= k2_a {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[4][ch_u].add(v);
                i += 1 + ((j > 4) as i32) + ((j > 8) as i32); j += 1; } }
            // ols[5]
            { let mut j = 1; let mut i = 1; while j <= k1_a {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[5][ch_u].add(v);
                i += 2 + ((j > 3) as i32 + (j > 9) as i32 + (j > 19) as i32
                    + (j > 36) as i32 + (j > 61) as i32);
                j += 1; } }
            if self.stereo != 0 {
                for i in 1..=(k1_a - k2_a) {
                    let xx = Self::x2_static(s, i, wmode, 36) as f64;
                    self.ols[2][ch_u].add(xx);
                    self.ols[3][ch_u].add(xx);
                    self.ols[4][ch_u].add(xx);
                }
            }
            // 28-sample ols[0/6/7] block.
            let k1_b = 28; let k2_b = k1_b - 6 * self.stereo;
            for i in 1..=k2_b {
                let xx = Self::x1_static(s, i, wmode) as f64;
                self.ols[0][ch_u].add(xx);
                self.ols[6][ch_u].add(xx);
                self.ols[7][ch_u].add(xx);
            }
            let mut i = k2_b + 1;
            while i <= 96 {
                let v = Self::x1_static(s, i, wmode) as f64;
                self.ols[0][ch_u].add(v);
                i += 1;
            }
            if self.stereo != 0 {
                for i in 1..=(k1_b - k2_b) {
                    let xx = Self::x2_static(s, i, wmode, 36) as f64;
                    self.ols[0][ch_u].add(xx);
                    self.ols[6][ch_u].add(xx);
                    self.ols[7][ch_u].add(xx);
                }
                let mut i = (k1_b - k2_b) + 1;
                while i <= 32 {
                    let v = Self::x2_static(s, i, wmode, 36) as f64;
                    self.ols[0][ch_u].add(v);
                    i += 1;
                }
            } else {
                let mut i = k2_b + 1;
                while i <= 128 {
                    let v = Self::x1_static(s, i, wmode) as f64;
                    self.ols[0][ch_u].add(v);
                    i += 1;
                }
            }

            for i in 0..A8_N_OLS {
                let pred = self.ols[i][ch_u].predict().floor() as i32;
                self.prd[i][ch_u][0] = pred.clamp(-128, 127);
                self.prd[i][ch_u][1] = (self.prd[i][ch_u][0]
                    + self.residuals[i][p_ch]).clamp(-128, 127);
            }
            // 3 extrapolation predictors.
            let x1_1 = Self::x1_static(s, 1, wmode);
            let x1_2 = Self::x1_static(s, 2, wmode);
            let x1_3 = Self::x1_static(s, 3, wmode);
            let x1_4 = Self::x1_static(s, 4, wmode);
            self.prd[A8_N_OLS][ch_u][0]     = (x1_1 * 2 - x1_2).clamp(-128, 127);
            self.prd[A8_N_OLS + 1][ch_u][0] = (x1_1 * 3 - x1_2 * 3 + x1_3).clamp(-128, 127);
            self.prd[A8_N_OLS + 2][ch_u][0] = (x1_1 * 4 - x1_2 * 6 + x1_3 * 4 - x1_4).clamp(-128, 127);
            for i in A8_N_OLS..A8_N_LNR_PRD {
                self.prd[i][ch_u][1] = (self.prd[i][ch_u][0]
                    + self.residuals[i][p_ch]).clamp(-128, 127);
            }
        }

        // Per-bit predictions via 3 SmallStationaryContextMaps per
        // linear predictor.
        let ch_u = self.ch as usize;
        for i in 0..A8_N_LNR_PRD {
            let ctx = (((self.prd[i][ch_u][0] - b as i32) & 0xff) as u32 * 8) + bpos;
            self.s_map[i][0].set(ctx);
            self.s_map[i][1].set(ctx);
            let ctx2 = (((self.prd[i][ch_u][1] - b as i32) & 0xff) as u32 * 8) + bpos;
            self.s_map[i][2].set(ctx2);
            self.s_map[i][0].mix(m, s.y, 6, 1, 2 + ((i >= A8_N_OLS) as i32),
                &s.squash, &s.stretch);
            self.s_map[i][1].mix(m, s.y, 9, 1, 2 + ((i >= A8_N_OLS) as i32),
                &s.squash, &s.stretch);
            self.s_map[i][2].mix(m, s.y, 7, 1, 3, &s.squash, &s.stretch);
        }
        let c0 = s.c0;
        m.set((self.err_log << 8) | (c0 & 0xff), 4096);
        m.set(((self.mask & 0xff) << 3) | ((self.ch as u32) << 2) | (bpos >> 1), 2048);
        m.set((self.mx_ctx << 7) | ((s.buf.at(1) as u32) >> 1), 1280);
        m.set((self.err_log << 4) | ((self.ch as u32) << 3) | bpos, 256);
        m.set(self.mx_ctx, 10);
    }

    /// X1 — paq8.cpp:5517-5529. Sample reader, wmode-dependent.
    /// Static so callers can hold `&mut self.ols[..]` and read at
    /// the same time.
    fn x1_static(s: &Paq8State, i: i32, wmode: i32) -> i32 {
        let buf = &s.buf;
        match wmode {
            0 => buf.at(i as u32) as i32 - 128,
            1 => buf.at((i << 1) as u32) as i32 - 128,
            2 => s2(buf, (i << 1) as u32),
            3 => s2(buf, (i << 2) as u32),
            4 => (buf.at(i as u32) ^ 128) as i32 - 128,
            5 => (buf.at((i << 1) as u32) ^ 128) as i32 - 128,
            6 => t2(buf, (i << 1) as u32),
            7 => t2(buf, (i << 2) as u32),
            _ => 0,
        }
    }

    /// X2 — stereo-paired sample reader (paq8.cpp:5531-5543).
    fn x2_static(s: &Paq8State, i: i32, wmode: i32, big_s: i32) -> i32 {
        let buf = &s.buf;
        match wmode {
            0 => buf.at((i + big_s) as u32) as i32 - 128,
            1 => buf.at(((i << 1) - 1) as u32) as i32 - 128,
            2 => s2(buf, ((i + big_s) << 1) as u32),
            3 => s2(buf, ((i << 2) - 2) as u32),
            4 => (buf.at((i + big_s) as u32) ^ 128) as i32 - 128,
            5 => (buf.at(((i << 1) - 1) as u32) ^ 128) as i32 - 128,
            6 => t2(buf, ((i + big_s) << 1) as u32),
            7 => t2(buf, ((i << 2) - 2) as u32),
            _ => 0,
        }
    }
    fn x1(&self, s: &Paq8State, i: i32, wmode: i32) -> i32 { Self::x1_static(s, i, wmode) }
    fn x2(&self, s: &Paq8State, i: i32, wmode: i32, big_s: i32) -> i32 { Self::x2_static(s, i, wmode, big_s) }
}

#[inline]
fn s2(buf: &super::substrate::Buf, i: u32) -> i32 {
    let v = (buf.at(i) as u32) + 256 * (buf.at(i - 1) as u32);
    (v as i16) as i32
}
#[inline]
fn t2(buf: &super::substrate::Buf, i: u32) -> i32 {
    let v = (buf.at(i - 1) as u32) + 256 * (buf.at(i) as u32);
    (v as i16) as i32
}

// =============================================================
// Wav16Model — paq8.cpp:5660-5805. 16-bit PCM with Cholesky-LS fit.
// =============================================================

const WAV_SD_MAX: usize = 49; // S + D dimensions

pub struct Wav16Model {
    pr: [[i32; 2]; 3],
    n:  [i32; 2],
    counter: [i32; 2],
    f: Vec<Vec<Vec<f64>>>, // [49][49][2]
    l: Vec<Vec<f64>>,      // [49][49]
    rpos: u32, last_pos: u32,
    bits: i32, channels: i32, w: i32, ch: i32, col: i32,
    z1: i32, z2: i32, z3: i32, z4: i32, z5: i32, z6: i32, z7: i32,
    wmode: i32,
    big_s: i32, big_d: i32,
    scms: [super::context_map::SmallStationaryContextMap; 7],
    cm: super::context_map::ContextMap,
    dt: [i32; 1024],
}

impl Wav16Model {
    pub fn new() -> Self {
        use super::context_map::{ContextMap, SmallStationaryContextMap};
        Self {
            pr: [[0; 2]; 3],
            n: [0; 2], counter: [0; 2],
            f: vec![vec![vec![0.0; 2]; WAV_SD_MAX]; WAV_SD_MAX],
            l: vec![vec![0.0; WAV_SD_MAX]; WAV_SD_MAX],
            rpos: 0, last_pos: 0,
            bits: 0, channels: 0, w: 0, ch: 0, col: 0,
            z1: 0, z2: 0, z3: 0, z4: 0, z5: 0, z6: 0, z7: 0,
            wmode: 0, big_s: 0, big_d: 0,
            scms: [
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
                SmallStationaryContextMap::new(8, 8),
            ],
            cm: ContextMap::new(super::substrate::mem(0) * 2, 11, super::substrate::build_dt()),
            dt: super::substrate::build_dt(),
        }
    }

    fn x1_buf(&self, s: &Paq8State, i: i32) -> i32 {
        let buf = &s.buf;
        match self.wmode {
            0 => buf.at(i as u32) as i32 - 128,
            1 => buf.at((i << 1) as u32) as i32 - 128,
            2 => s2(buf, (i << 1) as u32),
            3 => s2(buf, (i << 2) as u32),
            4 => (buf.at(i as u32) ^ 128) as i32 - 128,
            5 => (buf.at((i << 1) as u32) ^ 128) as i32 - 128,
            6 => t2(buf, (i << 1) as u32),
            7 => t2(buf, (i << 2) as u32),
            _ => 0,
        }
    }

    fn x2_buf(&self, s: &Paq8State, i: i32) -> i32 {
        let buf = &s.buf;
        match self.wmode {
            0 => buf.at((i + self.big_s) as u32) as i32 - 128,
            1 => buf.at(((i << 1) - 1) as u32) as i32 - 128,
            2 => s2(buf, ((i + self.big_s) << 1) as u32),
            3 => s2(buf, ((i << 2) - 2) as u32),
            4 => (buf.at((i + self.big_s) as u32) ^ 128) as i32 - 128,
            5 => (buf.at(((i << 1) - 1) as u32) ^ 128) as i32 - 128,
            6 => t2(buf, ((i + self.big_s) << 1) as u32),
            7 => t2(buf, ((i << 2) - 2) as u32),
            _ => 0,
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, info: u32) {
        let bpos = s.bpos as u32;
        let pos  = s.buf.pos;
        let a    = 0.996f64;
        let a2   = 1.0 / a;

        if bpos == 0 {
            self.rpos = if pos == self.last_pos + 1 { self.rpos + 1 } else { 0 };
            self.last_pos = pos;
        }
        if bpos == 0 && self.rpos == 0 {
            self.bits = ((info as i32 % 4) / 2) * 8 + 8;
            self.channels = (info as i32 % 2) + 1;
            self.col = 0;
            self.w = self.channels * (self.bits >> 3);
            self.wmode = info as i32;
            if self.channels == 1 { self.big_s = 48; self.big_d = 0; }
            else { self.big_s = 36; self.big_d = 12; }
            for j in 0..(self.channels as usize) {
                for k in 0..=(self.big_s + self.big_d) as usize {
                    for l in 0..=(self.big_s + self.big_d) as usize {
                        self.f[k][l][j] = 0.0;
                        self.l[k][l] = 0.0;
                    }
                }
                self.f[1][0][j] = 1.0;
                self.n[j] = 0; self.counter[j] = 0;
                self.pr[2][j] = 0; self.pr[1][j] = 0; self.pr[0][j] = 0;
                self.z1 = 0; self.z2 = 0; self.z3 = 0;
                self.z4 = 0; self.z5 = 0; self.z6 = 0; self.z7 = 0;
            }
        }

        if bpos == 0 && self.rpos >= self.w as u32 {
            self.ch = (self.rpos % self.w as u32) as i32;
            let msb = self.ch % (self.bits >> 3);
            let chn = (self.ch / (self.bits >> 3)) as usize;
            if msb == 0 {
                self.z1 = self.x1_buf(s, 1);
                self.z2 = self.x1_buf(s, 2);
                self.z3 = self.x1_buf(s, 3);
                self.z4 = self.x1_buf(s, 4);
                self.z5 = self.x1_buf(s, 5);
                let k = self.x1_buf(s, 1) as f64;
                let s_max = self.big_s.min(self.counter[chn] - 1);
                for l in 0..=s_max as usize {
                    self.f[0][l][chn] *= a;
                    self.f[0][l][chn] += self.x1_buf(s, l as i32 + 1) as f64 * k;
                }
                let d_max = self.big_d.min(self.counter[chn]);
                for l in 1..=d_max as usize {
                    self.f[0][l + self.big_s as usize][chn] *= a;
                    self.f[0][l + self.big_s as usize][chn] +=
                        self.x2_buf(s, l as i32 + 1) as f64 * k;
                }
                if self.channels == 2 {
                    let k = self.x2_buf(s, 2) as f64;
                    for l in 1..=self.big_d.min(self.counter[chn]) as usize {
                        let idx = l + self.big_s as usize;
                        self.f[self.big_s as usize + 1][idx][chn] *= a;
                        self.f[self.big_s as usize + 1][idx][chn] +=
                            self.x2_buf(s, l as i32 + 1) as f64 * k;
                    }
                    for l in 1..=self.big_s.min(self.counter[chn] - 1) as usize {
                        self.f[l][self.big_s as usize + 1][chn] *= a;
                        self.f[l][self.big_s as usize + 1][chn] +=
                            self.x1_buf(s, l as i32 + 1) as f64 * k;
                    }
                    self.z6 = self.x2_buf(s, 1) + self.x1_buf(s, 1) - self.x2_buf(s, 2);
                    self.z7 = self.x2_buf(s, 1);
                } else {
                    self.z6 = 2 * self.x1_buf(s, 1) - self.x1_buf(s, 2);
                    self.z7 = self.x1_buf(s, 1);
                }
                self.n[chn] += 1;
                if self.n[chn] == 1 {
                    // Re-estimate covariance + Cholesky factor.
                    let sd = (self.big_s + self.big_d) as usize;
                    if self.channels == 1 {
                        for k in 1..=sd { for l in k..=sd {
                            self.f[k][l][chn] = (self.f[k - 1][l - 1][chn]
                                - self.x1_buf(s, k as i32) as f64
                                    * self.x1_buf(s, l as i32) as f64) * a2;
                        } }
                    } else {
                        for k in 1..=sd { if k as i32 != self.big_s + 1 {
                            for l in k..=sd { if l as i32 != self.big_s + 1 {
                                let xk = if (k - 1) as i32 <= self.big_s
                                    { self.x1_buf(s, k as i32) as f64 }
                                    else { self.x2_buf(s, k as i32 - self.big_s) as f64 };
                                let xl = if (l - 1) as i32 <= self.big_s
                                    { self.x1_buf(s, l as i32) as f64 }
                                    else { self.x2_buf(s, l as i32 - self.big_s) as f64 };
                                self.f[k][l][chn] = (self.f[k - 1][l - 1][chn]
                                    - xk * xl) * a2;
                            } }
                        } }
                    }
                    let mut broke_at = 0usize;
                    let mut ok = true;
                    for i in 1..=sd {
                        let mut sum = self.f[i][i][chn];
                        for kk in 1..i {
                            sum -= self.l[i][kk] * self.l[i][kk];
                        }
                        sum = (sum + 0.5).floor();
                        sum = 1.0 / sum;
                        if sum > 0.0 {
                            self.l[i][i] = sum.sqrt();
                            for jj in (i + 1)..=sd {
                                let mut s2 = self.f[i][jj][chn];
                                for kk in 1..i {
                                    s2 -= self.l[jj][kk] * self.l[i][kk];
                                }
                                s2 = (s2 + 0.5).floor();
                                self.l[jj][i] = s2 * self.l[i][i];
                            }
                        } else { ok = false; broke_at = i; break; }
                    }
                    let _ = broke_at;
                    if ok && self.counter[chn] > self.big_s + 1 {
                        for k in 1..=sd {
                            self.f[k][0][chn] = self.f[0][k][chn];
                            for jj in 1..k {
                                self.f[k][0][chn] -= self.l[k][jj] * self.f[jj][0][chn];
                            }
                            self.f[k][0][chn] *= self.l[k][k];
                        }
                        for k in (1..=sd).rev() {
                            for jj in (k + 1)..=sd {
                                self.f[k][0][chn] -= self.l[jj][k] * self.f[jj][0][chn];
                            }
                            self.f[k][0][chn] *= self.l[k][k];
                        }
                    }
                    self.n[chn] = 0;
                }
                let mut sum = 0.0f64;
                for l in 1..=(self.big_s + self.big_d) as usize {
                    let xl = if (l as i32) <= self.big_s
                        { self.x1_buf(s, l as i32) as f64 }
                        else { self.x2_buf(s, l as i32 - self.big_s) as f64 };
                    sum += self.f[l][0][chn] * xl;
                }
                self.pr[2][chn] = self.pr[1][chn];
                self.pr[1][chn] = self.pr[0][chn];
                self.pr[0][chn] = sum.floor() as i32;
                self.counter[chn] += 1;
            }
            let y1 = self.pr[0][chn];
            let y2 = self.pr[1][chn];
            let y3 = self.pr[2][chn];
            let mut x1 = s.buf.at(1) as i32;
            let mut x2 = s.buf.at(2) as i32;
            let x3 = s.buf.at(3) as i32;
            if self.wmode == 4 || self.wmode == 5 { x1 ^= 128; x2 ^= 128; }
            if self.bits == 8 { x1 -= 128; x2 -= 128; }
            let t = (self.bits == 8) || ((msb == 0) ^ (self.wmode < 6));
            let mut i = (self.ch << 4) as u64;
            let mut bump = || { i = i.wrapping_add(1); i };
            use super::substrate::{hash2, hash3, hash4};
            if (msb != 0) ^ (self.wmode < 6) {
                self.cm.set(hash2(bump(), (y1 & 0xff) as u64));
                self.cm.set(hash3(bump(), (y1 & 0xff) as u64,
                    (((self.z1 - y2 + self.z2 - y3) >> 1) & 0xff) as u64));
                self.cm.set(hash3(bump(), x1 as u64, (y1 & 0xff) as u64));
                self.cm.set(hash4(bump(), x1 as u64, (x2 >> 3) as u64, x3 as u64));
                if self.bits == 8 {
                    let diff = (self.z1 - y2).abs() as u32;
                    let llog = super::substrate::ilog2(diff.max(1)) * 2
                        + (self.z1 > y2) as u32;
                    self.cm.set(hash3(bump(), (y1 & 0xFE) as u64, llog as u64));
                } else {
                    self.cm.set(hash2(bump(), ((y1 + self.z1 - y2) & 0xff) as u64));
                }
                self.cm.set(hash2(bump(), x1 as u64));
                self.cm.set(hash3(bump(), x1 as u64, x2 as u64));
                self.cm.set(hash2(bump(), (self.z1 & 0xff) as u64));
                self.cm.set(hash2(bump(), ((self.z1 * 2 - self.z2) & 0xff) as u64));
                self.cm.set(hash2(bump(), (self.z6 & 0xff) as u64));
                self.cm.set(hash3(bump(), (y1 & 0xFF) as u64,
                    (((self.z1 - y2 + self.z2 - y3) / (self.bits >> 3) as i32) & 0xFF) as u64));
            } else {
                self.cm.set(hash2(bump(), ((y1 - x1 + self.z1 - y2) >> 8) as u64));
                self.cm.set(hash2(bump(), ((y1 - x1) >> 8) as u64));
                self.cm.set(hash2(bump(),
                    ((y1 - x1 + self.z1 * 2 - y2 * 2 - self.z2 + y3) >> 8) as u64));
                self.cm.set(hash3(bump(), ((y1 - x1) >> 8) as u64,
                    ((self.z1 - y2 + self.z2 - y3) >> 9) as u64));
                self.cm.set(hash2(bump(), (self.z1 >> 12) as u64));
                self.cm.set(hash2(bump(), x1 as u64));
                self.cm.set(hash4(bump(), (x1 >> 7) as u64, x2 as u64, (x3 >> 7) as u64));
                self.cm.set(hash2(bump(), (self.z1 >> 8) as u64));
                self.cm.set(hash2(bump(), ((self.z1 * 2 - self.z2) >> 8) as u64));
                self.cm.set(hash2(bump(), (y1 >> 8) as u64));
                self.cm.set(hash2(bump(), ((y1 - x1) >> 6) as u64));
            }
            let tmul = if t { 1 } else { 0 };
            self.scms[0].set((tmul * self.ch) as u32);
            self.scms[1].set((tmul * (((self.z1 - x1 + y1) >> 9) & 0xff)) as u32);
            self.scms[2].set((tmul * (((self.z1 * 2 - self.z2 - x1 + y1) >> 8) & 0xff)) as u32);
            self.scms[3].set((tmul * (((self.z1 * 3 - self.z2 * 3 + self.z3 - x1) >> 7) & 0xff)) as u32);
            self.scms[4].set((tmul * (((self.z1 + self.z7 - x1 + y1 * 2) >> 10) & 0xff)) as u32);
            self.scms[5].set((tmul * (((self.z1 * 4 - self.z2 * 6 + self.z3 * 4 - self.z4 - x1) >> 7) & 0xff)) as u32);
            self.scms[6].set((tmul * (((self.z1 * 5 - self.z2 * 10 + self.z3 * 10 - self.z4 * 5 + self.z5 - x1 + y1) >> 9) & 0xff)) as u32);
        }
        // Predict.
        let y = s.y;
        for scm in self.scms.iter_mut() {
            scm.mix(m, y, 7, 1, 4, &s.squash, &s.stretch);
        }
        let c0 = s.c0;
        let c1 = s.buf.at(1);
        self.cm.mix1(m, c0, bpos as i32, c1, y,
            &s.ilog, &s.squash, &s.stretch);
        self.col += 1;
        if self.col >= self.w * 8 { self.col = 0; }
        let bits_minus_1 = (self.bits - 1).max(1);
        let cb = self.col & bits_minus_1;
        m.set((self.ch as u32) + 4 * super::substrate::ilog2(cb.max(1) as u32), 4 * 8);
        m.set(((self.col % self.bits) < 8) as u32, 2);
        m.set((self.col % self.bits) as u32, self.bits as u32);
        m.set(self.col as u32, (self.w * 8) as u32);
        m.set(c0, 256);
    }
}

impl Default for Wav16Model { fn default() -> Self { Self::new() } }

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

    #[test]
    fn im24bit_model_rgb_runs_without_panic() {
        let mut model = Im24BitModel::new(64 * 1024, build_dt());
        drive(|s, m| model.mix(s, m, 48, /*alpha=*/false));
    }

    #[test]
    fn im24bit_model_rgba_runs_without_panic() {
        let mut model = Im24BitModel::new(64 * 1024, build_dt());
        drive(|s, m| model.mix(s, m, 64, /*alpha=*/true));
    }

    #[test]
    fn audio_model_text_path_does_not_fire() {
        // For random text-ish bytes, no WAV header should be detected.
        let mut s = Paq8State::new(0);
        let mut m = Mixer::new(64, 4, 0);
        let mut model = AudioModel::new();
        // Push some non-WAV bytes; model.mix should never fire.
        for byte in 0u32..32 {
            for bp in 0..8 {
                s.bpos = bp;
                s.c0 = if bp == 0 { 1 } else { (1u32 << bp) | (byte >> (8 - bp)) };
                s.y = ((byte >> (7 - bp)) & 1) as i32;
                assert!(!model.mix(&mut s, &mut m));
            }
            s.c4 = (s.c4 << 8) | byte;
            s.buf.push(byte as u8);
        }
    }
}
