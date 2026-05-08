//! SSE (Secondary Symbol Estimation) probability smoother — port of
//! `mixer/sse.{h,cpp}` (originally written by Eugene Shelwien at
//! <https://encode.ru/threads/2515-mod_ppmd>).
//!
//! The CMIX SSE block is two stacked smoothers (`s6` + `mix1`,
//! followed by `s7` + `mix2`). Each takes the previous-stage
//! probability, looks it up in a `(prq, ffl, pc, j)`-keyed table of
//! `SSEi<7>` quantisers, runs a `Mixer`-driven blend with the
//! upstream probability, and feeds the smoothed result onward.
//!
//! The arrays are *large*: `s6 ≈ 25 M × 14 B = ~350 MB`,
//! `s7 ≈ 6.3 M × 14 B = ~88 MB`, plus the two `Mixer` arrays
//! (`mix1 ≈ 2.6 MB`, `mix2 ≈ 1.6 MB`). Construction touches every
//! cell once. Heap allocation for a single SSE instance is around
//! ~440 MB — fine on the development host but worth knowing
//! before you embed it in a test loop.

#![allow(dead_code)]

const SCALE_LOG: i32 = 15;
const SCALE: i32 = 1 << SCALE_LOG;
const H_SCALE: i32 = SCALE / 2;
const M_SCALE: i32 = SCALE - 1;
const SSE_QUANT: usize = 7;

const M_F0_C: i32     = 10240;
const M_F1_C: i32     = 7935;
const M_F2_C: i32     = 9592;
const M_SM6_WR_A: i32 = 0;
const M_SM6_WR_B: i32 = 106;
const M_SM6_MW: i32   = 0;
const M_SM6_C1: i32   = 8092;
const M_X1_W0: i32    = 7649; // (7648+1)
const M_X1_WR: i32    = 6202;
const M_F3_C: i32     = 8200;
const M_F4_C: i32     = 7677;
const M_SM7_WR_B: i32 = 127;
const M_SM7_MW: i32   = 8192;
const M_SM7_C1: i32   = 8202;
const M_X2_W0: i32    = 2561; // (2560+1)
const M_X2_WR: i32    = 8320;

const M_MIX1_VOLUME: usize = 1 * 4 * (1 << 8) * (1 << 3) * 79;       // 647_168
const M_MIX2_VOLUME: usize = 1 * 3 * (1 << 1) * (1 << 8) * 256;      // 393_216
const M_SM6X_VOLUME: usize = 1 * 3 * (1 << 7) * (1 << 8) * 256;      // 25_165_824
const M_SM7X_VOLUME: usize = 1 * 3 * (1 << 5) * (1 << 8) * 255;      // 6_266_880

// ---------- 256-byte mask tables (verbatim from upstream) ----------

const M_MX1_MASK0: [u8; 256] = [
    0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,31,32,32,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,
    47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,
    55,55,55,55,56,56,56,56,57,57,57,57,58,58,58,58,59,59,59,59,60,60,60,60,61,61,61,61,62,62,62,62,
    63,63,63,63,63,63,63,63,64,64,64,64,64,64,64,64,65,65,65,65,65,65,65,65,66,66,66,66,66,66,66,66,
    67,67,67,67,67,67,67,67,68,68,68,68,68,68,68,68,69,69,69,69,69,69,69,69,70,70,70,70,70,70,70,70,
    71,71,71,71,71,71,71,71,72,72,72,72,72,72,72,72,73,73,73,73,73,73,73,73,74,74,74,74,74,74,74,74,
    75,75,75,75,75,75,75,75,76,76,76,76,76,76,76,76,77,77,77,77,77,77,77,77,78,78,78,78,78,78,78,78,
];

const M_SM7_MASK0: [u8; 256] = [
    0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,
    63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,
    95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
    127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,
    159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
    191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,
    223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,
];

// ---------- Stretch / squash tables ------------------------------------

fn st_d(p: f64) -> f64 { ((1.0 - p) / p).ln() / std::f64::consts::LN_2 }
fn sq_d(p: f64) -> f64 { 1.0 / (1.0 + (p * std::f64::consts::LN_2).exp()) }

fn st_coef() -> f64 {
    (H_SCALE as f64 - 1.0) / (((SCALE - 1) as f64).ln() / std::f64::consts::LN_2)
}
fn sq_coef() -> f64 { 1.0 / st_coef() }

fn st_i(p: u32) -> u16 {
    let v = st_d(p as f64 / SCALE as f64) * st_coef() + H_SCALE as f64;
    v as u16
}
fn sq_i(p: u32) -> u16 {
    let v = sq_d((p as i32 - H_SCALE) as f64 * sq_coef()) * SCALE as f64;
    v as u16
}

/// Pair of stretch/squash lookup tables, populated as upstream does
/// in `Init_ST_SQ()`. Cached as a `&'static` after first
/// initialisation by `tables()`.
struct StSqTables {
    t_st: Vec<u16>,
    t_sq: Vec<u16>,
}

fn tables() -> &'static StSqTables {
    use std::sync::OnceLock;
    static TABLES: OnceLock<StSqTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let n = SCALE as usize;
        let mut t_st = vec![0u16; n];
        let mut t_sq = vec![0u16; n];

        for i in 1..n {
            t_sq[i] = sq_i(i as u32);
        }
        let mut x = 0usize;
        t_st[0] = 0;
        for i in 1..n {
            let s = st_i(i as u32);
            t_st[i] = s;
            if s != t_st[x] {
                let y = i - 1;
                t_sq[t_st[x] as usize] = ((x + y + 1) / 2) as u16;
                x = i;
            }
        }
        StSqTables { t_st, t_sq }
    })
}

// ---------- Helpers ----------------------------------------------------

fn extrap(p1: i32, c: i32) -> i32 {
    let v = (((p1 - H_SCALE) * c) >> 13) + H_SCALE;
    v.clamp(1, M_SCALE)
}

#[inline]
fn rdiv(x: i32, a: i32, d: i32) -> i32 {
    if x >= 0 { (x + a) >> d } else { -((-x + a) >> d) }
}

// ---------- SSEi<7> ----------------------------------------------------

#[derive(Clone, Copy)]
pub struct SSEi {
    p: [u16; SSE_QUANT],
}

#[derive(Clone, Copy, Default)]
struct UpdStr {
    p: i32,
    sw: i32,
    c1_idx: usize, // index within SSEi.p
}

impl SSEi {
    pub fn new() -> Self { Self { p: [0u16; SSE_QUANT] } }

    pub fn init(&mut self, wi: i32) {
        let scw = (SCALE - wi) / (SSE_QUANT as i32 - 1);
        let inc = wi / 2 + 8192;
        let mut p1 = inc;
        for i in 0..SSE_QUANT {
            self.p[i] = p1 as u16;
            p1 += scw;
        }
    }

    fn sse_pred(&self, ip: i32, x: &mut UpdStr) -> i32 {
        let sse_freq = (((SSE_QUANT as i32 - 1) * ip) >> SCALE_LOG) as usize;
        x.sw = ((SSE_QUANT as i32 - 1) * ip) & M_SCALE;
        x.c1_idx = sse_freq;
        let c0 = self.p[sse_freq] as i32;
        let c1 = self.p[sse_freq + 1] as i32;
        let mut f = (((SCALE - x.sw) * c0 + x.sw * c1) >> SCALE_LOG) - 8192;
        if f <= 0 { f = 1; }
        if f >= SCALE { f = M_SCALE; }
        x.p = f;
        f
    }

    fn sse_update(&mut self, c: u8, wr0: i32, x: &mut UpdStr) {
        x.p = (x.p * (SCALE - wr0)) >> SCALE_LOG;
        if c == 0 { x.p += wr0; }
        let c0 = self.p[x.c1_idx] as i32;
        let c1 = self.p[x.c1_idx + 1] as i32;
        let dc = c0 - c1;
        let sw_dc = (x.sw * dc + M_SCALE) >> SCALE_LOG;
        self.p[x.c1_idx]     = (x.p + sw_dc + 8192) as u16;
        self.p[x.c1_idx + 1] = (x.p - (dc - sw_dc) + 8192) as u16;
    }
}

// ---------- Mixer ------------------------------------------------------

#[derive(Clone, Copy)]
pub struct SseMixer {
    w: i32,
}

impl SseMixer {
    pub fn new() -> Self { Self { w: 0 } }
    pub fn init(&mut self, w0: i32) { self.w = w0 + H_SCALE; }
    pub fn mixup(&self, w: i32, s1: i32, s0: i32) -> i32 {
        let x = s1 + rdiv((w - H_SCALE) * (s0 - s1), 1 << (SCALE_LOG - 1), SCALE_LOG);
        if x > 0 { if x < SCALE { x } else { SCALE - 1 } } else { 1 }
    }
    pub fn update(&mut self, y: i32, p0: i32, p1: i32, wq: i32, pm: i32) {
        let py = SCALE - (y << SCALE_LOG);
        let e = py - pm;
        let mut d = rdiv(e * (p0 - p1), 1 << (SCALE_LOG - 1), SCALE_LOG);
        d = rdiv(d * wq, 1 << (SCALE_LOG - 1), SCALE_LOG);
        self.w += d;
    }
}

// ---------- Public SSE -------------------------------------------------

/// Two-stage SSE smoother. Public façade matching upstream
/// `class SSE`. Heap allocation is ~440 MB on `new()`.
pub struct SSE {
    s6: Vec<SSEi>,
    s7: Vec<SSEi>,
    x1: Vec<SseMixer>,
    x2: Vec<SseMixer>,
    su6: UpdStr,
    su7: UpdStr,
    sm6x: usize,
    mix1: usize,
    sm7x: usize,
    mix2: usize,
    mix1_s0: u32, mix1_s1: u32, mix1_p: u32,
    mix2_s0: u32, mix2_s1: u32, mix2_p: u32,
    j: u32,
    pc: u32,
    ffl: u32,
}

impl SSE {
    pub fn new() -> Self {
        let _ = tables(); // populate the lookup tables once.
        let mut s6 = vec![SSEi::new(); M_SM6X_VOLUME];
        for s in s6.iter_mut() { s.init(M_SM6_MW); }
        let mut s7 = vec![SSEi::new(); M_SM7X_VOLUME];
        for s in s7.iter_mut() { s.init(M_SM7_MW); }
        let mut x1 = vec![SseMixer::new(); M_MIX1_VOLUME];
        for m in x1.iter_mut() { m.init(M_X1_W0); }
        let mut x2 = vec![SseMixer::new(); M_MIX2_VOLUME];
        for m in x2.iter_mut() { m.init(M_X2_W0); }
        Self {
            s6, s7, x1, x2,
            su6: UpdStr::default(), su7: UpdStr::default(),
            sm6x: 0, mix1: 0, sm7x: 0, mix2: 0,
            mix1_s0: 0, mix1_s1: 0, mix1_p: 0,
            mix2_s0: 0, mix2_s1: 0, mix2_p: 0,
            j: 1, pc: 0, ffl: 0,
        }
    }

    fn estimate(&mut self, p: u32) -> u32 {
        let prq = p >> 11;
        let j   = self.j;
        let pc  = self.pc;
        let ffl = self.ffl;

        let prq_g0  = (prq > 0) as u32;
        let prq_g7  = (prq > 7) as u32;
        let prq_g14 = (prq > 14) as u32;

        // sm7x = ((prq>0 + prq>14) << 5) << 8 ... per upstream.
        self.sm7x = 0;
        self.sm7x = self.sm7x * 3 + (prq_g0 + prq_g14) as usize;
        self.sm7x = (self.sm7x << 5) + ((ffl) & 31) as usize;
        self.sm7x = (self.sm7x << 8) + ((pc) & 255) as usize;
        self.sm7x = (self.sm7x * 255) + M_SM7_MASK0[j as usize] as usize;

        self.mix2 = 0;
        self.mix2 = self.mix2 * 3 + (prq_g0 + prq_g14) as usize;
        self.mix2 = (self.mix2 << 1) + ((ffl) & 1) as usize;
        self.mix2 = (self.mix2 << 8) + ((pc) & 255) as usize;
        self.mix2 = (self.mix2 * 256) + (j) as usize;

        self.sm6x = 0;
        self.sm6x = self.sm6x * 3 + (prq_g0 + prq_g14) as usize;
        self.sm6x = (self.sm6x << 7) + ((ffl) & 127) as usize;
        self.sm6x = (self.sm6x << 8) + ((pc) & 255) as usize;
        self.sm6x = (self.sm6x * 256) + (j) as usize;

        self.mix1 = 0;
        self.mix1 = self.mix1 * 4 + (prq_g0 + prq_g7 + prq_g14) as usize;
        self.mix1 = (self.mix1 << 8) + ((ffl) & 255) as usize;
        self.mix1 = (self.mix1 << 3) + (((pc) >> 5) & 7) as usize;
        self.mix1 = (self.mix1 * 79) + M_MX1_MASK0[j as usize] as usize;

        let t = tables();
        let p0 = p as i32;

        let p1 = self.s6[self.sm6x].sse_pred(t.t_sq[extrap(t.t_st[p0 as usize] as i32, M_F0_C) as usize] as i32, &mut self.su6);
        let s0 = extrap(t.t_st[p0 as usize] as i32, M_F1_C);
        let s1 = extrap(t.t_st[p1 as usize] as i32, M_F2_C);
        self.mix1_s0 = s0 as u32;
        self.mix1_s1 = s1 as u32;
        let mut s2 = self.x1[self.mix1].mixup(self.x1[self.mix1].w, s0, s1);
        s2 = extrap(s2, M_SM6_C1);
        self.mix1_p = t.t_sq[s2 as usize] as u32;

        let p2 = self.s7[self.sm7x].sse_pred(t.t_sq[extrap(t.t_st[p0 as usize] as i32, M_F3_C) as usize] as i32, &mut self.su7);
        let s4 = extrap(t.t_st[p2 as usize] as i32, M_F4_C);
        self.mix2_s0 = s2 as u32;
        self.mix2_s1 = s4 as u32;
        let mut s5 = self.x2[self.mix2].mixup(self.x2[self.mix2].w, s2, s4);
        s5 = extrap(s5, M_SM7_C1);
        self.mix2_p = t.t_sq[s5 as usize] as u32;

        self.mix2_p
    }

    fn update(&mut self, bit: u32) {
        let su6_c1 = self.su6.c1_idx;
        let su7_c1 = self.su7.c1_idx;
        let mut su6 = self.su6;
        su6.c1_idx = su6_c1;
        self.s6[self.sm6x].sse_update(bit as u8, M_SM6_WR_A * 128 + M_SM6_WR_B, &mut su6);
        self.su6 = su6;

        self.x1[self.mix1].update(
            bit as i32, self.mix1_s0 as i32, self.mix1_s1 as i32,
            M_X1_WR, self.mix1_p as i32,
        );

        let mut su7 = self.su7;
        su7.c1_idx = su7_c1;
        self.s7[self.sm7x].sse_update(bit as u8, M_SM6_WR_A * 128 + M_SM7_WR_B, &mut su7);
        self.su7 = su7;

        self.x2[self.mix2].update(
            bit as i32, self.mix2_s0 as i32, self.mix2_s1 as i32,
            M_X2_WR, self.mix2_p as i32,
        );

        self.j = self.j.wrapping_mul(2).wrapping_add(bit);
        if self.j >= 256 {
            self.ffl = ((self.ffl * 2 + ((self.pc >= 0x40) as u32)) & 0xFF) as u32;
            self.pc = self.j & 0xFF;
            self.j = 1;
        }
    }

    /// Smooth `input ∈ [0, 1]` and return the smoothed probability.
    pub fn predict(&mut self, input: f32) -> f32 {
        let discrete = 1 + ((1.0 - input) * 32766.0) as u32;
        let estimate = self.estimate(discrete);
        1.0 - ((estimate as i32 - 1) as f32 / 32766.0)
    }

    /// Update internal state with the just-(en|de)coded bit.
    pub fn perceive(&mut self, bit: i32) {
        self.update(bit as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Allocate the full SSE instance and run a tiny round-trip
    /// (predict → perceive → predict). This is heavy: ~440 MB peak
    /// heap. Marked `#[ignore]` by default; run with `cargo test
    /// -- --ignored sse_smoke` if you want to exercise it.
    #[test] #[ignore]
    fn sse_smoke_full_alloc() {
        let mut sse = SSE::new();
        let p0 = sse.predict(0.5);
        sse.perceive(1);
        let p1 = sse.predict(0.5);
        assert!(p0.is_finite() && p0 >= 0.0 && p0 <= 1.0);
        assert!(p1.is_finite() && p1 >= 0.0 && p1 <= 1.0);
    }

    /// Cheap unit tests on the small primitives.
    #[test]
    fn ssei_init_distributes_evenly() {
        let mut s = SSEi::new();
        s.init(0);
        // P[0] = 0/2 + 8192 = 8192. P[6] = 8192 + 6 * (32768/6) = 8192 + 32766 = 40958.
        // Check monotonic and bounded.
        for i in 1..SSE_QUANT {
            assert!(s.p[i] >= s.p[i - 1], "P should be monotonic, got {:?}", s.p);
        }
    }

    #[test]
    fn st_sq_tables_are_inverses_in_middle_range() {
        let t = tables();
        for &p in &[1024u32, 4096, 8192, 16384, 24576, 30000] {
            let s = t.t_st[p as usize];
            let back = t.t_sq[s as usize];
            // Allow ~1% slack — the tables are coarse-grained.
            let diff = (back as i32 - p as i32).abs();
            assert!(diff < (SCALE / 100), "p={} → st={} → sq={} (diff {})",
                p, s, back, diff);
        }
    }

    #[test]
    fn mixer_init_centres_at_half() {
        let mut m = SseMixer::new();
        m.init(0);
        assert_eq!(m.w, H_SCALE);
    }
}
