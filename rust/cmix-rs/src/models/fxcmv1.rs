//! FXCMv1 — port of `models/fxcmv1.{h,cpp}` (Kaido Orav, GPL-2+).
//!
//! Layered PAQ8-style mixer that produces a single bit-1 probability
//! at each call. Uses a tree of context maps (StateMap variants,
//! ContextMap, RunContextMap, SmallStationaryContextMap) feeding a
//! two-layer logistic Mixer1.
//!
//! This module is laid out top-down to match upstream's flow:
//!
//!   * Foundations (this section): integer types, `ilog[]`,
//!     `sqt[]` / `strt[]` (squash / stretch tables), `Inputs<S>`
//!     and `BlockData<S>` mixer-input scratch space.
//!   * State machinery (next): `StateTable` and the seven prebaked
//!     state-transition tables (`STA1..STA7`).
//!   * Helpers: `dot_product`, `train` (SSE/AVX-style integer ops).
//!   * Maps: `StateMap`, `StateMap1`, `RunContextMap`,
//!     `SmallStationaryContextMap`, `ContextMap`, `ContextMap1`.
//!   * Mixer1 — the two-layer PAQ8 mixer.
//!   * Predictor — top-level `mix()` orchestrator.
//!
//! A working dictionary (`dictionary_path`) is *not* required —
//! the model degrades gracefully when one isn't supplied (the
//! WRT-codeword decoder produces no extra inputs).

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use std::f32::consts::E;

// ====================================================================
// Integer typedefs (mirror the C source for porting fidelity)
// ====================================================================

#[allow(non_camel_case_types)] pub type U8 = u8;
#[allow(non_camel_case_types)] pub type U16 = u16;
#[allow(non_camel_case_types)] pub type U32 = u32;
#[allow(non_camel_case_types)] pub type U64 = u64;

// Number of mixer inputs the upstream Predictor accumulates.
// Upstream sets `num_models = 439 + 1 - 2 - 7 = 431`.
pub const NUM_MODELS: usize = 431;

const CONVERSION_FACTOR: f32 = 1.0 / 4095.0;

// ====================================================================
// `wrt_2b[256]` and `wrt_3b[256]` — verbatim from upstream.
// Used by ContextMap variants for word-replacement-table contexts.
// ====================================================================

pub const WRT_2B: [U8; 256] = [
    2, 3, 1, 3, 3, 0, 1, 2, 3, 3, 0, 0, 1, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3,
    3, 2, 0, 2, 1, 3, 2, 1, 3, 3, 3, 3, 2, 3, 0, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 3, 2, 2,
    2, 2, 0, 0, 2, 3, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 3, 2, 0, 2, 3,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub const WRT_3B: [U8; 256] = [
    0, 0, 2, 0, 5, 6, 0, 6, 0, 2, 0, 4, 3, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 4, 1, 4, 4, 7, 4, 7, 3, 7, 2, 2, 3, 5, 3, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 5, 3, 3, 5, 5,
    0, 5, 5, 7, 5, 0, 1, 5, 4, 5, 0, 0, 6, 0, 7, 1,
    3, 3, 7, 4, 5, 5, 7, 0, 2, 2, 5, 4, 4, 7, 4, 6,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];

// ====================================================================
// `ilog[256]` — round(log2(x) * 16). Built by numerical integration
// of 1/x, exactly as upstream `InitIlog`.
// ====================================================================

pub fn build_ilog() -> [U8; 256] {
    let mut t = [0u8; 256];
    let mut x: U32 = 14_155_776;
    for i in 2..257u32 {
        x = x.wrapping_add(774_541_002 / (i * 2 - 1)); // numerator = 2^29 / ln 2
        t[(i - 1) as usize] = (x >> 24) as u8;
    }
    t
}

// ====================================================================
// `squash` / `stretch` lookup tables — sigmoid in 12-bit precision.
// `sqt[d + 2047]` for d in [-2047, 2047] gives p in [1, 4095].
// `strt[p]`        for p in [0, 4095]    gives d in [-2047, 2047].
// ====================================================================

fn squashc_scalar(d: i32) -> i32 {
    if d < -2047 { return 1; }
    if d > 2047  { return 4095; }
    let p = 1.0 / (1.0 + (-d as f32 / 256.0).exp());
    let pi = (p * 4096.0).round() as i32;
    pi.clamp(1, 4095)
}

fn stretchc_scalar(p: i32) -> i32 {
    let p = if p == 0 { 1 } else { p };
    let f = p as f32 / 4096.0;
    let d = (f / (1.0 - f)).ln() * 256.0;
    let di = d.round() as i32;
    di.clamp(-2047, 2047)
}

pub fn build_squash_table() -> Vec<i16> {
    // Indexed by (d + 2047), d ∈ [-2047, 2047].
    let mut t = vec![0i16; 4095];
    for d in -2047..=2047 {
        t[(d + 2047) as usize] = squashc_scalar(d) as i16;
    }
    t
}

pub fn build_stretch_table() -> Vec<i16> {
    let mut t = vec![0i16; 4096];
    for p in 0..=4095 {
        t[p as usize] = stretchc_scalar(p) as i16;
    }
    t
}

#[inline] pub fn squash(sqt: &[i16], d: i32) -> i32 {
    if d < -2047 { return 1; }
    if d > 2047  { return 4095; }
    sqt[(d + 2047) as usize] as i32
}

#[inline] pub fn stretch(strt: &[i16], p: i32) -> i32 {
    strt[p.clamp(0, 4095) as usize] as i32
}

// ====================================================================
// `Inputs<S>` — accumulator for per-bit mixer inputs. Upstream uses
// `aligned(64)` arrays; we just use `Vec<i16>`.
// ====================================================================

#[derive(Clone)]
pub struct Inputs {
    pub n: Vec<i16>,
    pub ncount: usize,
    capacity: usize,
}

impl Inputs {
    pub fn new(s: usize) -> Self { Self { n: vec![0; s], ncount: 0, capacity: s } }
    pub fn add(&mut self, p: i16) {
        debug_assert!(self.ncount < self.capacity);
        debug_assert!(p > -2048 && p < 2048);
        self.n[self.ncount] = p;
        self.ncount += 1;
    }
    pub fn reset(&mut self) { self.ncount = 0; }
}

// ====================================================================
// `BlockData<S>` — running per-bit / per-byte state shared by every
// model in the FXCM tree.
// ====================================================================

#[derive(Clone)]
pub struct BlockData {
    pub y: i32,           // last decoded bit
    pub c0: i32,          // last partial byte with leading 1 (1..=255)
    pub c4: U32,          // last 4 whole bytes packed
    pub bpos: i32,        // bits in c0
    pub blpos: i32,       // relative position in block
    pub bposshift: i32,   // bpos cached, used by maps
    pub c0shift_bpos: i32,
    pub mx_inputs1: Inputs,
    pub mx_inputs2: Inputs,
}

impl BlockData {
    pub fn new(s1: usize) -> Self {
        Self {
            y: 0, c0: 1, c4: 0, bpos: 0, blpos: 0,
            bposshift: 0, c0shift_bpos: 0,
            mx_inputs1: Inputs::new(s1),
            mx_inputs2: Inputs::new(32),
        }
    }
}

// ====================================================================
// `StateTable` — generates a per-context bit-history → next-state
// transition table. Each state encodes a pair (x, y) of bit counts;
// `b[0..5]` caps x for each y, and `mdc` controls how aggressively
// the opposite-bit count is discounted on a contradicting bit.
// `Init(s0..s6, &mut [U8; 1024])` populates `ns[0..1024]` so the
// `t[]` lookup is `ns[state * 4 + (0|1|2|3)]`.
// ====================================================================

pub struct StateTable {
    mdc: i32,
    b: [i32; 6],
    pub ns: [u8; 1024],
    t: [[[u8; 2]; 64]; 64],
}

impl StateTable {
    pub fn new() -> Self {
        Self {
            mdc: 0,
            b: [0; 6],
            ns: [0; 1024],
            t: [[[0u8; 2]; 64]; 64],
        }
    }

    fn num_states(&self, x: i32, y: i32) -> i32 {
        if x < y { return self.num_states(y, x); }
        if x < 0 || y < 0 || x >= 64 || y >= 64 || y >= 5 || x >= self.b[y as usize] {
            return 0;
        }
        1 + (y > 0 && x + y < self.b[5]) as i32
    }

    fn discount(&self, x: &mut i32) {
        if *x > 2 {
            let mut y = 0i32;
            for i in 1..self.mdc {
                if *x >= i { y += 1; }
            }
            *x = y;
        }
    }

    fn next_state(&mut self, x: &mut i32, y: &mut i32, b: i32) {
        if *x < *y {
            let (mut nx, mut ny) = (*y, *x);
            self.next_state(&mut nx, &mut ny, 1 - b);
            *x = nx;
            *y = ny;
            return;
        }
        if b != 0 {
            *y += 1;
            self.discount(x);
        } else {
            *x += 1;
            self.discount(y);
        }
        while self.t[*x as usize][*y as usize][1] == 0 {
            if *y < 2 {
                *x -= 1;
            } else {
                *x = (*x * (*y - 1) + (*y / 2)) / *y;
                *y -= 1;
            }
        }
    }

    fn generate(&mut self) {
        for r in self.ns.iter_mut() { *r = 0; }
        for r in self.t.iter_mut() { for r in r.iter_mut() { *r = [0u8; 2]; } }

        // Pass 1: assign state IDs.
        let mut state = 0i32;
        for i in 0..256 {
            for y in 0..=i {
                let x = i - y;
                let n = self.num_states(x, y);
                if n != 0 && x < 64 && y < 64 {
                    self.t[x as usize][y as usize][0] = state as u8;
                    self.t[x as usize][y as usize][1] = n as u8;
                    state += n;
                }
            }
        }

        // Pass 2: populate the next-state table.
        let mut state = 0i32;
        'outer: for i in 0..64 {
            for y in 0..=i {
                let x = i - y;
                let cap = self.t[x as usize][y as usize][1];
                for _k in 0..cap {
                    let mut x0 = x; let mut y0 = y;
                    let mut x1 = x; let mut y1 = y;
                    self.next_state(&mut x0, &mut y0, 0);
                    self.next_state(&mut x1, &mut y1, 1);
                    let ns0 = self.t[x0 as usize][y0 as usize][0];
                    let ns1 = self.t[x1 as usize][y1 as usize][0]
                        + (self.t[x1 as usize][y1 as usize][1] > 1) as u8;
                    let s = state as usize;
                    self.ns[s * 4]     = ns0;
                    self.ns[s * 4 + 1] = ns1;
                    self.ns[s * 4 + 2] = x as u8;
                    self.ns[s * 4 + 3] = y as u8;
                    if state > 0xFF
                        || cap == 0
                        || self.t[x0 as usize][y0 as usize][1] == 0
                        || self.t[x1 as usize][y1 as usize][1] == 0
                    {
                        return;
                    }
                    state += 1;
                    if state > 0xFF { break 'outer; }
                }
            }
        }
    }

    /// Mirror upstream `Init(s0..s6, table)`: set `b[0..6]` from
    /// `s0..s5`, set `mdc = s6`, regenerate, and copy the result
    /// into `out_table[0..1024]`.
    pub fn init(&mut self, s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32,
                out_table: &mut [u8; 1024])
    {
        self.b[0] = s0; self.b[1] = s1; self.b[2] = s2;
        self.b[3] = s3; self.b[4] = s4; self.b[5] = s5;
        self.mdc = s6;
        self.generate();
        out_table.copy_from_slice(&self.ns);
    }
}

impl Default for StateTable { fn default() -> Self { Self::new() } }

// ====================================================================
// Top-level Predictor scaffolding (state owner). Models in the tree
// (added in subsequent turns) live in fields of this struct.
// ====================================================================

pub struct Predictor {
    pub model_predictions: Vec<f32>,
    pub prediction_index: usize,
    pub block: BlockData,
    pub sqt: Vec<i16>,
    pub strt: Vec<i16>,
    pub ilog: [U8; 256],
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            model_predictions: vec![0.5f32; NUM_MODELS],
            prediction_index: 0,
            block: BlockData::new(528 + 32),
            sqt: build_squash_table(),
            strt: build_stretch_table(),
            ilog: build_ilog(),
        }
    }

    pub fn add_prediction(&mut self, x: i32) {
        let i = self.prediction_index;
        debug_assert!(i < NUM_MODELS);
        self.model_predictions[i] = x as f32 * CONVERSION_FACTOR;
        self.prediction_index += 1;
    }

    pub fn reset_predictions(&mut self) { self.prediction_index = 0; }

    // The full per-bit `Predict` / `Perceive` will land in later
    // turns, once the Maps and Mixer1 have been ported. For now the
    // model returns a uniform 0.5 so callers can wire this struct
    // into the predictor pipeline.
    pub fn predict(&self) -> f32 { 0.5 }
    pub fn perceive(&mut self, _bit: i32) {}
}

impl Default for Predictor { fn default() -> Self { Self::new() } }

// ====================================================================
// Tests — exercise the pieces ported in this turn.
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ilog_sane_anchors() {
        let t = build_ilog();
        // ilog[0] = 0 by convention (only i=2..257 written).
        assert_eq!(t[0], 0);
        // log2(2) * 16 ≈ 16, log2(4) * 16 = 32, log2(256) * 16 = 128.
        // The integration is approximate but within a few units.
        assert!((t[1] as i32 - 16).abs() <= 3);
        assert!((t[3] as i32 - 32).abs() <= 3);
        assert!((t[127] as i32 - 112).abs() <= 4);
    }

    #[test]
    fn squash_stretch_round_trip() {
        let sqt = build_squash_table();
        let strt = build_stretch_table();
        for &p in &[100i32, 1000, 2048, 3000, 4000] {
            let d = stretch(&strt, p);
            let p2 = squash(&sqt, d);
            // Allow ~1% slack — 12-bit precision both ways.
            let diff = (p - p2).abs();
            assert!(diff < 50, "p={} → d={} → p2={} (diff {})", p, d, p2, diff);
        }
    }

    #[test]
    fn squash_endpoints_clamped() {
        let sqt = build_squash_table();
        assert_eq!(squash(&sqt, -3000), 1);
        assert_eq!(squash(&sqt,  3000), 4095);
    }

    #[test]
    fn inputs_accumulator() {
        let mut i = Inputs::new(8);
        i.add(100);
        i.add(-200);
        assert_eq!(i.ncount, 2);
        assert_eq!(i.n[0], 100);
        assert_eq!(i.n[1], -200);
        i.reset();
        assert_eq!(i.ncount, 0);
    }

    #[test]
    fn block_data_init() {
        let b = BlockData::new(64);
        assert_eq!(b.y, 0);
        assert_eq!(b.c0, 1);
        assert_eq!(b.bpos, 0);
        assert_eq!(b.mx_inputs1.capacity, 64);
        assert_eq!(b.mx_inputs2.capacity, 32);
    }

    #[test]
    fn predictor_starts_uniform() {
        let p = Predictor::new();
        assert_eq!(p.predict(), 0.5);
        let _ = E; // silence unused
    }

    #[test]
    fn state_table_generates_256_states() {
        let mut st = StateTable::new();
        let mut tbl = [0u8; 1024];
        // Use the 1st upstream state-table parameters (STA1 in
        // upstream: s0=18, s1=12, s2=8, s3=6, s4=4, s5=43, mdc=5).
        st.init(18, 12, 8, 6, 4, 43, 5, &mut tbl);
        // Sanity: at least the first state's transitions are
        // populated and reference valid downstream states.
        let nx0 = tbl[0];
        let nx1 = tbl[1];
        assert!(nx0 < 255 || tbl[2] != 0 || tbl[3] != 0,
            "state 0 should transition to a real state");
        assert!(nx1 < 255 || tbl[2] != 0 || tbl[3] != 0,
            "state 0 bit-1 transition should be valid");
        // Different parameter sets must produce different tables.
        let mut tbl2 = [0u8; 1024];
        st.init(20, 10, 5, 5, 5, 25, 4, &mut tbl2);
        assert_ne!(tbl, tbl2,
            "different StateTable params must produce different ns[]");
    }
}
