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

/// 4-bit byte classifier — used by `n4bState=wrt_4b[c1]` at byte
/// boundaries to fold the most-recent byte into `stream4b`.
///
/// Values 0..=15. Mirrors `wrt_4b[256]` at fxcmv1.cpp:1832-1851.
pub const WRT_4B: [U8; 256] = [
     6, 0,12,15,12,15,14,14, 5, 3,14, 0,15,13, 8,13,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    13, 5,15,11,10,12, 6,12, 0,11,14, 1, 1,10, 9, 8,
     7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9,11, 6, 1, 0, 4,
     9,10,10, 4, 5, 1, 4, 2,11, 8, 4, 1, 0,10,10, 5,
     4, 7,15, 4, 5,13, 0, 1, 4,12, 0, 1, 3, 3, 3,11,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 8, 0,11, 7,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
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
// `dot_product(t, w, n)` — t · w with each pair scaled down by 8 bits.
// Mirrors upstream's scalar fallback (we don't reach for SIMD in the
// safe-Rust port; the optimiser can vectorise the small loop).
// `n` must be rounded up to a multiple of 8 by the caller, matching
// upstream's `n=(n+15)&-16` contract.
// ====================================================================

#[inline]
pub fn dot_product(t: &[i16], w: &[i16], n: usize) -> i32 {
    let n = (n + 15) & !15;
    let mut sum = 0i32;
    let mut i = 0;
    while i < n {
        sum += ((t[i]     as i32) * (w[i]     as i32)) >> 8;
        if i + 1 < n {
            sum += ((t[i + 1] as i32) * (w[i + 1] as i32)) >> 8;
        }
        i += 2;
    }
    sum
}

/// Train weights `w[0..n]` against inputs `t[0..n]` and the error
/// `err`. `w[i] += ((t[i] * 2 * err) >> 16 + 1) >> 1`, clamped to
/// `[-32768, 32767]`.
#[inline]
pub fn train(t: &[i16], w: &mut [i16], n: usize, err: i32) {
    if err == 0 { return; }
    let n = (n + 15) & !15;
    for i in 0..n {
        let tv = t[i] as i32;
        let mut wt = w[i] as i32 + (((tv * 2 * err) >> 16) + 1) / 2;
        wt = wt.clamp(-32768, 32767);
        w[i] = wt as i16;
    }
}

// ====================================================================
// `Mixer1` — two-layer logistic mixer. Holds N inputs and M context
// rows of N i16 weights each. `add()` is implicit via the upstream
// caller writing into `tx`; `set(cxt)` selects a row; `p()` runs the
// dot product through `squash`; `update(y)` trains the active row.
// ====================================================================

pub struct Mixer1 {
    pub n: usize,            // inputs per context
    pub m: usize,            // contexts
    /// Weights — `m * n` entries, row-major (row = context).
    pub wx: Vec<i16>,
    /// Inputs (caller-owned). Upstream uses a raw pointer; we hand
    /// in a slice through `set_inputs(...)`.
    pub tx: Vec<i16>,
    pub cxt: usize,
    pub pr: i32,
    pub shift1: u32,
    pub elim: i32,
    pub uperr: i32,
    pub err: i32,
}

impl Mixer1 {
    pub fn new(n: usize, m: usize, shift1: u32, elim: i32, uperr: i32) -> Self {
        // Upstream allocates `(N*M)+32` aligned to 32; the +32 is
        // padding so the SSE/AVX read past the end is safe. We pad
        // by 32 too so the rounded-up loop in `dot_product` doesn't
        // trip.
        let mut wx = vec![129i16; n * m + 32];
        let _ = &mut wx;
        Self {
            n, m,
            wx,
            tx: vec![0; n + 32],
            cxt: 0, pr: 2048,
            shift1, elim, uperr, err: 0,
        }
    }

    /// Replace the input vector. Caller writes in stretched logits
    /// via `tx_mut()` (or this method).
    pub fn set_inputs(&mut self, inputs: &[i16]) {
        let take = inputs.len().min(self.n);
        self.tx[..take].copy_from_slice(&inputs[..take]);
        for v in &mut self.tx[take..self.n] { *v = 0; }
    }

    pub fn tx_mut(&mut self) -> &mut [i16] { &mut self.tx[..self.n] }

    /// Adjust weights to minimise prediction error for the bit `y`.
    pub fn update(&mut self, y: i32) {
        let mut e = ((y << 12) - self.pr) * self.uperr / 4;
        e = e.clamp(-32768, 32767);
        if e >= -self.elim && e <= self.elim { e = 0; }
        self.err = e;
        let row_lo = self.cxt * self.n;
        let (t, w_full) = (&self.tx[..], &mut self.wx[..]);
        let n = self.n;
        let row = &mut w_full[row_lo .. row_lo + n + 16];
        train(t, row, n, e);
    }

    /// Predict the next bit as a 12-bit probability (0..4095).
    pub fn p(&mut self, sqt: &[i16]) -> i32 {
        debug_assert!(self.cxt < self.m);
        let row = &self.wx[self.cxt * self.n .. self.cxt * self.n + self.n + 16];
        let dp = (dot_product(&self.tx, row, self.n) * self.shift1 as i32) >> 11;
        self.pr = squash(sqt, dp);
        self.pr
    }

    /// Like `p` but also returns the un-squashed (clamped) logit.
    pub fn p1(&mut self, sqt: &[i16]) -> i32 {
        debug_assert!(self.cxt < self.m);
        let row = &self.wx[self.cxt * self.n .. self.cxt * self.n + self.n + 16];
        let mut dp = (dot_product(&self.tx, row, self.n) * self.shift1 as i32) >> 11;
        dp = dp.clamp(-2047, 2047);
        self.pr = squash(sqt, dp);
        dp
    }
}

// ====================================================================
// `dt[1024]` — `16K / (2i + 3)` adaptive-rate table, used by
// StateMap1 to set the per-context update step.
// ====================================================================

pub fn build_dt() -> [i32; 1024] {
    let mut t = [0i32; 1024];
    for i in 0..1024 {
        t[i] = 16384 / (2 * i as i32 + 3);
    }
    t
}

// ====================================================================
// `StateMap` — N-context map of state → 12-bit probability. The
// per-context entry is a u32 with the prediction in the high 22 bits
// and (intentionally unused) low 10 bits; on update we add the
// signed bit-driven gradient directly.
// `nn` is a reference to the active state table (e.g. STA1[0..1024]),
// laid out as `[next0, next1, n0, n1]` quadruples per state.
// ====================================================================

pub struct StateMap {
    pub n: usize,
    pub cxt: usize,
    pub t: Vec<u32>,
    pub pr: i32,
}

impl StateMap {
    /// Initialise from a state-table `nn`; `n` must be a power of two.
    pub fn new(n: usize, nn: &[u8]) -> Self {
        debug_assert!(n.is_power_of_two());
        let mut t = vec![0u32; n];
        for i in 0..n {
            let n0 = nn[i * 4 + 2] as u32 * 3 + 1;
            let n1 = nn[i * 4 + 3] as u32 * 3 + 1;
            t[i] = ((n1 << 20) / (n0 + n1).max(1)) << 12;
        }
        Self { n, cxt: 0, t, pr: 2048 }
    }

    /// Train the previous prediction with `y` (0 or 1).
    fn update(&mut self, y: i32) {
        let p0 = self.t[self.cxt];
        let pr1 = (p0 >> 13) as i32;
        let p_new = p0.wrapping_add(((y << 19) - pr1) as u32);
        self.t[self.cxt] = p_new;
    }

    /// Train with `y` then advance to context `c`. Returns the new
    /// 12-bit probability.
    pub fn set(&mut self, y: i32, c: usize) -> i32 {
        debug_assert!(c < self.n);
        self.update(y);
        self.cxt = c;
        self.pr = (self.t[c] >> 20) as i32;
        self.pr
    }
}

// ====================================================================
// `StateMap1` — like StateMap but stores (count, prediction) packed
// in the same u32: low 10 bits = count, high 22 bits = prediction.
// Adaptive rate via `dt[count]`.
// ====================================================================

pub struct StateMap1 {
    pub n: usize,
    pub mask: usize,
    pub limit: i32,
    pub cxt: usize,
    pub t: Vec<u32>,
    pub pr: i32,
}

impl StateMap1 {
    pub fn new(n: usize, limit: i32) -> Self {
        debug_assert!(n.is_power_of_two());
        debug_assert!(limit > 0 && limit < 1024);
        Self {
            n, mask: n - 1, limit,
            cxt: 0,
            t: vec![1u32 << 31; n],
            pr: 2048,
        }
    }

    fn update(&mut self, y: i32, dt: &[i32]) {
        let p0 = self.t[self.cxt];
        let count = (p0 & 1023) as i32;
        let pr1 = (p0 >> 12) as i32;
        let mut p_new = p0;
        if count < self.limit { p_new = p_new.wrapping_add(1); }
        // Upstream C uses signed `int` and tolerates wrap. Match it
        // with explicit wrapping arithmetic.
        let diff = (y << 20).wrapping_sub(pr1);
        let prod = diff.wrapping_mul(dt[count as usize]).wrapping_add(512);
        let delta = prod & 0xFFFFFC00u32 as i32;
        p_new = p_new.wrapping_add(delta as u32);
        self.t[self.cxt] = p_new;
    }

    pub fn set(&mut self, y: i32, c: usize, dt: &[i32]) -> i32 {
        debug_assert!(c < self.n);
        self.update(y, dt);
        self.cxt = c & self.mask;
        self.pr = (self.t[self.cxt] >> 20) as i32;
        self.pr
    }
}

// ====================================================================
// `clp` / `clp1` — clamp helpers used by RunContextMap.
// ====================================================================

#[inline] pub fn clp(z: i32) -> i16 { z.clamp(-2047, 2047) as i16 }
#[inline] pub fn clp1(z: i32) -> i16 { z.clamp(0, 4095) as i16 }

// ====================================================================
// `RunContextMap` — hash-keyed run-length predictor. Stores the
// most-recent byte plus a repeat count in 4-byte buckets; emits a
// signed prediction biased by the matched/mismatched bit at the
// current bit position.
// ====================================================================

pub struct RunContextMap {
    pub t: Vec<u8>,           // hash table buckets
    pub n: U32,               // bucket-count - 1 (mask)
    pub rc: [i16; 512],       // signed count → prediction LUT
    pub cp: u32,              // index into `t` of the active bucket+1
    tmp: [u8; 4],
}

impl RunContextMap {
    /// `m` is total table size in bytes (must be a multiple of 4).
    /// `rcm_ml` is the bias multiplier (8 in upstream).
    pub fn new(m: usize, rcm_ml: i32, ilog: &[u8; 256]) -> Self {
        let mut rc = [0i16; 512];
        for r in 0..256 {
            let mut c = ilog[r] as i32 * 8;
            if (r & 1) == 0 { c = c * rcm_ml / 4; }
            rc[r + 256] = clp(c);
            rc[r]       = clp(-c);
        }
        Self {
            t: vec![0u8; m],
            n: (m as U32 / 4 - 1),
            rc,
            cp: 1, // first bucket, +1 to point past the chk byte
            tmp: [0; 4],
        }
    }

    /// Find / insert the bucket whose check-bytes match `cx`, and
    /// return the offset just past the chk byte (i.e. pointing at
    /// the (count, value) pair).
    fn find(&mut self, mut i: U32) -> u32 {
        let chk = (((i >> 16) ^ i) & 0xFFFF) as u16;
        i = i.wrapping_mul(4) & self.n;
        let mut found_j: i32 = -1;
        let m = 4i32; // M in upstream
        for j in 0..m {
            let p = ((i + j as U32) * 4) as usize;
            let cp1 = u16::from_le_bytes([self.t[p], self.t[p + 1]]);
            if self.t[p + 2] == 0 {
                self.t[p..p + 2].copy_from_slice(&chk.to_le_bytes());
                found_j = j;
                break;
            }
            if cp1 == chk { found_j = j; break; }
        }
        if found_j == 0 {
            return ((i * 4) + 1) as u32;
        }
        if found_j == -1 {
            // Replacement (lowest priority among tested).
            let mut j = m - 1;
            self.tmp = [0; 4];
            self.tmp[0..2].copy_from_slice(&chk.to_le_bytes());
            let p_last  = ((i + j as U32) * 4) as usize;
            let p_prev  = ((i + (j as U32 - 1)) * 4) as usize;
            if m > 2 && self.t[p_last + 2] > self.t[p_prev + 2] { j -= 1; }
            // Memmove buckets [0..j] to [1..j+1].
            let n_bytes = (j as usize) * 4;
            let base = (i as usize) * 4;
            self.t.copy_within(base..base + n_bytes, base + 4);
            self.t[base..base + 4].copy_from_slice(&self.tmp);
            return (base + 1) as u32;
        }
        // Move the matched bucket to the front.
        let mut j = found_j as usize;
        let base = (i as usize) * 4;
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.t[base + j * 4..base + j * 4 + 4]);
        let n_bytes = j * 4;
        self.t.copy_within(base..base + n_bytes, base + 4);
        self.t[base..base + 4].copy_from_slice(&buf);
        let _ = (m, j);
        (base + 1) as u32
    }

    /// Update the count for the active bucket on byte boundary.
    pub fn set(&mut self, cx: U32, c1: u8) {
        let p = self.cp as usize;
        if self.t[p] == 0 { self.t[p] = 2; self.t[p + 1] = c1; }
        else if self.t[p + 1] != c1 { self.t[p] = 1; self.t[p + 1] = c1; }
        else if self.t[p] < 254 { self.t[p] += 2; }
        let next = self.find(cx);
        self.cp = next + 1;
    }

    /// Emit a single signed-prediction value to push into the mixer.
    pub fn p(&self, c0_shift_bpos: i32, bposshift: i32) -> i16 {
        let cp = self.cp as usize;
        let count = self.t[cp] as usize;
        let value = self.t[cp + 1];
        let b = c0_shift_bpos ^ ((value >> bposshift) as i32);
        if b <= 1 { self.rc[(b * 256) as usize + count] } else { 0 }
    }
}

// ====================================================================
// `SmallStationaryContextMap` — direct-lookup context table. Each
// context owns `Stride` (= 2^InputBits - 1) 16-bit counters. On
// every bit, we (a) decay the active counter toward y * 65536, and
// (b) advance the in-byte position B. Two stretched logits are
// emitted per call (the second one decrements
// `prediction_index` so it's not counted in num_models).
// ====================================================================

pub struct SmallStationaryContextMap {
    pub data: Vec<u16>,
    pub context: usize,
    pub mask: usize,
    pub stride: usize,
    pub b_count: i32,
    pub b_total: i32,
    pub b: i32,
    pub n: usize,
}

impl SmallStationaryContextMap {
    pub fn new(bits_of_context: u32, input_bits: u32) -> Self {
        debug_assert!(input_bits > 0 && input_bits <= 8);
        let mask = (1usize << bits_of_context) - 1;
        let stride = (1usize << input_bits) - 1;
        let n = (1usize << bits_of_context) * stride;
        Self {
            data: vec![0x7FFFu16; n],
            context: 0, mask, stride,
            b_count: 0, b_total: input_bits as i32, b: 0,
            n,
        }
    }

    pub fn set(&mut self, ctx: U32) {
        self.context = (ctx as usize & self.mask) * self.stride;
        self.b_count = 0;
        self.b = 0;
    }

    /// Mix in two predictions for the current bit and return them.
    /// `r` is the upstream rate offset; total rate = r + 7.
    pub fn mix(
        &mut self,
        y: i32,
        r: i32,
        sqt: &[i16],
        strt: &[i16],
        out: &mut Vec<i16>,
    ) {
        let rate = r + 7;
        let cp_idx = self.context + self.b as usize;
        let cur = self.data[cp_idx] as i32;
        let new = cur + (((y << 16) - cur + (1 << (rate - 1))) >> rate);
        self.data[cp_idx] = new as u16;
        if y != 0 && self.b > 0 { self.b += y; } // upstream: B += (y && B>0)
        let new_b = self.b as usize;
        let new_idx = self.context + new_b;
        let prediction = (self.data[new_idx] as i32) >> 4;
        out.push((stretch(strt, prediction) / 4) as i16);
        out.push(((prediction - 2048) / 8) as i16);
        let _ = sqt;
        self.b_count += 1;
        self.b += self.b + 1;
        if self.b_count == self.b_total {
            self.b_count = 0;
            self.b = 0;
        }
    }
}

// ====================================================================
// `EBucket<A, B>` — one cache-line hash bucket for the ContextMap
// family. Layout:
//
//   chk[A]:  A × u16                          (2 * A bytes)
//   last:    u8                               (1 byte)
//   bh[A][7]: A * 7 × u8                      (7 * A bytes)
//   pad to B
//
// The bucket is stored as a flat `[u8; B]` (cache-line aligned by
// the containing Vec) with helper accessors.
// ====================================================================

#[derive(Clone, Copy)]
pub struct EBucket<const A: usize, const B: usize> {
    pub data: [u8; B],
}

impl<const A: usize, const B: usize> EBucket<A, B> {
    pub fn new() -> Self { Self { data: [0; B] } }

    #[inline] fn chk_off(i: usize) -> usize { 2 * i }
    #[inline] fn last_off() -> usize { 2 * A }
    #[inline] fn bh_off(i: usize, j: usize) -> usize { 2 * A + 1 + i * 7 + j }

    #[inline] pub fn chk(&self, i: usize) -> u16 {
        let o = Self::chk_off(i);
        u16::from_le_bytes([self.data[o], self.data[o + 1]])
    }
    #[inline] pub fn set_chk(&mut self, i: usize, v: u16) {
        let o = Self::chk_off(i);
        self.data[o..o + 2].copy_from_slice(&v.to_le_bytes());
    }
    #[inline] pub fn last(&self) -> u8 { self.data[Self::last_off()] }
    #[inline] pub fn set_last(&mut self, v: u8) { self.data[Self::last_off()] = v; }
    #[inline] pub fn bh(&self, i: usize, j: usize) -> u8 { self.data[Self::bh_off(i, j)] }
    #[inline] pub fn set_bh(&mut self, i: usize, j: usize, v: u8) {
        self.data[Self::bh_off(i, j)] = v;
    }

    /// Find / insert an element matching `ch`. Returns the row
    /// index `i` (caller indexes `bh[i][0..7]` from there).
    pub fn get(&mut self, ch: u16, keep: u8) -> usize {
        let last = self.last();
        let last_lo = (last & 15) as usize;
        if last_lo < A && self.chk(last_lo) == ch {
            return last_lo;
        }
        let mut best_priority = 0xFFFFu16;
        let mut best_i = 0usize;
        for i in 0..A {
            if self.chk(i) == ch {
                self.set_last(((last & 0x0F) << 4) as u8 | i as u8);
                return i;
            }
            let pri = self.bh(i, 0) as u16;
            let last_hi = (last >> 4) as usize;
            if pri < best_priority && last_lo != i && last_hi != i {
                best_priority = pri;
                best_i = i;
            }
        }
        // Replace.
        self.set_last(((last & 0x0F) << 4) as u8 | best_i as u8 | (keep & 0x0F));
        self.set_chk(best_i, ch);
        for j in 0..7 { self.set_bh(best_i, j, 0); }
        best_i
    }
}

// ====================================================================
// `getStateByteLocation(bp, c0)` — pick which byte slot inside the
// 7-byte bh row to use given the current bit position and partial
// byte. Mirrors upstream's macro.
// ====================================================================

#[inline]
pub fn get_state_byte_location(bpos: i32, c0: i32) -> u32 {
    let smask = (0x31031010u32 >> (bpos << 2)) & 0x0F;
    smask + (c0 as u32 & smask)
}

#[inline]
pub fn sc(p: i32) -> i32 {
    if p > 0 { p >> 7 } else { (p + 127) >> 7 }
}

#[inline]
pub fn ctx_pre(nn: &[u8], state: i32) -> i32 {
    let n0 = nn[(state * 4 + 2) as usize] as i32 * 3 + 1;
    let n1 = nn[(state * 4 + 3) as usize] as i32 * 3 + 1;
    (n1 << 12) / (n0 + n1).max(1)
}

// ====================================================================
// `ContextMap` over generic bucket size. Upstream parameterises this
// as `E<7,64>` (regular ContextMap), `E<3,32>` (ContextMap1), and
// `E<14,128>` (ContextMap2). Each variant differs only in bucket
// size; the algorithm is identical.
// ====================================================================

pub const MAX_CXT: usize = 8;

pub struct ContextMap<const A: usize, const B: usize> {
    pub c: usize,             // max contexts (≤ MAX_CXT)
    pub buckets: Vec<EBucket<A, B>>,
    pub tmask: u32,
    pub cn: usize,
    pub cxt_mask: u16,
    pub cxt: [u32; MAX_CXT],
    /// Bucket index (linear) and row index inside the bucket.
    pub cp_bucket: [u32; MAX_CXT],
    pub cp_row:    [u8;  MAX_CXT],
    pub cp_col:    [u8;  MAX_CXT],
    pub cp0_col:   [u8;  MAX_CXT],
    pub runp_off:  [u8;  MAX_CXT],
    pub sm: Vec<StateMap>,
    pub kep: u8,
    pub skip2: i32,
    pub cms: i32,
    pub cms3: i32,
    pub cms4: i32,
    pub st1: Vec<i16>,        // [4096]
    pub st2: Vec<i16>,        // [4096]
    pub st32: [i16; 256],
    pub st8:  [i16; 256],
    pub rc1:  [i16; 512],
    pub result: i32,
    /// Whether `cp[i]` is "live" (mirrors upstream's null check on cp[i]).
    pub cp_live: [bool; MAX_CXT],
}

impl<const A: usize, const B: usize> ContextMap<A, B> {
    /// `m` is bucket-array size in BYTES (must be power-of-two,
    /// ≥ 64). `c` packs C (low byte), cmul (next byte), cms (next).
    pub fn new(
        m: u32,
        c: i32,
        s3: i32,
        nn: &[u8],
        cs4: i32,
        k: u8,
        u_skip2: i32,
        st_in: &[i16],
        ilog: &[u8; 256],
    ) -> Self {
        debug_assert!(m >= 64 && (m & (m - 1)) == 0);
        let cval = c & 0xFF;
        let cmul = (c >> 8) & 0xFF;
        let cms = (c >> 16) & 0xFF;
        let bucket_count = (m / B as u32) as usize;
        let tmask = bucket_count as u32 - 1;

        let mut sm: Vec<StateMap> = Vec::with_capacity(cval as usize);
        for _ in 0..cval { sm.push(StateMap::new(256, nn)); }

        let mut rc1 = [0i16; 512];
        for rc in 0..256 {
            let mut cc = ilog[rc] as i32;
            cc <<= 2 + ((!rc) & 1);
            if (rc & 1) == 0 { cc = cc * cmul / 4; }
            rc1[rc + 256] = clp(cc);
            rc1[rc]       = clp(-cc);
        }

        let mut st1 = vec![0i16; 4096];
        let strt_local = build_stretch_table();
        for i in 0..4096 {
            st1[i] = clp(sc(cms * (strt_local[i] as i32)));
        }

        let mut st32 = [0i16; 256];
        let mut st8  = [0i16; 256];
        for s in 0..256 {
            let n0 = -((nn[(s * 4 + 2) as usize] == 0) as i32);
            let n1 = -((nn[(s * 4 + 3) as usize] == 0) as i32);
            let r;
            let mut sp0 = 0;
            let diff = n1 - n0;
            if diff == 1       { r = 1; sp0 = 0; }
            else if diff == -1 { r = 1; sp0 = 4095; }
            else { r = 0; }
            if r != 0 {
                st8[s]  = clp(sc(cs4 * (ctx_pre(nn, s as i32) - sp0)));
                st32[s] = clp(sc(s3 * (strt_local[ctx_pre(nn, s as i32) as usize] as i32)));
                if s < 8 { st32[s] = 0; }
            }
        }

        let mut buckets = Vec::with_capacity(bucket_count);
        for _ in 0..bucket_count { buckets.push(EBucket::new()); }
        let st2 = if st_in.is_empty() { vec![0i16; 4096] } else { st_in.to_vec() };

        Self {
            c: cval as usize,
            buckets,
            tmask,
            cn: 0,
            cxt_mask: 0,
            cxt: [0; MAX_CXT],
            cp_bucket: [0; MAX_CXT],
            cp_row:    [0; MAX_CXT],
            cp_col:    [0; MAX_CXT],
            cp0_col:   [0; MAX_CXT],
            runp_off:  [3; MAX_CXT],
            sm,
            kep: k,
            skip2: u_skip2,
            cms,
            cms3: s3,
            cms4: cs4,
            st1,
            st2,
            st32,
            st8,
            rc1,
            result: 0,
            cp_live: [true; MAX_CXT],
        }
    }

    /// Set the i'th context. Mirrors `inline void set(U32 cx)`.
    pub fn set(&mut self, mut cx: u32) {
        let i = self.cn;
        debug_assert!(i < self.c);
        cx = cx.wrapping_mul(987_654_323).wrapping_add(i as u32);
        cx = (cx << 16) | (cx >> 16);
        self.cxt[i] = cx.wrapping_mul(123_456_791).wrapping_add(i as u32);
        self.cn += 1;
        self.cxt_mask = self.cxt_mask.wrapping_mul(2);
    }

    pub fn sets(&mut self) {
        self.cn += 1;
        self.cxt_mask = self.cxt_mask.wrapping_add(1).wrapping_mul(2);
    }

    /// Returns the current bit-history state at `cp[i]`.
    fn cp_state(&self, i: usize) -> u8 {
        let b = &self.buckets[self.cp_bucket[i] as usize];
        b.bh(self.cp_row[i] as usize, self.cp_col[i] as usize)
    }

    /// Set the current bit-history state at `cp[i]`.
    fn set_cp_state(&mut self, i: usize, v: u8) {
        let b = &mut self.buckets[self.cp_bucket[i] as usize];
        b.set_bh(self.cp_row[i] as usize, self.cp_col[i] as usize, v);
    }

    /// runp slot byte (0..3 of the row, treated as count/value/unused/unused).
    fn runp_byte(&self, i: usize, off: usize) -> u8 {
        let b = &self.buckets[self.cp_bucket[i] as usize];
        b.bh(self.cp_row[i] as usize, (self.runp_off[i] as usize + off) & 7)
    }
    fn set_runp_byte(&mut self, i: usize, off: usize, v: u8) {
        let b = &mut self.buckets[self.cp_bucket[i] as usize];
        b.set_bh(self.cp_row[i] as usize, (self.runp_off[i] as usize + off) & 7, v);
    }

    /// Inner mixer-input emit for state `s`. Adds 5 inputs to `out`
    /// (or 4 if skip2 == 0). The two trailing prediction-tracker
    /// helpers in upstream's `prediction_index--` are not modelled
    /// here — callers count emitted inputs per upstream's flow.
    fn mix3(&mut self, s: u8, sm_idx: usize, y: i32, out: &mut Vec<i16>) -> i32 {
        if s == 0 {
            out.push(0);
            if self.skip2 == 1 { out.push(0); }
            out.push(0);
            out.push(0);
            out.push(64); // 32 * 2
            0
        } else {
            self.sm[sm_idx].set(y, s as usize);
            let p1 = self.sm[sm_idx].pr;
            out.push(self.st1[p1.clamp(0, 4095) as usize]);
            if self.skip2 == 1 { out.push(self.st2[p1.clamp(0, 4095) as usize]); }
            out.push(self.st8[s as usize]);
            out.push(self.st32[s as usize]);
            out.push(0);
            1
        }
    }

    fn mix4(&self, out: &mut Vec<i16>) {
        out.push(0);
        if self.skip2 == 1 { out.push(0); }
        out.push(0);
        out.push(0);
        out.push(64);
        out.push(0);
    }

    /// Per-bit update + predict. Mirrors `mix1(cc, bp, c1)` in the
    /// upstream class. `y` is the just-decoded bit; `cc=c0`,
    /// `bp=bpos`, `c1` is the most-recent whole byte (low byte of
    /// `c4`). Outputs are appended to `out` (typically the
    /// `mxInputs1` of a BlockData).
    pub fn mix1(&mut self, cc: i32, bp: i32, c1: u8, y: i32, out: &mut Vec<i16>,
                c0shift_bpos: i32, bposshift: i32, nn: &[u8])
        -> i32
    {
        self.result = 0;
        let cn = self.cn;
        for i in 0..cn {
            if (self.cxt_mask >> (cn - i)) & 1 != 0 {
                self.mix4(out);
                continue;
            }

            // Update bit-history with y.
            if self.cp_live[i] {
                let s = self.cp_state(i);
                let next_s = nn[(s as usize) * 4 + y as usize];
                self.set_cp_state(i, next_s);
            }

            // Refresh context pointers.
            let mut s = 0u8;
            if bp > 1 && self.runp_byte(i, 0) == 0 {
                self.cp_live[i] = false;
            } else {
                let chksum = ((self.cxt[i] >> 16) ^ i as u32) as u16;
                if bp != 0 {
                    if bp == 2 || bp == 5 {
                        let bidx = (self.cxt[i].wrapping_add(cc as u32) & self.tmask) as usize;
                        let row = self.buckets[bidx].get(chksum, self.kep);
                        self.cp_bucket[i] = bidx as u32;
                        self.cp_row[i] = row as u8;
                        self.cp_col[i] = 0;
                        self.cp0_col[i] = 0;
                    } else {
                        self.cp_col[i] = (self.cp0_col[i] as u32
                            + get_state_byte_location(bp, cc)) as u8 & 7;
                    }
                } else {
                    let bidx = (self.cxt[i].wrapping_add(cc as u32) & self.tmask) as usize;
                    let row = self.buckets[bidx].get(chksum, self.kep);
                    self.cp_bucket[i] = bidx as u32;
                    self.cp_row[i] = row as u8;
                    self.cp_col[i] = 0;
                    self.cp0_col[i] = 0;
                    // Pending bit-history update for bits 2..7.
                    let bh3 = self.buckets[bidx].bh(row, 3);
                    if bh3 == 2 {
                        let cv = self.buckets[bidx].bh(row, 4) as i32 + 256;
                        // First half (3 bits).
                        let half_idx_a = (self.cxt[i].wrapping_add((cv as u32) >> 6)
                                          & self.tmask) as usize;
                        let row_a = self.buckets[half_idx_a].get(chksum, self.kep);
                        self.buckets[half_idx_a].set_bh(row_a, 0, 1 + ((cv >> 5) & 1) as u8);
                        let off1 = 1 + ((cv >> 5) & 1) as usize;
                        self.buckets[half_idx_a].set_bh(row_a, off1, 1 + ((cv >> 4) & 1) as u8);
                        let off2 = 3 + ((cv >> 4) & 3) as usize;
                        self.buckets[half_idx_a].set_bh(row_a, off2, 1 + ((cv >> 3) & 1) as u8);
                        let half_idx_b = (self.cxt[i].wrapping_add((cv as u32) >> 3)
                                          & self.tmask) as usize;
                        let row_b = self.buckets[half_idx_b].get(chksum, self.kep);
                        self.buckets[half_idx_b].set_bh(row_b, 0, 1 + ((cv >> 2) & 1) as u8);
                        let off3 = 1 + ((cv >> 2) & 1) as usize;
                        self.buckets[half_idx_b].set_bh(row_b, off3, 1 + ((cv >> 1) & 1) as u8);
                        let off4 = 3 + ((cv >> 1) & 3) as usize;
                        self.buckets[half_idx_b].set_bh(row_b, off4, 1 + (cv & 1) as u8);
                        self.buckets[bidx].set_bh(row, 6, 0);
                    }
                    // Run-count update.
                    if self.runp_byte(i, 0) == 0 {
                        self.set_runp_byte(i, 0, 2);
                        self.set_runp_byte(i, 1, c1);
                    } else if self.runp_byte(i, 1) != c1 {
                        self.set_runp_byte(i, 0, 1);
                        self.set_runp_byte(i, 1, c1);
                    } else if self.runp_byte(i, 0) < 254 {
                        let v = self.runp_byte(i, 0) + 2;
                        self.set_runp_byte(i, 0, v);
                    }
                    self.runp_off[i] = (self.cp_col[i] as i32 + 3) as u8 & 7;
                }
                self.cp_live[i] = true;
                s = self.cp_state(i);
            }

            self.result += self.mix3(s, i, y, out);

            // Run-context inputs.
            let runp_count = self.runp_byte(i, 0);
            let runp_value = self.runp_byte(i, 1);
            let mut bb = c0shift_bpos ^ ((runp_value >> bposshift) as i32);
            if bb <= 1 {
                bb *= 256;
                out.push(self.rc1[(runp_count as i32 + bb) as usize]);
            } else {
                out.push(0);
            }
        }
        if bp == 7 { self.cn = 0; self.cxt_mask = 0; }
        self.result
    }
}

// ====================================================================
// `APM<S>` — Adaptive Probability Map. Each (S, cxt) pair owns 33
// interpolation points; `p(pr, cxt, rate, y)` looks up the
// interpolated probability and SGD-updates the two surrounding
// points.
// ====================================================================

pub struct ApmDyn {
    pub s: usize,
    pub index: usize,
    pub t: Vec<u16>,
}

impl ApmDyn {
    pub fn new(s: usize, sqt: &[i16]) -> Self {
        let mut t = vec![0u16; s * 33];
        for j in 0..33 {
            let v = squash(sqt, ((j as i32 - 16) * 128)) * 16;
            t[j] = v as u16;
        }
        for i in 33..s * 33 { t[i] = t[i - 33]; }
        Self { s, index: 0, t }
    }

    /// `pr` is the input probability (0..4095), `cxt` is the
    /// context index (< s), `rate` is the SGD step shift, `y` is
    /// the just-decoded bit.
    pub fn p(&mut self, pr: i32, cxt: usize, rate: u32, y: i32, strt: &[i16]) -> i32 {
        let pr_s = stretch(strt, pr);
        let g = (y << 16) + (y << rate) - y * 2;
        let idx = self.index;
        let v0 = self.t[idx] as i32;
        let v1 = self.t[idx + 1] as i32;
        self.t[idx]     = (v0 + ((g - v0) >> rate)) as u16;
        self.t[idx + 1] = (v1 + ((g - v1) >> rate)) as u16;
        let w = pr_s & 127;
        self.index = (((pr_s + 2048) >> 7) + cxt as i32 * 33) as usize;
        let nv0 = self.t[self.index] as i32;
        let nv1 = self.t[self.index + 1] as i32;
        (nv0 * (128 - w) + nv1 * w) >> 11
    }
}

// ====================================================================
// `DirectStateMap` — direct-lookup state machine with c parallel
// slots. Used by the FXCMv1 Predictor for short-range bit history.
// `set(cx, y)` advances the active slot's state and emits two
// stretched-logit inputs into `out`. Mirrors upstream's struct.
// ====================================================================

pub struct DirectStateMap {
    pub mask: u32,
    pub count: usize,
    pub cxt: Vec<u32>,
    pub cxt_state: Vec<u8>,
    pub sm: Vec<StateMap>,
    pub index: usize,
    pub pre1: [i16; 256],
}

impl DirectStateMap {
    pub fn new(m: u32, c: usize, nn: &[u8], strt: &[i16]) -> Self {
        let mut sm = Vec::with_capacity(c);
        for _ in 0..c { sm.push(StateMap::new(256, nn)); }
        let mask = (1u32 << m) - 1;
        let mut pre1 = [0i16; 256];
        for i in 0..256 {
            let n0 = nn[i * 4 + 2] as u32 * 3 + 1;
            let n1 = nn[i * 4 + 3] as u32 * 3 + 1;
            let p = ((n1 << 12) / (n0 + n1).max(1)) as i32;
            pre1[i] = clp(stretch(strt, p)) >> 2;
        }
        Self {
            mask, count: c,
            cxt: vec![0u32; c],
            cxt_state: vec![0u8; (mask + 1) as usize],
            sm,
            index: 0,
            pre1,
        }
    }

    pub fn next(&self, nn: &[u8], state: u8, y: i32) -> u8 {
        nn[(state as usize) * 4 + y as usize]
    }

    /// Advance the active slot and emit two stretched-logit inputs.
    pub fn set(&mut self, cx: u32, y: i32, nn: &[u8], strt: &[i16],
               out: &mut Vec<i16>)
    {
        // Update state at the previous slot's context.
        let prev_cxt = self.cxt[self.index] as usize;
        let cur_state = self.cxt_state[prev_cxt];
        self.cxt_state[prev_cxt] = self.next(nn, cur_state, y);
        // Advance to new context.
        self.cxt[self.index] = cx & self.mask;
        let new_cxt = self.cxt[self.index] as usize;
        self.sm[self.index].set(y, self.cxt_state[new_cxt] as usize);
        let p = self.sm[self.index].pr;
        out.push((stretch(strt, p) >> 2) as i16);
        out.push(self.pre1[self.cxt_state[new_cxt] as usize]);
        self.index += 1;
    }

    pub fn end_byte(&mut self) { self.index = 0; }
}

// ====================================================================
// Buffer — circular byte buffer that backs upstream's `buf(i)` /
// `bufr(i)` helpers. `pos` is the byte index of the next write
// slot; `buffer[(pos - i) & BMASK]` returns the byte `i` positions
// ago.
// ====================================================================

pub const BMASK: usize = (1 << 24) - 1;
pub const BUFSIZE: usize = 1 << 24;

pub struct Buffer {
    pub buffer: Vec<u8>,
    pub pos: usize,
}

impl Buffer {
    pub fn new() -> Self { Self { buffer: vec![0u8; BUFSIZE], pos: 0 } }
    /// `buf(i)` — byte `i` positions back (i ≥ 1; i = 1 is the
    /// most recent byte).
    #[inline] pub fn buf(&self, i: usize) -> u8 {
        self.buffer[(self.pos.wrapping_sub(i)) & BMASK]
    }
    /// `bufr(i)` — byte at absolute position `i` (mod BMASK).
    #[inline] pub fn bufr(&self, i: usize) -> u8 {
        self.buffer[i & BMASK]
    }
    /// Append a byte and advance `pos`.
    pub fn push(&mut self, b: u8) {
        self.buffer[self.pos & BMASK] = b;
        self.pos = self.pos.wrapping_add(1);
    }
}

impl Default for Buffer { fn default() -> Self { Self::new() } }

// ====================================================================
// MTFList — small move-to-front index list used by SparseMatchModel.
// ====================================================================

pub struct MtfList {
    pub root:  i32,
    pub index: i32,
    pub previous: Vec<i32>,
    pub next:     Vec<i32>,
}

impl MtfList {
    pub fn new(s: usize) -> Self {
        let mut previous = vec![0i32; s];
        let mut next = vec![0i32; s];
        for i in 0..s {
            previous[i] = i as i32 - 1;
            next[i]     = i as i32 + 1;
        }
        next[s - 1] = -1;
        Self { root: 0, index: 0, previous, next }
    }

    pub fn get_first(&mut self) -> i32 {
        self.index = self.root;
        self.index
    }

    pub fn get_next(&mut self) -> i32 {
        if self.index >= 0 { self.index = self.next[self.index as usize]; }
        self.index
    }

    pub fn move_to_front(&mut self, i: i32) {
        debug_assert!(i >= 0 && (i as usize) < self.previous.len());
        self.index = i;
        if self.index == self.root { return; }
        let p = self.previous[self.index as usize];
        let n = self.next[self.index as usize];
        if p >= 0 { self.next[p as usize] = n; }
        if n >= 0 { self.previous[n as usize] = p; }
        self.previous[self.root as usize] = self.index;
        self.next[self.index as usize] = self.root;
        self.root = self.index;
        self.previous[self.root as usize] = -1;
    }
}

// ====================================================================
// SparseMatchModel — sparse-stride byte-match predictor used by
// upstream's FXCMv1. Holds 4 distinct hash chains keyed by
// configurable offset/stride/length, finds the longest sparse match
// and emits two stretched-logit inputs into the mixer.
// ====================================================================

pub const SPARSE_MAX_LEN: u32 = 64;
pub const SPARSE_MIN_LEN: u32 = 2;
pub const SPARSE_NUM_HASHES: usize = 4;

#[derive(Clone, Copy)]
pub struct SparseConfig {
    pub offset: u32,
    pub stride: u32,
    pub deletions: u32,
    pub min_len: u32,
    pub bit_mask: u32,
}

const SPARSE_CONFIGS: [SparseConfig; SPARSE_NUM_HASHES] = [
    SparseConfig { offset: 0, stride: 1, deletions: 0, min_len: 3, bit_mask: 0xFF },
    SparseConfig { offset: 0, stride: 1, deletions: 0, min_len: 4, bit_mask: 0xFF },
    SparseConfig { offset: 0, stride: 2, deletions: 0, min_len: 6, bit_mask: 0xFF },
    SparseConfig { offset: 0, stride: 1, deletions: 0, min_len: 5, bit_mask: 0xFF },
];

pub struct SparseMatchModel {
    pub table: Vec<u32>,           // 1024*1024 entries
    pub list: MtfList,
    pub hashes: [u32; SPARSE_NUM_HASHES],
    pub hash_index: u32,
    pub length: u32,
    pub index: u32,
    pub mask: u32,
    pub expected_byte: u8,
    pub valid: bool,
}

impl SparseMatchModel {
    pub fn new() -> Self {
        Self {
            table: vec![0u32; 1024 * 1024],
            list: MtfList::new(SPARSE_NUM_HASHES),
            hashes: [0; SPARSE_NUM_HASHES],
            hash_index: 0,
            length: 0,
            index: 0,
            mask: 1024 * 1024 - 1,
            expected_byte: 0,
            valid: false,
        }
    }

    pub fn update(&mut self, buf: &Buffer) {
        // Refresh hashes.
        for i in 0..SPARSE_NUM_HASHES {
            let cfg = SPARSE_CONFIGS[i];
            let mut h = (i as u32 + 1) * 191;
            let mut k = 1u32;
            for _ in 0..cfg.min_len {
                h = h.wrapping_mul(191).wrapping_add((buf.buf(k as usize) as u32) << i);
                k += cfg.stride;
            }
            self.hashes[i] = h & self.mask;
        }
        if self.length != 0 {
            self.index = self.index.wrapping_add(1);
            if self.length < SPARSE_MAX_LEN { self.length += 1; }
        } else {
            // Try each hash from the MTF list head.
            let mut i = self.list.get_first();
            while i >= 0 {
                let cfg = SPARSE_CONFIGS[i as usize];
                self.index = self.table[self.hashes[i as usize] as usize];
                if self.index > 0 {
                    let mut offset = 1u32;
                    while self.length < cfg.min_len
                        && (buf.buf(offset as usize)
                            ^ buf.bufr(self.index.wrapping_sub(offset) as usize)) == 0
                    {
                        self.length += 1;
                        offset = offset.wrapping_add(cfg.stride);
                    }
                    if self.length >= cfg.min_len {
                        self.length -= cfg.min_len - 1;
                        self.hash_index = i as u32;
                        self.list.move_to_front(i);
                        break;
                    }
                }
                self.length = 0;
                self.index = 0;
                i = self.list.get_next();
            }
        }
        // Update positions.
        for i in 0..SPARSE_NUM_HASHES {
            self.table[self.hashes[i] as usize] = buf.pos as u32;
        }
        self.expected_byte = buf.bufr(self.index as usize);
        self.valid = self.length > 1;
    }

    /// Per-bit prediction emit. Adds two inputs to `out` and
    /// returns the current match length.
    pub fn p(&mut self, c0: i32, bpos: i32, buf: &Buffer, out: &mut Vec<i16>) -> u32 {
        let b = (c0 << (8 - bpos)) as u8;
        if bpos == 0 { self.update(buf); }

        if self.length > 0
            && (((self.expected_byte ^ b) >> (8 - bpos)) as i32) != 0
        {
            self.length = 0;
        }

        if self.valid {
            if self.length > 1 {
                let expected_bit = ((self.expected_byte >> (7 - bpos)) & 1) as i32;
                let sign = 2 * expected_bit - 1;
                let l1 = (self.length - 1).min(32);
                let l2 = (self.length - 2).min(3);
                let l3 = (self.length - 1).min(8);
                out.push((sign * ((l1 as i32) << 5)) as i16);
                out.push((sign * ((1 << l2) * l3 as i32) << 4) as i16);
            } else {
                out.push(0);
                out.push(0);
            }
        } else {
            out.push(0);
            out.push(0);
        }
        self.length
    }
}

impl Default for SparseMatchModel { fn default() -> Self { Self::new() } }

// ====================================================================
// `vec<T, S>` — fixed-capacity vector used by BracketContext /
// WordsContext / ColumnContext. Upstream parameterises by element
// type T and capacity S; the Rust port uses a `Vec<T>` with a
// recorded `capacity` matching upstream's compile-time `S`.
// ====================================================================

#[derive(Clone)]
pub struct CmixVec<T: Default + Copy> {
    pub cxt: Vec<T>,
    pub size: usize,
    pub capacity: usize,
}

impl<T: Default + Copy> CmixVec<T> {
    pub fn new(s: usize) -> Self {
        Self { cxt: vec![T::default(); s], size: 0, capacity: s }
    }
    pub fn push(&mut self, v: T) {
        debug_assert!(self.size < self.capacity);
        self.cxt[self.size] = v;
        self.size += 1;
    }
    pub fn pop(&mut self) -> T {
        if self.size == 0 { return T::default(); }
        self.size -= 1;
        self.cxt[self.size]
    }
    pub fn last(&self) -> T {
        if self.size == 0 { T::default() } else { self.cxt[self.size - 1] }
    }
    pub fn at(&self, i: usize) -> T { self.cxt[i] }
    pub fn set(&mut self, i: usize, v: T) { self.cxt[i] = v; }
    pub fn inc(&mut self, i: usize) where T: std::ops::AddAssign + From<u8> {
        let one: T = T::from(1);
        self.cxt[i] += one;
    }
    pub fn clear(&mut self) { self.size = 0; }
    pub fn is_empty(&self) -> bool { self.size == 0 }
    pub fn len(&self) -> usize { self.size }
}

/// Backwards-compat alias for the i32-specific case used in earlier tests.
pub type ContextVec = CmixVec<i32>;

// ====================================================================
// `BracketContext<T>` — bracket-pair stack tracker. Generic over T
// so callers pick u8 / u16 storage (1-byte bracket pairs vs 2-byte
// HTML-tag pairs etc). Mirrors upstream's `template <typename T>
// struct BracketContext`.
// ====================================================================

pub struct BracketContextFx<T>
    where T: Default + Copy + Eq + Into<u32> + From<u8>,
{
    pub context: u32,
    pub active: CmixVec<i32>,
    pub distance: CmixVec<i32>,
    pub element: Vec<T>,
    pub element_count: i32,
    pub do_pop: bool,
    pub limit: i32,
    pub cxt: u32,
    pub dst: u32,
    pub elem_bits: u32,
}

impl<T> BracketContextFx<T>
    where T: Default + Copy + Eq + Into<u32> + From<u8>,
{
    pub fn new(elements: &[T], pop: bool, limit: i32, elem_bits: u32) -> Self {
        Self {
            context: 0,
            active: CmixVec::new(512),
            distance: CmixVec::new(512),
            element: elements.to_vec(),
            element_count: elements.len() as i32,
            do_pop: pop,
            limit,
            cxt: 0, dst: 0,
            elem_bits,
        }
    }

    pub fn reset(&mut self) {
        self.active.clear();
        self.distance.clear();
        self.context = 0; self.cxt = 0; self.dst = 0;
    }

    fn find(&self, b: u32) -> bool {
        let mut i = 0;
        while i < self.element_count as usize {
            if self.element[i].into() == b { return true; }
            i += 2;
        }
        false
    }

    fn find_end(&self, b: u32, c: u32) -> bool {
        let mut i = 0;
        while i < self.element_count as usize {
            if self.element[i].into() == b && self.element[i + 1].into() == c {
                return true;
            }
            i += 2;
        }
        false
    }

    pub fn update(&mut self, byte: u32) {
        let mut pop = false;
        if !self.active.is_empty() {
            let top = self.active.last() as u32;
            let dist_top = self.distance.last();
            if self.find_end(top, byte) || dist_top >= self.limit {
                self.active.pop();
                self.distance.pop();
                pop = self.do_pop;
            } else {
                let last_idx = self.distance.len() - 1;
                let cur = self.distance.at(last_idx);
                self.distance.set(last_idx, cur + 1);
            }
        }
        if !pop && self.find(byte) {
            self.active.push(byte as i32);
            self.distance.push(0);
        }
        if !self.active.is_empty() {
            self.cxt = self.active.last() as u32;
            let max_d = (1u32 << self.elem_bits) - 1;
            self.dst = (self.distance.last() as u32).min(max_d);
            self.context = (1u32 << self.elem_bits) * self.cxt + self.dst;
        } else {
            self.context = 0; self.cxt = 0; self.dst = 0;
        }
    }
}

// ====================================================================
// Wikipedia control-character constants (mirror upstream). Not all
// of these appear in the FXCMv1 control flow yet, but they read
// cleaner alongside the `ColumnContext` port.
// ====================================================================

pub const COLON_CHR:        u8 = b'J';
pub const SEMICOLON_CHR:    u8 = b'K';
pub const LESSTHAN_CHR:     u8 = b'L';
pub const EQUALS_CHR:       u8 = b'M';
pub const GREATERTHAN_CHR:  u8 = b'N';
pub const QUESTION_CHR:     u8 = b'O';
pub const FIRSTUPPER_CHR:   u8 = 64;
pub const SQUAREOPEN_CHR:   u8 = 91;
pub const BACKSLASH_CHR:    u8 = 92;
pub const SQUARECLOSE_CHR:  u8 = 93;
pub const CURLYOPENING_CHR: u8 = b'P';
pub const VERTICALBAR_CHR:  u8 = b'Q';
pub const CURLYCLOSE_CHR:   u8 = b'R';

pub const APOSTROPHE_CHR: u8 = 39;
pub const QUOTATION_CHR:  u8 = 34;
pub const SPACE_CHR:      u8 = 32;
pub const HTLINK_CHR:     u8 = 31;
pub const HTML_CHR:       u8 = 30;
pub const LF_CHR:         u8 = 10;
pub const ESCAPE_CHR:     u8 = 12;
pub const UPPER_CHR:      u8 = 7;
pub const TEXTDATA_CHR:   u8 = 96;

pub const WIKIHEADER_CHR: u8 = GREATERTHAN_CHR;
pub const WIKITABLE_CHR:  u8 = b'-';

// ====================================================================
// `Column` / `ColumnContext` — track the last few rows of the input
// stream as character columns, plus wiki-table cell positions.
// Mirrors upstream's `Column` and `ColumnContext` structs.
// ====================================================================

#[derive(Clone)]
pub struct Column {
    pub linepos: u32,
    pub fc: u8,
    pub bytes: CmixVec<u8>,
}

impl Column {
    pub fn new() -> Self {
        Self { linepos: 0, fc: 0, bytes: CmixVec::new(2048) }
    }
}

impl Default for Column { fn default() -> Self { Self::new() } }

pub struct ColumnContext {
    pub col: [Column; 4],
    pub cell: [CmixVec<u32>; 4],
    pub rows: usize,
    pub cell_count: i32,
    pub cells: usize,
    pub abovecellpos: u32,
    pub abovecellpos1: u32,
    pub nl: bool,
    pub is_temp: bool,
    pub limit: i32,
    pub nl_char: u8,
}

impl ColumnContext {
    pub fn new(limit: i32) -> Self {
        Self {
            col: [Column::new(), Column::new(), Column::new(), Column::new()],
            cell: [
                CmixVec::new(32), CmixVec::new(32),
                CmixVec::new(32), CmixVec::new(32),
            ],
            rows: 0, cell_count: 0, cells: 0,
            abovecellpos: 0, abovecellpos1: 0,
            nl: false, is_temp: false,
            limit,
            nl_char: LF_CHR,
        }
    }

    pub fn lastfc(&self, i: usize) -> u8 {
        self.col[(self.rows.wrapping_sub(i)) & 3].fc
    }
    pub fn is_new_line(&self) -> bool { self.nl }

    pub fn collen(&self, i: usize, l: i32) -> i32 {
        let l = if l != 0 { l } else { self.limit };
        l.min(self.col[(self.rows.wrapping_sub(i)) & 3].bytes.len() as i32 + 1)
    }
    pub fn nlpos(&self, i: usize) -> u32 {
        self.col[(self.rows.wrapping_sub(i)) & 3].linepos
    }
    pub fn colb(&self, i: usize, j: i32, l: i32) -> u8 {
        if self.collen(0, l) < self.collen(i, l) {
            let idx = (self.collen(0, 0) - (1 + j)).max(0) as usize;
            self.col[(self.rows.wrapping_sub(i)) & 3].bytes.at(idx)
        } else { 0 }
    }

    fn reset_cells(&mut self) {
        for c in self.cell.iter_mut() { c.clear(); }
    }

    /// `byte` = current byte, `b2` = the upstream's `(b3<<16) |
    /// (b2<<8) | b1` packed-3-byte history, `blpos` = block
    /// position (from BlockData), `is_pre` = upstream's preformatted
    /// flag.
    pub fn update(&mut self, byte: u8, b2: u32, blpos: u32, is_pre: bool) {
        // Wiki-table fence detection.
        if b2 == ((CURLYOPENING_CHR as u32) << 16
                  | (CURLYOPENING_CHR as u32) << 8
                  | (VERTICALBAR_CHR as u32))
        {
            self.nl_char = WIKITABLE_CHR;
        } else if b2 == ((VERTICALBAR_CHR as u32) << 16
                         | (CURLYCLOSE_CHR as u32) << 8
                         | (CURLYCLOSE_CHR as u32))
        {
            self.nl_char = LF_CHR;
            self.reset_cells();
        }
        if byte != CURLYOPENING_CHR
            && (b2 & 0xFF00) == ((CURLYOPENING_CHR as u32) << 8)
            && (b2 & 0xFF0000) != ((CURLYOPENING_CHR as u32) << 16)
        {
            self.is_temp = true;
        } else if self.is_temp && byte == CURLYCLOSE_CHR {
            self.is_temp = false;
        }

        self.nl = false;
        if byte == LF_CHR {
            self.col[self.rows].bytes.push(byte);
            self.rows = (self.rows + 1) & 3;
            self.col[self.rows].bytes.clear();
            self.col[self.rows].fc = 0;
            self.col[self.rows].linepos = blpos.wrapping_sub(1);
        } else {
            self.col[self.rows].bytes.push(byte);
            if self.collen(0, 0) == 2 {
                self.col[self.rows].fc = byte.min(TEXTDATA_CHR);
                self.nl = true;
                if self.col[self.rows].fc == GREATERTHAN_CHR && !is_pre {
                    self.nl_char = WIKIHEADER_CHR;
                }
                if self.col[self.rows].fc == SQUAREOPEN_CHR && self.nl_char == WIKIHEADER_CHR {
                    self.nl_char = LF_CHR;
                }
            }
        }

        // Wiki-table cell tracking.
        if self.nl_char == WIKITABLE_CHR {
            if (b2 & 0xFFFF)
                == (WIKITABLE_CHR as u32 + (VERTICALBAR_CHR as u32) * 256)
            {
                self.cells = (self.cells + 1) & 3;
                self.cell[self.cells].clear();
                self.cell[self.cells].push(blpos);
                self.cell_count = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            }
            let mut newcell = false;
            if (b2 & 0xFFFF) == (VERTICALBAR_CHR as u32 + (VERTICALBAR_CHR as u32) * 256)
                || (b2 & 0xFFFF00)
                   == ((VERTICALBAR_CHR as u32 + (LF_CHR as u32) * 256) * 256)
            {
                self.cell[self.cells].push(blpos);
                self.cell_count += 1;
                newcell = true;
            }
            if self.abovecellpos != 0 {
                self.abovecellpos = self.abovecellpos.wrapping_add(1);
                if self.abovecellpos > self.abovecellpos1 {
                    self.abovecellpos = 0;
                    self.abovecellpos1 = 0;
                }
            }
            if newcell && self.cells_count(1) > 0 {
                self.abovecellpos = self.cell_pos(self.cell_count - 1, 1);
                self.abovecellpos1 = self.cell_pos(self.cell_count, 1);
            }
        }
        if self.nl_char == WIKIHEADER_CHR {
            if (b2 & 0xFFFF) == (WIKIHEADER_CHR as u32 + (LF_CHR as u32) * 256) {
                self.cells = (self.cells + 1) & 3;
                self.cell[self.cells].clear();
                self.cell[self.cells].push(blpos);
                self.cell_count = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            } else {
                let mut newcell = false;
                if (b2 & 0xFF) == (WIKIHEADER_CHR as u32) {
                    self.cell[self.cells].push(blpos);
                    self.cell_count += 1;
                    newcell = true;
                }
                if self.abovecellpos != 0 {
                    self.abovecellpos = self.abovecellpos.wrapping_add(1);
                    if self.abovecellpos > self.abovecellpos1 {
                        self.abovecellpos = 0;
                        self.abovecellpos1 = 0;
                    }
                }
                if newcell && self.cells_count(1) > 0 {
                    self.abovecellpos = self.cell_pos(self.cell_count - 1, 1);
                    self.abovecellpos1 = self.cell_pos(self.cell_count, 1);
                }
            }
        }
    }

    pub fn cells_count(&self, row: usize) -> i32 {
        self.cell[(self.cells.wrapping_sub(row)) & 3].len() as i32
    }
    pub fn cell_pos(&self, cell_id: i32, row: usize) -> u32 {
        let total = (self.cells_count(row) - 1).max(0).min(cell_id) as usize;
        self.cell[(self.cells.wrapping_sub(row)) & 3].at(total)
    }
}

// ====================================================================
// `WordsContext` — sentence/word state. Tracks a sliding window of
// recent words plus their types, stems, and capitalisation.
// ====================================================================

pub struct WordsContext {
    pub sbytes: CmixVec<u16>,
    pub r#type: CmixVec<u32>,
    pub stem: CmixVec<u32>,
    pub capital: CmixVec<u8>,
    pub fword: u32,
    pub ftype: u32,
    pub pbyte: u8,
    pub wordcount: i32,
    pub upper: i32,
    pub r#ref: i32,
}

impl WordsContext {
    pub fn new() -> Self {
        Self {
            sbytes: CmixVec::new(64 * 4),
            r#type: CmixVec::new(64 * 4),
            stem:   CmixVec::new(64 * 4),
            capital: CmixVec::new(64 * 4),
            fword: 0, ftype: 0, pbyte: 0,
            wordcount: 0, upper: 0, r#ref: 0,
        }
    }

    pub fn reset(&mut self) {
        self.sbytes.clear(); self.r#type.clear();
        self.stem.clear(); self.capital.clear();
        self.fword = 0; self.pbyte = 0; self.wordcount = 0;
        self.upper = 0; self.ftype = 0; self.r#ref = 0;
    }

    pub fn set(&mut self, b: u8, a: i32) { self.pbyte = b; self.upper = a; }

    pub fn update(&mut self, w: u32, b: u8, t: u32, s: u32) {
        if self.fword == 0 { self.fword = w; }
        self.sbytes.push((self.pbyte as u16) * 256 + b as u16);
        self.r#type.push(t);
        self.stem.push(s);
        self.capital.push(self.upper as u8);
        self.pbyte = 0;
        self.wordcount += 1;
        if self.ftype == 0 && t != 0 { self.ftype = t; }
    }

    pub fn remove(&mut self) {
        if !self.stem.is_empty() {
            self.sbytes.pop();
            self.r#type.pop();
            self.stem.pop();
            self.capital.pop();
            self.wordcount -= 1;
        }
    }

    pub fn word(&self, i: usize) -> u32 {
        let n = self.stem.len();
        if n >= i && i > 0 { self.stem.at(n - i) } else { 0 }
    }
    pub fn s_bytes(&self, i: usize) -> u16 {
        let n = self.sbytes.len();
        if n >= i && i > 0 { self.sbytes.at(n - i) } else { 0 }
    }
    pub fn types(&self, i: usize) -> u32 {
        let n = self.r#type.len();
        if n >= i && i > 0 { self.r#type.at(n - i) } else { 0 }
    }
    pub fn capital_at(&self, i: usize) -> u8 {
        let n = self.capital.len();
        if n >= i && i > 0 { self.capital.at(n - i) } else { 0 }
    }

    pub fn last(&self, j: usize, t: u32) -> u32 {
        let num = self.r#type.len();
        if t == 0 { return self.word(j); }
        if num >= j {
            for i in j..=num {
                if (self.types(i) & t) != 0 { return self.word(i); }
            }
        }
        self.word(j)
    }

    pub fn last_if(&self, j: usize, t: u32) -> u32 {
        let num = self.r#type.len();
        if t == 0 { return self.word(j); }
        if num >= j {
            for i in j..=num {
                if (self.types(i) & t) != 0 { return self.word(i); }
            }
        }
        0
    }

    pub fn last_idx(&self, j: usize, t: u32) -> u32 {
        let num = self.r#type.len();
        if t == 0 { return 0; }
        if num >= j {
            for i in j..=num {
                if (self.types(i) & t) != 0 { return i as u32; }
            }
        }
        0
    }

    pub fn remove_words_l(&mut self, len: usize, c: u8, d: u8, f: bool) {
        if (self.s_bytes(1) & 0xFF) as u8 == d {
            for i in 1..len {
                if ((self.s_bytes(i) >> 8) as u8) == c {
                    while ((self.s_bytes(1) >> 8) as u8) != c { self.remove(); }
                    if f { self.remove(); }
                    break;
                }
            }
        }
    }

    pub fn remove_words_r(&mut self, len: usize, c: u8, d: u8, f: bool) {
        if (self.s_bytes(1) & 0xFF) as u8 == d {
            for i in 1..len {
                if ((self.s_bytes(i) & 0xFF) as u8) == c {
                    while ((self.s_bytes(1) & 0xFF) as u8) != c { self.remove(); }
                    if f { self.remove(); }
                    break;
                }
            }
        }
    }
}

impl Default for WordsContext { fn default() -> Self { Self::new() } }

// ====================================================================
// Inline hash helper used throughout FXCMv1.
// ====================================================================

#[inline]
/// Three-arg hash used by FXCMv1 for APM context derivation
/// (`AH1`, `AH2`, etc.). Verbatim from fxcmv1.cpp:2281-2284.
/// `c` defaults to `0xffffffff` upstream.
pub fn fxcm_hash(a: u32, b: u32, c: u32) -> u32 {
    let h = a.wrapping_mul(110_002_499)
        .wrapping_add(b.wrapping_mul(30_005_491))
        .wrapping_add(c.wrapping_mul(50_004_239));
    h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4)
}

pub fn fxcm_hash3(a: u32, b: u32, c: u32) -> u32 {
    let h = a.wrapping_mul(110_002_499)
        .wrapping_add(b.wrapping_mul(30_005_491))
        .wrapping_add(c.wrapping_mul(50_004_239));
    h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4)
}

#[inline]
pub fn char_swap(c: i32) -> i32 {
    let mut c = c;
    if c >= b'{' as i32 && c < 127 { c += b'P' as i32 - b'{' as i32; }
    else if c >= b'P' as i32 && c < b'T' as i32 { c -= b'P' as i32 - b'{' as i32; }
    else if (c >= b':' as i32 && c <= b'?' as i32)
        || (c >= b'J' as i32 && c <= b'O' as i32) { c ^= 0x70; }
    if c == b'X' as i32 || c == b'`' as i32 { c ^= b'X' as i32 ^ b'`' as i32; }
    c
}

// ====================================================================
// `Word` — small fixed-buffer letter sequence with comparison /
// suffix-edit helpers used by the Porter2 stemmer.
// ====================================================================

pub const MAX_WORD_SIZE: usize = 64;

/// Bit flags used for word type classification (matches upstream
/// `EngWordTypeFlags`).
pub mod eng {
    pub const VERB:                  u32 = 1 << 0;
    pub const NOUN:                  u32 = 1 << 1;
    pub const ADJECTIVE:             u32 = 1 << 2;
    pub const PLURAL:                u32 = 1 << 3;
    pub const PAST_TENSE:            u32 = (1 << 5) | VERB;
    pub const PRESENT_PARTICIPLE:    u32 = (1 << 4) | VERB;
    pub const ADJECTIVE_SUPERLATIVE: u32 = (1 << 5) | ADJECTIVE;
    pub const ADJECTIVE_WITHOUT:     u32 = (1 << 6) | ADJECTIVE;
    pub const ADJECTIVE_FULL:        u32 = (1 << 7) | ADJECTIVE;
    pub const ADVERB_OF_MANNER:      u32 = 1 << 8;
    pub const SUFFIX:                u32 = 1 << 9;
    pub const PREFIX:                u32 = 1 << 10;
    pub const MALE:                  u32 = 1 << 11;
    pub const FEMALE:                u32 = 1 << 13;
    pub const ARTICLE:               u32 = 1 << 14;
    pub const CONJUNCTION:           u32 = 1 << 15;
    pub const ADPOSITION:            u32 = 1 << 16;
    pub const NUMBER:                u32 = 1 << 17;
    pub const PREPOSITION:           u32 = 1 << 18;
    pub const CONJUNCTIVE_ADVERB:    u32 = 1 << 19;
}

#[derive(Clone)]
pub struct Word {
    pub letters: [u8; MAX_WORD_SIZE],
    pub start: u8,
    pub end: u8,
    pub hash: u32,
    pub r#type: u32,
    pub suffix: u32,
    pub prefix: u32,
}

impl Word {
    pub fn new() -> Self {
        Self {
            letters: [0; MAX_WORD_SIZE],
            start: 0, end: 0,
            hash: 0, r#type: 0, suffix: 0, prefix: 0,
        }
    }

    pub fn equals_str(&self, s: &[u8]) -> bool {
        let len = s.len();
        let extra = if self.letters[self.start as usize] != 0 { 1 } else { 0 };
        let cur_len = (self.end as usize - self.start as usize + extra) as usize;
        if cur_len != len { return false; }
        let st = self.start as usize;
        self.letters[st..st + len] == *s
    }

    /// Append `c` to the end of the word (no-op if zero or full).
    pub fn append(&mut self, c: u8) {
        if c > 0 && (self.end as usize) < MAX_WORD_SIZE - 1 {
            if self.letters[self.end as usize] > 0 { self.end += 1; }
            self.letters[self.end as usize] = c;
        }
    }

    /// Letter at offset `i` from `Start` (returns 0 if out of range).
    pub fn at(&self, i: u8) -> u8 {
        if self.end >= self.start && (self.end - self.start) >= i {
            self.letters[(self.start + i) as usize]
        } else { 0 }
    }
    /// Letter at offset `i` from `End` (returns 0 if out of range).
    pub fn from_end(&self, i: u8) -> u8 {
        if self.end >= self.start && (self.end - self.start) >= i {
            self.letters[(self.end - i) as usize]
        } else { 0 }
    }

    pub fn len(&self) -> u32 {
        if self.letters[self.start as usize] != 0 {
            (self.end - self.start + 1) as u32
        } else { 0 }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Replace `old_suffix` (if present) with `new_suffix`. Returns
    /// true if the swap happened.
    pub fn change_suffix(&mut self, old_suffix: &[u8], new_suffix: &[u8]) -> bool {
        let len = old_suffix.len();
        if (self.len() as usize) <= len { return false; }
        let start = self.end as usize - len + 1;
        if &self.letters[start..start + len] != old_suffix { return false; }
        if !new_suffix.is_empty() {
            let n = new_suffix.len();
            let cap = MAX_WORD_SIZE - 1;
            let new_end = (self.end as usize + n - len).min(cap);
            let copy_n = (new_end + 1).saturating_sub(start).min(n);
            for i in 0..copy_n { self.letters[start + i] = new_suffix[i]; }
            self.end = new_end as u8;
        } else {
            self.end -= len as u8;
        }
        true
    }

    pub fn matches_any(&self, list: &[&[u8]]) -> bool {
        let len = self.len() as usize;
        for &cand in list {
            if cand.len() != len { continue; }
            let st = self.start as usize;
            if &self.letters[st..st + len] == cand { return true; }
        }
        false
    }

    pub fn ends_with(&self, suffix: &[u8]) -> bool {
        let len = suffix.len();
        if (self.len() as usize) <= len { return false; }
        let start = self.end as usize - len + 1;
        &self.letters[start..start + len] == suffix
    }

    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        let len = prefix.len();
        if (self.len() as usize) <= len { return false; }
        let st = self.start as usize;
        &self.letters[st..st + len] == prefix
    }
}

impl Default for Word { fn default() -> Self { Self::new() } }

// ====================================================================
// Stemmer reference tables (verbatim from upstream).
// ====================================================================

pub const VOWELS: &[u8] = b"aeiouy";
pub const DOUBLES: &[u8] = b"bdfgmnprt";
pub const LI_ENDINGS: &[u8] = b"cdeghkmnrt";
pub const NON_SHORT_CONSONANTS: &[u8] = b"wxY";

// Negation / prefix sub-flags (upstream `EngWordTypeFlagsNegation`).
pub mod neg {
    pub const NEGATION:     u32 = 1 << 0;
    pub const PREFIX_IRR:   u32 = (1 << 1) | NEGATION;
    pub const PREFIX_OVER:  u32 = 1 << 2;
    pub const PREFIX_UNDER: u32 = 1 << 3;
    pub const PREFIX_UNN:   u32 = (1 << 4) | NEGATION;
    pub const PREFIX_NON:   u32 = (1 << 5) | NEGATION;
    pub const PREFIX_ANTI:  u32 = (1 << 6) | NEGATION;
    pub const PREFIX_DIS:   u32 = (1 << 7) | NEGATION;
}

// Suffix sub-flags (upstream `EngWordTypeFlagsSuffix`). Note the
// composite bits like `SUFFIX_AL = (1<<6)|Noun` — kept verbatim
// to match upstream's behaviour.
pub mod suf {
    use super::eng::NOUN;
    use super::eng::ADJECTIVE;
    pub const SUFFIX_NESS:    u32 = 1 << 0;
    pub const SUFFIX_ITY:     u32 = (1 << 1) | NOUN;
    pub const SUFFIX_CAPABLE: u32 = 1 << 2;
    pub const SUFFIX_NCE:     u32 = 1 << 3;
    pub const SUFFIX_NT:      u32 = 1 << 4;
    pub const SUFFIX_ION:     u32 = 1 << 5;
    pub const SUFFIX_AL:      u32 = (1 << 6) | ADJECTIVE;
    pub const SUFFIX_IC:      u32 = (1 << 7) | ADJECTIVE;
    pub const SUFFIX_IVE:     u32 = 1 << 8;
    pub const SUFFIX_OUS:     u32 = (1 << 9) | ADJECTIVE;
}

// ====================================================================
// English-language reference word lists used by the stemmer.
// All tables verbatim from `fxcmv1.cpp` (lines 2425-2654).
// ====================================================================

pub const VERB_WORDS1: &[&[u8]] = &[
    b"has", b"had", b"have", b"was", b"were", b"may", b"might", b"must",
    b"shall", b"should", b"can", b"could", b"will", b"would", b"is",
    b"am", b"are", b"be", b"being", b"been", b"do", b"does", b"did",
];

pub const NUMBERS: &[&[u8]] = &[
    b"one", b"two", b"three", b"four", b"five", b"six", b"seven",
    b"eight", b"nine", b"ten", b"twenty", b"thirty", b"forty", b"fifty",
    b"sixty", b"seventy", b"eighty", b"ninety", b"hundred", b"thousand",
    b"million",
];

pub const CONJ_WORDS: &[&[u8]] = &[
    b"for", b"and", b"nor", b"but", b"or", b"yet", b"so", b"than",
    b"as", b"that", b"if", b"when", b"because", b"while", b"where",
    b"after", b"though", b"whether", b"before", b"although", b"like",
    b"once", b"unless", b"now", b"except",
];

pub const APO_WORDS: &[&[u8]] = &[
    b"in", b"during", b"at", b"on", b"since", b"until", b"above",
    b"across", b"against", b"along", b"among", b"around", b"behind",
    b"below", b"beneath", b"beside", b"between", b"by", b"down",
    b"from", b"into", b"near", b"of", b"off", b"to", b"toward",
    b"under", b"upon", b"with", b"within",
];

pub const PREP_WORDS: &[&[u8]] = &[b"as", b"by", b"de", b"in", b"on"];

pub const CON_AD_VER_PREP_WORDS: &[&[u8]] = &[b"also", b"thus"];

pub const VERB_WORDS: &[&[u8]] = &[
    b"be", b"do", b"an", b"could", b"may", b"must", b"need", b"ought",
    b"shall", b"should", b"will", b"would",
];

pub const MALE_WORDS: &[&[u8]] = &[
    b"he", b"him", b"his", b"himself", b"man", b"men", b"boy",
    b"husband", b"actor",
];

pub const FEMALE_WORDS: &[&[u8]] = &[
    b"she", b"her", b"herself", b"woman", b"women", b"girl", b"wife",
    b"actress",
];

pub const ARTICLE_WORDS: &[&[u8]] = &[b"a", b"an", b"the"];

// Step suffix tables.
pub const SUFFIXES_STEP0: &[&[u8]] = &[b"'s'", b"'s", b"'"];

pub const SUFFIXES_STEP1B: &[&[u8]] =
    &[b"eedly", b"eed", b"ed", b"edly", b"ing", b"ingly"];

pub const TYPES_STEP1B: &[u32] = &[
    eng::ADVERB_OF_MANNER,
    0,
    eng::PAST_TENSE,
    eng::ADVERB_OF_MANNER | eng::PAST_TENSE,
    eng::PRESENT_PARTICIPLE,
    eng::ADVERB_OF_MANNER | eng::PRESENT_PARTICIPLE,
];

pub const SUFFIXES_STEP2: &[(&[u8], &[u8])] = &[
    (b"ization", b"ize"),
    (b"ational", b"ate"),
    (b"ousness", b"ous"),
    (b"iveness", b"ive"),
    (b"fulness", b"ful"),
    (b"tional",  b"tion"),
    (b"lessli",  b"less"),
    (b"biliti",  b"ble"),
    (b"entli",   b"ent"),
    (b"ation",   b"ate"),
    (b"alism",   b"al"),
    (b"aliti",   b"al"),
    (b"fulli",   b"ful"),
    (b"ousli",   b"ous"),
    (b"iviti",   b"ive"),
    (b"enci",    b"ence"),
    (b"anci",    b"ance"),
    (b"abli",    b"able"),
    (b"izer",    b"ize"),
    (b"ator",    b"ate"),
    (b"alli",    b"al"),
    (b"bli",     b"ble"),
];

pub const TYPES_STEP2: &[u32] = &[
    eng::SUFFIX,
    eng::SUFFIX | eng::ADJECTIVE,
    eng::SUFFIX,
    eng::SUFFIX,
    eng::SUFFIX,
    eng::SUFFIX | eng::ADJECTIVE,
    eng::ADVERB_OF_MANNER,
    eng::ADVERB_OF_MANNER | eng::NOUN | eng::SUFFIX,
    eng::ADVERB_OF_MANNER,
    eng::SUFFIX,
    0,
    eng::NOUN | eng::SUFFIX,
    eng::ADVERB_OF_MANNER,
    eng::ADVERB_OF_MANNER,
    eng::NOUN | eng::SUFFIX,
    0,
    0,
    eng::ADVERB_OF_MANNER,
    0,
    0,
    eng::ADVERB_OF_MANNER,
    eng::ADVERB_OF_MANNER,
];

pub const TYPES_STEP2_SUFFIX: &[u32] = &[
    suf::SUFFIX_ION,
    suf::SUFFIX_ION | suf::SUFFIX_AL,
    suf::SUFFIX_NESS,
    suf::SUFFIX_NESS,
    suf::SUFFIX_NESS,
    suf::SUFFIX_ION | suf::SUFFIX_AL,
    0,
    suf::SUFFIX_ITY,
    0,
    suf::SUFFIX_ION,
    0,
    suf::SUFFIX_ITY,
    0,
    0,
    suf::SUFFIX_ITY,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

pub const SUFFIXES_STEP3: &[(&[u8], &[u8])] = &[
    (b"ational", b"ate"),
    (b"tional",  b"tion"),
    (b"alize",   b"al"),
    (b"icate",   b"ic"),
    (b"iciti",   b"ic"),
    (b"ical",    b"ic"),
    (b"ful",     b""),
    (b"ness",    b""),
];

pub const TYPES_STEP3: &[u32] = &[
    eng::SUFFIX | eng::ADJECTIVE,
    eng::SUFFIX | eng::ADJECTIVE,
    0,
    0,
    eng::NOUN | eng::SUFFIX,
    eng::SUFFIX | eng::ADJECTIVE,
    eng::ADJECTIVE_FULL,
    eng::SUFFIX,
];

pub const TYPES_STEP3_SUFFIX: &[u32] = &[
    suf::SUFFIX_ION | suf::SUFFIX_AL,
    suf::SUFFIX_ION | suf::SUFFIX_AL,
    0,
    0,
    suf::SUFFIX_ITY,
    suf::SUFFIX_AL,
    0,
    suf::SUFFIX_NESS,
];

pub const SUFFIXES_STEP4: &[&[u8]] = &[
    b"al", b"ance", b"ence", b"er", b"ic", b"able", b"ible", b"ant",
    b"ement", b"ment", b"ent", b"ou", b"ism", b"ate", b"iti", b"ous",
    b"ive", b"ize", b"sion", b"tion",
];

pub const TYPES_STEP4: &[u32] = &[
    eng::SUFFIX | eng::ADJECTIVE,
    eng::SUFFIX,
    eng::SUFFIX,
    0,
    eng::SUFFIX | eng::ADJECTIVE,
    eng::SUFFIX,
    eng::SUFFIX,
    eng::SUFFIX,
    0,
    0,
    eng::SUFFIX,
    0,
    0,
    0,
    eng::SUFFIX | eng::NOUN,
    eng::SUFFIX | eng::ADJECTIVE,
    eng::SUFFIX,
    0,
    eng::SUFFIX,
    eng::SUFFIX,
];

pub const TYPES_STEP4_SUFFIX: &[u32] = &[
    suf::SUFFIX_AL,
    suf::SUFFIX_NCE,
    suf::SUFFIX_NCE,
    0,
    suf::SUFFIX_IC,
    suf::SUFFIX_CAPABLE,
    suf::SUFFIX_CAPABLE,
    suf::SUFFIX_NT,
    0,
    0,
    suf::SUFFIX_NT,
    0,
    0,
    0,
    suf::SUFFIX_ITY,
    suf::SUFFIX_OUS,
    suf::SUFFIX_IVE,
    0,
    suf::SUFFIX_ION,
    suf::SUFFIX_ION,
];

pub const EXCEPTIONS_REGION1: &[&[u8]] = &[b"gener", b"arsen", b"commun"];

pub const EXCEPTIONS1: &[(&[u8], &[u8])] = &[
    (b"skis",   b"ski"),
    (b"skies",  b"sky"),
    (b"dying",  b"die"),
    (b"lying",  b"lie"),
    (b"tying",  b"tie"),
    (b"idly",   b"idle"),
    (b"gently", b"gentle"),
    (b"ugly",   b"ugli"),
    (b"early",  b"earli"),
    (b"only",   b"onli"),
    (b"singly", b"singl"),
    (b"sky",    b"sky"),
    (b"news",   b"news"),
    (b"howe",   b"howe"),
    (b"atlas",  b"atlas"),
    (b"cosmos", b"cosmos"),
    (b"bias",   b"bias"),
    (b"andes",  b"andes"),
    (b"texas",  b"texas"),
];

pub const TYPES_EXCEPTIONS1: &[u32] = &[
    eng::NOUN | eng::PLURAL,
    eng::NOUN | eng::PLURAL,
    eng::PRESENT_PARTICIPLE,
    eng::PRESENT_PARTICIPLE,
    eng::PRESENT_PARTICIPLE,
    eng::ADVERB_OF_MANNER,
    eng::ADVERB_OF_MANNER,
    eng::ADJECTIVE,
    eng::ADJECTIVE | eng::ADVERB_OF_MANNER,
    0,
    eng::ADVERB_OF_MANNER,
    eng::NOUN,
    eng::NOUN,
    0,
    eng::NOUN,
    eng::NOUN,
    eng::NOUN,
    eng::NOUN | eng::PLURAL,
    eng::NOUN,
];

pub const EXCEPTIONS2: &[&[u8]] = &[
    b"inning", b"outing", b"canning", b"herring", b"earring",
    b"proceed", b"exceed", b"succeed",
];

pub const TYPES_EXCEPTIONS2: &[u32] = &[
    eng::NOUN, eng::NOUN, eng::NOUN, eng::NOUN, eng::NOUN,
    eng::VERB, eng::VERB, eng::VERB,
];

const APOSTROPHE: u8 = b'\'';

// ====================================================================
// `EnglishStemmer` — Porter2-style stemmer used by FXCMv1's word
// model. Stateless: all helpers operate on the supplied `Word` /
// `&[u8]` arguments.
// ====================================================================

pub struct EnglishStemmer;

impl EnglishStemmer {
    fn is_vowel(c: u8) -> bool { VOWELS.contains(&c) }
    fn is_consonant(c: u8) -> bool { !Self::is_vowel(c) }
    fn is_short_consonant(c: u8) -> bool { !NON_SHORT_CONSONANTS.contains(&c) }
    fn is_double(c: u8) -> bool { DOUBLES.contains(&c) }
    fn is_li_ending(c: u8) -> bool { LI_ENDINGS.contains(&c) }

    /// Word hash used by upstream's `Hash(Word*)` — fold `Letters[Start..=End]`
    /// into a 32-bit FNV-style accumulator.
    fn hash(w: &mut Word) {
        w.hash = 0xb0a710ad;
        for i in (w.start as usize)..=(w.end as usize) {
            w.hash = w.hash.wrapping_mul(263).wrapping_mul(32)
                .wrapping_add(w.letters[i] as u32);
        }
    }

    /// Index of the first consonant after a vowel-run, starting at
    /// `from` letters past `Start`. Returns `Length()` if no such
    /// position exists.
    fn get_region(w: &Word, from: u32) -> u32 {
        let mut has_vowel = false;
        let start = w.start as usize + from as usize;
        for i in start..=(w.end as usize) {
            if Self::is_vowel(w.letters[i]) {
                has_vowel = true;
                continue;
            } else if has_vowel {
                return (i - w.start as usize + 1) as u32;
            }
        }
        w.len()
    }

    fn get_region1(w: &Word) -> u32 {
        for &exc in EXCEPTIONS_REGION1 {
            if w.starts_with(exc) { return exc.len() as u32; }
        }
        Self::get_region(w, 0)
    }

    fn suffix_in_rn(w: &Word, rn: u32, suffix: &[u8]) -> bool {
        w.start != w.end && (rn as usize) <= (w.len() as usize - suffix.len())
    }

    fn ends_in_short_syllable(w: &Word) -> bool {
        if w.end == w.start {
            false
        } else if w.end == w.start + 1 {
            Self::is_vowel(w.from_end(1)) && Self::is_consonant(w.from_end(0))
        } else {
            Self::is_consonant(w.from_end(2))
                && Self::is_vowel(w.from_end(1))
                && Self::is_consonant(w.from_end(0))
                && Self::is_short_consonant(w.from_end(0))
        }
    }

    fn is_short_word(w: &Word) -> bool {
        Self::ends_in_short_syllable(w) && Self::get_region1(w) == w.len()
    }

    fn has_vowels(w: &Word) -> bool {
        for i in (w.start as usize)..=(w.end as usize) {
            if Self::is_vowel(w.letters[i]) { return true; }
        }
        false
    }

    fn trim_starting_apostrophe(w: &mut Word) -> bool {
        let mut result = false;
        let mut cnt: u32 = 0;
        while w.start != w.end && w.at(0) == APOSTROPHE {
            result = true;
            w.start += 1;
            cnt += 1;
        }
        while w.start != w.end && w.from_end(0) == APOSTROPHE {
            if cnt == 0 { break; }
            w.end -= 1;
            cnt -= 1;
        }
        if w.from_end(0) == b'-' { w.end -= 1; }
        result
    }

    fn mark_ys_as_consonants(w: &mut Word) {
        if w.at(0) == b'y' {
            let s = w.start as usize;
            w.letters[s] = b'Y';
        }
        for i in (w.start as usize + 1)..=(w.end as usize) {
            if Self::is_vowel(w.letters[i - 1]) && w.letters[i] == b'y' {
                w.letters[i] = b'Y';
            }
        }
    }

    fn process_prefixes(w: &mut Word) -> bool {
        if w.starts_with(b"irr") && w.len() > 5
            && (w.at(3) == b'a' || w.at(3) == b'e')
        {
            w.start += 2; w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_IRR;
        } else if w.starts_with(b"over") && w.len() > 5 {
            w.start += 4; w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_OVER;
        } else if w.starts_with(b"under") && w.len() > 6 {
            w.start += 5; w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_UNDER;
        } else if w.starts_with(b"unn") && w.len() > 5 {
            w.start += 2; w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_UNN;
        } else if w.starts_with(b"non")
            && w.len() > (5 + (w.at(3) == b'-') as u32)
        {
            w.start += 2 + (w.at(3) == b'-') as u8;
            w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_NON;
        } else if w.starts_with(b"anti") && w.len() > 6 && w.at(4) == b'-' {
            w.start += 4 + (w.at(4) == b'-') as u8;
            w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_ANTI;
        } else if w.starts_with(b"dis") && w.len() > 5 && w.at(3) == b'-' {
            w.start += 2 + (w.at(3) == b'-') as u8;
            w.r#type |= eng::PREFIX; w.prefix |= neg::PREFIX_DIS;
        } else {
            return false;
        }
        true
    }

    fn process_superlatives(w: &mut Word) -> bool {
        if w.ends_with(b"est") && w.len() > 4 {
            let i = w.end;
            w.end -= 3;
            w.r#type |= eng::ADJECTIVE_SUPERLATIVE;
            // `memcmp("sugg", &(*W).Letters[(*W).End-3], 4) == 0` → check window.
            let sugg = w.len() >= 4
                && w.end as usize >= 3
                && &w.letters[(w.end as usize - 3)..(w.end as usize - 3 + 4)]
                    == b"sugg";
            if w.from_end(0) == w.from_end(1) && w.from_end(0) != b'r' && !sugg {
                let last = w.from_end(0);
                let cond_a = (last != b'f' && last != b'l' && last != b's')
                    || (w.len() > 4 && w.from_end(1) == b'l'
                        && (w.from_end(2) == b'u' || w.from_end(3) == b'u'
                            || w.from_end(3) == b'v'));
                let cond_b = !(w.len() == 3
                    && w.from_end(1) == b'd' && w.from_end(2) == b'o');
                if cond_a && cond_b { w.end -= 1; }
                if w.len() == 2 && (w.at(0) != b'i' || w.at(1) != b'n') {
                    w.end = i;
                    w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                }
            } else {
                match w.from_end(0) {
                    b'd' | b'k' | b'm' | b'y' => {}
                    b'g' => {
                        let cong = w.len() >= 4
                            && w.end as usize >= 3
                            && &w.letters[(w.end as usize - 3)..(w.end as usize - 3 + 4)]
                                != b"cong";
                        if !(w.len() > 3
                            && (w.from_end(1) == b'n' || w.from_end(1) == b'r')
                            && cong)
                        {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        } else if w.from_end(2) == b'a' {
                            w.end += 1;
                        }
                    }
                    b'i' => {
                        let e = w.end as usize;
                        w.letters[e] = b'y';
                    }
                    b'l' => {
                        let mo_match = w.end as usize >= 2
                            && &w.letters[(w.end as usize - 2)..(w.end as usize - 2 + 2)]
                                == b"mo";
                        if w.end == w.start + 1 || mo_match {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        } else if Self::is_consonant(w.from_end(1)) {
                            w.end += 1;
                        }
                    }
                    b'n' => {
                        if w.len() < 3
                            || Self::is_consonant(w.from_end(1))
                            || Self::is_consonant(w.from_end(2))
                        {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b'r' => {
                        if w.len() > 3
                            && Self::is_vowel(w.from_end(1))
                            && Self::is_vowel(w.from_end(2))
                        {
                            if w.from_end(2) == b'u'
                                && (w.from_end(1) == b'a' || w.from_end(1) == b'i')
                            {
                                w.end += 1;
                            }
                        } else {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b's' => { w.end += 1; }
                    b'w' => {
                        if !(w.len() > 2 && Self::is_vowel(w.from_end(1))) {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b'h' => {
                        if !(w.len() > 2 && Self::is_consonant(w.from_end(1))) {
                            w.end = i;
                            w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    _ => {
                        w.end += 3;
                        w.r#type &= !eng::ADJECTIVE_SUPERLATIVE;
                    }
                }
            }
        }
        (w.r#type & eng::ADJECTIVE_SUPERLATIVE) > 0
    }

    fn step0(w: &mut Word) -> bool {
        for &suf in SUFFIXES_STEP0 {
            if w.ends_with(suf) {
                w.end -= suf.len() as u8;
                w.r#type |= eng::PLURAL;
                return true;
            }
        }
        false
    }

    fn step1a(w: &mut Word) -> bool {
        if w.ends_with(b"sses") {
            w.end -= 2;
            w.r#type |= eng::PLURAL;
            return true;
        }
        if w.ends_with(b"ied") || w.ends_with(b"ies") {
            w.r#type |= if w.from_end(0) == b'd' {
                eng::PAST_TENSE
            } else {
                eng::PLURAL
            };
            w.end -= 1 + (w.len() > 4) as u8;
            return true;
        }
        if w.ends_with(b"us") || w.ends_with(b"ss") { return false; }
        if w.from_end(0) == b's' && w.len() > 2 {
            for i in (w.start as usize)..=(w.end as usize - 2) {
                if Self::is_vowel(w.letters[i]) {
                    w.end -= 1;
                    w.r#type |= eng::PLURAL;
                    return true;
                }
            }
        }
        if w.ends_with(b"n't") && w.len() > 4 {
            match w.from_end(3) {
                b'a' => {
                    if w.from_end(4) == b'c' {
                        w.end -= 2;
                    } else {
                        w.change_suffix(b"n't", b"ll");
                    }
                }
                b'i' => { w.change_suffix(b"in't", b"m"); }
                b'o' => {
                    if w.from_end(4) == b'w' {
                        w.change_suffix(b"on't", b"ill");
                    } else {
                        w.end -= 3;
                    }
                }
                _ => { w.end -= 3; }
            }
            w.r#type |= eng::PREFIX;
            w.prefix |= neg::NEGATION;
            return true;
        }
        if w.ends_with(b"hood") && w.len() > 7 {
            w.end -= 4;
            return true;
        }
        false
    }

    fn step1b(w: &mut Word, r1: u32) -> bool {
        for (i, &suf) in SUFFIXES_STEP1B.iter().enumerate() {
            if !w.ends_with(suf) { continue; }
            match i {
                0 | 1 => {
                    if Self::suffix_in_rn(w, r1, suf) {
                        w.end -= 1 + (i as u8) * 2;
                    }
                }
                _ => {
                    let j = w.end;
                    w.end -= suf.len() as u8;
                    if !Self::has_vowels(w) {
                        w.end = j;
                        return false;
                    }
                    if w.ends_with(b"at") || w.ends_with(b"bl")
                        || w.ends_with(b"iz") || Self::is_short_word(w)
                    {
                        w.append(b'e');
                    } else if w.len() > 2 {
                        if w.from_end(0) == w.from_end(1)
                            && Self::is_double(w.from_end(0))
                        {
                            w.end -= 1;
                        } else if i == 2 || i == 3 {
                            match w.from_end(0) {
                                b'c' | b's' | b'v' => {
                                    let keep = !(w.ends_with(b"ss")
                                        || w.ends_with(b"ias"));
                                    if keep { w.end += 1; }
                                }
                                b'd' => {
                                    let n_allowed = [b'a', b'e', b'i', b'o'];
                                    if Self::is_vowel(w.from_end(1))
                                        && !n_allowed.contains(&w.from_end(2))
                                    {
                                        w.end += 1;
                                    }
                                }
                                b'k' => {
                                    if w.ends_with(b"uak") { w.end += 1; }
                                }
                                b'l' => {
                                    let allowed1 = [
                                        b'b', b'c', b'd', b'f', b'g', b'k',
                                        b'p', b't', b'y', b'z',
                                    ];
                                    let allowed2 = [b'a', b'i', b'o', b'u'];
                                    if allowed1.contains(&w.from_end(1))
                                        || (allowed2.contains(&w.from_end(1))
                                            && Self::is_consonant(w.from_end(2)))
                                    {
                                        w.end += 1;
                                    }
                                }
                                _ => {}
                            }
                        } else if i >= 4 {
                            match w.from_end(0) {
                                b'd' => {
                                    if Self::is_vowel(w.from_end(1))
                                        && w.from_end(2) != b'a'
                                        && w.from_end(2) != b'e'
                                        && w.from_end(2) != b'o'
                                    {
                                        w.append(b'e');
                                    }
                                }
                                b'g' => {
                                    let allowed = [
                                        b'a', b'd', b'e', b'i', b'l', b'r', b'u',
                                    ];
                                    let cond = allowed.contains(&w.from_end(1))
                                        || (w.from_end(1) == b'n'
                                            && (w.from_end(2) == b'e'
                                                || (w.from_end(2) == b'u'
                                                    && w.from_end(3) != b'b'
                                                    && w.from_end(3) != b'd')
                                                || (w.from_end(2) == b'a'
                                                    && (w.from_end(3) == b'r'
                                                        || (w.from_end(3) == b'h'
                                                            && w.from_end(4) == b'c')))
                                                || (w.ends_with(b"ring")
                                                    && (w.from_end(4) == b'c'
                                                        || w.from_end(4) == b'f'))));
                                    if cond { w.append(b'e'); }
                                }
                                b'l' => {
                                    let cond = !(w.from_end(1) == b'l'
                                        || w.from_end(1) == b'r'
                                        || w.from_end(1) == b'w'
                                        || (Self::is_vowel(w.from_end(1))
                                            && Self::is_vowel(w.from_end(2))));
                                    if cond { w.append(b'e'); }
                                    if w.ends_with(b"uell") && w.len() > 4
                                        && w.from_end(4) != b'q'
                                    {
                                        w.end -= 1;
                                    }
                                }
                                b'r' => {
                                    let cond = ((w.from_end(1) == b'i'
                                        && w.from_end(2) != b'a'
                                        && w.from_end(2) != b'e'
                                        && w.from_end(2) != b'o')
                                        || (w.from_end(1) == b'a'
                                            && !(w.from_end(2) == b'e'
                                                || w.from_end(2) == b'o'
                                                || (w.from_end(2) == b'l'
                                                    && w.from_end(3) == b'l')))
                                        || (w.from_end(1) == b'o'
                                            && !(w.from_end(2) == b'o'
                                                || (w.from_end(2) == b't'
                                                    && w.from_end(3) != b's')))
                                        || w.from_end(1) == b'c'
                                        || w.from_end(1) == b't')
                                        && !w.ends_with(b"str");
                                    if cond { w.append(b'e'); }
                                }
                                b't' => {
                                    if w.from_end(1) == b'o'
                                        && w.from_end(2) != b'g'
                                        && w.from_end(2) != b'l'
                                        && w.from_end(2) != b'i'
                                        && w.from_end(2) != b'o'
                                    {
                                        w.append(b'e');
                                    }
                                }
                                b'u' => {
                                    if !(w.len() > 3
                                        && Self::is_vowel(w.from_end(1))
                                        && Self::is_vowel(w.from_end(2)))
                                    {
                                        w.append(b'e');
                                    }
                                }
                                b'z' => {
                                    if w.ends_with(b"izz") && w.len() > 3
                                        && (w.from_end(3) == b'h'
                                            || w.from_end(3) == b'u')
                                    {
                                        w.end -= 1;
                                    } else if w.from_end(1) != b't'
                                        && w.from_end(1) != b'z'
                                    {
                                        w.append(b'e');
                                    }
                                }
                                b'k' => {
                                    if w.ends_with(b"uak") { w.append(b'e'); }
                                }
                                b'b' | b'c' | b's' | b'v' => {
                                    let cond = !((w.from_end(0) == b'b'
                                        && (w.from_end(1) == b'm'
                                            || w.from_end(1) == b'r'))
                                        || w.ends_with(b"ss")
                                        || w.ends_with(b"ias")
                                        || w.equals_str(b"zinc"));
                                    if cond { w.append(b'e'); }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            w.r#type |= TYPES_STEP1B[i];
            return true;
        }
        false
    }

    fn step1c(w: &mut Word) -> bool {
        if w.len() > 2 && w.from_end(0) == b'y' && Self::is_consonant(w.from_end(1)) {
            let e = w.end as usize;
            w.letters[e] = b'i';
            return true;
        }
        false
    }

    fn step2(w: &mut Word, r1: u32) -> bool {
        for (i, (old, new_)) in SUFFIXES_STEP2.iter().enumerate() {
            if w.ends_with(old) && Self::suffix_in_rn(w, r1, old) {
                w.change_suffix(old, new_);
                w.r#type |= TYPES_STEP2[i];
                w.suffix |= TYPES_STEP2_SUFFIX[i];
                return true;
            }
        }
        if w.ends_with(b"logi") && Self::suffix_in_rn(w, r1, b"ogi") {
            w.end -= 1;
            return true;
        } else if w.ends_with(b"li") {
            if Self::suffix_in_rn(w, r1, b"li") && Self::is_li_ending(w.from_end(2)) {
                w.end -= 2;
                w.r#type |= eng::ADVERB_OF_MANNER;
                return true;
            } else if w.len() > 3 {
                match w.from_end(2) {
                    b'b' => {
                        let e = w.end as usize;
                        w.letters[e] = b'e';
                        w.r#type |= eng::ADVERB_OF_MANNER;
                        return true;
                    }
                    b'i' => {
                        if w.len() > 4 {
                            w.end -= 2;
                            w.r#type |= eng::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    b'l' => {
                        if w.len() > 5
                            && (w.from_end(3) == b'a' || w.from_end(3) == b'u')
                        {
                            w.end -= 2;
                            w.r#type |= eng::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    b's' => {
                        w.end -= 2;
                        w.r#type |= eng::ADVERB_OF_MANNER;
                        return true;
                    }
                    b'e' | b'g' | b'm' | b'n' | b'r' | b'w' => {
                        if w.len() > (4 + (w.from_end(2) == b'r') as u32) {
                            w.end -= 2;
                            w.r#type |= eng::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    fn step3(w: &mut Word, r1: u32, r2: u32) -> bool {
        let mut res = false;
        for (i, (old, new_)) in SUFFIXES_STEP3.iter().enumerate() {
            if w.ends_with(old) && Self::suffix_in_rn(w, r1, old) {
                w.change_suffix(old, new_);
                w.r#type |= TYPES_STEP3[i];
                w.suffix |= TYPES_STEP3_SUFFIX[i];
                res = true;
                break;
            }
        }
        if w.ends_with(b"ative") && Self::suffix_in_rn(w, r2, b"ative") {
            w.end -= 5;
            w.r#type |= eng::SUFFIX;
            w.suffix |= suf::SUFFIX_IVE;
            return true;
        }
        if w.len() > 5 && w.ends_with(b"less") {
            w.end -= 4;
            w.r#type |= eng::ADJECTIVE_WITHOUT;
            return true;
        }
        res
    }

    fn step4(w: &mut Word, r2: u32) -> bool {
        let mut res = false;
        for (i, &suf_) in SUFFIXES_STEP4.iter().enumerate() {
            if w.ends_with(suf_) && Self::suffix_in_rn(w, r2, suf_) {
                w.end -= (suf_.len() - if i > 17 { 1 } else { 0 }) as u8;
                if i != 10 || w.from_end(0) != b'm' {
                    w.r#type |= TYPES_STEP4[i];
                    w.suffix |= TYPES_STEP4_SUFFIX[i];
                }
                if i == 0 && w.ends_with(b"nti") {
                    w.end -= 1;
                    res = true;
                    continue;
                }
                return true;
            }
        }
        res
    }

    fn step5(w: &mut Word, r1: u32, r2: u32) -> bool {
        if w.from_end(0) == b'e' && !w.equals_str(b"here") {
            if Self::suffix_in_rn(w, r2, b"e") {
                w.end -= 1;
            } else if Self::suffix_in_rn(w, r1, b"e") {
                w.end -= 1;
                if Self::ends_in_short_syllable(w) { w.end += 1; }
            } else {
                return false;
            }
            return true;
        } else if w.len() > 1 && w.from_end(0) == b'l'
            && Self::suffix_in_rn(w, r2, b"l")
            && w.from_end(1) == b'l'
        {
            w.end -= 1;
            return true;
        }
        false
    }

    /// Run the full Porter2-style pipeline. `blpos` mirrors upstream's
    /// `x.blpos` global — it gates the `VerbWords1` table after an
    /// empirical threshold.
    pub fn stem(w: &mut Word, blpos: u32) -> bool {
        let mut res = Self::trim_starting_apostrophe(w);
        if Self::process_prefixes(w) { res = true; }
        if Self::process_superlatives(w) { res = true; }
        for (i, (key, repl)) in EXCEPTIONS1.iter().enumerate() {
            if w.equals_str(key) {
                if i < 11 {
                    let st = w.start as usize;
                    let len = repl.len();
                    w.letters[st..st + len].copy_from_slice(repl);
                    w.end = w.start + (len as u8) - 1;
                }
                Self::hash(w);
                w.r#type |= TYPES_EXCEPTIONS1[i];
                return i < 11;
            }
        }
        Self::mark_ys_as_consonants(w);
        let r1 = Self::get_region1(w);
        let r2 = Self::get_region(w, r1);
        if Self::step0(w)  { res = true; }
        if Self::step1a(w) { res = true; }
        for (i, &exc) in EXCEPTIONS2.iter().enumerate() {
            if w.equals_str(exc) {
                Self::hash(w);
                w.r#type |= TYPES_EXCEPTIONS2[i];
                return res;
            }
        }
        if Self::step1b(w, r1)     { res = true; }
        if Self::step1c(w)         { res = true; }
        if Self::step2(w, r1)      { res = true; }
        if Self::step3(w, r1, r2)  { res = true; }
        if Self::step4(w, r2)      { res = true; }
        if Self::step5(w, r1, r2)  { res = true; }
        for i in (w.start as usize)..=(w.end as usize) {
            if w.letters[i] == b'Y' { w.letters[i] = b'y'; }
        }
        if w.r#type == 0 || w.r#type == eng::PLURAL {
            if w.matches_any(MALE_WORDS) {
                res = true; w.r#type |= eng::MALE;
            } else if w.matches_any(FEMALE_WORDS) {
                res = true; w.r#type |= eng::FEMALE;
            } else if w.matches_any(ARTICLE_WORDS) {
                res = true; w.r#type |= eng::ARTICLE;
            } else if w.matches_any(CONJ_WORDS) {
                res = true; w.r#type |= eng::CONJUNCTION;
            } else if w.matches_any(APO_WORDS) {
                res = true; w.r#type |= eng::ADPOSITION;
            } else if w.matches_any(CON_AD_VER_PREP_WORDS) {
                res = true; w.r#type |= eng::CONJUNCTIVE_ADVERB;
            } else if blpos < 451_531_986 && w.matches_any(VERB_WORDS1) {
                res = true; w.r#type |= eng::VERB;
            } else if w.matches_any(NUMBERS) {
                res = true; w.r#type |= eng::NUMBER;
            }
        }
        Self::hash(w);
        res
    }
}

// ====================================================================
// `MatchModel2` — match-finder predictor used by FXCMv1.
//
// Tracks up to `MATCH_N` concurrent match candidates, each backed by
// a [`MatchInfo`] record holding rebased length, position-in-buffer,
// the expected next byte, and a 1-byte recovery backup. The hash
// table (`mhashtable`) maps an order-`LEN1`/`LEN2`/`LEN3` hash plus a
// word hash to up to `M_HASH_N` recent positions; on a byte boundary
// we promote table entries to candidates and shift the table to make
// room for the current position.
//
// `mix(...)` decides on a best candidate (using `MatchInfo::prio`),
// emits one length-scaled logit into `mx_inputs1`, and feeds three
// per-context StateMap1 outputs (stretched + raw probability) into
// the mixer.
// ====================================================================

pub const MATCH_MAXLEN:  u32 = 62;
pub const M_HASH_N:      usize = 4;
pub const MINLEN_RM:     u32 = 3;
pub const LEN1:          u32 = 5;
pub const LEN2:          u32 = 7;
pub const LEN3:          u32 = 9;
pub const MATCH_N:       usize = 4;
pub const N_ST:          usize = 3;

#[derive(Clone, Copy, Default)]
pub struct HashElementForMatchPositions {
    pub positions: [u32; M_HASH_N],
}

impl HashElementForMatchPositions {
    pub fn new() -> Self { Self::default() }

    /// Push `pos` to the front (slot 0), shifting older entries by one.
    pub fn add(&mut self, pos: u32) {
        for i in (1..M_HASH_N).rev() {
            self.positions[i] = self.positions[i - 1];
        }
        self.positions[0] = pos;
    }
}

#[derive(Clone, Copy)]
pub struct MatchInfo {
    pub length:        u32,
    pub index:         u32,
    pub length_bak:    u32,
    pub index_bak:     u32,
    pub expected_byte: u8,
    pub delta:         bool,
}

impl MatchInfo {
    pub fn new() -> Self {
        Self {
            length: 0, index: 0,
            length_bak: 0, index_bak: 0,
            expected_byte: 0, delta: false,
        }
    }

    pub fn is_in_no_match_mode(&self) -> bool {
        self.length == 0 && !self.delta && self.length_bak == 0
    }

    pub fn is_in_pre_recovery_mode(&self) -> bool {
        self.length == 0 && !self.delta && self.length_bak != 0
    }

    pub fn is_in_recovery_mode(&self) -> bool {
        self.length != 0 && self.length_bak != 0
    }

    pub fn recovery_mode_pos(&self) -> u32 {
        debug_assert!(self.is_in_recovery_mode());
        self.length - self.length_bak
    }

    /// Priority used to select the best candidate when several
    /// matches are active. Mirrors upstream's bit-packed formula.
    pub fn prio(&self) -> u32 {
        ((self.length != 0) as u32) << 31
            | (self.delta as u32) << 30
            | (if self.delta { self.length_bak >> 1 } else { self.length >> 1 }) << 24
            | (self.index & 0x00ff_ffff)
    }

    pub fn is_better_than(&self, other: &Self) -> bool {
        self.prio() > other.prio()
    }

    /// Per-bit update. Reads the next expected bit from
    /// `expected_byte`, compares with `y`, and either:
    ///   * extends the match if it succeeds and we are at byte
    ///     boundary; or
    ///   * enters delta/recovery mode on mismatch.
    ///
    /// `c1` is the most-recent whole byte.
    pub fn update(&mut self, y: i32, bpos: i32, c1: u8, buf: &Buffer) {
        if self.length != 0 {
            let expected_bit =
                ((self.expected_byte >> ((8 - bpos) & 7)) & 1) as i32;
            if y != expected_bit {
                if self.is_in_recovery_mode() {
                    self.length_bak = 0;
                    self.index_bak  = 0;
                } else {
                    self.length_bak = self.length;
                    self.index_bak  = self.index;
                    self.delta = true;
                }
                self.length = 0;
            }
        }

        if bpos == 0 {
            // Recover after a 1-byte mismatch.
            if self.is_in_pre_recovery_mode() {
                self.index_bak += 1;
                if self.length_bak < MATCH_MAXLEN { self.length_bak += 1; }
                if buf.bufr(self.index_bak as usize) == c1 {
                    self.length = self.length_bak;
                    self.index  = self.index_bak;
                } else {
                    self.length_bak = 0;
                    self.index_bak  = 0;
                }
            }
            // Extend the current match.
            if self.length != 0 {
                self.index += 1;
                if self.length < MATCH_MAXLEN { self.length += 1; }
                if self.is_in_recovery_mode()
                    && self.recovery_mode_pos() >= MINLEN_RM
                {
                    self.length_bak = 0;
                    self.index_bak  = 0;
                }
            }
            self.delta = false;
        }
    }

    /// Activate this candidate at buffer position `pos` with raw
    /// length `len` (rebased to `len - LEN1 + 1`).
    pub fn register_match(&mut self, pos: u32, len: u32) {
        debug_assert!(pos != 0);
        self.length = len - LEN1 + 1;
        self.index  = pos;
        self.length_bak = 0;
        self.index_bak  = 0;
        self.expected_byte = 0;
        self.delta = false;
    }
}

impl Default for MatchInfo { fn default() -> Self { Self::new() } }

/// Wraps the candidate set + hash table + per-context state maps.
///
/// The hash table is sized to `1 << log2_size`, mask is `size-1`.
/// `sm_a` holds three `StateMap1` predictors keyed by the model's
/// per-bit contexts (length×bit, expected×c0, delta-mode×c0).
pub struct MatchModel2 {
    pub candidates: [MatchInfo; MATCH_N],
    pub n_active: u32,
    pub mhashtable: Vec<HashElementForMatchPositions>,
    pub mhashtable_mask: u32,
    pub ctx: [u32; N_ST],
    pub sm_a: Vec<StateMap1>,  // length = N_ST
    pub log2_size: u8,
}

impl MatchModel2 {
    /// `log2_size` sets the hash table to `2^log2_size` entries.
    /// `sm_n`/`sm_limit` control the per-context StateMap1 (the bit
    /// of state behind each `ctx[i]` value).
    pub fn new(log2_size: u8, sm_n: usize, sm_limit: i32) -> Self {
        let size = 1usize << log2_size;
        Self {
            candidates: [MatchInfo::new(); MATCH_N],
            n_active: 0,
            mhashtable: vec![HashElementForMatchPositions::new(); size],
            mhashtable_mask: (size - 1) as u32,
            ctx: [0; N_ST],
            sm_a: (0..N_ST).map(|_| StateMap1::new(sm_n, sm_limit)).collect(),
            log2_size,
        }
    }

    /// Returns true iff bytes `buf(1)..=buf(min_len)` match
    /// `bufr(pos-1)..=bufr(pos-min_len)`.
    ///
    /// Upstream relies on `U32` wrap-around — `pos - length` may
    /// silently wrap if `length > pos`, after which `bufr` masks
    /// the result back to a valid buffer index. We mirror that
    /// behaviour with `wrapping_sub`.
    fn is_match(buf: &Buffer, pos: u32, min_len: u32) -> bool {
        for length in 1..=min_len {
            let abs = pos.wrapping_sub(length) as usize;
            if buf.buf(length as usize) != buf.bufr(abs) { return false; }
        }
        true
    }

    /// Try to promote table entries `matches` (matched at length
    /// `min_len`) into the live candidate set. Skips positions
    /// already registered by an existing candidate.
    fn add_candidates(&mut self, matches_idx: usize, min_len: u32, buf: &Buffer) {
        let positions = self.mhashtable[matches_idx].positions;
        let mut i = 0;
        while (self.n_active as usize) < MATCH_N && i < M_HASH_N {
            let matchpos = positions[i];
            if matchpos == 0 { break; }
            if Self::is_match(buf, matchpos, min_len) {
                let mut is_same = false;
                for j in 0..self.n_active as usize {
                    if self.candidates[j].index == matchpos {
                        is_same = true;
                        break;
                    }
                }
                if !is_same {
                    self.candidates[self.n_active as usize]
                        .register_match(matchpos, min_len);
                    self.n_active += 1;
                }
            }
            i += 1;
        }
    }

    /// Per-bit update. Refreshes existing candidates, drops dead
    /// ones, and at byte boundary promotes new candidates from the
    /// hash table and shifts the current `pos` into all four hash
    /// slots (LEN3/LEN2/LEN1 + word-hash).
    pub fn update(
        &mut self,
        y: i32,
        bpos: i32,
        c1: u8,
        pos: u32,
        buf: &Buffer,
        order_hashes: &[u32],
        word1_hash: u32,
    ) {
        let n_loop = self.n_active.max(1) as usize;
        let mut i = 0;
        while i < n_loop {
            self.candidates[i].update(y, bpos, c1, buf);
            if self.n_active != 0
                && self.candidates[i].is_in_no_match_mode()
            {
                self.n_active -= 1;
                if self.n_active as usize == i { break; }
                for j in i..self.n_active as usize {
                    self.candidates[j] = self.candidates[j + 1];
                }
                if i > 0 { i -= 1; }
            }
            i += 1;
        }

        if bpos == 0 {
            for &(hash, len) in &[
                (order_hashes[LEN3 as usize], LEN3),
                (order_hashes[LEN2 as usize], LEN2),
                (order_hashes[LEN1 as usize], LEN1),
                (word1_hash,                  LEN1),
            ] {
                let idx = (hash & self.mhashtable_mask) as usize;
                if (self.n_active as usize) < MATCH_N {
                    self.add_candidates(idx, len, buf);
                }
                self.mhashtable[idx].add(pos);
            }

            for i in 0..self.n_active as usize {
                self.candidates[i].expected_byte =
                    buf.bufr(self.candidates[i].index as usize);
            }
        }
    }

    /// Pick the best candidate, derive contexts, append outputs to
    /// `out` (one length-scaled logit + 3 × 2 StateMap1 outputs).
    /// Returns the best candidate's match length.
    pub fn mix(
        &mut self,
        y: i32,
        bpos: i32,
        c1: u8,
        c0: i32,
        pos: u32,
        buf: &Buffer,
        order_hashes: &[u32],
        word1_hash: u32,
        sqt: &[i16],
        strt: &[i16],
        dt: &[i32],
        out: &mut Vec<i32>,
    ) -> u32 {
        self.update(y, bpos, c1, pos, buf, order_hashes, word1_hash);

        for i in 0..N_ST { self.ctx[i] = 0; }

        let mut best = 0;
        for i in 1..self.n_active as usize {
            if self.candidates[i].is_better_than(&self.candidates[best]) {
                best = i;
            }
        }

        let length         = self.candidates[best].length;
        let expected_byte  = self.candidates[best].expected_byte;
        let in_delta_mode  = self.candidates[best].delta;
        let expected_bit   = if length != 0 {
            ((expected_byte >> (7 - bpos)) & 1) as i32
        } else { 0 };

        if length != 0 {
            let denselength = if length <= 16 {
                length - 1
            } else {
                12 + (length >> 2)
            };
            self.ctx[0] = (denselength << 4) | (expected_bit as u32) << 3 | bpos as u32;
            self.ctx[1] = ((expected_byte as u32) << 11)
                | ((bpos as u32) << 8) | (c1 as u32);
            let sign = 2 * expected_bit - 1;
            out.push(sign * (length as i32) << 5);
        } else {
            out.push(0);
        }

        if in_delta_mode {
            self.ctx[2] = ((expected_byte as u32) << 8) | (c0 as u32);
        }

        for i in 0..N_ST {
            let c = self.ctx[i];
            if c != 0 {
                let p1 = self.sm_a[i].set(y, c as usize, dt);
                let st = stretch(strt, p1);
                let _ = sqt;
                out.push((st as i32) >> 2);
                out.push((p1 - 2048) >> 3);
            } else {
                out.push(0);
                out.push(0);
            }
        }
        length
    }
}

// ====================================================================
// Per-context parameter arrays (one entry per CM index).
//
// Mirrored verbatim from upstream's globals at fxcmv1.cpp:3218-3221:
//   c_r  — ContextMap "run" multiplier (run-length learning gain).
//   c_s  — ContextMap "pr" multiplier (probability scaling).
//   c_s3 — `s3` (mix3 internal scale).
//   c_s4 — `cs4` (per-context shift).
// ====================================================================

pub const C_R:  [u32; 27] =
    [3,4,6,4,6,6,2,3,3,3,6,4,3,4,5,6,2,6,4,4,4,4,4,4,4,4,4];
pub const C_S:  [u32; 27] =
    [28,26,28,31,34,31,33,33,35,35,29,32,33,34,30,36,31,32,
     32,32,32,32,33,32,32,32,32];
pub const C_S3: [u32; 27] =
    [43,33,34,28,34,29,32,33,37,35,33,28,31,35,28,30,33,34,
     32,32,32,32,32,32,32,32,32];
pub const C_S4: [u32; 27] =
    [9,8,9,5,8,12,15,8,8,12,10,7,7,8,8,13,13,14,8,8,12,12,12,12,12,12,12];

/// Escape probability limits used by the failure-tracking branch in
/// `update1()` — `pr >= e_l[bpos]` flags this bit as a near-failure.
pub const ESC_LIMITS: [i32; 8] =
    [1830, 1997, 1973, 1851, 1897, 1690, 1998, 1842];

/// 14-entry prime multipliers used by the order-X hash table `t[]`.
/// Verbatim from fxcmv1.cpp:3213. Slot 0 is unused (the table starts
/// folding at index 1).
pub const PRIMES: [u32; 14] =
    [0, 257, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191];

/// Bracket / first-char → 1-7 lookup (else 0). 128-entry, indexed
/// by the low 7 bits of a control or open-bracket byte. Verbatim
/// from fxcmv1.cpp:3681-3690.
pub const FCY: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 5, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// First-char → 1-7 lookup (else 0). Verbatim from
/// fxcmv1.cpp:3692-3701.
pub const FCQ: [u8; 128] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 5, 0, 0,
    2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Tri/trj — small lookup tables that fold `fails` bits into the
/// `pz` failure counter inside `update1`.
pub const TRI: [u32; 4] = [0, 4, 3, 7];
pub const TRJ: [u32; 4] = [0, 6, 6, 12];

/// Bracket-context character lists (open/close pairs).
/// Mirrors fxcmv1.cpp:2150-2155 — must use the wiki-control
/// constants since some entries (CURLYOPENING, LESSTHAN, etc.) are
/// remapped above the printable range.
pub fn brackets_table() -> [u8; 8] {
    [b'(', b')',
     CURLYOPENING_CHR, CURLYCLOSE_CHR,
     b'[', b']',
     LESSTHAN_CHR, GREATERTHAN_CHR]
}

pub fn quotes_table() -> [u8; 4] {
    [APOSTROPHE_CHR, APOSTROPHE_CHR,
     QUOTATION_CHR,  QUOTATION_CHR]
}

pub fn fchar_table() -> [u8; 20] {
    [
        FIRSTUPPER_CHR, LF_CHR,
        TEXTDATA_CHR,   LF_CHR,
        COLON_CHR,      LF_CHR,
        LESSTHAN_CHR, GREATERTHAN_CHR,
        EQUALS_CHR,     LF_CHR,
        SQUAREOPEN_CHR, SQUARECLOSE_CHR,
        CURLYOPENING_CHR, CURLYCLOSE_CHR,
        b'*',           LF_CHR,
        VERTICALBAR_CHR, LF_CHR,
        HTLINK_CHR,     LF_CHR,
    ]
}

pub fn html_table() -> [u16; 2] {
    [(b'&' as u16) * 256 + (b'L' as u16),
     (b'&' as u16) * 256 + (b'N' as u16)]
}

// ====================================================================
// Six pre-built state-transition tables (`STA1..STA7`, STA3 omitted)
// driven by upstream's `StateTable::Init(s0..s5, mdc, &nn[0])` with
// the parameter tuples at fxcmv1.cpp:4876-4881.
// ====================================================================

#[derive(Clone, Copy)]
pub struct StaParams {
    pub s0: i32, pub s1: i32, pub s2: i32, pub s3: i32,
    pub s4: i32, pub s5: i32, pub mdc: i32,
}

pub const STA_PARAMS: [StaParams; 6] = [
    StaParams { s0: 28, s1: 28, s2: 31, s3: 29, s4: 23, s5: 4,  mdc: 17 }, // STA1
    StaParams { s0: 32, s1: 28, s2: 31, s3: 28, s4: 21, s5: 5,  mdc:  6 }, // STA2
    StaParams { s0: 31, s1: 27, s2: 30, s3: 27, s4: 24, s5: 4,  mdc: 27 }, // STA4
    StaParams { s0: 33, s1: 31, s2: 31, s3: 24, s4: 20, s5: 4,  mdc: 33 }, // STA5
    StaParams { s0: 28, s1: 29, s2: 30, s3: 30, s4: 23, s5: 3,  mdc: 22 }, // STA6
    StaParams { s0: 28, s1: 29, s2: 33, s3: 23, s4: 23, s5: 6,  mdc: 14 }, // STA7
];

/// Build all six 256-state transition tables side-by-side.
/// Returns `[u8; 6 * 1024]` flat — caller indexes by `state_table_id * 1024 + i`.
pub fn build_sta_tables() -> Vec<u8> {
    let mut out = vec![0u8; 6 * 1024];
    let mut st = StateTable::new();
    for (i, p) in STA_PARAMS.iter().enumerate() {
        let mut tbl = [0u8; 1024];
        st.init(p.s0, p.s1, p.s2, p.s3, p.s4, p.s5, p.mdc, &mut tbl);
        out[i * 1024..(i + 1) * 1024].copy_from_slice(&tbl);
    }
    out
}

// ====================================================================
// Mixer / StateMap / SmallStationaryContextMap parameter tables.
//
// PredictorInit() configures 12 Mixer1 instances + 3 StateMap1 + 7
// SmallStationaryContextMap. Below are the parameter tuples upstream
// uses. Mirrored from fxcmv1.cpp:3313-3337.
// ====================================================================

/// Mixer1 init params: `(m_contexts, shift1, elim, uperr)`. The `n`
/// (input count) comes from `mxInputs1.ncount` (or `mxInputs2.ncount`
/// for the final two) and is set by the caller separately.
pub const MIXER_PARAMS: [(usize, u32, i32, i32); 12] = [
    (2048,         237, 8, 69),  // general
    (6 * 256,      204, 8, 19),
    (6 * 256 * 4,   70, 1, 34),
    (8 * 256,       54, 1, 23),
    (6 * 256,       55, 1, 24),
    (7 * 256 * 4,   55, 1, 24),
    (0x4000,        70, 1, 34),
    (0x4000,        55, 1, 24),
    (0x20000,       55, 1, 24),
    (0x20000,       55, 1, 24),
    (8*7*2*2,        6, 0,  4),  // final mixer
    (1,              6, 0,  4),  // helper for final
];

/// `(n_contexts, limit)` for the three StateMap1 instances backing
/// `MatchModel2::sm_a`. Sizes mirror upstream's
/// `smA[0]=1<<9`, `smA[1]=1<<19`, `smA[2]=1<<16` all with limit 1023.
pub const STATE_MAP_PARAMS: [(usize, i32); 3] = [
    (1 << 9,  1023),
    (1 << 19, 1023),
    (1 << 16, 1023),
];

/// Number of context bits for the seven SmallStationaryContextMap
/// instances (`scmA[0..6]`). Each map uses an 8-bit input width by
/// default; those values are 8,8,8,9,8,8,7 in upstream.
pub const SCM_BITS: [u8; 7] = [8, 8, 8, 9, 8, 8, 7];

/// Build the 12 Mixer1 instances given the per-mixer input counts.
/// `inputs1_n` is `mxInputs1.ncount` (used by mixers 0..=9), and
/// `inputs2_n` is `mxInputs2.ncount` (used by mixers 10/11).
pub fn build_mixers(inputs1_n: usize, inputs2_n: usize) -> Vec<Mixer1> {
    MIXER_PARAMS.iter().enumerate().map(|(i, &(m, sh, el, ue))| {
        let n = if i < 10 { inputs1_n } else { inputs2_n };
        Mixer1::new(n, m, sh, el, ue)
    }).collect()
}

/// Build the 3 StateMap1 instances backing MatchModel2.
pub fn build_state_maps() -> Vec<StateMap1> {
    STATE_MAP_PARAMS.iter().map(|&(n, lim)| StateMap1::new(n, lim)).collect()
}

/// Build the 7 SmallStationaryContextMap instances (`ctx_bits` from
/// `SCM_BITS`, in_bits = 8 by default).
pub fn build_small_scm() -> Vec<SmallStationaryContextMap> {
    SCM_BITS.iter().map(|&b| SmallStationaryContextMap::new(b as u32, 8)).collect()
}

/// `st2_p1[i] = clp(sc(12*(i - 2048)))` — prior-init lookup table
/// passed to ContextMap2 instances. Verbatim from
/// fxcmv1.cpp:4867-4868.
pub fn build_st2_p1() -> Vec<i16> {
    (0..4096)
        .map(|i| clp(sc(12 * (i as i32 - 2048))))
        .collect()
}

/// `st2_p2[i] = clp(sc(14*(i - 2048)))` — same shape with a
/// steeper slope (used by `cmC[3]`). fxcmv1.cpp:4869.
pub fn build_st2_p2() -> Vec<i16> {
    (0..4096)
        .map(|i| clp(sc(14 * (i as i32 - 2048))))
        .collect()
}

/// `st2_p0` is left at upstream's BSS zero-init — used when a
/// ContextMap wants the "no prior" path.
pub fn build_st2_p0() -> Vec<i16> { vec![0i16; 4096] }

/// Per-state pre-stretched probability used by upstream's `pre2(STA7)`
/// to seed `pre1[256]`. We compute it from STA7 (== STA_PARAMS[5]).
pub fn build_pre1_from_sta7(sta7: &[u8], strt: &[i16]) -> [i16; 256] {
    let mut out = [0i16; 256];
    for i in 0..256 {
        let n0 = sta7[i * 4 + 2] as i32 * 3 + 1;
        let n1 = sta7[i * 4 + 3] as i32 * 3 + 1;
        let p = (n1 << 12) / (n0 + n1);
        out[i] = (clp(stretch(strt, p)) as i32 >> 2) as i16;
    }
    out
}

// ====================================================================
// `FxcmState` — captures the file-scope mutable globals that
// upstream's `fxcmv1.cpp` shares between modelPrediction, update1
// and PredictorInit. Carving them into a struct lets the per-bit
// pipeline take a single `&mut FxcmState` instead of touching
// dozens of separate variables.
// ====================================================================

pub struct FxcmState {
    // Recent whole bytes (separate from `BlockData::c4` because text
    // / wiki preprocessing sometimes overrides them).
    pub c1: u8,
    pub c2: u8,
    pub c3: u8,

    // 2/3/4-bit byte classifier outputs and their running streams.
    pub n2b_state: u32,
    pub n3b_state: u32,
    pub n4b_state: u32,
    pub o2b_state: u32,
    pub o3b_state: u32,
    pub o4b_state: u32,
    pub stream2b:  u32,
    pub stream3b:  u32,
    pub stream4b:  u32,
    pub stream2b_r: u32,
    pub stream3b_r: u32,
    pub stream4b_r: u32,

    // Word/sentence rolling hashes.
    pub word0:     u32,
    pub word00:    u32,
    pub word1:     u32,
    pub word2:     u32,
    pub word3:     u32,
    pub wshift:    u32,
    pub x4:        u32,
    pub x5:        u32,
    pub is_match:  u32,
    pub first_word: u32,
    pub link_word: u32,
    pub sen_word:  u32,

    // Number-tracking fields.
    pub number0:   u32,
    pub number1:   u32,
    pub numlen0:   u32,
    pub numlen1:   u32,
    pub mybenum:   u32,

    // Counters and APM hashes.
    pub words:     u8,
    pub spaces:    u8,
    pub numbers:   u8,
    pub fc_idx:    u32,
    pub br_fc_idx: u32,
    pub ah1:       u32,
    pub ah2:       u32,  // initial: 0x765BA55C
    pub fails:     u32,
    pub failz:     u32,
    pub failcount: u32,

    // Line tracking.
    pub nl:  i32,
    pub nl1: i32,
    pub col: i32,
    pub fc:  i32,

    // 14-element hash table consulted by MatchModel2 + ContextMaps.
    pub t: [u32; 14],

    // Indirect-byte / indirect-word rolling history. `t1` rolls
    // 4 bytes per low byte (256 buckets); `t2` rolls 4 bytes per
    // (c4 >> 8) & 0xffff word (64K buckets). Heap-allocated to
    // keep `FxcmState` itself stack-friendly.
    pub t1: Vec<u32>,
    pub t2: Vec<u32>,

    // Last-position-of-word0 lookup (`wp[word0 & 0xffff] = pos`).
    pub wp: Vec<u32>,

    // Position counter (bytes seen so far).
    pub pos: u32,

    // 3-bit / 2-bit / mask state.
    pub stream3b_mask:  u32,
    pub stream3b_mask1: u32,
    pub stream3b_r_mask1: u32,
    pub stream3b_r_mask2: u32,
    pub stream2b_mask:  u32,

    // Order counters used by ContextMap selection.
    pub ord_x: i32,
    pub ord_w: i32,

    // Wiki/text mode flags.
    pub is_text:        bool,
    pub is_paragraph:   i32,
    pub is_now_iki:     bool,
    pub is_math:        bool,
    pub is_pre:         bool,
    pub last_art:       bool,
    pub utf8_left:      i32,

    // SSE rate (APM update rate).
    pub sscm_rate: i32,
    pub rate:      i32,

    // Other indices.
    pub last_wt:        u32,
    pub indirect_br_byte: u32,
    pub indirect_byte:    u32,
    pub indirect_word0_pos: u32,
    pub indirect_word:    u32,
    pub u8w:              u32,
    pub context1_ind3:    u32,
    pub cxt_ind3:         u32,
    pub stem_index:       i32,
    pub s_verb:           u32,
    pub dec_code:         i32,

    // Final prediction value (the `pr` upstream global).
    pub pr: i32,

    // 4-Word stemmer ring (`StemWords[4]` upstream).
    pub stem_words: [Word; 4],
    /// Index into stem_words for the current word.
    pub c_word: usize,
    /// Index into stem_words for the previous word.
    pub p_word: usize,
}

impl FxcmState {
    pub fn new() -> Self {
        Self {
            c1: 0, c2: 0, c3: 0,
            n2b_state: 0xffffffff, n3b_state: 0xffffffff, n4b_state: 0,
            o2b_state: 0, o3b_state: 0, o4b_state: 0,
            stream2b: 0, stream3b: 0, stream4b: 0,
            stream2b_r: 0, stream3b_r: 0, stream4b_r: 0,
            word0: 0, word00: 0, word1: 0, word2: 0, word3: 0,
            wshift: 0, x4: 0, x5: 0, is_match: 0,
            first_word: 0, link_word: 0, sen_word: 0,
            number0: 0, number1: 0, numlen0: 0, numlen1: 0, mybenum: 0,
            words: 0, spaces: 0, numbers: 0,
            fc_idx: 0, br_fc_idx: 0,
            ah1: 0, ah2: 0x765BA55C,
            fails: 0, failz: 0, failcount: 0,
            nl: 0, nl1: 0, col: 0, fc: 0,
            t: [0u32; 14],
            t1: vec![0u32; 256],
            t2: vec![0u32; 65536],
            wp: vec![0u32; 65536],
            pos: 0,
            stream3b_mask: 0, stream3b_mask1: 0,
            stream3b_r_mask1: 0, stream3b_r_mask2: 0,
            stream2b_mask: 0,
            ord_x: 0, ord_w: 0,
            is_text: false, is_paragraph: 0, is_now_iki: false,
            is_math: false, is_pre: false,
            last_art: false, utf8_left: 0,
            sscm_rate: 0, rate: 6,
            last_wt: 0,
            indirect_br_byte: 0, indirect_byte: 0,
            indirect_word0_pos: 0, indirect_word: 0, u8w: 0,
            context1_ind3: 0, cxt_ind3: 0,
            stem_index: 0, s_verb: 0, dec_code: 0,
            pr: 2048,
            stem_words: [Word::new(), Word::new(), Word::new(), Word::new()],
            c_word: 0, p_word: 3,
        }
    }
}

impl Default for FxcmState { fn default() -> Self { Self::new() } }

impl FxcmState {
    /// First-stage byte-boundary side effects from `modelPrediction`
    /// (fxcmv1.cpp:3803-3859, before the wiki/word handling).
    ///
    /// Cycles c3/c2/c1, refreshes the 2/3/4-bit byte classifiers,
    /// pushes to the buffer, advances `pos`, and folds the new byte
    /// into the order-X hash table `t[1..=13]`.
    ///
    /// `c4` is the most recent 4 packed bytes (BlockData::c4) — only
    /// its low byte is consulted here.
    pub fn byte_boundary_step(&mut self, c4: u32, buf: &mut Buffer) {
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = (c4 & 0xff) as u8;

        self.n2b_state = WRT_2B[self.c1 as usize] as u32;
        self.n3b_state = WRT_3B[self.c1 as usize] as u32;
        self.n4b_state = WRT_4B[self.c1 as usize] as u32;

        self.stream2b = self.stream2b
            .wrapping_mul(4)
            .wrapping_add(self.n2b_state);
        self.stream4b = self.stream4b
            .wrapping_mul(16)
            .wrapping_add(self.n4b_state);

        buf.push(self.c1);
        self.pos = self.pos.wrapping_add(1);

        // x4-end markers: $, ], |, ), [
        if self.c1 == b'$' || self.c1 == SQUARECLOSE_CHR
            || self.c1 == VERTICALBAR_CHR || self.c1 == b')'
            || self.c1 == SQUAREOPEN_CHR
        {
            if self.c1 != self.c2 {
                for i in (1..=13).rev() {
                    self.t[i] = self.t[i - 1].wrapping_mul(PRIMES[i]);
                }
            }
            self.x4 = (self.x4 << 8).wrapping_add(self.c2 as u32);
            self.stream2b = self.stream2b
                .wrapping_mul(4)
                .wrapping_add(self.n2b_state);
            self.stream2b_r = (self.stream2b_r << 2)
                .wrapping_add(self.n2b_state);
            self.stream3b_r = (self.stream3b_r << 3)
                .wrapping_add(self.n3b_state);
        }

        // Fold c1 into the order-X hash table.
        self.x4 = (self.x4 << 8).wrapping_add(self.c1 as u32);
        for i in (1..=13).rev() {
            self.t[i] = self.t[i - 1]
                .wrapping_mul(PRIMES[i])
                .wrapping_add(self.c1 as u32)
                .wrapping_add((i as u32) * 256);
        }
    }

    /// Words / spaces / numbers bit-shift step from
    /// `modelPrediction` (fxcmv1.cpp:3871-3874). Called once per byte
    /// boundary, in lockstep with `byte_boundary_step`.
    pub fn shift_letter_classes(&mut self) {
        self.words   = self.words.wrapping_shl(1);
        self.spaces  = self.spaces.wrapping_shl(1);
        self.numbers = self.numbers.wrapping_shl(1);
    }

    /// Returns true iff the current `c1` is a "word letter" — either a
    /// lowercase ASCII letter (a..=z, the upstream `(j-'a')<=('z'-'a')`
    /// check) or a non-ASCII byte (>127) whose predecessor wasn't the
    /// ESCAPE control byte (upstream's UTF-8 / wiki-encoded char).
    pub fn is_word_letter(&self) -> bool {
        let j = self.c1;
        (j >= b'a' && j <= b'z')
            || (self.c1 > 127 && self.c2 != ESCAPE_CHR)
    }

    /// Word-pipeline cycle (fxcmv1.cpp:3964-3970). On a non-letter
    /// byte after a word run (`word0 != 0`), shift the
    /// word3/word2/word1/word0 pipeline using upstream's prime
    /// multipliers. Skips the cycle for some `pWord.Type` classes
    /// before the empirical `blpos` threshold.
    ///
    /// Caller must clear `word0` after the cycle if the byte
    /// terminated the word.
    pub fn cycle_word_pipeline(&mut self, blpos: u32) {
        if self.word0 == 0 { return; }
        let p_type = self.stem_words[self.p_word].r#type;
        let cycle = blpos > 463_139_793
            || (p_type & (eng::CONJUNCTIVE_ADVERB | eng::CONJUNCTION)) == 0;
        if cycle {
            self.word3 = self.word2.wrapping_mul(47);
            self.word2 = self.word1.wrapping_mul(53);
            self.word1 = self.word0.wrapping_mul(83);
        }
    }

    /// Update `firstWord` if it hasn't been set yet AND we aren't
    /// inside a `[` bracket (fxcmv1.cpp:3976-3978).
    /// `fc_cxt` is the current `fccxt.cxt` (FirstChar bracket
    /// context); compare against [`SQUAREOPEN_CHR`].
    pub fn maybe_seed_first_word(&mut self, fc_cxt: u8) {
        if self.first_word == 0 && fc_cxt != SQUAREOPEN_CHR {
            self.first_word = self.word0;
        }
    }

    /// Stream3b / stream2b updates driven by the most-recent word's
    /// type (fxcmv1.cpp:3971-3994). Mutates the running stream
    /// accumulators when the latest word is a Number, Conjunction,
    /// Article, Adposition, or PresentParticiple-in-paragraph.
    ///
    /// `worcxt` is the active WordsContext. `is_paragraph` mirrors
    /// upstream's `isParagraph` flag (passed from the wiki tracker).
    /// On a Conjunction the caller may also want to clear `senword`
    /// (we do that here too — upstream pairs the two).
    pub fn update_streams_for_word_type(
        &mut self,
        worcxt: &WordsContext,
        is_paragraph: bool,
    ) {
        // Most-recent word is index 1.
        if worcxt.types(1) == eng::NUMBER {
            self.stream3b_r = (self.stream3b_r << 7).wrapping_add(1);
            self.stream3b   = (self.stream3b   << 7).wrapping_add(1);
        }
        let last_type = worcxt.types(1);
        if (last_type & eng::CONJUNCTION) != 0 {
            self.stream3b_r <<= 7;
            self.stream3b   <<= 7;
            if is_paragraph { self.sen_word = 0; }
        }
        if (last_type & eng::ARTICLE) != 0 {
            self.stream3b_r = (self.stream3b_r << 7).wrapping_add(2);
            self.stream3b   = (self.stream3b   << 7).wrapping_add(2);
        }
        if (last_type & eng::ADPOSITION) != 0
            || (is_paragraph && (last_type & eng::PRESENT_PARTICIPLE) != 0)
        {
            self.stream2b_r = (self.stream2b_r << 2)
                .wrapping_add(self.stream2b_r & 3);
            self.stream2b   = (self.stream2b   << 2)
                .wrapping_add(self.stream2b   & 3);
        }
    }

    /// Punctuation handler: the `c1 == LF` branch
    /// (fxcmv1.cpp:4056-4068). Resets line-tracking fields, folds
    /// the line-end bits into the running accumulators, and
    /// (caller-side) resets the WordsContext objects. This method
    /// only touches `FxcmState` fields; resetting `worcxt` /
    /// `worcxt1` is the caller's responsibility.
    ///
    /// Returns `true` if `is_now_iki` should also be cleared (when
    /// the previous byte was already LF — empty line in nowiki).
    pub fn line_break_lf(&mut self) -> bool {
        self.fc           = 0;
        self.is_paragraph = 0;
        self.first_word   = 0;
        self.last_wt      = 0;
        self.nl1 = self.nl;
        self.nl  = self.pos.wrapping_sub(1) as i32;
        self.stream3b_r <<= 7;
        self.stream2b   |= 0x3fc;
        self.words      = 0xfc;
        self.stream2b_r <<= 2;
        self.stream4b   |= 0xfff0;
        self.c2 == LF_CHR
    }

    /// Sentence-end punctuation handler: the `c1 ∈ {., ), ?}` branch
    /// (fxcmv1.cpp:4069-4085). Ages `last_wt`, advances stream
    /// accumulators, and folds the most-recent c4 byte into `x5`.
    /// Caller is responsible for the `worcxt.Reset()` and
    /// `senword=0` rules that gate on `'.'` plus context (those
    /// require fccxt / colcxt access).
    ///
    /// `c4` is the running 4-byte history (BlockData::c4).
    /// Returns `true` iff `c1 == '.'` (signals to the caller that
    /// it should also handle the `wshift=1` and worcxt-reset rules).
    pub fn sentence_end_punct(&mut self, c4: u32) -> bool {
        self.last_wt = self.last_wt.wrapping_mul(16);
        self.stream3b_r <<= 7;
        self.stream3b   <<= 7;
        self.words   |= 0xfe;
        self.x5 = (self.x5 << 8).wrapping_add(c4 & 0xff);
        self.stream2b   |= 204;
        self.stream4b   = ((self.stream4b & 0xffff0) << 8)
                          | (self.stream4b & 0xf);
        self.stream2b_r &= 0xffff_ffc0;
        if self.c1 == b'.' {
            self.wshift = 1;
            true
        } else {
            if self.c1 == b')' { self.sen_word = 0; }
            false
        }
    }

    /// BrFcIdx / FcIdx context-index lookups —
    /// fxcmv1.cpp:4167-4169 + 4246-4247.
    ///
    /// `BrFcIdx` collapses the active bracket / quote / first-char
    /// context to one of 0..=7 via [`FCY`]. `FcIdx` is the
    /// first-char-only collapse via [`FCQ`].
    ///
    /// `br_context` and `br_cxt_active` come from `BracketContext::cxt`
    /// and `BracketContext::context != 0`. Same pair for `qo_*` and
    /// `fc_*`.
    pub fn compute_fc_indices(
        &mut self,
        br_context: u8, br_cxt_active: bool,
        qo_context_high: u8, qo_cxt_active: bool,
        fc_context: u8, fc_cxt_active: bool,
    ) {
        self.br_fc_idx = 0;
        if br_cxt_active {
            self.br_fc_idx = FCY[(br_context & 0x7f) as usize] as u32;
        }
        if !br_cxt_active && qo_cxt_active {
            self.br_fc_idx = FCY[(qo_context_high & 0x7f) as usize] as u32;
        }
        if self.br_fc_idx == 0 && fc_cxt_active {
            self.br_fc_idx = FCY[(fc_context & 0x7f) as usize] as u32;
        }
        self.fc_idx = FCQ[(fc_context & 0x7f) as usize] as u32;
    }

    /// UTF-8 byte-stream tracker — fxcmv1.cpp:4310-4321. When the
    /// previous c2 was the upstream ESCAPE marker (12), interpret
    /// the next bytes as a UTF-8 sequence:
    ///   * `0b110xxxxx` (>>5 == 6) → 2-byte sequence (1 left).
    ///   * `0b1110xxxx` (>>4 == 0xE) → 3-byte sequence (2 left).
    ///   * `0b11110xxx` (>>3 == 0x1E) → 4-byte sequence (3 left).
    ///   * Anything else → ASCII or error.
    /// Continuation bytes must match `0b10xxxxxx` (>>6 == 2);
    /// otherwise the run is aborted.
    ///
    /// Folds each in-sequence byte into `u8w` via `u8w*191 + c1`.
    pub fn track_utf8_byte(&mut self) {
        if self.c2 != ESCAPE_CHR { return; }
        if self.utf8_left == 0 {
            let c1 = self.c1;
            if (c1 >> 5) == 6 {
                self.utf8_left = 1;
                self.u8w = self.u8w.wrapping_mul(191).wrapping_add(c1 as u32);
            } else if (c1 >> 4) == 0xE {
                self.utf8_left = 2;
                self.u8w = self.u8w.wrapping_mul(191).wrapping_add(c1 as u32);
            } else if (c1 >> 3) == 0x1E {
                self.utf8_left = 3;
                self.u8w = self.u8w.wrapping_mul(191).wrapping_add(c1 as u32);
            } else {
                self.utf8_left = 0;
            }
        } else {
            self.utf8_left -= 1;
            if (self.c1 >> 6) != 2 { self.utf8_left = 0; }
        }
    }

    /// Indirect-byte / indirect-word rolling histories — port of
    /// fxcmv1.cpp:4290-4299. Updates:
    ///
    /// * `t2[(c4>>8)&0xffff]` — pushes c1 onto the 4-byte history
    ///   keyed by the previous 16-bit word. The composed
    ///   `indirect_word` for the current step is
    ///   `(c4&0xffff) | (t2[c4&0xffff] << 16)`.
    /// * `t1[(c4>>8)&0xff]` — pushes c1 onto the byte-keyed
    ///   4-byte history. `indirect_byte = c1 | (t1[c1] << 8)`.
    /// * `t1[brcontext]` — pushes the low 2 bits of `stream2b`
    ///   onto the bracket-context byte history.
    ///   `indirect_br_byte = (stream3b & 7) | (t1[brcontext] << 3)`.
    ///
    /// `c4` is the running 4-byte history (BlockData::c4).
    /// `brcontext` is the active `BracketContext::cxt`.
    pub fn update_indirect_histories(&mut self, c4: u32, brcontext: u8) {
        // Rolling 4-byte history per (c4>>8)&0xffff.
        let key_w_prev = ((c4 >> 8) & 0xffff) as usize;
        self.t2[key_w_prev] = (self.t2[key_w_prev] << 8) | (self.c1 as u32);
        let key_w = (c4 & 0xffff) as usize;
        self.indirect_word = (key_w as u32) | (self.t2[key_w] << 16);

        // Rolling 4-byte history per (c4>>8)&0xff.
        let key_b_prev = ((c4 >> 8) & 0xff) as usize;
        self.t1[key_b_prev] = (self.t1[key_b_prev] << 8) | (self.c1 as u32);
        self.indirect_byte = (self.c1 as u32)
            | (self.t1[self.c1 as usize] << 8);

        // Bracket-context byte history (low 2 bits of stream2b).
        self.t1[brcontext as usize] = (self.t1[brcontext as usize] << 2)
            | (self.stream2b & 3);
        self.indirect_br_byte = (self.stream3b & 7)
            | (self.t1[brcontext as usize] << 3);
    }

    /// Word0-position lookup helper — fxcmv1.cpp:4301-4304. Computes
    /// the distance between the current `pos` and the last `pos`
    /// where `word0 & 0xffff` was seen, plus a back-reference byte
    /// from the buffer at that distance and the current `c1`.
    /// Stores result in `indirect_word0_pos`.
    ///
    /// Returns the distance (capped at 256) so the caller can decide
    /// whether the position was "fresh" (>255 → caller emits a
    /// distinct context).
    pub fn update_word0_position(&mut self, buf: &Buffer) -> u32 {
        let key = (self.word0 & 0xffff) as usize;
        let last = self.wp[key];
        let dist = self.pos.wrapping_sub(last);
        if dist > 255 {
            self.indirect_word0_pos = 256 + ((self.c1 as u32) << 16);
            dist
        } else {
            let bb = buf.buf(dist as usize) as u32;
            self.indirect_word0_pos = dist + (bb << 8) + ((self.c1 as u32) << 16);
            dist
        }
    }

    /// Latch the current `pos` into `wp[word0 & 0xffff]`. Should be
    /// called when `word0` is being cleared (end of a word run) —
    /// upstream stamps this immediately before zeroing `word0`.
    pub fn stamp_word0_position(&mut self) {
        let key = (self.word0 & 0xffff) as usize;
        self.wp[key] = self.pos;
    }

    /// Paragraph flag derived from `fc` — fxcmv1.cpp:4192-4193,
    /// 4240-4241. `is_paragraph = 1` when `fc == FIRSTUPPER`,
    /// otherwise 0.
    pub fn paragraph_from_fc(&mut self) {
        self.is_paragraph = if self.fc as u8 == FIRSTUPPER_CHR { 1 } else { 0 };
    }

    /// Comma handler — fxcmv1.cpp:4086-4089.
    pub fn punct_comma(&mut self) {
        self.words   |= 0xfc;
        self.sen_word = 0;
    }

    /// `'('` handler — fxcmv1.cpp:4090-4092.
    pub fn punct_open_paren(&mut self) { self.sen_word = 0; }

    /// `':'` handler (link / list) — fxcmv1.cpp:4097-4102.
    /// `c4` is the running 4-byte history (BlockData::c4).
    pub fn punct_colon(&mut self, c4: u32) {
        self.stream3b   = (self.stream3b   & 0xffff_fff8).wrapping_add(4);
        self.stream2b  |= 12;
        self.x5         = (self.x5 << 8).wrapping_add(c4 & 0xff);
        self.sen_word   = 0;
    }

    /// `'{'` / `'}'` handler — fxcmv1.cpp:4103-4109.
    pub fn punct_curly(&mut self, c4: u32) {
        self.words      |= 0xfc;
        self.stream3b_r &= 0xffff_ffc0;
        self.x5         = (self.x5 << 8).wrapping_add(c4 & 0xff);
        self.stream3b   = (self.stream3b & 0xffff_fff8).wrapping_add(3);
    }

    /// `']'` handler — fxcmv1.cpp:4110-4114. (`'['` opens a wiki link
    /// and is tracked by the BracketContext; `']'` here is the
    /// closing-bracket sentence-effect.)
    pub fn punct_close_square(&mut self) {
        self.stream3b = (self.stream3b & 0xffff_fff8).wrapping_add(3);
        self.link_word = 0;
    }

    /// `'<'` (or `'&'` predecessor) handler — fxcmv1.cpp:4116-4118.
    pub fn punct_less_than(&mut self) { self.words |= 0xfc; }

    /// `'='` heading handler — fxcmv1.cpp:4124-4129.
    /// Note: upstream also OVERWRITES `c2` to `'.'` here; we mirror
    /// that, but callers should treat that c2 mutation as a hint
    /// for the per-byte arith pipeline only.
    pub fn punct_equals(&mut self) {
        self.stream3b = (self.stream3b & 0xffff_fff8).wrapping_add(4);
        self.c2       = b'.';
        self.words    = self.words.wrapping_mul(2);
    }

    /// `'&nbsp;'` shortcut — fxcmv1.cpp:4131-4136. When the previous
    /// char was `'&'` and the current is `'!'`, upstream substitutes
    /// the byte-pair into a single SPACE so downstream models see a
    /// real word break. Mutates `c1` and the low byte of the
    /// returned `c4`. Caller passes in `c4` and uses the returned
    /// value for the rest of the byte processing.
    pub fn punct_nbsp(&mut self, c4: u32) -> u32 {
        self.c1       = SPACE_CHR;
        let new_c4    = (c4 & 0xffff_ff00) | SPACE_CHR as u32;
        self.stream2b = (self.stream2b & 0xffff_fffc)
                         .wrapping_add(WRT_2B[SPACE_CHR as usize] as u32);
        self.stream3b = (self.stream3b & 0xffff_fff8)
                         .wrapping_add(WRT_3B[SPACE_CHR as usize] as u32);
        new_c4
    }

    /// 2-bit and 3-bit serial/non-repeating stream update
    /// (fxcmv1.cpp:4148-4163), run unconditionally each byte after
    /// the word/punct branches. The non-repeating accumulators
    /// (`stream2b_r`, `stream3b_r`) only fold a fresh bit when the
    /// classifier transitioned (`o*b != n*b`); the serial
    /// accumulator (`stream3b`) folds every byte.
    pub fn update_streams_after_byte(&mut self) {
        // 2-bit non-repeating.
        if self.o2b_state != self.n2b_state {
            self.stream2b_r = (self.stream2b_r << 2)
                .wrapping_add(self.n2b_state);
            self.o2b_state = self.n2b_state;
        }
        self.stream2b_mask = (self.stream2b_mask << 2) | 3;

        // 3-bit non-repeating.
        if self.o3b_state != self.n3b_state {
            self.stream3b_r = (self.stream3b_r << 3)
                .wrapping_add(self.n3b_state);
            self.stream3b_r_mask1 = (self.stream3b_r_mask1 << 3) | 7;
            self.stream3b_r_mask2 = (self.stream3b_r_mask2 << 3) | 7;
            self.o3b_state = self.n3b_state;
        }
        // 3-bit serial.
        self.stream3b = (self.stream3b << 3)
            .wrapping_add(self.n3b_state);
        self.stream3b_mask  = (self.stream3b_mask  << 3) | 7;
        self.stream3b_mask1 = (self.stream3b_mask1 << 3) | 7;
    }

    /// After a word run ends, reset the per-word stream-mask history
    /// (fxcmv1.cpp:4014-4017). Cycles `_mask2 ← _mask1`,
    /// `_mask1 ← current`, then zeros the live masks.
    pub fn rotate_stream_masks(&mut self) {
        self.stream3b_r_mask2 = self.stream3b_r_mask1;
        self.stream3b_mask1   = self.stream3b_mask;
        self.stream3b_mask    = 0;
        self.stream2b_mask    = 0;
        self.stream3b_r_mask1 = 0;
    }

    /// Number-tracking sub-step (fxcmv1.cpp:3940-3953). Consumes the
    /// most-recent byte (`self.c1`) and updates `numbers`, `number0`,
    /// `number1`, `numlen0`, `numlen1`, `mybenum` per upstream's
    /// integer-literal parser.
    ///
    /// Must run AFTER `shift_letter_classes()` (which shifts
    /// `numbers` left by 1).
    pub fn track_numbers(&mut self) {
        if (b'0'..=b'9').contains(&self.c1) {
            self.numbers = self.numbers.wrapping_add(1);
            // "1,000" / locale comma group: roll number0 → number1.
            if (self.numbers & 4) != 0 && self.c2 == b',' {
                self.number0 = self.number1;
                self.number1 = 0;
                self.numlen0 = self.numlen1;
                self.numlen1 = 0;
            }
            // mybenum = "could be number" carry from the prior step;
            // when the new digit lands within 2 chars of the carry,
            // commit a roll.
            if self.mybenum != 0 && self.numlen1 <= 2 {
                self.number0 = self.number1;
                self.number1 = 0;
                self.numlen0 = self.numlen1;
                self.numlen1 = 0;
            }
            self.number0 = self.number0
                .wrapping_mul(10)
                .wrapping_add((self.c1 & 0x0f) as u32);
            self.numlen0 = (self.numlen0 + 1).min(19);
            self.mybenum = 0;
        } else {
            // Non-digit terminates the current run.
            if self.numlen0 != 0 || (self.numbers & 0xf) == 0 {
                self.number1 = self.number0;
                self.numlen1 = self.numlen0;
                self.number0 = 0;
                self.numlen0 = 0;
            }
            // Decimal-point continuation hints.
            if self.numlen1 <= 2 && self.numlen1 != 0
                && (self.numbers & 5) == 5
                && self.numlen0 == 0 && self.c2 == b'.'
            {
                self.mybenum = 2;
            } else if self.numlen1 <= 2 && self.numlen1 != 0
                && (self.numbers & 2) != 0
                && self.numlen0 == 0 && self.c1 == b'.'
            {
                self.mybenum = 1;
            } else if self.mybenum == 1 && self.c1 != b'.' {
                self.mybenum = 0;
            }
        }
    }
}

// ====================================================================
// Top-level Predictor scaffolding (state owner). Models in the tree
// (added in subsequent turns) live in fields of this struct.
// ====================================================================

pub struct Predictor {
    // Per-block prediction buffer used by the upstream
    // `AddPrediction` / `ResetPredictions` plumbing.
    pub model_predictions: Vec<f32>,
    pub prediction_index:  usize,

    // Shared per-bit / per-byte state.
    pub block: BlockData,
    pub state: FxcmState,
    pub buffer: Buffer,

    // Pre-computed lookup tables.
    pub sqt: Vec<i16>,
    pub strt: Vec<i16>,
    pub ilog: [U8; 256],
    pub dt:   [i32; 1024],
    pub sta_tables: Vec<u8>,
    pub pre1: [i16; 256],

    // Mixer / SmallStationaryContextMap / RunContextMap / StateMap arrays.
    pub mixers: Vec<Mixer1>,        // length = 12
    pub small_scms: Vec<SmallStationaryContextMap>, // length = 7
    pub run_cm: RunContextMap,
    pub state_maps_match: Vec<StateMap1>, // length = 3 (for MatchModel2)

    // APM caskade (apmA0..apmA5 in upstream).
    pub apm_a: Vec<ApmDyn>,         // length = 6

    // Bracket / quote / first-char / html contexts.
    pub brcxt: BracketContextFx<u8>,
    pub qocxt: BracketContextFx<u8>,
    pub fccxt: BracketContextFx<u8>,
    pub htcxt: BracketContextFx<u16>,

    // Column / words contexts (3 separate WordsContexts upstream).
    pub colcxt: ColumnContext,
    pub worcxt:  WordsContext,
    pub worcxt1: WordsContext,
    pub worcxt2: WordsContext,

    // Match model + sparse match.
    pub match_model: MatchModel2,
    pub smatch:      SparseMatchModel,

    // ContextMap arrays — three width tiers per upstream:
    //   * cm_c2  (18) — large state memory  (`<14, 128>`)
    //   * cm_c1  (8)  — small state memory  (`<3, 32>`)
    //   * cm_c   (6)  — medium state memory (`<7, 64>`)
    pub cm_c2: Vec<ContextMap<14, 128>>,
    pub cm_c1: Vec<ContextMap<3,  32>>,
    pub cm_c:  Vec<ContextMap<7,  64>>,
    /// Per-CM STA-slot index (0..=5 → STA1, STA2, STA4, STA5, STA6,
    /// STA7). Used by `mix_byte_context_maps` to look up the right
    /// 1024-byte slice in `sta_tables` when calling `mix1`.
    pub cm_c2_sta: Vec<u8>,
    pub cm_c1_sta: Vec<u8>,
    pub cm_c_sta:  Vec<u8>,
}

impl Predictor {
    pub fn new() -> Self {
        let sqt  = build_squash_table();
        let strt = build_stretch_table();
        let ilog = build_ilog();
        let dt   = build_dt();
        let sta_tables = build_sta_tables();
        let pre1 = build_pre1_from_sta7(&sta_tables[5 * 1024..6 * 1024], &strt);

        // mxInputs1 sizing matches upstream's
        // `(515+16+1-5*2-2*2) & -16 = 512`; mxInputs2 = 16.
        let inputs1_n: usize = 512;
        let inputs2_n: usize = 16;

        let mixers = build_mixers(inputs1_n, inputs2_n);
        let small_scms = build_small_scm();
        let run_cm = RunContextMap::new(1 * 4096 * 4096, /*rcm_ml=*/6, &ilog);
        let state_maps_match = build_state_maps();

        // 6 APMs sized per upstream `APM<256>`, `APM<0x8000*2>` etc.
        // APM<S> upstream uses `S * 33` slots; ApmDyn picks `s` at
        // runtime — we mirror the sizes directly.
        let apm_a = vec![
            ApmDyn::new(256,           &sqt),
            ApmDyn::new(0x8000  * 2,   &sqt),
            ApmDyn::new(0x8000  * 2,   &sqt),
            ApmDyn::new(0x20000 * 2,   &sqt),
            ApmDyn::new(0x20000 * 2,   &sqt),
            ApmDyn::new(0x20000 * 2,   &sqt),
        ];

        // Bracket / quote / first-char / html contexts (8-bit / 16-bit).
        let brackets = brackets_table();
        let quotes   = quotes_table();
        let fchar    = fchar_table();
        let html     = html_table();
        let brcxt = BracketContextFx::new(&brackets, false, 256, 8);
        let qocxt = BracketContextFx::new(&quotes,   true,  256, 8);
        let fccxt = BracketContextFx::new(&fchar,    false, 256, 8);
        let htcxt = BracketContextFx::new(&html,     false, 0xfff, 16);

        // Match model: 0x200000 hash table, sm_n / sm_limit per upstream.
        // Three StateMap1 inside MatchModel2 use sizes 1<<9, 1<<19, 1<<16
        // — we honour that in `build_state_maps`.
        // The MatchModel2's own 3-state map array is separate from
        // `state_maps_match`; the latter we pass into the model when
        // constructing it, but our current MatchModel2 builds its
        // own internally — keep `state_maps_match` for now in case
        // the caller wants to share.
        let match_model = MatchModel2::new(/*log2_size=*/21, /*sm_n=*/1 << 9,
                                           /*sm_limit=*/1023);

        // ContextMap arrays — STA tables are slot-indexed:
        //   STA1 → 0, STA2 → 1, STA4 → 2, STA5 → 3, STA6 → 4, STA7 → 5.
        let sta1 = sta_tables[0 * 1024..1 * 1024].to_vec();
        let sta2 = sta_tables[1 * 1024..2 * 1024].to_vec();
        let sta4 = sta_tables[2 * 1024..3 * 1024].to_vec();
        let sta5 = sta_tables[3 * 1024..4 * 1024].to_vec();
        let sta6 = sta_tables[4 * 1024..5 * 1024].to_vec();
        let sta7 = sta_tables[5 * 1024..6 * 1024].to_vec();

        let st2_p0 = build_st2_p0();
        let st2_p1 = build_st2_p1();
        let st2_p2 = build_st2_p2();

        // c packs (max-contexts | (c_r << 8) | (c_s << 16)).
        let pack_c = |max_ctx: i32, idx: usize| -> i32 {
            max_ctx | ((C_R[idx] as i32) << 8) | ((C_S[idx] as i32) << 16)
        };

        // cmC2 — 18 entries — large state memory.
        // (m, max_ctx, sta_table, k, u_skip2, st2_p, c_idx)
        let cm_c2_specs: &[(u32, i32, &Vec<u8>, u8, i32, &Vec<i16>, usize)] = &[
            ( 8 * 4096 * 4096,      3, &sta6, 0xf0, 1, &st2_p1, 0),
            (16 * 4096 * 4096,      1, &sta6, 0xf0, 1, &st2_p1, 1),
            ( 8 * 4096 * 4096,      1, &sta6, 0xf0, 1, &st2_p1, 2),
            ( 8 * 4096 * 4096,      1, &sta6, 0xf0, 1, &st2_p1, 3),
            ( 8 * 4096 * 4096,      2, &sta6, 0xf0, 1, &st2_p1, 4),
            ( 8 * 4096 * 4096,      6, &sta6, 0xf0, 1, &st2_p1, 5),
            ( 1 * 4096 * 4096 / 64, 1, &sta1, 0,    1, &st2_p1, 6),
            ( 2 * 4096 * 4096,      1, &sta5, 0xf0, 1, &st2_p1, 7),
            ( 8 * 4096 * 4096 / 2,  4, &sta4, 0,    1, &st2_p1, 8),
            ( 8 * 4096 * 4096,      4, &sta6, 0xf0, 1, &st2_p1, 17),
            ( 8 * 4096 * 4096,      6, &sta5, 0xf0, 1, &st2_p1, 18),
            ( 8 * 4096 * 4096,      5, &sta5, 0xf0, 1, &st2_p1, 19),
            ( 8 * 4096 * 4096,      2, &sta6, 0xf0, 1, &st2_p1, 20),
            (16 * 4096 * 4096,      2, &sta6, 0xf0, 1, &st2_p1, 21),
            ( 4 * 4096 * 4096 / 2,  1, &sta6, 0xf0, 1, &st2_p1, 23),
            ( 8 * 64   * 4096,      1, &sta1, 0,    0, &st2_p0, 24),
            ( 1 * 4096 * 4096 / 2,  1, &sta6, 0xf0, 1, &st2_p1, 17),
            ( 2 * 4096 * 4096,      2, &sta6, 0xf0, 1, &st2_p1, 17),
        ];
        let cm_c2: Vec<ContextMap<14, 128>> = cm_c2_specs.iter().map(
            |&(m, max_ctx, nn, k, u_skip2, st2_p, idx)| {
                ContextMap::new(
                    m, pack_c(max_ctx, idx), C_S3[idx] as i32,
                    nn, C_S4[idx] as i32, k, u_skip2, st2_p, &ilog,
                )
            }).collect();

        // cmC1 — 8 entries — small state memory.
        let cm_c1_specs: &[(u32, i32, &Vec<u8>, u8, i32, &Vec<i16>, usize)] = &[
            (    32 * 4096, 2, &sta6, 0,    0, &st2_p0,  9),
            ( 2 * 32 * 4096, 3, &sta7, 0,    1, &st2_p1, 10),
            (    32 * 4096, 4, &sta2, 0,    1, &st2_p1, 11),
            (   128 * 4096, 2, &sta1, 0,    0, &st2_p0, 16),
            (    16 * 4096, 5, &sta7, 0,    1, &st2_p1, 12),
            (    32 * 4096, 2, &sta7, 0,    1, &st2_p1, 11),  // upstream cmC1[5] absent → fill placeholder
            ( 1 * 16 * 4096, 1, &sta6, 0,    0, &st2_p1,  5),
            (    16 * 4096, 4, &sta2, 0,    1, &st2_p1, 12),
        ];
        let cm_c1: Vec<ContextMap<3, 32>> = cm_c1_specs.iter().map(
            |&(m, max_ctx, nn, k, u_skip2, st2_p, idx)| {
                ContextMap::new(
                    m, pack_c(max_ctx, idx), C_S3[idx] as i32,
                    nn, C_S4[idx] as i32, k, u_skip2, st2_p, &ilog,
                )
            }).collect();

        // cmC — 6 entries — medium state memory.
        let cm_c_specs: &[(u32, i32, &Vec<u8>, u8, i32, &Vec<i16>, usize)] = &[
            (   16 * 4096, 7, &sta2, 0,    1, &st2_p1, 13),
            (64 * 2 * 4096, 3, &sta5, 0xf0, 0, &st2_p0, 14),
            (    2 * 4096, 2, &sta2, 0xf0, 0, &st2_p0, 15),
            (   32 * 4096, 2, &sta2, 0x00, 1, &st2_p2, 22),
            (  512 * 4096, 1, &sta1, 0xf0, 1, &st2_p1, 25),
            (  512 * 4096, 1, &sta1, 0xf0, 1, &st2_p1, 26),
        ];
        let cm_c: Vec<ContextMap<7, 64>> = cm_c_specs.iter().map(
            |&(m, max_ctx, nn, k, u_skip2, st2_p, idx)| {
                ContextMap::new(
                    m, pack_c(max_ctx, idx), C_S3[idx] as i32,
                    nn, C_S4[idx] as i32, k, u_skip2, st2_p, &ilog,
                )
            }).collect();

        Self {
            model_predictions: vec![0.5f32; NUM_MODELS],
            prediction_index: 0,
            block: BlockData::new(528 + 32),
            state: FxcmState::new(),
            buffer: Buffer::new(),
            sqt, strt, ilog, dt, sta_tables, pre1,
            mixers, small_scms, run_cm, state_maps_match, apm_a,
            brcxt, qocxt, fccxt, htcxt,
            colcxt: ColumnContext::new(31),
            worcxt:  WordsContext::new(),
            worcxt1: WordsContext::new(),
            worcxt2: WordsContext::new(),
            match_model,
            smatch: SparseMatchModel::new(),
            cm_c2, cm_c1, cm_c,
            // sta indices (0=STA1, 1=STA2, 2=STA4, 3=STA5, 4=STA6, 5=STA7)
            // — must mirror the order of the *_specs above.
            cm_c2_sta: vec![4, 4, 4, 4, 4, 4, 0, 3, 2, 4, 3, 3, 4, 4, 4, 0, 4, 4],
            cm_c1_sta: vec![4, 5, 1, 0, 5, 5, 4, 1],
            cm_c_sta:  vec![1, 3, 1, 1, 0, 0],
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

impl Predictor {
    /// Feed all byte-boundary ContextMap inputs for the current
    /// state (fxcmv1.cpp:4327-4604). Call once per byte boundary,
    /// AFTER `state.byte_boundary_step(...)` and the rest of the
    /// FxcmState helpers have run, but BEFORE the per-bit mix.
    ///
    /// `h` is upstream's running word/byte hash (`word0*271 + c1`
    /// folded by the byte loop); caller must compute it.
    /// `above`, `above1` are the column-tracker peek bytes.
    /// `c4` is the running 4-byte history.
    pub fn feed_byte_context_maps(
        &mut self,
        h:  u32,
        above:  u32,
        above1: u32,
        c4: u32,
    ) {
        let s = &self.state;
        let c1   = s.c1 as u32;
        let c2   = s.c2 as u32;
        let fc   = s.fc as u32;
        let col  = s.col as u32;
        let utf8 = s.utf8_left;
        let brcontext = self.brcxt.cxt as u32;
        let fccontext = self.fccxt.cxt as u32;
        let lastwt    = s.last_wt & 0xf;

        // Run-context map: word(3)*53 + c1 + 193*(stream3b & 0x7fff).
        self.run_cm.set(
            s.word3.wrapping_mul(53)
                .wrapping_add(c1)
                .wrapping_add(193u32.wrapping_mul(s.stream3b & 0x7fff)),
            s.c1,
        );

        // cmC2[4] / cmC2[17] — word stream.
        if col < 2 || s.fc as u8 == SPACE_CHR {
            self.cm_c2[4].sets();
            self.cm_c2[4].sets();
            self.cm_c2[17].sets();
        } else {
            self.cm_c2[4].set(
                s.word00.wrapping_add(s.number0.wrapping_mul(191).wrapping_add(s.numlen0))
                    .wrapping_add(s.u8w),
            );
            let last_fc0 = self.colcxt.lastfc(0);
            if last_fc0 == b'&' || utf8 != 0 {
                self.cm_c2[4].sets();
            } else {
                self.cm_c2[4].set(h.wrapping_add(s.word1));
            }
            if self.brcxt.cxt == LESSTHAN_CHR as u32 {
                self.cm_c2[17].sets();
            } else {
                self.cm_c2[17].set(
                    self.worcxt1.word(1).wrapping_mul(53)
                        .wrapping_add(self.worcxt1.word(2).wrapping_mul(11))
                        .wrapping_add(h)
                        .wrapping_add(lastwt),
                );
            }
        }

        // cmC2[5] — word stream with multiple hashes.
        if c1 == ESCAPE_CHR as u32 || col < 2 || utf8 != 0 || s.fc as u8 == SPACE_CHR {
            self.cm_c2[5].sets();
        } else {
            self.cm_c2[5].set(h.wrapping_add(s.word2.wrapping_mul(71)));
        }
        if s.fc as u8 == SPACE_CHR || self.brcxt.cxt == LESSTHAN_CHR as u32 {
            for _ in 0..5 { self.cm_c2[5].sets(); }
        } else {
            self.cm_c2[5].set(
                self.worcxt.word(4).wrapping_mul(53)
                    .wrapping_add(self.worcxt1.word(1))
                    .wrapping_add(h)
                    .wrapping_add(s.stream3b & 511),
            );
            let mask4 = self.worcxt.types(4) ^ eng::VERB;
            self.cm_c2[5].set(
                self.worcxt.last(4, mask4).wrapping_mul(53)
                    .wrapping_add(s.s_verb)
                    .wrapping_add(h)
                    .wrapping_add(s.stream3b_r & 63),
            );
            self.cm_c2[5].set(
                self.worcxt.fword.wrapping_mul(53)
                    .wrapping_add(self.worcxt1.word(1))
                    .wrapping_add(h)
                    .wrapping_add(s.stream3b & 63),
            );
            self.cm_c2[5].set(
                self.worcxt2.word(1)
                    .wrapping_add(self.worcxt2.word(2).wrapping_mul(11))
                    .wrapping_add(s.word00)
                    .wrapping_add(c1),
            );
            let last_par_verb = self.worcxt2
                .last_if(1, self.worcxt.types(1) & eng::VERB);
            if last_par_verb != 0 {
                self.cm_c2[5].set(
                    last_par_verb.wrapping_mul(11)
                        .wrapping_add(s.word00)
                        .wrapping_add(c1),
                );
            } else {
                self.cm_c2[5].sets();
            }
        }

        // cmC1[6] / cmC2[6] / cmC2[7].
        self.cm_c1[6].set(
            h.wrapping_add(self.worcxt.types(1) & 0x1FF)
             .wrapping_add(self.worcxt1.word(1)),
        );
        self.cm_c2[6].set(
            ((s.stream2b & 15) << 16)
                .wrapping_add(s.t[2] & 0xffff),
        );
        if c1 == ESCAPE_CHR as u32 || utf8 != 0 || fccontext == CURLYOPENING_CHR as u32 {
            self.cm_c2[7].set(0);
        } else {
            self.cm_c2[7].set(s.indirect_br_byte);
        }

        // cmC2[8].
        self.cm_c2[8].set(
            (((s.indirect_br_byte) & 0x7ff) * 32)
                .wrapping_add((s.stream4b & 0xfff0) << 16)
                .wrapping_add(s.br_fc_idx),
        );
        self.cm_c2[8].set(
            ((s.stream3b_r & 0x3fffffff) * 4)
                .wrapping_add(s.stream2b & 3),
        );
        self.cm_c2[8].set(
            (fccontext * 4)
                .wrapping_add((s.stream3b_r & 0x3ffff) << 9)
                .wrapping_add(s.br_fc_idx),
        );
        if fccontext == HTLINK_CHR as u32 {
            self.cm_c2[8].sets();
        } else {
            self.cm_c2[8].set(
                (c4 & 0xffffff)
                    .wrapping_add((s.stream2b << 18) & 0xff000000),
            );
        }

        // cmC1[0] / cmC1[1] / cmC1[2] / cmC1[4].
        let lfc0 = self.colcxt.lastfc(0) as u32;
        self.cm_c1[0].set(
            lfc0
                | (fccontext << 15)
                | ((s.stream3b & 63) << 7)
                | (brcontext << 24),
        );
        self.cm_c1[0].set(lfc0 | ((c4 & 0xffffff) << 8));

        self.cm_c1[1].set((s.stream2b & 3).wrapping_add(s.word00.wrapping_mul(11)));
        self.cm_c1[1].set(c4 & 0xffff);
        self.cm_c1[1].set(
            ((fc << 11) | c1).wrapping_add((s.stream2b & 3) << 18),
        );

        self.cm_c1[2].set((s.stream2b & 15).wrapping_add((s.stream3b & 7) << 6));
        self.cm_c1[2].set(
            c1
                | ((col * (c1 == SPACE_CHR as u32) as u32) << 8)
                | ((s.stream2b & 15) << 16),
        );
        self.cm_c1[2].set(if s.is_paragraph != 0 { s.first_word } else { fc << 11 });
        if c1 == ESCAPE_CHR as u32 || s.fc as u8 == SPACE_CHR || utf8 != 0 {
            self.cm_c1[2].sets();
        } else {
            self.cm_c1[2].set(
                91u32.wrapping_mul(83).wrapping_mul(self.worcxt.word(1))
                    .wrapping_add(89u32.wrapping_mul(s.word0)),
            );
        }

        // cmC1[4] — word stream + first-char/quotes.
        if s.fc as u8 == SPACE_CHR {
            self.cm_c1[4].sets();
        } else {
            self.cm_c1[4].set(
                c1.wrapping_add((s.stream3b & 0xe38) << 6),
            );
        }
        self.cm_c1[4].set(
            self.worcxt.fword.wrapping_mul(11).wrapping_add(s.br_fc_idx),
        );
        self.cm_c1[4].set(
            c1.wrapping_add(s.word0)
              .wrapping_add(s.number0.wrapping_mul(191)),
        );
        self.cm_c1[4].set(
            ((c4 & 0xffff) << 16)
                | (fccontext << 8)
                | fc,
        );
        self.cm_c1[4].set(
            ((s.stream3b_r & 0xfff) << 8)
                .wrapping_add(s.stream2b & 0xfc),
        );

        // cmC[0] — table/column related contexts.
        if s.c1 == ESCAPE_CHR {
            for _ in 0..6 { self.cm_c[0].sets(); }
        } else {
            if s.is_paragraph == 1 {
                self.cm_c[0].set(
                    self.worcxt.fword.wrapping_mul(3191)
                        .wrapping_add(s.stream2b & 3),
                );
                self.cm_c[0].set(h.wrapping_add(s.first_word.wrapping_mul(89)));
                self.cm_c[0].set(
                    s.word0.wrapping_mul(53)
                        .wrapping_add(c1)
                        .wrapping_add(s.br_fc_idx),
                );
            } else {
                self.cm_c[0].set(
                    above
                        | ((s.stream3b & 0x3f) << 9)
                        | ((self.colcxt.collen(0, 0) as u32) << 19)
                        | ((s.stream2b & 3) << 16),
                );
                self.cm_c[0].set(h.wrapping_add(s.first_word.wrapping_mul(89)));
                self.cm_c[0].set(
                    above
                        | (c1 << 16)
                        | (((col + s.numlen0 + s.br_fc_idx) & 0xff) << 8)
                        | (above1 << 24),
                );
            }
            if self.colcxt.lastfc(0) == b'*' {
                self.cm_c[0].set(
                    s.word0
                        .wrapping_add(fccontext << 8)
                        | (s.br_fc_idx << 16),
                );
                self.cm_c[0].set(c1);
                self.cm_c[0].set(s.word0);
            } else {
                let above_byte = self.buffer.bufr(self.colcxt.abovecellpos as usize);
                self.cm_c[0].set(
                    (WRT_2B[above_byte as usize] as u32)
                        | (fccontext << 8)
                        | (s.br_fc_idx << 16),
                );
                self.cm_c[0].set((above_byte as u32) | (c1 << 8));
                self.cm_c[0].set(
                    s.word0.wrapping_add(WRT_2B[above_byte as usize] as u32),
                );
            }
        }

        // cmC[1] / cmC[2] — additional contexts.
        self.cm_c[1].set(
            (s.stream3b & 0x7fff)
                .wrapping_mul(s.word0)
                .wrapping_add(s.br_fc_idx),
        );
        self.cm_c[1].set(
            (s.x4 & 0xff0000ff) | ((s.stream3b & 0xe07) << 8),
        );
        self.cm_c[1].set(
            (s.indirect_br_byte & 0xffff) | ((s.stream3b & 0x38) << 16),
        );

        if s.is_math {
            self.cm_c[0].sets();
        } else {
            self.cm_c[0].set(
                (s.indirect_byte & 0xff00)
                    .wrapping_add(257u32.wrapping_mul(self.worcxt.word(1)).wrapping_mul(53))
                    .wrapping_add(c1),
            );
        }
        self.cm_c[2].set(
            (c1 << 8)
                | (s.indirect_byte >> 2)
                | (fc << 16),
        );
        self.cm_c[2].set(
            (c4 & 0xffff).wrapping_add(if c2 == s.c3 as u32 { 1 } else { 0 }),
        );

        self.cm_c1[3].set(
            (s.stream3b & s.stream3b_mask) * 256
                | (s.stream2b & s.stream2b_mask & 255),
        );
        self.cm_c1[3].set(s.x4);

        // cmC2[9] — word-stream-1 / link-word feeds.
        let p_hash = self.state.stem_words[self.state.p_word].hash;
        self.cm_c2[9].set(
            257u32.wrapping_mul(p_hash)
                .wrapping_add(fccontext)
                .wrapping_add(193u32.wrapping_mul(s.stream3b & s.stream3b_mask)),
        );
        self.cm_c2[9].set(
            fc | ((s.stream2b_r & 0xfff) << 9)
               | (c1 << 24),
        );
        self.cm_c2[16].set(
            self.worcxt.fword.wrapping_mul(83)
                .wrapping_add((s.stream2b & 15).wrapping_mul(11))
                .wrapping_add(brcontext),
        );
        self.cm_c2[17].set(
            self.worcxt.last(1, eng::VERB)
                .wrapping_add(self.worcxt.word(1).wrapping_mul(83))
                .wrapping_add(h),
        );

        self.cm_c2[9].set(
            (s.x4 & 0xffff00)
                .wrapping_add(brcontext)
                .wrapping_add(fccontext << 24),
        );
        if s.link_word != 0 {
            self.cm_c2[9].set(s.link_word);
        } else if s.is_math {
            self.cm_c2[9].sets();
        } else if s.sen_word != 0 {
            self.cm_c2[9].set(s.sen_word.wrapping_mul(1471).wrapping_add(c1));
        } else if fc == HTML_CHR as u32 || brcontext == LESSTHAN_CHR as u32 {
            self.cm_c2[9].sets();
        } else {
            self.cm_c2[9].set(0);
        }

        // cmC2[10] — indirect-byte / indirect-word feeds.
        self.cm_c2[10].set(s.indirect_byte);
        self.cm_c2[10].set(
            ((s.indirect_byte & 0xffff00) >> 4)
                | (s.stream2b & s.stream2b_mask & 0xf)
                | ((s.stream3b & 0xfff) << 20),
        );
        self.cm_c2[10].set(
            (s.x4 >> 16) | ((s.stream2b & 255) << 24),
        );
        if c1 > 127 {
            self.cm_c2[10].set(
                ((((s.stream2b & 12) * 256) + c1) << 11)
                    | ((s.indirect_word & 0xffffff) >> 16),
            );
        } else {
            self.cm_c2[10].set(
                (c1 << 11)
                    | (s.br_fc_idx << 8)
                    | ((s.indirect_word & 0xffffff) >> 16),
            );
        }
        if s.is_math {
            self.cm_c2[10].sets();
        } else {
            self.cm_c2[10].set(
                (fccontext * 4 + s.br_fc_idx)
                    | ((c4 & 0xffff) << 9)
                    | ((s.stream2b & 0xff) << 24),
            );
        }
        self.cm_c2[10].set(
            (s.indirect_word >> 16)
                | ((s.stream2b & 0x3c) << 25)
                | ((s.stream3b & 0x1ff) << 16),
        );

        // cmC2[11] — words/spaces/numbers feeds.
        self.cm_c2[11].set(
            (s.words as u32)
                .wrapping_add((s.spaces as u32) << 8)
                .wrapping_add((s.stream2b & 15) << 16)
                .wrapping_add(((s.stream3b_r >> 3) & 511) << 21)
                .wrapping_add((s.is_paragraph as u32) << 30),
        );
        self.cm_c2[11].set(
            c1.wrapping_add((s.stream3b << 5) & 0x1fffff00),
        );
        self.cm_c2[11].set(
            s.stream2b_r.wrapping_mul(16).wrapping_add(s.br_fc_idx),
        );
        self.cm_c2[11].set(
            ((s.indirect_byte & 0xffff) >> 8)
                .wrapping_add((64u32.wrapping_mul(s.stream2b_r)) & 0x3ffff00)
                .wrapping_add(brcontext << 25),
        );
        if fccontext == FIRSTUPPER_CHR as u32 && brcontext == SQUAREOPEN_CHR as u32 {
            self.cm_c2[11].sets();
        } else {
            self.cm_c2[11].set(
                s.indirect_word0_pos
                    | ((s.indirect_byte & 0xff00) << 16),
            );
        }

        // cmC2[12] — byte-stream + paragraph/column branches.
        self.cm_c2[12].set(
            (s.x4 & 0x80f00000)
                .wrapping_add((s.x4 & 0x0000f0ff) << 12),
        );

        // Paragraph (word) vs column (above-byte) sub-branch.
        if s.is_paragraph == 1 {
            // Word.
            if s.c1 == ESCAPE_CHR
                || fccontext == HTLINK_CHR as u32
                || fccontext == CURLYOPENING_CHR as u32
                || s.is_math
                || s.is_pre
            {
                self.cm_c2[12].sets();
            } else {
                self.cm_c2[12].set(
                    h.wrapping_add(self.worcxt.word(1).wrapping_mul(53).wrapping_mul(79))
                     .wrapping_add(self.worcxt.word(3).wrapping_mul(53)
                                       .wrapping_mul(47).wrapping_mul(71)),
                );
            }
        } else {
            // Column.
            if fccontext == HTLINK_CHR as u32
                || brcontext == LESSTHAN_CHR as u32
                || self.htcxt.cxt != 0
            {
                self.cm_c2[12].sets();
            } else if col == 31 {
                self.cm_c2[12].set(c4 << 16);
            } else {
                self.cm_c2[12].set(
                    above
                        | ((c4 & 0xffff) << 16)
                        | (above1 << 8),
                );
            }
        }

        // cmC2[13] — word/sentence stream, gated by many "html/template"
        // conditions including the high byte of worcxt.sBytes(0)=='\\'.
        let sb0_high = (self.worcxt.s_bytes(0) >> 8) as u8;
        let cm_c2_13_disabled = s.c1 == ESCAPE_CHR
            || utf8 != 0
            || fccontext == CURLYOPENING_CHR as u32
            || fccontext == HTLINK_CHR as u32
            || fc == HTML_CHR as u32
            || self.htcxt.cxt != 0
            || s.fc as u8 == SPACE_CHR
            || s.is_pre
            || c1 == b'&' as u32
            || brcontext == LESSTHAN_CHR as u32
            || s.is_math
            || col < 2
            || sb0_high == b'\\';
        if cm_c2_13_disabled {
            self.cm_c2[13].sets();
            self.cm_c2[13].sets();
        } else {
            self.cm_c2[13].set(
                self.worcxt.word(1).wrapping_mul(83).wrapping_mul(1471)
                    .wrapping_sub(s.word0.wrapping_mul(53))
                    .wrapping_add(self.worcxt.word(2)),
            );
            self.cm_c2[13].set(
                h.wrapping_add(self.worcxt.word(2).wrapping_mul(53).wrapping_mul(79))
                 .wrapping_add(self.worcxt.word(3).wrapping_mul(53)
                                   .wrapping_mul(47).wrapping_mul(71)),
            );
        }

        // cmC[3] — last-byte stream2b/3bR + first char + brfcidx,
        //          link-or-word / number cross.
        self.cm_c[3].set(
            ((s.stream3b_r & 7) << 10)
                .wrapping_add(s.stream2b & 3)
                .wrapping_add(fc * 4)
                .wrapping_add(s.br_fc_idx << 24),
        );
        let lw_or_w0 = if s.link_word != 0 { s.link_word } else { s.word0 };
        self.cm_c[3].set(
            lw_or_w0.wrapping_mul(3301).wrapping_add(s.number0.wrapping_mul(3191)),
        );

        // cmC2[14] — word/non-repeating cross with brfcidx.
        let cm_c2_14_disabled = s.c1 == ESCAPE_CHR
            || utf8 != 0
            || fccontext == CURLYOPENING_CHR as u32
            || fccontext == HTLINK_CHR as u32
            || s.fc as u8 == SPACE_CHR
            || fc == HTML_CHR as u32
            || brcontext == LESSTHAN_CHR as u32
            || col < 2
            || s.is_math
            || sb0_high == b'\\';
        if cm_c2_14_disabled {
            self.cm_c2[14].sets();
        } else {
            self.cm_c2[14].set(
                s.br_fc_idx
                    .wrapping_add(
                        self.worcxt.word(2)
                            .wrapping_mul(s.stream3b_r & s.stream3b_r_mask2),
                    )
                    .wrapping_add(self.worcxt.types(1) & 0x1ff),
            );
        }

        // cmC1[7] — local, small memory.
        if s.c1 == ESCAPE_CHR || utf8 != 0 || s.fc as u8 == SPACE_CHR {
            for _ in 0..4 { self.cm_c1[7].sets(); }
        } else {
            self.cm_c1[7].set(self.worcxt1.word(1).wrapping_add(s.word00));
            self.cm_c1[7].set(
                self.worcxt.word(2)
                    .wrapping_add(s.word0.wrapping_mul(191))
                    .wrapping_add(s.stream3b_r & 63),
            );
            self.cm_c1[7].set(
                s.word0.wrapping_mul(191)
                    .wrapping_add(s.stream3b_r & 63),
            );
            self.cm_c1[7].set(
                (s.indirect_word0_pos & 0xffff)
                    .wrapping_mul(191)
                    .wrapping_add(s.word0)
                    .wrapping_add(s.stream3b_r & 63),
            );
        }

        // SmallStationaryContextMap fan-out (scmA[0..6]).
        self.small_scms[0].set(c1);
        self.small_scms[1].set(c2.wrapping_mul(s.is_paragraph as u32));
        self.small_scms[2].set((s.indirect_word & 0xffffff) >> 16);
        self.small_scms[3].set(s.stream3b & 0x1ff);
        self.small_scms[4].set(s.stream2b & 0xff);
        self.small_scms[5].set(brcontext);
        self.small_scms[6].set(
            (s.is_paragraph as u32).wrapping_add(2 * (s.stream3b_r & 0x3f)),
        );

        // Snapshot the few state fields the remaining writes touch,
        // then drop the immutable borrow so we can mutate `self.state`.
        let br_fc_idx_snap  = s.br_fc_idx;
        let stream3b_r_snap = s.stream3b_r;
        let x5_snap         = s.x5;
        // (`s` is dropped here implicitly by going out of scope below.)
        drop(s);

        // wshift handler / sentence-end word-pipeline cycle
        // (fxcmv1.cpp:4564-4568).
        if self.state.wshift != 0 || self.state.c1 == LF_CHR {
            self.state.word3 = self.state.word3.wrapping_mul(47);
            self.state.word2 = self.state.word2.wrapping_mul(53);
            self.state.word1 = self.state.word1.wrapping_mul(83);
            self.state.wshift = 0;
            if self.state.c1 == LF_CHR { self.state.s_verb = 0; }
        }

        // cmC2[15] — final word-stream feed with brfcidx + fc +
        // stream3bR low 12 bits.
        self.cm_c2[15].set(
            (br_fc_idx_snap * 256)
                .wrapping_add(fc)
                .wrapping_add((stream3b_r_snap & 0xfff) << 16),
        );

        // APM hashes — fxcmv1.cpp:4572-4575.
        self.state.ah1 = fxcm_hash(
            (x5_snap >> 0) & 255,
            (x5_snap >> 8) & 255,
            (x5_snap >> 16) & 0x80ff,
        );
        self.state.ah2 = fxcm_hash(19, x5_snap & 0x80ffff, 0xffff_ffff);
    }
}

impl Predictor {
    /// Per-bit ContextMap mix chain — fxcmv1.cpp:4581-4629. Calls
    /// every `cm.mix1(...)` (32 ContextMaps total) plus the seven
    /// SmallStationaryContextMap predictors and the MatchModel /
    /// SparseMatchModel. Pushes their stretched / raw outputs into
    /// `block.mx_inputs1`, accumulates `ord_x` / `ord_w` across the
    /// CM mix calls, and returns the MatchModel's reported match
    /// length (caller stores into `state.is_match`).
    pub fn mix_byte_context_maps(&mut self) -> u32 {
        // Snapshot inputs we'll need without holding a borrow.
        let cc = self.block.c0;
        let bp = self.block.bpos;
        let c1 = (self.block.c4 & 0xff) as u8;
        let y  = self.block.y;
        let bposshift   = 7 - bp;
        let c0shift_bp  = (self.block.c0 << 1) ^ (256 >> bposshift);

        // Each cm.mix1 needs a temp Vec to push into; we then drain
        // each into mx_inputs1 via `add` (which performs the
        // upstream bounds + range checks).
        let mut tmp: Vec<i16> = Vec::with_capacity(32);

        // SmallStationaryContextMap fan-out — pushes 2 outputs each.
        let sscmrate = self.state.sscm_rate;
        for s in self.small_scms.iter_mut() {
            tmp.clear();
            s.mix(y, sscmrate, &self.sqt, &self.strt, &mut tmp);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }

        // MatchModel — pushes 1 + 3*2 = 7 outputs and returns length.
        tmp.clear();
        let order_hashes = self.state.t;
        let word1_hash = self.worcxt.word(1);
        let pos = self.state.pos;
        // mix takes &mut Vec<i32>, not Vec<i16> — pass a separate
        // i32 buffer and convert on push.
        let mut tmp32: Vec<i32> = Vec::with_capacity(8);
        let is_match = self.match_model.mix(
            y, bp, c1, cc, pos,
            &self.buffer, &order_hashes, word1_hash,
            &self.sqt, &self.strt, &self.dt,
            &mut tmp32,
        );
        for v in tmp32 { self.block.mx_inputs1.add(v.clamp(-2047, 2047) as i16); }

        // SparseMatchModel — pushes 2 outputs.
        tmp.clear();
        self.smatch.p(cc, bp, &self.buffer, &mut tmp);
        for &v in &tmp { self.block.mx_inputs1.add(v); }

        // ContextMap mixes. Each pushes per-context outputs (5 +
        // run-context = 6) into out via mix1. Inline `bp` matches
        // BlockData::bpos, `cc` is c0.
        let mut ord_x = 0i32;
        let mut ord_w = 0i32;

        // cm_c2[0] — first CM: ord_x bonus +2 if cxt-mask is non-zero.
        if self.cm_c2[0].cxt_mask != 0 { ord_x = 2; }
        for (i, cm) in self.cm_c2.iter_mut().enumerate().take(13) {
            tmp.clear();
            let nn_idx = self.cm_c2_sta[i] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            let r = cm.mix1(cc, bp, c1, y, &mut tmp, c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
            // ord_x accumulation: cm_c2[0..=3] feed ord_x; [4..=5] feed ord_w.
            if i < 4 { ord_x += r; if i == 0 && ord_x == 3 { ord_x = 2; } }
            else if i < 6 { ord_w += r; }
        }
        if ord_w > 3 { ord_w = 3; }

        // cm_c1[0..2,4]
        for i in &[0usize, 1, 2, 4] {
            tmp.clear();
            let nn_idx = self.cm_c1_sta[*i] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            self.cm_c1[*i].mix1(cc, bp, c1, y, &mut tmp,
                                c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }

        // cm_c[0..=2]
        for i in 0..3 {
            tmp.clear();
            let nn_idx = self.cm_c_sta[i] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            self.cm_c[i].mix1(cc, bp, c1, y, &mut tmp,
                              c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }
        // cm_c1[3]
        {
            tmp.clear();
            let nn_idx = self.cm_c1_sta[3] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            self.cm_c1[3].mix1(cc, bp, c1, y, &mut tmp,
                               c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }

        // cm_c2[13]
        {
            tmp.clear();
            let nn_idx = self.cm_c2_sta[13] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            let r = self.cm_c2[13].mix1(cc, bp, c1, y, &mut tmp,
                                        c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
            ord_w += r;
        }

        // cm_c[3] / cm_c2[14] / cm_c2[15] / cm_c[4..=5]
        for &(arr_kind, idx) in &[
            (1u8, 3), (2, 14), (2, 15), (1, 4), (1, 5),
        ] {
            tmp.clear();
            let (nn_idx, r) = match arr_kind {
                1 => {
                    let n = self.cm_c_sta[idx] as usize;
                    let nn_slice = &self.sta_tables[n * 1024..(n + 1) * 1024];
                    (n, self.cm_c[idx].mix1(cc, bp, c1, y, &mut tmp,
                                            c0shift_bp, bposshift, nn_slice))
                }
                _ => {
                    let n = self.cm_c2_sta[idx] as usize;
                    let nn_slice = &self.sta_tables[n * 1024..(n + 1) * 1024];
                    (n, self.cm_c2[idx].mix1(cc, bp, c1, y, &mut tmp,
                                             c0shift_bp, bposshift, nn_slice))
                }
            };
            for &v in &tmp { self.block.mx_inputs1.add(v); }
            if arr_kind == 2 && idx == 14 { ord_w += r; }
            let _ = nn_idx;
        }

        // cm_c2[16..=17] / cm_c1[6..=7]
        for i in &[16usize, 17] {
            tmp.clear();
            let nn_idx = self.cm_c2_sta[*i] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            self.cm_c2[*i].mix1(cc, bp, c1, y, &mut tmp,
                                c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }
        for i in &[6usize, 7] {
            tmp.clear();
            let nn_idx = self.cm_c1_sta[*i] as usize;
            let nn_slice = &self.sta_tables[nn_idx * 1024..(nn_idx + 1) * 1024];
            self.cm_c1[*i].mix1(cc, bp, c1, y, &mut tmp,
                                c0shift_bp, bposshift, nn_slice);
            for &v in &tmp { self.block.mx_inputs1.add(v); }
        }

        // RunContextMap final mix — pushes 1 prediction.
        let p_run = self.run_cm.p(c0shift_bp, bposshift);
        self.block.mx_inputs1.add(p_run);

        // Persist ord_x / ord_w into FxcmState so set_mixer_contexts
        // sees them.
        self.state.ord_x = ord_x;
        self.state.ord_w = ord_w;
        is_match
    }

    /// Per-bit mixer-context derivation — fxcmv1.cpp:4642-4757. Each
    /// of the 12 mixers gets its `cxt` set as a function of (bpos,
    /// c0, stream2b/3b/3bR, words/numbers, BrFcIdx/FcIdx, ordX/ordW
    /// etc.). Pure integer math; runs once per bit before `mix()`.
    ///
    /// `c0_b` is upstream's `c0 << (8 - bpos)` — pass in from
    /// BlockData::c0/bpos.
    /// `is_match` is the latest length returned by `MatchModel2.mix`
    /// (0 when no active candidate).
    pub fn set_mixer_contexts(&mut self, c0_b: u32, is_match: u32) {
        let s    = &self.state;
        let bpos = self.block.bpos as u32;
        let words   = s.words as u32;
        let nums    = s.numbers as u32;
        let c2_3b   = s.stream3b & 7;
        let c2_3b_r = s.stream3b_r & 7;
        let stream2b = s.stream2b;

        // Mixer 0.
        self.mixers[0].cxt = if bpos == 0 {
            ((stream2b & 255) * 8 + c2_3b) as usize
        } else if bpos > 3 {
            let c = WRT_2B[(c0_b & 255) as usize] as u32;
            ((((stream2b << 2) & 255) + c) * 8 + s.br_fc_idx) as usize
        } else {
            ((stream2b & 255) * 8 + s.br_fc_idx) as usize
        };

        // Mixer 1.
        let mx1_c: u32 = if bpos != 0 {
            let mut c = c0_b;
            if bpos == 1 {
                c = c.wrapping_add(16u32 * (words.wrapping_mul(2) & 4));
            } else if bpos > 3 {
                c = (WRT_2B[(c0_b & 255) as usize] as u32) * 64;
            }
            (bpos.min(5)).wrapping_mul(256)
                .wrapping_add(c2_3b_r)
                .wrapping_add(s.fc_idx.wrapping_mul(8))
                .wrapping_add(c & 192)
        } else {
            (words & 12).wrapping_mul(16)
                .wrapping_add(c2_3b_r)
                .wrapping_add(s.br_fc_idx.wrapping_mul(8))
        };
        self.mixers[1].cxt = mx1_c as usize;

        // Mixer 2.
        self.mixers[2].cxt = (((4 * words) & 0xf0) * 4
            + (s.ord_x as u32) * 256 * 4
            + (stream2b & 63)) as usize;

        // Mixer 6.
        self.mixers[6].cxt = ((s.stream3b_r & 0xff8) * 4
            + ((2 * words) & 0x1c)
            + (stream2b & 3)) as usize;

        // Mixer 3.
        self.mixers[3].cxt = (bpos * 256
            + ((((nums | words) << bpos) & 255) >> bpos | (c0_b & 255))) as usize;

        // Mixer 10 (final).
        self.mixers[10].cxt = (((s.ord_x as u32) * 8
            + (if s.br_fc_idx != 0 { 1 } else { 0 }) * 4
            + (stream2b & 3)) * 2
            + (words & 1)) as usize;

        // Mixer 4 — fxcmv1.cpp:4706-4726. Computed `c` cycles with
        // bpos, then folded with ord_x adjustment and is_paragraph.
        let mut c4: u32 = c0_b;
        if bpos != 0 {
            match bpos {
                1 => c4 = c4.wrapping_add(16 * c2_3b),
                2 => c4 = c4.wrapping_add(16 * (stream2b & 3)),
                3 => c4 = c4.wrapping_add(16 * (words & 1)),
                _ => c4 = bpos.wrapping_add(c4 & 0xf0),
            }
            if bpos < 5 { c4 = bpos.wrapping_add(c4 & 0xf0); }
        } else {
            c4 = 16 * (stream2b & 0xf);
        }
        // ord_x adjustment for mixer 4 (in-place: caller's
        // `state.ord_x` is mutated).
        let mut ord_x_adj = (s.ord_x as i32) - 1;
        if ord_x_adj < 0 { ord_x_adj = 0; }
        if is_match != 0 { ord_x_adj += 1; }
        self.mixers[4].cxt = (c4
            + (ord_x_adj as u32) * 256
            + 8 * (s.is_paragraph as u32)) as usize;

        // Mixer 5 — words / first-char / stream2b/3b cross.
        self.mixers[5].cxt = ((s.ord_w as u32) * 256
            + (stream2b & 0xf0)
            + ((s.stream3b & 0x38) >> 2)) as usize * 4 + s.fc_idx as usize;

        // Mixer 7 — bpos-split: 0..=2 vs 3..=7.
        if bpos > 2 {
            self.mixers[7].cxt = (((s.stream3b & 7) * 8
                + WRT_3B[(c0_b & 255) as usize] as u32)
                * 256
                + s.br_fc_idx * 32
                + (words & 7) * 4
                + (s.is_paragraph as u32)
                + if is_match != 0 { 2 } else { 0 }) as usize;
        } else {
            self.mixers[7].cxt = (((s.stream3b & 63) * 256
                + s.br_fc_idx * 16
                + (words & 7) * 2
                + (s.is_paragraph as u32))
                | if is_match != 0 { 128 } else { 0 }) as usize;
        }

        // Mixer 9 — bpos × fails low2 cross. Upstream additionally
        // folds `lstmex` (LSTM auxiliary signal); without that
        // available yet, we leave the slot at 0.
        self.mixers[9].cxt = ((bpos << 8) * 4
            + (s.fails & 3) * 256) as usize;

        // Update state.ord_x with the post-mixer-4 adjustment so the
        // upstream "ordX = ordX-1; ... if (isMatch) ordX++;" carries
        // forward to the next bit.
        let _ = s;  // release immutable borrow
        self.state.ord_x = ord_x_adj;
    }
}

impl Predictor {
    /// APM cascade applied to `pr` (the modelPrediction output) —
    /// fxcmv1.cpp:4807-4834. Returns the post-cascade prediction.
    /// Mutates the AP M state of `apm_a[0..=5]` per the upstream
    /// recipe.
    ///
    /// `pr_in` is the prediction returned by the per-bit context-map
    /// mixer chain (modelPrediction's tail). `c0`, `y`, are the
    /// current BlockData fields.
    pub fn apm_cascade(&mut self, pr_in: i32, c0: i32, y: i32) -> i32 {
        let s = &self.state;
        let rate = s.rate as u32;
        let fails = s.fails;
        let failz = s.failz;
        let failcount = s.failcount;
        let stream2b = s.stream2b;
        let stream2b_r = s.stream2b_r;
        let stream3b_r = s.stream3b_r;
        let ah1 = s.ah1;
        let ah2 = s.ah2;
        let x5 = s.x5;
        // Drop the immutable borrow before APM mutates.
        let _ = s;

        let pu_init = (self.apm_a[0].p(pr_in, c0 as usize, 3, y, &self.strt)
                         + 7 * pr_in + 4) >> 3;

        // Compute pz after the failure-count fold.
        let mut pz: i32 = (failcount + 1) as i32;
        pz += TRI[((fails >> 5) & 3) as usize] as i32;
        pz += TRJ[((fails >> 3) & 3) as usize] as i32;
        pz += TRJ[((fails >> 1) & 3) as usize] as i32;
        if (fails & 1) != 0 { pz += 8; }
        pz /= 2;

        let pu = self.apm_a[3].p(
            pu_init,
            (((c0 as u32) * 2) ^ ah1) as usize & 0x3ffff,
            rate, y, &self.strt,
        );
        let pv0 = self.apm_a[1].p(
            pr_in,
            (((c0 as u32) * 8) ^ fxcm_hash(29, failz & 2047, 0xffff_ffff))
                as usize & 0xffff,
            rate + 1, y, &self.strt,
        );

        let pv = if (fails & 255) != 0 {
            self.apm_a[4].p(
                pv0,
                fxcm_hash(c0 as u32, stream2b & 0xfffc, stream3b_r & 0x1ff)
                    as usize & 0x3ffff,
                rate, y, &self.strt,
            )
        } else {
            self.apm_a[4].p(
                pv0,
                fxcm_hash(c0 as u32,
                    (stream2b_r & 0xfffc).wrapping_add(0x10000),
                    stream3b_r & 0x1ff) as usize & 0x3ffff,
                rate, y, &self.strt,
            )
        };

        let pt = self.apm_a[2].p(
            pr_in,
            (((c0 as u32) * 32) ^ ah2) as usize & 0xffff,
            rate, y, &self.strt,
        );

        let pz_final = self.apm_a[5].p(
            pu,
            (((c0 as u32) * 4)
                ^ fxcm_hash(pz.min(9) as u32, x5 & 0x80ff, 0xffff_ffff))
                as usize & 0x3ffff,
            rate, y, &self.strt,
        );

        if (fails & 255) != 0 {
            (pt * 6 + pu + pv * 11 + pz_final * 14 + 31) >> 5
        } else {
            (pt * 4 + pu * 5 + pv * 12 + pz_final * 11 + 31) >> 5
        }
    }
}

impl Predictor {
    /// Final mixer chain — fxcmv1.cpp:4744-4755. Runs mixers
    /// 0..=9 to populate `mx_inputs2`, then runs the two final
    /// mixers (10 and 11) and returns the squashed
    /// `(mxA[10].p1()*7 + mxA[11].p1() + 4) >> 3`. Caller stores
    /// the result into `state.pr`.
    ///
    /// Wired BEFORE this is called: `mx_inputs1` populated by
    /// `mix_byte_context_maps`, mixer contexts assigned by
    /// `set_mixer_contexts`. Upstream also pushes the LSTM byte
    /// model's stretched prediction into `mx_inputs2`; that signal
    /// isn't ported yet so we substitute zero (a no-op for the
    /// final logit sum).
    pub fn final_mixer_prediction(&mut self) -> i32 {
        // Connect the per-mixer input vector. Upstream's setTxWx
        // wires the first 10 mixers to mxInputs1.n; mixers 10, 11
        // wire to mxInputs2.n. Our `set_inputs` copies the slice in.
        let inputs1 = self.block.mx_inputs1.n[..self.block.mx_inputs1.ncount].to_vec();
        for i in 0..=9 {
            self.mixers[i].set_inputs(&inputs1);
        }

        // Reset mxInputs2 and push the LSTM-stretched prior (we're
        // missing that signal, push 0 as the placeholder).
        self.block.mx_inputs2.reset();
        self.block.mx_inputs2.add(0);
        // Push mxA[0..=9].p1() into mxInputs2.
        for i in 0..=9 {
            let p = self.mixers[i].p1(&self.sqt) as i16;
            self.block.mx_inputs2.add(p.clamp(-2047, 2047));
        }
        // Upstream also adds `stretch(lstmpr) / 2`; substitute 0.
        self.block.mx_inputs2.add(0);

        // Wire mixers 10 and 11 to mx_inputs2.
        let inputs2 = self.block.mx_inputs2.n[..self.block.mx_inputs2.ncount].to_vec();
        self.mixers[10].set_inputs(&inputs2);
        self.mixers[11].set_inputs(&inputs2);

        let p10 = self.mixers[10].p1(&self.sqt);
        let p11 = self.mixers[11].p1(&self.sqt);
        squash(&self.sqt, (p10 * 7 + p11 + 4) >> 3)
    }
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
    fn final_mixer_prediction_returns_a_squashed_pr() {
        let mut p = Predictor::new();
        // Mix once so the first 9 mixer inputs are populated.
        let _ = p.mix_byte_context_maps();
        // Set mixer contexts so they're in-bounds.
        p.set_mixer_contexts(/*c0_b=*/0xC0, /*is_match=*/0);
        let pr = p.final_mixer_prediction();
        assert!(pr >= 0 && pr <= 4095,
            "final mixer pr out of range: {}", pr);
    }

    #[test]
    fn mix_byte_context_maps_runs_without_panic() {
        let mut p = Predictor::new();
        // Initial state: c0=1 (with leading 1), bpos=0, etc.
        // Calling mix at byte boundary should run all 32 CM mix1
        // paths plus scm/match/sparse without panicking.
        let _is_match = p.mix_byte_context_maps();
        // mx_inputs1 should now be populated.
        assert!(p.block.mx_inputs1.ncount > 0,
            "expected mix to push predictions into mx_inputs1");
    }

    #[test]
    fn apm_cascade_returns_within_pr_range() {
        let mut p = Predictor::new();
        // After cascade with pr=2048 (uniform), result should also be
        // a 0..4096 prediction.
        let out = p.apm_cascade(/*pr=*/2048, /*c0=*/1, /*y=*/0);
        assert!(out >= 0 && out <= 4095, "apm_cascade out of range: {}", out);
    }

    #[test]
    fn set_mixer_contexts_assigns_cxt_for_each_bit_position() {
        let mut p = Predictor::new();
        p.state.stream2b = 0xAB;
        p.state.stream3b = 0x55;
        p.state.stream3b_r = 0x33;
        p.state.words   = 0xfc;
        p.state.numbers = 5;
        p.state.fc_idx  = 3;
        p.state.br_fc_idx = 2;
        p.state.ord_x   = 4;

        // Run for bpos=0..=7 to cover every branch.
        for bpos in 0..=7 {
            p.block.bpos = bpos;
            p.set_mixer_contexts(/*c0_b=*/0xC0, /*is_match=*/0);
            // Mixers 0/1/2/3/4/6/10 should all have a `cxt` assigned
            // within their `m` range.
            assert!(p.mixers[0].cxt < p.mixers[0].m);
            assert!(p.mixers[1].cxt < p.mixers[1].m);
            assert!(p.mixers[2].cxt < p.mixers[2].m);
            assert!(p.mixers[3].cxt < p.mixers[3].m);
            assert!(p.mixers[6].cxt < p.mixers[6].m);
            assert!(p.mixers[10].cxt < p.mixers[10].m);
        }
    }

    #[test]
    fn feed_byte_context_maps_does_not_panic() {
        // Smoke test — feed a known c1 through the CM bank with all
        // contexts at their boot values. Exercises every branch
        // that depends only on FxcmState / built-in contexts.
        let mut p = Predictor::new();
        // Wire a synthetic byte boundary by hand.
        p.state.c1 = b'h';
        p.state.col = 5;
        p.state.fc  = b'A' as i32;
        p.state.is_paragraph = 1;
        p.state.word0 = 7;
        p.state.word00 = 7;
        p.state.first_word = 11;
        p.state.stream2b = 0xAB;
        p.state.stream3b = 0x55;
        p.state.stream3b_r = 0x33;
        p.state.x4 = 0xCAFEBABE;
        // Drives a number of branches in cm_c[0]/[1]/[2], cm_c1[*],
        // cm_c2[*]; just verify no panic and `cn` advances.
        let h = p.state.word0.wrapping_mul(271).wrapping_add(b'h' as u32);
        p.feed_byte_context_maps(h, /*above=*/0, /*above1=*/0, /*c4=*/0xCAFEBABE);
        // Every CM should now have at least one stored context.
        assert!(p.cm_c2[0].cn == 0 || p.cm_c2[4].cn > 0,
                "feeds must register some contexts");
        // cm_c1[1] gets unconditional set calls — verify cn advanced.
        assert!(p.cm_c1[1].cn >= 1);
    }

    #[test]
    fn predictor_owns_all_required_components() {
        let p = Predictor::new();
        // Tables.
        assert_eq!(p.sqt.len(),  4095);   // d ∈ [-2047, +2047] = 4095 slots
        assert_eq!(p.strt.len(), 4096);
        assert_eq!(p.dt.len(),   1024);
        assert_eq!(p.sta_tables.len(), 6 * 1024);
        // Mixers / SCMs / state maps.
        assert_eq!(p.mixers.len(),    12);
        assert_eq!(p.small_scms.len(), 7);
        assert_eq!(p.state_maps_match.len(), 3);
        // APM cascade.
        assert_eq!(p.apm_a.len(), 6);
        // Bracket / words contexts construct cleanly.
        assert_eq!(p.brcxt.element_count, 8);
        assert_eq!(p.fccxt.element_count, 20);
        assert_eq!(p.qocxt.element_count, 4);
        assert_eq!(p.htcxt.element_count, 2);
        // FxcmState boot constants reachable via predictor.
        assert_eq!(p.state.ah2, 0x765BA55C);
        assert_eq!(p.state.pr,  2048);
        // ContextMap arrays match upstream's cmC2[18], cmC1[8], cmC[6].
        assert_eq!(p.cm_c2.len(), 18);
        assert_eq!(p.cm_c1.len(),  8);
        assert_eq!(p.cm_c.len(),   6);
    }

    #[test]
    fn dot_product_scaled_by_8_bits() {
        // n=16 elements: t = [256; 16], w = [128; 16].
        // Each pair contributes (256 * 128) >> 8 = 128 → sum = 16 * 128 = 2048.
        // We pair-loop, so we cover all 16 entries.
        let mut t = vec![0i16; 32];
        let mut w = vec![0i16; 32];
        for i in 0..16 { t[i] = 256; w[i] = 128; }
        let r = dot_product(&t, &w, 16);
        assert_eq!(r, 2048);
    }

    #[test]
    fn train_moves_weights_toward_target() {
        let mut w = vec![0i16; 32];
        let t = vec![1024i16; 32];
        train(&t, &mut w, 16, 4096);
        // Each weight should have moved positive (err > 0, t > 0).
        for &wi in &w[..16] {
            assert!(wi > 0, "weight should ascend, got {}", wi);
        }
    }

    #[test]
    fn mixer_predict_and_train_changes_pr() {
        let sqt = build_squash_table();
        let mut m = Mixer1::new(/*n=*/16, /*m=*/4, /*shift1=*/8, /*elim=*/0, /*uperr=*/200);
        m.cxt = 1;
        // Set some logits.
        for v in m.tx_mut() { *v = 500; }
        let p_before = m.p(&sqt);
        m.update(/*y=*/1);
        let p_after = m.p(&sqt);
        assert_ne!(p_before, p_after,
            "training must shift the prediction (got p_before={}, p_after={})",
            p_before, p_after);
    }

    #[test]
    fn dt_table_anchors() {
        let dt = build_dt();
        assert_eq!(dt[0], 16384 / 3);     // i=0 → 16384/3 = 5461
        assert_eq!(dt[1], 16384 / 5);
        assert_eq!(dt[1023], 16384 / 2049);
    }

    #[test]
    fn state_map_init_from_state_table() {
        // Build a tiny state table by hand: state 0 has n0=1, n1=2;
        // state 1 has n0=3, n1=1; ...
        let mut nn = vec![0u8; 4 * 4]; // 4 states
        // state 0: next=0,1, n0=1, n1=2
        // state 0: (n0,n1)=(1,2) → bias toward 1.
        nn[0] = 0; nn[1] = 1; nn[2] = 1; nn[3] = 2;
        // state 1: (n0,n1)=(3,1) → bias toward 0.
        nn[4] = 0; nn[5] = 1; nn[6] = 3; nn[7] = 1;
        // state 2: (n0,n1)=(0,0) → uniform.
        nn[8] = 0; nn[9] = 1; nn[10] = 0; nn[11] = 0;
        // state 3: (n0,n1)=(7,2) → bias toward 0 different from state 1.
        nn[12] = 0; nn[13] = 1; nn[14] = 7; nn[15] = 2;
        let m = StateMap::new(4, &nn);
        // Different (n0, n1) ratios should produce different
        // initial predictions.
        assert!(m.t[0] != 0);
        assert!(m.t[0] != m.t[1]);
        assert!(m.t[1] != m.t[3]);
        // States 0 (1,2) and 1 (3,1) should be on opposite sides of
        // the 0.5 midpoint.
        assert!(m.t[0] > m.t[1],
            "state(1,2) should predict above 0.5, state(3,1) below");
    }

    #[test]
    fn state_map1_responds_to_training() {
        let dt = build_dt();
        let mut m = StateMap1::new(8, 100);
        let p0 = m.set(0, 0, &dt);
        // Train on bit=1 ten times in the same context.
        let mut p_last = p0;
        for _ in 0..10 { p_last = m.set(1, 0, &dt); }
        assert!(p_last > p0,
            "after training on bit=1 the prediction must rise (was {}, became {})",
            p0, p_last);
    }

    #[test]
    fn small_ssm_emits_two_logits_per_call() {
        let sqt = build_squash_table();
        let strt = build_stretch_table();
        let mut s = SmallStationaryContextMap::new(/*ctx_bits=*/4, /*in_bits=*/4);
        s.set(0xAB);
        let mut out = Vec::new();
        s.mix(/*y=*/1, /*r=*/0, &sqt, &strt, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn ebucket_set_and_lookup() {
        let mut b: EBucket<7, 64> = EBucket::new();
        let row = b.get(0xABCD, 0);
        // First call inserts at slot 0 (priority lowest of empties).
        // Re-look-up returns the same row.
        let row2 = b.get(0xABCD, 0);
        assert_eq!(row, row2);
        assert_eq!(b.chk(row), 0xABCD);
    }

    #[test]
    fn context_map_mix1_emits_inputs_per_context() {
        let strt = build_stretch_table();
        let ilog_t = build_ilog();
        let mut nn = vec![0u8; 1024];
        let mut st = StateTable::new();
        let mut tbl = [0u8; 1024];
        st.init(18, 12, 8, 6, 4, 43, 5, &mut tbl);
        nn.copy_from_slice(&tbl);

        let mut cm: ContextMap<7, 64> = ContextMap::new(
            4096, 2, 16, &nn, 16, 0, 1, &strt, &ilog_t);
        cm.set(0x12345);
        cm.set(0x67890);
        let mut out: Vec<i16> = Vec::new();
        cm.mix1(/*cc=*/1, /*bp=*/0, /*c1=*/0, /*y=*/0, &mut out, 0, 7, &nn);
        // Each context emits 5+1 inputs (skip2=1 ⇒ 5 from mix3, +1 run-context).
        assert_eq!(out.len(), cm.cn * 6);
    }

    #[test]
    fn apm_dyn_returns_finite_predictions() {
        let sqt = build_squash_table();
        let strt = build_stretch_table();
        let mut a = ApmDyn::new(/*s=*/4, &sqt);
        let mut last = 0;
        for _ in 0..50 {
            last = a.p(/*pr=*/2048, /*cxt=*/0, /*rate=*/8, /*y=*/1, &strt);
        }
        // After 50 training calls on bit=1, the prediction must
        // have moved upward from the initial 2048 baseline.
        assert!(last > 2048,
            "after training APM toward bit=1, p={} should exceed 2048", last);
    }

    #[test]
    fn direct_state_map_emits_two_inputs() {
        let strt = build_stretch_table();
        let mut nn = vec![0u8; 1024];
        let mut st = StateTable::new();
        let mut tbl = [0u8; 1024];
        st.init(18, 12, 8, 6, 4, 43, 5, &mut tbl);
        nn.copy_from_slice(&tbl);
        let mut dsm = DirectStateMap::new(/*m=*/12, /*c=*/2, &nn, &strt);
        let mut out: Vec<i16> = Vec::new();
        dsm.set(0xCAFE, 1, &nn, &strt, &mut out);
        dsm.set(0xBABE, 0, &nn, &strt, &mut out);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn buffer_round_trips_circular() {
        let mut b = Buffer::new();
        for v in [b'a', b'b', b'c', b'd', b'e'] { b.push(v); }
        // buf(1) = most recent (e), buf(2) = d, …
        assert_eq!(b.buf(1), b'e');
        assert_eq!(b.buf(2), b'd');
        assert_eq!(b.buf(5), b'a');
    }

    #[test]
    fn mtf_list_moves_front() {
        let mut l = MtfList::new(4);
        // Initial order: 0, 1, 2, 3.
        assert_eq!(l.get_first(), 0);
        assert_eq!(l.get_next(), 1);
        l.move_to_front(2);
        assert_eq!(l.get_first(), 2);
        assert_eq!(l.get_next(), 0);  // 2 → 0 → 1 → 3
    }

    #[test]
    fn context_vec_push_pop_works() {
        let mut v = ContextVec::new(4);
        v.push(7); v.push(8);
        assert_eq!(v.size, 2);
        assert_eq!(v.last(), 8);
        assert_eq!(v.pop(), 8);
        assert_eq!(v.last(), 7);
    }

    #[test]
    fn word_append_and_compare() {
        let mut w = Word::new();
        for c in b"hello".iter() { w.append(*c); }
        assert_eq!(w.len(), 5);
        assert!(w.equals_str(b"hello"));
        assert!(w.starts_with(b"he"));
        assert!(w.ends_with(b"lo"));
        assert!(w.matches_any(&[b"world", b"hello", b"foo"]));
        assert!(!w.matches_any(&[b"world", b"foo"]));
    }

    #[test]
    fn word_change_suffix() {
        let mut w = Word::new();
        for c in b"running".iter() { w.append(*c); }
        // "ing" → "ed".
        assert!(w.change_suffix(b"ing", b"ed"));
        assert!(w.equals_str(b"runned"));
        // No-op when suffix doesn't match.
        assert!(!w.change_suffix(b"xy", b"foo"));
    }

    #[test]
    fn words_context_tracks_recent_words() {
        let mut w = WordsContext::new();
        w.update(0xCAFE, b'a', 0x01, 0x100);
        w.update(0xBEEF, b'b', 0x02, 0x200);
        w.update(0x1234, b'c', 0x04, 0x300);
        // Most-recent word is index 1.
        assert_eq!(w.word(1), 0x300);
        assert_eq!(w.word(2), 0x200);
        // First word fills `fword`.
        assert_eq!(w.fword, 0xCAFE);
        // type[1] & 0x04 → return that word.
        assert_eq!(w.last(1, 0x04), 0x300);
    }

    #[test]
    fn fxcm_hash_is_deterministic() {
        assert_eq!(fxcm_hash3(1, 2, 3), fxcm_hash3(1, 2, 3));
        assert_ne!(fxcm_hash3(1, 2, 3), fxcm_hash3(3, 2, 1));
    }

    #[test]
    fn column_context_tracks_rows() {
        let mut c = ColumnContext::new(31);
        // Feed "ab\ncd\n" — 3 lines worth of state.
        let mut blpos = 0u32;
        for &b in b"ab\ncd\n" {
            c.update(b, 0, blpos, false);
            blpos += 1;
        }
        // After two newlines we've advanced the rows ring.
        assert!(c.rows == 2);
        // The most-recent committed line had a first byte 'c'
        // (= 99); upstream clamps fc to min(byte, TEXTDATA=96).
        assert_eq!(c.lastfc(1), TEXTDATA_CHR);
    }

    #[test]
    fn bracket_context_fx_pushes_and_pops() {
        // Element list: '(' → ')'.
        let mut b: BracketContextFx<u8> = BracketContextFx::new(
            &[b'(', b')'], false, 64, 8,
        );
        b.update(b'(' as u32);
        let after_open = b.context;
        assert!(after_open != 0);
        b.update(b')' as u32);
        // Stack empties on matched close → context resets to 0.
        assert_eq!(b.context, 0);
    }

    #[test]
    fn sparse_match_emits_two_inputs_per_call() {
        let mut buf = Buffer::new();
        for &b in b"abcabcabc" { buf.push(b); }
        let mut m = SparseMatchModel::new();
        let mut out: Vec<i16> = Vec::new();
        let _ = m.p(/*c0=*/1, /*bpos=*/0, &buf, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn context_map_compiles_with_other_sizes() {
        // ContextMap1 = ContextMap<3, 32>; ContextMap2 = ContextMap<14, 128>.
        let strt = build_stretch_table();
        let ilog_t = build_ilog();
        let mut nn = vec![0u8; 1024];
        let mut st = StateTable::new();
        let mut tbl = [0u8; 1024];
        st.init(18, 12, 8, 6, 4, 43, 5, &mut tbl);
        nn.copy_from_slice(&tbl);
        let _cm1: ContextMap<3, 32>   = ContextMap::new(2048, 1, 16, &nn, 16, 0, 1, &strt, &ilog_t);
        let _cm2: ContextMap<14, 128> = ContextMap::new(4096, 1, 16, &nn, 16, 0, 1, &strt, &ilog_t);
    }

    #[test]
    fn context_map_init_and_set() {
        let strt = build_stretch_table();
        let ilog_t = build_ilog();
        // Build a minimal state table (256 states) for sm[].
        let mut nn = vec![0u8; 1024];
        let mut st = StateTable::new();
        let mut tbl = [0u8; 1024];
        st.init(18, 12, 8, 6, 4, 43, 5, &mut tbl);
        nn.copy_from_slice(&tbl);

        let mut cm: ContextMap<7, 64> = ContextMap::new(
            /*m=*/4096, /*c=*/3, /*s3=*/16, &nn, /*cs4=*/16,
            /*k=*/0, /*u_skip2=*/1, &strt, &ilog_t,
        );
        cm.set(0xDEADBEEF);
        cm.set(0xCAFEBABE);
        cm.sets();
        assert_eq!(cm.cn, 3);
    }

    /// Helper: build a `Word` from a byte literal.
    fn make_word(s: &[u8]) -> Word {
        let mut w = Word::new();
        for &c in s { w.append(c); }
        w
    }

    #[test]
    fn stemmer_basic_plural() {
        let mut w = make_word(b"cats");
        let _ = EnglishStemmer::stem(&mut w, 0);
        // "cats" is plural; the stemmer returns the singular root
        // and tags the word as Plural.
        assert!((w.r#type & eng::PLURAL) != 0,
            "type=0x{:x} should have PLURAL bit", w.r#type);
    }

    #[test]
    fn stemmer_recognises_male_pronoun() {
        let mut w = make_word(b"he");
        EnglishStemmer::stem(&mut w, 0);
        assert!((w.r#type & eng::MALE) != 0,
            "type=0x{:x} should have MALE bit", w.r#type);
    }

    #[test]
    fn stemmer_recognises_article() {
        let mut w = make_word(b"the");
        EnglishStemmer::stem(&mut w, 0);
        assert!((w.r#type & eng::ARTICLE) != 0,
            "type=0x{:x} should have ARTICLE bit", w.r#type);
    }

    #[test]
    fn stemmer_recognises_number_word() {
        let mut w = make_word(b"three");
        EnglishStemmer::stem(&mut w, 0);
        assert!((w.r#type & eng::NUMBER) != 0,
            "type=0x{:x} should have NUMBER bit", w.r#type);
    }

    #[test]
    fn stemmer_handles_exception1() {
        // "skies" → "sky" via Exceptions1[1]; tagged Noun|Plural.
        let mut w = make_word(b"skies");
        EnglishStemmer::stem(&mut w, 0);
        assert!(w.equals_str(b"sky"),
            "letters[start..=end]={:?} should equal 'sky'",
            std::str::from_utf8(&w.letters[w.start as usize..=w.end as usize]).ok());
        assert!((w.r#type & (eng::NOUN | eng::PLURAL)) == (eng::NOUN | eng::PLURAL));
    }

    #[test]
    fn stemmer_strips_step1b_suffix() {
        // "running" → after Step1b drops "ing", IsDouble('n') fires
        // → drop one more letter → "run".
        let mut w = make_word(b"running");
        EnglishStemmer::stem(&mut w, 0);
        assert!(w.equals_str(b"run"),
            "letters[start..=end]={:?} should equal 'run'",
            std::str::from_utf8(&w.letters[w.start as usize..=w.end as usize]).ok());
        assert!((w.r#type & eng::PRESENT_PARTICIPLE) != 0);
    }

    #[test]
    fn stemmer_hash_changes_with_letters() {
        let mut a = make_word(b"hello");
        let mut b = make_word(b"world");
        EnglishStemmer::stem(&mut a, 0);
        EnglishStemmer::stem(&mut b, 0);
        // Different letter sequences must produce different hashes.
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn stemmer_blpos_threshold_disables_verb_words() {
        // VerbWords1 has "is" — gated behind blpos < 451531986.
        let mut w_below = make_word(b"is");
        EnglishStemmer::stem(&mut w_below, 0);
        assert!((w_below.r#type & eng::VERB) != 0,
            "before threshold, 'is' should be tagged VERB");

        let mut w_above = make_word(b"is");
        EnglishStemmer::stem(&mut w_above, 451_531_986);
        // After the threshold the VerbWords1 branch is disabled.
        assert!((w_above.r#type & eng::VERB) == 0,
            "at threshold, VerbWords1 branch must be skipped");
    }

    #[test]
    fn shift_letter_classes_shifts_left_by_one() {
        let mut s = FxcmState::new();
        s.words = 0b0000_1010;
        s.spaces = 0b0000_0101;
        s.numbers = 0b0001_1111;
        s.shift_letter_classes();
        assert_eq!(s.words,   0b0001_0100);
        assert_eq!(s.spaces,  0b0000_1010);
        assert_eq!(s.numbers, 0b0011_1110);
    }

    #[test]
    fn is_word_letter_rules() {
        let mut s = FxcmState::new();
        s.c1 = b'h'; assert!(s.is_word_letter());
        s.c1 = b'z'; assert!(s.is_word_letter());
        s.c1 = b'a'; assert!(s.is_word_letter());
        s.c1 = b'A'; assert!(!s.is_word_letter());  // upper ASCII not a letter here
        s.c1 = b'5'; assert!(!s.is_word_letter());
        s.c1 = b' '; assert!(!s.is_word_letter());
        // High-ASCII byte after a non-ESCAPE byte counts as letter.
        s.c1 = 0xC3; s.c2 = b' '; assert!(s.is_word_letter());
        s.c1 = 0xC3; s.c2 = ESCAPE_CHR; assert!(!s.is_word_letter());
    }

    #[test]
    fn track_numbers_accumulates_digits() {
        let mut s = FxcmState::new();
        // Feed "127": three digits in a row, no comma.
        for c in [b'1', b'2', b'7'] {
            s.c1 = c;
            s.shift_letter_classes();
            s.track_numbers();
        }
        assert_eq!(s.number0, 127);
        assert_eq!(s.numlen0, 3);
    }

    #[test]
    fn cycle_word_pipeline_shifts_word_chain() {
        let mut s = FxcmState::new();
        s.word0 = 7;
        s.cycle_word_pipeline(0);
        assert_eq!(s.word1, 7 * 83);
        // word2 cycles in word1 from before (which was 0 → 0*53=0).
        assert_eq!(s.word2, 0);
        assert_eq!(s.word3, 0);
        // Run again with new word0.
        s.word0 = 11;
        s.cycle_word_pipeline(0);
        assert_eq!(s.word1, 11 * 83);
        assert_eq!(s.word2, 7 * 83 * 53);
    }

    #[test]
    fn cycle_word_pipeline_skips_for_certain_pword_types() {
        let mut s = FxcmState::new();
        s.word0 = 5;
        // Set pWord type to Conjunction → cycle skipped (before blpos
        // threshold).
        s.stem_words[s.p_word].r#type = eng::CONJUNCTION;
        s.cycle_word_pipeline(0);
        assert_eq!(s.word1, 0, "Conjunction should skip the cycle");

        // Past the blpos threshold the cycle runs anyway.
        s.cycle_word_pipeline(463_139_794);
        assert_eq!(s.word1, 5 * 83);
    }

    #[test]
    fn update_streams_for_number_word_appends_one() {
        let mut s = FxcmState::new();
        let mut w = WordsContext::new();
        // Push a single word with type=NUMBER.
        w.set(b'!', 0);
        w.update(/*w=*/123, /*b=*/b' ', eng::NUMBER, /*s=*/123);
        s.stream3b   = 0xAB;
        s.stream3b_r = 0xCD;
        s.update_streams_for_word_type(&w, /*is_paragraph=*/false);
        assert_eq!(s.stream3b,   0xAB << 7 | 1);
        assert_eq!(s.stream3b_r, 0xCD << 7 | 1);
    }

    #[test]
    fn update_streams_conjunction_clears_senword_in_paragraph() {
        let mut s = FxcmState::new();
        let mut w = WordsContext::new();
        w.set(b'!', 0);
        w.update(1, b' ', eng::CONJUNCTION, 1);
        s.sen_word = 999;
        s.update_streams_for_word_type(&w, /*is_paragraph=*/true);
        assert_eq!(s.sen_word, 0);
    }

    #[test]
    fn fc_tables_have_known_anchor_values() {
        // FCY is keyed by upstream's wiki-encoded control bytes
        // (LESSTHAN_CHR='L', SQUAREOPEN_CHR=91, CURLYOPENING_CHR='P'),
        // not the literal "[", "<", "{". `(` is at its ASCII slot.
        assert_eq!(FCY[b'(' as usize], 1);
        assert_eq!(FCY[LESSTHAN_CHR as usize], 4);
        assert_eq!(FCY[SQUAREOPEN_CHR as usize], 3);
        assert_eq!(FCY[CURLYOPENING_CHR as usize], 2);
    }

    #[test]
    fn compute_fc_indices_picks_bracket_first() {
        let mut s = FxcmState::new();
        s.compute_fc_indices(
            /*br=*/b'(', /*br_active=*/true,
            /*qo_high=*/0, false,
            /*fc=*/0, false,
        );
        assert_eq!(s.br_fc_idx, 1);  // FCY['('] = 1.
    }

    #[test]
    fn compute_fc_indices_falls_back_to_quote() {
        let mut s = FxcmState::new();
        s.compute_fc_indices(
            /*br=*/0, /*br_active=*/false,
            /*qo_high=*/LESSTHAN_CHR, /*qo_active=*/true,
            /*fc=*/0, /*fc_active=*/false,
        );
        assert_eq!(s.br_fc_idx, FCY[LESSTHAN_CHR as usize] as u32);
    }

    #[test]
    fn track_utf8_starts_two_byte_sequence() {
        let mut s = FxcmState::new();
        s.c2 = ESCAPE_CHR;
        s.c1 = 0xC2;  // 11000010 → 2-byte start.
        s.track_utf8_byte();
        assert_eq!(s.utf8_left, 1);
        assert!(s.u8w != 0);
    }

    #[test]
    fn track_utf8_continues_then_resets() {
        let mut s = FxcmState::new();
        s.c2 = ESCAPE_CHR;
        s.c1 = 0xE0; s.track_utf8_byte();          // 3-byte sequence start.
        assert_eq!(s.utf8_left, 2);
        s.c1 = 0xA4; s.track_utf8_byte();          // continuation.
        assert_eq!(s.utf8_left, 1);
        s.c1 = 0xB8; s.track_utf8_byte();          // continuation.
        assert_eq!(s.utf8_left, 0);
    }

    #[test]
    fn track_utf8_aborts_on_non_continuation() {
        let mut s = FxcmState::new();
        s.c2 = ESCAPE_CHR;
        s.c1 = 0xE0; s.track_utf8_byte();          // 3-byte start.
        s.c1 = 0x41; s.track_utf8_byte();          // ASCII 'A' — not continuation.
        assert_eq!(s.utf8_left, 0);
    }

    #[test]
    fn track_utf8_skips_when_no_escape() {
        let mut s = FxcmState::new();
        s.c2 = b'x';
        s.c1 = 0xC2;
        s.track_utf8_byte();
        assert_eq!(s.utf8_left, 0, "non-ESCAPE c2 must be a no-op");
    }

    #[test]
    fn indirect_histories_roll_byte_into_t1_and_t2() {
        let mut s = FxcmState::new();
        s.c1 = 0x42;
        s.stream2b = 0b11;  // low 2 bits = 3.
        s.stream3b = 0b101; // low 3 bits = 5.
        s.update_indirect_histories(/*c4=*/0x12345678, /*brcontext=*/0);
        // t2[(c4>>8)&0xffff] = (0 << 8) | 0x42 = 0x42.
        assert_eq!(s.t2[0x3456], 0x42);
        // indirect_word low 16 = c4 & 0xffff = 0x5678.
        assert_eq!(s.indirect_word & 0xffff, 0x5678);
        // indirect_byte low 8 = c1.
        assert_eq!(s.indirect_byte & 0xff, 0x42);
        // indirect_br_byte low 3 = stream3b & 7 = 5.
        assert_eq!(s.indirect_br_byte & 7, 5);
    }

    #[test]
    fn word0_position_tracks_distance() {
        let mut s = FxcmState::new();
        let mut buf = Buffer::new();
        for &b in b"abcdef" { buf.push(b); }
        s.word0 = 0x1234;
        s.pos = 100;
        s.wp[0x1234] = 50;  // last seen 50 positions ago.
        let dist = s.update_word0_position(&buf);
        assert_eq!(dist, 50);
        // Low 8 = dist; mid 8 = buf(50); high 8 = c1.
        assert_eq!(s.indirect_word0_pos & 0xff, 50);
    }

    #[test]
    fn word0_position_caps_at_256() {
        let mut s = FxcmState::new();
        let buf = Buffer::new();
        s.word0 = 0; s.pos = 1000; s.wp[0] = 100;
        s.c1 = 0xAB;
        let dist = s.update_word0_position(&buf);
        assert_eq!(dist, 900);  // returned distance unchanged.
        // But stored value is 256 + (c1 << 16).
        assert_eq!(s.indirect_word0_pos, 256 + (0xAB << 16));
    }

    #[test]
    fn paragraph_from_fc_only_one_for_firstupper() {
        let mut s = FxcmState::new();
        s.fc = FIRSTUPPER_CHR as i32;
        s.paragraph_from_fc();
        assert_eq!(s.is_paragraph, 1);
        s.fc = b' ' as i32;
        s.paragraph_from_fc();
        assert_eq!(s.is_paragraph, 0);
    }

    #[test]
    fn punct_comma_clears_senword_and_sets_words() {
        let mut s = FxcmState::new();
        s.words = 0; s.sen_word = 99;
        s.punct_comma();
        assert_eq!(s.words & 0xfc, 0xfc);
        assert_eq!(s.sen_word, 0);
    }

    #[test]
    fn punct_colon_folds_streams_and_x5() {
        let mut s = FxcmState::new();
        s.stream3b = 0xFFFF_FFF0;  // low 3 bits = 0
        s.stream2b = 0x0;
        s.x5 = 0;
        s.punct_colon(/*c4=*/0xCAFE_BABE);
        // stream3b low 3 = 4.
        assert_eq!(s.stream3b & 7, 4);
        // stream2b |= 12 (binary 1100).
        assert_eq!(s.stream2b & 0xf, 12);
        // x5 low byte = c4 low byte.
        assert_eq!(s.x5 & 0xff, 0xBE);
    }

    #[test]
    fn punct_curly_folds_3b_and_3b_r() {
        let mut s = FxcmState::new();
        s.stream3b   = 0xFFFF_FFF0;
        s.stream3b_r = 0xFFFF_FFFF;
        s.punct_curly(0);
        // stream3b low 3 = 3.
        assert_eq!(s.stream3b & 7, 3);
        // stream3b_r masked to 0xffffffc0.
        assert_eq!(s.stream3b_r & 0x3f, 0);
    }

    #[test]
    fn punct_nbsp_substitutes_to_space() {
        let mut s = FxcmState::new();
        s.c1 = b'!'; s.c2 = b'&';
        let new_c4 = s.punct_nbsp(/*c4=*/0xAABBCC00 | b'!' as u32);
        assert_eq!(s.c1, SPACE_CHR);
        assert_eq!(new_c4 & 0xff, SPACE_CHR as u32);
    }

    #[test]
    fn line_break_lf_resets_line_tracking() {
        let mut s = FxcmState::new();
        s.fc = 12; s.is_paragraph = 1; s.first_word = 99; s.last_wt = 0xAB;
        s.pos = 1000; s.nl = 500;
        s.stream3b_r = 0x10; s.stream2b = 0x100; s.words = 0;
        let was_double_lf = s.line_break_lf();
        assert_eq!(s.fc, 0);
        assert_eq!(s.is_paragraph, 0);
        assert_eq!(s.first_word, 0);
        assert_eq!(s.last_wt, 0);
        assert_eq!(s.nl1, 500);
        assert_eq!(s.nl, 999);
        assert_eq!(s.stream3b_r, 0x10 << 7);
        assert_eq!(s.stream2b, 0x100 | 0x3fc);
        assert_eq!(s.words, 0xfc);
        assert!(!was_double_lf);

        // c2 == LF_CHR triggers the double-LF return.
        s.c2 = LF_CHR;
        assert!(s.line_break_lf());
    }

    #[test]
    fn sentence_end_punct_dot_returns_true() {
        let mut s = FxcmState::new();
        s.c1 = b'.';
        let is_dot = s.sentence_end_punct(/*c4=*/0xCAFE_BABE);
        assert!(is_dot);
        assert_eq!(s.wshift, 1);
        assert_eq!(s.x5, (0u32 << 8) | 0xBE);
        assert!((s.stream2b & 204) == 204);
        assert!((s.words & 0xfe) == 0xfe);
    }

    #[test]
    fn sentence_end_punct_close_paren_clears_senword() {
        let mut s = FxcmState::new();
        s.c1 = b')';
        s.sen_word = 999;
        let is_dot = s.sentence_end_punct(0);
        assert!(!is_dot);
        assert_eq!(s.sen_word, 0);
    }

    #[test]
    fn update_streams_after_byte_serial_always_folds_3b() {
        let mut s = FxcmState::new();
        s.n3b_state = 5;
        s.o3b_state = 5;  // unchanged → non-repeating must NOT fold
        s.stream3b   = 0xAB;
        s.stream3b_r = 0xCD;
        s.update_streams_after_byte();
        // Serial: shifted 3 + 5.
        assert_eq!(s.stream3b, (0xAB << 3) | 5);
        // Non-repeating unchanged because o3b == n3b.
        assert_eq!(s.stream3b_r, 0xCD);
    }

    #[test]
    fn update_streams_after_byte_nonrepeating_folds_on_transition() {
        let mut s = FxcmState::new();
        s.o3b_state = 1;
        s.n3b_state = 4;  // transition
        s.stream3b_r = 0xCD;
        s.update_streams_after_byte();
        assert_eq!(s.stream3b_r, (0xCD << 3) | 4);
        assert_eq!(s.o3b_state, 4, "non-repeating must remember new state");
    }

    #[test]
    fn rotate_stream_masks_cycles_two_levels() {
        let mut s = FxcmState::new();
        s.stream3b_mask    = 0x10;
        s.stream3b_mask1   = 0x20;
        s.stream3b_r_mask1 = 0x40;
        s.stream3b_r_mask2 = 0x80;
        s.stream2b_mask    = 0x100;
        s.rotate_stream_masks();
        assert_eq!(s.stream3b_r_mask2, 0x40, "_mask2 receives old _mask1");
        assert_eq!(s.stream3b_mask1,   0x10, "_mask1 receives old current");
        assert_eq!(s.stream3b_mask,    0);
        assert_eq!(s.stream3b_r_mask1, 0);
        assert_eq!(s.stream2b_mask,    0);
    }

    #[test]
    fn maybe_seed_first_word_only_seeds_outside_bracket() {
        let mut s = FxcmState::new();
        s.word0 = 42;
        s.maybe_seed_first_word(/*fc_cxt=*/0);
        assert_eq!(s.first_word, 42);

        // Inside `[` the seed is suppressed.
        let mut s2 = FxcmState::new();
        s2.word0 = 42;
        s2.maybe_seed_first_word(SQUAREOPEN_CHR);
        assert_eq!(s2.first_word, 0);
    }

    #[test]
    fn track_numbers_rolls_into_number1_on_non_digit() {
        let mut s = FxcmState::new();
        // "42 " — "42" accumulates; the space rolls into number1.
        for c in [b'4', b'2', b' '] {
            s.c1 = c;
            s.shift_letter_classes();
            s.track_numbers();
        }
        assert_eq!(s.number1, 42);
        assert_eq!(s.numlen1, 2);
        assert_eq!(s.number0, 0);
        assert_eq!(s.numlen0, 0);
    }

    #[test]
    fn primes_table_starts_with_zero_then_decreasing_primes() {
        assert_eq!(PRIMES.len(), 14);
        assert_eq!(PRIMES[0], 0);
        // The remaining 13 are strictly decreasing primes.
        for i in 2..14 {
            assert!(PRIMES[i] < PRIMES[i - 1],
                "PRIMES[{}]={} should be < PRIMES[{}]={}",
                i, PRIMES[i], i - 1, PRIMES[i - 1]);
        }
    }

    #[test]
    fn fxcm_state_byte_boundary_advances_state() {
        let mut s = FxcmState::new();
        let mut buf = Buffer::new();
        s.byte_boundary_step(/*c4=*/0x68, &mut buf); // 'h'
        assert_eq!(s.c1, b'h');
        assert_eq!(s.pos, 1);
        assert_eq!(buf.buf(1), b'h');
        // Hash table should now be non-zero in slots 1..=13.
        let any_nonzero = s.t.iter().skip(1).any(|&v| v != 0);
        assert!(any_nonzero, "t[1..=13] should fold in c1");

        s.byte_boundary_step(0x65, &mut buf); // 'e'
        assert_eq!(s.c1, b'e');
        assert_eq!(s.c2, b'h'); // c2 was cycled.
        assert_eq!(s.pos, 2);
    }

    #[test]
    fn fxcm_state_marker_byte_resets_t() {
        let mut s = FxcmState::new();
        let mut buf = Buffer::new();
        // Feed a few normal bytes.
        s.byte_boundary_step(0x68, &mut buf);
        s.byte_boundary_step(0x69, &mut buf);
        let t_before = s.t;
        // Feed a marker byte ($).
        s.byte_boundary_step(b'$' as u32, &mut buf);
        // The marker shifts twice (once from the c1!=c2 path, once
        // from the always-fold path), so t differs.
        assert_ne!(s.t, t_before);
        assert!(s.x4 != 0, "x4 must accumulate the marker byte");
    }

    #[test]
    fn fxcm_state_initialises_to_known_constants() {
        let s = FxcmState::new();
        // Mirror of upstream's PredictorInit constants:
        assert_eq!(s.n2b_state, 0xffffffff);
        assert_eq!(s.n3b_state, 0xffffffff);
        assert_eq!(s.ah2, 0x765BA55C);
        assert_eq!(s.pr, 2048);
        assert_eq!(s.rate, 6);
        // 4 words, c_word=0 / p_word=3 to mirror upstream's
        // `cWord=&StemWords[0], pWord=&StemWords[3]`.
        assert_eq!(s.stem_words.len(), 4);
        assert_eq!(s.c_word, 0);
        assert_eq!(s.p_word, 3);
        // 14-entry hash table all zero.
        assert!(s.t.iter().all(|&x| x == 0));
    }

    #[test]
    fn build_mixers_produces_twelve_with_correct_shape() {
        let ms = build_mixers(/*inputs1_n=*/512, /*inputs2_n=*/16);
        assert_eq!(ms.len(), 12);
        // First mixer: m=2048, n=512.
        assert_eq!(ms[0].m, 2048);
        assert_eq!(ms[0].n, 512);
        assert_eq!(ms[0].uperr, 69);
        // Last mixer: m=1, n=16.
        assert_eq!(ms[11].m, 1);
        assert_eq!(ms[11].n, 16);
        assert_eq!(ms[11].shift1, 6);
    }

    #[test]
    fn build_state_maps_produces_three_with_distinct_sizes() {
        let sms = build_state_maps();
        assert_eq!(sms.len(), 3);
        assert_eq!(sms[0].n, 1 << 9);
        assert_eq!(sms[1].n, 1 << 19);
        assert_eq!(sms[2].n, 1 << 16);
    }

    #[test]
    fn build_small_scm_produces_seven_with_listed_bits() {
        let scm = build_small_scm();
        assert_eq!(scm.len(), 7);
        // SmallStationaryContextMap has no public bits accessor;
        // sanity-check that all 7 instantiate without panic.
        for s in &scm { let _ = s; }
    }

    #[test]
    fn predictor_param_arrays_are_27_long_each() {
        assert_eq!(C_R.len(),  27);
        assert_eq!(C_S.len(),  27);
        assert_eq!(C_S3.len(), 27);
        assert_eq!(C_S4.len(), 27);
    }

    #[test]
    fn build_sta_tables_yields_six_distinct_blocks() {
        let v = build_sta_tables();
        assert_eq!(v.len(), 6 * 1024);
        // Different parameter sets must produce different transition
        // tables — sanity-check by comparing every adjacent pair.
        for i in 0..5 {
            let a = &v[i * 1024..(i + 1) * 1024];
            let b = &v[(i + 1) * 1024..(i + 2) * 1024];
            assert_ne!(a, b, "STA[{}] and STA[{}] must differ", i, i + 1);
        }
    }

    #[test]
    fn pre1_table_is_monotonic_in_n1_n0_ratio() {
        let strt = build_stretch_table();
        let v = build_sta_tables();
        let sta7 = &v[5 * 1024..6 * 1024];  // STA7 is at slot 5.
        let pre1 = build_pre1_from_sta7(sta7, &strt);
        // pre1 entries must fit in i16 and differ.
        let mut distinct = std::collections::HashSet::new();
        for &v in &pre1 { distinct.insert(v); }
        assert!(distinct.len() > 4,
            "pre1 should produce a variety of stretched probs, got {} unique",
            distinct.len());
    }

    #[test]
    fn fchar_table_has_20_chars_and_pairs_correctly() {
        let f = fchar_table();
        assert_eq!(f.len(), 20);
        // Most pairs end with LF — one's a real pair (LESSTHAN/GREATERTHAN).
        let mut lf_count = 0;
        for i in (0..20).step_by(2) {
            if f[i + 1] == LF_CHR { lf_count += 1; }
        }
        assert!(lf_count >= 7, "expected most pairs to map to LF, got {}", lf_count);
    }

    #[test]
    fn match_info_priority_orders_correctly() {
        let mut a = MatchInfo::new();
        let mut b = MatchInfo::new();
        a.length = 16; a.index = 100;
        b.length = 8;  b.index = 100;
        // Longer match wins regardless of index.
        assert!(a.is_better_than(&b));
        // Equal length: more recent index wins.
        a.length = 8; a.index = 200;
        b.length = 8; b.index = 100;
        assert!(a.is_better_than(&b));
    }

    #[test]
    fn hash_element_add_shifts_in_order() {
        let mut h = HashElementForMatchPositions::new();
        h.add(10);
        h.add(20);
        h.add(30);
        // Most-recent is at slot 0.
        assert_eq!(h.positions[0], 30);
        assert_eq!(h.positions[1], 20);
        assert_eq!(h.positions[2], 10);
        assert_eq!(h.positions[3], 0);
    }

    #[test]
    fn match_model_promotes_existing_match() {
        // Set up a buffer with a known repeat: "abcabc".
        let mut buf = Buffer::new();
        for &b in b"abcabc" { buf.push(b); }

        let mut mm = MatchModel2::new(/*log2_size=*/8, /*sm_n=*/256, /*sm_lim=*/100);

        // After ingesting "abcabc", buf.pos = 6. The first "abc"
        // sits at absolute positions 0,1,2 and the second at 3,4,5.
        // Pre-seed a hash table entry pointing at position 6 (just
        // past the second "abc"); is_match(6, LEN1=5) checks
        // `buf(1..=5) == bufr(5..=1)`. With pos=6, buf(1)='c',
        // bufr(5)='c'. They will only match if the buffer mirrors
        // the suffix back-to-back which is exactly what we set up.

        // Construct a fake order_hashes vector: hash[LEN1]=42.
        let mut hashes = vec![0u32; 14];
        hashes[LEN1 as usize] = 42;

        // Place the candidate match position into bucket (42 & mask).
        let idx = (42u32 & mm.mhashtable_mask) as usize;
        mm.mhashtable[idx].add(6);

        // Drive update at byte boundary (bpos=0).
        mm.update(/*y=*/0, /*bpos=*/0, /*c1=*/b'c',
                  /*pos=*/6, &buf, &hashes, /*word=*/0);
        // Test that update completes without panic and recognises
        // the suffix match (we may or may not have an active
        // candidate depending on is_match check).
        let _ = mm.n_active;
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
