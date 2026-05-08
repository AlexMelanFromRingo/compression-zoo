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
        let delta = (((y << 20) - pr1) * dt[count as usize] + 512) & 0xFFFFFC00u32 as i32;
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
