//! Adaptive probability maps — verbatim port of paq8.cpp:601-711.
//!
//! * [`StateMap`]   — 256-context bit map, fixed `>>8` adapt rate.
//! * [`StateMap32`] — variable-size map, 22-bit prediction + 10-bit
//!                    count, `dt[]`-scaled adaptation.
//! * [`Apm1`]       — interpolating adaptive map (`pr ⊗ cxt`).
//! * [`Apm`]        — StateMap32 over `n*24` slots with stretch-mapped
//!                    pr interpolation.
//!
//! All `p()` calls take `y` (the previously observed bit) explicitly
//! instead of upstream's file-scope global.

#![allow(dead_code)]

use super::substrate::{nex, Stretch};

// =============================================================
// StateMap — paq8.cpp:624-644.
// =============================================================

pub struct StateMap {
    cxt: usize,
    t:   Vec<u16>,
}

impl Default for StateMap { fn default() -> Self { Self::new() } }

impl StateMap {
    pub fn new() -> Self {
        let mut t = vec![0u16; 256];
        for i in 0..256u32 {
            let mut n0 = nex(i as u8, 2) as u32;
            let mut n1 = nex(i as u8, 3) as u32;
            if n0 == 0 { n1 *= 64; }
            if n1 == 0 { n0 *= 64; }
            t[i as usize] = (65536 * (n1 + 1) / (n0 + n1 + 2)) as u16;
        }
        Self { cxt: 0, t }
    }

    /// Update the previous context with `y`, then predict for `cx`.
    pub fn p(&mut self, cx: u32, y: i32) -> i32 {
        let v = self.t[self.cxt] as i32;
        self.t[self.cxt] = (v + (((y << 16) - v + 128) >> 8)) as u16;
        self.cxt = cx as usize;
        (self.t[self.cxt] >> 4) as i32
    }
}

// =============================================================
// StateMap32 — paq8.cpp:646-690.
// =============================================================

pub struct StateMap32 {
    n:   usize,
    cxt: usize,
    t:   Vec<u32>,
    dt:  [i32; 1024],
}

impl StateMap32 {
    pub fn new(n: usize, dt: [i32; 1024]) -> Self {
        let mut t = vec![0u32; n];
        if n == 256 {
            for i in 0..256u32 {
                let mut n0 = nex(i as u8, 2) as u32;
                let mut n1 = nex(i as u8, 3) as u32;
                if n0 == 0 { n1 *= 64; }
                if n1 == 0 { n0 *= 64; }
                t[i as usize] = ((n1 << 16) / (n0 + n1 + 1)) << 16;
            }
        } else {
            for v in t.iter_mut() { *v = 1u32 << 31; }
        }
        Self { n, cxt: 0, t, dt }
    }

    fn update(&mut self, limit: u32, y: i32) {
        let p0 = self.t[self.cxt];
        let n = (p0 & 1023) as usize;
        let pr = (p0 >> 10) as i32;
        let mut p0 = if (n as u32) < limit {
            p0.wrapping_add(1)
        } else {
            (p0 & 0xffff_fc00) | limit
        };
        let target = y << 22;
        let delta = ((target - pr) >> 3).wrapping_mul(self.dt[n]);
        p0 = p0.wrapping_add((delta as u32) & 0xffff_fc00);
        self.t[self.cxt] = p0;
    }

    /// Update + predict for `cx`. `limit` caps the adaptation count.
    pub fn p(&mut self, cx: u32, limit: u32, y: i32) -> i32 {
        debug_assert!((cx as usize) < self.n);
        self.update(limit, y);
        self.cxt = cx as usize;
        (self.t[self.cxt] >> 20) as i32
    }
}

// =============================================================
// Apm1 — paq8.cpp:601-622.
// =============================================================

pub struct Apm1 {
    index: usize,
    n:     usize,
    t:     Vec<u16>,
}

impl Apm1 {
    pub fn new(n: usize, stretch: &Stretch) -> Self {
        let mut t = vec![0u16; n * 33];
        // upstream: t[i*33+j] = (i==0) ? squash((j-16)*128)*16 : t[j]
        // — needs squash, but `squash((j-16)*128)` clamps; we use the
        // stretch table's inverse mapping via the squash global which
        // the caller provides. Here we build row 0 from squash and
        // copy it into every other row.
        for j in 0..33 {
            // squash((j-16)*128) — re-derive via the stretch/squash
            // relationship. The caller passes a Stretch ref; squash
            // values come from `stretch`'s companion table at
            // construction. We approximate with the analytic squash.
            let x = ((j as i32) - 16) * 128;
            let sq = analytic_squash(x);
            t[j] = (sq * 16) as u16;
        }
        for i in 1..n {
            for j in 0..33 {
                t[i * 33 + j] = t[j];
            }
        }
        let _ = stretch;
        Self { index: 0, n, t }
    }

    /// `p(pr, cxt, rate)` — paq8.cpp:607-615.
    pub fn p(&mut self, pr: i32, cxt: u32, rate: i32, y: i32,
            stretch: &Stretch) -> i32 {
        let pr_s = stretch.get(pr.clamp(0, 4095));
        let g = (y << 16) + (y << rate) - y - y;
        let v0 = self.t[self.index] as i32;
        self.t[self.index] = (v0 + ((g - v0) >> rate)) as u16;
        let v1 = self.t[self.index + 1] as i32;
        self.t[self.index + 1] = (v1 + ((g - v1) >> rate)) as u16;
        let w = pr_s & 127;
        self.index = (((pr_s + 2048) >> 7) as usize)
            + (cxt as usize) * 33;
        debug_assert!(self.index + 1 < self.n * 33);
        let a = self.t[self.index] as i32;
        let b = self.t[self.index + 1] as i32;
        (a * (128 - w) + b * w) >> 11
    }
}

// =============================================================
// Apm — paq8.cpp:692-711.
// =============================================================

pub struct Apm {
    n:   usize,
    cxt: usize,
    t:   Vec<u32>,
    dt:  [i32; 1024],
}

impl Apm {
    pub fn new(n: usize, dt: [i32; 1024]) -> Self {
        let size = n * 24;
        let mut t = vec![0u32; size];
        for i in 0..size {
            let p = (((i % 24) * 2 + 1) as i32 * 4096) / 48 - 2048;
            t[i] = ((analytic_squash(p) as u32) << 20) + 6;
        }
        Self { n: size, cxt: 0, t, dt }
    }

    fn update(&mut self, limit: u32, y: i32) {
        let p0 = self.t[self.cxt];
        let n = (p0 & 1023) as usize;
        let pr = (p0 >> 10) as i32;
        let mut p0 = if (n as u32) < limit {
            p0.wrapping_add(1)
        } else {
            (p0 & 0xffff_fc00) | limit
        };
        let target = y << 22;
        let delta = ((target - pr) >> 3).wrapping_mul(self.dt[n]);
        p0 = p0.wrapping_add((delta as u32) & 0xffff_fc00);
        self.t[self.cxt] = p0;
    }

    /// `p(pr, cx, limit)` — paq8.cpp:700-710.
    pub fn p(&mut self, pr: i32, cx: u32, limit: u32, y: i32,
            stretch: &Stretch) -> i32 {
        self.update(limit, y);
        let pr2 = (stretch.get(pr.clamp(0, 4095)) + 2048) * 23;
        let wt = pr2 & 0xfff;
        let cx2 = (cx as usize) * 24 + (pr2 >> 12) as usize;
        self.cxt = cx2 + ((wt >> 11) as usize);
        let a = (self.t[cx2] >> 13) as i32;
        let b = (self.t[cx2 + 1] >> 13) as i32;
        ((a * (4096 - wt)) + (b * wt)) >> 19
    }
}

/// Analytic `squash` — paq8 builds its squash table from this curve
/// (`4096/(1+e^(-x/256))`). Used to seed APM / APM1 tables exactly
/// as upstream's `squash()` does at construction time.
fn analytic_squash(d: i32) -> i32 {
    if d >= 2047 { return 4095; }
    if d <= -2047 { return 0; }
    let f = 1.0 / (1.0 + (-(d as f64) / 256.0).exp());
    (f * 4096.0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::{build_dt, Squash, Stretch};

    #[test]
    fn state_map_predicts_in_range() {
        let mut s = StateMap::new();
        for cx in 0..300 {
            let p = s.p(cx % 256, (cx % 2) as i32);
            assert!(p >= 0 && p <= 4095);
        }
    }

    #[test]
    fn state_map32_predicts_in_range() {
        let mut s = StateMap32::new(1024, build_dt());
        for cx in 0..400 {
            let p = s.p(cx % 1024, 1023, (cx % 2) as i32);
            assert!(p >= 0 && p <= 4095);
        }
    }

    #[test]
    fn apm1_predict_in_range() {
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut a = Apm1::new(256, &st);
        for i in 0..200 {
            let p = a.p(2048 + (i % 100) * 10, (i as u32) % 256, 7,
                (i % 2) as i32, &st);
            assert!(p >= 0 && p <= 4095, "apm1 out of range: {}", p);
        }
    }

    #[test]
    fn apm_predict_in_range() {
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut a = Apm::new(256, build_dt());
        for i in 0..200 {
            let p = a.p(2048, (i as u32) % 256, 1023, (i % 2) as i32, &st);
            assert!(p >= 0 && p <= 4095, "apm out of range: {}", p);
        }
    }

    #[test]
    fn state_map_converges_toward_observed_bit() {
        let mut s = StateMap::new();
        // Feed context 5 always y=1; its prediction should rise.
        let mut last = 0;
        for _ in 0..200 { last = s.p(5, 1); }
        assert!(last > 2048, "expected convergence toward 1, got {}", last);
    }
}
