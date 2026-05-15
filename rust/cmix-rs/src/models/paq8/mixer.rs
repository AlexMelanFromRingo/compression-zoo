//! `paq8::Mixer` — verbatim port of upstream paq8.cpp:514-599.
//!
//! Two-stage logistic mixer. The outer mixer holds `S` per-context
//! weight sets; when `S > 1` it owns an inner `mp` mixer (`S` inputs,
//! 1 set) that combines the per-set predictions. Weight pages are
//! lazily allocated per mixer-context, keyed in a `HashMap`.

#![allow(dead_code)]

use std::collections::HashMap;

use super::substrate::{dot_product, train, Squash, Stretch};

pub struct Mixer {
    /// Input count, rounded up to a multiple of 8 (upstream `(n+7)&-8`).
    n: usize,
    /// Number of mixer sets.
    s: usize,
    /// Initial weight value for fresh weight pages.
    init_w: i16,
    /// Stretched inputs for the current bit.
    tx: Vec<i16>,
    nx: usize,
    /// Active context per set (`base + cx`).
    cxt: Vec<usize>,
    ncxt: usize,
    base: usize,
    /// Per-set cached predictions (range `[0, 4095]`).
    pr: Vec<i32>,
    /// Per-context weight pages.
    wx: HashMap<usize, Vec<i16>>,
    /// Inner 2nd-stage mixer (present iff `s > 1`).
    mp: Option<Box<Mixer>>,
}

impl Mixer {
    pub fn new(n: usize, s: usize, init_w: i16) -> Self {
        // Upstream rounds N to a multiple of 8, but `dot_product`
        // rounds the iteration count up to 16; size the input and
        // weight buffers to a multiple of 16 so the trailing (zero)
        // slots are always in-bounds.
        let n_aligned = (n + 15) & !15;
        let mp = if s > 1 {
            Some(Box::new(Mixer::new(s, 1, 0x7fff)))
        } else {
            None
        };
        Self {
            n: n_aligned,
            s,
            init_w,
            tx: vec![0i16; n_aligned],
            nx: 0,
            cxt: vec![0usize; s],
            ncxt: 0,
            base: 0,
            pr: vec![2048i32; s],
            wx: HashMap::new(),
            mp,
        }
    }

    /// Forget the current bit's inputs / contexts (used by tests).
    pub fn reset(&mut self) {
        self.nx = 0;
        self.base = 0;
        self.ncxt = 0;
    }

    /// `add(x)` — push a stretched input.
    pub fn add(&mut self, x: i16) {
        if self.nx < self.tx.len() {
            self.tx[self.nx] = x;
            self.nx += 1;
        }
    }

    /// `set(cx, range)` — register a per-set context.
    pub fn set(&mut self, cx: u32, range: u32) {
        if self.ncxt < self.cxt.len() {
            self.cxt[self.ncxt] = self.base + cx as usize;
            self.ncxt += 1;
            self.base += range as usize;
        }
    }

    /// `update()` — paq8.cpp:528-541. Trains every active set's
    /// weight page toward the observed bit `y`, then clears the
    /// per-bit state.
    pub fn update(&mut self, y: i32) {
        let n = self.n;
        let iw = self.init_w;
        for i in 0..self.ncxt {
            let err = ((y << 12) - self.pr[i]) * 7;
            let ctx = self.cxt[i];
            let wts = self.wx.entry(ctx).or_insert_with(|| vec![iw; n]);
            train(&self.tx[..n], wts, self.nx, err);
        }
        self.nx = 0;
        self.base = 0;
        self.ncxt = 0;
    }

    /// `p()` — paq8.cpp:553-583. Returns the mixed bit-1 probability
    /// in `[0, 4095]`.
    pub fn p(&mut self, y: i32, squash: &Squash, stretch: &Stretch) -> i32 {
        // Pad inputs to a multiple of 8.
        while self.nx & 7 != 0 {
            if self.nx < self.tx.len() { self.tx[self.nx] = 0; }
            self.nx += 1;
        }
        let n = self.n;
        let iw = self.init_w;
        let nx = self.nx;

        if let Some(mut mp) = self.mp.take() {
            // Two-stage: train + run the inner mixer.
            mp.update(y);
            for i in 0..self.ncxt {
                let ctx = self.cxt[i];
                let wts = self.wx.entry(ctx).or_insert_with(|| vec![iw; n]);
                let z = dot_product(&self.tx[..n], &wts[..n], nx);
                self.pr[i] = squash.get((z * 9) >> 9);
                mp.add(stretch.get(self.pr[i]) as i16);
            }
            mp.set(0, 1);
            let result = mp.p(y, squash, stretch);
            self.mp = Some(mp);
            result
        } else {
            // Single-stage (this is the inner `mp`).
            let wts = self.wx.entry(0).or_insert_with(|| vec![iw; n]);
            let z = dot_product(&self.tx[..n], &wts[..n], nx);
            self.base = squash.get((z * 16) >> 13) as usize;
            self.pr[0] = squash.get(z >> 9);
            self.pr[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_constructs_two_stage_when_s_gt_1() {
        let m = Mixer::new(64, 8, 0);
        assert_eq!(m.n, 64);
        assert_eq!(m.s, 8);
        assert!(m.mp.is_some(), "S>1 must allocate an inner mixer");
        let single = Mixer::new(8, 1, 0);
        assert!(single.mp.is_none());
    }

    #[test]
    fn mixer_returns_probability_in_valid_range() {
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut m = Mixer::new(16, 2, 0);
        for cycle in 0..100 {
            m.update(cycle & 1);
            for i in 0..16 { m.add((i as i16 - 8) * 64); }
            m.set(0, 16);
            m.set(1, 16);
            let p = m.p(cycle & 1, &sq, &st);
            assert!(p >= 0 && p <= 4095, "mixer p out of range: {}", p);
        }
    }

    #[test]
    fn mixer_two_stage_stays_in_range_under_training() {
        // The realistic configuration: a 2-stage mixer (S>1) with the
        // production init weight, driven with a consistent bit-1
        // signal. The mixed output must stay a valid probability and
        // the training loop must not panic.
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        let mut m = Mixer::new(64, 4, 32);
        for _ in 0..400 {
            m.update(1);
            for _ in 0..64 { m.add(1024); }
            for set in 0..4 { m.set(set as u32, 16); }
            let p = m.p(1, &sq, &st);
            assert!(p >= 0 && p <= 4095, "mixer p out of range: {}", p);
        }
    }
}
