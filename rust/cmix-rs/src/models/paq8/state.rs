//! `paq8::Paq8State` — file-scope mutable globals from upstream
//! paq8.cpp packaged as a struct so each sub-model takes
//! `&mut Paq8State` instead of touching dozens of separate
//! `static`s.
//!
//! Mirrors:
//!
//! * `int level`, `int y`, `int c0`, `U32 c4`, `int bpos`,
//!   `int blpos`, `Buf buf`, `U8 grp0`, `int dt[1024]`.
//! * `U32 b2, b3, w4, w5, f4, x4, x5, tt, words`.
//! * `paq8::Predictor::stats` (ModelStats).
//! * `model_predictions[]` array + `prediction_index`.

#![allow(dead_code)]

use super::substrate::{Buf, build_dt, Ilog, Squash, Stretch, NUM_INPUTS, NUM_SETS};
use super::stats::ModelStats;

pub const PREDICTIONS_LEN: usize = NUM_INPUTS + NUM_SETS + 11;

pub struct Paq8State {
    pub level: u32,
    pub y:     i32,
    pub c0:    u32,
    pub c4:    u32,
    pub bpos:  i32,
    pub blpos: i32,
    pub buf:   Buf,
    pub grp0:  u8,
    pub dt:    [i32; 1024],

    /// Recent bytes / word state — upstream's `b2`, `b3`, `w4`, `w5`,
    /// `f4`, `x4`, `x5`, `tt`, `words`.
    pub b2: u32, pub b3: u32,
    pub w4: u32, pub w5: u32,
    pub f4: u32, pub x4: u32, pub x5: u32,
    pub tt: u32, pub words: u32,

    /// Text-layout globals — upstream's `frstchar`, `spafdo`,
    /// `spaces`, `col`, `spacecount`, `wordcount`, `wordlen`,
    /// `wordlen1` (paq8.cpp:3871-3873).
    pub frstchar:   u32,
    pub spafdo:     u32,
    pub spaces:     u32,
    pub col:        u32,
    pub spacecount: u32,
    pub wordcount:  u32,
    pub wordlen:    u32,
    pub wordlen1:   u32,

    pub stats: ModelStats,

    /// Cached pre-computed lookup tables.
    pub ilog:    Ilog,
    pub squash:  Squash,
    pub stretch: Stretch,

    /// Per-bit prediction outputs (`AddPrediction` / `ResetPredictions`).
    pub model_predictions: Vec<f32>,
    pub prediction_index:  usize,

    /// Cached last_prediction (upstream global).
    pub last_prediction: i32,
}

impl Paq8State {
    pub fn new(level: u32) -> Self {
        let squash  = Squash::new();
        let stretch = Stretch::new(&squash);
        let mut buf = Buf::new();
        // Upstream uses `paq8::buf.setsize(paq8::MEM()*8)` (PAQ8::new).
        let buf_size = (super::substrate::mem(level) as usize)
            .saturating_mul(8)
            .next_power_of_two();
        buf.set_size(buf_size);
        Self {
            level,
            y: 0,
            c0: 1,
            c4: 0,
            bpos: 0,
            blpos: 0,
            buf,
            grp0: 0,
            dt: build_dt(),
            b2: 0, b3: 0,
            w4: 0, w5: 0,
            f4: 0, x4: 0, x5: 0,
            tt: 0, words: 0,
            frstchar: 0, spafdo: 0, spaces: 0, col: 0,
            spacecount: 0, wordcount: 0, wordlen: 0, wordlen1: 0,
            stats: ModelStats::new(),
            ilog: Ilog::new(),
            squash, stretch,
            model_predictions: vec![0.5; PREDICTIONS_LEN],
            prediction_index:  0,
            last_prediction:   2048,
        }
    }

    #[inline]
    pub fn add_prediction(&mut self, x: i32) {
        if self.prediction_index < self.model_predictions.len() {
            self.model_predictions[self.prediction_index] =
                x as f32 * (1.0 / 4095.0);
            self.prediction_index += 1;
        }
    }

    #[inline]
    pub fn reset_predictions(&mut self) { self.prediction_index = 0; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paq8_state_initializes_with_uniform_predictions_small_level() {
        // level=0 → buf = 0x10000 * 8 = 512 KiB. Cheap for unit
        // tests; production uses level=11.
        let s = Paq8State::new(0);
        assert_eq!(s.model_predictions.len(), PREDICTIONS_LEN);
        assert_eq!(s.c0, 1);
        assert_eq!(s.last_prediction, 2048);
        for &p in &s.model_predictions { assert!((p - 0.5).abs() < 1e-6); }
    }

    #[test]
    fn paq8_state_buf_is_sized_power_of_two_small_level() {
        let s = Paq8State::new(0);
        let n = s.buf.size();
        assert!(n.is_power_of_two() || n == 0);
        assert!(n >= (1usize << 19)); // 512 KiB minimum
    }

    #[test]
    #[ignore = "allocates several GB; run with --ignored --test-threads=1"]
    fn paq8_state_initializes_at_production_level() {
        let s = Paq8State::new(11);
        assert!(s.buf.size() >= (1usize << 30)); // ≥ 1 GiB
    }
}
