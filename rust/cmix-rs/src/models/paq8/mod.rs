//! `models/paq8.{h,cpp}` — port of Matt Mahoney's PAQ8 sub-model bank
//! (`paq8::Predictor` + ~25 contextual sub-models + Mixer/APM stack).
//!
//! Sub-modules, in upstream dependency order:
//!
//! * [`substrate`]    — Random, Buf, Ilog, Squash/Stretch, dt[],
//!                      STATE_TABLE, hash helpers, dot_product/train.
//! * [`mixer`]        — two-stage logistic Mixer.
//! * [`apm`]          — StateMap / StateMap32 / APM1 / APM.
//! * [`context_map`]  — HashTableB / Bh / RunContextMap /
//!                      SmallStationaryContextMap / StationaryMap /
//!                      IndirectMap / ContextMap / ContextMap2.
//! * [`stats`]        — `ModelStats` shared per-byte state.
//! * [`state`]        — `Paq8State` (file-scope mutable globals).
//! * [`word`]         — `Word` + language flags + Segment/Sentence.
//! * [`stemmer`]      — English / French / German stemmers.
//! * [`util`]         — OLS, IndirectContext, MtfList, Cache.
//! * [`match_model`]  — MatchModel.
//! * [`sparse_match_model`] — SparseMatchModel.
//! * [`text_model`]   — TextModel.
//! * [`small_models`] — pic / distance / sparse / nest / indirect /
//!                      record / word / linear-prediction models.
//! * [`dmc`]          — dmcModel + dmcForest.
//! * [`xml_model`]    — XMLModel.
//! * [`exe_model`]    — x86/x64 instruction model.
//! * [`file_models`]  — image / jpeg / audio models.
//! * [`contextmodel`] — `contextModel2` + `Paq8Predictor` integration.

#![forbid(unsafe_code)]
#![allow(dead_code)]

pub mod substrate;
pub mod mixer;
pub mod apm;
pub mod context_map;
pub mod stats;
pub mod state;
pub mod word;
pub mod stemmer;
pub mod util;
pub mod match_model;
pub mod sparse_match_model;
pub mod text_model;
pub mod small_models;
pub mod dmc;
pub mod xml_model;
pub mod exe_model;
pub mod file_models;
pub mod contextmodel;

use super::Model;
use contextmodel::Paq8Predictor;

/// Top-level paq8 model — wraps the full [`Paq8Predictor`] pipeline.
pub struct Paq8 {
    predictor: Paq8Predictor,
    /// Single-element `outputs` slice carrying the latest bit-1
    /// probability (for the `Model` trait).
    output: [f32; 1],
}

impl Paq8 {
    /// `memory_level` mirrors upstream's `PAQ8(memory)` argument
    /// (production cmix uses `11`; tests use `0`).
    pub fn new(memory_level: u32) -> Self {
        Self {
            predictor: Paq8Predictor::new(memory_level),
            output: [0.5],
        }
    }

    /// Latest bit-1 probability in `[0, 1]`.
    pub fn predict_bit(&self) -> f32 { self.predictor.predict() }

    /// Advance the model by one observed bit.
    pub fn perceive_bit(&mut self, bit: i32) {
        self.predictor.update(bit);
    }
}

impl Default for Paq8 { fn default() -> Self { Self::new(11) } }

impl Model for Paq8 {
    fn outputs(&self) -> &[f32] { &self.output }
    fn perceive(&mut self, bit: i32) {
        self.predictor.update(bit);
        self.output[0] = self.predictor.predict();
    }
    fn byte_update(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paq8_drives_a_byte_through_the_full_pipeline_small_level() {
        // level 0 keeps the allocation footprint test-friendly.
        let mut p = Paq8::new(0);
        for &byte in b"paq8 integration" {
            for bp in (0..8).rev() {
                p.perceive(((byte >> bp) & 1) as i32);
                let pr = p.predict_bit();
                assert!(pr >= 0.0 && pr <= 1.0, "predict out of range: {}", pr);
            }
        }
        assert_eq!(p.outputs().len(), 1);
    }
}
