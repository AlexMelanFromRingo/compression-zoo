//! `predictor.cpp` — top-level Predictor orchestrator.
//!
//! Upstream `Predictor` adds ~100 sub-models through a `ContextManager`
//! and combines them through a 3-layer mixer tree + SSE + an LSTM
//! `ByteMixer`. This port wires the two fully-ported bit-level model
//! banks — `fxcmv1` (Kaido Orav's PAQ8-style mixer) and the complete
//! `paq8` sub-model bank (`contextModel2` + `Paq8Predictor`) — through
//! an adaptive 2-input logistic mixer (a `paq8::Mixer`), the same
//! mixing primitive upstream uses at every layer.
//!
//! Mathematical correctness of the encode→decode round-trip holds for
//! any predictor; combining the two model banks materially improves
//! the compression ratio over either alone. The remaining gap to
//! upstream cmix is the full `ContextManager` model bank + the LSTM
//! ByteMixer + the layered mixer tree.

#![forbid(unsafe_code)]

use crate::coder::Predictor as CoderPredictor;
use crate::models::fxcmv1;
use crate::models::paq8::mixer::Mixer;
use crate::models::paq8::substrate::{Squash, Stretch};
use crate::models::paq8::Paq8;

/// Production memory level for the paq8 sub-predictor (mirrors
/// upstream's `PAQ8(11)`).
const PAQ8_LEVEL: u32 = 11;

pub struct Predictor {
    fxcm: fxcmv1::Predictor,
    paq:  Paq8,
    /// 2-input single-stage adaptive logistic mixer (fxcm + paq8).
    mix:  Mixer,
    squash:  Squash,
    stretch: Stretch,
    /// Cached mixed bit-1 probability (0..=4095).
    pr: i32,
}

impl Predictor {
    pub fn new() -> Self {
        Self::with_paq8_level(PAQ8_LEVEL)
    }

    /// Construct with an explicit paq8 memory level — tests use a low
    /// level to keep allocations small.
    pub fn with_paq8_level(level: u32) -> Self {
        let squash  = Squash::new();
        let stretch = Stretch::new(&squash);
        Self {
            fxcm: fxcmv1::Predictor::new(),
            paq:  Paq8::new(level),
            // 2 inputs, 1 mixer set (single-stage), default init weight.
            mix:  Mixer::new(2, 1, 32),
            squash,
            stretch,
            pr: 2048,
        }
    }

    /// Construct with an optional WRT-style dictionary attached to the
    /// fxcmv1 sub-predictor.
    pub fn with_dictionary(data: &[u8]) -> Self {
        let mut p = Self::with_paq8_level(PAQ8_LEVEL);
        p.fxcm = fxcmv1::Predictor::new().with_dictionary_bytes(data);
        p
    }
}

impl Default for Predictor { fn default() -> Self { Self::new() } }

impl CoderPredictor for Predictor {
    fn predict(&mut self) -> f32 {
        // Each sub-model's current bit-1 probability, stretched and
        // fed into the adaptive mixer. The mixer's weights were
        // trained on the last observed bit in `perceive`.
        let fx = ((self.fxcm.predict() * 4095.0) as i32).clamp(0, 4095);
        let p8 = ((self.paq.predict_bit() * 4095.0) as i32).clamp(0, 4095);
        self.mix.add(self.stretch.get(fx) as i16);
        self.mix.add(self.stretch.get(p8) as i16);
        self.mix.set(0, 1);
        // Single-stage mixer: `p()` ignores its `y` argument.
        self.pr = self.mix.p(0, &self.squash, &self.stretch);
        (self.pr as f32 + 0.5) / 4096.0
    }

    fn perceive(&mut self, bit: i32) {
        // Train the top-level mixer on the bit just coded, then
        // advance both sub-model banks.
        self.mix.update(bit);
        self.fxcm.perceive(bit);
        self.paq.perceive_bit(bit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "fxcmv1 + paq8(11) allocate several GB; heavy test"]
    fn predictor_runs_a_byte_at_production_level() {
        let mut p = Predictor::new();
        for bit in [1, 0, 1, 1, 0, 0, 1, 0] {
            let pr = p.predict();
            assert!(pr >= 0.0 && pr <= 1.0);
            p.perceive(bit);
        }
    }

    #[test]
    #[ignore = "fxcmv1 still allocates GB-scale buckets; heavy test"]
    fn predictor_small_level_round_trip_shape() {
        let mut p = Predictor::with_paq8_level(0);
        for &byte in b"top-level orchestrator" {
            for bp in (0..8).rev() {
                let pr = p.predict();
                assert!(pr >= 0.0 && pr <= 1.0);
                p.perceive(((byte >> bp) & 1) as i32);
            }
        }
    }
}
