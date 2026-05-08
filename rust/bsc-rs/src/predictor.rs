//! Probability counter and 3-input logistic mixer, ported from
//! `plugins/bsc/upstream/libbsc/coder/common/predictor.h`.
//!
//! The C "ProbabilityCounter" is a free-functions namespace; we mirror
//! that with module-level helpers operating on a `&mut i16` (the
//! probability slot, in 12-bit fixed-point).
//!
//! `ProbabilityMixer` is a stateful 3-input mixer + 17-entry adaptive
//! map. The math here is identical to libbsc's; integer types are
//! widened to `i32` where the C code relies on implicit promotion of
//! `short`. Wrapping arithmetic is used to match C's signed-overflow
//! "undefined behaviour" that libbsc happens to depend on (the values
//! stay bounded by construction, so wrapping never triggers in
//! practice — but using `wrapping_*` makes the port crash-free under
//! debug overflow checks).
//!
//! The probability scale is 12 bits (range 0..=4096); stretched
//! probabilities are 12 bits signed (range -2047..=2047).

#![allow(dead_code)]

use crate::coder_tables::{squash, stretch};

// ---------------------------------------------------------------------
// ProbabilityCounter — bit-level updaters.
// ---------------------------------------------------------------------

/// `ProbabilityCounter::UpdateBit(bit, probability, t0, ar0, t1, ar1)`.
#[inline]
pub fn update_bit(
    bit: u32,
    probability: &mut i16,
    threshold0: i32,
    adaptation_rate0: i32,
    threshold1: i32,
    adaptation_rate1: i32,
) {
    let p = *probability as i32;
    let delta0 = p.wrapping_mul(adaptation_rate0)
        .wrapping_sub((4096 - threshold0).wrapping_mul(adaptation_rate0).wrapping_sub(4095));
    let delta1 = p.wrapping_mul(adaptation_rate1)
        .wrapping_sub(threshold1.wrapping_mul(adaptation_rate1));
    let delta = if bit != 0 { delta1 } else { delta0 };
    *probability = (p.wrapping_sub(delta >> 12)) as i16;
}

/// `ProbabilityCounter::UpdateBit0(probability, threshold, adaptation_rate)`.
#[inline]
pub fn update_bit_0(probability: &mut i16, threshold: i32, adaptation_rate: i32) {
    let p = *probability as i32;
    let new = p.wrapping_add(((4096 - threshold - p).wrapping_mul(adaptation_rate)) >> 12);
    *probability = new as i16;
}

/// `ProbabilityCounter::UpdateBit1(probability, threshold, adaptation_rate)`.
#[inline]
pub fn update_bit_1(probability: &mut i16, threshold: i32, adaptation_rate: i32) {
    let p = *probability as i32;
    let new = p.wrapping_sub(((p - threshold).wrapping_mul(adaptation_rate)) >> 12);
    *probability = new as i16;
}

/// `template <int R> ProbabilityCounter::UpdateBit(bit, probability, t0, t1)`.
#[inline]
pub fn update_bit_r(bit: u32, probability: &mut i16, threshold0: i32, threshold1: i32, r: u32) {
    let p = *probability as i32;
    let t = if bit != 0 { threshold1 } else { threshold0 };
    let new = p.wrapping_sub((p - t) >> r);
    *probability = new as i16;
}

/// `template <int R> ProbabilityCounter::UpdateBit(probability, threshold)`.
#[inline]
pub fn update_bit_simple_r(probability: &mut i16, threshold: i32, r: u32) {
    let p = *probability as i32;
    let new = p.wrapping_sub((p - threshold) >> r);
    *probability = new as i16;
}

// ---------------------------------------------------------------------
// ProbabilityMixer — 3-input logistic mixer with 17-entry map.
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ProbabilityMixer {
    stretched_probability0: i16,
    stretched_probability1: i16,
    stretched_probability2: i16,
    mixed_probability: i32,
    index: i32,

    probability_map: [i16; 17],

    weight0: i32,
    weight1: i32,
    weight2: i32,
}

impl ProbabilityMixer {
    pub fn new() -> Self {
        let mut me = Self {
            stretched_probability0: 0,
            stretched_probability1: 0,
            stretched_probability2: 0,
            mixed_probability: 0,
            index: 0,
            probability_map: [0; 17],
            weight0: 2048 << 5,
            weight1: 2048 << 5,
            weight2: 0,
        };
        for p in 0..17i32 {
            me.probability_map[p as usize] = squash((p - 8) * 256) as i16;
        }
        me
    }

    /// Reset to construction state.
    pub fn init(&mut self) {
        *self = Self::new();
    }

    /// `Mixup(p0, p1, p2)` — returns a new mixed probability in
    /// the 12-bit scale (0..=4095).
    pub fn mixup(&mut self, p0: i32, p1: i32, p2: i32) -> i32 {
        self.stretched_probability0 = stretch(p0) as i16;
        self.stretched_probability1 = stretch(p1) as i16;
        self.stretched_probability2 = stretch(p2) as i16;

        let mut stretched = (
            (self.stretched_probability0 as i32) * self.weight0
            + (self.stretched_probability1 as i32) * self.weight1
            + (self.stretched_probability2 as i32) * self.weight2
        ) >> 17;

        // libbsc casts to short here; reproduce the wrap semantics.
        let stretched_s = stretched as i16 as i32;
        stretched = stretched_s.clamp(-2047, 2047);

        self.index = (stretched + 2048) >> 8;
        let weight = stretched & 255;
        let probability = squash(stretched);
        let m_lo = self.probability_map[self.index as usize] as i32;
        let m_hi = self.probability_map[(self.index + 1) as usize] as i32;
        let mapped = m_lo + (((m_hi - m_lo) * weight) >> 8);
        self.mixed_probability = (3 * probability + mapped) >> 2;
        self.mixed_probability
    }

    /// `MixupAndUpdateBit0(...)`.
    pub fn mixup_and_update_bit_0(
        &mut self,
        p0: i32, p1: i32, p2: i32,
        learning_rate0: i32, learning_rate1: i32, learning_rate2: i32,
        threshold: i32, adaptation_rate: i32,
    ) -> i32 {
        let s0 = stretch(p0) as i16;
        let s1 = stretch(p1) as i16;
        let s2 = stretch(p2) as i16;

        let mut stretched = (
            (s0 as i32) * self.weight0
            + (s1 as i32) * self.weight1
            + (s2 as i32) * self.weight2
        ) >> 17;
        let stretched_s = stretched as i16 as i32;
        stretched = stretched_s.clamp(-2047, 2047);

        let weight = stretched & 255;
        let index = (stretched + 2048) >> 8;
        let probability = squash(stretched);
        let m_lo = self.probability_map[index as usize] as i32;
        let m_hi = self.probability_map[(index + 1) as usize] as i32;
        let mapped = m_lo + (((m_hi - m_lo) * weight) >> 8);
        let mixed = (3 * probability + mapped) >> 2;

        update_bit_0(&mut self.probability_map[index as usize], threshold, adaptation_rate);
        update_bit_0(&mut self.probability_map[(index + 1) as usize], threshold, adaptation_rate);

        let eps = mixed - 4095;
        self.weight0 -= (learning_rate0 * eps * (s0 as i32)) >> 16;
        self.weight1 -= (learning_rate1 * eps * (s1 as i32)) >> 16;
        self.weight2 -= (learning_rate2 * eps * (s2 as i32)) >> 16;

        mixed
    }

    /// `MixupAndUpdateBit1(...)`.
    pub fn mixup_and_update_bit_1(
        &mut self,
        p0: i32, p1: i32, p2: i32,
        learning_rate0: i32, learning_rate1: i32, learning_rate2: i32,
        threshold: i32, adaptation_rate: i32,
    ) -> i32 {
        let s0 = stretch(p0) as i16;
        let s1 = stretch(p1) as i16;
        let s2 = stretch(p2) as i16;

        let mut stretched = (
            (s0 as i32) * self.weight0
            + (s1 as i32) * self.weight1
            + (s2 as i32) * self.weight2
        ) >> 17;
        let stretched_s = stretched as i16 as i32;
        stretched = stretched_s.clamp(-2047, 2047);

        let weight = stretched & 255;
        let index = (stretched + 2048) >> 8;
        let probability = squash(stretched);
        let m_lo = self.probability_map[index as usize] as i32;
        let m_hi = self.probability_map[(index + 1) as usize] as i32;
        let mapped = m_lo + (((m_hi - m_lo) * weight) >> 8);
        let mixed = (3 * probability + mapped) >> 2;

        update_bit_1(&mut self.probability_map[index as usize], threshold, adaptation_rate);
        update_bit_1(&mut self.probability_map[(index + 1) as usize], threshold, adaptation_rate);

        let eps = mixed - 1;
        self.weight0 -= (learning_rate0 * eps * (s0 as i32)) >> 16;
        self.weight1 -= (learning_rate1 * eps * (s1 as i32)) >> 16;
        self.weight2 -= (learning_rate2 * eps * (s2 as i32)) >> 16;

        mixed
    }

    /// `UpdateBit0(...)` — re-uses cached stretched probabilities and index.
    pub fn update_bit_0(
        &mut self,
        learning_rate0: i32, learning_rate1: i32, learning_rate2: i32,
        threshold: i32, adaptation_rate: i32,
    ) {
        update_bit_0(&mut self.probability_map[self.index as usize], threshold, adaptation_rate);
        update_bit_0(&mut self.probability_map[(self.index + 1) as usize], threshold, adaptation_rate);
        let eps = self.mixed_probability - 4095;
        self.weight0 -= (learning_rate0 * eps * (self.stretched_probability0 as i32)) >> 16;
        self.weight1 -= (learning_rate1 * eps * (self.stretched_probability1 as i32)) >> 16;
        self.weight2 -= (learning_rate2 * eps * (self.stretched_probability2 as i32)) >> 16;
    }

    /// `UpdateBit1(...)`.
    pub fn update_bit_1(
        &mut self,
        learning_rate0: i32, learning_rate1: i32, learning_rate2: i32,
        threshold: i32, adaptation_rate: i32,
    ) {
        update_bit_1(&mut self.probability_map[self.index as usize], threshold, adaptation_rate);
        update_bit_1(&mut self.probability_map[(self.index + 1) as usize], threshold, adaptation_rate);
        let eps = self.mixed_probability - 1;
        self.weight0 -= (learning_rate0 * eps * (self.stretched_probability0 as i32)) >> 16;
        self.weight1 -= (learning_rate1 * eps * (self.stretched_probability1 as i32)) >> 16;
        self.weight2 -= (learning_rate2 * eps * (self.stretched_probability2 as i32)) >> 16;
    }
}

impl Default for ProbabilityMixer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_bit_0_pulls_probability_up() {
        let mut p: i16 = 2000;
        update_bit_0(&mut p, 0, 1024);
        // (4096 - 0 - 2000) * 1024 >> 12 = 2096 * 1024 >> 12 = 524 ≈
        // expectation: probability rose toward 4096.
        assert!(p as i32 > 2000);
    }

    #[test]
    fn update_bit_1_pulls_probability_down() {
        let mut p: i16 = 2000;
        update_bit_1(&mut p, 0, 1024);
        // (2000 - 0) * 1024 >> 12 = 500 → p drops by 500.
        assert!((p as i32) < 2000);
    }

    #[test]
    fn mixer_init_returns_neutral_50_percent() {
        let mut m = ProbabilityMixer::new();
        // squash(0) = 2048 (50% in 12-bit scale)
        let p = m.mixup(2048, 2048, 2048);
        // Mix of three 50% probabilities ought to land near 50%.
        assert!((p - 2048).abs() < 200, "mixed = {}", p);
    }

    #[test]
    fn mixer_responds_to_consistent_signal() {
        let mut m = ProbabilityMixer::new();
        // Drive 300 bit-0 events at strong "expect 0" inputs and check
        // the mixer narrows toward 0 (high probability of bit=0 ⇒
        // mixed_probability close to 4095).
        for _ in 0..300 {
            m.mixup_and_update_bit_0(3500, 3500, 2048, 16, 16, 16, 64, 1024);
        }
        let p = m.mixup(3500, 3500, 2048);
        assert!(p > 3000, "expected mixer to learn bit-0 bias, got {}", p);
    }
}
