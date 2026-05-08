//! Online context mixer — port of `mixer/{mixer,mixer-input}.{h,cpp}`.
//!
//! Submodules:
//!   * [`lstm_layer`] — single LSTM layer with Adam + BPTT.
//!
//! The full LSTM stack and LSTM-driven byte mixer are TODO; see
//! HANDOFF.
//!
//! [`MixerInput`] gathers per-model bit-1 probabilities (one per
//! sub-model) plus a small set of "extra" stretched inputs, all
//! mapped through the [`crate::sigmoid::Sigmoid`] logit table so
//! the mixer can sum them in logit space.
//!
//! [`Mixer`] applies a per-context weight vector (and a separate
//! extra-input weight vector), stored in a hashmap keyed by the
//! caller-supplied context. `mix` returns the weighted sum (still
//! in logit space — caller logistics it for use as a probability).
//! `perceive(bit)` SGD-updates the weights with a step size that
//! decays both globally (with steps so far) and per-context
//! (proportional to that context's hit count).

#![allow(dead_code)]

pub mod byte_mixer;
pub mod lstm;
pub mod lstm_layer;
pub mod sse;

use std::collections::HashMap;

use crate::sigmoid::Sigmoid;

/// Per-context weight vector + step counter.
#[derive(Clone)]
pub struct ContextData {
    pub steps: u64,
    pub weights: Vec<f32>,
    pub extra_weights: Vec<f32>,
}

impl ContextData {
    pub fn new(input_size: usize, extra_input_size: usize) -> Self {
        Self {
            steps: 0,
            weights: vec![0.0; input_size],
            extra_weights: vec![0.0; extra_input_size],
        }
    }
}

/// Stretched-logit inputs for the mixer.
pub struct MixerInput<'s> {
    inputs: Vec<f32>,
    extra_inputs: Vec<f32>,
    sigmoid: &'s Sigmoid,
    min: f32,
    max: f32,
    stretched_min: f32,
    stretched_max: f32,
}

impl<'s> MixerInput<'s> {
    pub fn new(sigmoid: &'s Sigmoid, eps: f32) -> Self {
        Self {
            inputs: vec![0.5],
            extra_inputs: Vec::new(),
            min: eps,
            max: 1.0 - eps,
            stretched_min: sigmoid.logit(0.0),
            stretched_max: sigmoid.logit(1.0),
            sigmoid,
        }
    }

    pub fn set_num_models(&mut self, n: usize) {
        self.inputs.resize(n, 0.5);
    }

    /// Stretch `p` (probability in `[min, max]` after clamp) and
    /// store at slot `index`.
    pub fn set_input(&mut self, index: usize, p: f32) {
        let p = p.clamp(self.min, self.max);
        self.inputs[index] = self.sigmoid.logit(p);
    }

    /// Like `set_input` but `p` is already a stretched logit.
    pub fn set_stretched_input(&mut self, index: usize, p: f32) {
        let p = p.clamp(self.stretched_min, self.stretched_max);
        self.inputs[index] = p;
    }

    pub fn set_extra_input(&mut self, p: f32) {
        let p = p.clamp(self.stretched_min, self.stretched_max);
        self.extra_inputs.push(p);
    }

    pub fn clear_extra_inputs(&mut self) { self.extra_inputs.clear(); }

    pub fn inputs(&self) -> &[f32] { &self.inputs }
    pub fn extra_inputs(&self) -> &[f32] { &self.extra_inputs }
}

/// Online weight-mixing predictor.
///
/// Holds per-context weight vectors in a `HashMap<u64, ContextData>`,
/// matching upstream's `unordered_map<unsigned int, ...>` shape (we
/// widen to `u64` because Rust's hashable u64 has the same domain).
pub struct Mixer {
    p: f32,
    learning_rate: f32,
    max_steps: u64,
    steps: u64,
    context_map: HashMap<u64, ContextData>,
    /// Cached input/extra sizes used to allocate fresh `ContextData`.
    input_size: usize,
    extra_input_size: usize,
}

impl Mixer {
    pub fn new(input_size: usize, extra_input_size: usize, learning_rate: f32) -> Self {
        Self {
            p: 0.5,
            learning_rate,
            max_steps: 1,
            steps: 0,
            context_map: HashMap::new(),
            input_size,
            extra_input_size,
        }
    }

    /// Read-only view of the cached `Mix()` output (a logit). Set by
    /// the most recent `mix(...)` call.
    pub fn last_logit(&self) -> f32 { self.p }

    /// Look up (or insert) the weight set for `context`. Mirrors
    /// upstream's GetContextData with its 10,000-context cap and
    /// `0xDEADBEEF` overflow bucket.
    fn ensure_context(&mut self, context: u64) -> &mut ContextData {
        const LIMIT: usize = 10_000;
        const OVERFLOW: u64 = 0xDEADBEEF;
        let key = if self.context_map.len() >= LIMIT
                  && !self.context_map.contains_key(&context) {
            OVERFLOW
        } else {
            context
        };
        self.context_map.entry(key).or_insert_with(||
            ContextData::new(self.input_size, self.extra_input_size))
    }

    /// Compute `p = Σ inputs[i] * weights[i] + Σ extra[i] * extra_w[i]`
    /// and cache it for the next `perceive`. Returns the raw logit
    /// — caller applies `Sigmoid::logistic` if a probability is wanted.
    pub fn mix(&mut self, inputs: &[f32], extra_inputs: &[f32], context: u64) -> f32 {
        let data = self.ensure_context(context);
        let mut p = 0.0f32;
        for i in 0..inputs.len() {
            p += inputs[i] * data.weights[i];
        }
        let mut e = 0.0f32;
        for i in 0..extra_inputs.len().min(data.extra_weights.len()) {
            e += extra_inputs[i] * data.extra_weights[i];
        }
        self.p = p + e;
        self.p
    }

    /// Update the active context's weights via the same online SGD
    /// rule as upstream.
    pub fn perceive(&mut self, bit: i32, inputs: &[f32], extra_inputs: &[f32], context: u64) {
        // Snapshot the bits we'll need before borrowing the context.
        let p = self.p;
        let logistic = Sigmoid::logistic(p);
        let decay_global = 0.9 / (0.0000001 * self.steps as f32 + 0.8).powf(0.8);
        let max_steps_snapshot = self.max_steps;
        let learning_rate = self.learning_rate;

        let data = self.ensure_context(context);
        let decay_local = 1.5 - (data.steps as f32 / max_steps_snapshot as f32);
        let update = decay_global * decay_local * learning_rate
            * (logistic - bit as f32);

        for i in 0..inputs.len() {
            data.weights[i] -= update * inputs[i];
        }
        for i in 0..extra_inputs.len().min(data.extra_weights.len()) {
            data.extra_weights[i] -= update * extra_inputs[i];
        }
        // Periodic L2-style decay (every 1024 steps).
        if (data.steps & 1023) == 0 {
            for w in data.weights.iter_mut()       { *w *= 1.0 - 3.0e-6; }
            for w in data.extra_weights.iter_mut() { *w *= 1.0 - 3.0e-6; }
        }

        data.steps += 1;
        let new_steps = data.steps;
        if new_steps > self.max_steps {
            self.max_steps = new_steps;
        }
        self.steps += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_input_clamping() {
        let s = Sigmoid::new(4096);
        let mut mi = MixerInput::new(&s, 0.001);
        mi.set_num_models(3);
        mi.set_input(0, 0.5);
        // Below `min_` clamps to `min_` then logit; just verify no panic.
        mi.set_input(1, 0.0);
        mi.set_input(2, 1.0);
        let xs = mi.inputs();
        assert_eq!(xs.len(), 3);
        // logit(eps) is large negative; logit(1-eps) is large positive.
        assert!(xs[1] < 0.0);
        assert!(xs[2] > 0.0);
    }

    /// Mixer over a single repeated context with always-bit-1 input
    /// should drive the weight up over many `perceive` calls so the
    /// next `mix` is more positive than the first.
    #[test]
    fn mixer_learns_a_constant_signal() {
        let mut m = Mixer::new(1, 0, 0.005);
        let inputs = vec![1.0f32];
        let extra: Vec<f32> = Vec::new();
        let p0 = m.mix(&inputs, &extra, 42);
        for _ in 0..200 {
            m.mix(&inputs, &extra, 42);
            m.perceive(1, &inputs, &extra, 42);
        }
        let p_after = m.mix(&inputs, &extra, 42);
        // Initial weights are 0 so p0 == 0; after training on bit=1
        // the logit should drift positive.
        assert!(p0 == 0.0);
        assert!(p_after > 0.0,
            "p_after = {} (should be > 0 after training)", p_after);
    }
}
