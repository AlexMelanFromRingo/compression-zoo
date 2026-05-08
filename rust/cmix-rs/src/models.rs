//! Per-context probability models — port of `models/*.{h,cpp}`.
//!
//! What's included this session: the four small models used by
//! every CMIX configuration as the foundation of the per-byte
//! probability mix:
//!
//!   * [`Direct`]      — `(byte_context, bit_context) → probability`,
//!     with an adaptive divisor that ramps up toward `limit + delta`.
//!   * [`DirectHash`]  — same shape, but `byte_context` is hashed
//!     into `size` slots with 20-probe linear-probing checksum
//!     collision detection.
//!   * [`Indirect`]    — a state-machine probability where the
//!     state index lives in a caller-owned byte map. Each
//!     `perceive(bit)` walks the state via a [`crate::state::State`]
//!     transition.
//!   * [`Match`]       — longest-match predictor backed by a sliding
//!     history buffer.
//!
//! Bigger models (PAQ8, FXCMv1, PPMd, ByteModel/Bracket with its
//! vocabulary stack) are deferred to follow-up sessions and tracked
//! in `HANDOFF.md`.

#![allow(dead_code)]

use crate::state::State;

// ---------------- Model trait -----------------------------------------

/// Probability-output model. The default emits a constant 0.5 so
/// callers can compose stub instances. `predict` returns one or
/// more bit-1 probabilities; `perceive(bit)` updates internal state
/// after each bit; `byte_update()` is invoked once per byte boundary.
pub trait Model {
    /// Read the model's current bit-1 probability output(s).
    fn outputs(&self) -> &[f32];
    /// Update internal state after observing `bit`.
    fn perceive(&mut self, _bit: i32) {}
    /// End-of-byte hook.
    fn byte_update(&mut self) {}
}

// ---------------- Direct ----------------------------------------------

#[derive(Clone)]
pub struct Direct {
    output: [f32; 1],
    limit: i32,
    delta: f32,
    divisor: f32,
    predictions: Vec<[f32; 256]>,
    counts: Vec<[u8; 256]>,
}

impl Direct {
    pub fn new(limit: i32, delta: f32, size: usize) -> Self {
        Self {
            output: [0.5],
            limit,
            delta,
            divisor: 1.0 / (limit as f32 + delta),
            predictions: vec![[0.5; 256]; size],
            counts: vec![[0u8; 256]; size],
        }
    }
    /// Reads the prediction for the given `(byte_context, bit_context)`.
    pub fn predict(&mut self, byte_context: usize, bit_context: usize) -> f32 {
        let p = self.predictions[byte_context][bit_context];
        self.output[0] = p;
        p
    }
    /// SGD-style update of the cell at `(byte_context, bit_context)`.
    pub fn perceive(&mut self, bit: i32, byte_context: usize, bit_context: usize) {
        let mut divisor = self.divisor;
        let cnt = &mut self.counts[byte_context][bit_context];
        if (*cnt as i32) < self.limit {
            *cnt += 1;
            divisor = 1.0 / (*cnt as f32 + self.delta);
        }
        let p = &mut self.predictions[byte_context][bit_context];
        *p += (bit as f32 - *p) * divisor;
    }
}
impl Model for Direct {
    fn outputs(&self) -> &[f32] { &self.output }
}

// ---------------- DirectHash ------------------------------------------

#[derive(Clone)]
pub struct DirectHash {
    output: [f32; 1],
    index: usize,
    limit: i32,
    delta: f32,
    divisor: f32,
    predictions: Vec<[f32; 256]>,
    counts: Vec<[u8; 256]>,
    checksums: Vec<u64>,
}

impl DirectHash {
    pub fn new(limit: i32, delta: f32, size: usize) -> Self {
        Self {
            output: [0.5],
            index: 0,
            limit,
            delta,
            divisor: 1.0 / (limit as f32 + delta),
            predictions: vec![[0.5; 256]; size],
            counts: vec![[0u8; 256]; size],
            checksums: vec![0u64; size],
        }
    }
    pub fn predict(&mut self, bit_context: usize) -> f32 {
        let p = self.predictions[self.index][bit_context];
        self.output[0] = p;
        p
    }
    pub fn perceive(&mut self, bit: i32, bit_context: usize) {
        let mut divisor = self.divisor;
        let cnt = &mut self.counts[self.index][bit_context];
        if (*cnt as i32) < self.limit {
            *cnt += 1;
            divisor = 1.0 / (*cnt as f32 + self.delta);
        }
        let p = &mut self.predictions[self.index][bit_context];
        *p += (bit as f32 - *p) * divisor;
    }
    /// Hashes `byte_context` into the slot table with 20-probe
    /// linear-probing checksum collision detection. After this call,
    /// `predict`/`perceive` work on the resolved slot.
    pub fn byte_update(&mut self, byte_context: u64) {
        let n = self.predictions.len();
        let mut idx = (byte_context as usize) % n;
        for i in 0..20 {
            if self.checksums[idx] == 0 {
                self.checksums[idx] = byte_context;
                break;
            }
            if self.checksums[idx] == byte_context { break; }
            if i == 19 {
                self.predictions[idx] = [0.5; 256];
                self.counts[idx] = [0u8; 256];
                self.checksums[idx] = byte_context;
                break;
            }
            idx = if idx + 1 == n { 0 } else { idx + 1 };
        }
        self.index = idx;
    }
}
impl Model for DirectHash {
    fn outputs(&self) -> &[f32] { &self.output }
}

// ---------------- Indirect --------------------------------------------

/// State-machine-driven probability. `map[map_index]` is the state
/// that tracks the bit history at the current `(byte, bit)` cell;
/// the `State` trait turns that into a bit-1 probability.
pub struct Indirect<S: State> {
    output: [f32; 1],
    map_index: usize,
    map_offset: usize,
    divisor: f32,
    state: S,
    predictions: [f32; 256],
    bit_context: usize,
}

impl<S: State> Indirect<S> {
    /// `map.len()` must be > 257. `map_offset` becomes
    /// `seed % (map.len() - 257)` (upstream uses libc rand() — we
    /// take an explicit seed for determinism).
    pub fn new(state: S, delta: f32, map_size: usize, seed: u64) -> Self {
        let mut predictions = [0.5f32; 256];
        for i in 0..256 {
            predictions[i] = state.init_probability(i as i32);
        }
        let map_offset = (seed as usize) % (map_size.saturating_sub(257).max(1));
        Self {
            output: [0.5],
            map_index: 0,
            map_offset,
            divisor: 1.0 / delta,
            state,
            predictions,
            bit_context: 0,
        }
    }

    /// Returns the bit-1 probability at the current map cell shifted
    /// by `bit_context`.
    pub fn predict(&mut self, map: &[u8], bit_context: usize) -> f32 {
        self.bit_context = bit_context;
        self.map_index = self.map_index.wrapping_add(bit_context);
        let s = map[self.map_index] as usize;
        self.output[0] = self.predictions[s];
        self.output[0]
    }

    /// SGD update + state transition. `map` is mutably borrowed —
    /// the caller owns the shared state-history map across all
    /// `Indirect` instances.
    pub fn perceive(&mut self, bit: i32, map: &mut [u8]) {
        let s = map[self.map_index] as usize;
        self.predictions[s] += (bit as f32 - self.predictions[s]) * self.divisor;
        let next = self.state.next(s as i32, bit) as u8;
        map[self.map_index] = next;
        self.map_index = self.map_index.wrapping_sub(self.bit_context);
    }

    pub fn byte_update(&mut self, byte_context: u64, map_len: usize) {
        let modulus = map_len.saturating_sub(257).max(1);
        self.map_index = ((257 * byte_context as usize) + self.map_offset) % modulus;
    }
}

// ---------------- ByteModel -------------------------------------------

/// Byte-level model — maintains a 256-entry probability vector per
/// byte and emits the bit-1 probability for the current binary
/// search position via `predict`. Used as the base class for
/// vocabulary-aware models like `Bracket` upstream; here we expose
/// the byte distribution directly for callers that need it.
///
/// Mirrors `models/byte-model.{h,cpp}`.
#[derive(Clone)]
pub struct ByteModel {
    output: [f32; 1],
    /// Highest possible byte value still in the active range.
    pub top: i32,
    /// Last computed midpoint of `[bot, top]`.
    pub mid: i32,
    /// Lowest possible byte value still in the active range.
    pub bot: i32,
    /// Vocabulary mask — `vocab[b] = false` zeros that entry on
    /// each byte boundary. Caller-owned so several `ByteModel`s
    /// can share one.
    pub probs: [f32; 256],
    /// Index of the most-probable byte from the last `predict`. Mirrors
    /// upstream's `int ex` field; useful for debugging/stats.
    pub ex: i32,
}

impl ByteModel {
    pub fn new() -> Self {
        Self {
            output: [0.5],
            top: 255,
            mid: 0,
            bot: 0,
            probs: [1.0 / 256.0; 256],
            ex: 0,
        }
    }

    pub fn byte_predict(&self) -> &[f32; 256] { &self.probs }

    /// Recompute the bit-1 probability based on the binary-search
    /// midpoint `mid = bot + (top - bot)/2`. Probability of bit = 1
    /// is `Σ probs[mid+1..top+1] / Σ probs[bot..top+1]`.
    pub fn predict(&mut self) -> f32 {
        let mid = self.bot + (self.top - self.bot) / 2;
        let mut num = 0.0f32;
        for i in (mid + 1)..=self.top { num += self.probs[i as usize]; }
        let mut denom = num;
        for i in self.bot..=mid { denom += self.probs[i as usize]; }
        // Track the most-probable byte for upstream parity.
        self.ex = self.bot;
        let mut max_p = self.probs[self.bot as usize];
        for i in (self.bot + 1)..=self.top {
            if self.probs[i as usize] > max_p {
                max_p = self.probs[i as usize];
                self.ex = i;
            }
        }
        self.output[0] = if denom == 0.0 { 0.5 } else { num / denom };
        self.output[0]
    }

    /// Narrow the `[bot, top]` range based on the just-decoded bit.
    pub fn perceive(&mut self, bit: i32) {
        self.mid = self.bot + (self.top - self.bot) / 2;
        if bit != 0 { self.bot = self.mid + 1; } else { self.top = self.mid; }
    }

    /// End-of-byte hook. Resets `[bot, top]` to `[0, 255]` and
    /// zeros every byte that's not in the vocabulary.
    pub fn byte_update(&mut self, vocab: &[bool; 256]) {
        self.top = 255;
        self.bot = 0;
        for i in 0..256 {
            if !vocab[i] { self.probs[i] = 0.0; }
        }
    }
}

impl Default for ByteModel { fn default() -> Self { Self::new() } }
impl Model for ByteModel {
    fn outputs(&self) -> &[f32] { &self.output }
}

// ---------------- Match -----------------------------------------------

/// Longest-match predictor against a sliding history buffer.
pub struct Match {
    output: [f32; 1],
    history_pos: u64,
    cur_match: u64,
    cur_byte: u8,
    bit_pos: u8,
    match_length: u8,
    pub longest_match: u64,
    limit: i32,
    delta: f32,
    divisor: f32,
    map: Vec<u32>,
    predictions: [f32; 256],
    counts: [i32; 256],
}

impl Match {
    pub fn new(limit: i32, delta: f32, map_size: usize) -> Self {
        let mut predictions = [0.0f32; 256];
        for i in 0..256 {
            predictions[i] = 0.5 + (i as f32 + 0.5) / 512.0;
        }
        Self {
            output: [0.5],
            history_pos: 0,
            cur_match: 0,
            cur_byte: 0,
            bit_pos: 128,
            match_length: 0,
            longest_match: 0,
            limit,
            delta,
            divisor: 1.0 / (limit as f32 + delta),
            map: vec![0u32; map_size],
            predictions,
            counts: [0i32; 256],
        }
    }

    pub fn predict(&mut self) -> f32 {
        let p = self.predictions[self.match_length as usize];
        self.output[0] = if (self.cur_byte & self.bit_pos) != 0 { p }
                         else { 1.0 - p };
        self.output[0]
    }

    pub fn perceive(&mut self, bit: i32, bit_context: u32, byte_context: u64) {
        let predicted = ((self.cur_byte & self.bit_pos) != 0) as i32;
        let m = if bit == predicted { 1 } else { 0 };
        self.bit_pos /= 2;

        let cnt = &mut self.counts[self.match_length as usize];
        let mut divisor = self.divisor;
        if *cnt < self.limit {
            *cnt += 1;
            divisor = 1.0 / (*cnt as f32 + self.delta);
        }
        let p = &mut self.predictions[self.match_length as usize];
        *p += (m as f32 - *p) * divisor;

        if m == 1 {
            if self.match_length < 255 { self.match_length += 1; }
        } else {
            self.match_length = 0;
        }

        if bit_context >= 128 {
            let idx = (byte_context as usize) % self.map.len();
            self.map[idx] = self.history_pos as u32;
            self.history_pos += 1;
        }
    }

    pub fn byte_update(&mut self, byte_context: u64, history: &[u8]) {
        if self.match_length < 8 {
            let idx = (byte_context as usize) % self.map.len();
            self.cur_match = self.map[idx] as u64;
        } else {
            self.cur_match += 1;
        }
        if !history.is_empty() {
            self.cur_match %= history.len() as u64;
            self.cur_byte = history[self.cur_match as usize];
        }
        self.bit_pos = 128;
        let match_context = (self.match_length / 32) as u64;
        if match_context > self.longest_match { self.longest_match = match_context; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::run_map::RunMap;

    #[test]
    fn direct_converges_on_constant_signal() {
        // delta = 1.0 → divisor ramps up faster, hitting 0.9+ within
        // the test window. Upstream typical configs use larger delta
        // (slower learning) for stability; we just want to verify
        // monotonic convergence here.
        let mut d = Direct::new(255, 1.0, 256);
        for _ in 0..2000 { d.perceive(1, 5, 7); }
        let p = d.predict(5, 7);
        assert!(p > 0.9, "p = {} (expected > 0.9)", p);
    }

    #[test]
    fn direct_hash_handles_collision_via_probe() {
        let mut h = DirectHash::new(255, 100.0, 4);
        // Train two contexts that hash to the same slot — verify the
        // second one finds an unused neighbour rather than corrupting.
        // Upstream uses checksum 0 as the "empty" sentinel, so callers
        // must avoid byte_context = 0 — we use 4 and 8 here instead
        // (both hash to slot 0 with size 4).
        h.byte_update(4);
        for _ in 0..50 { h.perceive(1, 0); }
        h.byte_update(8);
        // First probe at slot 0 is taken (checksums[0] = 4); the loop
        // moves to slot 1 and inserts 8 there.
        assert_eq!(h.index, 1);
    }

    #[test]
    fn indirect_uses_state_init_probability() {
        let map = vec![64u8; 1024]; // every cell starts at state 64.
        let ind = Indirect::new(RunMap::new(), 100.0, map.len(), 12345);
        // RunMap::init_probability(64) = (128-64)/256 = 0.25.
        assert!((ind.predictions[64] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn match_predict_returns_a_probability() {
        let history = vec![b'a'; 64];
        let mut m = Match::new(64, 100.0, 16);
        m.byte_update(0, &history);
        let p = m.predict();
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn byte_model_round_trip_a_byte() {
        // Bias the distribution sharply toward byte 0x42, then walk
        // the binary-search loop bit-by-bit and verify it ends on 0x42.
        let mut bm = ByteModel::new();
        for i in 0..256 {
            bm.probs[i] = if i == 0x42 { 0.5 } else { 0.5 / 255.0 };
        }
        let target: u8 = 0x42;
        for i in (0..8).rev() {
            let _ = bm.predict();
            let bit = ((target >> i) & 1) as i32;
            bm.perceive(bit);
        }
        // After 8 binary-search narrowings, bot == top == target.
        assert_eq!(bm.bot, target as i32);
        assert_eq!(bm.top, target as i32);
    }

    #[test]
    fn byte_model_byte_update_masks_vocab() {
        let mut bm = ByteModel::new();
        let mut vocab = [true; 256];
        vocab[0x80] = false;
        bm.probs[0x80] = 0.5;
        bm.byte_update(&vocab);
        assert_eq!(bm.probs[0x80], 0.0);
        assert_eq!(bm.bot, 0);
        assert_eq!(bm.top, 255);
    }
}
