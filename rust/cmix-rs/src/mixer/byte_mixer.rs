//! Byte-level LSTM-driven mixer — port of `mixer/byte-mixer.{h,cpp}`.
//!
//! Wraps a [`crate::mixer::lstm::Lstm`] to produce a 256-entry
//! probability distribution at each byte boundary, masked by the
//! caller-supplied vocabulary. Inputs from upstream models are
//! accumulated through `set_input`, scaled, fed to the LSTM, and the
//! softmax output is unpacked back into the 256-entry probs array.

#![allow(dead_code)]

use crate::mixer::lstm::Lstm;
use crate::models::ByteModel;

pub struct ByteMixer {
    pub byte_model: ByteModel,
    lstm: Lstm,
    /// `byte_map[i]` = compact index of byte `i` within the vocab,
    /// or anything if `vocab[i] == false`.
    byte_map: Vec<i32>,
    inputs: Vec<f32>,
    num_models: u32,
    vocab_size: usize,
    offset: usize,
    vocab: Vec<bool>,
}

impl ByteMixer {
    pub fn new(num_models: u32, vocab: Vec<bool>, vocab_size: usize, lstm: Lstm) -> Self {
        let mut byte_map = vec![0i32; 256];
        let mut offset = 0usize;
        for i in 0..256 {
            byte_map[i] = offset as i32;
            if vocab[i] { offset += 1; }
        }
        Self {
            byte_model: ByteModel::new(),
            lstm,
            byte_map,
            inputs: vec![0.0; vocab_size],
            num_models,
            vocab_size,
            offset: 0,
            vocab,
        }
    }

    /// Accumulate one model's vote for byte `index`. The cyclic
    /// `offset` advances with each call so successive `set_input`
    /// calls land in successive slots of the LSTM input vector.
    pub fn set_input(&mut self, index: usize, val: f32) {
        if !self.vocab[index] { return; }
        self.inputs[self.offset] += val;
        self.offset += 1;
        if self.offset == self.vocab_size { self.offset = 0; }
    }

    /// End-of-byte hook: scales accumulated inputs, feeds them to
    /// the LSTM via `set_input` + `perceive`, and unpacks the
    /// softmax output back into the byte-model's `probs[256]`.
    pub fn byte_update(&mut self, byte_context: u32) {
        let scale = 2.0 / self.num_models as f32;
        for v in self.inputs.iter_mut() { *v *= scale; }

        self.lstm.set_input(&self.inputs);
        for v in self.inputs.iter_mut() { *v = 0.0; }

        let symbol = self.byte_map[byte_context as usize] as u32;
        let output = self.lstm.perceive(symbol);

        let mut k = 0usize;
        for i in 0..256 {
            self.byte_model.probs[i] = if self.vocab[i] {
                let v = output[k];
                k += 1;
                v
            } else {
                0.0
            };
        }
        self.offset = 0;
        self.byte_model.byte_update(&self.vocab_array());
    }

    /// Helper to convert `Vec<bool>` to the `[bool; 256]` shape that
    /// `ByteModel::byte_update` expects.
    fn vocab_array(&self) -> [bool; 256] {
        let mut a = [false; 256];
        for i in 0..256.min(self.vocab.len()) { a[i] = self.vocab[i]; }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke: construct a small ByteMixer with a 4-byte vocab,
    /// pump some inputs, and verify the resulting probs are
    /// non-negative and zero outside the vocab.
    #[test]
    fn byte_mixer_smoke() {
        let mut vocab = vec![false; 256];
        vocab[b'a' as usize] = true;
        vocab[b'b' as usize] = true;
        vocab[b'c' as usize] = true;
        vocab[b'd' as usize] = true;
        let lstm = Lstm::new(
            /*input_size=*/4,
            /*output_size=*/4,
            /*num_cells=*/2,
            /*num_layers=*/1,
            /*horizon=*/2,
            /*learning_rate=*/0.01,
            /*gradient_clip=*/2.0,
        );
        let mut bm = ByteMixer::new(2, vocab, 4, lstm);
        bm.set_input(b'a' as usize, 0.3);
        bm.set_input(b'b' as usize, 0.4);
        bm.byte_update(b'a' as u32);
        for (i, p) in bm.byte_model.probs.iter().enumerate() {
            assert!(p.is_finite() && *p >= 0.0,
                "probs[{}] = {} (must be non-negative finite)", i, p);
            if i != b'a' as usize && i != b'b' as usize
                && i != b'c' as usize && i != b'd' as usize
            {
                assert_eq!(*p, 0.0, "non-vocab byte {} should be zero", i);
            }
        }
    }
}
