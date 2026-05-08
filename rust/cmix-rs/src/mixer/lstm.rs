//! Multi-layer LSTM with horizon-based BPTT — port of
//! `mixer/lstm.{h,cpp}`.
//!
//! Owns a stack of [`LstmLayer`]s plus the cross-layer hidden state,
//! the per-timestep `output_layer` weight tensor, and the cyclic
//! input/output history buffer. Every `perceive(input)` call:
//!   1. Backpropagates through `horizon` timesteps (only at epoch
//!      wrap-around — a coarse-grained TBPTT trade-off that mirrors
//!      upstream).
//!   2. SGD-updates the output layer with the latest training pair.
//!   3. Returns the next-step softmax probability vector via
//!      [`Self::predict`].

#![allow(dead_code)]

use crate::mixer::lstm_layer::LstmLayer;

pub struct Lstm {
    layers: Vec<LstmLayer>,
    input_history: Vec<u32>,
    /// Concatenated hidden state across layers, plus a bias 1 in the
    /// trailing slot. Length `num_cells * num_layers + 1`.
    hidden: Vec<f32>,
    hidden_error: Vec<f32>,
    /// Per-epoch per-layer concatenated input. Shape
    /// `[horizon][num_layers][input_size + 1 + num_cells * 2]`.
    layer_input: Vec<Vec<Vec<f32>>>,
    /// Per-epoch output-layer weights. Shape `[horizon][output_size][hidden.len()]`.
    output_layer: Vec<Vec<Vec<f32>>>,
    output: Vec<Vec<f32>>,
    learning_rate: f32,
    num_cells: usize,
    epoch: usize,
    horizon: usize,
    input_size: usize,
    output_size: usize,
}

impl Lstm {
    pub fn new(
        input_size: usize,
        output_size: usize,
        num_cells: usize,
        num_layers: usize,
        horizon: usize,
        learning_rate: f32,
        gradient_clip: f32,
    ) -> Self {
        let hidden_len = num_cells * num_layers + 1;
        let mut hidden = vec![0.0; hidden_len];
        hidden[hidden_len - 1] = 1.0;

        // layer_input: each layer's input vector size is
        //   layer 0: 1 + num_cells + input_size
        //   layer >0: input_size + 1 + num_cells * 2
        let mut layer_input = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let mut per_layer = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let sz = if layer == 0 {
                    1 + num_cells + input_size
                } else {
                    input_size + 1 + num_cells * 2
                };
                let mut v = vec![0.0; sz];
                v[sz - 1] = 1.0; // bias trailing slot.
                per_layer.push(v);
            }
            layer_input.push(per_layer);
        }

        // output_layer: per-epoch [output_size × hidden_len].
        let output_layer = (0..horizon).map(|_| {
            (0..output_size).map(|_| vec![0.0; hidden_len]).collect()
        }).collect();

        // output: per-epoch uniform 1/output_size.
        let inv_n = 1.0 / output_size as f32;
        let output = (0..horizon).map(|_|
            vec![inv_n; output_size]).collect();

        // Build layers.
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_input_size = if i == 0 {
                1 + num_cells + input_size
            } else {
                input_size + 1 + num_cells * 2
            };
            let total = layer_input_size + output_size;
            layers.push(LstmLayer::new(
                total,
                input_size,
                output_size,
                num_cells,
                horizon,
                gradient_clip,
                learning_rate,
                0xC0FFEE_u64.wrapping_add(i as u64 * 7919),
            ));
        }

        Self {
            layers,
            input_history: vec![0u32; horizon],
            hidden,
            hidden_error: vec![0.0; num_cells],
            layer_input,
            output_layer,
            output,
            learning_rate,
            num_cells,
            epoch: 0,
            horizon,
            input_size,
            output_size,
        }
    }

    pub fn set_input(&mut self, input: &[f32]) {
        // upstream: copy `input_size` floats into each layer's input
        // buffer (offset 0). Each layer reads its own copy.
        let n = self.input_size.min(input.len());
        for layer in 0..self.layers.len() {
            for j in 0..n {
                self.layer_input[self.epoch][layer][j] = input[j];
            }
        }
    }

    /// Run one forward pass for `input` (a discrete symbol index)
    /// and return the resulting softmax over `output_size` symbols.
    pub fn predict(&mut self, input: u32) -> Vec<f32> {
        for i in 0..self.layers.len() {
            // Copy hidden[i*num_cells..(i+1)*num_cells] into the layer's
            // input vector at offset input_size.
            let start = i * self.num_cells;
            for j in 0..self.num_cells {
                self.layer_input[self.epoch][i][self.input_size + j] = self.hidden[start + j];
            }
            // Borrow hidden mutably for forward_pass; copy out the
            // input row first to avoid aliasing self.layer_input.
            let layer_in = self.layer_input[self.epoch][i].clone();
            self.layers[i].forward_pass(
                &layer_in, input as usize, &mut self.hidden, i * self.num_cells,
            );
            // Copy hidden[i*num_cells..(i+1)*num_cells] into next
            // layer's input vector at offset num_cells + input_size.
            if i + 1 < self.layers.len() {
                for j in 0..self.num_cells {
                    self.layer_input[self.epoch][i + 1][self.num_cells + self.input_size + j]
                        = self.hidden[start + j];
                }
            }
        }
        // Softmax over output_size.
        let mut max_out = 0.0f32;
        let mut out = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = 0.0f32;
            for j in 0..self.hidden.len() {
                sum += self.hidden[j] * self.output_layer[self.epoch][i][j];
            }
            out[i] = sum;
            if sum > max_out { max_out = sum; }
        }
        for v in out.iter_mut() { *v = (*v - max_out).exp(); }
        let sum: f32 = out.iter().sum();
        if sum > 0.0 { for v in out.iter_mut() { *v /= sum; } }
        self.output[self.epoch] = out.clone();

        let cur = self.epoch;
        self.epoch += 1;
        if self.epoch == self.horizon { self.epoch = 0; }
        self.output[cur].clone()
    }

    /// Update on input (the just-observed symbol) and return the
    /// next-step softmax. Mirrors upstream `Perceive`.
    pub fn perceive(&mut self, input: u32) -> Vec<f32> {
        let last_epoch = if self.epoch == 0 { self.horizon - 1 } else { self.epoch - 1 };
        let old_input = self.input_history[last_epoch];
        self.input_history[last_epoch] = input;

        // Backwards pass through horizon: only when we wrap (epoch_ == 0).
        if self.epoch == 0 {
            for epoch in (0..self.horizon).rev() {
                for layer in (0..self.layers.len()).rev() {
                    let offset = layer * self.num_cells;
                    for h in self.hidden_error.iter_mut() { *h = 0.0; }
                    for i in 0..self.output_size {
                        let target = self.input_history[epoch];
                        let err = if i as u32 == target {
                            self.output[epoch][i] - 1.0
                        } else {
                            self.output[epoch][i]
                        };
                        for j in 0..self.hidden_error.len() {
                            self.hidden_error[j] +=
                                self.output_layer[epoch][i][j + offset] * err;
                        }
                    }
                    let prev_epoch = if epoch == 0 { self.horizon - 1 } else { epoch - 1 };
                    let input_symbol = if epoch == 0 { old_input } else { self.input_history[prev_epoch] };
                    let layer_in = self.layer_input[epoch][layer].clone();
                    let mut he = self.hidden_error.clone();
                    self.layers[layer].backward_pass(
                        &layer_in, epoch, layer, input_symbol as usize, &mut he,
                    );
                    self.hidden_error = he;
                }
            }
        }

        // Output-layer SGD update: w[i] = prev_w[i] - lr * err * hidden.
        for i in 0..self.output_size {
            let err = if i as u32 == input {
                self.output[last_epoch][i] - 1.0
            } else {
                self.output[last_epoch][i]
            };
            for j in 0..self.hidden.len() {
                let prev = self.output_layer[last_epoch][i][j];
                self.output_layer[self.epoch][i][j] =
                    prev - self.learning_rate * err * self.hidden[j];
            }
        }
        self.predict(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a tiny LSTM and run a few perceive() calls. We just
    /// check the outputs stay finite + sum ≈ 1.
    #[test]
    fn forward_smoke() {
        let mut l = Lstm::new(
            /*input_size=*/4,
            /*output_size=*/8,
            /*num_cells=*/4,
            /*num_layers=*/2,
            /*horizon=*/3,
            /*learning_rate=*/0.01,
            /*gradient_clip=*/2.0,
        );
        l.set_input(&[0.1f32; 4]);
        let _ = l.predict(0);
        l.set_input(&[0.2f32; 4]);
        let p = l.perceive(1);
        assert_eq!(p.len(), 8);
        for v in &p { assert!(v.is_finite() && *v >= 0.0); }
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "softmax must sum to 1, got {}", s);
    }
}
