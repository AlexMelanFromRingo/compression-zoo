//! Single LSTM layer — port of `mixer/lstm-layer.{h,cpp}`.
//!
//! Each layer holds three neuron groups (`forget_gate`, `input_node`,
//! `output_gate`), each backed by a weight matrix + per-cell
//! normalisation params + Adam optimiser state. Forward and
//! backward passes step through `horizon` timesteps in a circular
//! buffer for BPTT.
//!
//! Upstream uses `valarray<valarray<float>>` for 2-D matrices; the
//! Rust port flattens these into `Vec<f32>` with explicit
//! `row * cols + col` indexing.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::sigmoid::Sigmoid;

/// Adam optimiser update. Mirrors the anonymous-namespace `Adam` in
/// `lstm-layer.cpp:11`.
fn adam(
    g: &[f32], m: &mut [f32], v: &mut [f32], w: &mut [f32],
    learning_rate: f32, t: f32, update_limit: u64,
) {
    const BETA1: f32 = 0.025;
    const BETA2: f32 = 0.9999;
    const EPS: f32   = 1e-6;

    let alpha = if t < update_limit as f32 {
        learning_rate * 0.1 / (5e-5 * t + 1.0).sqrt()
    } else {
        learning_rate * 0.1 / (5e-5 * update_limit as f32 + 1.0).sqrt()
    };
    for i in 0..g.len() {
        m[i] = m[i] * BETA1 + (1.0 - BETA1) * g[i];
        v[i] = v[i] * BETA2 + (1.0 - BETA2) * g[i] * g[i];
    }
    let t_for_bias = if t < update_limit as f32 { t } else { update_limit as f32 };
    let bias1 = 1.0 - BETA1.powf(t_for_bias);
    let bias2 = 1.0 - BETA2.powf(t_for_bias);
    for i in 0..w.len() {
        w[i] -= alpha * (m[i] / bias1) / ((v[i] / bias2 + EPS).sqrt());
    }
}

/// One gate's worth of state — three of these per [`LstmLayer`].
pub(crate) struct NeuronLayer {
    pub error: Vec<f32>,         // [num_cells]
    pub ivar: Vec<f32>,          // [horizon]
    pub gamma: Vec<f32>,         // [num_cells]
    pub gamma_u: Vec<f32>,
    pub gamma_m: Vec<f32>,
    pub gamma_v: Vec<f32>,
    pub beta: Vec<f32>,
    pub beta_u: Vec<f32>,
    pub beta_m: Vec<f32>,
    pub beta_v: Vec<f32>,
    pub weights: Vec<f32>,       // [num_cells * input_size]
    pub state: Vec<f32>,         // [horizon * num_cells]
    pub update: Vec<f32>,        // [num_cells * input_size]
    pub m: Vec<f32>,             // [num_cells * input_size]
    pub v: Vec<f32>,             // [num_cells * input_size]
    pub transpose: Vec<f32>,     // [(input_size - offset) * num_cells]
    pub norm: Vec<f32>,          // [horizon * num_cells]
    pub input_size: usize,
    pub num_cells: usize,
    pub horizon: usize,
    pub offset: usize,
}

impl NeuronLayer {
    pub fn new(input_size: usize, num_cells: usize, horizon: usize, offset: usize) -> Self {
        Self {
            error: vec![0.0; num_cells],
            ivar: vec![0.0; horizon],
            gamma: vec![1.0; num_cells],
            gamma_u: vec![0.0; num_cells],
            gamma_m: vec![0.0; num_cells],
            gamma_v: vec![0.0; num_cells],
            beta: vec![0.0; num_cells],
            beta_u: vec![0.0; num_cells],
            beta_m: vec![0.0; num_cells],
            beta_v: vec![0.0; num_cells],
            weights: vec![0.0; num_cells * input_size],
            state: vec![0.0; horizon * num_cells],
            update: vec![0.0; num_cells * input_size],
            m: vec![0.0; num_cells * input_size],
            v: vec![0.0; num_cells * input_size],
            transpose: vec![0.0; (input_size - offset) * num_cells],
            norm: vec![0.0; horizon * num_cells],
            input_size, num_cells, horizon, offset,
        }
    }
    /// `weights[cell, j]`.
    #[inline] pub fn w(&self, cell: usize, j: usize) -> f32 {
        self.weights[cell * self.input_size + j]
    }
    #[inline] pub fn w_mut(&mut self, cell: usize, j: usize) -> &mut f32 {
        &mut self.weights[cell * self.input_size + j]
    }
    #[inline] pub fn st(&self, epoch: usize, cell: usize) -> f32 {
        self.state[epoch * self.num_cells + cell]
    }
    #[inline] pub fn nm(&self, epoch: usize, cell: usize) -> f32 {
        self.norm[epoch * self.num_cells + cell]
    }
    #[inline] pub fn st_mut(&mut self, epoch: usize, cell: usize) -> &mut f32 {
        &mut self.state[epoch * self.num_cells + cell]
    }
    #[inline] pub fn nm_mut(&mut self, epoch: usize, cell: usize) -> &mut f32 {
        &mut self.norm[epoch * self.num_cells + cell]
    }
}

pub struct LstmLayer {
    pub state: Vec<f32>,         // [num_cells]
    pub state_error: Vec<f32>,
    pub stored_error: Vec<f32>,
    pub tanh_state: Vec<f32>,    // [horizon * num_cells]
    pub input_gate_state: Vec<f32>,
    pub last_state: Vec<f32>,
    pub gradient_clip: f32,
    pub learning_rate: f32,
    pub num_cells: usize,
    pub epoch: usize,
    pub horizon: usize,
    pub input_size: usize,       // = auxiliary_input_size from upstream
    pub output_size: usize,
    pub update_steps: u64,
    pub update_limit: u64,
    pub forget_gate: NeuronLayer,
    pub input_node: NeuronLayer,
    pub output_gate: NeuronLayer,
}

impl LstmLayer {
    pub fn new(
        input_size: usize,
        auxiliary_input_size: usize,
        output_size: usize,
        num_cells: usize,
        horizon: usize,
        gradient_clip: f32,
        learning_rate: f32,
        seed: u64,
    ) -> Self {
        let offset = output_size + auxiliary_input_size;
        let mut s = Self {
            state:           vec![0.0; num_cells],
            state_error:     vec![0.0; num_cells],
            stored_error:    vec![0.0; num_cells],
            tanh_state:      vec![0.0; horizon * num_cells],
            input_gate_state:vec![0.0; horizon * num_cells],
            last_state:      vec![0.0; horizon * num_cells],
            gradient_clip,
            learning_rate,
            num_cells,
            epoch: 0,
            horizon,
            input_size: auxiliary_input_size,
            output_size,
            update_steps: 0,
            update_limit: 3000,
            forget_gate: NeuronLayer::new(input_size, num_cells, horizon, offset),
            input_node:  NeuronLayer::new(input_size, num_cells, horizon, offset),
            output_gate: NeuronLayer::new(input_size, num_cells, horizon, offset),
        };

        // Glorot-style init: range = 2*sqrt(6/(input_size+output_size)).
        let val = (6.0 / (s.input_size as f32 + s.output_size as f32)).sqrt();
        let low = -val;
        let range = 2.0 * val;
        // Deterministic LCG so tests are reproducible (upstream uses
        // libc rand() which is non-deterministic across runs anyway).
        let mut rng = seed;
        let mut next_rand = || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng >> 16) as f32 / 65536.0
        };
        for i in 0..num_cells {
            for j in 0..input_size {
                let r = low + next_rand() * range;
                *s.forget_gate.w_mut(i, j) = r;
                let r = low + next_rand() * range;
                *s.input_node.w_mut(i, j) = r;
                let r = low + next_rand() * range;
                *s.output_gate.w_mut(i, j) = r;
            }
            // Forget gate's last weight is initialised to 1 (the
            // "remember by default" trick).
            *s.forget_gate.w_mut(i, input_size - 1) = 1.0;
        }
        s
    }

    fn forward_neuron(neurons: &mut NeuronLayer, input: &[f32], input_symbol: usize, epoch: usize) {
        // 1) Linear forward: f = w[input_symbol] + Σ input * w[output_size + j].
        let output_size = neurons.input_size - input.len() - 0; // not used; see offset
        let _ = output_size;
        let in_sz = neurons.input_size;
        for i in 0..neurons.num_cells {
            let mut f = neurons.weights[i * in_sz + input_symbol];
            for j in 0..input.len() {
                // upstream: weights[i][output_size + j]. The "output_size"
                // here is the number of input-symbol slots before the
                // floating input vector starts.
                f += input[j] * neurons.weights[i * in_sz + (in_sz - input.len()) + j];
            }
            *neurons.nm_mut(epoch, i) = f;
        }
        // 2) Per-timestep layer normalisation: ivar = 1/sqrt(mean(norm^2) + eps).
        let mut sum_sq = 0.0f32;
        for i in 0..neurons.num_cells {
            let v = neurons.nm(epoch, i);
            sum_sq += v * v;
        }
        neurons.ivar[epoch] = 1.0 / (sum_sq / neurons.num_cells as f32 + 1e-5).sqrt();
        // 3) Apply normalisation + scale + shift.
        let ivar = neurons.ivar[epoch];
        for i in 0..neurons.num_cells {
            let nm = neurons.nm(epoch, i) * ivar;
            *neurons.nm_mut(epoch, i) = nm;
            *neurons.st_mut(epoch, i) = nm * neurons.gamma[i] + neurons.beta[i];
        }
    }

    /// Public forward pass — runs all three gates and writes the
    /// next hidden state into `hidden[hidden_start .. hidden_start + num_cells]`.
    pub fn forward_pass(
        &mut self,
        input: &[f32],
        input_symbol: usize,
        hidden: &mut [f32],
        hidden_start: usize,
    ) {
        let epoch = self.epoch;
        // Save previous state.
        for i in 0..self.num_cells {
            self.last_state[epoch * self.num_cells + i] = self.state[i];
        }
        Self::forward_neuron(&mut self.forget_gate, input, input_symbol, epoch);
        Self::forward_neuron(&mut self.input_node,  input, input_symbol, epoch);
        Self::forward_neuron(&mut self.output_gate, input, input_symbol, epoch);

        for i in 0..self.num_cells {
            let f = Sigmoid::logistic(self.forget_gate.st(epoch, i));
            *self.forget_gate.st_mut(epoch, i) = f;
            let g = self.input_node.st(epoch, i).tanh();
            *self.input_node.st_mut(epoch, i) = g;
            let o = Sigmoid::logistic(self.output_gate.st(epoch, i));
            *self.output_gate.st_mut(epoch, i) = o;
        }

        // input_gate_state = 1 - forget_gate_state.
        for i in 0..self.num_cells {
            self.input_gate_state[epoch * self.num_cells + i] =
                1.0 - self.forget_gate.st(epoch, i);
        }

        // state = state * f + input_node * (1 - f).
        for i in 0..self.num_cells {
            let f = self.forget_gate.st(epoch, i);
            let g = self.input_node.st(epoch, i);
            let one_minus_f = self.input_gate_state[epoch * self.num_cells + i];
            self.state[i] = self.state[i] * f + g * one_minus_f;
        }

        // tanh_state = tanh(state); hidden = output_gate * tanh_state.
        for i in 0..self.num_cells {
            let t = self.state[i].tanh();
            self.tanh_state[epoch * self.num_cells + i] = t;
            hidden[hidden_start + i] = self.output_gate.st(epoch, i) * t;
        }

        self.epoch += 1;
        if self.epoch == self.horizon { self.epoch = 0; }
    }

    pub fn backward_pass(
        &mut self,
        input: &[f32],
        epoch: usize,
        layer: usize,
        input_symbol: usize,
        hidden_error: &mut [f32],
    ) {
        if epoch == self.horizon - 1 {
            self.stored_error.copy_from_slice(hidden_error);
            for x in self.state_error.iter_mut() { *x = 0.0; }
        } else {
            for i in 0..self.num_cells { self.stored_error[i] += hidden_error[i]; }
        }

        for i in 0..self.num_cells {
            let o   = self.output_gate.st(epoch, i);
            let t   = self.tanh_state[epoch * self.num_cells + i];
            let g   = self.input_node.st(epoch, i);
            let f   = self.forget_gate.st(epoch, i);
            let igs = self.input_gate_state[epoch * self.num_cells + i];
            let ls  = self.last_state[epoch * self.num_cells + i];
            self.output_gate.error[i] = t * self.stored_error[i] * o * (1.0 - o);
            self.state_error[i] += self.stored_error[i] * o * (1.0 - t * t);
            self.input_node.error[i]  = self.state_error[i] * igs * (1.0 - g * g);
            self.forget_gate.error[i] = (ls - g) * self.state_error[i] * f * igs;
        }

        for x in hidden_error.iter_mut() { *x = 0.0; }
        if epoch > 0 {
            for i in 0..self.num_cells {
                self.state_error[i] *= self.forget_gate.st(epoch, i);
            }
            for x in self.stored_error.iter_mut() { *x = 0.0; }
        } else if self.update_steps < self.update_limit {
            self.update_steps += 1;
        }

        Self::backward_neuron(
            &mut self.forget_gate, &mut self.stored_error,
            input, epoch, layer, input_symbol, hidden_error,
            self.num_cells, self.input_size, self.output_size,
            self.learning_rate, self.update_steps, self.update_limit,
        );
        Self::backward_neuron(
            &mut self.input_node, &mut self.stored_error,
            input, epoch, layer, input_symbol, hidden_error,
            self.num_cells, self.input_size, self.output_size,
            self.learning_rate, self.update_steps, self.update_limit,
        );
        Self::backward_neuron(
            &mut self.output_gate, &mut self.stored_error,
            input, epoch, layer, input_symbol, hidden_error,
            self.num_cells, self.input_size, self.output_size,
            self.learning_rate, self.update_steps, self.update_limit,
        );

        let clip = self.gradient_clip;
        Self::clip_gradients(&mut self.state_error, clip);
        Self::clip_gradients(&mut self.stored_error, clip);
        Self::clip_gradients(hidden_error, clip);
    }

    fn clip_gradients(arr: &mut [f32], clip: f32) {
        for v in arr.iter_mut() {
            if      *v < -clip { *v = -clip; }
            else if *v >  clip { *v =  clip; }
        }
    }

    fn backward_neuron(
        neurons: &mut NeuronLayer,
        stored_error: &mut [f32],
        input: &[f32],
        epoch: usize,
        layer: usize,
        input_symbol: usize,
        hidden_error: &mut [f32],
        num_cells: usize,
        input_size: usize,
        output_size: usize,
        learning_rate: f32,
        update_steps: u64,
        update_limit: u64,
    ) {
        if epoch == neurons.horizon - 1 {
            for v in neurons.gamma_u.iter_mut() { *v = 0.0; }
            for v in neurons.beta_u.iter_mut()  { *v = 0.0; }
            for i in 0..num_cells {
                for j in 0..neurons.input_size {
                    neurons.update[i * neurons.input_size + j] = 0.0;
                }
                let off = output_size + input_size;
                let trans_rows = neurons.input_size - off;
                for j in 0..trans_rows {
                    neurons.transpose[j * num_cells + i] =
                        neurons.weights[i * neurons.input_size + j + off];
                }
            }
        }
        for i in 0..num_cells {
            neurons.beta_u[i]  += neurons.error[i];
            neurons.gamma_u[i] += neurons.error[i] * neurons.nm(epoch, i);
            neurons.error[i] *= neurons.gamma[i] * neurons.ivar[epoch];
        }
        let mean = {
            let mut s = 0.0f32;
            for i in 0..num_cells { s += neurons.error[i] * neurons.nm(epoch, i); }
            s / num_cells as f32
        };
        for i in 0..num_cells {
            neurons.error[i] -= mean * neurons.nm(epoch, i);
        }

        if layer > 0 {
            for i in 0..num_cells {
                let mut f = 0.0f32;
                for j in 0..num_cells {
                    f += neurons.error[j]
                        * neurons.transpose[(num_cells + i) * num_cells + j];
                }
                hidden_error[i] += f;
            }
        }
        if epoch > 0 {
            for i in 0..num_cells {
                let mut f = 0.0f32;
                for j in 0..num_cells {
                    f += neurons.error[j] * neurons.transpose[i * num_cells + j];
                }
                stored_error[i] += f;
            }
        }

        // update[i][output_size + k] += error[i] * input[k];  update[i][input_symbol] += error[i].
        for i in 0..num_cells {
            for k in 0..input.len() {
                neurons.update[i * neurons.input_size + output_size + k]
                    += neurons.error[i] * input[k];
            }
            neurons.update[i * neurons.input_size + input_symbol] += neurons.error[i];
        }

        if epoch == 0 {
            for i in 0..num_cells {
                let row_w  = i * neurons.input_size .. (i + 1) * neurons.input_size;
                // Borrow update / m / v / weights as separate slices.
                let (g, m, v, w);
                {
                    let n = neurons.input_size;
                    let lo = i * n;
                    let hi = lo + n;
                    g = neurons.update[lo..hi].to_vec();
                    m = neurons.m[lo..hi].to_vec();
                    v = neurons.v[lo..hi].to_vec();
                    w = neurons.weights[lo..hi].to_vec();
                }
                let mut m = m; let mut v = v; let mut w = w;
                adam(&g, &mut m, &mut v, &mut w,
                     learning_rate, update_steps as f32, update_limit);
                neurons.m[row_w.clone()].copy_from_slice(&m);
                neurons.v[row_w.clone()].copy_from_slice(&v);
                neurons.weights[row_w].copy_from_slice(&w);
            }
            adam(&neurons.gamma_u.clone(), &mut neurons.gamma_m, &mut neurons.gamma_v,
                 &mut neurons.gamma, learning_rate, update_steps as f32, update_limit);
            adam(&neurons.beta_u.clone(), &mut neurons.beta_m, &mut neurons.beta_v,
                 &mut neurons.beta,  learning_rate, update_steps as f32, update_limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward pass on a fresh layer should produce finite outputs
    /// and not explode the state. We don't check exact numerical
    /// equivalence with upstream (different rand stream); we just
    /// confirm the BPTT pipeline doesn't NaN/Inf on a tiny config.
    #[test]
    fn forward_pass_smoke() {
        let mut l = LstmLayer::new(
            /*input_size=*/8 + 4,         // output_size + aux
            /*aux=*/4,
            /*output_size=*/8,
            /*num_cells=*/4,
            /*horizon=*/3,
            /*gradient_clip=*/2.0,
            /*learning_rate=*/0.01,
            /*seed=*/0xCAFE,
        );
        let input = vec![0.1f32; 4];
        let mut hidden = vec![0.0f32; 4];
        l.forward_pass(&input, 0, &mut hidden, 0);
        for h in hidden { assert!(h.is_finite()); }
    }
}
