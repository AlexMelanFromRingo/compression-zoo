//! `dmcModel` + `dmcForest` — paq8.cpp:7615-7814.
//!
//! Dynamic Markov Compression: a bitwise context represented by a
//! state graph that clones nodes as the cloning threshold is reached.
//! `dmcForest` runs 10 `dmcModel`s with different parameters in
//! tandem (8 reset periodically, 2 never reset).

#![allow(dead_code)]

use super::apm::StateMap32;
use super::mixer::Mixer;
use super::state::Paq8State;
use super::substrate::{mem, nex};

const DMC_NODES_BASE: u64 = 255 * 256; // 65280
const DMC_NODES_MAX:  u64 = (1u64 << 31) / 12; // sizeof(DMCNode) = 12

/// DMC state-graph node — paq8.cpp:7615-7633. The bit-history state
/// byte is split across the low nibble of `nx0`/`nx1`; here it is a
/// plain field since Rust doesn't need the C++ bit-packing.
#[derive(Clone, Copy, Default)]
struct DmcNode {
    c0:    u16,
    c1:    u16,
    nx0:   u32,
    nx1:   u32,
    state: u8,
}

pub struct DmcModel {
    t:              Vec<DmcNode>,
    sm:             StateMap32,
    top:            u32,
    curr:           u32,
    threshold:      u32,
    threshold_fine: u32,
    extra:          u32,
}

impl DmcModel {
    pub fn new(dmc_nodes: u64, th_start: u32, dt: [i32; 1024]) -> Self {
        let size = (dmc_nodes + DMC_NODES_BASE).min(DMC_NODES_MAX) as usize;
        let mut m = Self {
            t:  vec![DmcNode::default(); size],
            sm: StateMap32::new(256, dt),
            top: 0, curr: 0,
            threshold: 0, threshold_fine: 0, extra: 0,
        };
        m.reset_state_graph(th_start);
        m
    }

    #[inline]
    fn increment_counter(x: u32, increment: u32) -> u32 {
        (((x << 6) - x) >> 6) + (increment << 10)
    }

    /// Initialize the state graph to a bytewise order-1 model.
    pub fn reset_state_graph(&mut self, th_start: u32) {
        self.top = 0; self.curr = 0; self.extra = 0;
        self.threshold = th_start;
        self.threshold_fine = th_start << 11;
        let init_count: u16 = if th_start < 1024 { 2048 } else { 512 };
        for _j in 0..256 {
            for i in 0..255u32 {
                let top = self.top as usize;
                if i < 127 {
                    self.t[top].nx0 = self.top + i + 1;
                    self.t[top].nx1 = self.top + i + 2;
                } else {
                    let linked_tree_root = (i - 127) * 2 * 255;
                    self.t[top].nx0 = linked_tree_root;
                    self.t[top].nx1 = linked_tree_root + 255;
                }
                self.t[top].c0 = init_count;
                self.t[top].c1 = init_count;
                self.t[top].state = 0;
                self.top += 1;
            }
        }
    }

    fn update_y(&mut self, y: u32) {
        let curr = self.curr as usize;
        let c0 = self.t[curr].c0 as u32;
        let c1 = self.t[curr].c1 as u32;
        let n = if y == 0 { c0 } else { c1 };

        self.t[curr].c0 = Self::increment_counter(c0, 1 - y) as u16;
        self.t[curr].c1 = Self::increment_counter(c1, y) as u16;
        self.t[curr].state = nex(self.t[curr].state, y as usize);

        if n > self.threshold {
            let next = if y == 0 {
                self.t[curr].nx0
            } else {
                self.t[curr].nx1
            } as usize;
            let mut c0n = self.t[next].c0 as u32;
            let mut c1n = self.t[next].c1 as u32;
            let nn = c0n + c1n;
            if nn > n + self.threshold {
                if self.top as usize != self.t.len() {
                    let top = self.top as usize;
                    let c0_top = (c0n as u64 * n as u64 / nn as u64) as u32;
                    let c1_top = (c1n as u64 * n as u64 / nn as u64) as u32;
                    c0n -= c0_top;
                    c1n -= c1_top;
                    self.t[top].c0 = c0_top as u16;
                    self.t[top].c1 = c1_top as u16;
                    self.t[next].c0 = c0n as u16;
                    self.t[next].c1 = c1n as u16;
                    self.t[top].nx0 = self.t[next].nx0;
                    self.t[top].nx1 = self.t[next].nx1;
                    self.t[top].state = self.t[next].state;
                    if y == 0 {
                        self.t[curr].nx0 = self.top;
                    } else {
                        self.t[curr].nx1 = self.top;
                    }
                    self.top += 1;
                    if self.threshold < 8 * 1024 {
                        self.threshold_fine += 1;
                        self.threshold = self.threshold_fine >> 11;
                    }
                } else {
                    self.extra += nn >> 10;
                }
            }
        }
        self.curr = if y == 0 {
            self.t[curr].nx0
        } else {
            self.t[curr].nx1
        };
    }

    pub fn is_full(&self) -> bool {
        (self.extra >> 7) as usize > self.t.len()
    }

    fn pr1(&self) -> i32 {
        let n0 = self.t[self.curr as usize].c0 as u32 + 1;
        let n1 = self.t[self.curr as usize].c1 as u32 + 1;
        ((n1 << 12) / (n0 + n1)) as i32
    }

    fn pr2(&mut self, y: i32) -> i32 {
        let state = self.t[self.curr as usize].state;
        self.sm.p(state as u32, 256, y)
    }

    /// `st()` — update + return the averaged stretched prediction.
    pub fn st(&mut self, y: i32, stretch: &super::substrate::Stretch) -> i32 {
        self.update_y(y as u32);
        stretch.get(self.pr1()) + stretch.get(self.pr2(y))
    }
}

pub struct DmcForest {
    models:    Vec<DmcModel>,
    dmcparams: [u32; 10],
}

impl DmcForest {
    pub fn new(level: u32, dt: [i32; 1024]) -> Self {
        let dmcparams: [u32; 10] = [2, 32, 64, 4, 128, 8, 256, 16, 1024, 1536];
        let dmcmem: [u64; 10] = [6, 10, 11, 7, 12, 8, 13, 9, 2, 2];
        let mem_q = mem(level) >> 2;
        let mut models = Vec::with_capacity(10);
        for i in 0..10 {
            models.push(DmcModel::new(mem_q / dmcmem[i], dmcparams[i], dt));
        }
        Self { models, dmcparams }
    }

    /// `mix` — paq8.cpp:7795-7813.
    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        let y = s.y;
        let stretch = &s.stretch;
        let mut i = 10;
        // The two slow models predict individually.
        i -= 1;
        let v = self.models[i].st(y, stretch);
        m.add((v >> 3) as i16);
        i -= 1;
        let v = self.models[i].st(y, stretch);
        m.add((v >> 3) as i16);
        // Fast models combined in pairs.
        while i > 0 {
            i -= 1;
            let pr1 = self.models[i].st(y, stretch);
            i -= 1;
            let pr2 = self.models[i].st(y, stretch);
            m.add(((pr1 + pr2) >> 4) as i16);
        }
        // Reset full fast models on byte boundary.
        if s.bpos == 0 {
            for j in (0..=7).rev() {
                if self.models[j].is_full() {
                    let p = self.dmcparams[j];
                    self.models[j].reset_state_graph(p);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn dmc_model_resets_and_updates() {
        let dt = build_dt();
        let mut m = DmcModel::new(1000, 32, dt);
        for i in 0..2000 {
            m.update_y((i & 1) as u32);
        }
        // After updates the model should still be in-bounds.
        assert!((m.curr as usize) < m.t.len());
    }

    #[test]
    fn dmc_forest_mixes_without_panic() {
        let dt = build_dt();
        let mut forest = DmcForest::new(0, dt); // small level for tests
        let mut s = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        for byte in 0u32..40 {
            for bp in 0..8 {
                s.bpos = bp;
                s.y = ((byte >> (7 - bp)) & 1) as i32;
                forest.mix(&mut s, &mut mixer);
            }
        }
    }
}
