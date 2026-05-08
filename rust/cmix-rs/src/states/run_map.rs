//! Run-length state map — port of `states/run-map.{h,cpp}`. The
//! 512-entry transition table is computed from the constructor
//! formula; it's small enough to keep at runtime rather than
//! pre-baking into a constant.

#![allow(dead_code)]

use crate::state::State;

#[derive(Clone)]
pub struct RunMap {
    table: [u8; 512],
}

impl RunMap {
    pub fn new() -> Self {
        let mut table = [0u8; 512];
        for i in 0..512 {
            let mut state = i / 2;
            if i % 2 == 0 {
                if state < 127       { state += 1; }
                else if state >= 128 { state = 0; }
            } else {
                if state < 128       { state = 128; }
                else if state < 255  { state += 1; }
            }
            table[i] = state as u8;
        }
        Self { table }
    }
}

impl Default for RunMap { fn default() -> Self { Self::new() } }

impl State for RunMap {
    fn init_probability(&self, state: i32) -> f32 {
        if state < 128 { (128.0 - state as f32) / 256.0 }
        else { state as f32 / 256.0 }
    }
    fn next(&self, state: i32, bit: i32) -> i32 {
        self.table[(state * 2 + bit) as usize] as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_matches_upstream_formula() {
        let m = RunMap::new();
        // Fresh state = 0, bit 0 → 1 (upstream: state 0 < 127, ++state).
        assert_eq!(m.next(0, 0), 1);
        // Fresh state = 0, bit 1 → 128 (jump into the upper half).
        assert_eq!(m.next(0, 1), 128);
        // State 127, bit 0 → 0 (else-clause: ≥128 ⇒ wrap).
        // Upstream: i=254, state=127, bit 0; state<127 fails, state>=128 fails ⇒ unchanged 127. So next=127.
        assert_eq!(m.next(127, 0), 127);
        // State 128, bit 1 → 129.
        assert_eq!(m.next(128, 1), 129);
        // State 255, bit 1 → 255 (saturated).
        assert_eq!(m.next(255, 1), 255);
    }

    #[test]
    fn init_probability_symmetric() {
        let m = RunMap::new();
        // State 0 → 0.5, state 64 → 0.25, state 128 → 0.5, state 192 → 0.75.
        assert!((m.init_probability(0)   - 0.5).abs() < 1e-6);
        assert!((m.init_probability(64)  - 0.25).abs() < 1e-6);
        assert!((m.init_probability(128) - 0.5).abs() < 1e-6);
        assert!((m.init_probability(192) - 0.75).abs() < 1e-6);
    }
}
