//! Sigmoid table — port of `mixer/sigmoid.{h,cpp}`. Used by the
//! mixer to convert clamped probabilities to logits via a precomputed
//! lookup table; the inverse logistic is computed scalar-style with
//! `f32::exp`.

#![allow(dead_code)]

#[derive(Clone)]
pub struct Sigmoid {
    logit_size: i32,
    logit_table: Vec<f32>,
}

impl Sigmoid {
    pub fn new(logit_size: i32) -> Self {
        let n = logit_size as usize;
        let mut table = vec![0f32; n];
        for i in 0..n {
            let p = (i as f32 + 0.5) / logit_size as f32;
            table[i] = (p / (1.0 - p)).ln();
        }
        Self { logit_size, logit_table: table }
    }

    /// Logit of `p ∈ [0, 1]`, looked up from the table at the
    /// nearest `logit_size`-bin slot.
    pub fn logit(&self, p: f32) -> f32 {
        let mut idx = (p * self.logit_size as f32) as i32;
        if idx >= self.logit_size { idx = self.logit_size - 1; }
        else if idx < 0 { idx = 0; }
        self.logit_table[idx as usize]
    }

    /// Logistic of `p`. Stateless, computed via `exp`. Mirrors
    /// upstream `static float Logistic`.
    pub fn logistic(p: f32) -> f32 {
        1.0 / (1.0 + (-p).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_inits() {
        let s = Sigmoid::new(4096);
        // logit(p) ≈ -logit(1-p); test a few values.
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let l = s.logit(p);
            let inv = Sigmoid::logistic(l);
            assert!((inv - p).abs() < 0.01,
                "logistic(logit({})) = {}, expected ≈ {}", p, inv, p);
        }
    }

    #[test]
    fn logit_bounds() {
        let s = Sigmoid::new(4096);
        let _ = s.logit(0.0);     // Smallest bin.
        let _ = s.logit(1.0);     // Clamped to `logit_size - 1`.
        let _ = s.logit(-0.5);    // Clamped to 0.
        let _ = s.logit(1.5);     // Clamped to `logit_size - 1`.
    }
}
