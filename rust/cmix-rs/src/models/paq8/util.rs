//! Small paq8 utilities — OLS, IndirectContext, MTFList, Cache.
//! paq8.cpp:1365-1529, 3007-3028.

#![allow(dead_code)]

// =============================================================
// OLS — Ordinary Least Squares predictor (paq8.cpp:1365-1466).
// =============================================================

/// Recursive-least-squares linear predictor with Cholesky solve.
/// `sub` is upstream's mean-offset constant (`0` for the
/// `hasZeroMean = true` instantiations, `1 << (bits-1)` otherwise).
pub struct Ols {
    n:      usize,
    kmax:   i32,
    km:     i32,
    index:  usize,
    lambda: f64,
    nu:     f64,
    sub:    f64,
    x:      Vec<f64>,
    w:      Vec<f64>,
    b:      Vec<f64>,
    m_cov:  Vec<Vec<f64>>,
    m_chol: Vec<Vec<f64>>,
}

impl Ols {
    const FTOL: f64 = 1e-8;

    pub fn new(n: usize, kmax: i32, lambda: f64, nu: f64, sub: f64) -> Self {
        Self {
            n, kmax, km: 0, index: 0, lambda, nu, sub,
            x: vec![0.0; n],
            w: vec![0.0; n],
            b: vec![0.0; n],
            m_cov:  vec![vec![0.0; n]; n],
            m_chol: vec![vec![0.0; n]; n],
        }
    }

    fn factor(&mut self) -> i32 {
        for i in 0..self.n {
            for j in 0..self.n {
                self.m_chol[i][j] = self.m_cov[i][j];
            }
        }
        for i in 0..self.n {
            self.m_chol[i][i] += self.nu;
        }
        for i in 0..self.n {
            for j in 0..i {
                let mut sum = self.m_chol[i][j];
                for k in 0..j {
                    sum -= self.m_chol[i][k] * self.m_chol[j][k];
                }
                self.m_chol[i][j] = sum / self.m_chol[j][j];
            }
            let mut sum = self.m_chol[i][i];
            for k in 0..i {
                sum -= self.m_chol[i][k] * self.m_chol[i][k];
            }
            if sum > Self::FTOL {
                self.m_chol[i][i] = sum.sqrt();
            } else {
                return 1;
            }
        }
        0
    }

    fn solve(&mut self) {
        for i in 0..self.n {
            let mut sum = self.b[i];
            for j in 0..i {
                sum -= self.m_chol[i][j] * self.w[j];
            }
            self.w[i] = sum / self.m_chol[i][i];
        }
        for i in (0..self.n).rev() {
            let mut sum = self.w[i];
            for j in (i + 1)..self.n {
                sum -= self.m_chol[j][i] * self.w[j];
            }
            self.w[i] = sum / self.m_chol[i][i];
        }
    }

    pub fn add(&mut self, val: f64) {
        if self.index < self.n {
            self.x[self.index] = val - self.sub;
            self.index += 1;
        }
    }

    /// `Predict()` with already-loaded `x` — paq8.cpp:1447-1453.
    pub fn predict(&mut self) -> f64 {
        self.index = 0;
        let mut sum = 0.0;
        for i in 0..self.n {
            sum += self.w[i] * self.x[i];
        }
        sum + self.sub
    }

    /// `Predict(p)` — refresh `x` from `p` then dot with `w`.
    pub fn predict_from(&mut self, p: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.n {
            self.x[i] = p[i] - self.sub;
            sum += self.w[i] * self.x[i];
        }
        sum + self.sub
    }

    pub fn update(&mut self, val: f64) {
        for j in 0..self.n {
            for i in 0..self.n {
                self.m_cov[j][i] = self.lambda * self.m_cov[j][i]
                    + (1.0 - self.lambda) * (self.x[j] * self.x[i]);
            }
        }
        for i in 0..self.n {
            self.b[i] = self.lambda * self.b[i]
                + (1.0 - self.lambda) * (self.x[i] * (val - self.sub));
        }
        self.km += 1;
        if self.km >= self.kmax {
            if self.factor() == 0 {
                self.solve();
            }
            self.km = 0;
        }
    }
}

// =============================================================
// IndirectContext — paq8.cpp:1471-1495.
// =============================================================

/// Indirect-context history table. `bits_per_context` sizes the
/// table; `input_bits` controls the shift width; `storage_bits` is
/// the width of the stored cell (upstream's `T` — 8 for
/// `IndirectContext<U8>`, 16 for `<U16>`, 32 for `<U32>`). Cells are
/// truncated to `storage_bits` after each `add`, mirroring C++
/// integer-type wraparound.
pub struct IndirectContext {
    data:        Vec<u32>,
    ctx_idx:     usize,
    ctx_mask:    u32,
    input_mask:  u32,
    input_bits:  u32,
    storage_mask: u32,
}

impl IndirectContext {
    pub fn new(bits_per_context: u32, input_bits: u32,
                storage_bits: u32) -> Self {
        Self {
            data: vec![0u32; 1usize << bits_per_context],
            ctx_idx: 0,
            ctx_mask: (1u32 << bits_per_context) - 1,
            input_mask: (1u32 << input_bits) - 1,
            input_bits,
            storage_mask: if storage_bits >= 32 {
                u32::MAX
            } else {
                (1u32 << storage_bits) - 1
            },
        }
    }

    /// `operator+=(i)` — fold `i` into the current ctx slot.
    pub fn add(&mut self, i: u32) {
        let v = &mut self.data[self.ctx_idx];
        *v = ((*v << self.input_bits) | (i & self.input_mask)) & self.storage_mask;
    }

    /// `operator=(i)` — repoint ctx at `data[i & ctxMask]`.
    pub fn set(&mut self, i: u32) {
        self.ctx_idx = (i & self.ctx_mask) as usize;
    }

    /// `operator()` — current ctx value.
    pub fn get(&self) -> u32 { self.data[self.ctx_idx] }
}

// =============================================================
// MTFList — move-to-front list (paq8.cpp:1499-1529).
// =============================================================

pub struct MtfList {
    root:     i32,
    index:    i32,
    previous: Vec<i32>,
    next:     Vec<i32>,
}

impl MtfList {
    pub fn new(n: usize) -> Self {
        let mut previous = vec![0i32; n];
        let mut next = vec![0i32; n];
        for i in 0..n {
            previous[i] = i as i32 - 1;
            next[i] = i as i32 + 1;
        }
        next[n - 1] = -1;
        Self { root: 0, index: 0, previous, next }
    }

    pub fn get_first(&mut self) -> i32 {
        self.index = self.root;
        self.index
    }

    pub fn get_next(&mut self) -> i32 {
        if self.index >= 0 {
            self.index = self.next[self.index as usize];
        }
        self.index
    }

    pub fn move_to_front(&mut self, i: i32) {
        self.index = i;
        if i == self.root { return; }
        let idx = self.index as usize;
        let p = self.previous[idx];
        let n = self.next[idx];
        if p >= 0 { self.next[p as usize] = n; }
        if n >= 0 { self.previous[n as usize] = p; }
        self.previous[self.root as usize] = self.index;
        self.next[idx] = self.root;
        self.root = self.index;
        self.previous[self.root as usize] = -1;
    }
}

// =============================================================
// Cache<T, Size> — paq8.cpp:3007-3028. Ring of `Size` elements.
// =============================================================

pub struct Cache<T: Clone + Default> {
    data:  Vec<T>,
    index: u32,
    size:  u32,
}

impl<T: Clone + Default> Cache<T> {
    /// `size` must be a power of two > 1.
    pub fn new(size: usize) -> Self {
        debug_assert!(size > 1 && size.is_power_of_two());
        Self { data: vec![T::default(); size], index: 0, size: size as u32 }
    }

    /// `operator()(i)` — element `i` slots back.
    pub fn at(&self, i: u32) -> &T {
        &self.data[((self.index.wrapping_sub(i)) & (self.size - 1)) as usize]
    }
    pub fn at_mut(&mut self, i: u32) -> &mut T {
        let idx = ((self.index.wrapping_sub(i)) & (self.size - 1)) as usize;
        &mut self.data[idx]
    }

    /// `operator++` — advance the ring index.
    pub fn advance(&mut self) { self.index = self.index.wrapping_add(1); }
    /// `operator--` — rewind the ring index.
    pub fn retreat(&mut self) { self.index = self.index.wrapping_sub(1); }

    /// `Next()` — advance + clear-and-return the new slot.
    pub fn next(&mut self) -> &mut T {
        self.index = self.index.wrapping_add(1);
        let idx = (self.index & (self.size - 1)) as usize;
        self.data[idx] = T::default();
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_predicts_linear_relation() {
        // Train OLS on y = 2*x0 + 3*x1 with zero-mean (sub=0).
        let mut ols = Ols::new(2, 1, 0.998, 0.001, 0.0);
        for t in 0..2000 {
            let x0 = ((t * 7) % 11) as f64;
            let x1 = ((t * 3) % 13) as f64;
            ols.add(x0);
            ols.add(x1);
            let _ = ols.predict();
            ols.update(2.0 * x0 + 3.0 * x1);
        }
        ols.add(4.0);
        ols.add(5.0);
        let pred = ols.predict();
        // 2*4 + 3*5 = 23. Allow generous slack — RLS convergence.
        assert!((pred - 23.0).abs() < 3.0, "pred = {}", pred);
    }

    #[test]
    fn indirect_context_shift_and_repoint() {
        // 32-bit storage — no truncation on a 16-bit accumulation.
        let mut ic = IndirectContext::new(8, 8, 32);
        ic.set(5);
        ic.add(0xAB);
        ic.add(0xCD);
        assert_eq!(ic.get(), 0xABCD);
        ic.set(6);
        assert_eq!(ic.get(), 0); // fresh slot
        ic.set(5);
        assert_eq!(ic.get(), 0xABCD); // slot 5 preserved
    }

    #[test]
    fn indirect_context_u8_storage_truncates() {
        // 8-bit storage with 1-bit input — an 8-deep bit history.
        let mut ic = IndirectContext::new(19, 1, 8);
        ic.set(0);
        for _ in 0..16 { ic.add(1); }
        assert_eq!(ic.get(), 0xFF, "U8 storage truncates to 8 bits");
    }

    #[test]
    fn mtf_list_moves_to_front() {
        let mut m = MtfList::new(8);
        assert_eq!(m.get_first(), 0);
        m.move_to_front(5);
        assert_eq!(m.get_first(), 5);
        assert_eq!(m.get_next(), 0);
    }

    #[test]
    fn cache_ring_indexing() {
        let mut c: Cache<u32> = Cache::new(4);
        *c.next() = 10;
        *c.next() = 20;
        *c.next() = 30;
        assert_eq!(*c.at(0), 30);
        assert_eq!(*c.at(1), 20);
        assert_eq!(*c.at(2), 10);
    }
}
