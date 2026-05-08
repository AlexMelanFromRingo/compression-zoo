//! Concrete context-hash implementations — port of `contexts/*.{h,cpp}`.
//!
//! Upstream wires each context to externally-managed state via
//! `const unsigned long long&` references. Rust's borrow checker
//! makes that fragile across many concurrently-mutating contexts,
//! so each port instead holds its *own* state and the caller passes
//! the necessary byte / byte-context / recent-contexts on each
//! `update(...)` call. The `context()` / `size()` accessors mirror
//! upstream's `GetContext` / `Size`.
//!
//! The `Context` trait is intentionally minimal — concrete types
//! expose richer `update_with_*` methods because each one needs a
//! different argument shape.

#![allow(dead_code)]

pub trait Context {
    fn context(&self) -> u64;
    fn size(&self) -> u64;
}

// ---------------- BitContext ------------------------------------------

/// `(byte_context << 8) + bit_context` — one of the simplest mixers
/// of the byte-so-far with the byte-being-decoded.
#[derive(Clone, Default)]
pub struct BitContext {
    context: u64,
    size: u64,
}

impl BitContext {
    pub fn new(byte_context_size: u64) -> Self {
        Self { context: 0, size: 256 * byte_context_size }
    }
    pub fn update(&mut self, bit_context: u64, byte_context: u64) {
        self.context = (byte_context << 8) + bit_context;
    }
}
impl Context for BitContext {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- ContextHash -----------------------------------------

/// `context = (context * 2^hash_size + byte) mod size`. A rolling
/// `order`-byte hash with `hash_size` bits per byte.
#[derive(Clone)]
pub struct ContextHash {
    context: u64,
    size: u64,
    hash_size: u32,
}

impl ContextHash {
    pub fn new(order: u32, hash_size: u32) -> Self {
        let size = 1u64 << (hash_size * order);
        Self { context: 0, size, hash_size }
    }
    pub fn update(&mut self, byte: u64) {
        self.context = (self.context * (1u64 << self.hash_size) + byte) % self.size;
    }
}
impl Context for ContextHash {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- CombinedContext -------------------------------------

/// `(context2 << shift) + context1` — packs two contexts together,
/// with `shift` chosen as the smallest power of two ≥ context1_size.
#[derive(Clone, Default)]
pub struct CombinedContext {
    context: u64,
    size: u64,
    shift: u32,
}

impl CombinedContext {
    pub fn new(context1_size: u64, context2_size: u64) -> Self {
        let mut shift = 1u32;
        while (1u64 << shift) < context1_size { shift += 1; }
        Self { context: 0, size: context1_size * context2_size, shift }
    }
    pub fn update(&mut self, context1: u64, context2: u64) {
        self.context = (context2 << self.shift) + context1;
    }
}
impl Context for CombinedContext {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- Interval --------------------------------------------

/// Shifts a `map_[byte]` token into a sliding `num_bits`-wide
/// window. `map_` lets the caller bucket bytes into intervals
/// (e.g. lowercase / uppercase / digit / punct).
#[derive(Clone)]
pub struct Interval {
    context: u64,
    size: u64,
    map: Vec<i32>,
    mask: u64,
    shift: u32,
}

impl Interval {
    pub fn new(map: Vec<i32>, num_bits: u32) -> Self {
        let max_value = map.iter().copied().max().unwrap_or(0);
        let mut shift = 1u32;
        while (1i32 << shift) <= max_value { shift += 1; }
        let size = 1u64 << num_bits;
        Self { context: 0, size, map, mask: size - 1, shift }
    }
    pub fn update(&mut self, byte: usize) {
        self.context = self.mask
            & ((self.context << self.shift) + self.map[byte] as u64);
    }
}
impl Context for Interval {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- IntervalHash ----------------------------------------

/// `Interval` followed by a `ContextHash`-style rolling hash.
#[derive(Clone)]
pub struct IntervalHash {
    context: u64,
    size: u64,
    map: Vec<i32>,
    mask: u64,
    hash_size: u32,
    interval: u64,
    shift: u32,
}

impl IntervalHash {
    pub fn new(map: Vec<i32>, num_bits: u32, order: u32, hash_size: u32) -> Self {
        let max_value = map.iter().copied().max().unwrap_or(0);
        let mut shift = 1u32;
        while (1i32 << shift) <= max_value { shift += 1; }
        let mask = (1u64 << num_bits) - 1;
        let size = 1u64 << (hash_size * order);
        Self { context: 0, size, map, mask, hash_size, interval: 0, shift }
    }
    pub fn update(&mut self, byte: usize) {
        self.interval = self.mask
            & ((self.interval << self.shift) + self.map[byte] as u64);
        self.context = (self.context * (1u64 << self.hash_size) + self.interval)
            % self.size;
    }
}
impl Context for IntervalHash {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- IndirectHash ----------------------------------------

/// Two-level rolling hash: an `order1` outer hash indexes a small
/// table; the table entry feeds an `order2` inner hash.
#[derive(Clone)]
pub struct IndirectHash {
    context: u64,
    size: u64,
    context1: u64,
    hash_size1: u32,
    hash_size2: u32,
    size1: u64,
    hashes: Vec<u64>,
}

impl IndirectHash {
    pub fn new(order1: u32, hash_size1: u32, order2: u32, hash_size2: u32) -> Self {
        let size1 = 1u64 << (hash_size1 * order1);
        let size  = 1u64 << (hash_size2 * order2);
        Self {
            context: 0, size, context1: 0,
            hash_size1, hash_size2, size1,
            hashes: vec![0u64; size1 as usize],
        }
    }
    pub fn update(&mut self, byte: u64) {
        let idx = self.context1 as usize;
        self.hashes[idx] =
            (self.context * (1u64 << self.hash_size2) + byte) % self.size;
        self.context1 = (self.context1 * (1u64 << self.hash_size1) + byte) % self.size1;
        self.context = self.hashes[self.context1 as usize];
    }
}
impl Context for IndirectHash {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- Sparse ----------------------------------------------

/// Linear combination of recent context values. Upstream's
/// `recent_contexts` slot 0..N stores the last 0..N context-hash
/// values; this picks `orders` of them and sums with prime factors.
#[derive(Clone)]
pub struct Sparse {
    context: u64,
    size: u64,
    orders: Vec<u32>,
    factors: [u64; 6],
}

impl Sparse {
    pub fn new(orders: Vec<u32>) -> Self {
        let factors = [
            1, 256, 29 * 31, 29 * 31 * 37,
            29 * 31 * 37 * 41, 29 * 31 * 37 * 41 * 43,
        ];
        Self { context: 0, size: u64::MAX, orders, factors }
    }
    pub fn update(&mut self, recent_contexts: &[u64]) {
        let mut ctx = recent_contexts[self.orders[0] as usize];
        for i in 1..self.orders.len() {
            ctx = ctx.wrapping_add(
                self.factors[i.min(5)]
                    .wrapping_mul(recent_contexts[self.orders[i] as usize])
            );
        }
        self.context = ctx;
    }
}
impl Context for Sparse {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

// ---------------- BracketContext --------------------------------------

/// Tracks open/close brackets `(){}[]<>` to give a context based on
/// the most-recent unmatched opener and the distance back to it.
/// Useful for code/markup-heavy inputs.
#[derive(Clone)]
pub struct BracketContext {
    context: u64,
    size: u64,
    distance_limit: u32,
    stack_limit: u32,
    active: Vec<u32>,
    distance: Vec<u32>,
}

impl BracketContext {
    pub fn new(distance_limit: u32, stack_limit: u32) -> Self {
        Self {
            context: 0,
            size: 257 * distance_limit as u64,
            distance_limit,
            stack_limit,
            active: Vec::new(),
            distance: Vec::new(),
        }
    }

    /// Lookup table for matching-close character. Mirrors upstream
    /// `{('(',')'), ('{','}'), ('[',']'), ('<','>')}`.
    fn close_for(byte: u32) -> Option<u32> {
        match byte as u8 {
            b'(' => Some(b')' as u32),
            b'{' => Some(b'}' as u32),
            b'[' => Some(b']' as u32),
            b'<' => Some(b'>' as u32),
            _    => None,
        }
    }

    /// Same lookup but checks if `byte` matches the stack top (i.e.
    /// `byte == close_for(top)`).
    fn matches_open(top: u32, byte: u32) -> bool {
        Self::close_for(top) == Some(byte)
    }

    pub fn update(&mut self, byte: u32) {
        if let (Some(&top), Some(&dist_top)) = (self.active.last(), self.distance.last()) {
            if Self::matches_open(top, byte) || dist_top >= self.distance_limit - 1 {
                self.active.pop();
                self.distance.pop();
            } else {
                let n = self.distance.len();
                self.distance[n - 1] += 1;
            }
        }
        if Self::close_for(byte).is_some() {
            self.active.push(byte);
            self.distance.push(0);
            if self.active.len() as u32 > self.stack_limit {
                self.active.remove(0);
                self.distance.remove(0);
            }
        }
        self.context = if let (Some(&top), Some(&dist)) =
            (self.active.last(), self.distance.last())
        {
            self.distance_limit as u64 * (top as u64 + 1) + dist as u64
        } else {
            0
        };
    }
}
impl Context for BracketContext {
    fn context(&self) -> u64 { self.context }
    fn size(&self) -> u64 { self.size }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_context_packs_byte_and_bit() {
        let mut bc = BitContext::new(256);
        bc.update(0xAB, 0x12);
        assert_eq!(bc.context(), (0x12 << 8) + 0xAB);
    }

    #[test]
    fn context_hash_advances_modulo_size() {
        let mut ch = ContextHash::new(2, 8); // size = 1 << 16
        for &b in &[1u64, 2, 3, 4] {
            ch.update(b);
        }
        assert_eq!(ch.size(), 1 << 16);
        // 4-byte rolling hash with 8-bit shift: ((((1*256)+2)*256+3)*256+4) mod 65536.
        let want = ((((1u64 * 256) + 2) * 256 + 3) * 256 + 4) % 65536;
        assert_eq!(ch.context(), want);
    }

    #[test]
    fn combined_context_packs() {
        let mut cc = CombinedContext::new(256, 256); // shift = 8
        cc.update(0x42, 0x33);
        assert_eq!(cc.context(), (0x33 << 8) + 0x42);
    }

    #[test]
    fn interval_window_slides() {
        // map: every byte → its low 4 bits.
        let map: Vec<i32> = (0..256).map(|i| (i & 0x0F) as i32).collect();
        let mut iv = Interval::new(map, 16);
        iv.update(0xAB);
        iv.update(0xCD);
        // shift=4 (max value 15 = 1111). After two updates:
        //   ctx = ((0 << 4) | 0xB) << 4 | 0xD = 0xBD.
        assert_eq!(iv.context(), 0xBD);
    }

    #[test]
    fn bracket_tracks_open_close() {
        let mut b = BracketContext::new(8, 4);
        b.update(b'(' as u32);
        let after_open = b.context();
        b.update(b')' as u32);
        // After matched close, the stack empties → context resets to 0.
        assert_eq!(b.context(), 0);
        assert!(after_open != 0);
    }

    #[test]
    fn indirect_hash_sane_size() {
        let h = IndirectHash::new(2, 8, 3, 8);
        assert_eq!(h.size(), 1 << 24);
    }

    #[test]
    fn sparse_combines_recent_contexts() {
        let mut s = Sparse::new(vec![0, 1, 2]);
        let recent = [10u64, 20, 30, 40, 50];
        s.update(&recent);
        // ctx = 10 + 256*20 + 29*31*30
        let want = 10u64.wrapping_add(256u64.wrapping_mul(20))
            .wrapping_add((29u64 * 31).wrapping_mul(30));
        assert_eq!(s.context(), want);
    }
}
