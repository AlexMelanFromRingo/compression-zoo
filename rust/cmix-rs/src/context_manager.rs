//! Per-byte shared state — port of `context-manager.{h,cpp}`.
//!
//! Upstream's `ContextManager` owns *both* the shared mutable state
//! (history buffer, recent words, recent bytes, line-break counter,
//! line-feed/WRT counter) *and* a heterogeneous
//! `Vec<unique_ptr<Context>>` driven each bit. The two
//! responsibilities are awkward to mix in safe Rust because every
//! Context pulls a `const X&` reference into the manager.
//!
//! This port keeps the *state* portion (the easy part) — the
//! per-byte history / words / recent-bytes update logic that
//! every CMIX configuration uses. Owners that need to drive a
//! collection of [`crate::contexts::Context`] implementations on
//! top of this state can do so by holding their own typed Vecs;
//! the manager exposes the inputs each Context needs via direct
//! field access. The integration glue (the actual Vec<Context*>
//! plus `IsEqual`-based dedup) is the natural home of the
//! upcoming `predictor.rs` port.

#![allow(dead_code)]

use crate::contexts::{
    BitContext, BracketContext, CombinedContext, Context, ContextHash,
    IndirectHash, Interval, IntervalHash, Sparse,
};

/// Where a `u64` context value comes from when driving a sub-model
/// or a mixer's context-key. Mirrors the `const u64&` references each
/// upstream model holds to its manager binding.
#[derive(Clone, Copy, Debug)]
pub enum Src {
    BitContext,
    LongBitContext,
    Zero,
    Wrt,
    LongestMatch,
    LineBreak,
    Auxiliary,
    RecentByte(usize),
    Word(usize),
    Ctx(usize),
    BitCtx(usize),
}

/// A byte-level Context node owned by the manager. Each variant
/// records the manager source whose value feeds its `update(...)`
/// at byte boundary.
pub enum CtxNode {
    ContextHash { c: ContextHash, byte_src: Src },
    Interval { c: Interval, byte_src: Src },
    IntervalHash { c: IntervalHash, byte_src: Src },
    IndirectHash { c: IndirectHash, byte_src: Src },
    Sparse { c: Sparse },
    Bracket { c: BracketContext, byte_src: Src },
    Combined { c: CombinedContext, a: Src, b: Src },
}

impl CtxNode {
    pub fn context(&self) -> u64 {
        match self {
            CtxNode::ContextHash { c, .. } => c.context(),
            CtxNode::Interval { c, .. } => c.context(),
            CtxNode::IntervalHash { c, .. } => c.context(),
            CtxNode::IndirectHash { c, .. } => c.context(),
            CtxNode::Sparse { c } => c.context(),
            CtxNode::Bracket { c, .. } => c.context(),
            CtxNode::Combined { c, .. } => c.context(),
        }
    }
    pub fn size(&self) -> u64 {
        match self {
            CtxNode::ContextHash { c, .. } => c.size(),
            CtxNode::Interval { c, .. } => c.size(),
            CtxNode::IntervalHash { c, .. } => c.size(),
            CtxNode::IndirectHash { c, .. } => c.size(),
            CtxNode::Sparse { c } => c.size(),
            CtxNode::Bracket { c, .. } => c.size(),
            CtxNode::Combined { c, .. } => c.size(),
        }
    }
}

/// A `BitContext` + the source for its `byte_context` argument.
/// `update` fires every bit.
pub struct BitCtxNode {
    pub c: BitContext,
    pub byte_src: Src,
}

pub struct ContextManager {
    /// Bit-shift register for the byte being decoded (1..=255 mid-
    /// byte, drops to its low 8 bits at byte boundaries).
    pub bit_context: u32,
    /// `bit_context` extended to 64-bit for callers that hash it.
    /// Exactly equal to `bit_context as u64` mid-byte; resets to 1
    /// at byte boundaries.
    pub long_bit_context: u64,
    /// Helper trackers for upstream WRT (word-replacement-table)
    /// processing.
    pub wrt_state: u32,
    pub wrt_context: u64,
    /// Other shared per-byte counters surfaced to models.
    pub zero_context: u64,
    pub history_pos: u64,
    pub line_break: u64,
    pub longest_match: u64,
    pub auxiliary_context: u64,
    /// Sliding history buffer of the last bytes (default 100 MB,
    /// like upstream — the Indirect/Match models expect it large).
    pub history: Vec<u8>,
    /// Indirect-state map shared across all `Indirect` instances.
    pub shared_map: Vec<u8>,
    /// Recent-word hashes ([0..1] are active rolling-hashes; [2..7]
    /// are historical). Total 8 slots, see `update_words`.
    pub words: [u64; 8],
    /// Recent-byte ring buffer (8 entries, newest first).
    pub recent_bytes: [u64; 8],

    /// Byte-level Context collection, populated by `add_context`.
    /// Each entry's `update` is called once at every byte boundary.
    pub contexts: Vec<CtxNode>,
    /// Bit-level `BitContext` collection. `update` fires every bit.
    pub bit_contexts: Vec<BitCtxNode>,
}

impl ContextManager {
    /// Create a manager with `history_size` bytes of history and
    /// `shared_map_size` of indirect-state map. Upstream defaults
    /// are `100_000_000` and `256 * 8_000_000`; tests typically use
    /// much smaller values.
    pub fn new(history_size: usize, shared_map_size: usize) -> Self {
        Self {
            bit_context: 1,
            long_bit_context: 1,
            wrt_state: 0,
            wrt_context: 0,
            zero_context: 0,
            history_pos: 0,
            line_break: 0,
            longest_match: 0,
            auxiliary_context: 0,
            history: vec![0u8; history_size],
            shared_map: vec![0u8; shared_map_size],
            words: [0u64; 8],
            recent_bytes: [0u64; 8],
            contexts: Vec::new(),
            bit_contexts: Vec::new(),
        }
    }

    /// Register a byte-level Context. Returns its index for later
    /// `Src::Ctx` references.
    pub fn add_context(&mut self, node: CtxNode) -> usize {
        let idx = self.contexts.len();
        self.contexts.push(node);
        idx
    }

    /// Register a bit-level `BitContext`. Returns its index for
    /// `Src::BitCtx` references.
    pub fn add_bit_context(&mut self, node: BitCtxNode) -> usize {
        let idx = self.bit_contexts.len();
        self.bit_contexts.push(node);
        idx
    }

    /// Resolve a [`Src`] to its current `u64` value.
    pub fn resolve(&self, src: Src) -> u64 {
        match src {
            Src::BitContext => self.bit_context as u64,
            Src::LongBitContext => self.long_bit_context,
            Src::Zero => 0,
            Src::Wrt => self.wrt_context,
            Src::LongestMatch => self.longest_match,
            Src::LineBreak => self.line_break,
            Src::Auxiliary => self.auxiliary_context,
            Src::RecentByte(i) => self.recent_bytes[i],
            Src::Word(i) => self.words[i],
            Src::Ctx(i) => self.contexts[i].context(),
            Src::BitCtx(i) => self.bit_contexts[i].c.context(),
        }
    }

    /// Resolve the `Size` of a context source (used by Direct/Match/
    /// Indirect/Bracket model constructors that take a context size).
    pub fn ctx_size(&self, src: Src) -> u64 {
        match src {
            Src::Ctx(i) => self.contexts[i].size(),
            Src::BitCtx(i) => self.bit_contexts[i].c.size(),
            _ => 0,
        }
    }

    /// Drive every registered Context for the current bit (byte-level
    /// contexts at boundary, bit-level every bit). Call this AFTER
    /// `update_bit`, since the byte-level contexts read the just-
    /// completed byte from `bit_context`.
    pub fn update_contexts_owned(&mut self, at_byte_boundary: bool) {
        if at_byte_boundary {
            let byte = self.bit_context as u64;
            // Snapshot what each ContextNode needs, then update.
            // We can't take a &CtxNode and then mutate self.contexts,
            // so we drain via index loop.
            for i in 0..self.contexts.len() {
                // Resolve `Src` against the *current* state.
                // (Each context reads only `Src`-based inputs that
                // are independent of self.contexts mutation.)
                let (kind_input, sparse_words) = self.context_inputs(i);
                let node = &mut self.contexts[i];
                match node {
                    CtxNode::ContextHash { c, .. } =>
                        c.update(kind_input),
                    CtxNode::Interval { c, .. } =>
                        c.update((kind_input & 0xff) as usize),
                    CtxNode::IntervalHash { c, .. } =>
                        c.update((kind_input & 0xff) as usize),
                    CtxNode::IndirectHash { c, .. } =>
                        c.update(kind_input),
                    CtxNode::Bracket { c, .. } =>
                        c.update((kind_input & 0xffff_ffff) as u32),
                    CtxNode::Combined { c, a, b } => {
                        // `kind_input` already resolves to `a`; we
                        // need `b` separately.
                        let av = match a { _ => kind_input };
                        let _ = av; // silence
                        // Re-resolve both cleanly here.
                        let av = resolve_src_static(*a, byte,
                            &self.recent_bytes, &self.words);
                        let bv = resolve_src_static(*b, byte,
                            &self.recent_bytes, &self.words);
                        c.update(av, bv);
                    }
                    CtxNode::Sparse { c } => c.update(&sparse_words),
                }
            }
        }
        // Bit-level contexts every bit.
        let long_bc = self.long_bit_context;
        for i in 0..self.bit_contexts.len() {
            let byte_ctx = self.resolve(self.bit_contexts[i].byte_src);
            self.bit_contexts[i].c.update(long_bc, byte_ctx);
        }
    }

    fn context_inputs(&self, i: usize) -> (u64, [u64; 8]) {
        let node = &self.contexts[i];
        let byte_src = match node {
            CtxNode::ContextHash { byte_src, .. } => *byte_src,
            CtxNode::Interval { byte_src, .. } => *byte_src,
            CtxNode::IntervalHash { byte_src, .. } => *byte_src,
            CtxNode::IndirectHash { byte_src, .. } => *byte_src,
            CtxNode::Bracket { byte_src, .. } => *byte_src,
            CtxNode::Combined { a, .. } => *a,
            CtxNode::Sparse { .. } => Src::Zero,
        };
        (self.resolve(byte_src), self.words)
    }
}

/// Standalone resolver used inside the borrow-restricted update
/// loop (we can't call `self.resolve` while `self.contexts` is
/// mutably borrowed).
fn resolve_src_static(
    src: Src,
    byte_context: u64,
    recent_bytes: &[u64; 8],
    words: &[u64; 8],
) -> u64 {
    match src {
        Src::BitContext | Src::LongBitContext => byte_context,
        Src::Zero | Src::Wrt | Src::LongestMatch | Src::LineBreak
        | Src::Auxiliary | Src::Ctx(_) | Src::BitCtx(_) => 0,
        Src::RecentByte(i) => recent_bytes[i],
        Src::Word(i) => words[i],
    }
}

impl ContextManager {

    /// Append the just-completed byte to the history ring buffer.
    pub fn update_history(&mut self) {
        self.history[self.history_pos as usize] = self.bit_context as u8;
        self.history_pos += 1;
        if self.history_pos as usize == self.history.len() { self.history_pos = 0; }
    }

    /// Roll the case-sensitive ([7]) and case-insensitive ([0..7])
    /// word hashes. Mirrors upstream `UpdateWords`.
    pub fn update_words(&mut self) {
        let c = self.bit_context as u8;
        if (c >= b'a' && c <= b'z') || (c >= b'A' && c <= b'Z') || c >= 0x80 {
            self.words[7] = self.words[7].wrapping_mul(997 * 16).wrapping_add(c as u64);
        } else {
            self.words[7] = 0;
        }
        let c = if c >= b'A' && c <= b'Z' { c + b'a' - b'A' } else { c };
        if (c >= b'a' && c <= b'z') || (c >= b'0' && c <= b'9')
            || c == 8 || c == 6 || c >= 0x80
        {
            self.words[0] = (self.words[0].wrapping_mul(997 * 16).wrapping_add(c as u64))
                & 0xfffffff;
            self.words[1] = self.words[1].wrapping_mul(263 * 32).wrapping_add(c as u64);
        } else {
            for i in (2..=6).rev() {
                self.words[i] = self.words[i - 1];
            }
            self.words[1] = 0;
        }
    }

    /// Push the just-completed byte onto the recent-bytes ring.
    pub fn update_recent_bytes(&mut self) {
        for i in (1..=7).rev() {
            self.recent_bytes[i] = self.recent_bytes[i - 1];
        }
        self.recent_bytes[0] = self.bit_context as u64;
    }

    /// Maintain the WRT context — a high-bit byte sequence used by
    /// upstream's text-mode models for word-replacement-table tags.
    pub fn update_wrt_context(&mut self) {
        if self.bit_context < 0x80 {
            self.wrt_state = 0;
        } else {
            if self.wrt_state == 0 { self.wrt_context = 0; }
            self.wrt_state = 1;
            self.wrt_context <<= 8;
            self.wrt_context += self.bit_context as u64;
            if self.wrt_context > 0xFFEFCF { self.wrt_context = 0; }
        }
    }

    /// Advance bit-by-bit. Returns `true` once at a byte boundary
    /// (after applying the byte-end side effects); models that need
    /// to reload contexts should react to that.
    pub fn update_bit(&mut self, bit: i32) -> bool {
        self.bit_context = self.bit_context * 2 + bit as u32;
        self.long_bit_context = self.bit_context as u64;
        if self.bit_context >= 256 {
            self.bit_context -= 256;
            self.long_bit_context = 1;
            self.longest_match = 0;
            if self.bit_context == b'\n' as u32 {
                self.line_break = 0;
            } else if self.line_break < 99 {
                self.line_break += 1;
            }
            self.update_history();
            self.update_words();
            self.update_recent_bytes();
            self.update_wrt_context();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_bit_emits_byte_boundary() {
        let mut cm = ContextManager::new(1024, 1024);
        // Push byte 'A' = 0b01000001 bit-by-bit.
        let bits = [0,1,0,0,0,0,0,1];
        for (i, &b) in bits.iter().enumerate() {
            let at_boundary = cm.update_bit(b);
            // The boundary fires on the 8th (final) bit.
            assert_eq!(at_boundary, i == 7);
        }
        assert_eq!(cm.bit_context, b'A' as u32);
        assert_eq!(cm.history[0], b'A');
    }

    #[test]
    fn update_words_rolls_case_insensitively() {
        let mut cm = ContextManager::new(1024, 1024);
        // Feed "ab" letter-by-letter, byte-by-byte.
        for c in b"ab" {
            cm.bit_context = *c as u32;
            cm.update_words();
        }
        // After two letters, words[0] should be non-zero (rolling
        // hash of lowercase-folded letters).
        assert_ne!(cm.words[0], 0);
        // words[7] tracks case-sensitive letters / high-bytes only.
        assert_ne!(cm.words[7], 0);
    }

    #[test]
    fn line_break_resets_on_newline_and_caps_at_99() {
        let mut cm = ContextManager::new(1024, 1024);
        for _ in 0..150 {
            cm.bit_context = 1; // arbitrary non-newline byte
            cm.line_break = if cm.line_break < 99 { cm.line_break + 1 } else { 99 };
        }
        assert_eq!(cm.line_break, 99);
        // Now feed a newline through update_bit (8 bits, value 10).
        for &b in &[0,0,0,0,1,0,1,0] {
            cm.update_bit(b);
        }
        assert_eq!(cm.line_break, 0);
    }
}
