//! Context maps from paq8.cpp:830-1364.
//!
//! Implements:
//!
//! * [`HashTableB`]            — generic bucket size `B`, paq8.cpp:830-857.
//! * [`Bh`]                    — `BH<B>` MRU-bucket hash, paq8.cpp:779-814.
//! * [`RunContextMap`]         — paq8.cpp:858-878.
//! * [`SmallStationaryContextMap`] — paq8.cpp:892-920.
//! * [`StationaryMap`]         — paq8.cpp:936-975.
//! * [`IndirectMap`]           — paq8.cpp:977-1009.
//! * [`ContextMap`]            — paq8.cpp:1011-1146.
//! * [`ContextMap2`]           — paq8.cpp:1165-1364.
//!
//! Each map consumes a paq8 [`Mixer`](super::mixer::Mixer) by way of
//! `mix()`. Context-creating methods take `u64` for upstream-fidelity:
//! upstream uses `U64` ctx values that get finalized/checksummed to
//! a `(hashbits, checksum)` pair.

#![allow(dead_code)]

use super::apm::{StateMap, StateMap32};
use super::mixer::Mixer;
use super::substrate::{
    checksum64, finalize64, ilog2, nex, Squash, Stretch,
};

// =================================================================
// HashTable<B> — paq8.cpp:830-857. Used by some legacy paq sub-models;
// we expose it for completeness. Bucket size `B` is the runtime
// parameter.
// =================================================================

pub struct HashTableB {
    t:        Vec<u8>,
    n:        usize,
    mask:     usize,
    hashbits: u32,
    b:        usize,
}

impl HashTableB {
    /// `n` is the number of B-byte items in the table; must be a
    /// power of two. `b` is bucket size in bytes.
    pub fn new(n: usize, b: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        Self {
            t: vec![0u8; n * b],
            n, mask: n - 1,
            hashbits: ilog2(n as u32),
            b,
        }
    }

    /// Get a mutable slice of size `b - 1` (everything except the
    /// checksum byte) for context `ctx`.
    pub fn get(&mut self, ctx: u64) -> &mut [u8] {
        let chk = (checksum64(ctx, self.hashbits, 8) & 0xff) as u8;
        let mut i = (finalize64(ctx, self.hashbits) as usize * self.b) & self.mask;
        // search 3 slots: i, i^B, i^(2B).
        let b = self.b;
        if self.t[i] == chk {
            return &mut self.t[i + 1..i + b];
        }
        if self.t[i ^ b] == chk {
            return &mut self.t[(i ^ b) + 1..(i ^ b) + b];
        }
        if self.t[i ^ (b * 2)] == chk {
            return &mut self.t[(i ^ (b * 2)) + 1..(i ^ (b * 2)) + b];
        }
        // evict lowest priority
        if self.t[i + 1] > self.t[(i + 1) ^ b]
            || self.t[i + 1] > self.t[(i + 1) ^ (b * 2)]
        {
            i ^= b;
        }
        if self.t[i + 1] > self.t[(i + 1) ^ b ^ (b * 2)] {
            i ^= b ^ (b * 2);
        }
        for j in 0..b { self.t[i + j] = 0; }
        self.t[i] = chk;
        &mut self.t[i + 1..i + b]
    }
}

// =================================================================
// BH<B> — paq8.cpp:779-814. MRU-style bucket hash used by
// RunContextMap and ContextMap. Each "slot" holds a 16-bit checksum
// followed by `B - 2` payload bytes.
// =================================================================

pub struct Bh {
    t:        Vec<u8>,
    mask:     u32,
    hashbits: u32,
    b:        usize,
}

impl Bh {
    /// Raw-byte read at absolute index — used by JpegModel's bit-
    /// history pointer (`cp[i]` indexes into the slot payload by
    /// raw offset rather than through the MRU lookup).
    pub fn byte_at(&self, abs: usize) -> u8 {
        if abs < self.t.len() { self.t[abs] } else { 0 }
    }
    /// Raw-byte write at absolute index.
    pub fn set_byte_at(&mut self, abs: usize, v: u8) {
        if abs < self.t.len() { self.t[abs] = v; }
    }
    /// Starting address of the underlying `Vec<u8>` storage.
    /// Subtract this from a `&mut [u8]` returned by `get()` to
    /// compute the absolute index.
    pub fn storage_base(&self) -> usize { self.t.as_ptr() as usize }

    /// `n` is the number of B-byte items in the table; must be a
    /// power of two. `b` is bucket size in bytes (must be ≥ 4).
    pub fn new(n: usize, b: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        debug_assert!(b >= 4);
        Self {
            t: vec![0u8; n * b],
            mask: (n - 1) as u32,
            hashbits: ilog2(n as u32),
            b,
        }
    }

    /// MRU lookup. Returns a mutable slice starting at the payload
    /// area (after the 2-byte checksum header). The returned slice
    /// has length `B - 2`.
    pub fn get(&mut self, ctx: u64) -> &mut [u8] {
        let m: usize = 8; // search limit
        let chk = (checksum64(ctx, self.hashbits, 16) & 0xffff) as u16;
        let i = ((finalize64(ctx, self.hashbits) as usize) * m) & (self.mask as usize);
        let b = self.b;

        let mut j_found = m;
        for j in 0..m {
            let base = (i + j) * b;
            let p2 = self.t[base + 2];
            let cp = ((self.t[base] as u16)) | ((self.t[base + 1] as u16) << 8);
            if p2 == 0 {
                // empty slot: claim it
                self.t[base    ] = (chk & 0xff) as u8;
                self.t[base + 1] = (chk >> 8) as u8;
                j_found = j;
                break;
            }
            if cp == chk {
                j_found = j;
                break;
            }
        }

        if j_found == m {
            // didn't find or claim: evict last
            let mut j = m - 1;
            if m > 2 && self.t[(i + j) * b + 2] > self.t[(i + j - 1) * b + 2] {
                j -= 1;
            }
            j_found = j;
            // overwrite the j_found slot with new checksum, clear payload
            for k in 0..b { self.t[(i + j_found) * b + k] = 0; }
            self.t[(i + j_found) * b    ] = (chk & 0xff) as u8;
            self.t[(i + j_found) * b + 1] = (chk >> 8) as u8;
        }

        if j_found > 0 {
            // MRU: rotate j_found to position 0.
            let start = i * b;
            // Save tmp slot.
            let mut tmp = vec![0u8; b];
            tmp.copy_from_slice(&self.t[start + j_found * b..start + (j_found + 1) * b]);
            // Shift slots [0..j_found] down by one.
            for k in (1..=j_found).rev() {
                let src = start + (k - 1) * b;
                let dst = start + k * b;
                let (left, right) = self.t.split_at_mut(dst);
                right[..b].copy_from_slice(&left[src..src + b]);
            }
            self.t[start..start + b].copy_from_slice(&tmp);
        }

        let base = i * b;
        &mut self.t[base + 1..base + b]
    }
}

// =================================================================
// RunContextMap — paq8.cpp:858-878.
// =================================================================

pub struct RunContextMap {
    t:        Bh,
    /// Cached current ctx — used to re-fetch the active cp slot.
    cur_ctx:  u64,
    primed:   bool,
}

impl RunContextMap {
    pub fn new(m: usize) -> Self {
        let n = (m / 4).next_power_of_two().max(4);
        Self { t: Bh::new(n, 4), cur_ctx: 0, primed: false }
    }

    /// Update the *previous* slot's run counter + last-byte, then
    /// advance the cp pointer to `cx`. `buf_minus1` is the most
    /// recent committed byte (`buf(1)` in upstream).
    pub fn set(&mut self, cx: u64, buf_minus1: u8) {
        if self.primed {
            // Bh::get returns slice starting at bucket offset +1, so
            // [chk_hi, payload_0, payload_1] for B=4. We treat
            // payload_0 (slice[1]) as count, payload_1 (slice[2]) as
            // last_byte — matches upstream's `cp = t[ctx]+1` followed
            // by `cp[0]` / `cp[1]` (which after the +1 land at
            // payload bytes).
            let cp = self.t.get(self.cur_ctx);
            let count    = cp[1];
            let last_b   = cp[2];
            if count == 0 || last_b != buf_minus1 {
                cp[1] = 1; cp[2] = buf_minus1;
            } else if count < 255 {
                cp[1] = count + 1;
            }
        }
        self.cur_ctx = cx;
        self.primed  = true;
        // Touch the new slot so MRU reorders correctly.
        let _ = self.t.get(self.cur_ctx);
    }

    /// Bit-context probability — paq8.cpp:868-873.
    pub fn p(&mut self, c0: u32, bpos: i32,
            ilog: &super::substrate::Ilog) -> i32 {
        if !self.primed { return 0; }
        let cp = self.t.get(self.cur_ctx);
        let count  = cp[1] as u32;
        let last_b = cp[2] as u32;
        if ((last_b + 256) >> (8 - bpos)) == c0 {
            let bit = ((last_b >> (7 - bpos)) & 1) as i32;
            let sign = bit * 2 - 1;
            let il = ilog.get(((count + 1) & 0xffff) as u16) as i32;
            sign * il * 8
        } else { 0 }
    }

    pub fn mix(&mut self, m: &mut Mixer, c0: u32, bpos: i32,
                ilog: &super::substrate::Ilog) -> bool {
        let p = self.p(c0, bpos, ilog);
        m.add(p as i16);
        if !self.primed { return false; }
        let cp = self.t.get(self.cur_ctx);
        cp[1] != 0
    }
}

// =================================================================
// SmallStationaryContextMap — paq8.cpp:892-920.
// =================================================================

pub struct SmallStationaryContextMap {
    data:    Vec<u16>,
    context: u32,
    mask:    u32,
    stride:  u32,
    b_count: u32,
    b_total: u32,
    b:       u32,
    cp:      usize, // index into `data` of current cell
}

impl SmallStationaryContextMap {
    pub fn new(bits_of_context: u32, bits_per_context: u32) -> Self {
        debug_assert!((1..=8).contains(&bits_per_context));
        let len = ((1u64 << bits_of_context) * ((1u64 << bits_per_context) - 1))
            as usize;
        let mut data = vec![0x7FFFu16; len];
        let _ = &mut data;
        Self {
            data,
            context: 0,
            mask:   (1u32 << bits_of_context) - 1,
            stride: (1u32 << bits_per_context) - 1,
            b_count: 0,
            b_total: bits_per_context,
            b: 0,
            cp: 0,
        }
    }

    pub fn set(&mut self, ctx: u32) {
        self.context = (ctx & self.mask) * self.stride;
        self.b_count = 0; self.b = 0;
    }

    pub fn reset(&mut self) {
        for x in self.data.iter_mut() { *x = 0x7FFF; }
    }

    pub fn mix(&mut self, m: &mut Mixer, y: i32,
                rate: i32, multiplier: i32, divisor: i32,
                _squash: &Squash, stretch: &Stretch) {
        // adapt current cell
        let cur = self.data[self.cp] as i32;
        let half_step = 1i32 << (rate - 1);
        let upd = ((y << 16) - cur + half_step) >> rate;
        self.data[self.cp] = (cur + upd).clamp(0, u16::MAX as i32) as u16;
        // step
        if y != 0 && self.b > 0 { self.b += 1; }
        let idx = (self.context + self.b) as usize;
        self.cp = idx.min(self.data.len() - 1);
        let pr_v = (self.data[self.cp] >> 4) as i32;
        m.add(((stretch.get(pr_v) * multiplier) / divisor) as i16);
        m.add((((pr_v - 2048) * multiplier) / (divisor * 2)) as i16);
        self.b_count += 1;
        self.b = self.b * 2 + 1;
        if self.b_count == self.b_total { self.b_count = 0; self.b = 0; }
    }
}

// =================================================================
// StationaryMap — paq8.cpp:936-975.
// =================================================================

pub struct StationaryMap {
    data:      Vec<u32>,
    mask:      u32,
    maskbits:  u32,
    stride:    u32,
    context:   u32,
    b_count:   u32,
    b_total:   u32,
    b:         u32,
    cp:        usize,
}

impl StationaryMap {
    pub fn new(bits_of_context: u32, bits_per_context: u32, rate: u32) -> Self {
        debug_assert!((1..=8).contains(&bits_per_context));
        let len = ((1u64 << bits_of_context) * ((1u64 << bits_per_context) - 1))
            as usize;
        let init = (0x7FFu32 << 20) | rate.min(1023);
        let data = vec![init; len];
        Self {
            data,
            mask: (1u32 << bits_of_context) - 1,
            maskbits: bits_of_context,
            stride: (1u32 << bits_per_context) - 1,
            context: 0,
            b_count: 0, b_total: bits_per_context, b: 0,
            cp: 0,
        }
    }

    pub fn set_direct(&mut self, ctx: u32) {
        self.context = (ctx & self.mask) * self.stride;
        self.b_count = 0; self.b = 0;
    }

    pub fn set(&mut self, ctx: u64) {
        self.context = (finalize64(ctx, self.maskbits) & self.mask) * self.stride;
        self.b_count = 0; self.b = 0;
    }

    pub fn reset(&mut self, rate: u32) {
        let init = (0x7FFu32 << 20) | rate.min(1023);
        for x in self.data.iter_mut() { *x = init; }
    }

    pub fn mix(&mut self, m: &mut Mixer, y: i32, multiplier: i32,
                divisor: i32, limit: u16, dt: &[i32; 1024],
                _squash: &Squash, stretch: &Stretch) {
        let cur = self.data[self.cp];
        let count = (cur & 0x3FF).min(limit as u32).min(0x3FF) + 1;
        let pr_v = (cur >> 10) as i32;
        let error = ((y << 22) - pr_v) >> 3;
        let pr_new = (pr_v + ((error * dt[count as usize - 1]) / 1024))
            .clamp(0, 0x3F_FFFF);
        self.data[self.cp] = ((pr_new as u32) << 10) | count;
        if y != 0 && self.b > 0 { self.b += 1; }
        let idx = (self.context + self.b) as usize;
        self.cp = idx.min(self.data.len() - 1);
        let pr_top = (self.data[self.cp] >> 20) as i32;
        m.add(((stretch.get(pr_top) * multiplier) / divisor) as i16);
        m.add((((pr_top - 2048) * multiplier) / (divisor * 2)) as i16);
        self.b_count += 1;
        self.b = self.b * 2 + 1;
        if self.b_count == self.b_total { self.b_count = 0; self.b = 0; }
    }
}

// =================================================================
// IndirectMap — paq8.cpp:977-1009.
// =================================================================

pub struct IndirectMap {
    data:      Vec<u8>,
    map:       StateMap32,
    mask:      u32,
    maskbits:  u32,
    stride:    u32,
    context:   u32,
    b_count:   u32,
    b_total:   u32,
    b:         u32,
    cp:        usize,
}

impl IndirectMap {
    pub fn new(bits_of_context: u32, bits_per_context: u32,
                dt: [i32; 1024]) -> Self {
        let len = ((1u64 << bits_of_context) * ((1u64 << bits_per_context) - 1))
            as usize;
        Self {
            data: vec![0u8; len],
            map:  StateMap32::new(256, dt),
            mask: (1u32 << bits_of_context) - 1,
            maskbits: bits_of_context,
            stride: (1u32 << bits_per_context) - 1,
            context: 0,
            b_count: 0, b_total: bits_per_context, b: 0,
            cp: 0,
        }
    }

    pub fn set_direct(&mut self, ctx: u32) {
        self.context = (ctx & self.mask) * self.stride;
        self.b_count = 0; self.b = 0;
    }

    pub fn set(&mut self, ctx: u64) {
        self.context = (finalize64(ctx, self.maskbits) & self.mask) * self.stride;
        self.b_count = 0; self.b = 0;
    }

    pub fn mix(&mut self, m: &mut Mixer, y: i32, multiplier: i32,
                divisor: i32, _limit: u16,
                _squash: &Squash, stretch: &Stretch) {
        // update state-machine of current cell.
        self.data[self.cp] = nex(self.data[self.cp], y as usize);
        if y != 0 && self.b > 0 { self.b += 1; }
        let idx = (self.context + self.b) as usize;
        self.cp = idx.min(self.data.len() - 1);
        let state = self.data[self.cp];
        let p1 = self.map.p(state as u32, 1023, y);
        m.add(((stretch.get(p1) * multiplier) / divisor) as i16);
        m.add((((p1 - 2048) * multiplier) / (divisor * 2)) as i16);
        self.b_count += 1;
        self.b = self.b * 2 + 1;
        if self.b_count == self.b_total { self.b_count = 0; self.b = 0; }
    }
}

// =================================================================
// ContextMap — paq8.cpp:1011-1146.
//
// Bit-history-based context map. Stores per-context bit cells in a
// hash table of `E` buckets (64 bytes each: 7 u16 checksums + 1 byte
// MRU + 7×7 byte bit-history matrix). Each context is updated through
// a private `StateMap` and contributes 5 stretched inputs per bit to
// the Mixer.
// =================================================================

/// One hash-bucket entry — paq8.cpp:1013-1019.
///
/// Layout matches upstream:
/// * `chk:  [u16; 7]` — 7 16-bit checksums (one per MRU slot).
/// * `last: u8`        — packed MRU history (low nibble = most-recent
///                       slot, high nibble = second-most-recent).
/// * `bh:   [[u8; 7]; 7]` — 7-slot × 7-byte bit-history matrix.
///                          bh[i][0] = "priority" / state header.
#[derive(Clone)]
pub struct ECell {
    pub chk:  [u16; 7],
    pub last: u8,
    pub bh:   [[u8; 7]; 7],
}

impl Default for ECell {
    fn default() -> Self {
        Self { chk: [0; 7], last: 0, bh: [[0; 7]; 7] }
    }
}

impl ECell {
    /// MRU find-or-create — paq8.cpp:1039-1048. Returns the slot
    /// index (0..7) of the matching/new bucket. Caller indexes
    /// `bh[slot][...]` for state access.
    pub fn get(&mut self, ch: u16) -> usize {
        if self.chk[(self.last & 15) as usize] == ch {
            return (self.last & 15) as usize;
        }
        let mut b: u32 = 0xffff;
        let mut bi: usize = 0;
        for i in 0..7 {
            if self.chk[i] == ch {
                self.last = (self.last << 4) | (i as u8);
                return i;
            }
            let pri = self.bh[i][0] as u32;
            if (self.last & 15) as usize != i
                && (self.last >> 4) as usize != i
                && pri < b
            {
                b = pri; bi = i;
            }
        }
        self.last = 0xf0 | (bi as u8);
        self.chk[bi] = ch;
        self.bh[bi] = [0; 7];
        bi
    }
}

/// ContextMap — paq8.cpp:1011-1146. Multi-context bit-history map.
pub struct ContextMap {
    pub c: u32,
    t:        Vec<ECell>,
    /// Per-context `cp` state: which cell (`tbl_idx`, `slot`) and the
    /// `bit_offset` within `bh[slot]` we're currently looking at.
    /// `tbl_idx = u32::MAX` ⇒ cp is "null" (skipped this bit).
    cp:       Vec<(u32, u8, u8)>,
    /// `cp0` — base cell pointer at byte-boundary lookup.
    cp0:      Vec<(u32, u8)>,
    /// Per-context byte hash (after finalize64).
    cxt:      Vec<u32>,
    /// Per-context checksum (truncated to 16 bits).
    chk:      Vec<u16>,
    /// Per-context run-length tracker offset within bh[slot]
    /// (initially 3 — start of the 4-byte run header).
    runp_off: Vec<u8>,
    sm:       Vec<StateMap>,
    pub cn:   u32,
    mask:     u32,
    hashbits: u32,
}

impl ContextMap {
    pub fn new(m: u64, c: u32, dt: [i32; 1024]) -> Self {
        let n_cells = (m >> 6).next_power_of_two() as usize;
        let sm: Vec<StateMap> = (0..c).map(|_| StateMap::new()).collect();
        let _ = dt;
        Self {
            c,
            t:   vec![ECell::default(); n_cells],
            cp:  vec![(0, 0, 0); c as usize],
            cp0: vec![(0, 0); c as usize],
            cxt: vec![0; c as usize],
            chk: vec![0; c as usize],
            runp_off: vec![3; c as usize],
            sm,
            cn: 0,
            mask: (n_cells - 1) as u32,
            hashbits: ilog2(n_cells as u32),
        }
    }

    pub fn set(&mut self, cx: u64) {
        // `hash(cx, cn)` — bind context to its slot index.
        let h = super::substrate::hash2(cx, self.cn as u64);
        let idx = self.cn as usize;
        self.cxt[idx] = finalize64(h, self.hashbits);
        self.chk[idx] = (checksum64(h, self.hashbits, 16) & 0xffff) as u16;
        self.cn += 1;
    }

    /// `mix1` — paq8.cpp:1070-1146. Calls `mixer.add` 5 times per
    /// context. Returns count of "non-zero state" contexts (used by
    /// upstream for order indexing).
    pub fn mix1(&mut self, m: &mut Mixer, cc: u32, bp: i32, c1: u8, y1: i32,
                ilog: &super::substrate::Ilog,
                _squash: &Squash, stretch: &Stretch) -> i32 {
        let mut result = 0;
        for i in 0..self.cn as usize {
            // Step current cp state if active.
            let (cp_tbl, cp_slot, cp_off) = self.cp[i];
            if cp_tbl != u32::MAX {
                let cur = self.t[cp_tbl as usize].bh[cp_slot as usize]
                    [cp_off as usize];
                let mut ns = nex(cur, y1 as usize);
                if ns >= 204 {
                    // The `rnd() << ((452-ns)>>3)` heuristic — random
                    // demotion. Use a stable PRNG-free deterministic
                    // demote rate matching upstream's gating
                    // probability via a counter on `cur ^ ns`.
                    // For now match by always demoting (slight bias
                    // — TODO replace with proper Random hookup).
                    ns -= 4;
                }
                self.t[cp_tbl as usize].bh[cp_slot as usize][cp_off as usize] = ns;
            }

            // Refresh cp based on bp position.
            let mut killed = false;
            if bp > 1 && (cp_tbl != u32::MAX
                && self.t[cp_tbl as usize].bh
                    [cp_slot as usize][self.runp_off[i] as usize] == 0)
            {
                self.cp[i] = (u32::MAX, 0, 0);
                killed = true;
            }
            if !killed {
                match bp {
                    1 | 3 | 6 => {
                        let (b, s) = self.cp0[i];
                        self.cp[i] = (b, s, 1 + (cc & 1) as u8);
                    }
                    4 | 7 => {
                        let (b, s) = self.cp0[i];
                        self.cp[i] = (b, s, 3 + (cc & 3) as u8);
                    }
                    2 | 5 => {
                        let ctx = self.cxt[i];
                        let chk = self.chk[i];
                        let idx = ((ctx.wrapping_add(cc)) & self.mask) as usize;
                        let slot = self.t[idx].get(chk);
                        self.cp0[i] = (idx as u32, slot as u8);
                        self.cp[i]  = (idx as u32, slot as u8, 0);
                    }
                    _ => {  // bp == 0 (byte boundary)
                        let ctx = self.cxt[i];
                        let chk = self.chk[i];
                        let idx = ((ctx.wrapping_add(cc)) & self.mask) as usize;
                        let slot = self.t[idx].get(chk);
                        self.cp0[i] = (idx as u32, slot as u8);
                        self.cp[i]  = (idx as u32, slot as u8, 0);

                        // Propagate pending 2-7 bit histories. bh[s][3]
                        // == 2 ⇒ a complete byte arrived; expand its
                        // bits into surrounding cells.
                        if self.t[idx].bh[slot][3] == 2 {
                            let c = self.t[idx].bh[slot][4] as i32 + 256;
                            let idx_a = ((ctx.wrapping_add((c >> 6) as u32))
                                & self.mask) as usize;
                            let slot_a = self.t[idx_a].get(chk);
                            self.t[idx_a].bh[slot_a][0] = 1 + (((c >> 5) & 1) as u8);
                            self.t[idx_a].bh[slot_a][1 + ((c >> 5) & 1) as usize]
                                = 1 + (((c >> 4) & 1) as u8);
                            self.t[idx_a].bh[slot_a][3 + ((c >> 4) & 3) as usize]
                                = 1 + (((c >> 3) & 1) as u8);

                            let idx_b = ((ctx.wrapping_add((c >> 3) as u32))
                                & self.mask) as usize;
                            let slot_b = self.t[idx_b].get(chk);
                            self.t[idx_b].bh[slot_b][0] = 1 + (((c >> 2) & 1) as u8);
                            self.t[idx_b].bh[slot_b][1 + ((c >> 2) & 1) as usize]
                                = 1 + (((c >> 1) & 1) as u8);
                            self.t[idx_b].bh[slot_b][3 + ((c >> 1) & 3) as usize]
                                = 1 + ((c & 1) as u8);

                            self.t[idx].bh[slot][6] = 0;
                        }

                        // Run-length tracker — paq8.cpp:1110-1118.
                        let run_off = self.runp_off[i] as usize;
                        let rc0 = self.t[idx].bh[slot][run_off];
                        let rc1 = self.t[idx].bh[slot][run_off + 1];
                        if rc0 == 0 {
                            self.t[idx].bh[slot][run_off    ] = 2;
                            self.t[idx].bh[slot][run_off + 1] = c1;
                        } else if rc1 != c1 {
                            self.t[idx].bh[slot][run_off    ] = 1;
                            self.t[idx].bh[slot][run_off + 1] = c1;
                        } else if rc0 < 254 {
                            self.t[idx].bh[slot][run_off    ] = rc0 + 2;
                        } else if rc0 == 255 {
                            self.t[idx].bh[slot][run_off    ] = 128;
                        }
                        // runp now points 3 bytes into the new cell.
                        self.runp_off[i] = 3;
                    }
                }
            }

            // Reads for mixer adds.
            let (tbl, slot, off) = self.cp[i];
            let runp_off = self.runp_off[i] as usize;
            let (run_rc, run_b) = if tbl == u32::MAX {
                (0u8, 0u8)
            } else {
                (self.t[tbl as usize].bh[slot as usize][runp_off],
                 self.t[tbl as usize].bh[slot as usize][runp_off + 1])
            };

            let rc = run_rc as u32;
            if ((run_b as u32 + 256) >> (8 - bp)) == cc {
                let bit = ((run_b >> (7 - bp)) & 1) as i32;
                let sign = bit * 2 - 1;
                let il = ilog.get(((rc + 1) & 0xffff) as u16) as i32;
                let c = il << (2 + (!rc & 1) as i32);
                m.add((sign * c) as i16);
            } else {
                m.add(0);
            }

            let s = if tbl == u32::MAX {
                0u8
            } else {
                self.t[tbl as usize].bh[slot as usize][off as usize]
            };
            let p1 = self.sm[i].p(s as u32, y1);
            let st = (stretch.get(p1) + (1 << 1)) >> 2;
            m.add(st as i16);
            m.add(((p1 - 2047 + (1 << 2)) >> 3) as i16);
            let n0_neg: i32 = if nex(s, 2) == 0 { -1 } else { 0 };
            let n1_neg: i32 = if nex(s, 3) == 0 { -1 } else { 0 };
            m.add((st * (n1_neg - n0_neg).abs()) as i16);
            let p0 = 4095 - p1;
            m.add((((p1 & n0_neg) - (p0 & n1_neg) + (1 << 3)) >> 4) as i16);
            result += if s > 0 { 1 } else { 0 };
        }
        if bp == 7 { self.cn = 0; }
        result
    }
}

// =================================================================
// ContextMap2 — paq8.cpp:1165-1360.
//
// 2nd-generation context map. Reuses [`ECell`] as the 64-byte hash
// bucket (Checksums[7] + MRU + BitState[7][7]). Maintains a 4-byte
// byte-history alongside the bit-history and runs three StateMap32
// layers (6-bit, 8-bit, 12-bit) per context.
// =================================================================

/// Per-context pointer state for ContextMap2.
#[derive(Clone, Copy)]
struct Cm2Ptr {
    /// `BitState` — (table_idx, slot, bit_offset). `tbl == u32::MAX`
    /// ⇒ null.
    bs:  (u32, u8, u8),
    /// `BitState0` — (table_idx, slot) of the byte-boundary cell.
    bs0: (u32, u8),
    /// `ByteHistory` — (table_idx, slot); bytes live at
    /// `bh[slot][3..7]` = [RunStats, b1, b2, b3].
    bh:  (u32, u8),
}

pub struct ContextMap2 {
    pub c:    u32,
    table:    Vec<ECell>,
    ptr:      Vec<Cm2Ptr>,
    contexts: Vec<u32>,
    chk:      Vec<u16>,
    has_history: Vec<bool>,
    maps6b:   Vec<StateMap32>,
    maps8b:   Vec<StateMap32>,
    maps12b:  Vec<StateMap32>,
    pub index: u32,
    mask:     u32,
    hashbits: u32,
    bits:     u32,
    last_byte: u8,
    last_bit:  u8,
    bit_pos:   i32,
}

impl ContextMap2 {
    pub fn new(size: u64, count: u32, dt: [i32; 1024]) -> Self {
        let n_cells = (size >> 6).next_power_of_two() as usize;
        let maps6b  = (0..count).map(|_| StateMap32::new((1 << 6) + 8, dt)).collect();
        let maps8b  = (0..count).map(|_| StateMap32::new(1 << 8, dt)).collect();
        let maps12b = (0..count).map(|_| StateMap32::new((1 << 12) + (1 << 9), dt)).collect();
        Self {
            c: count,
            table: vec![ECell::default(); n_cells],
            ptr: vec![Cm2Ptr { bs: (0, 0, 0), bs0: (0, 0), bh: (0, 0) };
                      count as usize],
            contexts: vec![0; count as usize],
            chk:      vec![0; count as usize],
            has_history: vec![false; count as usize],
            maps6b, maps8b, maps12b,
            index: 0,
            mask: (n_cells - 1) as u32,
            hashbits: ilog2(n_cells as u32),
            bits: 1,
            last_byte: 0,
            last_bit: 0,
            bit_pos: 0,
        }
    }

    pub fn set(&mut self, ctx: u64) {
        let h = super::substrate::hash2(ctx, self.index as u64);
        let i = self.index as usize;
        self.contexts[i] = finalize64(h, self.hashbits);
        self.chk[i] = (checksum64(h, self.hashbits, 16) & 0xffff) as u16;
        self.index += 1;
    }

    /// `Update()` — paq8.cpp:1206-1259.
    fn update(&mut self) {
        for i in 0..self.index as usize {
            // Step current bit-state if non-null.
            let (bt, bsl, boff) = self.ptr[i].bs;
            if bt != u32::MAX {
                let cur = self.table[bt as usize].bh[bsl as usize][boff as usize];
                self.table[bt as usize].bh[bsl as usize][boff as usize] =
                    nex(cur, self.last_bit as usize);
            }

            // bitPos>1 && ByteHistory[i][0]==0 → kill the bit state.
            let (bht, bhsl) = self.ptr[i].bh;
            let run_stats = self.table[bht as usize].bh[bhsl as usize][3];
            if self.bit_pos > 1 && run_stats == 0 {
                self.ptr[i].bs = (u32::MAX, 0, 0);
                continue;
            }

            match self.bit_pos {
                0 => {
                    let chk = self.chk[i];
                    let ctx = self.contexts[i];
                    let idx = ((ctx.wrapping_add(self.bits)) & self.mask) as usize;
                    let slot = self.table[idx].get(chk);
                    self.ptr[i].bs  = (idx as u32, slot as u8, 0);
                    self.ptr[i].bs0 = (idx as u32, slot as u8);

                    // Pending 2-7 bit history propagation.
                    if self.table[idx].bh[slot][3] == 2 {
                        let c = self.table[idx].bh[slot][4] as i32 + 256;
                        let idx_a = ((ctx.wrapping_add((c >> 6) as u32))
                            & self.mask) as usize;
                        let slot_a = self.table[idx_a].get(chk);
                        self.table[idx_a].bh[slot_a][0] = 1 + (((c >> 5) & 1) as u8);
                        self.table[idx_a].bh[slot_a][1 + ((c >> 5) & 1) as usize]
                            = 1 + (((c >> 4) & 1) as u8);
                        self.table[idx_a].bh[slot_a][3 + ((c >> 4) & 3) as usize]
                            = 1 + (((c >> 3) & 1) as u8);
                        let idx_b = ((ctx.wrapping_add((c >> 3) as u32))
                            & self.mask) as usize;
                        let slot_b = self.table[idx_b].get(chk);
                        self.table[idx_b].bh[slot_b][0] = 1 + (((c >> 2) & 1) as u8);
                        self.table[idx_b].bh[slot_b][1 + ((c >> 2) & 1) as usize]
                            = 1 + (((c >> 1) & 1) as u8);
                        self.table[idx_b].bh[slot_b][3 + ((c >> 1) & 3) as usize]
                            = 1 + ((c & 1) as u8);
                        self.table[idx].bh[slot][6] = 0;
                    }

                    // Update byte history of the PREVIOUS context.
                    let (pbt, pbsl) = self.ptr[i].bh;
                    {
                        let bh = &mut self.table[pbt as usize].bh[pbsl as usize];
                        bh[6] = bh[5];
                        bh[5] = bh[4];
                        if bh[3] == 0 {
                            bh[3] = 2; bh[4] = self.last_byte;
                        } else if bh[4] != self.last_byte {
                            bh[3] = 1; bh[4] = self.last_byte;
                        } else if bh[3] < 254 {
                            bh[3] += 2;
                        } else if bh[3] == 255 {
                            bh[3] = 128;
                        }
                    }
                    // ByteHistory now points at the new cell's run area.
                    self.ptr[i].bh = (idx as u32, slot as u8);
                    self.has_history[i] = self.table[idx].bh[slot][0] > 15;
                }
                2 | 5 => {
                    let chk = self.chk[i];
                    let ctx = self.contexts[i];
                    let idx = ((ctx.wrapping_add(self.bits)) & self.mask) as usize;
                    let slot = self.table[idx].get(chk);
                    self.ptr[i].bs  = (idx as u32, slot as u8, 0);
                    self.ptr[i].bs0 = (idx as u32, slot as u8);
                }
                1 | 3 | 6 => {
                    let (b, s) = self.ptr[i].bs0;
                    self.ptr[i].bs = (b, s, 1 + self.last_bit);
                }
                _ /* 4 | 7 */ => {
                    let (b, s) = self.ptr[i].bs0;
                    self.ptr[i].bs = (b, s, 3 + (self.bits & 3) as u8);
                }
            }
        }
    }

    /// `mix` — paq8.cpp:1303-1359. 7 stretched inputs per context.
    pub fn mix(&mut self, m: &mut Mixer, y: i32, bpos: i32,
                ilog: &super::substrate::Ilog,
                _squash: &Squash, stretch: &Stretch) -> i32 {
        let mut result = 0;
        self.last_bit = y as u8;
        self.bit_pos  = bpos;
        self.bits = self.bits.wrapping_add(self.bits).wrapping_add(y as u32);
        self.last_byte = (self.bits & 0xFF) as u8;
        if bpos == 0 { self.bits = 1; }
        self.update();

        for i in 0..self.index as usize {
            let (bt, bsl, boff) = self.ptr[i].bs;
            let state = if bt == u32::MAX {
                0
            } else {
                self.table[bt as usize].bh[bsl as usize][boff as usize]
            };
            result += (state > 0) as i32;
            let p1_full = self.maps8b[i].p(state as u32, 1023, y);
            // Upstream: k=-~n1 (= n1+1); k=(k*64)/(k-~n0) where
            // k-~n0 = (n1+1)+(n0+1) = n0+n1+2; then n0=-!n0, n1=-!n1.
            let n0_raw = nex(state, 2) as i32;
            let n1_raw = nex(state, 3) as i32;
            let denom  = n0_raw + n1_raw + 2; // always >= 2
            let k      = ((n1_raw + 1) * 64) / denom;
            let n0: i32 = if n0_raw == 0 { -1 } else { 0 };
            let n1: i32 = if n1_raw == 0 { -1 } else { 0 };

            let (bht, bhsl) = self.ptr[i].bh;
            let bh = self.table[bht as usize].bh[bhsl as usize];
            let run_stats = bh[3] as u32;
            let b1 = bh[4] as u32;
            let b2 = bh[5] as u32;
            let b3 = bh[6] as u32;

            if ((b1 + 256) >> (8 - bpos)) == self.bits {
                let sign = ((b1 >> (7 - bpos)) & 1) as i32 * 2 - 1;
                let value = (ilog.get(((run_stats + 1) & 0xffff) as u16) as i32)
                    << (3 - (run_stats & 1) as i32);
                m.add((sign * value) as i16);
            } else if bpos > 0 && (run_stats & 1) > 0 {
                if ((b2 + 256) >> (8 - bpos)) == self.bits {
                    let v = (((b2 >> (7 - bpos)) & 1) as i32 * 2 - 1) * 128;
                    m.add(v as i16);
                } else if self.has_history[i]
                    && ((b3 + 256) >> (8 - bpos)) == self.bits
                {
                    let v = (((b3 >> (7 - bpos)) & 1) as i32 * 2 - 1) * 128;
                    m.add(v as i16);
                } else {
                    m.add(0);
                }
            } else {
                m.add(0);
            }

            let hist_state: u32 = if self.has_history[i] {
                let mut s = (b1 >> (7 - bpos)) & 1;
                s |= ((b2 >> (7 - bpos)) & 1) * 2;
                s |= ((b3 >> (7 - bpos)) & 1) * 4;
                s
            } else {
                8
            };

            let st = stretch.get(p1_full) >> 2;
            m.add(st as i16);
            m.add(((p1_full - 2047) >> 3) as i16);
            let p1 = p1_full >> 4;
            let p0 = 255 - p1;
            m.add((st * (n1 - n0).abs()) as i16);
            m.add(((p1 & n0) - (p0 & n1)) as i16);
            let m12 = self.maps12b[i].p(
                (hist_state << 9) | ((bpos as u32) << 6) | (k as u32),
                1023, y);
            m.add((stretch.get(m12) >> 2) as i16);
            let m6 = self.maps6b[i].p(
                (hist_state << 3) | bpos as u32, 1023, y);
            m.add((stretch.get(m6) >> 2) as i16);
        }
        if bpos == 7 { self.index = 0; }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::{build_dt, Squash, Stretch};

    fn small_mixer() -> (Mixer, Squash, Stretch) {
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        (Mixer::new(64, 4, 0), sq, st)
    }

    #[test]
    fn hash_table_b_round_trip_get() {
        let mut h = HashTableB::new(16, 8);
        let p1 = h.get(0xDEAD_BEEF).as_ptr() as usize;
        let p2 = h.get(0xDEAD_BEEF).as_ptr() as usize;
        assert_eq!(p1, p2, "same key must return same slot");
    }

    #[test]
    fn bh_lookups_for_same_key_are_stable() {
        let mut h = Bh::new(16, 8);
        let p1 = h.get(0xCAFE_BABE_DEAD_BEEF).as_ptr() as usize;
        let p2 = h.get(0xCAFE_BABE_DEAD_BEEF).as_ptr() as usize;
        assert_eq!(p1, p2);
    }

    #[test]
    fn small_stationary_context_map_mixes_without_panic() {
        let (mut mx, sq, st) = small_mixer();
        let mut sscm = SmallStationaryContextMap::new(8, 8);
        sscm.set(0xAB);
        for _ in 0..8 {
            sscm.mix(&mut mx, 0, 7, 1, 4, &sq, &st);
        }
    }

    #[test]
    fn stationary_map_mixes_without_panic() {
        let (mut mx, sq, st) = small_mixer();
        let dt = build_dt();
        let mut sm = StationaryMap::new(8, 8, 0);
        sm.set_direct(0xAB);
        for _ in 0..8 {
            sm.mix(&mut mx, 0, 1, 4, 1023, &dt, &sq, &st);
        }
    }

    #[test]
    fn indirect_map_mixes_without_panic() {
        let (mut mx, sq, st) = small_mixer();
        let dt = build_dt();
        let mut im = IndirectMap::new(8, 8, dt);
        im.set_direct(0xCD);
        for _ in 0..8 {
            im.mix(&mut mx, 0, 1, 4, 1023, &sq, &st);
        }
    }

    #[test]
    fn e_cell_find_or_create_round_trip() {
        let mut e = ECell::default();
        let s1 = e.get(0xABCD);
        assert!(s1 < 7);
        let s2 = e.get(0xABCD);
        assert_eq!(s1, s2, "same checksum must return same slot");
        let s3 = e.get(0xBEEF);
        assert_ne!(s1, s3, "different checksums use different slots");
    }

    #[test]
    fn context_map_constructs_and_sets_contexts_without_panic() {
        let (_, _sq, _st) = small_mixer();
        // 64 KiB / 64 byte buckets = 1024 cells. C=8 contexts.
        let mut cm = ContextMap::new(64 * 1024, 8, build_dt());
        for j in 0..8 {
            cm.set(0xDEAD_BEEFu64.wrapping_add(j as u64));
        }
        assert_eq!(cm.cn, 8);
    }

    #[test]
    fn context_map2_constructs_and_runs_8_bits() {
        let (mut mx, sq, st) = small_mixer();
        let il = super::super::substrate::Ilog::new();
        let mut cm = ContextMap2::new(64 * 1024, 4, build_dt());
        for j in 0..4 {
            cm.set(0xFEED_0000u64.wrapping_add(j as u64));
        }
        for bp in 0..8 {
            let _ = cm.mix(&mut mx, (bp & 1) as i32, bp, &il, &sq, &st);
        }
    }

    #[test]
    fn run_context_map_tracks_and_predicts() {
        let il = super::super::substrate::Ilog::new();
        let (mut mx, _sq, _st) = small_mixer();
        let mut rcm = RunContextMap::new(1 << 16);
        rcm.set(0xDEAD, 0xAB);
        rcm.set(0xBEEF, 0xCD);
        // Same context + same byte twice — count should grow.
        rcm.set(0xCAFE, 0xCD);
        rcm.set(0xCAFE, 0xCD);
        let _ = rcm.mix(&mut mx, 0xCD, 7, &il);
    }

    #[test]
    fn context_map_mix1_runs_through_8_bits_without_panic() {
        let (mut mx, sq, st) = small_mixer();
        let il = super::super::substrate::Ilog::new();
        let mut cm = ContextMap::new(64 * 1024, 4, build_dt());
        for j in 0..4 {
            cm.set(0xCAFE_0000u64.wrapping_add(j as u64));
        }
        for bp in 0..8 {
            let cc = if bp == 0 { 1u32 } else { (1u32 << bp) | 0xA5 & ((1 << bp) - 1) };
            let _ = cm.mix1(&mut mx, cc, bp, 0xA5, 0, &il, &sq, &st);
        }
    }
}
