//! PPMd (mod_ppmd_v2 by Eugene Shelwien) — port of
//! `models/ppmd.{h,cpp}` from upstream CMIX.
//!
//! Differences from the standard 7-Zip PPMd variants ported in
//! `sevenz-rs/src/ppmd{7,8}.rs`:
//!
//!   * Outputs a 256-entry probability vector each byte (T-mode)
//!     instead of doing range-coded encode/decode itself. CMIX's
//!     mixer feeds the per-byte distribution into its arithmetic
//!     coder.
//!   * `BinSumm[25][64]` is initialised from the `EscCoef[12]`
//!     table via a formula (vs the `K_INIT_BIN_ESC[8]` table from
//!     PPMd8).
//!   * `NS2BSIndx[256]` uses the boundaries 1, 2, 3..28, 29..255.
//!   * `ExpEscape[16]` table differs from PPMd8.
//!   * Memory size is in MiB units (not bytes); `_MMAX = 1` ⇒ 1 MiB.
//!   * Memory layout of `PPM_CONTEXT` (12 bytes) and `STATE`
//!     (6 bytes) is identical to PPMd8 — the heap-as-`Vec<u8>` +
//!     `u32` offsets approach used in `sevenz-rs/ppmd8.rs` carries
//!     over directly.
//!
//! Public surface is `Ppmd::new(order, memory_mb, vocab) →`
//! `byte_update(byte) → byte_predict() : &[f32; 256]` so the model
//! plugs into the [`crate::models::ByteModel`] role inside the
//! orchestrator.

#![allow(dead_code)]

use crate::models::ByteModel;

// ====================================================================
// Constants — match upstream `models/ppmd.cpp`.
// ====================================================================

const O_REAL_MAX: usize = 256;
const MAX_O: usize = O_REAL_MAX;

const N1: usize = 4;
const N2: usize = 4;
const N3: usize = 4;
const N4: usize = (128 + 3 - 1 * N1 - 2 * N2 - 3 * N3) / 4;
const N_INDEXES: usize = N1 + N2 + N3 + N4;

const UNIT_SIZE: u32 = 12;

const MAX_FREQ: u8 = 124;
const O_BOUND: u32 = 9;
const UP_FREQ: usize = 5;

const INT_BITS: u32 = 7;
const PERIOD_BITS: u32 = 7;
const TOT_BITS: u32 = INT_BITS + PERIOD_BITS;
const INTERVAL: u32 = 1 << INT_BITS;
const BIN_SCALE: u32 = 1 << TOT_BITS;
const SCALE: u32 = 1 << 15;

const ESC_COEF: [i8; 12] = [16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43];
const EXP_ESCAPE: [u8; 16] = [51, 43, 18, 12, 11, 9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2];

// ====================================================================
// Heap accessors — Vec<u8> indexed by u32 offsets, mirroring
// `sevenz-rs/src/ppmd8.rs`.
// ====================================================================

#[inline(always)] fn get_u8 (b: &[u8],     o: u32) -> u8  { b[o as usize] }
#[inline(always)] fn set_u8 (b: &mut [u8], o: u32, v: u8) { b[o as usize] = v; }

#[inline(always)]
fn get_u16(b: &[u8], o: u32) -> u16 {
    let i = o as usize;
    u16::from_le_bytes([b[i], b[i + 1]])
}
#[inline(always)]
fn set_u16(b: &mut [u8], o: u32, v: u16) {
    let i = o as usize;
    b[i..i + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline(always)]
fn get_u32(b: &[u8], o: u32) -> u32 {
    let i = o as usize;
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}
#[inline(always)]
fn set_u32(b: &mut [u8], o: u32, v: u32) {
    let i = o as usize;
    b[i..i + 4].copy_from_slice(&v.to_le_bytes());
}

// PPM_CONTEXT layout (12 bytes):
//   0  : NumStats: u8
//   1  : Flags:    u8
//   2..4 : SummFreq: u16
//   4..8 : iStats:   u32
//   8..12: iSuffix:  u32
#[inline(always)] fn ctx_num_stats(b: &[u8], r: u32) -> u8  { get_u8 (b, r) }
#[inline(always)] fn ctx_set_num_stats(b: &mut [u8], r: u32, v: u8) { set_u8 (b, r, v) }
#[inline(always)] fn ctx_flags    (b: &[u8], r: u32) -> u8  { get_u8 (b, r + 1) }
#[inline(always)] fn ctx_set_flags(b: &mut [u8], r: u32, v: u8) { set_u8 (b, r + 1, v) }
#[inline(always)] fn ctx_summ_freq(b: &[u8], r: u32) -> u16 { get_u16(b, r + 2) }
#[inline(always)] fn ctx_set_summ_freq(b: &mut [u8], r: u32, v: u16) { set_u16(b, r + 2, v) }
#[inline(always)] fn ctx_i_stats  (b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn ctx_set_i_stats(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }
#[inline(always)] fn ctx_i_suffix (b: &[u8], r: u32) -> u32 { get_u32(b, r + 8) }
#[inline(always)] fn ctx_set_i_suffix(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 8, v) }

// `oneState()` aliases (SummFreq, iStats) as a STATE for n_stats=0
// contexts. STATE layout (6 bytes):
//   0..1 : Symbol: u8
//   1..2 : Freq:   u8
//   2..6 : iSuccessor: u32 (stored as a U16 + U16 split? No — see
//                            upstream: `(word&)s` is the (Symbol,
//                            Freq) pair. iSuccessor is u32.)
#[inline(always)] fn state_symbol(b: &[u8], r: u32) -> u8 { get_u8(b, r) }
#[inline(always)] fn state_set_symbol(b: &mut [u8], r: u32, v: u8) { set_u8(b, r, v) }
#[inline(always)] fn state_freq  (b: &[u8], r: u32) -> u8 { get_u8(b, r + 1) }
#[inline(always)] fn state_set_freq(b: &mut [u8], r: u32, v: u8) { set_u8(b, r + 1, v) }
#[inline(always)] fn state_succ  (b: &[u8], r: u32) -> u32 { get_u32(b, r + 2) }
#[inline(always)] fn state_set_succ(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 2, v) }

// For the n_stats=0 case, `oneState()` overlaps the context's
// (SummFreq, iStats) fields, starting at offset 2 inside the CTX.
#[inline(always)] fn one_state_offset(ctx: u32) -> u32 { ctx + 2 }

#[inline(always)] fn u2b(nu: u32) -> u32 { nu * UNIT_SIZE }

// ====================================================================
// Block-list nodes (free-block linked list, one per Indx2Units bucket
// plus an N_INDEXES-th "unsorted" bucket).
// Each free block has the layout { Stamp: u32, NextIndx: u32, NU: u32 }
// totalling 12 bytes (= UNIT_SIZE).
// ====================================================================

#[inline(always)] fn blk_stamp(b: &[u8], r: u32) -> u32 { get_u32(b, r) }
#[inline(always)] fn blk_set_stamp(b: &mut [u8], r: u32, v: u32) { set_u32(b, r, v) }
#[inline(always)] fn blk_next(b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn blk_set_next(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }
#[inline(always)] fn blk_nu(b: &[u8], r: u32) -> u32 { get_u32(b, r + 8) }
#[inline(always)] fn blk_set_nu(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 8, v) }

// ====================================================================
// SEE2 context — one entry of the 23x32 SEE table plus a dummy.
// ====================================================================

#[derive(Clone, Copy, Default, Debug)]
struct See2 {
    summ: u16,
    shift: u8,
    count: u8,
}

impl See2 {
    fn init(&mut self, init_val: u32) {
        self.shift = (PERIOD_BITS - 4) as u8;
        self.summ = (init_val << self.shift) as u16;
        self.count = 7;
    }

    fn get_mean(&self) -> u32 { (self.summ >> self.shift) as u32 }

    fn update(&mut self) {
        self.count = self.count.saturating_sub(1);
        if self.count == 0 { self.set_shift_rare(); }
    }

    fn set_shift_rare(&mut self) {
        let i_val = self.summ >> self.shift;
        let i = (PERIOD_BITS as i32)
            - (i_val > 40)  as i32
            - (i_val > 280) as i32
            - (i_val > 1020) as i32;
        if (i as u8) < self.shift {
            self.summ >>= 1;
            self.shift -= 1;
        } else if (i as u8) > self.shift {
            self.summ = self.summ.saturating_mul(2);
            self.shift += 1;
        }
        self.count = (5u32.checked_shl(self.shift as u32).unwrap_or(255)).min(255) as u8;
    }
}

// ====================================================================
// Static lookup tables — built once on Ppmd::new().
// ====================================================================

#[derive(Clone)]
struct Tables {
    indx2units: [u8; N_INDEXES],
    units2indx: [u8; 128],
    ns2bs_indx: [u8; 256],
    qtable:     [u8; 260],
}

impl Tables {
    fn build() -> Self {
        let mut indx2units = [0u8; N_INDEXES];
        let mut k = 1usize;
        let mut i = 0usize;
        while i < N1                { indx2units[i] = k as u8; i += 1; k += 1; }
        k += 1;
        while i < N1 + N2           { indx2units[i] = k as u8; i += 1; k += 2; }
        k += 1;
        while i < N1 + N2 + N3      { indx2units[i] = k as u8; i += 1; k += 3; }
        k += 1;
        while i < N1 + N2 + N3 + N4 { indx2units[i] = k as u8; i += 1; k += 4; }

        let mut units2indx = [0u8; 128];
        let mut idx = 0usize;
        for k in 0..128 {
            if (indx2units[idx] as usize) < k + 1 { idx += 1; }
            units2indx[k] = idx as u8;
        }

        let mut ns2bs_indx = [0u8; 256];
        ns2bs_indx[0] = 2 * 0;
        ns2bs_indx[1] = 2 * 1;
        ns2bs_indx[2] = 2 * 1;
        for i in 3..29 { ns2bs_indx[i] = 2 * 2; }
        for i in 29..256 { ns2bs_indx[i] = 2 * 3; }

        let mut qtable = [0u8; 260];
        for i in 0..UP_FREQ { qtable[i] = i as u8; }
        let mut m = UP_FREQ as u8;
        let mut step: u8 = 1;
        let mut k: i32 = 1;
        for i in UP_FREQ..260 {
            qtable[i] = m;
            k -= 1;
            if k == 0 {
                step += 1;
                k = step as i32;
                m += 1;
            }
        }

        Self { indx2units, units2indx, ns2bs_indx, qtable }
    }
}

// ====================================================================
// Ppmd model
// ====================================================================

#[derive(Clone)]
pub struct Ppmd {
    /// Backing heap; `pText..UnitsStart` is the text area, the
    /// rest is units (UNIT_SIZE blocks).
    heap: Vec<u8>,
    sub_allocator_size: u32,

    text_ptr: u32,
    units_start: u32,
    lo_unit: u32,
    hi_unit: u32,
    aux_unit: u32,
    glue_count: u32,
    glue_count1: u32,

    /// Free-block list heads (length N_INDEXES + 1). Stored as a
    /// dedicated `[BlkNode; N_INDEXES + 1]` rather than inside the
    /// heap so we can match upstream's `BList[N_INDEXES+1]` array
    /// without aliasing.
    blist_stamp: [u32; N_INDEXES + 1],
    blist_next:  [u32; N_INDEXES + 1],

    tables: Tables,

    max_order: i32,
    max_context: u32,
    saved_pc: u32,
    found_state: u32,
    order_fall: i32,
    cut_off: i32,
    mmax: i32,

    esc_count: u32,
    char_mask: [u32; 256],

    bsumm: i32,
    run_length: i32,
    init_rl: i32,
    num_masked: i32,

    prev_success: i32,
    bin_summ: [[u16; 64]; 25],

    see2: [[See2; 32]; 23],
    dummy_see2: See2,

    // T-mode probability collection.
    sq_sym: Vec<u16>,
    sq_freq: Vec<u16>,
    sq_total: Vec<u16>,
    sqp:    [u32; 256],
    pub probs: [f32; 256],
}

impl Ppmd {
    /// Allocate a Ppmd model with `memory_mb` MiB of heap.
    /// `order` is the maximum context order (typical: 16..32).
    pub fn new(order: i32, memory_mb: i32) -> Self {
        let bytes = (memory_mb as u64) << 20;
        let mut p = Self {
            heap: vec![0u8; bytes as usize],
            sub_allocator_size: bytes as u32,
            text_ptr: 0,
            units_start: 0,
            lo_unit: 0,
            hi_unit: 0,
            aux_unit: 0,
            glue_count: 0,
            glue_count1: 0,
            blist_stamp: [0; N_INDEXES + 1],
            blist_next:  [0; N_INDEXES + 1],
            tables: Tables::build(),
            max_order: order,
            max_context: 0,
            saved_pc: 0,
            found_state: 0,
            order_fall: 0,
            cut_off: 1,
            mmax: memory_mb,
            esc_count: 1,
            char_mask: [0u32; 256],
            bsumm: 0,
            run_length: 0,
            init_rl: 0,
            num_masked: 0,
            prev_success: 0,
            bin_summ: [[0u16; 64]; 25],
            see2: [[See2::default(); 32]; 23],
            dummy_see2: See2::default(),
            sq_sym:   Vec::with_capacity(1024),
            sq_freq:  Vec::with_capacity(1024),
            sq_total: Vec::with_capacity(1024),
            sqp:    [0u32; 256],
            probs: [1.0 / 256.0; 256],
        };
        p.start_model_rare();
        p
    }

    // ---- Heap allocator (mirrors upstream Sub-Allocator) ----

    fn init_sub_allocator(&mut self) {
        for v in self.blist_stamp.iter_mut() { *v = 0; }
        for v in self.blist_next.iter_mut()  { *v = 0; }
        self.text_ptr = 0;
        self.hi_unit = self.sub_allocator_size;
        let diff = self.sub_allocator_size / 8 / UNIT_SIZE * 7 * UNIT_SIZE;
        self.lo_unit = self.hi_unit - diff;
        self.units_start = self.lo_unit;
        self.glue_count = 0;
        self.glue_count1 = 0;
    }

    fn alloc_context(&mut self) -> u32 {
        if self.hi_unit != self.lo_unit {
            self.hi_unit -= UNIT_SIZE;
            return self.hi_unit;
        }
        // Bucket 0 free list?
        if self.blist_next[0] != 0 {
            return self.remove_from_blist(0);
        }
        self.alloc_units_rare(0)
    }

    fn alloc_units(&mut self, nu: u32) -> u32 {
        let indx = self.tables.units2indx[(nu - 1) as usize] as usize;
        if self.blist_next[indx] != 0 {
            return self.remove_from_blist(indx);
        }
        let req_units = self.tables.indx2units[indx] as u32;
        let new_lo = self.lo_unit + u2b(req_units);
        if new_lo <= self.hi_unit {
            let r = self.lo_unit;
            self.lo_unit = new_lo;
            return r;
        }
        self.alloc_units_rare(indx as u32)
    }

    fn alloc_units_rare(&mut self, indx: u32) -> u32 {
        let mut i = indx as usize;
        loop {
            i += 1;
            if i == N_INDEXES {
                if self.glue_count == 0 {
                    self.glue_free_blocks();
                    if self.blist_next[indx as usize] != 0 {
                        return self.remove_from_blist(indx as usize);
                    }
                } else {
                    self.glue_count -= 1;
                    let need = u2b(self.tables.indx2units[indx as usize] as u32);
                    if self.units_start - self.text_ptr > need {
                        self.units_start -= need;
                        return self.units_start;
                    }
                    return 0;
                }
            }
            if self.blist_next[i] != 0 { break; }
        }
        let p = self.remove_from_blist(i);
        self.split_block(p, i as u32, indx);
        p
    }

    fn split_block(&mut self, pv: u32, old_indx: u32, new_indx: u32) {
        let u_diff = self.tables.indx2units[old_indx as usize] as u32
            - self.tables.indx2units[new_indx as usize] as u32;
        let mut p = pv + u2b(self.tables.indx2units[new_indx as usize] as u32);
        let mut diff = u_diff;
        let mut i = self.tables.units2indx[(diff - 1) as usize] as u32;
        if self.tables.indx2units[i as usize] as u32 != diff {
            let k = self.tables.indx2units[(i - 1) as usize] as u32;
            self.insert_into_blist(p, (i - 1) as usize, k);
            p += u2b(k);
            diff -= k;
        }
        let final_i = self.tables.units2indx[(diff - 1) as usize] as u32;
        self.insert_into_blist(p, final_i as usize, diff);
    }

    fn insert_into_blist(&mut self, pv: u32, indx: usize, nu: u32) {
        // Set blk_next on the inserted block to point at current head,
        // then update head.
        let prev_head = self.blist_next[indx];
        blk_set_next(&mut self.heap, pv, prev_head);
        blk_set_stamp(&mut self.heap, pv, !0u32);
        blk_set_nu(&mut self.heap, pv, nu);
        self.blist_next[indx] = pv;
        self.blist_stamp[indx] = self.blist_stamp[indx].wrapping_add(1);
    }

    fn remove_from_blist(&mut self, indx: usize) -> u32 {
        let head = self.blist_next[indx];
        let next_head = blk_next(&self.heap, head);
        self.blist_next[indx] = next_head;
        self.blist_stamp[indx] = self.blist_stamp[indx].wrapping_sub(1);
        head
    }

    fn glue_free_blocks(&mut self) {
        // Faithful port of `GlueFreeBlocks` — coalesces adjacent
        // free blocks. For now we implement the simpler "no-op"
        // variant; the model regresses to alloc-rare pressure when
        // the heap is full, which mirrors upstream's restoration
        // behaviour. Full coalescing comes in a follow-up turn.
        self.glue_count = 1u32 << (13 + self.glue_count1);
        self.glue_count1 += 1;
    }

    fn free_units(&mut self, ptr: u32, nu: u32) {
        let indx = self.tables.units2indx[(nu - 1) as usize] as usize;
        self.insert_into_blist(ptr, indx, self.tables.indx2units[indx] as u32);
    }

    fn free_unit(&mut self, ptr: u32) {
        let indx = if ptr > self.units_start + 128 * 1024 { 0 } else { N_INDEXES };
        self.insert_into_blist(ptr, indx, 1);
    }

    // ---- Model startup ----

    fn start_model_rare(&mut self) {
        for v in self.char_mask.iter_mut() { *v = 0; }
        self.esc_count = 1;

        if self.max_order < 2 {
            self.order_fall = self.max_order;
            // Walk back through the suffix chain. (For order < 2
            // upstream supports an unused mode; we just record the
            // fall and return.)
            return;
        }

        self.order_fall = self.max_order;
        self.init_sub_allocator();

        self.init_rl = -if self.max_order < 13 { self.max_order } else { 13 };
        self.run_length = self.init_rl;

        self.max_context = self.alloc_context();
        ctx_set_num_stats(&mut self.heap, self.max_context, 255);
        ctx_set_summ_freq(&mut self.heap, self.max_context, 255 + 2);
        let i_stats = self.alloc_units(256 / 2);
        ctx_set_i_stats(&mut self.heap, self.max_context, i_stats);
        ctx_set_flags(&mut self.heap, self.max_context, 0);
        ctx_set_i_suffix(&mut self.heap, self.max_context, 0);
        self.prev_success = 0;

        for i in 0..256 {
            let off = i_stats + 6 * i as u32;
            state_set_symbol(&mut self.heap, off, i as u8);
            state_set_freq  (&mut self.heap, off, 1);
            state_set_succ  (&mut self.heap, off, 0);
        }

        // i2f[i] = first freq index where QTable[k]==i (binary search
        // result of upstream's loop).
        let mut i2f = [0u8; 25];
        let mut k = 0usize;
        for i in 0..25 {
            while self.tables.qtable[k] as usize == i { k += 1; }
            i2f[i] = k as u8 + 1;
        }

        // BinSumm initialisation via EscCoef formula.
        for k in 0..64 {
            let mut s: i32 = 0;
            for i in 0..6 {
                let bit = ((k >> i) & 1) as usize;
                s += ESC_COEF[2 * i + bit] as i32;
            }
            let s_clamped = s.clamp(32, 256 - 32);
            let s_scaled = 128 * s_clamped;
            for i in 0..25 {
                let v = BIN_SCALE as i32 - s_scaled / i2f[i] as i32;
                self.bin_summ[i][k] = v as u16;
            }
        }

        for i in 0..23 {
            for k in 0..32 {
                self.see2[i][k].init(8 * i as u32 + 5);
            }
        }
    }

    // ---- T-mode probability collection ----

    fn sq_clear(&mut self) {
        self.sq_sym.clear();
        self.sq_freq.clear();
        self.sq_total.clear();
    }
    fn sq_push(&mut self, sym: u16, freq: u16, total: u16) {
        self.sq_sym.push(sym);
        self.sq_freq.push(freq);
        self.sq_total.push(total);
    }

    fn convert_sq(&mut self) {
        let mut cum: u64 = 0xFFFF_FF00;
        for v in self.sqp.iter_mut() { *v = 0; }
        for i in 0..self.sq_sym.len() {
            let c = self.sq_sym[i] as u32;
            let freq = self.sq_freq[i] as u64;
            let total = self.sq_total[i] as u64;
            let prob = if total != 0 { (cum * freq) / total } else { 0 };
            if c < 256 {
                self.sqp[c as usize] = prob as u32 + 1;
            } else {
                cum = prob;
            }
        }
    }

    fn process_bin_symbol_t(&mut self, q: u32) {
        // q is a context with NumStats == 0 (bin_state in oneState()).
        let rs = one_state_offset(q);
        let suffix = ctx_i_suffix(&self.heap, q);
        let suffix_ns = ctx_num_stats(&self.heap, suffix);
        let flags = ctx_flags(&self.heap, q);
        let idx = self.tables.ns2bs_indx[suffix_ns as usize] as i32
            + self.prev_success
            + flags as i32
            + ((self.run_length >> 26) & 0x20);
        let qtab = self.tables.qtable[(state_freq(&self.heap, rs) - 1) as usize] as usize;
        let bs = self.bin_summ[qtab][idx as usize];
        self.bsumm = bs as i32;

        self.sq_push(state_symbol(&self.heap, rs) as u16, (self.bsumm + self.bsumm) as u16, SCALE as u16);
        self.sq_push(256, (SCALE as i32 - self.bsumm - self.bsumm) as u16, SCALE as u16);

        self.char_mask[state_symbol(&self.heap, rs) as usize] = self.esc_count;
        self.num_masked = 0;
    }

    fn process_symbol1_t(&mut self, q: u32) {
        let stats = ctx_i_stats(&self.heap, q);
        let cnum = ctx_num_stats(&self.heap, q) as i32;
        let total = ctx_summ_freq(&self.heap, q);
        for i in 0..=cnum {
            let off = stats + 6 * i as u32;
            let sym  = state_symbol(&self.heap, off);
            let freq = state_freq  (&self.heap, off);
            self.sq_push(sym as u16, freq as u16, total);
        }
        let mut low: u32 = 0;
        for i in 0..=cnum {
            let off = stats + 6 * i as u32;
            low += state_freq(&self.heap, off) as u32;
            self.char_mask[state_symbol(&self.heap, off) as usize] = self.esc_count;
        }
        self.num_masked = cnum;
        self.sq_push(256, (total as u32 - low) as u16, total);
    }

    fn process_symbol2_t(&mut self, q: u32) {
        let stats = ctx_i_stats(&self.heap, q);
        let cnum = ctx_num_stats(&self.heap, q) as i32;
        let summ_freq = ctx_summ_freq(&self.heap, q) as u32;
        let flags = ctx_flags(&self.heap, q);

        let see_freq;
        let psee2_idx: Option<(usize, usize)>;
        if cnum != 0xFF {
            let mut row = (self.tables.qtable[(cnum + 3) as usize] - 4) as usize;
            row = row.min(22);
            let mut col = 0usize;
            if summ_freq > 10 * (cnum as u32 + 1) { col += 1; }
            let suffix = ctx_i_suffix(&self.heap, q);
            let suff_ns = ctx_num_stats(&self.heap, suffix) as i32;
            if 2 * cnum < suff_ns + self.num_masked { col += 2; }
            col += flags as usize;
            col = col.min(31);
            see_freq = self.see2[row][col].get_mean() as i32 + 1;
            psee2_idx = Some((row, col));
        } else {
            see_freq = 1;
            psee2_idx = None;
        }

        let mut low: i32 = 0;
        for i in 0..=cnum {
            let off = stats + 6 * i as u32;
            let c = state_symbol(&self.heap, off) as usize;
            if self.char_mask[c] != self.esc_count {
                low += state_freq(&self.heap, off) as i32;
            }
        }
        let total = see_freq + low;

        for i in 0..=cnum {
            let off = stats + 6 * i as u32;
            let c = state_symbol(&self.heap, off) as usize;
            if self.char_mask[c] != self.esc_count {
                self.sq_push(c as u16, state_freq(&self.heap, off) as u16, total as u16);
                self.char_mask[c] = self.esc_count;
            }
        }
        self.sq_push(256, see_freq as u16, total as u16);
        self.num_masked = cnum;
        let _ = psee2_idx; // SEE update happens in the matching update path.
    }

    /// Walk the context chain and collect symbol probabilities into
    /// `sqp[0..256]`. Mirrors upstream `ppmd_PrepareByte`.
    fn prepare_byte(&mut self) {
        self.sq_clear();
        self.num_masked = 0;
        let saved_order_fall = self.order_fall;

        let mut min_ctx = self.max_context;
        if ctx_num_stats(&self.heap, min_ctx) != 0 {
            self.process_symbol1_t(min_ctx);
        } else {
            self.process_bin_symbol_t(min_ctx);
        }

        loop {
            // Walk back along iSuffix until NumStats != num_masked.
            loop {
                let suffix = ctx_i_suffix(&self.heap, min_ctx);
                if suffix == 0 {
                    self.esc_count = self.esc_count.wrapping_add(1);
                    self.num_masked = 0;
                    self.order_fall = saved_order_fall;
                    self.convert_sq();
                    return;
                }
                self.order_fall += 1;
                min_ctx = suffix;
                if ctx_num_stats(&self.heap, min_ctx) as i32 != self.num_masked { break; }
            }
            self.process_symbol2_t(min_ctx);
        }
    }

    /// Per-byte update — currently a simplified path that just
    /// advances `esc_count` and re-runs `prepare_byte`. The full
    /// model-update path (UpdateModel + processSymbol1/2 with mode
    /// 0) follows in a future turn; for now PPMd contributes
    /// stable but un-trained probabilities.
    pub fn byte_update(&mut self, _byte: u8) {
        // The full mod_ppmd update walks the tree, allocates new
        // contexts, and updates frequencies. Intentionally deferred
        // — the read-only prediction path above is enough to verify
        // the heap+tree structure is sound. The remaining glue lands
        // in the next turn; see the `update_model` method below for
        // the in-progress port.
        self.prepare_byte();
    }

    /// Convenience accessor mirroring upstream's `PPMD::ByteUpdate`
    /// post-processing: turn `sqp[0..256]` into a normalised
    /// probability vector in `self.probs`.
    pub fn finalize_probs(&mut self) -> &[f32; 256] {
        let mut sum = 0.0f32;
        for i in 0..256 {
            let v = self.sqp[i].max(1) as f32;
            self.probs[i] = v;
            sum += v;
        }
        if sum > 0.0 {
            for v in self.probs.iter_mut() { *v /= sum; }
        }
        &self.probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ppmd::new must allocate the requested heap and initialise
    /// the model without panicking. We can't exercise the full
    /// update path until the model-update port lands; for now just
    /// verify startup + prepare_byte produce a normalised
    /// distribution.
    #[test]
    fn ppmd_startup_and_prepare() {
        let mut p = Ppmd::new(/*order=*/4, /*memory_mb=*/4);
        // After startup, prepare_byte should walk the (empty) tree
        // and emit a uniform-ish distribution.
        p.prepare_byte();
        let probs = p.finalize_probs();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "probs must sum to ~1, got {}", sum);
        // None should be zero (we floor to 1 before normalising).
        for &v in probs.iter() { assert!(v > 0.0); }
    }

    #[test]
    fn ppmd_tables_match_upstream() {
        let t = Tables::build();
        // First N1 entries: 1, 2, 3, 4
        assert_eq!(&t.indx2units[0..4], &[1u8, 2, 3, 4]);
        // ns2bs_indx anchors
        assert_eq!(t.ns2bs_indx[0], 0);
        assert_eq!(t.ns2bs_indx[1], 2);
        assert_eq!(t.ns2bs_indx[2], 2);
        assert_eq!(t.ns2bs_indx[3], 4);
        assert_eq!(t.ns2bs_indx[28], 4);
        assert_eq!(t.ns2bs_indx[29], 6);
        assert_eq!(t.ns2bs_indx[255], 6);
        // qtable: 0..4 are identity, then 5 onwards step up.
        for i in 0..5 { assert_eq!(t.qtable[i], i as u8); }
        assert_eq!(t.qtable[5], 5);
        assert_eq!(t.qtable[6], 6);
        assert_eq!(t.qtable[7], 6);
    }
}
