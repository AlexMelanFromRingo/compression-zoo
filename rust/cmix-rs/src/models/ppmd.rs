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
        let i = self.tables.units2indx[(diff - 1) as usize] as u32;
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

    /// Coalesce adjacent free blocks. Mirrors upstream's
    /// `GlueFreeBlocks`. The algorithm:
    ///   1. Drain every BList[i] into a temporary linked list off
    ///      a stack-local sentinel.
    ///   2. While walking the temp list, if `p + p->NU` lands on
    ///      another free block (Stamp == ~0), absorb it.
    ///   3. Re-distribute by NU into the appropriate buckets,
    ///      splitting > 128-unit megablocks into 128-unit chunks
    ///      first.
    fn glue_free_blocks(&mut self) {
        // We can't write a sentinel "off the heap" cheaply — instead
        // we collect everything into a Vec<(ptr, nu)>, coalesce on
        // the heap, then redistribute. This deviates from upstream's
        // in-place linked-list dance but produces the same end state.

        // Drop a NUL terminator at LoUnit (upstream "if LoUnit!=HiUnit").
        if self.lo_unit != self.hi_unit {
            self.heap[self.lo_unit as usize] = 0;
        }

        let mut all: Vec<u32> = Vec::new();
        for i in 0..=N_INDEXES {
            while self.blist_next[i] != 0 {
                let p = self.remove_from_blist(i);
                if blk_nu(&self.heap, p) != 0 {
                    all.push(p);
                }
            }
        }

        // Coalesce: for each block, walk forward absorbing adjacent
        // (heap_addr == p + nu*UNIT_SIZE) Stamp==~0 blocks.
        for &p in &all {
            loop {
                let nu = blk_nu(&self.heap, p);
                let next_addr = p + u2b(nu);
                if next_addr as usize + 12 > self.heap.len() { break; }
                if blk_stamp(&self.heap, next_addr) != !0u32 { break; }
                let next_nu = blk_nu(&self.heap, next_addr);
                if next_nu == 0 { break; }
                blk_set_nu(&mut self.heap, p, nu + next_nu);
                blk_set_nu(&mut self.heap, next_addr, 0);
                blk_set_stamp(&mut self.heap, next_addr, 0);
            }
        }

        // Redistribute back into the buckets.
        for &raw_p in &all {
            let mut p = raw_p;
            let mut sz = blk_nu(&self.heap, p);
            if sz == 0 { continue; }
            // Cleave megablocks into 128-unit chunks.
            while sz > 128 {
                self.insert_into_blist(p, N_INDEXES - 1, 128);
                p += u2b(128);
                sz -= 128;
            }
            let mut i = self.tables.units2indx[(sz - 1) as usize] as usize;
            if self.tables.indx2units[i] as u32 != sz {
                let bucket_units = self.tables.indx2units[i - 1] as u32;
                let k = sz - bucket_units;
                self.insert_into_blist(p + u2b(sz - k), (k - 1) as usize, k);
                i -= 1;
            }
            self.insert_into_blist(p, i, self.tables.indx2units[i] as u32);
        }

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

    fn units_cpy(&mut self, dest: u32, src: u32, nu: u32) {
        let n = u2b(nu) as usize;
        // Need split borrow because src and dest live in the same Vec.
        let (lo, hi, lo_off, hi_off) = if dest < src {
            (dest, src, 0usize, 0usize)
        } else {
            (src, dest, 0usize, 0usize)
        };
        let _ = (lo, hi, lo_off, hi_off);
        // Easiest path: copy via a small temporary buffer.
        let mut buf = vec![0u8; n];
        buf.copy_from_slice(&self.heap[src as usize..src as usize + n]);
        self.heap[dest as usize..dest as usize + n].copy_from_slice(&buf);
    }

    fn expand_units(&mut self, old_ptr: u32, old_nu: u32) -> u32 {
        let i0 = self.tables.units2indx[(old_nu - 1) as usize];
        let i1 = self.tables.units2indx[old_nu as usize];
        if i0 == i1 { return old_ptr; }
        let ptr = self.alloc_units(old_nu + 1);
        if ptr != 0 {
            self.units_cpy(ptr, old_ptr, old_nu);
            self.insert_into_blist(old_ptr, i0 as usize,
                self.tables.indx2units[i0 as usize] as u32);
        }
        ptr
    }

    fn shrink_units(&mut self, old_ptr: u32, old_nu: u32, new_nu: u32) -> u32 {
        let i0 = self.tables.units2indx[(old_nu - 1) as usize] as usize;
        let i1 = self.tables.units2indx[(new_nu - 1) as usize] as usize;
        if i0 == i1 { return old_ptr; }
        if self.blist_next[i1] != 0 {
            let ptr = self.remove_from_blist(i1);
            self.units_cpy(ptr, old_ptr, new_nu);
            self.insert_into_blist(old_ptr, i0,
                self.tables.indx2units[i0] as u32);
            ptr
        } else {
            self.split_block(old_ptr, i0 as u32, i1 as u32);
            old_ptr
        }
    }

    fn move_units_up(&mut self, old_ptr: u32, nu: u32) -> u32 {
        let indx = self.tables.units2indx[(nu - 1) as usize] as usize;
        if old_ptr > self.units_start + 128 * 1024
            || old_ptr > self.blist_next[indx]
        {
            return old_ptr;
        }
        let ptr = self.remove_from_blist(indx);
        self.units_cpy(ptr, old_ptr, nu);
        self.insert_into_blist(old_ptr, N_INDEXES,
            self.tables.indx2units[indx] as u32);
        ptr
    }

    fn prepare_text_area(&mut self) {
        self.aux_unit = self.alloc_context();
        if self.aux_unit == 0 {
            self.aux_unit = self.units_start;
        } else if self.aux_unit == self.units_start {
            self.units_start += UNIT_SIZE;
            self.aux_unit = self.units_start;
        }
    }

    fn expand_text_area(&mut self) {
        let mut count = [0u32; N_INDEXES];

        if self.aux_unit != self.units_start {
            // *(uint*)AuxUnit != ~uint(0) ?
            if get_u32(&self.heap, self.aux_unit) != !0u32 {
                self.units_start += UNIT_SIZE;
            } else {
                self.insert_into_blist(self.aux_unit, 0, 1);
            }
        }

        // While first units_start block has Stamp==~0, absorb it.
        loop {
            let p = self.units_start;
            if p as usize + 12 > self.heap.len() { break; }
            if blk_stamp(&self.heap, p) != !0u32 { break; }
            let nu = blk_nu(&self.heap, p);
            self.units_start = p + u2b(nu);
            count[self.tables.units2indx[(nu - 1) as usize] as usize] += 1;
            blk_set_stamp(&mut self.heap, p, 0);
        }

        // Walk the N_INDEXES (last) bucket and remove any zero-stamp
        // entries, decrementing count and stamps[N_INDEXES].
        // (The full upstream logic uses linked-list traversal; we
        // rebuild the list while filtering.)
        if count.iter().any(|&c| c != 0) {
            let mut head = self.blist_next[N_INDEXES];
            let mut new_head: u32 = 0;
            let mut removed: u32 = 0;
            while head != 0 {
                let next = blk_next(&self.heap, head);
                if blk_stamp(&self.heap, head) == 0 {
                    removed += 1;
                } else {
                    blk_set_next(&mut self.heap, head, new_head);
                    new_head = head;
                }
                head = next;
            }
            self.blist_next[N_INDEXES] = new_head;
            self.blist_stamp[N_INDEXES] = self.blist_stamp[N_INDEXES].wrapping_sub(removed);

            for i in 0..N_INDEXES {
                if count[i] == 0 { continue; }
                let mut h = self.blist_next[i];
                let mut nh: u32 = 0;
                let mut left = count[i];
                let mut removed: u32 = 0;
                while h != 0 && left > 0 {
                    let next = blk_next(&self.heap, h);
                    if blk_stamp(&self.heap, h) == 0 {
                        removed += 1;
                        left -= 1;
                    } else {
                        blk_set_next(&mut self.heap, h, nh);
                        nh = h;
                    }
                    h = next;
                }
                // Append remaining (rest of original list) untouched.
                while h != 0 {
                    let next = blk_next(&self.heap, h);
                    blk_set_next(&mut self.heap, h, nh);
                    nh = h;
                    h = next;
                }
                self.blist_next[i] = nh;
                self.blist_stamp[i] = self.blist_stamp[i].wrapping_sub(removed);
            }
        }
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

    // ---- Update-path helpers (mode 0 = actually train the model) ----

    /// Rescale a context's stats: halve every freq, re-sort, prune
    /// zeros, possibly collapse to a single-state context. Mirrors
    /// upstream `STATE* rescale(...)`.
    fn rescale(&mut self, q: u32, found_state: u32) -> u32 {
        // Move found_state to the head of its stats list.
        let cur_flags = ctx_flags(&self.heap, q);
        ctx_set_flags(&mut self.heap, q, cur_flags & 0x14);
        let p1 = ctx_i_stats(&self.heap, q);
        // tmp = found_state[0]; shift everything down to make room
        // at p1 for the (formerly) found_state.
        let mut tmp = [0u8; 6];
        for i in 0..6 { tmp[i] = self.heap[(found_state + i as u32) as usize]; }
        let mut p = found_state;
        while p != p1 {
            // p[0] = p[-1]
            for i in 0..6 {
                self.heap[(p + i as u32) as usize]
                    = self.heap[(p - 6 + i as u32) as usize];
            }
            p -= 6;
        }
        for i in 0..6 { self.heap[(p1 + i as u32) as usize] = tmp[i]; }

        let order_fall = self.order_fall;
        let of: i32 = if order_fall != 0 { 1 } else { 0 };
        let f0 = state_freq(&self.heap, p1) as i32;
        let sf_initial = ctx_summ_freq(&self.heap, q) as i32;
        let mut sf = sf_initial;
        let mut esc_freq = sf - f0;
        let new_f0 = ((f0 + of) >> 1) as u8;
        state_set_freq(&mut self.heap, p1, new_f0);
        ctx_set_summ_freq(&mut self.heap, q, new_f0 as u16);

        let num_stats = ctx_num_stats(&self.heap, q) as i32;
        let mut p = p1;
        for _ in 0..num_stats {
            p += 6;
            let mut a = state_freq(&self.heap, p) as i32;
            esc_freq -= a;
            a = (a + of) >> 1;
            state_set_freq(&mut self.heap, p, a as u8);
            let mut sumf = ctx_summ_freq(&self.heap, q) as i32;
            sumf += a;
            ctx_set_summ_freq(&mut self.heap, q, sumf as u16);
            if a != 0 {
                let sym = state_symbol(&self.heap, p);
                if sym >= 0x40 {
                    let cf = ctx_flags(&self.heap, q);
                    ctx_set_flags(&mut self.heap, q, cf | 0x08);
                }
            }
            if a > state_freq(&self.heap, p - 6) as i32 {
                // Bubble up.
                let mut tmp = [0u8; 6];
                for i in 0..6 { tmp[i] = self.heap[(p + i as u32) as usize]; }
                let mut p1c = p;
                while p1c > q + 4 + 6 {  // > stats start
                    let prev = p1c - 6;
                    if tmp[1] as i32 <= state_freq(&self.heap, prev) as i32 { break; }
                    for i in 0..6 {
                        self.heap[(p1c + i as u32) as usize]
                            = self.heap[(prev + i as u32) as usize];
                    }
                    p1c = prev;
                }
                for i in 0..6 { self.heap[(p1c + i as u32) as usize] = tmp[i]; }
            }
        }

        // Trim tail zeros.
        if state_freq(&self.heap, p) == 0 {
            let mut zero_count = 0i32;
            while state_freq(&self.heap, p) == 0 {
                zero_count += 1;
                p -= 6;
            }
            esc_freq += zero_count;
            let a_units = (num_stats + 2) >> 1;
            let new_ns = num_stats - zero_count;
            ctx_set_num_stats(&mut self.heap, q, new_ns as u8);
            if new_ns == 0 {
                let stats = ctx_i_stats(&self.heap, q);
                let s_freq = state_freq(&self.heap, stats);
                let s_sym  = state_symbol(&self.heap, stats);
                let s_succ = state_succ(&self.heap, stats);
                let new_freq = (((2 * s_freq as i32 + esc_freq - 1) / esc_freq.max(1))
                    .min(MAX_FREQ as i32 / 3)) as u8;
                let new_flags = ctx_flags(&self.heap, q) & 0x18;
                ctx_set_flags(&mut self.heap, q, new_flags);
                self.free_units(stats, a_units as u32);
                let one = one_state_offset(q);
                state_set_symbol(&mut self.heap, one, s_sym);
                state_set_freq  (&mut self.heap, one, new_freq);
                state_set_succ  (&mut self.heap, one, s_succ);
                return one;
            }
            let shrunk = self.shrink_units(
                ctx_i_stats(&self.heap, q),
                a_units as u32,
                ((new_ns + 2) >> 1) as u32,
            );
            ctx_set_i_stats(&mut self.heap, q, shrunk);
        }

        let new_summ = ctx_summ_freq(&self.heap, q) as i32 + ((esc_freq + 1) >> 1);
        ctx_set_summ_freq(&mut self.heap, q, new_summ as u16);

        let order_fall = self.order_fall;
        let flags_q = ctx_flags(&self.heap, q);
        let stats_q = ctx_i_stats(&self.heap, q);
        let head_freq = state_freq(&self.heap, stats_q) as i32;
        let a;
        if order_fall != 0 || (flags_q & 0x04) == 0 {
            sf -= esc_freq;
            let denom = (sf - f0).max(1);
            a = ((f0 * (ctx_summ_freq(&self.heap, q) as i32)
                  - sf * head_freq + denom - 1) / denom)
                .clamp(2, MAX_FREQ as i32 / 2 - 18) as u8;
        } else {
            a = 2;
        }
        let new_head = state_freq(&self.heap, stats_q).saturating_add(a);
        state_set_freq(&mut self.heap, stats_q, new_head);
        let new_summ = ctx_summ_freq(&self.heap, q) as i32 + a as i32;
        ctx_set_summ_freq(&mut self.heap, q, new_summ as u16);
        ctx_set_flags(&mut self.heap, q, flags_q | 0x04);
        stats_q
    }

    /// `processBinSymbol<0>` — update path for binary contexts.
    fn process_bin_symbol(&mut self, q: u32, symbol: u8) {
        let rs = one_state_offset(q);
        let suffix = ctx_i_suffix(&self.heap, q);
        let suffix_ns = ctx_num_stats(&self.heap, suffix);
        let flags_q = ctx_flags(&self.heap, q);
        let idx = self.tables.ns2bs_indx[suffix_ns as usize] as i32
            + self.prev_success
            + flags_q as i32
            + ((self.run_length >> 26) & 0x20);
        let qtab = self.tables.qtable[(state_freq(&self.heap, rs) - 1) as usize] as usize;
        let bs = self.bin_summ[qtab][idx as usize] as i32;
        self.bsumm = bs;
        // Apply BinSumm decay.
        self.bin_summ[qtab][idx as usize] =
            (bs as i32 - ((bs + 64) >> PERIOD_BITS)) as u16;

        let rs_sym = state_symbol(&self.heap, rs);
        if rs_sym != symbol {
            self.char_mask[rs_sym as usize] = self.esc_count;
            self.num_masked = 0;
            self.prev_success = 0;
            self.found_state = 0;
        } else {
            // Boost the BinSumm entry, increment freq up to 196,
            // bump run-length, mark success.
            let new_bs = self.bin_summ[qtab][idx as usize] as i32 + INTERVAL as i32;
            self.bin_summ[qtab][idx as usize] = new_bs as u16;
            let f = state_freq(&self.heap, rs);
            if f < 196 { state_set_freq(&mut self.heap, rs, f + 1); }
            self.run_length = self.run_length.wrapping_add(1);
            self.prev_success = 1;
            self.found_state = rs;
        }
    }

    /// `processSymbol1<0>` — update path for n_stats > 0 contexts.
    fn process_symbol1(&mut self, q: u32, symbol: u8) {
        let stats = ctx_i_stats(&self.heap, q);
        let cnum = ctx_num_stats(&self.heap, q) as i32;
        let p0 = stats;
        let head_sym = state_symbol(&self.heap, p0);

        if head_sym == symbol {
            self.prev_success = 0;
            let f = state_freq(&self.heap, p0).saturating_add(4);
            state_set_freq(&mut self.heap, p0, f);
            let new_summ = ctx_summ_freq(&self.heap, q).saturating_add(4);
            ctx_set_summ_freq(&mut self.heap, q, new_summ);
            self.found_state = p0;
            if f > MAX_FREQ {
                self.found_state = self.rescale(q, p0);
            }
            return;
        }

        self.prev_success = 0;
        let mut found = 0u32;
        let mut found_idx: i32 = -1;
        for i in 1..=cnum {
            let p = stats + 6 * i as u32;
            if state_symbol(&self.heap, p) == symbol {
                found = p;
                found_idx = i;
                break;
            }
        }

        if found != 0 {
            let f = state_freq(&self.heap, found).saturating_add(4);
            state_set_freq(&mut self.heap, found, f);
            let new_summ = ctx_summ_freq(&self.heap, q).saturating_add(4);
            ctx_set_summ_freq(&mut self.heap, q, new_summ);
            // If we beat the previous-position freq, swap up.
            let prev = found - 6;
            if state_freq(&self.heap, found) > state_freq(&self.heap, prev) {
                let mut tmp = [0u8; 6];
                for i in 0..6 { tmp[i] = self.heap[(found + i as u32) as usize]; }
                for i in 0..6 {
                    self.heap[(found + i as u32) as usize]
                        = self.heap[(prev + i as u32) as usize];
                }
                for i in 0..6 { self.heap[(prev + i as u32) as usize] = tmp[i]; }
                self.found_state = prev;
            } else {
                self.found_state = found;
            }
            if f > MAX_FREQ {
                self.found_state = self.rescale(q, self.found_state);
            }
            let _ = found_idx;
        } else {
            // Symbol not in this context's stats — mask all and
            // recurse outward.
            self.num_masked = cnum;
            for i in 0..=cnum {
                let p = stats + 6 * i as u32;
                self.char_mask[state_symbol(&self.heap, p) as usize] = self.esc_count;
            }
            self.found_state = 0;
        }
    }

    /// `processSymbol2<0>` — update path for masked contexts.
    fn process_symbol2(&mut self, q: u32, symbol: u8) {
        let stats = ctx_i_stats(&self.heap, q);
        let cnum = ctx_num_stats(&self.heap, q) as i32;
        let summ_freq = ctx_summ_freq(&self.heap, q) as u32;
        let flags_q = ctx_flags(&self.heap, q);

        // SEE lookup.
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
            col += flags_q as usize;
            col = col.min(31);
            see_freq = self.see2[row][col].get_mean() as i32 + 1;
            psee2_idx = Some((row, col));
        } else {
            see_freq = 1;
            psee2_idx = None;
        }

        let mut low = 0i32;
        let mut found = 0u32;
        for i in 0..=cnum {
            let p = stats + 6 * i as u32;
            let c = state_symbol(&self.heap, p);
            if self.char_mask[c as usize] != self.esc_count {
                self.char_mask[c as usize] = self.esc_count;
                low += state_freq(&self.heap, p) as i32;
                if c == symbol { found = p; }
            }
        }
        let total = see_freq + low;

        if found != 0 {
            if let Some((r, c)) = psee2_idx {
                if see_freq > 2 { self.see2[r][c].summ -= see_freq as u16; }
                self.see2[r][c].update();
            }
            let f = state_freq(&self.heap, found).saturating_add(4);
            state_set_freq(&mut self.heap, found, f);
            let new_summ = ctx_summ_freq(&self.heap, q).saturating_add(4);
            ctx_set_summ_freq(&mut self.heap, q, new_summ);
            self.found_state = found;
            if f > MAX_FREQ {
                self.found_state = self.rescale(q, found);
            }
            self.run_length = self.init_rl;
            self.esc_count = self.esc_count.wrapping_add(1);
        } else {
            self.num_masked = cnum;
            if let Some((r, c)) = psee2_idx {
                self.see2[r][c].summ = (self.see2[r][c].summ as i32 + (total - see_freq)) as u16;
            }
            let _ = total;
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

    /// `CreateSuccessors` — create new contexts for higher orders
    /// when a symbol is seen. Returns the resulting context offset.
    fn create_successors(&mut self, skip: bool, p_state: u32, mut pc: u32) -> u32 {
        let mut ps: [u32; MAX_O] = [0; MAX_O];
        let mut pps_n: usize = 0;

        let sym = state_symbol(&self.heap, self.found_state);
        let i_up_branch = state_succ(&self.heap, self.found_state);

        if !skip {
            ps[pps_n] = self.found_state;
            pps_n += 1;
            if ctx_i_suffix(&self.heap, pc) == 0 {
                return self.create_successors_no_loop(ps, pps_n, pc, sym, i_up_branch);
            }
        }

        let mut p = p_state;
        let mut entered = p_state != 0;
        if entered { pc = ctx_i_suffix(&self.heap, pc); }

        loop {
            if !entered {
                pc = ctx_i_suffix(&self.heap, pc);
                if ctx_num_stats(&self.heap, pc) != 0 {
                    let stats = ctx_i_stats(&self.heap, pc);
                    p = stats;
                    while state_symbol(&self.heap, p) != sym { p += 6; }
                    let f = state_freq(&self.heap, p);
                    let bump = if f < MAX_FREQ - 1 { 2 } else { 0 };
                    state_set_freq(&mut self.heap, p, f + bump);
                    let new_summ = ctx_summ_freq(&self.heap, pc).saturating_add(bump as u16);
                    ctx_set_summ_freq(&mut self.heap, pc, new_summ);
                } else {
                    p = one_state_offset(pc);
                    let suff_pc = ctx_i_suffix(&self.heap, pc);
                    let suff_ns = ctx_num_stats(&self.heap, suff_pc);
                    let f = state_freq(&self.heap, p);
                    if suff_ns == 0 && f < 16 {
                        state_set_freq(&mut self.heap, p, f + 1);
                    }
                }
            }
            entered = false;

            if state_succ(&self.heap, p) != i_up_branch {
                pc = state_succ(&self.heap, p);
                break;
            }
            ps[pps_n] = p;
            pps_n += 1;
            if ctx_i_suffix(&self.heap, pc) == 0 { break; }
        }

        self.create_successors_no_loop(ps, pps_n, pc, sym, i_up_branch)
    }

    fn create_successors_no_loop(
        &mut self,
        ps: [u32; MAX_O],
        pps_n: usize,
        mut pc: u32,
        sym: u8,
        i_up_branch: u32,
    ) -> u32 {
        if pps_n == 0 { return pc; }

        // Build a temp PPM_CONTEXT (12 bytes).
        let mut ct = [0u8; 12];
        // NumStats = 0
        ct[0] = 0;
        let upbyte_addr = i_up_branch;
        let next_byte = self.heap[upbyte_addr as usize];
        ct[1] = 0x10 * (sym >= 0x40) as u8 | 0x08 * (next_byte >= 0x40) as u8;
        // oneState() at offset 2: Symbol, Freq, iSuccessor.
        ct[2] = next_byte;
        // ct[3] = freq, set later.
        let succ = i_up_branch + 1;
        ct[4..8].copy_from_slice(&succ.to_le_bytes());
        // ct[8..12] = iSuffix, set later per allocated context.

        let cf;
        if ctx_num_stats(&self.heap, pc) != 0 {
            let stats = ctx_i_stats(&self.heap, pc);
            let mut p = stats;
            while state_symbol(&self.heap, p) != sym { p += 6; }
            let cf_v = state_freq(&self.heap, p) as i32 - 1;
            let s0 = ctx_summ_freq(&self.heap, pc) as i32
                - ctx_num_stats(&self.heap, pc) as i32 - cf_v;
            cf = if 2 * cf_v < s0 {
                1 + (12 * cf_v > s0) as i32
            } else {
                1 + 2 + cf_v / s0.max(1)
            };
        } else {
            cf = state_freq(&self.heap, one_state_offset(pc)) as i32;
        }
        ct[3] = cf.min(7) as u8;

        let mut chain_count = pps_n;
        while chain_count > 0 {
            let pc1 = self.alloc_context();
            if pc1 == 0 { return 0; }
            for i in 0..12 {
                self.heap[(pc1 + i as u32) as usize] = ct[i];
            }
            ctx_set_i_suffix(&mut self.heap, pc1, pc);
            pc = pc1;
            chain_count -= 1;
            state_set_succ(&mut self.heap, ps[chain_count], pc);
        }
        pc
    }

    /// `ReduceOrder` — fall back to the lowest matching context.
    fn reduce_order(&mut self, mut p_state: u32, pc_in: u32) -> u32 {
        let mut pc = pc_in;
        let pc1 = pc;
        state_set_succ(&mut self.heap, self.found_state, self.text_ptr);
        let sym = state_symbol(&self.heap, self.found_state);
        let i_up_branch = state_succ(&self.heap, self.found_state);
        self.order_fall += 1;

        let mut entered = p_state != 0;
        if entered { pc = ctx_i_suffix(&self.heap, pc); }

        loop {
            if !entered {
                if ctx_i_suffix(&self.heap, pc) == 0 { return pc; }
                pc = ctx_i_suffix(&self.heap, pc);
                if ctx_num_stats(&self.heap, pc) != 0 {
                    let stats = ctx_i_stats(&self.heap, pc);
                    p_state = stats;
                    while state_symbol(&self.heap, p_state) != sym { p_state += 6; }
                    let f = state_freq(&self.heap, p_state);
                    let bump = if f < MAX_FREQ - 3 { 2 } else { 0 };
                    state_set_freq(&mut self.heap, p_state, f + bump);
                    let new_summ = ctx_summ_freq(&self.heap, pc).saturating_add(bump as u16);
                    ctx_set_summ_freq(&mut self.heap, pc, new_summ);
                } else {
                    p_state = one_state_offset(pc);
                    let f = state_freq(&self.heap, p_state);
                    if f < 11 { state_set_freq(&mut self.heap, p_state, f + 1); }
                }
            }
            entered = false;
            if state_succ(&self.heap, p_state) != 0 { break; }
            state_set_succ(&mut self.heap, p_state, i_up_branch);
            self.order_fall += 1;
        }

        let succ = state_succ(&self.heap, p_state);
        if succ <= i_up_branch {
            let saved = self.found_state;
            self.found_state = p_state;
            let new_succ = self.create_successors(false, 0, pc);
            state_set_succ(&mut self.heap, p_state, new_succ);
            self.found_state = saved;
        }

        if self.order_fall == 1 && pc1 == self.max_context {
            let new_succ = state_succ(&self.heap, p_state);
            state_set_succ(&mut self.heap, self.found_state, new_succ);
            self.text_ptr = self.text_ptr.saturating_sub(1);
        }

        state_succ(&self.heap, p_state)
    }

    /// `UpdateModel` — extend the context tree with the new symbol.
    fn update_model(&mut self, min_context: u32) -> u32 {
        let f_symbol = state_symbol(&self.heap, self.found_state);
        let f_freq = state_freq(&self.heap, self.found_state);
        let i_f_succ = state_succ(&self.heap, self.found_state);

        let mut p_state: u32 = 0;
        let mut pc: u32 = 0;
        if ctx_i_suffix(&self.heap, min_context) != 0 {
            pc = ctx_i_suffix(&self.heap, min_context);
            if ctx_num_stats(&self.heap, pc) != 0 {
                let stats = ctx_i_stats(&self.heap, pc);
                let mut p = stats;
                if state_symbol(&self.heap, p) != f_symbol {
                    p += 6;
                    while state_symbol(&self.heap, p) != f_symbol { p += 6; }
                    if state_freq(&self.heap, p) >= state_freq(&self.heap, p - 6) {
                        let mut tmp = [0u8; 6];
                        for i in 0..6 { tmp[i] = self.heap[(p + i as u32) as usize]; }
                        for i in 0..6 {
                            self.heap[(p + i as u32) as usize]
                                = self.heap[(p - 6 + i as u32) as usize];
                        }
                        for i in 0..6 { self.heap[(p - 6 + i as u32) as usize] = tmp[i]; }
                        p -= 6;
                    }
                }
                if state_freq(&self.heap, p) < MAX_FREQ - 3 {
                    let cf = 2 + (f_freq < 28) as u8;
                    let new_f = state_freq(&self.heap, p) + cf;
                    state_set_freq(&mut self.heap, p, new_f);
                    let new_summ = ctx_summ_freq(&self.heap, pc).saturating_add(cf as u16);
                    ctx_set_summ_freq(&mut self.heap, pc, new_summ);
                }
                p_state = p;
            } else {
                p_state = one_state_offset(pc);
                let f = state_freq(&self.heap, p_state);
                if f < 14 { state_set_freq(&mut self.heap, p_state, f + 1); }
            }
        }

        if self.order_fall == 0 && i_f_succ != 0 {
            let new_succ = self.create_successors(true, p_state, min_context);
            state_set_succ(&mut self.heap, self.found_state, new_succ);
            if new_succ == 0 {
                self.saved_pc = pc;
                return 0;
            }
            self.max_context = state_succ(&self.heap, self.found_state);
            return self.max_context;
        }

        // Append the symbol to text area.
        if (self.text_ptr as usize) < self.heap.len() {
            self.heap[self.text_ptr as usize] = f_symbol;
        }
        self.text_ptr += 1;
        let mut i_succ = self.text_ptr;
        if self.text_ptr >= self.units_start {
            self.saved_pc = pc;
            return 0;
        }

        let mut i_f_succ_local = i_f_succ;
        if i_f_succ_local != 0 {
            if i_f_succ_local < self.units_start {
                let new_succ = self.create_successors(false, p_state, min_context);
                i_f_succ_local = new_succ;
            }
        } else {
            i_f_succ_local = self.reduce_order(p_state, min_context);
        }

        if i_f_succ_local == 0 {
            self.saved_pc = pc;
            return 0;
        }

        if self.order_fall > 0 {
            self.order_fall -= 1;
            if self.order_fall == 0 {
                i_succ = i_f_succ_local;
                if self.max_context != min_context {
                    self.text_ptr -= 1;
                }
            }
        }

        let s0 = (ctx_summ_freq(&self.heap, min_context) as i32) - f_freq as i32;
        let ns = ctx_num_stats(&self.heap, min_context) as i32;
        let flag = 0x08 * (f_symbol >= 0x40) as u8;

        let mut walk = self.max_context;
        while walk != min_context {
            let ns1 = ctx_num_stats(&self.heap, walk) as i32;
            if ns1 != 0 {
                if ns1 & 1 != 0 {
                    let stats = ctx_i_stats(&self.heap, walk);
                    let p = self.expand_units(stats, ((ns1 + 1) >> 1) as u32);
                    if p == 0 { self.saved_pc = walk; return 0; }
                    ctx_set_i_stats(&mut self.heap, walk, p);
                }
                let q_inc = self.tables.qtable[(ns + 4) as usize] as i32 >> 3;
                let new_summ = ctx_summ_freq(&self.heap, walk).saturating_add(q_inc as u16);
                ctx_set_summ_freq(&mut self.heap, walk, new_summ);
            } else {
                let p = self.alloc_units(1);
                if p == 0 { self.saved_pc = walk; return 0; }
                let one = one_state_offset(walk);
                for i in 0..6 {
                    self.heap[(p + i as u32) as usize] = self.heap[(one + i as u32) as usize];
                }
                ctx_set_i_stats(&mut self.heap, walk, p);
                let f = state_freq(&self.heap, p);
                let new_freq = if f <= MAX_FREQ / 3 {
                    (2 * f).saturating_sub(1)
                } else {
                    MAX_FREQ - 15
                };
                state_set_freq(&mut self.heap, p, new_freq);
                let exp_idx = self.tables.qtable[(self.bsumm >> 8) as usize] as usize;
                let exp_e = EXP_ESCAPE[exp_idx.min(15)] as u16;
                let new_summ_freq = new_freq as u16 + (ns > 1) as u16 + exp_e;
                ctx_set_summ_freq(&mut self.heap, walk, new_summ_freq);
            }

            let pc_summ = ctx_summ_freq(&self.heap, walk) as i32;
            let cf_init = (f_freq as i32 - 1) * (5 + pc_summ);
            let sf = s0 + pc_summ;
            let cf;
            if cf_init <= 3 * sf {
                cf = 1 + (2 * cf_init > sf) as i32 + (2 * cf_init > 3 * sf) as i32;
                let nf = ctx_summ_freq(&self.heap, walk).saturating_add(4);
                ctx_set_summ_freq(&mut self.heap, walk, nf);
            } else {
                cf = 5
                    + (cf_init > 5 * sf) as i32
                    + (cf_init > 6 * sf) as i32
                    + (cf_init > 8 * sf) as i32
                    + (cf_init > 10 * sf) as i32
                    + (cf_init > 12 * sf) as i32;
                let nf = ctx_summ_freq(&self.heap, walk).saturating_add(cf as u16);
                ctx_set_summ_freq(&mut self.heap, walk, nf);
            }

            let new_ns = ctx_num_stats(&self.heap, walk) + 1;
            ctx_set_num_stats(&mut self.heap, walk, new_ns);
            let stats = ctx_i_stats(&self.heap, walk);
            let p = stats + 6 * new_ns as u32;
            state_set_succ  (&mut self.heap, p, i_succ);
            state_set_symbol(&mut self.heap, p, f_symbol);
            state_set_freq  (&mut self.heap, p, cf as u8);
            let cf_w = ctx_flags(&self.heap, walk);
            ctx_set_flags(&mut self.heap, walk, cf_w | flag);

            walk = ctx_i_suffix(&self.heap, walk);
        }

        self.max_context = i_f_succ_local;
        self.max_context
    }

    /// Public per-byte update: consumes the just-(en|de)coded byte,
    /// advances the model, then re-runs `prepare_byte` so the new
    /// byte distribution is ready for the next call to
    /// [`Self::finalize_probs`].
    pub fn byte_update(&mut self, byte: u8) {
        let mut min_ctx = self.max_context;
        if ctx_num_stats(&self.heap, min_ctx) != 0 {
            self.process_symbol1(min_ctx, byte);
        } else {
            self.process_bin_symbol(min_ctx, byte);
        }

        while self.found_state == 0 {
            // Walk back along iSuffix until we find a context with
            // a previously-unmasked symbol.
            loop {
                self.order_fall += 1;
                let suffix = ctx_i_suffix(&self.heap, min_ctx);
                if suffix == 0 {
                    // Fall back to a fresh start; PPMd treats this
                    // as a model restart.
                    self.start_model_rare();
                    self.prepare_byte();
                    return;
                }
                min_ctx = suffix;
                if ctx_num_stats(&self.heap, min_ctx) as i32 != self.num_masked { break; }
            }
            self.process_symbol2(min_ctx, byte);
        }

        if self.order_fall != 0
            || state_succ(&self.heap, self.found_state) < self.units_start
        {
            let p = self.update_model(min_ctx);
            if p != 0 { self.max_context = p; }
            else if self.cut_off != 0 {
                // Out of memory: full model restart for now (a real
                // RestoreModelRare comes once the cut-off path lands).
                self.start_model_rare();
            } else {
                self.start_model_rare();
            }
        } else {
            self.max_context = state_succ(&self.heap, self.found_state);
        }

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

    /// Feed a repetitive sequence ("ababab...") and verify that
    /// after training, the model heavily predicts the next char.
    /// Pattern ends with 'b', so 'a' (which always follows 'b' in
    /// the pattern) should be the dominant prediction.
    #[test]
    fn ppmd_learns_repetitive_input() {
        let mut p = Ppmd::new(/*order=*/4, /*memory_mb=*/4);
        let pattern = b"ababababababababababab";
        for &b in pattern { p.byte_update(b); }
        let probs = p.finalize_probs();
        let p_a = probs[b'a' as usize];
        let p_z = probs[b'z' as usize];
        let unif = 1.0 / 256.0;
        // 'a' always follows 'b' — should dominate the distribution.
        assert!(p_a > unif * 4.0,
            "p(a) = {} should be well above uniform {}", p_a, unif);
        // 'z' was never seen; should be essentially baseline.
        assert!(p_z < p_a,
            "p(z) = {} should be less than p(a) = {}", p_z, p_a);
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
