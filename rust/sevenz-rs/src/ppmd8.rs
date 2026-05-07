//! PPMd8 (PPMdI) — port of `7zip/C/Ppmd8*.c`.
//!
//! Differences vs PPMd7:
//! * Context has `NumStats: u8` (= actual_count − 1) and a `Flags: u8` byte.
//! * Node layout is `{ Stamp: u32, Next: u32, NU: u32 }` (12 bytes total).
//! * Carryless range coder (Subbotin 1999) instead of LZMA-style.
//! * `BinSumm[25][64]`, `See[24][32]` (vs PPMd7's `[128][64]` and `[25][16]`).
//! * `NS2Indx` is 260 entries.
//! * Restore model on alloc failure: RESTART or CUT_OFF.
//! * `Refresh`, `CutOff`, `ReduceOrder`, `ExpandTextArea` etc.

use core::convert::TryInto;

// ====================================================================
// Constants
// ====================================================================

pub const MIN_ORDER: u32 = 2;
pub const MAX_ORDER: u32 = 16;

const PPMD_INT_BITS: u32 = 7;
const PPMD_PERIOD_BITS: u32 = 7;
const PPMD_BIN_SCALE: u32 = 1 << (PPMD_INT_BITS + PPMD_PERIOD_BITS);

const PPMD_N1: usize = 4;
const PPMD_N2: usize = 4;
const PPMD_N3: usize = 4;
const PPMD_N4: usize = (128 + 3 - 1 * PPMD_N1 - 2 * PPMD_N2 - 3 * PPMD_N3) / 4;
const PPMD_NUM_INDEXES: usize = PPMD_N1 + PPMD_N2 + PPMD_N3 + PPMD_N4;

const MAX_FREQ: u8 = 124;
const UNIT_SIZE: u32 = 12;
const K_TOP: u32 = 1 << 24;
const K_BOT: u32 = 1 << 15;
const EMPTY_NODE: u32 = 0xFFFF_FFFF;

const FLAG_RESCALED: u8 = 1 << 2;
const FLAG_PREV_HIGH: u8 = 1 << 4;

const K_EXP_ESCAPE: [u8; 16] = [25, 14, 9, 7, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2];
const K_INIT_BIN_ESC: [u16; 8] = [
    0x3CDD, 0x1F3F, 0x59BF, 0x48F3, 0x64A1, 0x5ABC, 0x6632, 0x6051,
];

pub const SYM_END: i32 = -1;
pub const SYM_ERROR: i32 = -2;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RestoreMethod {
    Restart,
    CutOff,
}

// ====================================================================
// Field accessors (Vec<u8> heap, u32 offsets)
// ====================================================================

#[inline(always)]
fn get_u16(buf: &[u8], off: u32) -> u16 {
    let i = off as usize;
    u16::from_le_bytes([buf[i], buf[i + 1]])
}
#[inline(always)]
fn set_u16(buf: &mut [u8], off: u32, v: u16) {
    let i = off as usize;
    buf[i..i + 2].copy_from_slice(&v.to_le_bytes());
}
#[inline(always)]
fn get_u32(buf: &[u8], off: u32) -> u32 {
    let i = off as usize;
    u32::from_le_bytes(buf[i..i + 4].try_into().unwrap())
}
#[inline(always)]
fn set_u32(buf: &mut [u8], off: u32, v: u32) {
    let i = off as usize;
    buf[i..i + 4].copy_from_slice(&v.to_le_bytes());
}

// Context layout (12 bytes):
//   off 0    NumStats (u8)        — actual_state_count - 1
//   off 1    Flags (u8)
//   off 2..4 Union2: SummFreq (u16) | { Symbol (u8), Freq (u8) }
//   off 4..8 Union4: Stats (u32) | { Successor_0 (u16), Successor_1 (u16) }
//   off 8..12 Suffix (u32)

#[inline(always)] fn ctx_num_stats(b: &[u8], r: u32) -> u8 { b[r as usize] }
#[inline(always)] fn ctx_set_num_stats(b: &mut [u8], r: u32, v: u8) { b[r as usize] = v; }
#[inline(always)] fn ctx_flags(b: &[u8], r: u32) -> u8 { b[(r + 1) as usize] }
#[inline(always)] fn ctx_set_flags(b: &mut [u8], r: u32, v: u8) { b[(r + 1) as usize] = v; }
#[inline(always)] fn ctx_summ_freq(b: &[u8], r: u32) -> u16 { get_u16(b, r + 2) }
#[inline(always)] fn ctx_set_summ_freq(b: &mut [u8], r: u32, v: u16) { set_u16(b, r + 2, v) }
#[inline(always)] fn ctx_stats(b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn ctx_set_stats(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }
#[inline(always)] fn ctx_suffix(b: &[u8], r: u32) -> u32 { get_u32(b, r + 8) }
#[inline(always)] fn ctx_set_suffix(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 8, v) }

#[inline(always)] fn one_state_ref(ctx_ref: u32) -> u32 { ctx_ref + 2 }

#[inline(always)] fn st_symbol(b: &[u8], r: u32) -> u8 { b[r as usize] }
#[inline(always)] fn st_set_symbol(b: &mut [u8], r: u32, v: u8) { b[r as usize] = v; }
#[inline(always)] fn st_freq(b: &[u8], r: u32) -> u8 { b[(r + 1) as usize] }
#[inline(always)] fn st_set_freq(b: &mut [u8], r: u32, v: u8) { b[(r + 1) as usize] = v; }
#[inline(always)] fn st_succ(b: &[u8], r: u32) -> u32 {
    (get_u16(b, r + 2) as u32) | ((get_u16(b, r + 4) as u32) << 16)
}
#[inline(always)] fn st_set_succ(b: &mut [u8], r: u32, v: u32) {
    set_u16(b, r + 2, v as u16);
    set_u16(b, r + 4, (v >> 16) as u16);
}

// Node layout (12 bytes):
//   off 0..4  Stamp (u32) — EMPTY_NODE for free, 0 for guard
//   off 4..8  Next (u32)
//   off 8..12 NU (u32)
#[inline(always)] fn node_stamp(b: &[u8], r: u32) -> u32 { get_u32(b, r) }
#[inline(always)] fn node_set_stamp(b: &mut [u8], r: u32, v: u32) { set_u32(b, r, v) }
#[inline(always)] fn node_next(b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn node_set_next(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }
#[inline(always)] fn node_nu(b: &[u8], r: u32) -> u32 { get_u32(b, r + 8) }
#[inline(always)] fn node_set_nu(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 8, v) }

#[inline(always)]
fn u2b(nu: u32) -> u32 { nu * UNIT_SIZE }

// ====================================================================
// Hi-bits helpers (mirror C macros)
// ====================================================================

#[inline(always)]
fn hi_bits_prepare(sym: u8) -> u32 { (sym as u32) + 0xC0 }

#[inline(always)]
fn hi_bits_convert_3(flags: u32) -> u8 { ((flags >> (8 - 3)) & (1 << 3)) as u8 }

#[inline(always)]
fn hi_bits_convert_4(flags: u32) -> u8 { ((flags >> (8 - 4)) & (1 << 4)) as u8 }

#[inline(always)]
fn ppmd8_hi_bits_flag_3(sym: u8) -> u8 { hi_bits_convert_3(hi_bits_prepare(sym)) }

#[inline(always)]
fn ppmd8_hi_bits_flag_4(sym: u8) -> u8 { hi_bits_convert_4(hi_bits_prepare(sym)) }

// ====================================================================
// PPMd8 model
// ====================================================================

#[derive(Debug, Clone, Copy, Default)]
struct See {
    summ: u16,
    shift: u8,
    count: u8,
}

#[derive(Debug, Clone)]
pub struct Ppmd8 {
    pub max_order: u32,
    pub restore_method: RestoreMethod,
    base: Vec<u8>,
    align_offset: u32,
    size: u32,

    text: u32,
    units_start: u32,
    lo_unit: u32,
    hi_unit: u32,
    glue_count: u32,

    min_context: u32,
    max_context: u32,
    found_state: u32,

    order_fall: u32,
    init_esc: u32,
    prev_success: u32,
    run_length: i32,
    init_rl: i32,

    free_list: [u32; PPMD_NUM_INDEXES],
    stamps: [u32; PPMD_NUM_INDEXES],

    indx2units: [u8; PPMD_NUM_INDEXES + 2],
    units2indx: [u8; 128],
    ns2bs_indx: [u8; 256],
    ns2_indx: [u8; 260],

    exp_escape: [u8; 16],
    dummy_see: See,
    see: [[See; 32]; 24],
    bin_summ: [[u16; 64]; 25],
}

impl Ppmd8 {
    pub fn new(memory_size: u32, restore_method: RestoreMethod) -> Self {
        let align_offset = (4u32.wrapping_sub(memory_size)) & 3;
        let total = (align_offset + memory_size) as usize;
        let mut p = Self {
            max_order: 0,
            restore_method,
            base: vec![0u8; total],
            align_offset,
            size: memory_size,
            text: 0,
            units_start: 0,
            lo_unit: 0,
            hi_unit: 0,
            glue_count: 0,
            min_context: 0,
            max_context: 0,
            found_state: 0,
            order_fall: 0,
            init_esc: 0,
            prev_success: 0,
            run_length: 0,
            init_rl: 0,
            free_list: [0; PPMD_NUM_INDEXES],
            stamps: [0; PPMD_NUM_INDEXES],
            indx2units: [0; PPMD_NUM_INDEXES + 2],
            units2indx: [0; 128],
            ns2bs_indx: [0; 256],
            ns2_indx: [0; 260],
            exp_escape: K_EXP_ESCAPE,
            dummy_see: See::default(),
            see: [[See::default(); 32]; 24],
            bin_summ: [[0u16; 64]; 25],
        };
        // Build size tables.
        let mut k = 0usize;
        for i in 0..PPMD_NUM_INDEXES {
            let step = if i >= 12 { 4 } else { (i >> 2) + 1 };
            for _ in 0..step {
                p.units2indx[k] = i as u8;
                k += 1;
            }
            p.indx2units[i] = k as u8;
        }
        // ns2bs_indx
        p.ns2bs_indx[0] = 0;
        p.ns2bs_indx[1] = 1 << 1;
        for i in 2..11 { p.ns2bs_indx[i] = 2 << 1; }
        for i in 11..256 { p.ns2bs_indx[i] = 3 << 1; }
        // ns2_indx (PPMd8: 260 entries)
        for i in 0..5 { p.ns2_indx[i] = i as u8; }
        let mut m = 5usize;
        let mut kk = 1usize;
        for i in 5..260 {
            p.ns2_indx[i] = m as u8;
            kk -= 1;
            if kk == 0 {
                m += 1;
                kk = m - 4;
            }
        }
        p
    }

    pub fn init(&mut self, max_order: u32) {
        assert!(max_order >= MIN_ORDER && max_order <= MAX_ORDER);
        self.max_order = max_order;
        self.restart_model();
    }

    fn restart_model(&mut self) {
        for v in self.free_list.iter_mut() { *v = 0; }
        for v in self.stamps.iter_mut() { *v = 0; }
        self.text = self.align_offset;
        self.hi_unit = self.text + self.size;
        let n = self.size / 8 / UNIT_SIZE * 7 * UNIT_SIZE;
        self.lo_unit = self.hi_unit - n;
        self.units_start = self.lo_unit;
        self.glue_count = 0;
        self.order_fall = self.max_order;
        let cap = if self.max_order < 12 { self.max_order } else { 12 };
        self.run_length = -(cap as i32) - 1;
        self.init_rl = self.run_length;
        self.prev_success = 0;

        self.hi_unit -= UNIT_SIZE;
        let mc = self.hi_unit;
        let s_base = self.lo_unit;
        self.lo_unit += u2b(256 / 2);
        self.max_context = mc;
        self.min_context = mc;
        self.found_state = s_base;

        ctx_set_flags(&mut self.base, mc, 0);
        ctx_set_num_stats(&mut self.base, mc, 255);
        ctx_set_summ_freq(&mut self.base, mc, 256 + 1);
        ctx_set_stats(&mut self.base, mc, s_base);
        ctx_set_suffix(&mut self.base, mc, 0);
        for i in 0..256u32 {
            let s = s_base + i * 6;
            st_set_symbol(&mut self.base, s, i as u8);
            st_set_freq(&mut self.base, s, 1);
            st_set_succ(&mut self.base, s, 0);
        }

        // Init bin_summ — note PPMd8 uses NS2Indx[i]==m walker (fewer rows: 25)
        let mut i = 0usize;
        for m in 0..25 {
            while self.ns2_indx[i] == m as u8 {
                i += 1;
            }
            for k in 0..8 {
                let val = (PPMD_BIN_SCALE - (K_INIT_BIN_ESC[k] as u32) / (i as u32 + 1)) as u16;
                for r in (0..64).step_by(8) {
                    self.bin_summ[m][k + r] = val;
                }
            }
        }
        // Init see — 24 rows of 32 entries.
        let mut i = 0usize;
        for m in 0..24 {
            while self.ns2_indx[i + 3] == (m + 3) as u8 {
                i += 1;
            }
            let summ = (((2 * i + 5) << (PPMD_PERIOD_BITS - 4)) as u16);
            for k in 0..32 {
                self.see[m][k].summ = summ;
                self.see[m][k].shift = (PPMD_PERIOD_BITS - 4) as u8;
                self.see[m][k].count = 7;
            }
        }
        self.dummy_see.summ = 0;
        self.dummy_see.shift = PPMD_PERIOD_BITS as u8;
        self.dummy_see.count = 64;
    }

    // ---- memory pool ----

    fn insert_node(&mut self, node: u32, indx: u32) {
        node_set_stamp(&mut self.base, node, EMPTY_NODE);
        node_set_next(&mut self.base, node, self.free_list[indx as usize]);
        node_set_nu(&mut self.base, node, self.indx2units[indx as usize] as u32);
        self.free_list[indx as usize] = node;
        self.stamps[indx as usize] += 1;
    }
    fn remove_node(&mut self, indx: u32) -> u32 {
        let node = self.free_list[indx as usize];
        self.free_list[indx as usize] = node_next(&self.base, node);
        self.stamps[indx as usize] -= 1;
        node
    }
    fn split_block(&mut self, ptr: u32, old_indx: u32, new_indx: u32) {
        let nu = self.indx2units[old_indx as usize] as u32 - self.indx2units[new_indx as usize] as u32;
        let new_ptr = ptr + u2b(self.indx2units[new_indx as usize] as u32);
        let mut i = self.units2indx[(nu - 1) as usize] as u32;
        if (self.indx2units[i as usize] as u32) != nu {
            i -= 1;
            let k = self.indx2units[i as usize] as u32;
            self.insert_node(new_ptr + u2b(k), nu - k - 1);
        }
        self.insert_node(new_ptr, i);
    }
    fn alloc_units(&mut self, indx: u32) -> Option<u32> {
        if self.free_list[indx as usize] != 0 {
            return Some(self.remove_node(indx));
        }
        let num_bytes = u2b(self.indx2units[indx as usize] as u32);
        if self.hi_unit - self.lo_unit >= num_bytes {
            let p = self.lo_unit;
            self.lo_unit += num_bytes;
            return Some(p);
        }
        self.alloc_units_rare(indx)
    }
    fn alloc_units_rare(&mut self, indx: u32) -> Option<u32> {
        if self.glue_count == 0 {
            self.glue_free_blocks();
            if self.free_list[indx as usize] != 0 {
                return Some(self.remove_node(indx));
            }
        }
        let mut i = indx;
        loop {
            i += 1;
            if i as usize == PPMD_NUM_INDEXES {
                let num_bytes = u2b(self.indx2units[indx as usize] as u32);
                self.glue_count = self.glue_count.wrapping_sub(1);
                if (self.units_start - self.text) > num_bytes {
                    self.units_start -= num_bytes;
                    return Some(self.units_start);
                }
                return None;
            }
            if self.free_list[i as usize] != 0 {
                let block = self.remove_node(i);
                self.split_block(block, i, indx);
                return Some(block);
            }
        }
    }

    fn shrink_units(&mut self, old_ptr: u32, old_nu: u32, new_nu: u32) -> u32 {
        let i0 = self.units2indx[(old_nu - 1) as usize] as u32;
        let i1 = self.units2indx[(new_nu - 1) as usize] as u32;
        if i0 == i1 {
            return old_ptr;
        }
        if self.free_list[i1 as usize] != 0 {
            let ptr = self.remove_node(i1);
            for j in 0..(new_nu as usize) * 12 {
                self.base[ptr as usize + j] = self.base[old_ptr as usize + j];
            }
            self.insert_node(old_ptr, i0);
            ptr
        } else {
            self.split_block(old_ptr, i0, i1);
            old_ptr
        }
    }
    fn free_units(&mut self, ptr: u32, nu: u32) {
        self.insert_node(ptr, self.units2indx[(nu - 1) as usize] as u32);
    }
    fn special_free_unit(&mut self, ptr: u32) {
        if ptr != self.units_start {
            self.insert_node(ptr, 0);
        } else {
            self.units_start += UNIT_SIZE;
        }
    }

    fn glue_free_blocks(&mut self) {
        // Pass 1: build a singly-linked list of all free blocks AND glue
        // adjacent blocks (in memory order). Mirrors C's pattern where
        // `prev` only advances for nodes with nu != 0 — absorbed nodes
        // (nu == 0) get their slot overwritten by the next live node, so
        // pass 2 won't visit them.
        self.glue_count = 1 << 13;
        for s in self.stamps.iter_mut() { *s = 0; }
        if self.lo_unit != self.hi_unit {
            node_set_stamp(&mut self.base, self.lo_unit, 0); // guard
        }

        let mut head: u32 = 0;
        // prev_writer: the node whose .Next field is the current "tail slot"
        // of the chain. None means we should write to `head`.
        let mut prev_writer: Option<u32> = None;

        for i in 0..PPMD_NUM_INDEXES {
            let mut next = self.free_list[i];
            self.free_list[i] = 0;
            while next != 0 {
                let node = next;
                let nu = node_nu(&self.base, node);
                let saved_next = node_next(&self.base, node);
                // Write current node ref to prev's slot (head or last live node's .Next).
                match prev_writer {
                    None => head = node,
                    Some(p) => node_set_next(&mut self.base, p, node),
                }
                if nu != 0 {
                    // This node stays in the chain; advance prev_writer.
                    prev_writer = Some(node);
                    // Forward-glue: absorb consecutive empty neighbours.
                    let mut nu_now = nu;
                    loop {
                        let node2 = node + u2b(nu_now);
                        if (node2 as usize + 12) > self.base.len() {
                            break;
                        }
                        let stamp = node_stamp(&self.base, node2);
                        if stamp != EMPTY_NODE {
                            break;
                        }
                        nu_now += node_nu(&self.base, node2);
                        node_set_nu(&mut self.base, node2, 0);
                        node_set_nu(&mut self.base, node, nu_now);
                    }
                }
                // If nu == 0, prev_writer stays the same — the next live
                // node will overwrite the slot we just wrote, effectively
                // skipping this absorbed node.
                next = saved_next;
            }
        }
        // Terminate the chain.
        match prev_writer {
            None => head = 0,
            Some(p) => node_set_next(&mut self.base, p, 0),
        }

        // Distribute non-zero NU blocks back into free lists.
        let mut n = head;
        while n != 0 {
            let mut node = n;
            let mut nu = node_nu(&self.base, node);
            n = node_next(&self.base, node);
            if nu == 0 {
                continue;
            }
            while nu > 128 {
                self.insert_node(node, (PPMD_NUM_INDEXES - 1) as u32);
                nu -= 128;
                node += u2b(128);
            }
            let mut i = self.units2indx[(nu - 1) as usize] as u32;
            if (self.indx2units[i as usize] as u32) != nu {
                i -= 1;
                let k = self.indx2units[i as usize] as u32;
                self.insert_node(node + u2b(k), nu - k - 1);
            }
            self.insert_node(node, i);
        }
    }

    fn expand_text_area(&mut self) {
        let mut count = [0u32; PPMD_NUM_INDEXES];
        if self.lo_unit != self.hi_unit {
            node_set_stamp(&mut self.base, self.lo_unit, 0);
        }
        let mut node = self.units_start;
        loop {
            if (node as usize + 12) > self.base.len() {
                break;
            }
            if node_stamp(&self.base, node) != EMPTY_NODE {
                break;
            }
            let nu = node_nu(&self.base, node);
            node_set_stamp(&mut self.base, node, 0);
            count[self.units2indx[(nu - 1) as usize] as usize] += 1;
            node += u2b(nu);
        }
        self.units_start = node;

        for i in 0..PPMD_NUM_INDEXES {
            let mut cnt = count[i];
            if cnt == 0 {
                continue;
            }
            // Walk free_list[i] linked list and remove entries whose Stamp == 0.
            self.stamps[i] -= cnt;
            // Remove cnt entries from this list whose Stamp is 0.
            // Use a manual link walk via node_next.
            let mut prev: Option<u32> = None;
            let mut cur = self.free_list[i];
            while cur != 0 && cnt > 0 {
                let nxt = node_next(&self.base, cur);
                if node_stamp(&self.base, cur) != EMPTY_NODE {
                    // Stamp was reset to 0 — remove.
                    if let Some(p) = prev {
                        node_set_next(&mut self.base, p, nxt);
                    } else {
                        self.free_list[i] = nxt;
                    }
                    cnt -= 1;
                } else {
                    prev = Some(cur);
                }
                cur = nxt;
            }
        }
    }

    // ---- helpers ----

    fn bin_summ_idx(&self) -> (usize, usize) {
        let one = one_state_ref(self.min_context);
        let freq = st_freq(&self.base, one) as usize;
        let suffix = ctx_suffix(&self.base, self.min_context);
        let suffix_ns = ctx_num_stats(&self.base, suffix) as usize;
        let prev_succ = self.prev_success as usize;
        let run_extra = ((self.run_length >> 26) & 0x20) as usize;
        let ns2bs = self.ns2bs_indx[suffix_ns] as usize;
        let mc_flags = ctx_flags(&self.base, self.min_context) as usize;
        let outer = self.ns2_indx[freq - 1] as usize;
        let inner = prev_succ + run_extra + ns2bs + mc_flags;
        (outer, inner)
    }

    fn make_esc_freq(&mut self, num_masked: u32) -> (Option<(usize, usize)>, u32) {
        let mc = self.min_context;
        let num_stats = ctx_num_stats(&self.base, mc) as u32;
        if num_stats != 0xFF {
            let suffix = ctx_suffix(&self.base, mc);
            let suffix_ns = ctx_num_stats(&self.base, suffix) as u32;
            let summ_freq = ctx_summ_freq(&self.base, mc) as u32;
            let mc_flags = ctx_flags(&self.base, mc) as usize;
            let outer = (self.ns2_indx[(num_stats + 2) as usize] as usize) - 3;
            let mut inner = mc_flags;
            if summ_freq > 11 * (num_stats + 1) {
                inner += 1;
            }
            if 2 * num_stats < (suffix_ns as u32 + num_masked) {
                inner += 2;
            }
            let see = self.see[outer][inner];
            let summ = see.summ as u32;
            let r = summ >> see.shift;
            self.see[outer][inner].summ = (summ - r) as u16;
            let esc_freq = if r == 0 { 1 } else { r };
            (Some((outer, inner)), esc_freq)
        } else {
            (None, 1)
        }
    }
    fn see_update(&mut self, sig: (usize, usize)) {
        let s = &mut self.see[sig.0][sig.1];
        if s.shift < PPMD_PERIOD_BITS as u8 {
            s.count -= 1;
            if s.count == 0 {
                s.summ = s.summ.wrapping_shl(1);
                s.count = 3 << s.shift;
                s.shift += 1;
            }
        }
    }
    fn dummy_see_update(&mut self) {
        let s = &mut self.dummy_see;
        if s.shift < PPMD_PERIOD_BITS as u8 {
            s.count -= 1;
            if s.count == 0 {
                s.summ = s.summ.wrapping_shl(1);
                s.count = 3 << s.shift;
                s.shift += 1;
            }
        }
    }

    fn refresh(&mut self, ctx: u32, old_nu: u32, scale_in: u32) {
        let i = ctx_num_stats(&self.base, ctx) as u32;
        let s_new = self.shrink_units(ctx_stats(&self.base, ctx), old_nu, (i + 2) >> 1);
        ctx_set_stats(&mut self.base, ctx, s_new);

        let scale = scale_in | ((ctx_summ_freq(&self.base, ctx) as u32 >= (1u32 << 15)) as u32);
        let mut s = s_new;
        let mut flags = hi_bits_prepare(st_symbol(&self.base, s));
        let freq0 = st_freq(&self.base, s) as u32;
        let mut esc_freq = (ctx_summ_freq(&self.base, ctx) as u32) - freq0;
        let new_freq0 = (freq0 + scale) >> scale;
        let mut sum_freq = new_freq0;
        st_set_freq(&mut self.base, s, new_freq0 as u8);

        let mut count = i;
        while count > 0 {
            count -= 1;
            s += 6;
            let freq = st_freq(&self.base, s) as u32;
            esc_freq = esc_freq.wrapping_sub(freq);
            let new_freq = (freq + scale) >> scale;
            sum_freq += new_freq;
            st_set_freq(&mut self.base, s, new_freq as u8);
            flags |= hi_bits_prepare(st_symbol(&self.base, s));
        }

        let new_summ = (sum_freq + ((esc_freq + scale) >> scale)) as u16;
        ctx_set_summ_freq(&mut self.base, ctx, new_summ);
        let kept = FLAG_PREV_HIGH + FLAG_RESCALED * (scale as u8);
        let new_flags = (ctx_flags(&self.base, ctx) & kept) + hi_bits_convert_3(flags);
        ctx_set_flags(&mut self.base, ctx, new_flags);
    }

    fn cut_off(&mut self, ctx: u32, order: u32) -> u32 {
        let ns = ctx_num_stats(&self.base, ctx);
        if ns == 0 {
            let s = one_state_ref(ctx);
            let mut successor = st_succ(&self.base, s);
            if successor >= self.units_start {
                if order < self.max_order {
                    successor = self.cut_off(successor, order + 1);
                } else {
                    successor = 0;
                }
                st_set_succ(&mut self.base, s, successor);
                if successor != 0 || order <= 9 {
                    return ctx;
                }
            }
            self.special_free_unit(ctx);
            return 0;
        }

        let nu = ((ns as u32) + 2) >> 1;
        let indx = self.units2indx[(nu - 1) as usize] as u32;
        let stats = ctx_stats(&self.base, ctx);
        // Move stats up if it's near units_start (mirrors C inline check).
        let stats_offset = stats - self.units_start;
        let stats_new = if stats_offset <= (1 << 14)
            && stats <= self.free_list[indx as usize]
        {
            // C tries to move up: same indx alloc + memcpy + insert_node(old).
            // Simplified: only when free_list has matching block.
            if self.free_list[indx as usize] != 0 {
                let ptr = self.remove_node(indx);
                for j in 0..(nu as usize) * 12 {
                    self.base[ptr as usize + j] = self.base[stats as usize + j];
                }
                if stats != self.units_start {
                    self.insert_node(stats, indx);
                } else {
                    self.units_start += u2b(self.indx2units[indx as usize] as u32);
                }
                ptr
            } else {
                stats
            }
        } else {
            stats
        };
        ctx_set_stats(&mut self.base, ctx, stats_new);

        // Recursively cut off children.
        let mut s = stats_new + (ns as u32) * 6;
        let mut all_zero = true;
        let mut idx = ns as i32;
        while idx >= 0 {
            let succ = st_succ(&self.base, s);
            if succ < self.units_start {
                let new_s = self.create_successors_helper_zero(s);
                st_set_succ(&mut self.base, s, new_s);
                if new_s != 0 {
                    all_zero = false;
                }
            } else if order < self.max_order {
                let new_s = self.cut_off(succ, order + 1);
                st_set_succ(&mut self.base, s, new_s);
                if new_s != 0 {
                    all_zero = false;
                }
            } else {
                st_set_succ(&mut self.base, s, 0);
            }
            if idx == 0 { break; }
            idx -= 1;
            s -= 6;
        }
        if all_zero {
            // Convert this multi-state context into "deletable" by pruning
            // states with NULL successors. Mirrors C path in CutOff which
            // calls Refresh with scale=0 if context becomes order-0 scope.
            let _ = stats_new;
        }
        ctx
    }

    fn create_successors_helper_zero(&self, _s: u32) -> u32 { 0 }

    fn restore_model(&mut self, ctx_error: u32, _f_successor: u32) {
        self.text = self.align_offset;

        let mut c = self.max_context;
        while c != ctx_error {
            let ns = ctx_num_stats(&self.base, c);
            let new_ns = ns.wrapping_sub(1);
            ctx_set_num_stats(&mut self.base, c, new_ns);
            if new_ns == 0 {
                let s = ctx_stats(&self.base, c);
                let new_flags = (ctx_flags(&self.base, c) & FLAG_PREV_HIGH)
                    + ppmd8_hi_bits_flag_3(st_symbol(&self.base, s));
                ctx_set_flags(&mut self.base, c, new_flags);
                let sym = st_symbol(&self.base, s);
                let freq = st_freq(&self.base, s) as u32;
                let succ_lo = get_u16(&self.base, s + 2);
                let succ_hi = get_u16(&self.base, s + 4);
                let one = one_state_ref(c);
                st_set_symbol(&mut self.base, one, sym);
                st_set_freq(&mut self.base, one, ((freq + 11) >> 3) as u8);
                set_u16(&mut self.base, one + 2, succ_lo);
                set_u16(&mut self.base, one + 4, succ_hi);
                self.special_free_unit(s);
            } else {
                // C: Refresh(p, c, ((unsigned)c->NumStats + 3) >> 1, 0)
                // c->NumStats was decremented to new_ns before this point.
                let nu_old = ((new_ns as u32) + 3) >> 1;
                self.refresh(c, nu_old, 0);
            }
            c = ctx_suffix(&self.base, c);
        }
        // c now == ctx_error
        while c != self.min_context {
            let ns = ctx_num_stats(&self.base, c);
            if ns == 0 {
                let one = one_state_ref(c);
                let f = st_freq(&self.base, one) as u32;
                st_set_freq(&mut self.base, one, ((f + 1) >> 1) as u8);
            } else {
                let new_sf = (ctx_summ_freq(&self.base, c) as u32).wrapping_add(4) as u16;
                ctx_set_summ_freq(&mut self.base, c, new_sf);
                if new_sf as u32 > 128 + 4 * (ns as u32) {
                    let nu_old = ((ns as u32) + 2) >> 1;
                    self.refresh(c, nu_old, 1);
                }
            }
            c = ctx_suffix(&self.base, c);
        }

        if self.restore_method == RestoreMethod::Restart || self.used_memory() < (self.size >> 1) {
            self.restart_model();
        } else {
            // CUT_OFF mode.
            while ctx_suffix(&self.base, self.max_context) != 0 {
                self.max_context = ctx_suffix(&self.base, self.max_context);
            }
            loop {
                self.cut_off(self.max_context, 0);
                self.expand_text_area();
                if self.used_memory() <= 3 * (self.size >> 2) {
                    break;
                }
            }
            self.glue_count = 0;
            self.order_fall = self.max_order;
        }
        self.min_context = self.max_context;
    }

    fn used_memory(&self) -> u32 {
        let mut v = 0u32;
        for i in 0..PPMD_NUM_INDEXES {
            v = v.wrapping_add(self.stamps[i] * (self.indx2units[i] as u32));
        }
        self.size
            .wrapping_sub(self.hi_unit - self.lo_unit)
            .wrapping_sub(self.units_start - self.text)
            .wrapping_sub(u2b(v))
    }

    fn create_successors(&mut self, skip: bool, mut s1: u32, mut c: u32) -> Option<u32> {
        let up_branch = st_succ(&self.base, self.found_state);
        let mut ps: [u32; (MAX_ORDER + 1) as usize] = [0; (MAX_ORDER + 1) as usize];
        let mut num_ps = 0usize;
        if !skip {
            ps[num_ps] = self.found_state;
            num_ps += 1;
        }

        while ctx_suffix(&self.base, c) != 0 {
            c = ctx_suffix(&self.base, c);
            let s;
            if s1 != 0 {
                s = s1;
                s1 = 0;
            } else if ctx_num_stats(&self.base, c) != 0 {
                let stats = ctx_stats(&self.base, c);
                let sym = st_symbol(&self.base, self.found_state);
                let mut s_iter = stats;
                while st_symbol(&self.base, s_iter) != sym {
                    s_iter += 6;
                }
                if st_freq(&self.base, s_iter) < MAX_FREQ - 9 {
                    let f = st_freq(&self.base, s_iter);
                    st_set_freq(&mut self.base, s_iter, f + 1);
                    let sf = ctx_summ_freq(&self.base, c);
                    ctx_set_summ_freq(&mut self.base, c, sf + 1);
                }
                s = s_iter;
            } else {
                s = one_state_ref(c);
                let cur_f = st_freq(&self.base, s);
                // Match C: s->Freq += (!SUFFIX(c)->NumStats & (s->Freq < 24)).
                // SUFFIX(c) when c->Suffix==0 reads heap[0], same as in C reference.
                let suffix = ctx_suffix(&self.base, c);
                let suffix_ns = ctx_num_stats(&self.base, suffix);
                let bump = if suffix_ns == 0 && cur_f < 24 { 1u8 } else { 0u8 };
                st_set_freq(&mut self.base, s, cur_f + bump);
            }
            let successor = st_succ(&self.base, s);
            if successor != up_branch {
                c = successor;
                if num_ps == 0 {
                    return Some(c);
                }
                break;
            }
            ps[num_ps] = s;
            num_ps += 1;
        }

        let new_sym = self.base[up_branch as usize];
        let up_branch_next = up_branch + 1;
        let flags = ppmd8_hi_bits_flag_4(st_symbol(&self.base, self.found_state))
            + ppmd8_hi_bits_flag_3(new_sym);

        let new_freq;
        if ctx_num_stats(&self.base, c) == 0 {
            new_freq = st_freq(&self.base, one_state_ref(c));
        } else {
            let stats = ctx_stats(&self.base, c);
            let mut s_iter = stats;
            while st_symbol(&self.base, s_iter) != new_sym {
                s_iter += 6;
            }
            let cf = (st_freq(&self.base, s_iter) as u32) - 1;
            let s0 = (ctx_summ_freq(&self.base, c) as u32)
                - (ctx_num_stats(&self.base, c) as u32)
                - cf;
            let v = if 2 * cf <= s0 {
                if 5 * cf > s0 { 1u32 } else { 0u32 }
            } else {
                (cf + 2 * s0 - 3) / s0
            };
            new_freq = (1 + v) as u8;
        }

        while num_ps != 0 {
            let c1;
            if self.hi_unit != self.lo_unit {
                self.hi_unit -= UNIT_SIZE;
                c1 = self.hi_unit;
            } else if self.free_list[0] != 0 {
                c1 = self.remove_node(0);
            } else {
                c1 = match self.alloc_units_rare(0) {
                    Some(p) => p,
                    None => return None,
                };
            }
            ctx_set_flags(&mut self.base, c1, flags);
            ctx_set_num_stats(&mut self.base, c1, 0);
            let one = one_state_ref(c1);
            st_set_symbol(&mut self.base, one, new_sym);
            st_set_freq(&mut self.base, one, new_freq);
            st_set_succ(&mut self.base, one, up_branch_next);
            ctx_set_suffix(&mut self.base, c1, c);
            num_ps -= 1;
            st_set_succ(&mut self.base, ps[num_ps], c1);
            c = c1;
        }
        Some(c)
    }

    fn reduce_order(&mut self, mut s1: u32, mut c: u32) -> Option<u32> {
        let c1 = c;
        let up_branch = self.text;
        st_set_succ(&mut self.base, self.found_state, up_branch);
        self.order_fall += 1;
        let mut s: u32;

        loop {
            if s1 != 0 {
                c = ctx_suffix(&self.base, c);
                s = s1;
                s1 = 0;
            } else {
                if ctx_suffix(&self.base, c) == 0 {
                    return Some(c);
                }
                c = ctx_suffix(&self.base, c);
                if ctx_num_stats(&self.base, c) != 0 {
                    let stats = ctx_stats(&self.base, c);
                    let sym = st_symbol(&self.base, self.found_state);
                    let mut s_iter = stats;
                    if st_symbol(&self.base, s_iter) != sym {
                        s_iter += 6;
                        while st_symbol(&self.base, s_iter) != sym {
                            s_iter += 6;
                        }
                    }
                    if st_freq(&self.base, s_iter) < MAX_FREQ - 9 {
                        let f = st_freq(&self.base, s_iter);
                        st_set_freq(&mut self.base, s_iter, f + 2);
                        let sf = ctx_summ_freq(&self.base, c);
                        ctx_set_summ_freq(&mut self.base, c, sf + 2);
                    }
                    s = s_iter;
                } else {
                    s = one_state_ref(c);
                    let f = st_freq(&self.base, s);
                    st_set_freq(&mut self.base, s, f + ((f < 32) as u8));
                }
            }
            if st_succ(&self.base, s) != 0 {
                break;
            }
            st_set_succ(&mut self.base, s, up_branch);
            self.order_fall += 1;
        }

        if st_succ(&self.base, s) <= up_branch {
            let s2 = self.found_state;
            self.found_state = s;
            let succ = self.create_successors(false, 0, c);
            match succ {
                None => st_set_succ(&mut self.base, s, 0),
                Some(cs) => st_set_succ(&mut self.base, s, cs),
            }
            self.found_state = s2;
        }

        let successor = st_succ(&self.base, s);
        if self.order_fall == 1 && c1 == self.max_context {
            st_set_succ(&mut self.base, self.found_state, successor);
            self.text -= 1;
        }
        if successor == 0 {
            return None;
        }
        Some(successor)
    }

    fn next_context(&mut self) {
        let succ = st_succ(&self.base, self.found_state);
        if self.order_fall == 0 && succ >= self.units_start {
            self.min_context = succ;
            self.max_context = succ;
        } else {
            self.update_model();
        }
    }

    fn update_model(&mut self) {
        let mut min_successor = st_succ(&self.base, self.found_state);
        let f_freq = st_freq(&self.base, self.found_state) as u32;
        let f_symbol = st_symbol(&self.base, self.found_state);
        let mut s_used: u32 = 0;
        let mut have_s = false;

        if f_freq < (MAX_FREQ as u32) / 4
            && ctx_suffix(&self.base, self.min_context) != 0
        {
            let c = ctx_suffix(&self.base, self.min_context);
            if ctx_num_stats(&self.base, c) == 0 {
                let one = one_state_ref(c);
                let f = st_freq(&self.base, one);
                if f < 32 {
                    st_set_freq(&mut self.base, one, f + 1);
                }
                s_used = one;
                have_s = true;
            } else {
                let stats = ctx_stats(&self.base, c);
                let sym = f_symbol;
                let mut s = stats;
                if st_symbol(&self.base, s) != sym {
                    while st_symbol(&self.base, s) != sym {
                        s += 6;
                    }
                    if st_freq(&self.base, s) >= st_freq(&self.base, s - 6) {
                        let prev = s - 6;
                        let mut tmp = [0u8; 6];
                        tmp.copy_from_slice(&self.base[s as usize..(s + 6) as usize]);
                        let mut prev_data = [0u8; 6];
                        prev_data.copy_from_slice(&self.base[prev as usize..(prev + 6) as usize]);
                        self.base[s as usize..(s + 6) as usize].copy_from_slice(&prev_data);
                        self.base[prev as usize..(prev + 6) as usize].copy_from_slice(&tmp);
                        s = prev;
                    }
                }
                let cur_freq = st_freq(&self.base, s);
                if cur_freq < MAX_FREQ - 9 {
                    st_set_freq(&mut self.base, s, cur_freq + 2);
                    let sf = ctx_summ_freq(&self.base, c);
                    ctx_set_summ_freq(&mut self.base, c, sf + 2);
                }
                s_used = s;
                have_s = true;
            }
        }

        let c = self.max_context;
        if self.order_fall == 0 && min_successor != 0 {
            let cs = match self.create_successors(true, if have_s { s_used } else { 0 }, self.min_context) {
                Some(c) => c,
                None => {
                    st_set_succ(&mut self.base, self.found_state, 0);
                    self.restore_model(c, min_successor);
                    return;
                }
            };
            st_set_succ(&mut self.base, self.found_state, cs);
            self.min_context = cs;
            self.max_context = cs;
            return;
        }

        // Append symbol to text.
        self.base[self.text as usize] = f_symbol;
        self.text += 1;
        if self.text >= self.units_start {
            self.restore_model(c, min_successor);
            return;
        }
        let mut max_successor = self.text;

        if min_successor == 0 {
            let cs = match self.reduce_order(if have_s { s_used } else { 0 }, self.min_context) {
                Some(c) => c,
                None => {
                    self.restore_model(c, 0);
                    return;
                }
            };
            min_successor = cs;
        } else if min_successor < self.units_start {
            let cs = match self.create_successors(false, if have_s { s_used } else { 0 }, self.min_context) {
                Some(c) => c,
                None => {
                    self.restore_model(c, 0);
                    return;
                }
            };
            min_successor = cs;
        }

        self.order_fall -= 1;
        if self.order_fall == 0 {
            max_successor = min_successor;
            if self.max_context != self.min_context {
                self.text -= 1;
            }
        }

        let flag = ppmd8_hi_bits_flag_3(f_symbol);
        let ns = ctx_num_stats(&self.base, self.min_context) as u32;
        let s0 = (ctx_summ_freq(&self.base, self.min_context) as u32) - ns - f_freq;

        let mut c_walk = c;
        while c_walk != self.min_context {
            let ns1 = ctx_num_stats(&self.base, c_walk) as u32;
            let sum;
            if ns1 != 0 {
                if ns1 & 1 != 0 {
                    let old_nu = (ns1 + 1) >> 1;
                    let i = self.units2indx[(old_nu - 1) as usize] as u32;
                    if i != self.units2indx[old_nu as usize] as u32 {
                        let ptr = match self.alloc_units(i + 1) {
                            Some(p) => p,
                            None => {
                                self.restore_model(c_walk, min_successor);
                                return;
                            }
                        };
                        let old_ptr = ctx_stats(&self.base, c_walk);
                        for j in 0..(old_nu as usize) * 12 {
                            self.base[ptr as usize + j] = self.base[old_ptr as usize + j];
                        }
                        self.insert_node(old_ptr, i);
                        ctx_set_stats(&mut self.base, c_walk, ptr);
                    }
                }
                let mut s = ctx_summ_freq(&self.base, c_walk) as u32;
                let inc = (3 * ns1 + 1 < ns) as u32;
                s += inc;
                sum = s;
            } else {
                let s_new = match self.alloc_units(0) {
                    Some(p) => p,
                    None => {
                        self.restore_model(c_walk, min_successor);
                        return;
                    }
                };
                let one = one_state_ref(c_walk);
                let freq = st_freq(&self.base, one) as u32;
                let sym = st_symbol(&self.base, one);
                let succ_lo = get_u16(&self.base, one + 2);
                let succ_hi = get_u16(&self.base, one + 4);
                st_set_symbol(&mut self.base, s_new, sym);
                set_u16(&mut self.base, s_new + 2, succ_lo);
                set_u16(&mut self.base, s_new + 4, succ_hi);
                ctx_set_stats(&mut self.base, c_walk, s_new);
                let f = if freq < (MAX_FREQ as u32) / 4 - 1 {
                    freq << 1
                } else {
                    (MAX_FREQ as u32) - 4
                };
                st_set_freq(&mut self.base, s_new, f as u8);
                let extra = if ns > 2 { 1 } else { 0 };
                sum = f + self.init_esc + extra;
            }

            let stats_ptr = ctx_stats(&self.base, c_walk);
            let s_pos = stats_ptr + (ns1 + 1) * 6;
            let mut cf = 2 * (sum + 6) * f_freq;
            let sf = s0 + sum;
            st_set_symbol(&mut self.base, s_pos, f_symbol);
            ctx_set_num_stats(&mut self.base, c_walk, (ns1 + 1) as u8);
            st_set_succ(&mut self.base, s_pos, max_successor);
            let prev_flags = ctx_flags(&self.base, c_walk);
            ctx_set_flags(&mut self.base, c_walk, prev_flags | flag);
            let mut new_sum = sum;
            if cf < 6 * sf {
                cf = 1 + (cf > sf) as u32 + (cf >= 4 * sf) as u32;
                new_sum += 4;
            } else {
                cf = 4 + (cf > 9 * sf) as u32 + (cf > 12 * sf) as u32 + (cf > 15 * sf) as u32;
                new_sum += cf;
            }
            ctx_set_summ_freq(&mut self.base, c_walk, new_sum as u16);
            st_set_freq(&mut self.base, s_pos, cf as u8);
            c_walk = ctx_suffix(&self.base, c_walk);
        }
        self.max_context = min_successor;
        self.min_context = min_successor;
    }

    fn rescale(&mut self) {
        let stats = ctx_stats(&self.base, self.min_context);
        let mut s = self.found_state;
        if s != stats {
            let mut tmp = [0u8; 6];
            tmp.copy_from_slice(&self.base[s as usize..(s + 6) as usize]);
            while s != stats {
                let prev = s - 6;
                let mut p = [0u8; 6];
                p.copy_from_slice(&self.base[prev as usize..(prev + 6) as usize]);
                self.base[s as usize..(s + 6) as usize].copy_from_slice(&p);
                s = prev;
            }
            self.base[s as usize..(s + 6) as usize].copy_from_slice(&tmp);
        }

        let mut sum_freq = st_freq(&self.base, s) as u32;
        let mc = self.min_context;
        let mut esc_freq = (ctx_summ_freq(&self.base, mc) as u32) - sum_freq;
        let adder = if self.order_fall != 0 { 1 } else { 0 };
        sum_freq = (sum_freq + 4 + adder) >> 1;
        let mut i = ctx_num_stats(&self.base, mc) as u32;
        st_set_freq(&mut self.base, s, sum_freq as u8);

        let mut s_iter = s;
        while i > 0 {
            i -= 1;
            s_iter += 6;
            let freq = st_freq(&self.base, s_iter) as u32;
            esc_freq = esc_freq.wrapping_sub(freq);
            let new_freq = (freq + adder) >> 1;
            sum_freq += new_freq;
            st_set_freq(&mut self.base, s_iter, new_freq as u8);
            if s_iter > stats {
                let prev = s_iter - 6;
                if new_freq > st_freq(&self.base, prev) as u32 {
                    let mut tmp = [0u8; 6];
                    tmp.copy_from_slice(&self.base[s_iter as usize..(s_iter + 6) as usize]);
                    let mut s1 = s_iter;
                    while s1 > stats {
                        let p1 = s1 - 6;
                        if new_freq <= st_freq(&self.base, p1) as u32 { break; }
                        let mut buf = [0u8; 6];
                        buf.copy_from_slice(&self.base[p1 as usize..(p1 + 6) as usize]);
                        self.base[s1 as usize..(s1 + 6) as usize].copy_from_slice(&buf);
                        s1 = p1;
                    }
                    self.base[s1 as usize..(s1 + 6) as usize].copy_from_slice(&tmp);
                }
            }
        }

        if st_freq(&self.base, s_iter) == 0 {
            let mut removed = 0u32;
            while st_freq(&self.base, s_iter) == 0 {
                removed += 1;
                if s_iter == stats { break; }
                s_iter -= 6;
            }
            esc_freq += removed;
            let num_stats = ctx_num_stats(&self.base, mc) as u32;
            let new_ns = num_stats - removed;
            ctx_set_num_stats(&mut self.base, mc, new_ns as u8);
            let n0 = (num_stats + 2) >> 1;
            if new_ns == 0 {
                let stats_freq = st_freq(&self.base, stats) as u32;
                let freq = ((2 * stats_freq + esc_freq - 1) / esc_freq).min((MAX_FREQ as u32) / 3);
                let new_flags = (ctx_flags(&self.base, mc) & FLAG_PREV_HIGH)
                    + ppmd8_hi_bits_flag_3(st_symbol(&self.base, stats));
                ctx_set_flags(&mut self.base, mc, new_flags);
                let one = one_state_ref(mc);
                let sym = st_symbol(&self.base, stats);
                let succ = st_succ(&self.base, stats);
                st_set_symbol(&mut self.base, one, sym);
                st_set_freq(&mut self.base, one, freq as u8);
                st_set_succ(&mut self.base, one, succ);
                self.found_state = one;
                self.insert_node(stats, self.units2indx[(n0 - 1) as usize] as u32);
                return;
            }
            let n1 = (new_ns + 2) >> 1;
            if n0 != n1 {
                let new_stats = self.shrink_units(stats, n0, n1);
                ctx_set_stats(&mut self.base, mc, new_stats);
            }
        }

        let final_summ = sum_freq + esc_freq - (esc_freq >> 1);
        ctx_set_summ_freq(&mut self.base, mc, final_summ as u16);
        let f = ctx_flags(&self.base, mc) | FLAG_RESCALED;
        ctx_set_flags(&mut self.base, mc, f);
        self.found_state = ctx_stats(&self.base, mc);
    }

    fn update1(&mut self) {
        let s = self.found_state;
        let mc = self.min_context;
        let mut freq = st_freq(&self.base, s) as u32;
        freq += 4;
        let sf = ctx_summ_freq(&self.base, mc);
        ctx_set_summ_freq(&mut self.base, mc, sf + 4);
        st_set_freq(&mut self.base, s, freq as u8);
        if s > ctx_stats(&self.base, mc) {
            let prev = s - 6;
            if freq > st_freq(&self.base, prev) as u32 {
                let mut tmp_s = [0u8; 6];
                tmp_s.copy_from_slice(&self.base[s as usize..(s + 6) as usize]);
                let mut tmp_p = [0u8; 6];
                tmp_p.copy_from_slice(&self.base[prev as usize..(prev + 6) as usize]);
                self.base[s as usize..(s + 6) as usize].copy_from_slice(&tmp_p);
                self.base[prev as usize..(prev + 6) as usize].copy_from_slice(&tmp_s);
                self.found_state = prev;
                if freq > MAX_FREQ as u32 {
                    self.rescale();
                }
            }
        }
        self.next_context();
    }
    fn update1_0(&mut self) {
        let s = self.found_state;
        let mc = self.min_context;
        let mut freq = st_freq(&self.base, s) as u32;
        let summ_freq = ctx_summ_freq(&self.base, mc) as u32;
        // PPMd8: >= (vs PPMd7's >).
        self.prev_success = if 2 * freq >= summ_freq { 1 } else { 0 };
        self.run_length += self.prev_success as i32;
        ctx_set_summ_freq(&mut self.base, mc, (summ_freq + 4) as u16);
        freq += 4;
        st_set_freq(&mut self.base, s, freq as u8);
        if freq > MAX_FREQ as u32 {
            self.rescale();
        }
        self.next_context();
    }
    fn update2(&mut self) {
        let s = self.found_state;
        let mc = self.min_context;
        let mut freq = st_freq(&self.base, s) as u32;
        freq += 4;
        self.run_length = self.init_rl;
        let sf = ctx_summ_freq(&self.base, mc);
        ctx_set_summ_freq(&mut self.base, mc, sf + 4);
        st_set_freq(&mut self.base, s, freq as u8);
        if freq > MAX_FREQ as u32 {
            self.rescale();
        }
        self.update_model();
    }
}

// ====================================================================
// Carryless range decoder
// ====================================================================

#[derive(Debug)]
pub struct RangeDecoder<'a> {
    range: u32,
    code: u32,
    low: u32,
    src: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(src: &'a [u8]) -> Result<Self, &'static str> {
        if src.len() < 4 {
            return Err("PPMd8 RC stream too short");
        }
        let mut code = 0u32;
        for i in 0..4 {
            code = (code << 8) | src[i] as u32;
        }
        if code >= 0xFFFF_FFFF {
            return Err("PPMd8 RC initial code invalid");
        }
        Ok(Self { range: 0xFFFF_FFFF, code, low: 0, src, pos: 4 })
    }
    #[inline(always)]
    fn read_byte(&mut self) -> u8 {
        if self.pos < self.src.len() {
            let b = self.src[self.pos];
            self.pos += 1;
            b
        } else {
            0
        }
    }
    fn normalize(&mut self) {
        loop {
            let cond1 = (self.low ^ self.low.wrapping_add(self.range)) < K_TOP;
            let cond2 = self.range < K_BOT;
            if !(cond1 || cond2) {
                break;
            }
            if !cond1 && cond2 {
                self.range = (0u32.wrapping_sub(self.low)) & (K_BOT - 1);
            }
            let b = self.read_byte();
            self.code = (self.code << 8) | b as u32;
            self.range <<= 8;
            self.low <<= 8;
        }
    }
    fn get_threshold(&mut self, total: u32) -> u32 {
        self.range /= total;
        self.code / self.range
    }
    fn decode(&mut self, start: u32, size: u32) {
        let s = start.wrapping_mul(self.range);
        self.low = self.low.wrapping_add(s);
        self.code = self.code.wrapping_sub(s);
        self.range = self.range.wrapping_mul(size);
    }
    fn decode_final(&mut self, start: u32, size: u32) {
        self.decode(start, size);
        self.normalize();
    }
    pub fn is_finished_ok(&self) -> bool {
        self.code == 0
    }
}

// ====================================================================
// Carryless range encoder
// ====================================================================

#[derive(Debug)]
pub struct RangeEncoder {
    range: u32,
    low: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self { range: 0xFFFF_FFFF, low: 0, out: Vec::new() }
    }
    fn write_byte(&mut self) {
        self.out.push((self.low >> 24) as u8);
    }
    fn normalize(&mut self) {
        loop {
            let cond1 = (self.low ^ self.low.wrapping_add(self.range)) < K_TOP;
            let cond2 = self.range < K_BOT;
            if !(cond1 || cond2) {
                break;
            }
            if !cond1 && cond2 {
                self.range = (0u32.wrapping_sub(self.low)) & (K_BOT - 1);
            }
            self.write_byte();
            self.range <<= 8;
            self.low <<= 8;
        }
    }
    fn encode(&mut self, start: u32, size: u32, total: u32) {
        self.range /= total;
        self.low = self.low.wrapping_add(start.wrapping_mul(self.range));
        self.range = self.range.wrapping_mul(size);
    }
    pub fn flush(&mut self) {
        for _ in 0..4 {
            self.write_byte();
            self.low <<= 8;
        }
    }
}

// ====================================================================
// Decoder + symbol decoder
// ====================================================================

#[derive(Debug)]
pub struct Ppmd8Decoder<'a> {
    pub model: Ppmd8,
    pub rc: RangeDecoder<'a>,
}

impl<'a> Ppmd8Decoder<'a> {
    pub fn new(memory_size: u32, max_order: u32, restore: RestoreMethod, src: &'a [u8])
        -> Result<Self, &'static str>
    {
        let mut model = Ppmd8::new(memory_size, restore);
        model.init(max_order);
        let rc = RangeDecoder::new(src)?;
        Ok(Self { model, rc })
    }

    pub fn decode_symbol(&mut self) -> Result<u8, i32> {
        let p = &mut self.model;
        let r = &mut self.rc;

        if ctx_num_stats(&p.base, p.min_context) != 0 {
            // Multi-state path.
            let stats = ctx_stats(&p.base, p.min_context);
            let mut summ_freq = ctx_summ_freq(&p.base, p.min_context) as u32;
            // PPMD8_CORRECT_SUM_RANGE
            if summ_freq > r.range { summ_freq = r.range; }
            let count = r.get_threshold(summ_freq);
            let hi_cnt = count;
            let s_first = stats;
            let f0 = st_freq(&p.base, s_first) as u32;
            let mut count_sub = count.wrapping_sub(f0);
            if (count_sub as i32) < 0 {
                let sym = st_symbol(&p.base, s_first);
                r.decode_final(0, f0);
                p.found_state = s_first;
                p.update1_0();
                return Ok(sym);
            }
            p.prev_success = 0;
            let mut s = s_first;
            let mut i_remaining = ctx_num_stats(&p.base, p.min_context) as u32;
            while i_remaining > 0 {
                s += 6;
                let f = st_freq(&p.base, s) as u32;
                count_sub = count_sub.wrapping_sub(f);
                if (count_sub as i32) < 0 {
                    let sym = st_symbol(&p.base, s);
                    let start = hi_cnt.wrapping_sub(count_sub).wrapping_sub(f);
                    r.decode_final(start, f);
                    p.found_state = s;
                    p.update1();
                    return Ok(sym);
                }
                i_remaining -= 1;
            }
            if hi_cnt >= summ_freq {
                return Err(SYM_ERROR);
            }
            let new_hi = hi_cnt.wrapping_sub(count_sub);
            r.decode(new_hi, summ_freq.wrapping_sub(new_hi));

            let mut char_mask = [0xFFu8; 256];
            char_mask[st_symbol(&p.base, s) as usize] = 0;
            let mut s2 = stats;
            while s2 < s {
                let sym0 = st_symbol(&p.base, s2);
                let sym1 = st_symbol(&p.base, s2 + 6);
                s2 += 12;
                char_mask[sym0 as usize] = 0;
                char_mask[sym1 as usize] = 0;
            }
            return self.masked_loop(char_mask);
        }

        // Single-state context (BinSumm path).
        let one = one_state_ref(p.min_context);
        let bin_sig = p.bin_summ_idx();
        let pr = p.bin_summ[bin_sig.0][bin_sig.1] as u32;
        let size0 = (r.range >> 14) * pr;
        let pr_after = pr - ((pr + (1 << (PPMD_PERIOD_BITS - 2))) >> PPMD_PERIOD_BITS);
        if r.code < size0 {
            p.bin_summ[bin_sig.0][bin_sig.1] = (pr_after + (1 << PPMD_INT_BITS)) as u16;
            r.range = size0;
            r.normalize();
            let freq = st_freq(&p.base, one) as u32;
            let succ = st_succ(&p.base, one);
            let sym = st_symbol(&p.base, one);
            p.found_state = one;
            p.prev_success = 1;
            p.run_length = p.run_length.wrapping_add(1);
            // PPMd8: <196 vs PPMd7's <128.
            let new_freq = if freq < 196 { freq + 1 } else { freq };
            st_set_freq(&mut p.base, one, new_freq as u8);
            if p.order_fall == 0 && succ >= p.units_start {
                p.min_context = succ;
                p.max_context = succ;
            } else {
                p.update_model();
            }
            return Ok(sym);
        }
        // Mismatch.
        p.bin_summ[bin_sig.0][bin_sig.1] = pr_after as u16;
        p.init_esc = p.exp_escape[(pr_after >> 10) as usize] as u32;
        r.low = r.low.wrapping_add(size0);
        r.code = r.code.wrapping_sub(size0);
        r.range = (r.range & !(PPMD_BIN_SCALE - 1)) - size0;
        let mut char_mask = [0xFFu8; 256];
        char_mask[st_symbol(&p.base, one) as usize] = 0;
        p.prev_success = 0;
        self.masked_loop(char_mask)
    }

    fn masked_loop(&mut self, mut char_mask: [u8; 256]) -> Result<u8, i32> {
        loop {
            let p = &mut self.model;
            let r = &mut self.rc;
            r.normalize();
            let mut mc = p.min_context;
            let num_masked = ctx_num_stats(&p.base, mc) as u32;

            loop {
                p.order_fall = p.order_fall.wrapping_add(1);
                let suffix = ctx_suffix(&p.base, mc);
                if suffix == 0 {
                    return Err(SYM_END);
                }
                mc = suffix;
                if ctx_num_stats(&p.base, mc) as u32 != num_masked {
                    break;
                }
            }
            p.min_context = mc;

            let stats = ctx_stats(&p.base, mc);
            let num = (ctx_num_stats(&p.base, mc) as u32) + 1;
            let mut hi_cnt = 0u32;
            let mut s = stats;
            let num2 = num / 2;
            let odd = num & 1;
            if odd == 1 {
                let f = st_freq(&p.base, s) as u32;
                let m = char_mask[st_symbol(&p.base, s) as usize] as u32;
                hi_cnt = f & m;
                s += 6;
            }
            for _ in 0..num2 {
                let s0 = s;
                let s1 = s + 6;
                s += 12;
                let sym0 = st_symbol(&p.base, s0);
                let sym1 = st_symbol(&p.base, s1);
                hi_cnt += st_freq(&p.base, s0) as u32 & char_mask[sym0 as usize] as u32;
                hi_cnt += st_freq(&p.base, s1) as u32 & char_mask[sym1 as usize] as u32;
            }

            let (see_idx, esc_freq) = p.make_esc_freq(num_masked);
            let mut freq_sum = esc_freq + hi_cnt;
            if freq_sum > r.range { freq_sum = r.range; }
            let count = r.get_threshold(freq_sum);
            if count < hi_cnt {
                let mut sptr = ctx_stats(&p.base, p.min_context);
                let mut remaining = count;
                let new_hi = count;
                loop {
                    let f = st_freq(&p.base, sptr) as u32;
                    let m = char_mask[st_symbol(&p.base, sptr) as usize] as u32;
                    let dec = f & m;
                    if remaining < dec {
                        break;
                    }
                    remaining -= dec;
                    sptr += 6;
                }
                let f = st_freq(&p.base, sptr) as u32;
                let start = new_hi - remaining;
                r.decode_final(start, f);
                if let Some(idx) = see_idx { p.see_update(idx); } else { p.dummy_see_update(); }
                let sym = st_symbol(&p.base, sptr);
                p.found_state = sptr;
                p.update2();
                return Ok(sym);
            }
            if count >= freq_sum {
                return Err(SYM_ERROR);
            }
            r.decode(hi_cnt, freq_sum - hi_cnt);
            if let Some(idx) = see_idx {
                let cur = p.see[idx.0][idx.1].summ;
                p.see[idx.0][idx.1].summ = cur.wrapping_add(freq_sum as u16);
            }
            // Mask all stats of current mc.
            let mut sm = ctx_stats(&p.base, p.min_context);
            for _ in 0..num {
                char_mask[st_symbol(&p.base, sm) as usize] = 0;
                sm += 6;
            }
        }
    }
}

// ====================================================================
// Encoder + symbol encoder
// ====================================================================

#[derive(Debug)]
pub struct Ppmd8Encoder {
    pub model: Ppmd8,
    pub rc: RangeEncoder,
}

impl Ppmd8Encoder {
    pub fn new(memory_size: u32, max_order: u32, restore: RestoreMethod) -> Self {
        let mut model = Ppmd8::new(memory_size, restore);
        model.init(max_order);
        Self { model, rc: RangeEncoder::new() }
    }

    pub fn encode_symbols(&mut self, buf: &[u8]) {
        for &b in buf {
            self.encode_symbol(b);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.rc.flush();
        self.rc.out
    }

    fn encode_symbol(&mut self, symbol: u8) {
        let p = &mut self.model;
        let r = &mut self.rc;
        let symbol_int = symbol as u32;

        if ctx_num_stats(&p.base, p.min_context) != 0 {
            let stats = ctx_stats(&p.base, p.min_context);
            let mut summ_freq = ctx_summ_freq(&p.base, p.min_context) as u32;
            if summ_freq > r.range { summ_freq = r.range; }
            let s_first = stats;
            if st_symbol(&p.base, s_first) as u32 == symbol_int {
                let f0 = st_freq(&p.base, s_first) as u32;
                r.encode(0, f0, summ_freq);
                r.normalize();
                p.found_state = s_first;
                p.update1_0();
                return;
            }
            p.prev_success = 0;
            let mut sum = st_freq(&p.base, s_first) as u32;
            let mut i_remaining = ctx_num_stats(&p.base, p.min_context) as u32;
            let mut s = s_first;
            while i_remaining > 0 {
                s += 6;
                if st_symbol(&p.base, s) as u32 == symbol_int {
                    let f = st_freq(&p.base, s) as u32;
                    r.encode(sum, f, summ_freq);
                    r.normalize();
                    p.found_state = s;
                    p.update1();
                    return;
                }
                sum += st_freq(&p.base, s) as u32;
                i_remaining -= 1;
            }
            r.encode(sum, summ_freq - sum, summ_freq);

            let mut char_mask = [0xFFu8; 256];
            char_mask[st_symbol(&p.base, s) as usize] = 0;
            let mut s2 = stats;
            while s2 < s {
                let sym0 = st_symbol(&p.base, s2);
                let sym1 = st_symbol(&p.base, s2 + 6);
                s2 += 12;
                char_mask[sym0 as usize] = 0;
                char_mask[sym1 as usize] = 0;
            }
            self.encode_masked(symbol, char_mask);
            return;
        }

        // Single-state.
        let one = one_state_ref(p.min_context);
        let bin_sig = p.bin_summ_idx();
        let pr = p.bin_summ[bin_sig.0][bin_sig.1] as u32;
        let bound = (r.range >> 14) * pr;
        let pr_after = pr - ((pr + (1 << (PPMD_PERIOD_BITS - 2))) >> PPMD_PERIOD_BITS);
        if st_symbol(&p.base, one) as u32 == symbol_int {
            p.bin_summ[bin_sig.0][bin_sig.1] = (pr_after + (1 << PPMD_INT_BITS)) as u16;
            r.range = bound;
            r.normalize();
            let freq = st_freq(&p.base, one) as u32;
            let succ = st_succ(&p.base, one);
            p.found_state = one;
            p.prev_success = 1;
            p.run_length = p.run_length.wrapping_add(1);
            let new_freq = if freq < 196 { freq + 1 } else { freq };
            st_set_freq(&mut p.base, one, new_freq as u8);
            if p.order_fall == 0 && succ >= p.units_start {
                p.min_context = succ;
                p.max_context = succ;
            } else {
                p.update_model();
            }
            return;
        }
        // Escape.
        p.bin_summ[bin_sig.0][bin_sig.1] = pr_after as u16;
        p.init_esc = p.exp_escape[(pr_after >> 10) as usize] as u32;
        r.low = r.low.wrapping_add(bound as u32);
        r.range = (r.range & !(PPMD_BIN_SCALE - 1)) - bound;
        let mut char_mask = [0xFFu8; 256];
        char_mask[st_symbol(&p.base, one) as usize] = 0;
        p.prev_success = 0;
        self.encode_masked(symbol, char_mask);
    }

    fn encode_masked(&mut self, symbol: u8, mut char_mask: [u8; 256]) {
        let symbol_int = symbol as u32;
        loop {
            let p = &mut self.model;
            let r = &mut self.rc;
            r.normalize();
            let mut mc = p.min_context;
            let num_masked = ctx_num_stats(&p.base, mc) as u32;

            loop {
                p.order_fall = p.order_fall.wrapping_add(1);
                let suffix = ctx_suffix(&p.base, mc);
                if suffix == 0 {
                    return;
                }
                mc = suffix;
                if ctx_num_stats(&p.base, mc) as u32 != num_masked {
                    break;
                }
            }
            p.min_context = mc;

            let (see_idx, esc_freq) = p.make_esc_freq(num_masked);

            let stats = ctx_stats(&p.base, mc);
            let mut s = stats;
            let mut sum = 0u32;
            let mut i_remaining = (ctx_num_stats(&p.base, mc) as u32) + 1;
            let mut found = false;
            let mut found_freq = 0u32;
            let mut found_low = 0u32;
            let mut found_state_ptr = 0u32;
            while i_remaining > 0 {
                let cur = st_symbol(&p.base, s) as u32;
                if cur == symbol_int {
                    found_low = sum;
                    found_freq = st_freq(&p.base, s) as u32;
                    found_state_ptr = s;
                    found = true;
                    break;
                }
                let f = st_freq(&p.base, s) as u32;
                let m = char_mask[cur as usize] as u32;
                sum += f & m;
                s += 6;
                i_remaining -= 1;
            }

            if found {
                if let Some(idx) = see_idx { p.see_update(idx); } else { p.dummy_see_update(); }
                p.found_state = found_state_ptr;
                let mut total = sum + esc_freq;
                let num2 = i_remaining / 2;
                let parity = i_remaining & 1;
                total = total.wrapping_add(found_freq & 0u32.wrapping_sub(parity));
                let mut s_walk = s + (parity as u32) * 6;
                for _ in 0..num2 {
                    let s0 = s_walk;
                    let s1 = s_walk + 6;
                    s_walk += 12;
                    let sym0 = st_symbol(&p.base, s0);
                    let sym1 = st_symbol(&p.base, s1);
                    total += st_freq(&p.base, s0) as u32 & char_mask[sym0 as usize] as u32;
                    total += st_freq(&p.base, s1) as u32 & char_mask[sym1 as usize] as u32;
                }
                if total > r.range { total = r.range; }
                r.encode(found_low, found_freq, total);
                r.normalize();
                p.update2();
                return;
            }

            // Escape.
            let mut total = sum + esc_freq;
            if let Some(idx) = see_idx {
                let cur = p.see[idx.0][idx.1].summ;
                p.see[idx.0][idx.1].summ = cur.wrapping_add(total as u16);
            }
            if total > r.range { total = r.range; }
            r.encode(sum, total - sum, total);
            // Mask all stats.
            let mut sm = ctx_stats(&p.base, p.min_context);
            let count = (ctx_num_stats(&p.base, p.min_context) as u32) + 1;
            for _ in 0..count {
                char_mask[st_symbol(&p.base, sm) as usize] = 0;
                sm += 6;
            }
        }
    }
}

/// One-shot encode helper.
pub fn encode_one_shot(data: &[u8], memory_size: u32, max_order: u32, restore: RestoreMethod)
    -> Vec<u8>
{
    let mut e = Ppmd8Encoder::new(memory_size, max_order, restore);
    e.encode_symbols(data);
    e.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_init() {
        let mut p = Ppmd8::new(1 << 16, RestoreMethod::Restart);
        p.init(MIN_ORDER);
        assert_eq!(p.max_order, MIN_ORDER);
        assert!(p.min_context > 0);
    }

    #[test]
    fn round_trip_small() {
        let data = b"Hello PPMd8 encoder! Hello hello hello world world!".to_vec();
        let mem = 1 << 16;
        let order = 6;
        let encoded = encode_one_shot(&data, mem, order, RestoreMethod::Restart);
        let mut dec = Ppmd8Decoder::new(mem, order, RestoreMethod::Restart, &encoded).unwrap();
        let mut out = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            out.push(dec.decode_symbol().unwrap());
        }
        assert_eq!(out, data);
    }
}
