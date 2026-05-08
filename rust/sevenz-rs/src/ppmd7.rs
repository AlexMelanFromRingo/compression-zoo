//! PPMd7 (PPMdH with the 7-Zip range coder) — port of `7zip/C/Ppmd7*.c`.
//!
//! Manual heap arithmetic from C (`Base + offset`) is replaced with `Vec<u8>`
//! indexed by `u32` offsets — fully memory-safe.  Cross-checked against the
//! reference encoder (`Ppmd7z_EncodeSymbols`) on random / text / zero
//! inputs up to 1 MiB.
//!
//! Public API:
//! * [`Ppmd7Decoder`] — streaming decoder for the 7z PPMd variant.

use core::convert::TryInto;

// =====================================================================
// Constants
// =====================================================================

pub const MIN_ORDER: u32 = 2;
pub const MAX_ORDER: u32 = 64;
pub const MIN_MEM_SIZE: u32 = 1 << 11;

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
const TOP_VALUE: u32 = 1 << 24;
const EMPTY_NODE: u16 = 0;

// PPMd model tables.
const K_EXP_ESCAPE: [u8; 16] = [25, 14, 9, 7, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2];
const K_INIT_BIN_ESC: [u16; 8] = [
    0x3CDD, 0x1F3F, 0x59BF, 0x48F3, 0x64A1, 0x5ABC, 0x6632, 0x6051,
];

pub const SYM_END: i32 = -1;
pub const SYM_ERROR: i32 = -2;

// =====================================================================
// Field accessors (Vec<u8> as the heap, u32 offsets as references)
// =====================================================================

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
//   off 0..2   NumStats (u16)
//   off 2..4   Union2: SummFreq (u16) | { Symbol (u8), Freq (u8) }
//   off 4..8   Union4: Stats (u32) | { Successor_0 (u16), Successor_1 (u16) }
//   off 8..12  Suffix (u32)

#[inline(always)] fn ctx_num_stats(b: &[u8], r: u32) -> u16 { get_u16(b, r) }
#[inline(always)] fn ctx_set_num_stats(b: &mut [u8], r: u32, v: u16) { set_u16(b, r, v) }
#[inline(always)] fn ctx_summ_freq(b: &[u8], r: u32) -> u16 { get_u16(b, r + 2) }
#[inline(always)] fn ctx_set_summ_freq(b: &mut [u8], r: u32, v: u16) { set_u16(b, r + 2, v) }
#[inline(always)] fn ctx_stats(b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn ctx_set_stats(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }
#[inline(always)] fn ctx_suffix(b: &[u8], r: u32) -> u32 { get_u32(b, r + 8) }
#[inline(always)] fn ctx_set_suffix(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 8, v) }

// One-state inline view (when NumStats == 1): base = ctx_ref + 2.
#[inline(always)] fn one_state_ref(ctx_ref: u32) -> u32 { ctx_ref + 2 }
#[inline(always)] fn st_symbol(b: &[u8], r: u32) -> u8 { b[r as usize] }
#[inline(always)] fn st_set_symbol(b: &mut [u8], r: u32, v: u8) { b[r as usize] = v; }
#[inline(always)] fn st_freq(b: &[u8], r: u32) -> u8 { b[(r + 1) as usize] }
#[inline(always)] fn st_set_freq(b: &mut [u8], r: u32, v: u8) { b[(r + 1) as usize] = v; }
#[inline(always)] fn st_succ(b: &[u8], r: u32) -> u32 {
    // Successor is two u16 fields, joined LE.
    (get_u16(b, r + 2) as u32) | ((get_u16(b, r + 4) as u32) << 16)
}
#[inline(always)] fn st_set_succ(b: &mut [u8], r: u32, v: u32) {
    set_u16(b, r + 2, v as u16);
    set_u16(b, r + 4, (v >> 16) as u16);
}

// Node layout (12 bytes):
//   off 0..2   Stamp (u16)
//   off 2..4   NU (u16)
//   off 4..8   Next (u32)
//   off 8..12  Prev (u32, unused in PPMd7)
#[inline(always)] fn node_stamp(b: &[u8], r: u32) -> u16 { get_u16(b, r) }
#[inline(always)] fn node_set_stamp(b: &mut [u8], r: u32, v: u16) { set_u16(b, r, v) }
#[inline(always)] fn node_nu(b: &[u8], r: u32) -> u16 { get_u16(b, r + 2) }
#[inline(always)] fn node_set_nu(b: &mut [u8], r: u32, v: u16) { set_u16(b, r + 2, v) }
#[inline(always)] fn node_next(b: &[u8], r: u32) -> u32 { get_u32(b, r + 4) }
#[inline(always)] fn node_set_next(b: &mut [u8], r: u32, v: u32) { set_u32(b, r + 4, v) }

// =====================================================================
// PPMd7 model state
// =====================================================================

#[derive(Debug, Clone)]
pub struct Ppmd7 {
    pub max_order: u32,
    base: Vec<u8>,
    align_offset: u32,
    size: u32,

    text: u32,        // current "raw" position
    units_start: u32, // boundary between heap units and raw text
    lo_unit: u32,
    hi_unit: u32,
    glue_count: u32,

    min_context: u32,
    max_context: u32,
    found_state: u32,

    order_fall: u32,
    init_esc: u32,
    prev_success: u32,
    hi_bits_flag: u32,
    run_length: i32,
    init_rl: i32,

    free_list: [u32; PPMD_NUM_INDEXES],

    indx2units: [u8; PPMD_NUM_INDEXES + 2],
    units2indx: [u8; 128],
    ns2bs_indx: [u8; 256],
    ns2_indx: [u8; 256],

    exp_escape: [u8; 16],
    dummy_see: See,
    see: [[See; 16]; 25],
    bin_summ: [[u16; 64]; 128],
}

#[derive(Debug, Clone, Copy, Default)]
struct See {
    summ: u16,
    shift: u8,
    count: u8,
}

#[inline(always)]
fn u2b(nu: u32) -> u32 { nu * UNIT_SIZE }

impl Ppmd7 {
    pub fn new(memory_size: u32) -> Self {
        let align_offset = (4u32.wrapping_sub(memory_size)) & 3;
        let total = (align_offset + memory_size) as usize;
        let mut p = Self {
            max_order: 0,
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
            hi_bits_flag: 0,
            run_length: 0,
            init_rl: 0,
            free_list: [0; PPMD_NUM_INDEXES],
            indx2units: [0; PPMD_NUM_INDEXES + 2],
            units2indx: [0; 128],
            ns2bs_indx: [0; 256],
            ns2_indx: [0; 256],
            exp_escape: K_EXP_ESCAPE,
            dummy_see: See::default(),
            see: [[See::default(); 16]; 25],
            bin_summ: [[0u16; 64]; 128],
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
        // ns2_indx
        for i in 0..3 { p.ns2_indx[i] = i as u8; }
        let mut m = 3usize;
        let mut kk = 1usize;
        for i in 3..256 {
            p.ns2_indx[i] = m as u8;
            kk -= 1;
            if kk == 0 {
                m += 1;
                kk = m - 2;
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

        // Allocate root context (NumStats=256) and 256 states.
        self.hi_unit -= UNIT_SIZE;
        let mc = self.hi_unit;
        let s_base = self.lo_unit;
        self.lo_unit += u2b(256 / 2);
        self.max_context = mc;
        self.min_context = mc;
        self.found_state = s_base;

        ctx_set_num_stats(&mut self.base, mc, 256);
        ctx_set_summ_freq(&mut self.base, mc, 256 + 1);
        ctx_set_stats(&mut self.base, mc, s_base);
        ctx_set_suffix(&mut self.base, mc, 0);
        for i in 0..256u32 {
            let s = s_base + i * 6;
            st_set_symbol(&mut self.base, s, i as u8);
            st_set_freq(&mut self.base, s, 1);
            st_set_succ(&mut self.base, s, 0);
        }

        // Initialize bin_summ.
        for i in 0..128 {
            for k in 0..8 {
                let val = PPMD_BIN_SCALE - (K_INIT_BIN_ESC[k] as u32) / (i as u32 + 2);
                for m in (0..64).step_by(8) {
                    self.bin_summ[i][k + m] = val as u16;
                }
            }
        }
        // Initialize see.
        for i in 0..25 {
            let summ = ((5 * i + 10) << (PPMD_PERIOD_BITS - 4)) as u16;
            for k in 0..16 {
                self.see[i][k].summ = summ;
                self.see[i][k].shift = (PPMD_PERIOD_BITS - 4) as u8;
                self.see[i][k].count = 4;
            }
        }
        self.dummy_see.summ = 0;
        self.dummy_see.shift = PPMD_PERIOD_BITS as u8;
        self.dummy_see.count = 64;
    }

    // ---- memory pool ----

    fn insert_node(&mut self, node: u32, indx: u32) {
        set_u32(&mut self.base, node, self.free_list[indx as usize]);
        self.free_list[indx as usize] = node;
    }
    fn remove_node(&mut self, indx: u32) -> u32 {
        let node = self.free_list[indx as usize];
        self.free_list[indx as usize] = get_u32(&self.base, node);
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

    fn glue_free_blocks(&mut self) {
        self.glue_count = 255;
        if self.lo_unit != self.hi_unit {
            // Set guard node at lo_unit.
            node_set_stamp(&mut self.base, self.lo_unit, 1);
        }
        let mut head: u32 = 0;
        let mut n: u32 = 0;
        for i in 0..PPMD_NUM_INDEXES {
            let nu = self.indx2units[i] as u16;
            let mut next = self.free_list[i];
            self.free_list[i] = 0;
            while next != 0 {
                let node = next;
                // Free-list next pointer lives at offset 0 of the block (set
                // by `insert_node`); we must read it BEFORE we overwrite
                // bytes 0..1 with the Stamp / 2..3 with NU.
                let prev_next = get_u32(&self.base, node);
                node_set_stamp(&mut self.base, node, EMPTY_NODE);
                node_set_nu(&mut self.base, node, nu);
                node_set_next(&mut self.base, node, n);
                n = node;
                next = prev_next;
            }
        }
        head = n;
        // Glue free blocks.
        let mut prev_link = (&mut head) as *mut u32; // we re-walk via local indexing below
        let _ = prev_link;
        // To stay safe, we re-implement using two passes: walk list, glue adjacent.
        // First, build linked list as Vec for easier manipulation.
        let mut list = Vec::new();
        let mut cur = head;
        while cur != 0 {
            list.push(cur);
            cur = node_next(&self.base, cur);
        }
        // Glue: merge consecutive (by address) free nodes.
        let mut merged = Vec::with_capacity(list.len());
        for &node in &list {
            let mut nu = node_nu(&self.base, node) as u32;
            if nu == 0 {
                continue;
            }
            loop {
                let node2 = node + u2b(nu);
                if (node2 as usize + 4) > self.base.len() {
                    break;
                }
                let stamp = node_stamp(&self.base, node2);
                let nu2 = node_nu(&self.base, node2) as u32;
                let total = nu + nu2;
                if stamp != EMPTY_NODE || total >= 0x10000 {
                    break;
                }
                nu = total;
                node_set_nu(&mut self.base, node, nu as u16);
                node_set_nu(&mut self.base, node2, 0);
            }
            merged.push(node);
        }
        // Fill free lists.
        for node in merged {
            let mut nu = node_nu(&self.base, node) as u32;
            if nu == 0 {
                continue;
            }
            let mut node = node;
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

    // ---- helpers ----

    #[inline(always)]
    fn one_state_freq(&self, ctx: u32) -> u8 {
        st_freq(&self.base, one_state_ref(ctx))
    }
    #[inline(always)]
    fn one_state_symbol(&self, ctx: u32) -> u8 {
        st_symbol(&self.base, one_state_ref(ctx))
    }

    // Build BinSumm pointer for the current MinContext (single-state).
    fn bin_summ_idx(&mut self) -> (usize, usize) {
        let one = one_state_ref(self.min_context);
        let freq = st_freq(&self.base, one) as usize;
        let suffix = ctx_suffix(&self.base, self.min_context);
        let suffix_ns = ctx_num_stats(&self.base, suffix) as usize;
        let prev_succ = self.prev_success as usize;
        let run_extra = ((self.run_length >> 26) & 0x20) as usize;
        let ns2bs = self.ns2bs_indx[suffix_ns - 1] as usize;
        let one_sym = self.one_state_symbol(self.min_context);
        let one_hi = (((one_sym as u32) + 0xC0) >> (8 - 4)) & (1 << 4);
        let found_sym = st_symbol(&self.base, self.found_state);
        let hi_bits = (((found_sym as u32) + 0xC0) >> (8 - 3)) & (1 << 3);
        self.hi_bits_flag = hi_bits;
        let i = freq - 1;
        let k = prev_succ + run_extra + ns2bs + one_hi as usize + hi_bits as usize;
        (i, k)
    }

    fn make_esc_freq(&mut self, num_masked: u32) -> (See, u32) {
        let mc = self.min_context;
        let num_stats = ctx_num_stats(&self.base, mc) as u32;
        if num_stats != 256 {
            let non_masked = num_stats - num_masked;
            let suffix = ctx_suffix(&self.base, mc);
            let suffix_ns = ctx_num_stats(&self.base, suffix) as u32;
            let summ_freq = ctx_summ_freq(&self.base, mc) as u32;
            let see_idx_outer = self.ns2_indx[(non_masked - 1) as usize] as usize;
            let mut see_idx_inner = 0usize;
            if non_masked < suffix_ns - num_stats {
                see_idx_inner += 1;
            }
            if summ_freq < 11 * num_stats {
                see_idx_inner += 2;
            }
            if num_masked > non_masked {
                see_idx_inner += 4;
            }
            see_idx_inner += self.hi_bits_flag as usize;
            let see = self.see[see_idx_outer][see_idx_inner];
            let summ = see.summ as u32;
            let r = summ >> see.shift;
            let new_summ = (summ - r) as u16;
            let esc_freq = if r == 0 { 1 } else { r };
            self.see[see_idx_outer][see_idx_inner].summ = new_summ;
            (see, esc_freq)
        } else {
            (self.dummy_see, 1)
        }
    }
    fn see_update(&mut self, see_signature: (usize, usize)) {
        let s = &mut self.see[see_signature.0][see_signature.1];
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

    fn create_successors(&mut self) -> Option<u32> {
        let mut c = self.min_context;
        let up_branch = st_succ(&self.base, self.found_state);
        let mut ps: [u32; MAX_ORDER as usize] = [0; MAX_ORDER as usize];
        let mut num_ps = 0usize;
        if self.order_fall != 0 {
            ps[num_ps] = self.found_state;
            num_ps += 1;
        }
        let new_sym;
        let new_freq;
        // Mirrors C's `while (c->Suffix) { ... }`: stop the walk when the
        // current context has no suffix (root).  The caller then proceeds
        // to create the new single-symbol contexts pointing at `c`.
        while ctx_suffix(&self.base, c) != 0 {
            c = ctx_suffix(&self.base, c);
            let s;
            let ns = ctx_num_stats(&self.base, c);
            if ns != 1 {
                let stats = ctx_stats(&self.base, c);
                let sym = st_symbol(&self.base, self.found_state);
                let mut s_iter = stats;
                while st_symbol(&self.base, s_iter) != sym {
                    s_iter += 6;
                }
                s = s_iter;
            } else {
                s = one_state_ref(c);
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
        new_sym = self.base[up_branch as usize];
        let up_branch_next = up_branch + 1;
        if ctx_num_stats(&self.base, c) == 1 {
            new_freq = self.one_state_freq(c);
        } else {
            let stats = ctx_stats(&self.base, c);
            let mut s_iter = stats;
            while st_symbol(&self.base, s_iter) != new_sym {
                s_iter += 6;
            }
            let cf = (st_freq(&self.base, s_iter) as u32) - 1;
            let s0 = (ctx_summ_freq(&self.base, c) as u32) - (ctx_num_stats(&self.base, c) as u32) - cf;
            let v = if 2 * cf <= s0 {
                if 5 * cf > s0 { 1u32 } else { 0u32 }
            } else {
                (2 * cf + s0 - 1) / (2 * s0) + 1
            };
            new_freq = (1 + v) as u8;
        }
        // Loop creating contexts.
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
            ctx_set_num_stats(&mut self.base, c1, 1);
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

    fn next_context(&mut self) {
        let succ = st_succ(&self.base, self.found_state);
        if self.order_fall == 0 && succ > self.text {
            self.min_context = succ;
            self.max_context = succ;
        } else {
            self.update_model();
        }
    }

    fn update_model(&mut self) {
        let found = self.found_state;
        let found_freq = st_freq(&self.base, found);
        let found_sym = st_symbol(&self.base, found);
        if found_freq < MAX_FREQ / 4 && ctx_suffix(&self.base, self.min_context) != 0 {
            let c = ctx_suffix(&self.base, self.min_context);
            if ctx_num_stats(&self.base, c) == 1 {
                let one = one_state_ref(c);
                let f = st_freq(&self.base, one);
                if f < 32 {
                    st_set_freq(&mut self.base, one, f + 1);
                }
            } else {
                let stats = ctx_stats(&self.base, c);
                let mut s = stats;
                if st_symbol(&self.base, s) != found_sym {
                    while st_symbol(&self.base, s) != found_sym {
                        s += 6;
                    }
                    if st_freq(&self.base, s) >= st_freq(&self.base, s - 6) {
                        // Swap states.
                        let prev = s - 6;
                        let mut tmp = [0u8; 6];
                        tmp.copy_from_slice(&self.base[s as usize..(s + 6) as usize]);
                        let cur_prev = {
                            let mut p = [0u8; 6];
                            p.copy_from_slice(&self.base[prev as usize..(prev + 6) as usize]);
                            p
                        };
                        self.base[s as usize..(s + 6) as usize].copy_from_slice(&cur_prev);
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
            }
        }
        if self.order_fall == 0 {
            let cs = match self.create_successors() {
                Some(c) => c,
                None => { self.restart_model(); return; }
            };
            self.max_context = cs;
            self.min_context = cs;
            st_set_succ(&mut self.base, self.found_state, cs);
            return;
        }
        let max_successor;
        // Append symbol to text.
        self.base[self.text as usize] = found_sym;
        self.text += 1;
        if self.text >= self.units_start {
            self.restart_model();
            return;
        }
        max_successor = self.text;
        let mut min_successor = st_succ(&self.base, self.found_state);
        let mut max_successor = max_successor;
        if min_successor != 0 {
            if min_successor <= max_successor {
                let cs = match self.create_successors() {
                    Some(c) => c,
                    None => { self.restart_model(); return; }
                };
                min_successor = cs;
            }
            self.order_fall -= 1;
            if self.order_fall == 0 {
                // We've climbed back to MAX order: the new context can be
                // used directly as the successor for this and the parent
                // contexts, so roll back the raw-text byte we appended.
                max_successor = min_successor;
                if self.max_context != self.min_context {
                    self.text -= 1;
                }
            }
        } else {
            st_set_succ(&mut self.base, self.found_state, max_successor);
            min_successor = self.min_context;
        }
        let mc = self.min_context;
        let c_save = self.max_context;
        self.max_context = min_successor;
        self.min_context = min_successor;
        if c_save == mc {
            return;
        }
        // Update the chain.
        let ns = ctx_num_stats(&self.base, mc);
        let s0 = (ctx_summ_freq(&self.base, mc) as u32)
            - (ns as u32)
            - (st_freq(&self.base, found) as u32 - 1);
        let mut c = c_save;
        while c != mc {
            let ns1 = ctx_num_stats(&self.base, c) as u32;
            let sum;
            if ns1 != 1 {
                if ns1 & 1 == 0 {
                    let old_nu = ns1 >> 1;
                    let i = self.units2indx[(old_nu - 1) as usize] as u32;
                    if i != self.units2indx[(old_nu) as usize] as u32 {
                        let ptr = match self.alloc_units(i + 1) {
                            Some(p) => p,
                            None => { self.restart_model(); return; }
                        };
                        let old_ptr = ctx_stats(&self.base, c);
                        // Copy old_nu * 12 bytes.
                        for j in 0..(old_nu as usize) * 12 {
                            self.base[ptr as usize + j] = self.base[old_ptr as usize + j];
                        }
                        self.insert_node(old_ptr, i);
                        ctx_set_stats(&mut self.base, c, ptr);
                    }
                }
                let mut s = ctx_summ_freq(&self.base, c) as u32;
                let inc = ((2 * ns1 < ns as u32) as u32)
                    + 2 * ((4 * ns1 <= ns as u32) as u32 & (s <= 8 * ns1) as u32);
                s += inc;
                sum = s;
            } else {
                let s_new = match self.alloc_units(0) {
                    Some(p) => p,
                    None => { self.restart_model(); return; }
                };
                let one = one_state_ref(c);
                let freq = st_freq(&self.base, one) as u32;
                let sym = st_symbol(&self.base, one);
                let succ_lo = get_u16(&self.base, one + 2);
                let succ_hi = get_u16(&self.base, one + 4);
                st_set_symbol(&mut self.base, s_new, sym);
                set_u16(&mut self.base, s_new + 2, succ_lo);
                set_u16(&mut self.base, s_new + 4, succ_hi);
                ctx_set_stats(&mut self.base, c, s_new);
                let f = if freq < (MAX_FREQ as u32) / 4 - 1 {
                    freq << 1
                } else {
                    (MAX_FREQ as u32) - 4
                };
                st_set_freq(&mut self.base, s_new, f as u8);
                let extra = if ns > 3 { 1 } else { 0 };
                sum = f + self.init_esc + extra;
            }
            let stats_ptr = ctx_stats(&self.base, c);
            let s = stats_ptr + (ns1 * 6);
            let mut cf = 2 * (sum + 6) * (st_freq(&self.base, found) as u32);
            let sf = s0 + sum;
            st_set_symbol(&mut self.base, s, found_sym);
            ctx_set_num_stats(&mut self.base, c, (ns1 + 1) as u16);
            st_set_succ(&mut self.base, s, max_successor);
            let mut new_sum = sum;
            if cf < 6 * sf {
                cf = 1 + (cf > sf) as u32 + (cf >= 4 * sf) as u32;
                new_sum += 3;
            } else {
                cf = 4 + (cf >= 9 * sf) as u32 + (cf >= 12 * sf) as u32 + (cf >= 15 * sf) as u32;
                new_sum += cf;
            }
            ctx_set_summ_freq(&mut self.base, c, new_sum as u16);
            st_set_freq(&mut self.base, s, cf as u8);
            c = ctx_suffix(&self.base, c);
        }
    }

    fn rescale(&mut self) {
        let stats = ctx_stats(&self.base, self.min_context);
        let mut s = self.found_state;
        // Move found state to the front (sort by Freq).
        if s != stats {
            // Copy s to tmp, shift right.
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
        let mut i = ctx_num_stats(&self.base, mc) as u32 - 1;
        st_set_freq(&mut self.base, s, sum_freq as u8);
        let mut s_iter = s;
        while i > 0 {
            i -= 1;
            s_iter += 6;
            let mut freq = st_freq(&self.base, s_iter) as u32;
            esc_freq = esc_freq.wrapping_sub(freq);
            freq = (freq + adder) >> 1;
            sum_freq += freq;
            st_set_freq(&mut self.base, s_iter, freq as u8);
            // Bubble up if needed.
            if s_iter > stats {
                let prev = s_iter - 6;
                if freq > st_freq(&self.base, prev) as u32 {
                    let mut tmp = [0u8; 6];
                    tmp.copy_from_slice(&self.base[s_iter as usize..(s_iter + 6) as usize]);
                    let mut s1 = s_iter;
                    while s1 > stats {
                        let p1 = s1 - 6;
                        if freq <= st_freq(&self.base, p1) as u32 { break; }
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
            // Remove zero-freq tail.
            let mut removed = 0u32;
            while st_freq(&self.base, s_iter) == 0 {
                removed += 1;
                if s_iter == stats { break; }
                s_iter -= 6;
            }
            esc_freq += removed;
            let num_stats = ctx_num_stats(&self.base, mc) as u32;
            let new_ns = num_stats - removed;
            ctx_set_num_stats(&mut self.base, mc, new_ns as u16);
            let n0 = (num_stats + 1) >> 1;
            if new_ns == 1 {
                let stats_freq = st_freq(&self.base, stats) as u32;
                let mut freq = stats_freq;
                while esc_freq > 1 {
                    esc_freq >>= 1;
                    freq = (freq + 1) >> 1;
                }
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
            let n1 = (new_ns + 1) >> 1;
            if n0 != n1 {
                let i0 = self.units2indx[(n0 - 1) as usize] as u32;
                let i1 = self.units2indx[(n1 - 1) as usize] as u32;
                if i0 != i1 {
                    if self.free_list[i1 as usize] != 0 {
                        let ptr = self.remove_node(i1);
                        ctx_set_stats(&mut self.base, mc, ptr);
                        for j in 0..(n1 as usize) * 12 {
                            self.base[ptr as usize + j] = self.base[stats as usize + j];
                        }
                        self.insert_node(stats, i0);
                    } else {
                        self.split_block(stats, i0, i1);
                    }
                }
            }
        }
        let final_summ = sum_freq + esc_freq - (esc_freq >> 1);
        ctx_set_summ_freq(&mut self.base, mc, final_summ as u16);
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
                // Swap.
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
        self.prev_success = if 2 * freq > summ_freq { 1 } else { 0 };
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

// =====================================================================
// Range decoder (7z variant)
// =====================================================================

#[derive(Debug)]
pub struct RangeDecoder<'a> {
    range: u32,
    code: u32,
    src: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(src: &'a [u8]) -> Result<Self, &'static str> {
        if src.len() < 5 {
            return Err("PPMd7 RC stream too short");
        }
        if src[0] != 0 {
            return Err("PPMd7 RC stream first byte must be 0");
        }
        let mut code = 0u32;
        for i in 1..5 {
            code = (code << 8) | src[i] as u32;
        }
        if code == 0xFFFF_FFFF {
            return Err("PPMd7 RC initial code invalid");
        }
        Ok(Self { range: 0xFFFF_FFFF, code, src, pos: 5 })
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
    #[inline(always)]
    fn normalize(&mut self) {
        while self.range < TOP_VALUE {
            let b = self.read_byte();
            self.code = (self.code << 8) | b as u32;
            self.range <<= 8;
        }
    }
    #[inline(always)]
    fn get_threshold(&mut self, total: u32) -> u32 {
        self.range /= total;
        self.code / self.range
    }
    #[inline(always)]
    fn decode(&mut self, start: u32, size: u32) {
        self.code = self.code.wrapping_sub(start.wrapping_mul(self.range));
        self.range = self.range.wrapping_mul(size);
    }
    #[inline(always)]
    fn decode_final(&mut self, start: u32, size: u32) {
        self.decode(start, size);
        self.normalize();
    }
    pub fn is_finished_ok(&self) -> bool {
        self.code == 0
    }
}

// =====================================================================
// Decoder
// =====================================================================

#[derive(Debug)]
pub struct Ppmd7Decoder<'a> {
    model: Ppmd7,
    rc: RangeDecoder<'a>,
}

impl<'a> Ppmd7Decoder<'a> {
    pub fn new(memory_size: u32, max_order: u32, src: &'a [u8]) -> Result<Self, &'static str> {
        let mut model = Ppmd7::new(memory_size);
        model.init(max_order);
        let rc = RangeDecoder::new(src)?;
        Ok(Self { model, rc })
    }

    /// Decode the next symbol. Returns:
    /// * `Ok(0..=255)` — decoded byte;
    /// * `Err(SYM_END)` — end of stream;
    /// * `Err(SYM_ERROR)` — data error.
    pub fn decode_symbol(&mut self) -> Result<u8, i32> {
        let p = &mut self.model;
        let r = &mut self.rc;

        let mc = p.min_context;
        let num_stats = ctx_num_stats(&p.base, mc);
        if num_stats != 1 {
            let stats = ctx_stats(&p.base, mc);
            let summ_freq = ctx_summ_freq(&p.base, mc) as u32;
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
            let total = num_stats as u32 - 1;
            for _ in 0..total {
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
            }
            if hi_cnt >= summ_freq {
                return Err(SYM_ERROR);
            }
            let new_hi = hi_cnt - count_sub;
            r.decode(new_hi, summ_freq - new_hi);
            // Set up char_mask and continue with masked-symbol path.
            // NOTE: C uses `p->FoundState->Symbol` here (the symbol from the
            // previous decode), not the last state of the current context.
            let prev_found_sym = st_symbol(&p.base, p.found_state);
            p.hi_bits_flag =
                (((prev_found_sym as u32) + 0xC0) >> (8 - 3)) & (1 << 3);
            let mut char_mask = [0xFFu8; 256];
            // Mask out symbols already considered.
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

        // Single-state context: BinSumm path.
        let one = one_state_ref(mc);
        let bin_sig = p.bin_summ_idx();
        let prob = p.bin_summ[bin_sig.0][bin_sig.1];
        let pr = prob as u32;
        let size0 = (r.range >> 14) * pr;
        let pr_new = pr + (1 << PPMD_INT_BITS) - ((pr + (1 << (PPMD_PERIOD_BITS - 2))) >> PPMD_PERIOD_BITS);
        if r.code < size0 {
            p.bin_summ[bin_sig.0][bin_sig.1] = pr_new as u16;
            r.range = size0;
            // Single-byte normalize OK because min(BinSumm) > (1<<6).
            r.normalize();
            let freq = st_freq(&p.base, one) as u32;
            let succ = st_succ(&p.base, one);
            let sym = st_symbol(&p.base, one);
            p.found_state = one;
            p.prev_success = 1;
            p.run_length = p.run_length.wrapping_add(1);
            let new_freq = if freq < 128 { freq + 1 } else { freq };
            st_set_freq(&mut p.base, one, new_freq as u8);
            if p.order_fall == 0 && succ > p.text {
                p.min_context = succ;
                p.max_context = succ;
            } else {
                p.update_model();
            }
            return Ok(sym);
        }
        // Mismatch — masked path.  C updates `pr` first (subtract mean),
        // writes that back as the new prob, then uses *that* updated value
        // to index ExpEscape.
        let pr2 = pr - ((pr + (1 << (PPMD_PERIOD_BITS - 2))) >> PPMD_PERIOD_BITS);
        p.bin_summ[bin_sig.0][bin_sig.1] = pr2 as u16;
        p.init_esc = p.exp_escape[(pr2 >> 10) as usize] as u32;
        r.code = r.code.wrapping_sub(size0);
        r.range = r.range.wrapping_sub(size0);
        r.normalize();
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
            let mut num_masked = ctx_num_stats(&p.base, mc) as u32;
            // Walk to a context with more states than num_masked.
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
            let stats = ctx_stats(&p.base, mc);
            let num = ctx_num_stats(&p.base, mc) as u32;
            let mut hi_cnt = 0u32;
            let mut s = stats;
            // Step pairwise:
            let num2 = num / 2;
            let odd = num & 1;
            if odd == 1 {
                let f = st_freq(&p.base, s) as u32;
                let m = char_mask[st_symbol(&p.base, s) as usize] as u32;
                hi_cnt = f & m;
                s += 6;
            }
            p.min_context = mc;
            for _ in 0..num2 {
                let s0 = s;
                let s1 = s + 6;
                s += 12;
                let sym0 = st_symbol(&p.base, s0);
                let sym1 = st_symbol(&p.base, s1);
                hi_cnt += st_freq(&p.base, s0) as u32 & char_mask[sym0 as usize] as u32;
                hi_cnt += st_freq(&p.base, s1) as u32 & char_mask[sym1 as usize] as u32;
            }
            // Compute escape freq via SEE.
            let (see, esc_freq) = p.make_esc_freq(num_masked);
            let _ = see;
            let freq_sum = esc_freq + hi_cnt;
            let count = r.get_threshold(freq_sum);
            if count < hi_cnt {
                // Decoded a symbol.
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
                // Update SEE.
                if num != 256 {
                    let non_masked = num - num_masked;
                    let suffix = ctx_suffix(&p.base, p.min_context);
                    let suffix_ns = ctx_num_stats(&p.base, suffix) as u32;
                    let summ_freq = ctx_summ_freq(&p.base, p.min_context) as u32;
                    let see_idx_outer = p.ns2_indx[(non_masked - 1) as usize] as usize;
                    let mut see_idx_inner = 0usize;
                    if non_masked < suffix_ns - num { see_idx_inner += 1; }
                    if summ_freq < 11 * num { see_idx_inner += 2; }
                    if num_masked > non_masked { see_idx_inner += 4; }
                    see_idx_inner += p.hi_bits_flag as usize;
                    p.see_update((see_idx_outer, see_idx_inner));
                } else {
                    p.dummy_see_update();
                }
                let sym = st_symbol(&p.base, sptr);
                p.found_state = sptr;
                p.update2();
                return Ok(sym);
            }
            if count >= freq_sum {
                return Err(SYM_ERROR);
            }
            r.decode(hi_cnt, freq_sum - hi_cnt);
            // Update SEE for escape.
            if num != 256 {
                let non_masked = num - num_masked;
                let suffix = ctx_suffix(&p.base, p.min_context);
                let suffix_ns = ctx_num_stats(&p.base, suffix) as u32;
                let summ_freq = ctx_summ_freq(&p.base, p.min_context) as u32;
                let see_idx_outer = p.ns2_indx[(non_masked - 1) as usize] as usize;
                let mut see_idx_inner = 0usize;
                if non_masked < suffix_ns - num { see_idx_inner += 1; }
                if summ_freq < 11 * num { see_idx_inner += 2; }
                if num_masked > non_masked { see_idx_inner += 4; }
                see_idx_inner += p.hi_bits_flag as usize;
                let s_ref = &mut p.see[see_idx_outer][see_idx_inner];
                s_ref.summ = s_ref.summ.wrapping_add(freq_sum as u16);
            }
            // Mask all symbols of the current context.
            let mut sm = ctx_stats(&p.base, p.min_context);
            for _ in 0..num {
                char_mask[st_symbol(&p.base, sm) as usize] = 0;
                sm += 6;
            }
            num_masked = num;
        }
    }

    pub fn rc(&self) -> &RangeDecoder<'a> {
        &self.rc
    }

    /// Diagnostic accessor for tests.
    pub fn debug_summ_freq(&self) -> u32 {
        let mc = self.model.min_context;
        if ctx_num_stats(&self.model.base, mc) == 1 {
            self.model.one_state_freq(mc) as u32
        } else {
            ctx_summ_freq(&self.model.base, mc) as u32
        }
    }
    /// Inspect state at index of min_context's stats array.
    pub fn debug_state_at(&self, idx: usize) -> (u8, u8) {
        let mc = self.model.min_context;
        let stats = ctx_stats(&self.model.base, mc);
        let s = stats + (idx as u32) * 6;
        (st_symbol(&self.model.base, s), st_freq(&self.model.base, s))
    }
    /// Returns (NumStats, SummFreq, OrderFall, RunLength, range, code).
    pub fn debug_state(&self) -> (u32, u32, u32, i32, u32, u32) {
        let mc = self.model.min_context;
        let ns = ctx_num_stats(&self.model.base, mc) as u32;
        let sf = if ns == 1 { 0 } else { ctx_summ_freq(&self.model.base, mc) as u32 };
        (ns, sf, self.model.order_fall, self.model.run_length, self.rc.range, self.rc.code)
    }
    /// (NumStats, SummFreq, FoundFreq, OrderFall, root_sf) — same fields as the C trace.
    pub fn debug_state_short(&self) -> (u32, u32, u32, u32, u32) {
        let mc = self.model.min_context;
        let ns = ctx_num_stats(&self.model.base, mc) as u32;
        let sf = if ns == 1 { 0 } else { ctx_summ_freq(&self.model.base, mc) as u32 };
        let ff = if self.model.found_state == 0 { 0 } else { st_freq(&self.model.base, self.model.found_state) as u32 };
        let root = self.model.align_offset + self.model.size - UNIT_SIZE;
        let root_sf = ctx_summ_freq(&self.model.base, root) as u32;
        (ns, sf, ff, self.model.order_fall, root_sf)
    }
    /// First N (sym, freq) pairs of the root context's stats array.
    pub fn debug_root_states(&self, n: usize) -> Vec<(u8, u8)> {
        let root = self.model.align_offset + self.model.size - UNIT_SIZE;
        let stats = ctx_stats(&self.model.base, root);
        (0..n).map(|i| {
            let s = stats + (i as u32) * 6;
            (st_symbol(&self.model.base, s), st_freq(&self.model.base, s))
        }).collect()
    }
    /// (sym, freq) of min_context.OneState (only meaningful if NumStats==1).
    pub fn debug_one_state(&self) -> (u8, u8) {
        let mc = self.model.min_context;
        let one = one_state_ref(mc);
        (st_symbol(&self.model.base, one), st_freq(&self.model.base, one))
    }
    /// Dump min_context's stats array (first N entries).
    pub fn debug_mc_states(&self, n: usize) -> Vec<(u8, u8, u32)> {
        let mc = self.model.min_context;
        let stats = ctx_stats(&self.model.base, mc);
        (0..n).map(|i| {
            let s = stats + (i as u32) * 6;
            (
                st_symbol(&self.model.base, s),
                st_freq(&self.model.base, s),
                st_succ(&self.model.base, s),
            )
        }).collect()
    }
    /// (mc_off, ns, sf, suffix, found_freq, found_off, of, root_sf).
    pub fn debug_state_full(&self) -> (u32, u32, u32, u32, u32, u32, u32, u32) {
        let mc = self.model.min_context;
        let ns = ctx_num_stats(&self.model.base, mc) as u32;
        let sf = if ns == 1 { 0 } else { ctx_summ_freq(&self.model.base, mc) as u32 };
        let suf = ctx_suffix(&self.model.base, mc);
        let fs = self.model.found_state;
        let ff = if fs == 0 { 0 } else { st_freq(&self.model.base, fs) as u32 };
        let root = self.model.align_offset + self.model.size - UNIT_SIZE;
        let root_sf = ctx_summ_freq(&self.model.base, root) as u32;
        (mc, ns, sf, suf, ff, fs, self.model.order_fall, root_sf)
    }
}

// =====================================================================
// Range encoder (7z variant) + symbol encoder
// =====================================================================

#[derive(Debug)]
pub struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u32,
    out: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 1, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        let low_lo = self.low as u32;
        let low_hi = (self.low >> 32) as u32;
        if low_lo < 0xFF00_0000 || low_hi != 0 {
            let mut temp = self.cache;
            loop {
                self.out.push(temp.wrapping_add(low_hi as u8));
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 { break; }
            }
            self.cache = (low_lo >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = (low_lo << 8) as u64;
    }

    fn encode(&mut self, start: u32, size: u32) {
        self.low = self.low.wrapping_add((start as u64) * (self.range as u64));
        self.range = self.range.wrapping_mul(size);
    }

    fn normalize(&mut self) {
        while self.range < TOP_VALUE {
            self.range <<= 8;
            self.shift_low();
        }
    }

    pub fn flush(&mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }
}

#[derive(Debug)]
pub struct Ppmd7Encoder {
    pub model: Ppmd7,
    pub rc: RangeEncoder,
}

impl Ppmd7Encoder {
    pub fn new(memory_size: u32, max_order: u32) -> Self {
        let mut model = Ppmd7::new(memory_size);
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

        if ctx_num_stats(&p.base, p.min_context) != 1 {
            // Multi-state path.
            let stats = ctx_stats(&p.base, p.min_context);
            let summ_freq = ctx_summ_freq(&p.base, p.min_context) as u32;
            r.range /= summ_freq;
            let s_first = stats;
            if st_symbol(&p.base, s_first) as u32 == symbol_int {
                let f0 = st_freq(&p.base, s_first) as u32;
                r.encode(0, f0);
                r.normalize();
                p.found_state = s_first;
                p.update1_0();
                return;
            }
            p.prev_success = 0;
            let mut sum = st_freq(&p.base, s_first) as u32;
            let total = ctx_num_stats(&p.base, p.min_context) as u32 - 1;
            let mut s = s_first;
            for _ in 0..total {
                s += 6;
                if st_symbol(&p.base, s) as u32 == symbol_int {
                    let f = st_freq(&p.base, s) as u32;
                    r.encode(sum, f);
                    r.normalize();
                    p.found_state = s;
                    p.update1();
                    return;
                }
                sum += st_freq(&p.base, s) as u32;
            }
            // Escape — encode the escape interval.
            r.encode(sum, summ_freq - sum);

            // Set up char_mask from MinContext stats; mask the LAST scanned
            // symbol explicitly because the loop pairs walk only up to but
            // not including it.
            let prev_found_sym = st_symbol(&p.base, p.found_state);
            p.hi_bits_flag =
                (((prev_found_sym as u32) + 0xC0) >> (8 - 3)) & (1 << 3);
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

        // Single-state context (BinSumm path).
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
            let new_freq = if freq < 128 { freq + 1 } else { freq };
            st_set_freq(&mut p.base, one, new_freq as u8);
            if p.order_fall == 0 && succ > p.text {
                p.min_context = succ;
                p.max_context = succ;
            } else {
                p.update_model();
            }
            return;
        }
        // Mismatch — escape, then masked.
        p.bin_summ[bin_sig.0][bin_sig.1] = pr_after as u16;
        p.init_esc = p.exp_escape[(pr_after >> 10) as usize] as u32;
        r.low = r.low.wrapping_add(bound as u64);
        r.range = r.range.wrapping_sub(bound);
        // (RC_NORM_LOCAL is empty; RC_NORM_REMOTE happens at the start of
        // the masked loop below.)
        let mut char_mask = [0xFFu8; 256];
        char_mask[st_symbol(&p.base, one) as usize] = 0;
        p.prev_success = 0;
        self.encode_masked(symbol, char_mask);
    }

    /// Symmetric to `Ppmd7Decoder::masked_loop` — walks suffix chain, encodes
    /// the symbol using SEE/escape probabilities, masking visited symbols on
    /// each escape.
    fn encode_masked(&mut self, symbol: u8, mut char_mask: [u8; 256]) {
        let symbol_int = symbol as u32;
        loop {
            let p = &mut self.model;
            let r = &mut self.rc;
            r.normalize();
            let mut mc = p.min_context;
            let num_masked = ctx_num_stats(&p.base, mc) as u32;
            let mut i;
            loop {
                p.order_fall = p.order_fall.wrapping_add(1);
                let suffix = ctx_suffix(&p.base, mc);
                if suffix == 0 {
                    return; // end-of-stream marker (caller doesn't actually emit one)
                }
                mc = suffix;
                i = ctx_num_stats(&p.base, mc) as u32;
                if i != num_masked {
                    break;
                }
            }
            p.min_context = mc;

            // Compute SEE entry (mirrors Ppmd7_MakeEscFreq, encoder-side).
            let see_idx;
            let esc_freq;
            if i != 256 {
                let non_masked = i - num_masked;
                let suffix = ctx_suffix(&p.base, mc);
                let suffix_ns = ctx_num_stats(&p.base, suffix) as u32;
                let summ_freq = ctx_summ_freq(&p.base, mc) as u32;
                let outer = p.ns2_indx[(non_masked - 1) as usize] as usize;
                let mut inner = p.hi_bits_flag as usize;
                if non_masked < suffix_ns - i { inner += 1; }
                if summ_freq < 11 * i { inner += 2; }
                if num_masked > non_masked { inner += 4; }
                see_idx = Some((outer, inner));
                let summ = p.see[outer][inner].summ as u32;
                let r_val = summ >> p.see[outer][inner].shift;
                p.see[outer][inner].summ = (summ - r_val) as u16;
                esc_freq = if r_val == 0 { 1 } else { r_val };
            } else {
                see_idx = None;
                esc_freq = 1;
            }

            // Walk states looking for the target symbol.  `s` is the current
            // state pointer; `i_remaining` is the number of states from `s`
            // onward (matches the C variable `i` used in the if-block).
            let stats = ctx_stats(&p.base, mc);
            let mut s = stats;
            let mut sum = 0u32;
            let mut i_remaining = i;
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
                // Mirrors the C `if ((int)cur == symbol) { ... }` block.
                // `i_remaining` here matches C's `i` at break time:
                //   i = NumStats - k where k is the matched state index.
                if let Some(idx) = see_idx { p.see_update(idx); } else { p.dummy_see_update(); }
                p.found_state = found_state_ptr;
                let mut rest_sum = sum + esc_freq;
                let num2 = i_remaining / 2;
                let parity = i_remaining & 1;
                rest_sum = rest_sum.wrapping_add(found_freq & 0u32.wrapping_sub(parity));
                let mut s_walk = s + (parity as u32) * 6;
                for _ in 0..num2 {
                    let s0 = s_walk;
                    let s1 = s_walk + 6;
                    s_walk += 12;
                    let sym0 = st_symbol(&p.base, s0);
                    let sym1 = st_symbol(&p.base, s1);
                    rest_sum += st_freq(&p.base, s0) as u32
                        & char_mask[sym0 as usize] as u32;
                    rest_sum += st_freq(&p.base, s1) as u32
                        & char_mask[sym1 as usize] as u32;
                }
                r.range /= rest_sum;
                r.encode(found_low, found_freq);
                r.normalize();
                p.update2();
                return;
            }

            // Symbol not in this context — encode the escape.
            let total = sum + esc_freq;
            if let Some(idx) = see_idx {
                p.see[idx.0][idx.1].summ = p.see[idx.0][idx.1].summ.wrapping_add(total as u16);
            }
            r.range /= total;
            r.encode(sum, esc_freq);

            // Mask all symbols of the current context.
            let mut sm = ctx_stats(&p.base, p.min_context);
            for _ in 0..i {
                char_mask[st_symbol(&p.base, sm) as usize] = 0;
                sm += 6;
            }
        }
    }
}

/// One-shot encode helper.
pub fn encode_one_shot(data: &[u8], memory_size: u32, max_order: u32) -> Vec<u8> {
    let mut e = Ppmd7Encoder::new(memory_size, max_order);
    e.encode_symbols(data);
    e.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_init() {
        let mut p = Ppmd7::new(MIN_MEM_SIZE);
        p.init(MIN_ORDER);
        assert_eq!(p.max_order, MIN_ORDER);
        assert!(p.min_context > 0);
    }

    #[test]
    fn round_trip_small() {
        let data = b"Hello PPMd7 encoder! Hello hello hello world world!".to_vec();
        let mem = 1 << 16; // 64 KiB
        let order = 6;
        let encoded = encode_one_shot(&data, mem, order);
        let mut dec = Ppmd7Decoder::new(mem, order, &encoded).unwrap();
        let mut out = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            out.push(dec.decode_symbol().unwrap());
        }
        assert_eq!(out, data);
    }
}
