//! LZMA encoder — port of `7zip/C/LzmaEnc.c`.
//!
//! This is a *compatible* encoder rather than a byte-exact mirror of the C
//! reference: we produce LZMA streams that any conformant decoder (including
//! our own [`crate::lzma_dec`] and the reference `7lzma`) accepts. The
//! parsing strategy is greedy with a hash-chain match finder, comparable to
//! the SDK's `algo=0` "fast" mode.

use crate::lzma_dec::{Properties, PROPS_SIZE};

const TOP_VALUE: u32 = 1 << 24;
const NUM_BIT_MODEL_TOTAL_BITS: u32 = 11;
const BIT_MODEL_TOTAL: u32 = 1 << NUM_BIT_MODEL_TOTAL_BITS;
const NUM_MOVE_BITS: u32 = 5;
const PROB_INIT: u16 = (BIT_MODEL_TOTAL >> 1) as u16;

const NUM_STATES: u32 = 12;
const NUM_LIT_STATES: u32 = 7;
const NUM_POS_BITS_MAX: u32 = 4;
const NUM_POS_STATES_MAX: usize = 1 << NUM_POS_BITS_MAX;

const LEN_NUM_LOW_BITS: u32 = 3;
const LEN_NUM_LOW_SYMBOLS: u32 = 1 << LEN_NUM_LOW_BITS;
const LEN_NUM_HIGH_BITS: u32 = 8;
const LEN_NUM_HIGH_SYMBOLS: u32 = 1 << LEN_NUM_HIGH_BITS;

const LEN_LOW: usize = 0;
const LEN_HIGH: usize = LEN_LOW + 2 * (NUM_POS_STATES_MAX << LEN_NUM_LOW_BITS);
const NUM_LEN_PROBS: usize = LEN_HIGH + LEN_NUM_HIGH_SYMBOLS as usize;
const LEN_CHOICE: usize = LEN_LOW;
const LEN_CHOICE2: usize = LEN_LOW + (1 << LEN_NUM_LOW_BITS);

const NUM_FULL_DISTANCES: usize = 1 << (14 >> 1);
const START_POS_MODEL_INDEX: u32 = 4;
const END_POS_MODEL_INDEX: u32 = 14;
const NUM_POS_SLOT_BITS: u32 = 6;
const NUM_LEN_TO_POS_STATES: u32 = 4;
const NUM_ALIGN_BITS: u32 = 4;
const ALIGN_TABLE_SIZE: usize = 1 << NUM_ALIGN_BITS;

const MATCH_MIN_LEN: u32 = 2;

// Same layout as the decoder, see `lzma_dec.rs`.
const SPEC_POS: usize = 0;
const IS_REP0_LONG: usize = SPEC_POS + NUM_FULL_DISTANCES;
const REP_LEN_CODER: usize = IS_REP0_LONG + 16 * NUM_POS_STATES_MAX;
const LEN_CODER: usize = REP_LEN_CODER + NUM_LEN_PROBS;
const IS_MATCH: usize = LEN_CODER + NUM_LEN_PROBS;
const ALIGN_OFF: usize = IS_MATCH + 16 * NUM_POS_STATES_MAX;
const IS_REP: usize = ALIGN_OFF + ALIGN_TABLE_SIZE;
const IS_REP_G0: usize = IS_REP + NUM_STATES as usize;
const IS_REP_G1: usize = IS_REP_G0 + NUM_STATES as usize;
const IS_REP_G2: usize = IS_REP_G1 + NUM_STATES as usize;
const POS_SLOT: usize = IS_REP_G2 + NUM_STATES as usize;
const LITERAL: usize = POS_SLOT + (NUM_LEN_TO_POS_STATES as usize) * (1 << NUM_POS_SLOT_BITS);
const LZMA_LIT_SIZE: usize = 0x300;

// =====================================================================
// Range encoder
// =====================================================================

#[derive(Debug)]
struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: u64,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        Self { low: 0, range: 0xFFFF_FFFF, cache: 0, cache_size: 0, out: Vec::new() }
    }

    fn shift_low(&mut self) {
        let low = self.low as u32;
        let high = (self.low >> 32) as u32;
        self.low = (low << 8) as u64;
        if low < 0xFF00_0000 || high != 0 {
            // Emit the queued cache byte (with possible +1 from carry).
            self.out.push(self.cache.wrapping_add(high as u8));
            self.cache = (low >> 24) as u8;
            if self.cache_size == 0 {
                return;
            }
            // Drain any pending 0xFF (or 0x00 after carry) bytes.
            let extra = high as u8;
            loop {
                self.out.push((0xFFu8).wrapping_add(extra));
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    return;
                }
            }
        }
        self.cache_size += 1;
    }

    #[inline]
    fn normalize(&mut self) {
        if self.range < TOP_VALUE {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn encode_bit(&mut self, prob: &mut u16, bit: u32) {
        let p = *prob as u32;
        let new_bound = (self.range >> NUM_BIT_MODEL_TOTAL_BITS) * p;
        if bit == 0 {
            self.range = new_bound;
            *prob = (p + ((BIT_MODEL_TOTAL - p) >> NUM_MOVE_BITS)) as u16;
        } else {
            self.low = self.low.wrapping_add(new_bound as u64);
            self.range -= new_bound;
            *prob = (p - (p >> NUM_MOVE_BITS)) as u16;
        }
        self.normalize();
    }

    fn encode_direct_bits(&mut self, value: u32, num_bits: u32) {
        for i in (0..num_bits).rev() {
            self.range >>= 1;
            let bit = (value >> i) & 1;
            // bit=1 → low += range; bit=0 → no change.
            self.low = self.low.wrapping_add((self.range as u64) & (0u64.wrapping_sub(bit as u64)));
            self.normalize();
        }
    }

    fn flush(&mut self) {
        for _ in 0..5 {
            self.shift_low();
        }
    }
}

// =====================================================================
// Probability layout helpers
// =====================================================================

fn num_probs(prop: &Properties) -> usize {
    LITERAL + LZMA_LIT_SIZE * (1usize << (prop.lc + prop.lp))
}

#[inline]
fn calc_pos_state(processed_pos: u32, pb_mask: u32) -> usize {
    ((processed_pos & pb_mask) << 4) as usize
}

// =====================================================================
// Match finder: hash-chain on a 3-byte prefix
// =====================================================================

const HASH_BITS: u32 = 18;
const HASH_SIZE: usize = 1 << HASH_BITS;

#[derive(Debug)]
struct MatchFinder {
    hash: Vec<i32>,  // -1 = empty, else last absolute position with this hash
    chain: Vec<i32>, // chain[pos & mask] = previous position
    chain_mask: usize,
    max_chain_len: u32,
    nice_len: u32,
    dict_size: usize,
}

impl MatchFinder {
    fn new(dict_size: u32, nice_len: u32, max_chain_len: u32) -> Self {
        // Power-of-two chain capacity ≥ dict_size.
        let mut chain_size = 1usize;
        while chain_size < dict_size as usize {
            chain_size <<= 1;
        }
        Self {
            hash: vec![-1; HASH_SIZE],
            chain: vec![-1; chain_size],
            chain_mask: chain_size - 1,
            max_chain_len,
            nice_len,
            dict_size: dict_size as usize,
        }
    }

    #[inline(always)]
    fn hash3(b0: u8, b1: u8, b2: u8) -> usize {
        let h = (b0 as u32) ^ ((b1 as u32) << 8) ^ ((b2 as u32) << 5).wrapping_mul(0x9E37_79B9);
        (h.wrapping_mul(0x9E37_79B9) >> (32 - HASH_BITS)) as usize
    }

    /// Find the longest match at `pos` (max length capped by `max_len`).
    /// Returns (length, distance) where distance = pos - match_pos - 1.
    fn find(&mut self, input: &[u8], pos: usize, max_len: usize) -> Option<(usize, usize)> {
        if pos + 3 > input.len() {
            return None;
        }
        let max_len = max_len.min(input.len() - pos);
        let h = Self::hash3(input[pos], input[pos + 1], input[pos + 2]);
        let mut prev = self.hash[h];
        // Insert current position into the chain.
        self.chain[pos & self.chain_mask] = prev;
        self.hash[h] = pos as i32;

        if max_len < MATCH_MIN_LEN as usize {
            return None;
        }
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        let mut chain_iters = self.max_chain_len;
        while prev >= 0 && chain_iters > 0 {
            let p = prev as usize;
            let dist = pos - p;
            if dist > self.dict_size {
                break;
            }
            // Quick reject: peek at the byte one past current best length
            // (skip if it would go past either buffer's end).
            if best_len > 0 && pos + best_len < input.len() {
                if input[p + best_len] != input[pos + best_len] {
                    prev = self.chain[p & self.chain_mask];
                    chain_iters -= 1;
                    continue;
                }
            }
            // Find common prefix length.
            let mut len = 0;
            while len < max_len && input[p + len] == input[pos + len] {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_dist = dist - 1;
                if len >= self.nice_len as usize {
                    break;
                }
            }
            prev = self.chain[p & self.chain_mask];
            chain_iters -= 1;
        }
        if best_len >= MATCH_MIN_LEN as usize {
            Some((best_len, best_dist))
        } else {
            None
        }
    }

    /// Update the chain for position `pos` without searching (used to skip
    /// across a match without scanning).
    fn skip(&mut self, input: &[u8], pos: usize) {
        if pos + 3 > input.len() {
            return;
        }
        let h = Self::hash3(input[pos], input[pos + 1], input[pos + 2]);
        let prev = self.hash[h];
        self.chain[pos & self.chain_mask] = prev;
        self.hash[h] = pos as i32;
    }
}

// =====================================================================
// LZMA encoder
// =====================================================================

#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub lc: u8,
    pub lp: u8,
    pub pb: u8,
    pub dict_size: u32,
    pub nice_len: u32,
    pub max_chain_len: u32,
    pub write_end_mark: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            lc: 3,
            lp: 0,
            pb: 2,
            dict_size: 1 << 23, // 8 MiB
            nice_len: 32,
            max_chain_len: 64,
            write_end_mark: true,
        }
    }
}

pub fn encode_props(cfg: &EncoderConfig) -> [u8; PROPS_SIZE] {
    let mut out = [0u8; PROPS_SIZE];
    out[0] = (cfg.pb * 5 + cfg.lp) * 9 + cfg.lc;
    out[1..5].copy_from_slice(&cfg.dict_size.to_le_bytes());
    out
}

#[derive(Debug)]
pub struct Encoder {
    cfg: EncoderConfig,
    rc: RangeEncoder,
    probs: Vec<u16>,
    state: u32,
    reps: [u32; 4],
    processed_pos: u32,
    pb_mask: u32,
    lp_mask: u32,
    lc: u32,
    mf: MatchFinder,
}

impl Encoder {
    pub fn new(cfg: EncoderConfig) -> Self {
        assert!(cfg.lc + cfg.lp <= 4 || cfg.lc <= 8 && cfg.lp <= 4);
        let prop = Properties { lc: cfg.lc, lp: cfg.lp, pb: cfg.pb, dic_size: cfg.dict_size };
        let n = num_probs(&prop);
        let pb_mask = (1u32 << cfg.pb) - 1;
        let lc = cfg.lc as u32;
        let lp_mask = (0x100u32 << cfg.lp) - (0x100u32 >> lc);
        Self {
            cfg,
            rc: RangeEncoder::new(),
            probs: vec![PROB_INIT; n],
            state: 0,
            reps: [0; 4],
            processed_pos: 0,
            pb_mask,
            lp_mask,
            lc,
            mf: MatchFinder::new(cfg.dict_size, cfg.nice_len, cfg.max_chain_len),
        }
    }

    fn lit_context_off(&self, prev: u8) -> usize {
        if self.processed_pos == 0 {
            return LITERAL;
        }
        let prev_u = prev as u32;
        let lit_state = (((self.processed_pos << 8) + prev_u) & self.lp_mask) << self.lc;
        LITERAL + 3 * lit_state as usize
    }

    fn encode_literal(&mut self, sym: u8, prev: u8, match_byte: Option<u8>) {
        let prob_off = self.lit_context_off(prev);
        // Tree (8 bits MSB-first), with optional matched-mode early bits.
        let s = sym as u32;
        let mut symbol: u32 = 1;
        match match_byte {
            None => {
                for i in (0..8).rev() {
                    let bit = (s >> i) & 1;
                    let idx = prob_off + symbol as usize;
                    let mut p = self.probs[idx];
                    self.rc.encode_bit(&mut p, bit);
                    self.probs[idx] = p;
                    symbol = (symbol + symbol) | bit;
                }
            }
            Some(mb) => {
                // Mirrors the C `LitEnc_EncodeMatched`: prob index uses
                // `offs + (matchByte & offs) + (sym >> 8)`.  `offs` tracks
                // "still in match mode" — initially 0x100, falls to 0 on the
                // first bit that diverges from the matched-byte's bit.
                let mut match_byte = mb as u32;
                let mut offs: u32 = 0x100;
                for i in (0..8).rev() {
                    match_byte <<= 1;
                    let bit_match = (match_byte >> 8) & 1;
                    let bit = (s >> i) & 1;
                    let prob_idx =
                        prob_off + (offs + (match_byte & offs) + symbol) as usize;
                    let mut p = self.probs[prob_idx];
                    self.rc.encode_bit(&mut p, bit);
                    self.probs[prob_idx] = p;
                    symbol = (symbol + symbol) | bit;
                    if bit != bit_match {
                        offs = 0;
                    }
                }
            }
        }
    }

    fn encode_len(&mut self, base: usize, pos_state: usize, len: u32) {
        let len = len - MATCH_MIN_LEN; // 0-based
        let choice_idx = base + LEN_CHOICE;
        let choice2_idx = base + LEN_CHOICE2;
        if len < LEN_NUM_LOW_SYMBOLS {
            let mut p = self.probs[choice_idx];
            self.rc.encode_bit(&mut p, 0);
            self.probs[choice_idx] = p;
            let probs_low = base + LEN_LOW + pos_state;
            self.encode_tree_msb(probs_low, LEN_NUM_LOW_BITS, len);
        } else if len < LEN_NUM_LOW_SYMBOLS * 2 {
            let mut p = self.probs[choice_idx];
            self.rc.encode_bit(&mut p, 1);
            self.probs[choice_idx] = p;
            let mut p2 = self.probs[choice2_idx];
            self.rc.encode_bit(&mut p2, 0);
            self.probs[choice2_idx] = p2;
            let probs_mid = base + LEN_LOW + pos_state + (1 << LEN_NUM_LOW_BITS);
            self.encode_tree_msb(probs_mid, LEN_NUM_LOW_BITS, len - LEN_NUM_LOW_SYMBOLS);
        } else {
            let mut p = self.probs[choice_idx];
            self.rc.encode_bit(&mut p, 1);
            self.probs[choice_idx] = p;
            let mut p2 = self.probs[choice2_idx];
            self.rc.encode_bit(&mut p2, 1);
            self.probs[choice2_idx] = p2;
            let probs_high = base + LEN_HIGH;
            self.encode_tree_msb(probs_high, LEN_NUM_HIGH_BITS, len - LEN_NUM_LOW_SYMBOLS * 2);
        }
    }

    /// Decoder's `TREE_GET_BIT` walks a binary tree decoding MSB-first; the
    /// inverse encoder must do the same.
    fn encode_tree_msb(&mut self, base: usize, num_bits: u32, value: u32) {
        let mut sym: u32 = 1;
        for i in (0..num_bits).rev() {
            let bit = (value >> i) & 1;
            let idx = base + sym as usize;
            let mut p = self.probs[idx];
            self.rc.encode_bit(&mut p, bit);
            self.probs[idx] = p;
            sym = (sym << 1) | bit;
        }
    }

    /// Reverse-tree encode — inverse of decoder's `REV_BIT_VAR` walk.
    /// `value` is the LSB-first packed bit value.
    fn encode_reverse_tree(&mut self, base: usize, num_bits: u32, value: u32) {
        let mut m: u32 = 1;
        let mut i: u32 = 1;
        for k in 0..num_bits {
            let bit = (value >> k) & 1;
            let idx = base + i as usize;
            let mut p = self.probs[idx];
            self.rc.encode_bit(&mut p, bit);
            self.probs[idx] = p;
            if bit == 0 {
                i += m;
                m += m;
            } else {
                m += m;
                i += m;
            }
        }
    }

    /// Encode a match distance. `distance` is the value such that
    /// the decoder will compute `rep0 = distance + 1` (i.e. byte offset
    /// `distance + 1` back).
    fn encode_distance(&mut self, distance: u32, len: u32) {
        let len_to_pos_state = if len < NUM_LEN_TO_POS_STATES + MATCH_MIN_LEN {
            len - MATCH_MIN_LEN
        } else {
            NUM_LEN_TO_POS_STATES - 1
        };
        let probs_pos_slot = POS_SLOT + ((len_to_pos_state) << NUM_POS_SLOT_BITS) as usize;
        let pos_slot = pos_slot_for(distance);
        self.encode_tree_msb(probs_pos_slot, NUM_POS_SLOT_BITS, pos_slot);

        if pos_slot >= START_POS_MODEL_INDEX {
            let num_direct_bits = (pos_slot >> 1) - 1;
            let base = (2u32 | (pos_slot & 1)) << num_direct_bits;
            let lower = distance - base;
            if pos_slot < END_POS_MODEL_INDEX {
                // Reverse tree using SpecPos probs at offset (base + 1) in
                // decoder layout: decoder iterates `distance++` then walks
                // `prob[distance]` — equivalent to indexing probs at
                // SPEC_POS + base.
                let probs_off = SPEC_POS + base as usize;
                self.encode_reverse_tree(probs_off, num_direct_bits, lower);
            } else {
                let direct_bits = num_direct_bits - NUM_ALIGN_BITS;
                let direct_part = lower >> NUM_ALIGN_BITS;
                self.rc.encode_direct_bits(direct_part, direct_bits);
                let align_part = lower & ((1 << NUM_ALIGN_BITS) - 1);
                self.encode_reverse_tree(ALIGN_OFF, NUM_ALIGN_BITS, align_part);
            }
        }
    }

    fn state_after_lit(state: u32) -> u32 {
        if state < NUM_LIT_STATES {
            state - if state < 4 { state } else { 3 }
        } else {
            state - if state < 10 { 3 } else { 6 }
        }
    }
    fn state_after_match(state: u32) -> u32 {
        if state < NUM_LIT_STATES { 7 } else { 10 }
    }

    /// Encode the entire `data` slice as one LZMA payload (no header).
    pub fn encode(&mut self, data: &[u8]) -> Result<Vec<u8>, &'static str> {
        let mut pos = 0usize;
        let mut prev_byte: u8 = 0;
        while pos < data.len() {
            let pos_state = calc_pos_state(self.processed_pos, self.pb_mask);
            let combined_ps = pos_state + self.state as usize;
            let is_match_idx = IS_MATCH + combined_ps;

            let nice = self.cfg.nice_len as usize;
            let max_len = nice.min(data.len() - pos);
            let m = if max_len >= MATCH_MIN_LEN as usize {
                self.mf.find(data, pos, max_len)
            } else {
                None
            };

            if let Some((mlen, mdist)) = m {
                if mlen >= MATCH_MIN_LEN as usize {
                    // Encode match.
                    let mut p = self.probs[is_match_idx];
                    self.rc.encode_bit(&mut p, 1);
                    self.probs[is_match_idx] = p;

                    let is_rep_idx = IS_REP + self.state as usize;
                    let mut p_rep = self.probs[is_rep_idx];
                    self.rc.encode_bit(&mut p_rep, 0); // non-rep
                    self.probs[is_rep_idx] = p_rep;
                    self.encode_len(LEN_CODER, pos_state, mlen as u32);
                    self.encode_distance(mdist as u32, mlen as u32);

                    self.reps[3] = self.reps[2];
                    self.reps[2] = self.reps[1];
                    self.reps[1] = self.reps[0];
                    self.reps[0] = (mdist as u32) + 1;
                    self.state = Self::state_after_match(self.state);
                    self.processed_pos = self.processed_pos.wrapping_add(mlen as u32);

                    // Skip ahead in the match finder (insert positions into
                    // the chain so future searches see them).
                    for i in 1..mlen {
                        self.mf.skip(data, pos + i);
                    }
                    pos += mlen;
                    prev_byte = data[pos - 1];
                    continue;
                }
            }
            // Literal path.
            let mut p = self.probs[is_match_idx];
            self.rc.encode_bit(&mut p, 0);
            self.probs[is_match_idx] = p;
            let sym = data[pos];
            let match_byte = if self.state >= NUM_LIT_STATES && self.reps[0] != 0 {
                let r0 = self.reps[0] as usize;
                if pos >= r0 {
                    Some(data[pos - r0])
                } else {
                    None
                }
            } else {
                None
            };
            self.encode_literal(sym, prev_byte, match_byte);
            self.state = Self::state_after_lit(self.state);
            self.processed_pos = self.processed_pos.wrapping_add(1);
            prev_byte = sym;
            pos += 1;
        }

        if self.cfg.write_end_mark {
            // Encode end-of-stream: a non-rep match with distance == 0xFFFFFFFF
            // and the smallest length (MATCH_MIN_LEN).  The decoder treats
            // this as the "kMatchSpecLenStart" terminator.
            let pos_state = calc_pos_state(self.processed_pos, self.pb_mask);
            let combined_ps = pos_state + self.state as usize;
            let is_match_idx = IS_MATCH + combined_ps;
            let mut p = self.probs[is_match_idx];
            self.rc.encode_bit(&mut p, 1);
            self.probs[is_match_idx] = p;
            let is_rep_idx = IS_REP + self.state as usize;
            let mut p_rep = self.probs[is_rep_idx];
            self.rc.encode_bit(&mut p_rep, 0);
            self.probs[is_rep_idx] = p_rep;
            self.encode_len(LEN_CODER, pos_state, MATCH_MIN_LEN);
            // Distance = 0xFFFFFFFF — same control path as a normal match,
            // ending up in the slot=63 / numDirectBits=30 branch.
            self.encode_distance(0xFFFF_FFFF, MATCH_MIN_LEN);
        }
        self.rc.flush();
        let out = std::mem::take(&mut self.rc.out);
        Ok(out)
    }
}

#[inline]
fn pos_slot_for(distance: u32) -> u32 {
    if distance < 2 {
        return distance;
    }
    // floor(log2(distance)) * 2 + ((distance >> floor(log2(distance))) - 2)... actually:
    // for distance >= 2: pos_slot = (numHighBit-1)*2 + ((distance >> (numHighBit-1)) & 1)
    let bsr = 31 - distance.leading_zeros();
    bsr * 2 + ((distance >> (bsr - 1)) & 1)
}

/// Encode an LZMA-Alone stream (5-byte properties + 8-byte size + payload),
/// the same format as `7lzma e` and our decoder's [`decode_lzma_alone`].
pub fn encode_lzma_alone(data: &[u8], cfg: EncoderConfig) -> Result<Vec<u8>, &'static str> {
    let mut enc = Encoder::new(cfg);
    let payload = enc.encode(data)?;
    let mut out = Vec::with_capacity(13 + payload.len());
    out.extend_from_slice(&encode_props(&cfg));
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_slot_smoke() {
        assert_eq!(pos_slot_for(0), 0);
        assert_eq!(pos_slot_for(1), 1);
        assert_eq!(pos_slot_for(2), 2);
        assert_eq!(pos_slot_for(3), 3);
        assert_eq!(pos_slot_for(4), 4);
        assert_eq!(pos_slot_for(5), 4);
        assert_eq!(pos_slot_for(6), 5);
        assert_eq!(pos_slot_for(7), 5);
        assert_eq!(pos_slot_for(8), 6);
    }

    #[test]
    fn round_trip_literals_only() {
        // Force "no match finder" by setting nice_len high but max_chain_len = 0.
        let mut cfg = EncoderConfig::default();
        cfg.max_chain_len = 0;
        let data = b"Hello".to_vec();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_simple_match() {
        // "abcabc" — at position 3 we should find a 3-byte match for "abc".
        let cfg = EncoderConfig::default();
        let data = b"abcabc".to_vec();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        eprintln!("encoded {} -> {} bytes", data.len(), encoded.len());
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn round_trip_via_our_decoder_small() {
        let data = b"Hello world! Hello hello hello world world!".to_vec();
        let cfg = EncoderConfig::default();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn bisect_failing() {
        let full = b"Hello world! Hello hello hello world world!";
        for n in 1..=full.len() {
            let cfg = EncoderConfig::default();
            let data = &full[..n];
            let encoded = encode_lzma_alone(data, cfg).unwrap();
            match crate::lzma_dec::decode_lzma_alone(&encoded) {
                Ok(decoded) if decoded == data => eprintln!("ok n={n}"),
                Ok(decoded) => panic!("size {n} bytes: mismatch ({:?} vs {:?})", &decoded, data),
                Err(e) => panic!("size {n} bytes: decode error: {e:?}"),
            }
        }
    }

    #[test]
    fn match_at_distance_zero() {
        // "AA" — the second A could be encoded as a match at distance 0.
        let cfg = EncoderConfig::default();
        let data = b"AA".to_vec();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
    #[test]
    fn match_at_distance_one() {
        // "ABABAB" — repeated 2-char pattern, distances 2.
        let cfg = EncoderConfig::default();
        let data = b"ABABABABAB".to_vec();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        eprintln!("enc {:?}", encoded);
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
    #[test]
    fn match_no_end_mark() {
        let mut cfg = EncoderConfig::default();
        cfg.write_end_mark = false;
        let data = b"ABABABABAB".to_vec();
        let encoded = encode_lzma_alone(&data, cfg).unwrap();
        eprintln!("enc-no-em {:?}", encoded);
        let decoded = crate::lzma_dec::decode_lzma_alone(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
