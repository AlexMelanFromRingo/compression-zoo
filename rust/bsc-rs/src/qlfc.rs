//! QLFC (Quantized Local Frequency Coding) static decoder, ported
//! from `plugins/bsc/upstream/libbsc/coder/qlfc/qlfc.cpp`'s
//! `bsc_qlfc_static_decode` (the scalar, non-SIMD reference).
//!
//! This is the second-stage entropy decoder libbsc applies to the
//! BWT-sorted bytes. It produces the original (post-LZP) byte stream
//! when invoked with the same model parameters as the encoder.
//!
//! Only the static decode path is implemented; encode + adaptive +
//! fast variants are TODO.

#![allow(dead_code)]

use crate::coder_tables::{model_rank_state, model_run_state};
use crate::predictor::{
    update_bit, update_bit_0, update_bit_1, update_bit_r, update_bit_simple_r,
};
use crate::qlfc_model::*;
use crate::rangecoder::{RangeDecoder, RangeEncoder};

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum QlfcError {
    /// Decoder requested more output than the caller-provided buffer
    /// could hold (the encoded `n` was inconsistent with the buffer
    /// size).
    OutputOverflow,
    /// Wire format claims `n > i32::MAX` or some other shape we don't
    /// trust.
    DataCorrupt,
}

#[inline]
fn bit_scan_reverse(x: u32) -> i32 {
    debug_assert!(x > 0);
    31 - (x.leading_zeros() as i32)
}

/// Static QLFC decoder: read libbsc's wire format from `input` and
/// write the reconstructed byte stream into `output`. Returns the
/// number of bytes written (the `n` recorded in the bitstream header).
pub fn static_decode(input: &[u8], output: &mut [u8]) -> Result<usize, QlfcError> {
    // Allocate the model on the heap; libbsc's `bsc_malloc` does the
    // same (the struct is multi-megabyte and would blow the stack).
    let mut model = QlfcStatisticalModel1::boxed_init();

    let mut mtf_table = [0u8; ALPHABET_SIZE];

    let mut context_rank0: i32 = 0;
    let mut context_rank4: i32 = 0;
    let mut context_run:   i32 = 0;
    let mut max_rank:      i32 = 7;
    let mut avg_rank:      i32 = 0;

    let mut rank_history = [0u8; ALPHABET_SIZE];
    let mut run_history  = [0u8; ALPHABET_SIZE];

    let mut coder = RangeDecoder::new(input);
    let n = coder.decode_word() as i32;
    if n < 0 {
        return Err(QlfcError::DataCorrupt);
    }
    let n = n as usize;
    if n > output.len() {
        return Err(QlfcError::OutputOverflow);
    }

    // ---------------- 1) Decode the MTF alphabet header ------------
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;

    let mut alphabet_filled = ALPHABET_SIZE;
    for rank in 0..ALPHABET_SIZE {
        let mut current_char: i32 = 0;

        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if current_char == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                current_char = current_char + current_char + coder.decode_bit() as i32;
            } else if bit0 {
                current_char += current_char;
            } else if bit1 {
                current_char = current_char + current_char + 1;
            }
        }

        mtf_table[rank] = current_char as u8;

        if current_char == prev_char {
            // Sentinel: alphabet fully described in the first `rank` slots.
            // libbsc calls `bsc_bit_scan_reverse(rank - 1)`. For
            // single-character alphabets `rank == 1`, so the C code
            // computes bsr(0) — undefined; on x86 the `bsr`
            // instruction leaves the destination register unchanged,
            // so libbsc keeps its initial `maxRank = 7`. We match
            // that: only update when `rank >= 2`.
            if rank >= 2 {
                max_rank = bit_scan_reverse(rank as u32 - 1);
            }
            alphabet_filled = rank;
            break;
        }

        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }
    let _ = alphabet_filled; // keeps the assignment for potential future use

    // ---------------- 2) Decode rank+run pairs --------------------
    let mut i: usize = 0;
    while i < n {
        let current_char = mtf_table[0] as usize;
        let history = rank_history[current_char] as i32;
        let state = model_rank_state(context_rank4, context_run, history) as i32;

        let mut rank: i32 = 1;

        if avg_rank < 32 {
            // Rank is a 1 or > 1 (with mantissa+exponent encoding).
            let prob = mix_lr_3(
                model.rank.char_model[current_char] as i32,
                model.rank.state_model[state as usize] as i32,
                model.rank.static_model as i32,
                F_RANK_TM_LR0, F_RANK_TM_LR1, F_RANK_TM_LR2,
            );

            if coder.decode_bit_prob(prob as u32) != 0 {
                update_bit_1(&mut model.rank.state_model[state as usize], F_RANK_TS_TH1, F_RANK_TS_AR1);
                update_bit_1(&mut model.rank.char_model[current_char],     F_RANK_TC_TH1, F_RANK_TC_AR1);
                update_bit_1(&mut model.rank.static_model,                  F_RANK_TP_TH1, F_RANK_TP_AR1);

                // ------------- Exponent bits (unary; bitRankSize = log2(rank)+1) -
                let mut bit_rank_size: i32 = 1;
                let exp_state_base = (state as usize) * 8;
                let exp_char_base  = current_char * 8;
                loop {
                    if bit_rank_size == max_rank { break; }

                    let off = (bit_rank_size - 1) as usize;
                    let prob = mix_lr_3(
                        model.rank.exponent.char_model[exp_char_base + off] as i32,
                        model.rank.exponent.state_model[exp_state_base + off] as i32,
                        model.rank.exponent.static_model[off] as i32,
                        F_RANK_EM_LR0, F_RANK_EM_LR1, F_RANK_EM_LR2,
                    );
                    if coder.decode_bit_prob(prob as u32) != 0 {
                        update_bit_1(&mut model.rank.exponent.state_model[exp_state_base + off], F_RANK_ES_TH1, F_RANK_ES_AR1);
                        update_bit_1(&mut model.rank.exponent.char_model[exp_char_base + off],   F_RANK_EC_TH1, F_RANK_EC_AR1);
                        update_bit_1(&mut model.rank.exponent.static_model[off],                 F_RANK_EP_TH1, F_RANK_EP_AR1);
                        bit_rank_size += 1;
                    } else {
                        update_bit_0(&mut model.rank.exponent.state_model[exp_state_base + off], F_RANK_ES_TH0, F_RANK_ES_AR0);
                        update_bit_0(&mut model.rank.exponent.char_model[exp_char_base + off],   F_RANK_EC_TH0, F_RANK_EC_AR0);
                        update_bit_0(&mut model.rank.exponent.static_model[off],                 F_RANK_EP_TH0, F_RANK_EP_AR0);
                        break;
                    }
                }

                rank_history[current_char] = bit_rank_size as u8;

                // ---------- Mantissa bits (rank's lower bits) ----
                let m = bit_rank_size as usize;
                debug_assert!(m < 8);
                let mant_state_base = (state as usize) * ALPHABET_SIZE;
                let mant_char_base  = current_char * ALPHABET_SIZE;

                for bit in (0..bit_rank_size).rev() {
                    let r = rank as usize;
                    let prob = mix_lr_3(
                        model.rank.mantissa[m].char_model[mant_char_base + r] as i32,
                        model.rank.mantissa[m].state_model[mant_state_base + r] as i32,
                        model.rank.mantissa[m].static_model[r] as i32,
                        F_RANK_MM_LR0, F_RANK_MM_LR1, F_RANK_MM_LR2,
                    );
                    let b = coder.decode_bit_prob(prob as u32);

                    update_bit(b, &mut model.rank.mantissa[m].state_model[mant_state_base + r],
                               F_RANK_MS_TH0, F_RANK_MS_AR0, F_RANK_MS_TH1, F_RANK_MS_AR1);
                    update_bit(b, &mut model.rank.mantissa[m].char_model[mant_char_base + r],
                               F_RANK_MC_TH0, F_RANK_MC_AR0, F_RANK_MC_TH1, F_RANK_MC_AR1);
                    update_bit(b, &mut model.rank.mantissa[m].static_model[r],
                               F_RANK_MP_TH0, F_RANK_MP_AR0, F_RANK_MP_TH1, F_RANK_MP_AR1);

                    rank = rank + rank + b as i32;
                    let _ = bit;
                }
            } else {
                // rank == 1 path.
                rank_history[current_char] = 0;
                update_bit_0(&mut model.rank.state_model[state as usize], F_RANK_TS_TH0, F_RANK_TS_AR0);
                update_bit_0(&mut model.rank.char_model[current_char],     F_RANK_TC_TH0, F_RANK_TC_AR0);
                update_bit_0(&mut model.rank.static_model,                  F_RANK_TP_TH0, F_RANK_TP_AR0);
            }
        } else {
            // Escape path (avg_rank >= 32): read all max_rank+1 bits flat.
            let esc_state_base = (state as usize) * ALPHABET_SIZE;
            let esc_char_base  = current_char * ALPHABET_SIZE;
            rank = 0;
            let mut context: i32 = 1;
            for bit in (0..=max_rank).rev() {
                let ctx = context as usize;
                let prob = mix_lr_3(
                    model.rank.escape.char_model[esc_char_base + ctx] as i32,
                    model.rank.escape.state_model[esc_state_base + ctx] as i32,
                    model.rank.escape.static_model[ctx] as i32,
                    F_RANK_PM_LR0, F_RANK_PM_LR1, F_RANK_PM_LR2,
                );
                let b = coder.decode_bit_prob(prob as u32);

                update_bit(b, &mut model.rank.escape.state_model[esc_state_base + ctx],
                           F_RANK_PS_TH0, F_RANK_PS_AR0, F_RANK_PS_TH1, F_RANK_PS_AR1);
                update_bit(b, &mut model.rank.escape.char_model[esc_char_base + ctx],
                           F_RANK_PC_TH0, F_RANK_PC_AR0, F_RANK_PC_TH1, F_RANK_PC_AR1);
                update_bit(b, &mut model.rank.escape.static_model[ctx],
                           F_RANK_PP_TH0, F_RANK_PP_AR0, F_RANK_PP_TH1, F_RANK_PP_AR1);

                context = context + context + b as i32;
                rank    = rank + rank + b as i32;
                let _ = bit;
            }
            // libbsc: rank_history[currentChar] = bsc_bit_scan_reverse(rank);
            // Undefined for rank==0; treat 0 as "history = 0".
            rank_history[current_char] = if rank > 0 { bit_scan_reverse(rank as u32) as u8 } else { 0 };
        }

        // ---------------- 3) MTF: shift currentChar to position `rank`
        // (scalar fallback, mirrors the C `for r=0..rank: ...`).
        let r = rank as usize;
        if r >= ALPHABET_SIZE { return Err(QlfcError::DataCorrupt); }
        for k in 0..r {
            mtf_table[k] = mtf_table[k + 1];
        }
        mtf_table[r] = current_char as u8;

        avg_rank = (avg_rank * 124 + rank * 4) >> 7;
        let rank_for_run = rank - 1;
        let history = run_history[current_char] as i32;
        let state = model_run_state(context_rank0, context_run, rank_for_run, history) as i32;

        let prob = mix_lr_3(
            model.run.char_model[current_char] as i32,
            model.run.state_model[state as usize] as i32,
            model.run.static_model as i32,
            F_RUN_TM_LR0, F_RUN_TM_LR1, F_RUN_TM_LR2,
        );

        if coder.decode_bit_prob(prob as u32) != 0 {
            update_bit_1(&mut model.run.state_model[state as usize], F_RUN_TS_TH1, F_RUN_TS_AR1);
            update_bit_1(&mut model.run.char_model[current_char],     F_RUN_TC_TH1, F_RUN_TC_AR1);
            update_bit_1(&mut model.run.static_model,                  F_RUN_TP_TH1, F_RUN_TP_AR1);

            // Run > 1: exponent + mantissa.
            let mut run_size: i32 = 1;
            let mut bit_run_size: i32 = 1;

            let exp_state_base = (state as usize) * 32;
            let exp_char_base  = current_char * 32;
            loop {
                let off = (bit_run_size - 1) as usize;
                let prob = mix_lr_3(
                    model.run.exponent.char_model[exp_char_base + off] as i32,
                    model.run.exponent.state_model[exp_state_base + off] as i32,
                    model.run.exponent.static_model[off] as i32,
                    F_RUN_EM_LR0, F_RUN_EM_LR1, F_RUN_EM_LR2,
                );
                if coder.decode_bit_prob(prob as u32) != 0 {
                    update_bit_1(&mut model.run.exponent.state_model[exp_state_base + off], F_RUN_ES_TH1, F_RUN_ES_AR1);
                    update_bit_1(&mut model.run.exponent.char_model[exp_char_base + off],   F_RUN_EC_TH1, F_RUN_EC_AR1);
                    update_bit_1(&mut model.run.exponent.static_model[off],                 F_RUN_EP_TH1, F_RUN_EP_AR1);
                    bit_run_size += 1;
                } else {
                    update_bit_0(&mut model.run.exponent.state_model[exp_state_base + off], F_RUN_ES_TH0, F_RUN_ES_AR0);
                    update_bit_0(&mut model.run.exponent.char_model[exp_char_base + off],   F_RUN_EC_TH0, F_RUN_EC_AR0);
                    update_bit_0(&mut model.run.exponent.static_model[off],                 F_RUN_EP_TH0, F_RUN_EP_AR0);
                    break;
                }
            }
            run_history[current_char] =
                ((run_history[current_char] as i32 + 3 * bit_run_size + 3) >> 2) as u8;

            // Mantissa loop. Note: libbsc's context advancement in this
            // branch is unusual — only the first 5 levels get a true
            // context update; thereafter `context` is incremented by 1
            // regardless of `b`. Mirror exactly.
            let m = bit_run_size as usize;
            debug_assert!(m < 32);
            let mant_state_base = (state as usize) * 32;
            let mant_char_base  = current_char * 32;

            let mut context: i32 = 1;
            for bit in (0..bit_run_size).rev() {
                let ctx = context as usize;
                let prob = mix_lr_3(
                    model.run.mantissa[m].char_model[mant_char_base + ctx] as i32,
                    model.run.mantissa[m].state_model[mant_state_base + ctx] as i32,
                    model.run.mantissa[m].static_model[ctx] as i32,
                    F_RUN_MM_LR0, F_RUN_MM_LR1, F_RUN_MM_LR2,
                );
                let b = coder.decode_bit_prob(prob as u32);

                update_bit(b, &mut model.run.mantissa[m].state_model[mant_state_base + ctx],
                           F_RUN_MS_TH0, F_RUN_MS_AR0, F_RUN_MS_TH1, F_RUN_MS_AR1);
                update_bit(b, &mut model.run.mantissa[m].char_model[mant_char_base + ctx],
                           F_RUN_MC_TH0, F_RUN_MC_AR0, F_RUN_MC_TH1, F_RUN_MC_AR1);
                update_bit(b, &mut model.run.mantissa[m].static_model[ctx],
                           F_RUN_MP_TH0, F_RUN_MP_AR0, F_RUN_MP_TH1, F_RUN_MP_AR1);

                run_size = run_size + run_size + b as i32;
                let new_ctx = context + context + b as i32;
                let next_context = context + 1;
                context = if bit_run_size <= 5 { new_ctx } else { next_context };
                let _ = bit;
            }

            context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
            context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
            context_run   = ((context_run   << 1) | (if run_size < 3 { 1 } else { 0 })) & 0xf;

            for _ in 0..run_size {
                if i >= output.len() { return Err(QlfcError::OutputOverflow); }
                output[i] = current_char as u8;
                i += 1;
            }
        } else {
            // Run == 1.
            run_history[current_char] = ((run_history[current_char] as i32 + 2) >> 2) as u8;
            update_bit_0(&mut model.run.state_model[state as usize], F_RUN_TS_TH0, F_RUN_TS_AR0);
            update_bit_0(&mut model.run.char_model[current_char],     F_RUN_TC_TH0, F_RUN_TC_AR0);
            update_bit_0(&mut model.run.static_model,                  F_RUN_TP_TH0, F_RUN_TP_AR0);

            context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
            context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
            context_run   = ((context_run   << 1) | 1) & 0xf;

            if i >= output.len() { return Err(QlfcError::OutputOverflow); }
            output[i] = current_char as u8;
            i += 1;
        }
    }

    Ok(n)
}

/// `(p0 * lr0 + p1 * lr1 + p2 * lr2) >> 5` — the static-decoder's
/// "logistic mix" replacement (no per-mixer state, just a fixed
/// linear blend). Returns a 12-bit probability.
#[inline]
fn mix_lr_3(p0: i32, p1: i32, p2: i32, lr0: i32, lr1: i32, lr2: i32) -> i32 {
    (p0 * lr0 + p1 * lr1 + p2 * lr2) >> 5
}

// ===================================================================
// QLFC transform (rank+run preparation for the encoder).
// Mirrors the scalar fallback `bsc_qlfc_transform` (qlfc.cpp:469).
// ===================================================================

/// Output of the QLFC transform: the rank array (one byte per maximal
/// run of identical bytes in the input) and the MTF alphabet header
/// the encoder writes before the rank+run pairs.
pub struct QlfcTransform {
    /// One rank per run of identical bytes (left-to-right order).
    pub rank_array: Vec<u8>,
    /// Final MTF table contents — the encoder uses this to write the
    /// alphabet header (cf. the static-decoder header reader).
    pub mtf_table: [u8; ALPHABET_SIZE],
}

/// Run the QLFC transform: count runs of identical bytes (right-to-
/// left so the run-end is `currentChar` matching the preceding byte
/// — same convention as libbsc), maintain a moving MTF table, and
/// emit one rank per run.
///
/// Mirrors the scalar `bsc_qlfc_transform` exactly:
///   * Initial `MTFTable[i] = i`. If the last byte of input is `0`,
///     swap `MTFTable[0]` and `MTFTable[1]` so byte 0 has rank 1
///     (libbsc historical detail).
///   * For each new symbol's first occurrence, the rank is set to
///     `nSymbols++` (the running count of distinct symbols seen).
///   * After the loop, find the first `MTFTable[r]` whose symbol
///     hasn't been used yet and copy `MTFTable[r - 1]` over it. This
///     produces the "duplicate sentinel" the decoder watches for to
///     terminate the alphabet header.
pub fn transform(input: &[u8]) -> QlfcTransform {
    let n = input.len();
    let mut flag = [0u8; ALPHABET_SIZE];
    let mut mtf_table = [0u8; ALPHABET_SIZE];
    for i in 0..ALPHABET_SIZE { mtf_table[i] = i as u8; }

    if n > 0 && input[n - 1] == 0 {
        mtf_table[0] = 1;
        mtf_table[1] = 0;
    }

    // Scan right-to-left, run by run. We don't know how many runs in
    // advance, so we buffer them in a Vec and reverse at the end.
    let mut rank_array_rev: Vec<u8> = Vec::new();
    let mut n_symbols: u8 = 0;
    let mut i = n;
    while i > 0 {
        i -= 1;
        let current_char = input[i];
        // Skip back over the rest of this run.
        while i > 0 && input[i - 1] == current_char {
            i -= 1;
        }

        // Find current_char's rank in mtf_table by walking from the
        // front. As we move, shift the prefix one slot down so that
        // current_char ends up at index 0.
        let mut rank: u8 = 1;
        let mut previous = mtf_table[0];
        mtf_table[0] = current_char;
        loop {
            let t = mtf_table[rank as usize];
            mtf_table[rank as usize] = previous;
            if t == current_char { break; }
            previous = t;
            rank = rank.wrapping_add(1);
        }

        // First time seeing this symbol → override rank with the
        // running symbol counter (used by the alphabet header).
        if flag[current_char as usize] == 0 {
            flag[current_char as usize] = 1;
            rank = n_symbols;
            n_symbols = n_symbols.wrapping_add(1);
        }

        rank_array_rev.push(rank);
    }

    let mut rank_array: Vec<u8> = rank_array_rev.into_iter().rev().collect();
    if !rank_array.is_empty() {
        // libbsc overwrites the last rank to 1 — it's the sentinel
        // that the decoder reads after the alphabet header.
        let last = rank_array.len() - 1;
        rank_array[last] = 1;
    }

    // Patch the MTF alphabet so the decoder sees a duplicate at the
    // first unused slot — that's how the alphabet header terminates.
    for r in 1..ALPHABET_SIZE {
        if flag[mtf_table[r] as usize] == 0 {
            mtf_table[r] = mtf_table[r - 1];
            break;
        }
    }

    QlfcTransform { rank_array, mtf_table }
}

// ===================================================================
// QLFC static encoder — port of `bsc_qlfc_static_encode` (scalar).
// Mirrors the decoder structurally, just calling encode_bit_* paths
// instead of decode_bit_*.
// ===================================================================

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum QlfcEncodeError {
    /// Output buffer overflowed; libbsc treats this as
    /// `LIBBSC_NOT_COMPRESSIBLE` and falls back to bsc_store.
    OutputOverflow,
}

/// Static QLFC encoder. Writes the QLFC bitstream into `out`,
/// producing the same wire format that `static_decode` reads.
///
/// `output_capacity` is the upper bound on bytes written, mirroring
/// libbsc's `outputSize` parameter. If the encoder would exceed that
/// (typical for incompressible random data) we return
/// `OutputOverflow` and the caller is expected to fall back to a
/// store-mode block.
pub fn static_encode(
    input: &[u8],
    out: &mut Vec<u8>,
    output_capacity: usize,
) -> Result<usize, QlfcEncodeError> {
    let n = input.len();
    let mut model = QlfcStatisticalModel1::boxed_init();

    let mut context_rank0: i32 = 0;
    let mut context_rank4: i32 = 0;
    let mut context_run:   i32 = 0;
    let mut max_rank:      i32 = 7;
    let mut avg_rank:      i32 = 0;
    let mut rank_history = [0u8; ALPHABET_SIZE];
    let mut run_history  = [0u8; ALPHABET_SIZE];

    let xform = transform(input);
    let mtf_table = xform.mtf_table;
    let rank_array = xform.rank_array;

    let start_len = out.len();
    let mut coder = RangeEncoder::new(out, output_capacity);
    coder.encode_word(n as u32);

    // ----- 1) Emit the MTF alphabet header -----------------------
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;
    for rank in 0..ALPHABET_SIZE {
        let current_char = mtf_table[rank] as i32;

        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if (current_char >> (bit + 1)) == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                coder.encode_bit(((current_char >> bit) & 1) as u32);
            }
        }

        if current_char == prev_char {
            // libbsc's bsr(rank-1) is UB when rank==1; in that case
            // it leaves maxRank at its initial 7 (matching decoder).
            if rank >= 2 {
                max_rank = bit_scan_reverse(rank as u32 - 1);
            }
            break;
        }

        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }

    // ----- 2) Encode (rank, runSize) pairs -----------------------
    let mut rank_idx = 0usize;
    let mut input_ptr = 0usize;
    while rank_idx < rank_array.len() {
        if coder.check_eob() {
            return Err(QlfcEncodeError::OutputOverflow);
        }
        if input_ptr >= n {
            // Inconsistent transform output — treat as overflow.
            return Err(QlfcEncodeError::OutputOverflow);
        }

        // Find run length at input_ptr.
        let current_char = input[input_ptr];
        let run_start = input_ptr;
        input_ptr += 1;
        while input_ptr < n && input[input_ptr] == current_char {
            input_ptr += 1;
        }
        let run_size = (input_ptr - run_start) as i32;

        let rank = rank_array[rank_idx] as i32;
        rank_idx += 1;

        let cc = current_char as usize;
        let history = rank_history[cc] as i32;
        let state = model_rank_state(context_rank4, context_run, history) as i32;

        if avg_rank < 32 {
            if rank == 1 {
                rank_history[cc] = 0;
                let p0 = model.rank.char_model[cc] as i32;
                let p1 = model.rank.state_model[state as usize] as i32;
                let p2 = model.rank.static_model as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RANK_TM_LR0, F_RANK_TM_LR1, F_RANK_TM_LR2);
                update_bit_0(&mut model.rank.state_model[state as usize], F_RANK_TS_TH0, F_RANK_TS_AR0);
                update_bit_0(&mut model.rank.char_model[cc],               F_RANK_TC_TH0, F_RANK_TC_AR0);
                update_bit_0(&mut model.rank.static_model,                  F_RANK_TP_TH0, F_RANK_TP_AR0);
                coder.encode_bit_0(prob as u32);
            } else {
                let p0 = model.rank.char_model[cc] as i32;
                let p1 = model.rank.state_model[state as usize] as i32;
                let p2 = model.rank.static_model as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RANK_TM_LR0, F_RANK_TM_LR1, F_RANK_TM_LR2);
                update_bit_1(&mut model.rank.state_model[state as usize], F_RANK_TS_TH1, F_RANK_TS_AR1);
                update_bit_1(&mut model.rank.char_model[cc],               F_RANK_TC_TH1, F_RANK_TC_AR1);
                update_bit_1(&mut model.rank.static_model,                  F_RANK_TP_TH1, F_RANK_TP_AR1);
                coder.encode_bit_1(prob as u32);

                let bit_rank_size = bit_scan_reverse(rank as u32);
                rank_history[cc] = bit_rank_size as u8;

                let exp_state_base = (state as usize) * 8;
                let exp_char_base  = cc * 8;

                // bit = 1..bitRankSize (exclusive), encode "continue" 1.
                for bit in 1..bit_rank_size {
                    let off = (bit - 1) as usize;
                    let p0 = model.rank.exponent.char_model[exp_char_base + off] as i32;
                    let p1 = model.rank.exponent.state_model[exp_state_base + off] as i32;
                    let p2 = model.rank.exponent.static_model[off] as i32;
                    let prob = mix_lr_3(p0, p1, p2, F_RANK_EM_LR0, F_RANK_EM_LR1, F_RANK_EM_LR2);
                    update_bit_1(&mut model.rank.exponent.state_model[exp_state_base + off], F_RANK_ES_TH1, F_RANK_ES_AR1);
                    update_bit_1(&mut model.rank.exponent.char_model[exp_char_base + off],   F_RANK_EC_TH1, F_RANK_EC_AR1);
                    update_bit_1(&mut model.rank.exponent.static_model[off],                 F_RANK_EP_TH1, F_RANK_EP_AR1);
                    coder.encode_bit_1(prob as u32);
                }
                // Stop bit (only when bit_rank_size < max_rank).
                if bit_rank_size < max_rank {
                    let off = (bit_rank_size - 1) as usize;
                    let p0 = model.rank.exponent.char_model[exp_char_base + off] as i32;
                    let p1 = model.rank.exponent.state_model[exp_state_base + off] as i32;
                    let p2 = model.rank.exponent.static_model[off] as i32;
                    let prob = mix_lr_3(p0, p1, p2, F_RANK_EM_LR0, F_RANK_EM_LR1, F_RANK_EM_LR2);
                    update_bit_0(&mut model.rank.exponent.state_model[exp_state_base + off], F_RANK_ES_TH0, F_RANK_ES_AR0);
                    update_bit_0(&mut model.rank.exponent.char_model[exp_char_base + off],   F_RANK_EC_TH0, F_RANK_EC_AR0);
                    update_bit_0(&mut model.rank.exponent.static_model[off],                 F_RANK_EP_TH0, F_RANK_EP_AR0);
                    coder.encode_bit_0(prob as u32);
                }

                // Mantissa.
                let m = bit_rank_size as usize;
                let mant_state_base = (state as usize) * ALPHABET_SIZE;
                let mant_char_base  = cc * ALPHABET_SIZE;
                let mut context: i32 = 1;
                for bit in (0..bit_rank_size).rev() {
                    let ctx = context as usize;
                    let p0 = model.rank.mantissa[m].char_model[mant_char_base + ctx] as i32;
                    let p1 = model.rank.mantissa[m].state_model[mant_state_base + ctx] as i32;
                    let p2 = model.rank.mantissa[m].static_model[ctx] as i32;
                    let prob = mix_lr_3(p0, p1, p2, F_RANK_MM_LR0, F_RANK_MM_LR1, F_RANK_MM_LR2);
                    let b = ((rank >> bit) & 1) as u32;
                    update_bit(b, &mut model.rank.mantissa[m].state_model[mant_state_base + ctx], F_RANK_MS_TH0, F_RANK_MS_AR0, F_RANK_MS_TH1, F_RANK_MS_AR1);
                    update_bit(b, &mut model.rank.mantissa[m].char_model[mant_char_base + ctx],   F_RANK_MC_TH0, F_RANK_MC_AR0, F_RANK_MC_TH1, F_RANK_MC_AR1);
                    update_bit(b, &mut model.rank.mantissa[m].static_model[ctx],                  F_RANK_MP_TH0, F_RANK_MP_AR0, F_RANK_MP_TH1, F_RANK_MP_AR1);
                    context = context + context + b as i32;
                    coder.encode_bit_prob(b, prob as u32);
                }
            }
        } else {
            // Escape path.
            rank_history[cc] = if rank > 0 { bit_scan_reverse(rank as u32) as u8 } else { 0 };
            let esc_state_base = (state as usize) * ALPHABET_SIZE;
            let esc_char_base  = cc * ALPHABET_SIZE;
            let mut context: i32 = 1;
            for bit in (0..=max_rank).rev() {
                let ctx = context as usize;
                let p0 = model.rank.escape.char_model[esc_char_base + ctx] as i32;
                let p1 = model.rank.escape.state_model[esc_state_base + ctx] as i32;
                let p2 = model.rank.escape.static_model[ctx] as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RANK_PM_LR0, F_RANK_PM_LR1, F_RANK_PM_LR2);
                let b = ((rank >> bit) & 1) as u32;
                update_bit(b, &mut model.rank.escape.state_model[esc_state_base + ctx], F_RANK_PS_TH0, F_RANK_PS_AR0, F_RANK_PS_TH1, F_RANK_PS_AR1);
                update_bit(b, &mut model.rank.escape.char_model[esc_char_base + ctx],   F_RANK_PC_TH0, F_RANK_PC_AR0, F_RANK_PC_TH1, F_RANK_PC_AR1);
                update_bit(b, &mut model.rank.escape.static_model[ctx],                 F_RANK_PP_TH0, F_RANK_PP_AR0, F_RANK_PP_TH1, F_RANK_PP_AR1);
                context = context + context + b as i32;
                coder.encode_bit_prob(b, prob as u32);
            }
        }

        avg_rank = (avg_rank * 124 + rank * 4) >> 7;
        let rank_for_run = rank - 1;
        let history = run_history[cc] as i32;
        let state = model_run_state(context_rank0, context_run, rank_for_run, history) as i32;

        if run_size == 1 {
            run_history[cc] = ((run_history[cc] as i32 + 2) >> 2) as u8;
            let p0 = model.run.char_model[cc] as i32;
            let p1 = model.run.state_model[state as usize] as i32;
            let p2 = model.run.static_model as i32;
            let prob = mix_lr_3(p0, p1, p2, F_RUN_TM_LR0, F_RUN_TM_LR1, F_RUN_TM_LR2);
            update_bit_0(&mut model.run.state_model[state as usize], F_RUN_TS_TH0, F_RUN_TS_AR0);
            update_bit_0(&mut model.run.char_model[cc],               F_RUN_TC_TH0, F_RUN_TC_AR0);
            update_bit_0(&mut model.run.static_model,                  F_RUN_TP_TH0, F_RUN_TP_AR0);
            coder.encode_bit_0(prob as u32);
        } else {
            let p0 = model.run.char_model[cc] as i32;
            let p1 = model.run.state_model[state as usize] as i32;
            let p2 = model.run.static_model as i32;
            let prob = mix_lr_3(p0, p1, p2, F_RUN_TM_LR0, F_RUN_TM_LR1, F_RUN_TM_LR2);
            update_bit_1(&mut model.run.state_model[state as usize], F_RUN_TS_TH1, F_RUN_TS_AR1);
            update_bit_1(&mut model.run.char_model[cc],               F_RUN_TC_TH1, F_RUN_TC_AR1);
            update_bit_1(&mut model.run.static_model,                  F_RUN_TP_TH1, F_RUN_TP_AR1);
            coder.encode_bit_1(prob as u32);

            let bit_run_size = bit_scan_reverse(run_size as u32);
            run_history[cc] = ((run_history[cc] as i32 + 3 * bit_run_size + 3) >> 2) as u8;

            let exp_state_base = (state as usize) * 32;
            let exp_char_base  = cc * 32;

            for bit in 1..bit_run_size {
                let off = (bit - 1) as usize;
                let p0 = model.run.exponent.char_model[exp_char_base + off] as i32;
                let p1 = model.run.exponent.state_model[exp_state_base + off] as i32;
                let p2 = model.run.exponent.static_model[off] as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RUN_EM_LR0, F_RUN_EM_LR1, F_RUN_EM_LR2);
                update_bit_1(&mut model.run.exponent.state_model[exp_state_base + off], F_RUN_ES_TH1, F_RUN_ES_AR1);
                update_bit_1(&mut model.run.exponent.char_model[exp_char_base + off],   F_RUN_EC_TH1, F_RUN_EC_AR1);
                update_bit_1(&mut model.run.exponent.static_model[off],                 F_RUN_EP_TH1, F_RUN_EP_AR1);
                coder.encode_bit_1(prob as u32);
            }
            // Stop bit (always emitted in run path — note libbsc
            // encodes it unconditionally, unlike the rank path).
            {
                let off = (bit_run_size - 1) as usize;
                let p0 = model.run.exponent.char_model[exp_char_base + off] as i32;
                let p1 = model.run.exponent.state_model[exp_state_base + off] as i32;
                let p2 = model.run.exponent.static_model[off] as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RUN_EM_LR0, F_RUN_EM_LR1, F_RUN_EM_LR2);
                update_bit_0(&mut model.run.exponent.state_model[exp_state_base + off], F_RUN_ES_TH0, F_RUN_ES_AR0);
                update_bit_0(&mut model.run.exponent.char_model[exp_char_base + off],   F_RUN_EC_TH0, F_RUN_EC_AR0);
                update_bit_0(&mut model.run.exponent.static_model[off],                 F_RUN_EP_TH0, F_RUN_EP_AR0);
                coder.encode_bit_0(prob as u32);
            }

            let m = bit_run_size as usize;
            let mant_state_base = (state as usize) * 32;
            let mant_char_base  = cc * 32;
            let mut context: i32 = 1;
            for bit in (0..bit_run_size).rev() {
                let ctx = context as usize;
                let p0 = model.run.mantissa[m].char_model[mant_char_base + ctx] as i32;
                let p1 = model.run.mantissa[m].state_model[mant_state_base + ctx] as i32;
                let p2 = model.run.mantissa[m].static_model[ctx] as i32;
                let prob = mix_lr_3(p0, p1, p2, F_RUN_MM_LR0, F_RUN_MM_LR1, F_RUN_MM_LR2);
                let b = ((run_size >> bit) & 1) as u32;
                update_bit(b, &mut model.run.mantissa[m].state_model[mant_state_base + ctx], F_RUN_MS_TH0, F_RUN_MS_AR0, F_RUN_MS_TH1, F_RUN_MS_AR1);
                update_bit(b, &mut model.run.mantissa[m].char_model[mant_char_base + ctx],   F_RUN_MC_TH0, F_RUN_MC_AR0, F_RUN_MC_TH1, F_RUN_MC_AR1);
                update_bit(b, &mut model.run.mantissa[m].static_model[ctx],                  F_RUN_MP_TH0, F_RUN_MP_AR0, F_RUN_MP_TH1, F_RUN_MP_AR1);
                let new_ctx = context + context + b as i32;
                let next_context = context + 1;
                context = if bit_run_size <= 5 { new_ctx } else { next_context };
                coder.encode_bit_prob(b, prob as u32);
            }
        }

        context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
        context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
        context_run   = ((context_run   << 1) | (if run_size < 3 { 1 } else { 0 })) & 0xf;
    }

    let _ = coder.finish();
    Ok(out.len() - start_len)
}

// ===================================================================
// Adaptive encoder — port of `bsc_qlfc_adaptive_encode` (scalar).
// Mirrors the static encoder but routes probabilities through the
// per-context ProbabilityMixers and updates them after each bit.
// ===================================================================

/// Adaptive QLFC encoder.
pub fn adaptive_encode(
    input: &[u8],
    out: &mut Vec<u8>,
    output_capacity: usize,
) -> Result<usize, QlfcEncodeError> {
    let n = input.len();
    let mut model = QlfcStatisticalModel1::boxed_init();

    let mut context_rank0: i32 = 0;
    let mut context_rank4: i32 = 0;
    let mut context_run:   i32 = 0;
    let mut max_rank:      i32 = 7;
    let mut avg_rank:      i32 = 0;
    let mut rank_history = [0u8; ALPHABET_SIZE];
    let mut run_history  = [0u8; ALPHABET_SIZE];

    let xform = transform(input);
    let mtf_table = xform.mtf_table;
    let rank_array = xform.rank_array;

    let start_len = out.len();
    let mut coder = RangeEncoder::new(out, output_capacity);
    coder.encode_word(n as u32);

    // Alphabet header (identical bit shape to static encoder).
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;
    for rank in 0..ALPHABET_SIZE {
        let current_char = mtf_table[rank] as i32;
        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if (current_char >> (bit + 1)) == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                coder.encode_bit(((current_char >> bit) & 1) as u32);
            }
        }
        if current_char == prev_char {
            if rank >= 2 { max_rank = bit_scan_reverse(rank as u32 - 1); }
            break;
        }
        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }

    let mut rank_idx = 0usize;
    let mut input_ptr = 0usize;
    while rank_idx < rank_array.len() {
        if coder.check_eob() { return Err(QlfcEncodeError::OutputOverflow); }
        if input_ptr >= n { return Err(QlfcEncodeError::OutputOverflow); }

        let current_char = input[input_ptr];
        let run_start = input_ptr;
        input_ptr += 1;
        while input_ptr < n && input[input_ptr] == current_char { input_ptr += 1; }
        let run_size = (input_ptr - run_start) as i32;

        let rank = rank_array[rank_idx] as i32;
        rank_idx += 1;

        let cc = current_char as usize;
        let history = rank_history[cc] as i32;
        let state = model_rank_state(context_rank4, context_run, history) as i32;

        if avg_rank < 32 {
            if rank == 1 {
                rank_history[cc] = 0;
                let p0 = model.rank.char_model[cc] as i32;
                let p1 = model.rank.state_model[state as usize] as i32;
                let p2 = model.rank.static_model as i32;
                update_bit_0(&mut model.rank.state_model[state as usize], M_RANK_TS_TH0, M_RANK_TS_AR0);
                update_bit_0(&mut model.rank.char_model[cc],               M_RANK_TC_TH0, M_RANK_TC_AR0);
                update_bit_0(&mut model.rank.static_model,                  M_RANK_TP_TH0, M_RANK_TP_AR0);
                let prob = model.mixer_of_rank[cc].mixup_and_update_bit_0(
                    p0, p1, p2,
                    M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2,
                    M_RANK_TM_TH0, M_RANK_TM_AR0);
                coder.encode_bit_0(prob as u32);
            } else {
                let p0 = model.rank.char_model[cc] as i32;
                let p1 = model.rank.state_model[state as usize] as i32;
                let p2 = model.rank.static_model as i32;
                update_bit_1(&mut model.rank.state_model[state as usize], M_RANK_TS_TH1, M_RANK_TS_AR1);
                update_bit_1(&mut model.rank.char_model[cc],               M_RANK_TC_TH1, M_RANK_TC_AR1);
                update_bit_1(&mut model.rank.static_model,                  M_RANK_TP_TH1, M_RANK_TP_AR1);
                let prob = model.mixer_of_rank[cc].mixup_and_update_bit_1(
                    p0, p1, p2,
                    M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2,
                    M_RANK_TM_TH1, M_RANK_TM_AR1);
                coder.encode_bit_1(prob as u32);

                let bit_rank_size = bit_scan_reverse(rank as u32);
                rank_history[cc] = bit_rank_size as u8;

                let exp_state_base = (state as usize) * 8;
                let exp_char_base  = cc * 8;

                // Mixer at bit=1: [history<1?1:history][1].
                let mut mixer_idx = (if history < 1 { 1 } else { history } as usize) * 8 + 1;

                for bit in 1..bit_rank_size {
                    let off = (bit - 1) as usize;
                    let p0 = model.rank.exponent.char_model[exp_char_base + off] as i32;
                    let p1 = model.rank.exponent.state_model[exp_state_base + off] as i32;
                    let p2 = model.rank.exponent.static_model[off] as i32;
                    update_bit_1(&mut model.rank.exponent.state_model[exp_state_base + off], M_RANK_ES_TH1, M_RANK_ES_AR1);
                    update_bit_1(&mut model.rank.exponent.char_model[exp_char_base + off],   M_RANK_EC_TH1, M_RANK_EC_AR1);
                    update_bit_1(&mut model.rank.exponent.static_model[off],                 M_RANK_EP_TH1, M_RANK_EP_AR1);
                    let prob = model.mixer_of_rank_exponent[mixer_idx].mixup_and_update_bit_1(
                        p0, p1, p2,
                        M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2,
                        M_RANK_EM_TH1, M_RANK_EM_AR1);
                    coder.encode_bit_1(prob as u32);

                    // After encoding bit 1, mixer = [hist<=bit?bit+1:hist][bit+1].
                    let h = if history <= bit { bit + 1 } else { history };
                    mixer_idx = (h as usize) * 8 + (bit + 1) as usize;
                }
                if bit_rank_size < max_rank {
                    let off = (bit_rank_size - 1) as usize;
                    let p0 = model.rank.exponent.char_model[exp_char_base + off] as i32;
                    let p1 = model.rank.exponent.state_model[exp_state_base + off] as i32;
                    let p2 = model.rank.exponent.static_model[off] as i32;
                    update_bit_0(&mut model.rank.exponent.state_model[exp_state_base + off], M_RANK_ES_TH0, M_RANK_ES_AR0);
                    update_bit_0(&mut model.rank.exponent.char_model[exp_char_base + off],   M_RANK_EC_TH0, M_RANK_EC_AR0);
                    update_bit_0(&mut model.rank.exponent.static_model[off],                 M_RANK_EP_TH0, M_RANK_EP_AR0);
                    let prob = model.mixer_of_rank_exponent[mixer_idx].mixup_and_update_bit_0(
                        p0, p1, p2,
                        M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2,
                        M_RANK_EM_TH0, M_RANK_EM_AR0);
                    coder.encode_bit_0(prob as u32);
                }

                // Mantissa.
                let m = bit_rank_size as usize;
                let mant_state_base = (state as usize) * ALPHABET_SIZE;
                let mant_char_base  = cc * ALPHABET_SIZE;
                let mut context: i32 = 1;
                for bit in (0..bit_rank_size).rev() {
                    let ctx = context as usize;
                    let p0 = model.rank.mantissa[m].char_model[mant_char_base + ctx] as i32;
                    let p1 = model.rank.mantissa[m].state_model[mant_state_base + ctx] as i32;
                    let p2 = model.rank.mantissa[m].static_model[ctx] as i32;
                    let b = (rank >> bit) & 1;
                    if b != 0 {
                        update_bit_1(&mut model.rank.mantissa[m].state_model[mant_state_base + ctx], M_RANK_MS_TH1, M_RANK_MS_AR1);
                        update_bit_1(&mut model.rank.mantissa[m].char_model[mant_char_base + ctx],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        update_bit_1(&mut model.rank.mantissa[m].static_model[ctx],                  M_RANK_MP_TH1, M_RANK_MP_AR1);
                        let prob = model.mixer_of_rank_mantissa[m].mixup_and_update_bit_1(
                            p0, p1, p2,
                            M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2,
                            M_RANK_MM_TH1, M_RANK_MM_AR1);
                        coder.encode_bit_1(prob as u32);
                        context = context + context + 1;
                    } else {
                        update_bit_0(&mut model.rank.mantissa[m].state_model[mant_state_base + ctx], M_RANK_MS_TH0, M_RANK_MS_AR0);
                        update_bit_0(&mut model.rank.mantissa[m].char_model[mant_char_base + ctx],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        update_bit_0(&mut model.rank.mantissa[m].static_model[ctx],                  M_RANK_MP_TH0, M_RANK_MP_AR0);
                        let prob = model.mixer_of_rank_mantissa[m].mixup_and_update_bit_0(
                            p0, p1, p2,
                            M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2,
                            M_RANK_MM_TH0, M_RANK_MM_AR0);
                        coder.encode_bit_0(prob as u32);
                        context = context + context;
                    }
                }
            }
        } else {
            // Escape path (rank packed as max_rank+1 bits flat).
            rank_history[cc] = if rank > 0 { bit_scan_reverse(rank as u32) as u8 } else { 0 };
            let esc_state_base = (state as usize) * ALPHABET_SIZE;
            let esc_char_base  = cc * ALPHABET_SIZE;
            let mut context: i32 = 1;
            for bit in (0..=max_rank).rev() {
                let ctx = context as usize;
                let p0 = model.rank.escape.char_model[esc_char_base + ctx] as i32;
                let p1 = model.rank.escape.state_model[esc_state_base + ctx] as i32;
                let p2 = model.rank.escape.static_model[ctx] as i32;
                let b = (rank >> bit) & 1;
                if b != 0 {
                    update_bit_1(&mut model.rank.escape.state_model[esc_state_base + ctx], M_RANK_PS_TH1, M_RANK_PS_AR1);
                    update_bit_1(&mut model.rank.escape.char_model[esc_char_base + ctx],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    update_bit_1(&mut model.rank.escape.static_model[ctx],                 M_RANK_PP_TH1, M_RANK_PP_AR1);
                    let prob = model.mixer_of_rank_escape[ctx].mixup_and_update_bit_1(
                        p0, p1, p2,
                        M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2,
                        M_RANK_PM_TH1, M_RANK_PM_AR1);
                    coder.encode_bit_1(prob as u32);
                    context = context + context + 1;
                } else {
                    update_bit_0(&mut model.rank.escape.state_model[esc_state_base + ctx], M_RANK_PS_TH0, M_RANK_PS_AR0);
                    update_bit_0(&mut model.rank.escape.char_model[esc_char_base + ctx],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    update_bit_0(&mut model.rank.escape.static_model[ctx],                 M_RANK_PP_TH0, M_RANK_PP_AR0);
                    let prob = model.mixer_of_rank_escape[ctx].mixup_and_update_bit_0(
                        p0, p1, p2,
                        M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2,
                        M_RANK_PM_TH0, M_RANK_PM_AR0);
                    coder.encode_bit_0(prob as u32);
                    context = context + context;
                }
            }
        }

        avg_rank = (avg_rank * 124 + rank * 4) >> 7;
        let rank_for_run = rank - 1;
        let history = run_history[cc] as i32;
        let state = model_run_state(context_rank0, context_run, rank_for_run, history) as i32;

        if run_size == 1 {
            run_history[cc] = ((run_history[cc] as i32 + 2) >> 2) as u8;
            let p0 = model.run.char_model[cc] as i32;
            let p1 = model.run.state_model[state as usize] as i32;
            let p2 = model.run.static_model as i32;
            update_bit_0(&mut model.run.state_model[state as usize], M_RUN_TS_TH0, M_RUN_TS_AR0);
            update_bit_0(&mut model.run.char_model[cc],               M_RUN_TC_TH0, M_RUN_TC_AR0);
            update_bit_0(&mut model.run.static_model,                  M_RUN_TP_TH0, M_RUN_TP_AR0);
            let prob = model.mixer_of_run[cc].mixup_and_update_bit_0(
                p0, p1, p2,
                M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2,
                M_RUN_TM_TH0, M_RUN_TM_AR0);
            coder.encode_bit_0(prob as u32);
        } else {
            let p0 = model.run.char_model[cc] as i32;
            let p1 = model.run.state_model[state as usize] as i32;
            let p2 = model.run.static_model as i32;
            update_bit_1(&mut model.run.state_model[state as usize], M_RUN_TS_TH1, M_RUN_TS_AR1);
            update_bit_1(&mut model.run.char_model[cc],               M_RUN_TC_TH1, M_RUN_TC_AR1);
            update_bit_1(&mut model.run.static_model,                  M_RUN_TP_TH1, M_RUN_TP_AR1);
            let prob = model.mixer_of_run[cc].mixup_and_update_bit_1(
                p0, p1, p2,
                M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2,
                M_RUN_TM_TH1, M_RUN_TM_AR1);
            coder.encode_bit_1(prob as u32);

            let bit_run_size = bit_scan_reverse(run_size as u32);
            run_history[cc] = ((run_history[cc] as i32 + 3 * bit_run_size + 3) >> 2) as u8;

            let exp_state_base = (state as usize) * 32;
            let exp_char_base  = cc * 32;
            let mut mixer_idx = (if history < 1 { 1 } else { history } as usize) * 32 + 1;

            for bit in 1..bit_run_size {
                let off = (bit - 1) as usize;
                let p0 = model.run.exponent.char_model[exp_char_base + off] as i32;
                let p1 = model.run.exponent.state_model[exp_state_base + off] as i32;
                let p2 = model.run.exponent.static_model[off] as i32;
                update_bit_1(&mut model.run.exponent.state_model[exp_state_base + off], M_RUN_ES_TH1, M_RUN_ES_AR1);
                update_bit_1(&mut model.run.exponent.char_model[exp_char_base + off],   M_RUN_EC_TH1, M_RUN_EC_AR1);
                update_bit_1(&mut model.run.exponent.static_model[off],                 M_RUN_EP_TH1, M_RUN_EP_AR1);
                let prob = model.mixer_of_run_exponent[mixer_idx].mixup_and_update_bit_1(
                    p0, p1, p2,
                    M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2,
                    M_RUN_EM_TH1, M_RUN_EM_AR1);
                coder.encode_bit_1(prob as u32);

                let h = if history <= bit { bit + 1 } else { history };
                mixer_idx = (h as usize) * 32 + (bit + 1) as usize;
            }
            // Stop bit (run path always emits it).
            {
                let off = (bit_run_size - 1) as usize;
                let p0 = model.run.exponent.char_model[exp_char_base + off] as i32;
                let p1 = model.run.exponent.state_model[exp_state_base + off] as i32;
                let p2 = model.run.exponent.static_model[off] as i32;
                update_bit_0(&mut model.run.exponent.state_model[exp_state_base + off], M_RUN_ES_TH0, M_RUN_ES_AR0);
                update_bit_0(&mut model.run.exponent.char_model[exp_char_base + off],   M_RUN_EC_TH0, M_RUN_EC_AR0);
                update_bit_0(&mut model.run.exponent.static_model[off],                 M_RUN_EP_TH0, M_RUN_EP_AR0);
                let prob = model.mixer_of_run_exponent[mixer_idx].mixup_and_update_bit_0(
                    p0, p1, p2,
                    M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2,
                    M_RUN_EM_TH0, M_RUN_EM_AR0);
                coder.encode_bit_0(prob as u32);
            }

            // Mantissa.
            let m = bit_run_size as usize;
            let mant_state_base = (state as usize) * 32;
            let mant_char_base  = cc * 32;
            let mut context: i32 = 1;
            for bit in (0..bit_run_size).rev() {
                let ctx = context as usize;
                let p0 = model.run.mantissa[m].char_model[mant_char_base + ctx] as i32;
                let p1 = model.run.mantissa[m].state_model[mant_state_base + ctx] as i32;
                let p2 = model.run.mantissa[m].static_model[ctx] as i32;
                let b = (run_size >> bit) & 1;
                if b != 0 {
                    update_bit_1(&mut model.run.mantissa[m].state_model[mant_state_base + ctx], M_RUN_MS_TH1, M_RUN_MS_AR1);
                    update_bit_1(&mut model.run.mantissa[m].char_model[mant_char_base + ctx],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    update_bit_1(&mut model.run.mantissa[m].static_model[ctx],                  M_RUN_MP_TH1, M_RUN_MP_AR1);
                    let prob = model.mixer_of_run_mantissa[m].mixup_and_update_bit_1(
                        p0, p1, p2,
                        M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2,
                        M_RUN_MM_TH1, M_RUN_MM_AR1);
                    coder.encode_bit_1(prob as u32);
                    context = if bit_run_size <= 5 { context + context + 1 } else { context + 1 };
                } else {
                    update_bit_0(&mut model.run.mantissa[m].state_model[mant_state_base + ctx], M_RUN_MS_TH0, M_RUN_MS_AR0);
                    update_bit_0(&mut model.run.mantissa[m].char_model[mant_char_base + ctx],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    update_bit_0(&mut model.run.mantissa[m].static_model[ctx],                  M_RUN_MP_TH0, M_RUN_MP_AR0);
                    let prob = model.mixer_of_run_mantissa[m].mixup_and_update_bit_0(
                        p0, p1, p2,
                        M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2,
                        M_RUN_MM_TH0, M_RUN_MM_AR0);
                    coder.encode_bit_0(prob as u32);
                    context = if bit_run_size <= 5 { context + context } else { context + 1 };
                }
            }
        }

        context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
        context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
        context_run   = ((context_run   << 1) | (if run_size < 3 { 1 } else { 0 })) & 0xf;
    }

    let _ = coder.finish();
    Ok(out.len() - start_len)
}

// ===================================================================
// Fast encoder — port of `bsc_qlfc_fast_encode` (scalar fallback).
// Uses Model2 + the templated P=11/13 range coder. No mixers.
// ===================================================================

/// Fast QLFC encoder.
pub fn fast_encode(
    input: &[u8],
    out: &mut Vec<u8>,
    output_capacity: usize,
) -> Result<usize, QlfcEncodeError> {
    let n = input.len();
    let mut model = QlfcStatisticalModel2::boxed_init();

    let xform = transform(input);
    let mtf_table = xform.mtf_table;
    let rank_array = xform.rank_array;

    let start_len = out.len();
    let mut coder = RangeEncoder::new(out, output_capacity);
    coder.encode_word(n as u32);

    // Alphabet header — fast variant uses P=1, prob=1 (50/50).
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;
    for rank in 0..ALPHABET_SIZE {
        let current_char = mtf_table[rank] as i32;
        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if (current_char >> (bit + 1)) == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                coder.encode_bit_p(((current_char >> bit) & 1) as u32, 1, 1);
            }
        }
        if current_char == prev_char {
            // Fast variant: no maxRank update (the decoder doesn't
            // need it — Rank exponent is hard-capped at 7).
            break;
        }
        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }

    let mut rank_idx = 0usize;
    let mut input_ptr = 0usize;
    while rank_idx < rank_array.len() {
        if coder.check_eob() { return Err(QlfcEncodeError::OutputOverflow); }
        if input_ptr >= n { return Err(QlfcEncodeError::OutputOverflow); }

        let current_char = input[input_ptr];
        let run_start = input_ptr;
        input_ptr += 1;
        while input_ptr < n && input[input_ptr] == current_char { input_ptr += 1; }
        let run_size = (input_ptr - run_start) as u32;

        let current_rank = rank_array[rank_idx] as u32;
        rank_idx += 1;

        let cc = current_char as usize;

        // ---------- Rank ----------
        let rank_exp_base = cc * 8;
        if current_rank == 1 {
            let p0 = model.rank_exponent[rank_exp_base] as u32;
            update_bit_simple_r(&mut model.rank_exponent[rank_exp_base], 8016, 4);
            coder.encode_bit_0_p(p0, 13);
        } else {
            let p0 = model.rank_exponent[rank_exp_base] as u32;
            update_bit_simple_r(&mut model.rank_exponent[rank_exp_base], 83, 4);
            coder.encode_bit_1_p(p0, 13);

            let bit_rank_size = bit_scan_reverse(current_rank);
            for bit in 1..bit_rank_size {
                let off = bit as usize;
                let p = model.rank_exponent[rank_exp_base + off] as u32;
                update_bit_simple_r(&mut model.rank_exponent[rank_exp_base + off], 122, 4);
                coder.encode_bit_1_p(p, 13);
            }
            if bit_rank_size < 7 {
                let off = bit_rank_size as usize;
                let p = model.rank_exponent[rank_exp_base + off] as u32;
                update_bit_simple_r(&mut model.rank_exponent[rank_exp_base + off], 8114, 4);
                coder.encode_bit_0_p(p, 13);
            }

            let mant_base = cc * 8 * ALPHABET_SIZE
                + (bit_rank_size as usize) * ALPHABET_SIZE;
            let mut context: u32 = 1;
            for bit in (0..bit_rank_size).rev() {
                let b = (current_rank >> bit) & 1;
                let p = model.rank_mantissa[mant_base + context as usize] as u32;
                update_bit_r(b, &mut model.rank_mantissa[mant_base + context as usize], 7999, 235, 7);
                coder.encode_bit_p(b, p, 13);
                context = context + context + b;
            }
        }

        // ---------- Run ----------
        let run_exp_base = cc * 32;
        if run_size == 1 {
            let p0 = model.run_exponent[run_exp_base] as u32;
            update_bit_simple_r(&mut model.run_exponent[run_exp_base], 2025, 5);
            coder.encode_bit_0_p(p0, 11);
        } else {
            let p0 = model.run_exponent[run_exp_base] as u32;
            update_bit_simple_r(&mut model.run_exponent[run_exp_base], 42, 5);
            coder.encode_bit_1_p(p0, 11);

            let bit_run_size = bit_scan_reverse(run_size);
            for bit in 1..bit_run_size {
                let off = bit as usize;
                let p = model.run_exponent[run_exp_base + off] as u32;
                update_bit_simple_r(&mut model.run_exponent[run_exp_base + off], 142, 4);
                coder.encode_bit_1_p(p, 11);
            }
            // Stop bit (always emitted).
            {
                let off = bit_run_size as usize;
                let p = model.run_exponent[run_exp_base + off] as u32;
                update_bit_simple_r(&mut model.run_exponent[run_exp_base + off], 1962, 4);
                coder.encode_bit_0_p(p, 11);
            }

            let mant_base = cc * 32 * 32 + (bit_run_size as usize) * 32;
            if bit_run_size <= 5 {
                let mut context: u32 = 1;
                for bit in (0..bit_run_size).rev() {
                    let b = (run_size >> bit) & 1;
                    let p = model.run_mantissa[mant_base + context as usize] as u32;
                    update_bit_r(b, &mut model.run_mantissa[mant_base + context as usize], 1951, 147, 6);
                    coder.encode_bit_p(b, p, 11);
                    context = context + context + b;
                }
            } else {
                let mut context: u32 = 1;
                for bit in (0..bit_run_size).rev() {
                    let b = (run_size >> bit) & 1;
                    let p = model.run_mantissa[mant_base + context as usize] as u32;
                    update_bit_r(b, &mut model.run_mantissa[mant_base + context as usize], 1987, 46, 5);
                    coder.encode_bit_p(b, p, 11);
                    context += 1;
                }
            }
        }
    }

    let _ = coder.finish();
    Ok(out.len() - start_len)
}

// ===================================================================
// Fast decoder — port of `bsc_qlfc_fast_decode` (scalar fallback).
// Uses QlfcStatisticalModel2 (no mixers, smaller); range coder runs
// at P=13 for rank bits and P=11 for run bits.
// ===================================================================

/// Fast QLFC decoder.
pub fn fast_decode(input: &[u8], output: &mut [u8]) -> Result<usize, QlfcError> {
    let mut model = QlfcStatisticalModel2::boxed_init();
    let mut mtf_table = [0u8; ALPHABET_SIZE];
    let mut coder = RangeDecoder::new(input);
    let n = coder.decode_word() as i32;
    if n < 0 { return Err(QlfcError::DataCorrupt); }
    let n = n as usize;
    if n > output.len() { return Err(QlfcError::OutputOverflow); }

    // ----- MTF alphabet header (uses P=1, prob=1 → fair coin) -----
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;
    for rank in 0..ALPHABET_SIZE {
        let mut current_char: i32 = 0;
        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if current_char == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                current_char = current_char + current_char + coder.decode_bit_p(1, 1) as i32;
            } else if bit0 {
                current_char += current_char;
            } else if bit1 {
                current_char = current_char + current_char + 1;
            }
        }
        mtf_table[rank] = current_char as u8;
        if current_char == prev_char { break; }
        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }

    let mut i: usize = 0;
    while i < n {
        let current_char = mtf_table[0] as usize;

        // ---------- Rank ------------------------------------------
        let rank_exp_base = current_char * 8;
        let p0 = model.rank_exponent[rank_exp_base] as u32;
        let bit = coder.peak_bit_p(p0, 13);
        if bit != 0 {
            update_bit_simple_r(&mut model.rank_exponent[rank_exp_base], 83, 4);
            coder.decode_bit_1_p(p0, 13);

            let mut bit_rank_size: i32 = 1;
            while bit_rank_size < 7 {
                let off = bit_rank_size as usize;
                let p = model.rank_exponent[rank_exp_base + off] as u32;
                if coder.peak_bit_p(p, 13) != 0 {
                    update_bit_simple_r(&mut model.rank_exponent[rank_exp_base + off], 122, 4);
                    bit_rank_size += 1;
                    coder.decode_bit_1_p(p, 13);
                } else {
                    update_bit_simple_r(&mut model.rank_exponent[rank_exp_base + off], 8114, 4);
                    coder.decode_bit_0_p(p, 13);
                    break;
                }
            }

            // Mantissa.
            let mant_base = current_char * 8 * ALPHABET_SIZE
                + (bit_rank_size as usize) * ALPHABET_SIZE;
            let mut rank: u32 = 1;
            let mut bit = bit_rank_size - 1;
            while bit >= 0 {
                let p = model.rank_mantissa[mant_base + rank as usize] as u32;
                let b = coder.decode_bit_p(p, 13);
                update_bit_r(b, &mut model.rank_mantissa[mant_base + rank as usize], 7999, 235, 7);
                rank = rank + rank + b;
                bit -= 1;
            }

            let r = rank as usize;
            if r >= ALPHABET_SIZE { return Err(QlfcError::DataCorrupt); }
            for k in 0..r { mtf_table[k] = mtf_table[k + 1]; }
            mtf_table[r] = current_char as u8;
        } else {
            // rank = 1 special case: swap [0] and [1].
            mtf_table[0] = mtf_table[1];
            mtf_table[1] = current_char as u8;
            update_bit_simple_r(&mut model.rank_exponent[rank_exp_base], 8016, 4);
            coder.decode_bit_0_p(p0, 13);
        }

        // ---------- Run -------------------------------------------
        let run_exp_base = current_char * 32;
        let p0 = model.run_exponent[run_exp_base] as u32;
        if coder.peak_bit_p(p0, 11) != 0 {
            update_bit_simple_r(&mut model.run_exponent[run_exp_base], 42, 5);
            coder.decode_bit_1_p(p0, 11);

            let mut bit_run_size: i32 = 1;
            loop {
                let off = bit_run_size as usize;
                let p = model.run_exponent[run_exp_base + off] as u32;
                if coder.peak_bit_p(p, 11) != 0 {
                    update_bit_simple_r(&mut model.run_exponent[run_exp_base + off], 142, 4);
                    bit_run_size += 1;
                    coder.decode_bit_1_p(p, 11);
                } else {
                    update_bit_simple_r(&mut model.run_exponent[run_exp_base + off], 1962, 4);
                    coder.decode_bit_0_p(p, 11);
                    break;
                }
            }

            let mant_base = current_char * 32 * 32 + (bit_run_size as usize) * 32;
            let mut run_size: u32 = 1;
            if bit_run_size <= 5 {
                let mut bit = bit_run_size - 1;
                while bit >= 0 {
                    let p = model.run_mantissa[mant_base + run_size as usize] as u32;
                    let b = coder.decode_bit_p(p, 11);
                    update_bit_r(b, &mut model.run_mantissa[mant_base + run_size as usize], 1951, 147, 6);
                    run_size = run_size + run_size + b;
                    bit -= 1;
                }
            } else {
                let mut context: i32 = 1;
                while context <= bit_run_size {
                    let p = model.run_mantissa[mant_base + context as usize] as u32;
                    let b = coder.decode_bit_p(p, 11);
                    update_bit_r(b, &mut model.run_mantissa[mant_base + context as usize], 1987, 46, 5);
                    run_size = run_size + run_size + b;
                    context += 1;
                }
            }

            for _ in 0..run_size {
                if i >= output.len() { return Err(QlfcError::OutputOverflow); }
                output[i] = current_char as u8;
                i += 1;
            }
        } else {
            // run = 1.
            if i >= output.len() { return Err(QlfcError::OutputOverflow); }
            output[i] = current_char as u8;
            i += 1;
            update_bit_simple_r(&mut model.run_exponent[run_exp_base], 2025, 5);
            coder.decode_bit_0_p(p0, 11);
        }
    }

    Ok(n)
}

// ===================================================================
// Adaptive decoder — mirrors `bsc_qlfc_adaptive_decode` (scalar
// fallback) in upstream qlfc.cpp. Uses the same model as the static
// decoder but plumbs its probabilities through the per-context
// `ProbabilityMixer`s and updates them after each bit.
// ===================================================================

/// Adaptive QLFC decoder.
pub fn adaptive_decode(input: &[u8], output: &mut [u8]) -> Result<usize, QlfcError> {
    let mut model = QlfcStatisticalModel1::boxed_init();
    let mut mtf_table = [0u8; ALPHABET_SIZE];

    let mut context_rank0: i32 = 0;
    let mut context_rank4: i32 = 0;
    let mut context_run:   i32 = 0;
    let mut max_rank:      i32 = 7;
    let mut avg_rank:      i32 = 0;

    let mut rank_history = [0u8; ALPHABET_SIZE];
    let mut run_history  = [0u8; ALPHABET_SIZE];

    let mut coder = RangeDecoder::new(input);
    let n = coder.decode_word() as i32;
    if n < 0 { return Err(QlfcError::DataCorrupt); }
    let n = n as usize;
    if n > output.len() { return Err(QlfcError::OutputOverflow); }

    // --------- Decode MTF alphabet header (identical to static) ----
    let mut used_char = [0u8; ALPHABET_SIZE];
    let mut prev_char: i32 = -1;
    for rank in 0..ALPHABET_SIZE {
        let mut current_char: i32 = 0;
        for bit in (0..8i32).rev() {
            let mut bit0 = false;
            let mut bit1 = false;
            for c in 0..ALPHABET_SIZE as i32 {
                if c == prev_char || used_char[c as usize] == 0 {
                    if current_char == (c >> (bit + 1)) {
                        if (c & (1 << bit)) != 0 { bit1 = true; } else { bit0 = true; }
                        if bit0 && bit1 { break; }
                    }
                }
            }
            if bit0 && bit1 {
                current_char = current_char + current_char + coder.decode_bit() as i32;
            } else if bit0 {
                current_char += current_char;
            } else if bit1 {
                current_char = current_char + current_char + 1;
            }
        }
        mtf_table[rank] = current_char as u8;
        if current_char == prev_char {
            if rank >= 2 { max_rank = bit_scan_reverse(rank as u32 - 1); }
            break;
        }
        prev_char = current_char;
        used_char[current_char as usize] = 1;
    }

    // --------- rank+run loop --------------------------------------
    let mut i: usize = 0;
    while i < n {
        let current_char = mtf_table[0] as usize;
        let history = rank_history[current_char] as i32;
        let state = model_rank_state(context_rank4, context_run, history) as i32;

        let mut rank: i32 = 1;

        if avg_rank < 32 {
            // Trinary path: rank == 1 vs > 1.
            let mixer = &mut model.mixer_of_rank[current_char];
            let prob = mixer.mixup(
                model.rank.char_model[current_char] as i32,
                model.rank.state_model[state as usize] as i32,
                model.rank.static_model as i32,
            );
            let b = coder.decode_bit_prob(prob as u32);
            if b != 0 {
                update_bit_1(&mut model.rank.state_model[state as usize], M_RANK_TS_TH1, M_RANK_TS_AR1);
                update_bit_1(&mut model.rank.char_model[current_char],     M_RANK_TC_TH1, M_RANK_TC_AR1);
                update_bit_1(&mut model.rank.static_model,                  M_RANK_TP_TH1, M_RANK_TP_AR1);
                model.mixer_of_rank[current_char].update_bit_1(
                    M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH1, M_RANK_TM_AR1);

                // ------------- Exponent ---------------------------
                let mut bit_rank_size: i32 = 1;
                let exp_state_base = (state as usize) * 8;
                let exp_char_base  = current_char * 8;

                let hist_clamp_initial = if history < 1 { 1 } else { history };
                let mut mixer_idx = (hist_clamp_initial as usize) * 8 + 1;

                loop {
                    if bit_rank_size == max_rank { break; }
                    let off = (bit_rank_size - 1) as usize;
                    let prob = model.mixer_of_rank_exponent[mixer_idx].mixup(
                        model.rank.exponent.char_model[exp_char_base + off] as i32,
                        model.rank.exponent.state_model[exp_state_base + off] as i32,
                        model.rank.exponent.static_model[off] as i32,
                    );
                    let b = coder.decode_bit_prob(prob as u32);
                    if b != 0 {
                        update_bit_1(&mut model.rank.exponent.state_model[exp_state_base + off], M_RANK_ES_TH1, M_RANK_ES_AR1);
                        update_bit_1(&mut model.rank.exponent.char_model[exp_char_base + off],   M_RANK_EC_TH1, M_RANK_EC_AR1);
                        update_bit_1(&mut model.rank.exponent.static_model[off],                 M_RANK_EP_TH1, M_RANK_EP_AR1);
                        model.mixer_of_rank_exponent[mixer_idx].update_bit_1(
                            M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH1, M_RANK_EM_AR1);
                        bit_rank_size += 1;
                        let hist_clamp = if history < bit_rank_size { bit_rank_size } else { history };
                        mixer_idx = (hist_clamp as usize) * 8 + (bit_rank_size as usize);
                    } else {
                        update_bit_0(&mut model.rank.exponent.state_model[exp_state_base + off], M_RANK_ES_TH0, M_RANK_ES_AR0);
                        update_bit_0(&mut model.rank.exponent.char_model[exp_char_base + off],   M_RANK_EC_TH0, M_RANK_EC_AR0);
                        update_bit_0(&mut model.rank.exponent.static_model[off],                 M_RANK_EP_TH0, M_RANK_EP_AR0);
                        model.mixer_of_rank_exponent[mixer_idx].update_bit_0(
                            M_RANK_EM_LR0, M_RANK_EM_LR1, M_RANK_EM_LR2, M_RANK_EM_TH0, M_RANK_EM_AR0);
                        break;
                    }
                }
                rank_history[current_char] = bit_rank_size as u8;

                // ------------- Mantissa --------------------------
                let m = bit_rank_size as usize;
                let mant_state_base = (state as usize) * ALPHABET_SIZE;
                let mant_char_base  = current_char * ALPHABET_SIZE;

                for _bit in (0..bit_rank_size).rev() {
                    let r = rank as usize;
                    let prob = model.mixer_of_rank_mantissa[m].mixup(
                        model.rank.mantissa[m].char_model[mant_char_base + r] as i32,
                        model.rank.mantissa[m].state_model[mant_state_base + r] as i32,
                        model.rank.mantissa[m].static_model[r] as i32,
                    );
                    let b = coder.decode_bit_prob(prob as u32);
                    if b != 0 {
                        update_bit_1(&mut model.rank.mantissa[m].state_model[mant_state_base + r], M_RANK_MS_TH1, M_RANK_MS_AR1);
                        update_bit_1(&mut model.rank.mantissa[m].char_model[mant_char_base + r],   M_RANK_MC_TH1, M_RANK_MC_AR1);
                        update_bit_1(&mut model.rank.mantissa[m].static_model[r],                  M_RANK_MP_TH1, M_RANK_MP_AR1);
                        model.mixer_of_rank_mantissa[m].update_bit_1(
                            M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH1, M_RANK_MM_AR1);
                        rank = rank + rank + 1;
                    } else {
                        update_bit_0(&mut model.rank.mantissa[m].state_model[mant_state_base + r], M_RANK_MS_TH0, M_RANK_MS_AR0);
                        update_bit_0(&mut model.rank.mantissa[m].char_model[mant_char_base + r],   M_RANK_MC_TH0, M_RANK_MC_AR0);
                        update_bit_0(&mut model.rank.mantissa[m].static_model[r],                  M_RANK_MP_TH0, M_RANK_MP_AR0);
                        model.mixer_of_rank_mantissa[m].update_bit_0(
                            M_RANK_MM_LR0, M_RANK_MM_LR1, M_RANK_MM_LR2, M_RANK_MM_TH0, M_RANK_MM_AR0);
                        rank = rank + rank;
                    }
                }
            } else {
                rank_history[current_char] = 0;
                update_bit_0(&mut model.rank.state_model[state as usize], M_RANK_TS_TH0, M_RANK_TS_AR0);
                update_bit_0(&mut model.rank.char_model[current_char],     M_RANK_TC_TH0, M_RANK_TC_AR0);
                update_bit_0(&mut model.rank.static_model,                  M_RANK_TP_TH0, M_RANK_TP_AR0);
                model.mixer_of_rank[current_char].update_bit_0(
                    M_RANK_TM_LR0, M_RANK_TM_LR1, M_RANK_TM_LR2, M_RANK_TM_TH0, M_RANK_TM_AR0);
            }
        } else {
            // Escape path.
            let esc_state_base = (state as usize) * ALPHABET_SIZE;
            let esc_char_base  = current_char * ALPHABET_SIZE;
            rank = 0;
            let mut context: i32 = 1;
            for _bit in (0..=max_rank).rev() {
                let ctx = context as usize;
                let prob = model.mixer_of_rank_escape[ctx].mixup(
                    model.rank.escape.char_model[esc_char_base + ctx] as i32,
                    model.rank.escape.state_model[esc_state_base + ctx] as i32,
                    model.rank.escape.static_model[ctx] as i32,
                );
                let b = coder.decode_bit_prob(prob as u32);
                if b != 0 {
                    update_bit_1(&mut model.rank.escape.state_model[esc_state_base + ctx], M_RANK_PS_TH1, M_RANK_PS_AR1);
                    update_bit_1(&mut model.rank.escape.char_model[esc_char_base + ctx],   M_RANK_PC_TH1, M_RANK_PC_AR1);
                    update_bit_1(&mut model.rank.escape.static_model[ctx],                 M_RANK_PP_TH1, M_RANK_PP_AR1);
                    model.mixer_of_rank_escape[ctx].update_bit_1(
                        M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH1, M_RANK_PM_AR1);
                    context = context + context + 1;
                    rank    = rank + rank + 1;
                } else {
                    update_bit_0(&mut model.rank.escape.state_model[esc_state_base + ctx], M_RANK_PS_TH0, M_RANK_PS_AR0);
                    update_bit_0(&mut model.rank.escape.char_model[esc_char_base + ctx],   M_RANK_PC_TH0, M_RANK_PC_AR0);
                    update_bit_0(&mut model.rank.escape.static_model[ctx],                 M_RANK_PP_TH0, M_RANK_PP_AR0);
                    model.mixer_of_rank_escape[ctx].update_bit_0(
                        M_RANK_PM_LR0, M_RANK_PM_LR1, M_RANK_PM_LR2, M_RANK_PM_TH0, M_RANK_PM_AR0);
                    context = context + context;
                    rank    = rank + rank;
                }
            }
            rank_history[current_char] = if rank > 0 { bit_scan_reverse(rank as u32) as u8 } else { 0 };
        }

        // ------------- MTF shift ---------------------------------
        let r = rank as usize;
        if r >= ALPHABET_SIZE { return Err(QlfcError::DataCorrupt); }
        for k in 0..r { mtf_table[k] = mtf_table[k + 1]; }
        mtf_table[r] = current_char as u8;

        avg_rank = (avg_rank * 124 + rank * 4) >> 7;
        let rank_for_run = rank - 1;
        let history = run_history[current_char] as i32;
        let state = model_run_state(context_rank0, context_run, rank_for_run, history) as i32;

        let mixer = &mut model.mixer_of_run[current_char];
        let prob = mixer.mixup(
            model.run.char_model[current_char] as i32,
            model.run.state_model[state as usize] as i32,
            model.run.static_model as i32,
        );
        let b = coder.decode_bit_prob(prob as u32);
        if b != 0 {
            update_bit_1(&mut model.run.state_model[state as usize], M_RUN_TS_TH1, M_RUN_TS_AR1);
            update_bit_1(&mut model.run.char_model[current_char],     M_RUN_TC_TH1, M_RUN_TC_AR1);
            update_bit_1(&mut model.run.static_model,                  M_RUN_TP_TH1, M_RUN_TP_AR1);
            model.mixer_of_run[current_char].update_bit_1(
                M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH1, M_RUN_TM_AR1);

            let mut run_size: i32 = 1;
            let mut bit_run_size: i32 = 1;
            let exp_state_base = (state as usize) * 32;
            let exp_char_base  = current_char * 32;
            let hist_clamp_initial = if history < 1 { 1 } else { history };
            let mut mixer_idx = (hist_clamp_initial as usize) * 32 + 1;
            loop {
                let off = (bit_run_size - 1) as usize;
                let prob = model.mixer_of_run_exponent[mixer_idx].mixup(
                    model.run.exponent.char_model[exp_char_base + off] as i32,
                    model.run.exponent.state_model[exp_state_base + off] as i32,
                    model.run.exponent.static_model[off] as i32,
                );
                let b = coder.decode_bit_prob(prob as u32);
                if b != 0 {
                    update_bit_1(&mut model.run.exponent.state_model[exp_state_base + off], M_RUN_ES_TH1, M_RUN_ES_AR1);
                    update_bit_1(&mut model.run.exponent.char_model[exp_char_base + off],   M_RUN_EC_TH1, M_RUN_EC_AR1);
                    update_bit_1(&mut model.run.exponent.static_model[off],                 M_RUN_EP_TH1, M_RUN_EP_AR1);
                    model.mixer_of_run_exponent[mixer_idx].update_bit_1(
                        M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH1, M_RUN_EM_AR1);
                    bit_run_size += 1;
                    let hist_clamp = if history < bit_run_size { bit_run_size } else { history };
                    mixer_idx = (hist_clamp as usize) * 32 + (bit_run_size as usize);
                } else {
                    update_bit_0(&mut model.run.exponent.state_model[exp_state_base + off], M_RUN_ES_TH0, M_RUN_ES_AR0);
                    update_bit_0(&mut model.run.exponent.char_model[exp_char_base + off],   M_RUN_EC_TH0, M_RUN_EC_AR0);
                    update_bit_0(&mut model.run.exponent.static_model[off],                 M_RUN_EP_TH0, M_RUN_EP_AR0);
                    model.mixer_of_run_exponent[mixer_idx].update_bit_0(
                        M_RUN_EM_LR0, M_RUN_EM_LR1, M_RUN_EM_LR2, M_RUN_EM_TH0, M_RUN_EM_AR0);
                    break;
                }
            }

            run_history[current_char] =
                ((run_history[current_char] as i32 + 3 * bit_run_size + 3) >> 2) as u8;

            let m = bit_run_size as usize;
            let mant_state_base = (state as usize) * 32;
            let mant_char_base  = current_char * 32;
            let mut context: i32 = 1;
            for _bit in (0..bit_run_size).rev() {
                let ctx = context as usize;
                let prob = model.mixer_of_run_mantissa[m].mixup(
                    model.run.mantissa[m].char_model[mant_char_base + ctx] as i32,
                    model.run.mantissa[m].state_model[mant_state_base + ctx] as i32,
                    model.run.mantissa[m].static_model[ctx] as i32,
                );
                let b = coder.decode_bit_prob(prob as u32);
                if b != 0 {
                    update_bit_1(&mut model.run.mantissa[m].state_model[mant_state_base + ctx], M_RUN_MS_TH1, M_RUN_MS_AR1);
                    update_bit_1(&mut model.run.mantissa[m].char_model[mant_char_base + ctx],   M_RUN_MC_TH1, M_RUN_MC_AR1);
                    update_bit_1(&mut model.run.mantissa[m].static_model[ctx],                  M_RUN_MP_TH1, M_RUN_MP_AR1);
                    model.mixer_of_run_mantissa[m].update_bit_1(
                        M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH1, M_RUN_MM_AR1);
                    run_size = run_size + run_size + 1;
                    context = if bit_run_size <= 5 { context + context + 1 } else { context + 1 };
                } else {
                    update_bit_0(&mut model.run.mantissa[m].state_model[mant_state_base + ctx], M_RUN_MS_TH0, M_RUN_MS_AR0);
                    update_bit_0(&mut model.run.mantissa[m].char_model[mant_char_base + ctx],   M_RUN_MC_TH0, M_RUN_MC_AR0);
                    update_bit_0(&mut model.run.mantissa[m].static_model[ctx],                  M_RUN_MP_TH0, M_RUN_MP_AR0);
                    model.mixer_of_run_mantissa[m].update_bit_0(
                        M_RUN_MM_LR0, M_RUN_MM_LR1, M_RUN_MM_LR2, M_RUN_MM_TH0, M_RUN_MM_AR0);
                    run_size = run_size + run_size;
                    context = if bit_run_size <= 5 { context + context } else { context + 1 };
                }
            }

            context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
            context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
            context_run   = ((context_run   << 1) | (if run_size < 3 { 1 } else { 0 })) & 0xf;

            for _ in 0..run_size {
                if i >= output.len() { return Err(QlfcError::OutputOverflow); }
                output[i] = current_char as u8;
                i += 1;
            }
        } else {
            run_history[current_char] = ((run_history[current_char] as i32 + 2) >> 2) as u8;
            update_bit_0(&mut model.run.state_model[state as usize], M_RUN_TS_TH0, M_RUN_TS_AR0);
            update_bit_0(&mut model.run.char_model[current_char],     M_RUN_TC_TH0, M_RUN_TC_AR0);
            update_bit_0(&mut model.run.static_model,                  M_RUN_TP_TH0, M_RUN_TP_AR0);
            model.mixer_of_run[current_char].update_bit_0(
                M_RUN_TM_LR0, M_RUN_TM_LR1, M_RUN_TM_LR2, M_RUN_TM_TH0, M_RUN_TM_AR0);

            context_rank0 = ((context_rank0 << 1) | (if rank_for_run == 0 { 1 } else { 0 })) & 0x7;
            context_rank4 = ((context_rank4 << 2) | (if rank_for_run < 3 { rank_for_run } else { 3 })) & 0xff;
            context_run   = ((context_run   << 1) | 1) & 0xf;

            if i >= output.len() { return Err(QlfcError::OutputOverflow); }
            output[i] = current_char as u8;
            i += 1;
        }
    }

    Ok(n)
}
