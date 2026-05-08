//! Schindler sort transform inverse, port of `bsc_st_decode` and the
//! `bsc_unst_*_serial` helpers from
//! `plugins/bsc/upstream/libbsc/st/st.cpp`.
//!
//! ST is a generalised BWT that sorts suffixes only by their first
//! `k` characters (`3..=8`). The inverse is more involved than BWT
//! inverse: it reconstructs the original by combining a 256×256
//! bucket histogram of consecutive byte pairs with iteratively
//! refined order bits packed into a `P[]` array of 32-bit slots.
//!
//! The C source picks one of three reconstruct cases depending on
//! `n` and a `failBack` flag (set when any single byte's frequency
//! reaches `0x800000` and the 23-bit packed pointers can't fit). We
//! port all three so the decoder handles small and large blocks.
//!
//! Forward ST is left for a follow-up (most bsc archives use
//! `LIBBSC_BLOCKSORTER_BWT`, not ST).

#![allow(dead_code)]

const ALPHABET_SIZE: usize = 256;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum StError {
    /// `index` is out of `0..n` or `k` is outside `3..=8`.
    BadParameter,
}

/// Inverse Schindler sort transform.
///
/// `t` is rewritten in place with the original bytes. `index` is the
/// libbsc primary index (0-based, in `0..n`), `k` is the sorting
/// depth (3..=8).
pub fn unst(t: &mut [u8], index: i32, k: i32) -> Result<(), StError> {
    let n = t.len();
    if (index < 0) || (n > 0 && index as usize >= n) {
        return Err(StError::BadParameter);
    }
    if (k < 3) || (k > 8) {
        return Err(StError::BadParameter);
    }
    if n <= 1 {
        return Ok(());
    }

    let mut p: Vec<u32> = vec![0; n];
    // bucket[256][256] — flattened to [256*256].
    let mut bucket: Vec<u32> = vec![0; ALPHABET_SIZE * ALPHABET_SIZE];
    let mut count = [0u32; ALPHABET_SIZE];

    let fail_back = unst_sort_serial(t, &mut p, &mut count, &mut bucket, n, k);
    unst_reconstruct_serial(t, &mut p, &count, n, index as u32, fail_back);

    Ok(())
}

/// Sorting pass — builds `P[]` so each entry encodes a refinement
/// bit for the suffix's k-gram order. Returns `true` when any
/// character count >= 0x800000, in which case the 23-bit packed
/// pointer encoding can't fit and the decoder falls back to case 3.
fn unst_sort_serial(
    t: &[u8],
    p: &mut [u32],
    count: &mut [u32; ALPHABET_SIZE],
    bucket: &mut [u32],
    n: usize,
    k: i32,
) -> bool {
    let mut fail_back = false;

    // Frequency count.
    for i in 0..n { count[t[i] as usize] += 1; }

    // Convert count to start offsets and accumulate bucket pairs.
    let mut sum: u32 = 0;
    for c in 0..ALPHABET_SIZE {
        if count[c] >= 0x0080_0000 { fail_back = true; }
        let tmp = sum;
        sum += count[c];
        count[c] = tmp;
        if (count[c] as i32) != sum as i32 {
            // bucket[c << 8 + T[i]]++ for i in [count[c], sum).
            let base = c << 8;
            for i in count[c]..sum {
                bucket[base + t[i as usize] as usize] += 1;
            }
        }
    }

    // Symmetrise bucket: swap (d,c) ↔ (c,d) for d < c.
    for c in 0..ALPHABET_SIZE {
        for d in 0..c {
            let a = (d << 8) | c;
            let b = (c << 8) | d;
            bucket.swap(a, b);
        }
    }

    if k == 3 {
        // K=3 fast path: P[bucket_start] = 1 marks each non-empty
        // bucket; nothing else needed.
        let mut sum: u32 = 0;
        for w in 0..(ALPHABET_SIZE * ALPHABET_SIZE) {
            if bucket[w] > 0 {
                p[sum as usize] = 1;
                sum += bucket[w];
            }
        }
        return fail_back;
    }

    // K >= 4: refine the order with extra rounds.
    let mut index_arr = *count;
    let mut group = [-1i32; ALPHABET_SIZE];

    let mut sum: u32 = 0;
    for w in 0..(ALPHABET_SIZE * ALPHABET_SIZE) {
        let tmp = sum;
        sum += bucket[w];
        bucket[w] = tmp;
        for i in bucket[w]..sum {
            let c = t[i as usize] as usize;
            if group[c] != w as i32 {
                group[c] = w as i32;
                p[index_arr[c] as usize] = 0x8000_0000;
            }
            index_arr[c] += 1;
        }
    }

    let mut mask0: u32 = 0x8000_0000;
    let mut mask1: u32 = 0x4000_0000;
    let mut round = 4;
    while round < k {
        index_arr = *count;
        for v in group.iter_mut() { *v = -1; }

        let mut g = 0i32;
        for i in 0..n {
            if (p[i] & mask0) != 0 { g = i as i32; }
            let c = t[i] as usize;
            if group[c] != g {
                group[c] = g;
                p[index_arr[c] as usize] = p[index_arr[c] as usize].wrapping_add(mask1);
            }
            index_arr[c] += 1;
        }

        mask0 >>= 1;
        mask1 >>= 1;
        round += 1;
    }

    fail_back
}

fn unst_reconstruct_serial(
    t: &mut [u8],
    p: &mut [u32],
    count: &[u32; ALPHABET_SIZE],
    n: usize,
    index: u32,
    fail_back: bool,
) {
    if n < 0x0080_0000 {
        unst_reconstruct_case1_serial(t, p, count, n, index);
    } else if !fail_back {
        unst_reconstruct_case2_serial(t, p, count, n, index);
    } else {
        unst_reconstruct_case3_serial(t, p, count, n, index);
    }
}

fn unst_reconstruct_case1_serial(
    t: &mut [u8],
    p: &mut [u32],
    count: &[u32; ALPHABET_SIZE],
    n: usize,
    start: u32,
) {
    let mut index_arr = *count;
    let mut group = [-1i32; ALPHABET_SIZE];

    let mut g = 0i32;
    for i in 0..n {
        if p[i] > 0 { g = i as i32; }
        let c = t[i] as usize;
        if group[c] < g {
            group[c] = i as i32;
            p[i] = ((c as u32) << 24) | index_arr[c];
        } else {
            p[i] = ((c as u32) << 24) | 0x0080_0000 | (group[c] as u32);
            p[group[c] as usize] = p[group[c] as usize].wrapping_add(1);
        }
        index_arr[c] += 1;
    }

    let mut p_idx = start as usize;
    for i in (0..n).rev() {
        let mut u = p[p_idx];
        if (u & 0x0080_0000) != 0 {
            p_idx = (u & 0x007F_FFFF) as usize;
            u = p[p_idx];
        }
        t[i] = (u >> 24) as u8;
        p[p_idx] = p[p_idx].wrapping_sub(1);
        p_idx = (u & 0x007F_FFFF) as usize;
    }
}

fn unst_reconstruct_case2_serial(
    t: &mut [u8],
    p: &mut [u32],
    count: &[u32; ALPHABET_SIZE],
    n: usize,
    start: u32,
) {
    let mut index_arr = [0u32; ALPHABET_SIZE];
    let mut group = [-1i32; ALPHABET_SIZE];

    let mut g = 0i32;
    for i in 0..n {
        if p[i] > 0 { g = i as i32; }
        let c = t[i] as usize;
        if group[c] < g {
            group[c] = i as i32;
            p[i] = ((c as u32) << 24) | index_arr[c];
        } else {
            // Delta encoding: i - group[c].
            let delta = (i as i32 - group[c]) as u32;
            p[i] = ((c as u32) << 24) | 0x0080_0000 | delta;
            p[group[c] as usize] = p[group[c] as usize].wrapping_add(1);
        }
        index_arr[c] += 1;
    }

    let mut p_idx = start as usize;
    for i in (0..n).rev() {
        let mut u = p[p_idx];
        if (u & 0x0080_0000) != 0 {
            let delta = (u & 0x007F_FFFF) as usize;
            p_idx = p_idx - delta;
            u = p[p_idx];
        }
        let c = (u >> 24) as u8;
        t[i] = c;
        p[p_idx] = p[p_idx].wrapping_sub(1);
        p_idx = ((u & 0x007F_FFFF) + count[c as usize]) as usize;
    }
}

const ST_NUM_FASTBITS: u32 = 10;

#[inline]
fn unst_search(start_index: usize, p: &[u32], v: u32) -> i32 {
    let mut idx = start_index;
    while p[idx] <= v { idx += 1; }
    idx as i32
}

fn unst_reconstruct_case3_serial(
    t: &mut [u8],
    p: &mut [u32],
    count: &[u32; ALPHABET_SIZE],
    n: usize,
    mut start: u32,
) {
    let mut fastbits = [0u8; 1 << ST_NUM_FASTBITS];
    let mut index_arr = *count;
    let mut group = [-1i32; ALPHABET_SIZE];

    let mut g = 0i32;
    for i in 0..n {
        if p[i] > 0 { g = i as i32; }
        let c = t[i] as usize;
        if group[c] < g {
            group[c] = i as i32;
            p[i] = index_arr[c];
        } else {
            p[i] = 0x8000_0000 | (group[c] as u32);
            p[group[c] as usize] = p[group[c] as usize].wrapping_add(1);
        }
        index_arr[c] += 1;
    }

    let mut shift: u32 = 0;
    while ((n as u32 - 1) >> shift) >= (1u32 << ST_NUM_FASTBITS) { shift += 1; }

    let mut v = 0u32;
    for c in 0..ALPHABET_SIZE {
        index_arr[c] = if c + 1 < ALPHABET_SIZE { count[c + 1] } else { n as u32 };
        if count[c] != index_arr[c] {
            while v <= ((index_arr[c] - 1) >> shift) {
                fastbits[v as usize] = c as u8;
                v += 1;
            }
        }
    }

    if (p[start as usize] & 0x8000_0000) != 0 {
        start = p[start as usize] & 0x7FFF_FFFF;
    }

    t[0] = unst_search(fastbits[(start >> shift) as usize] as usize, p, start) as u8;
    p[start as usize] = p[start as usize].wrapping_sub(1);
    start = p[start as usize] + 1;

    let mut p_idx = start;
    for i in (1..n).rev() {
        let mut u = p[p_idx as usize];
        if (u & 0x8000_0000) != 0 {
            p_idx = u & 0x7FFF_FFFF;
            u = p[p_idx as usize];
        }
        t[i] = unst_search(fastbits[(p_idx >> shift) as usize] as usize, p, p_idx) as u8;
        p[p_idx as usize] = p[p_idx as usize].wrapping_sub(1);
        p_idx = u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Forward ST is intentionally not implemented (most bsc archives
    // don't use it). Cross-language correctness is verified by the
    // `st_xcheck` binary against `bsc_st_encode` on real fixtures —
    // see the build matrix in the workspace test runner.

    #[test]
    fn rejects_bad_params() {
        let mut t = vec![1u8, 2, 3, 4];
        assert_eq!(unst(&mut t, -1, 3), Err(StError::BadParameter));
        assert_eq!(unst(&mut t, 5, 3),  Err(StError::BadParameter));
        assert_eq!(unst(&mut t, 0, 2),  Err(StError::BadParameter));
        assert_eq!(unst(&mut t, 0, 9),  Err(StError::BadParameter));
    }

    #[test]
    fn empty_and_singleton() {
        let mut t: Vec<u8> = Vec::new();
        assert!(unst(&mut t, 0, 3).is_ok());
        let mut t2 = vec![42u8];
        assert!(unst(&mut t2, 0, 3).is_ok());
        assert_eq!(t2, vec![42]);
    }
}
