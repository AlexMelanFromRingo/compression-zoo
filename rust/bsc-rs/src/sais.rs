//! Suffix-array construction by SA-IS (Nong, Zhang, Chan 2009 —
//! *Two Efficient Algorithms for Linear Time Suffix Array
//! Construction*).
//!
//! Replaces the prefix-doubling implementation in `bwt.rs::suffix_array`
//! with O(n) construction. The output convention matches the existing
//! `suffix_array` API: `sais_u8(text)` returns `text.len()` indices
//! with the implicit "shorter suffix is smaller" tiebreak (i.e. the
//! sentinel sorts before every real byte). This is what `libsais` and
//! libbsc both use.
//!
//! The algorithm is recursive on the named LMS-substring string. At
//! each level the array shrinks by ≥ 2× so total memory is O(n).
//! There is no global state and no `unsafe`.
//!
//! This is the *unoptimised* SA-IS — clean and obviously correct,
//! roughly the speed of a two-pass radix sort. libsais's heavy
//! optimisations (cache-aware bucket layout, parallel passes,
//! gap-array tricks) are intentionally not ported; if profiling
//! later shows this matters they can be added incrementally.

#![allow(dead_code)]

/// Suffix array of `text` over the byte alphabet (256 symbols),
/// using an implicit smaller-than-everything sentinel at position
/// `text.len()`. Output length is `text.len()`.
pub fn sais_u8(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![0]; }

    // Internal alphabet: shift bytes up by 1 so 0 is reserved for
    // the sentinel. Length n+1, with t[n] = 0.
    let mut t = Vec::with_capacity(n + 1);
    for &b in text { t.push((b as i32) + 1); }
    t.push(0);

    let sa_full = sais_main(&t, 257);
    debug_assert_eq!(sa_full.len(), n + 1);
    debug_assert_eq!(sa_full[0], n as i32, "sentinel must sort first");

    // Drop the sentinel slot (always sa[0]).
    sa_full.into_iter().skip(1).map(|i| i as usize).collect()
}

/// Recursive core. `t` has alphabet `[0, k)` and assumes its last
/// element (`t[n-1]`) is a unique smallest sentinel (i.e. `t[n-1] = 0`
/// and no other position equals 0).
fn sais_main(t: &[i32], k: usize) -> Vec<i32> {
    let n = t.len();
    if n == 1 { return vec![0]; }

    // ------- Combined classify + bucket-count + LMS-collect.
    //
    // Single right-to-left scan that fills:
    //   * `is_s[i]` — L/S type of position i
    //   * `counts[c]` — number of positions with t[i]==c
    //   * `lms_positions` — positions i where i is an LMS suffix start
    //
    // Replaces three separate passes in the textbook reference.
    let mut is_s = vec![false; n];
    let mut counts = vec![0i32; k];
    let mut lms_positions: Vec<usize> = Vec::new();

    is_s[n - 1] = true;
    counts[t[n - 1] as usize] += 1;
    let mut prev_s = true;
    for i in (0..n - 1).rev() {
        let s = if t[i] < t[i + 1] {
            true
        } else if t[i] > t[i + 1] {
            false
        } else {
            prev_s
        };
        is_s[i] = s;
        counts[t[i] as usize] += 1;
        // LMS at position i+1 ⇔ is_s[i+1] && !is_s[i]
        // (we know prev_s = is_s[i+1] from the previous iteration).
        if prev_s && !s {
            lms_positions.push(i + 1);
        }
        prev_s = s;
    }
    // Fix order: we pushed LMS positions back-to-front while scanning
    // right-to-left. Reverse for input-order.
    lms_positions.reverse();

    // ------- Place LMS positions at bucket tails (any order), induce.
    let mut sa = vec![-1i32; n];
    let mut tails_buf = bucket_tails(&counts);
    for &i in &lms_positions {
        let c = t[i] as usize;
        sa[tails_buf[c]] = i as i32;
        if tails_buf[c] > 0 { tails_buf[c] -= 1; }
    }
    induce_sort_l(&mut sa, t, &is_s, &counts);
    induce_sort_s(&mut sa, t, &is_s, &counts);

    // ------- Name the sorted LMS substrings.
    //
    // Walk SA, picking out the LMS positions; each new one gets the
    // next name unless its substring matches the previous LMS's. We
    // store the name *in the position-indexed* `name_of` array so we
    // can read it back in input order in the next loop.
    let mut name_of = vec![-1i32; n];
    let mut cur_name: i32 = -1;
    let mut prev_pos: i32 = -1;
    for &raw in sa.iter() {
        if raw < 0 { continue; }
        let pos = raw as usize;
        if !is_lms_pos_inline(&is_s, pos) { continue; }

        let same = if prev_pos < 0 {
            false
        } else {
            lms_substring_equal(t, &is_s, prev_pos as usize, pos)
        };
        if !same { cur_name += 1; }
        name_of[pos] = cur_name;
        prev_pos = pos as i32;
    }

    let n_lms = lms_positions.len();
    let mut named: Vec<i32> = Vec::with_capacity(n_lms);
    for &p in &lms_positions {
        named.push(name_of[p]);
    }

    // ------- Recurse if any two LMS substrings shared a name.
    let sub_sa: Vec<i32> = if (cur_name + 1) as usize == n_lms {
        // All names unique — the SA of `named` is the inverse perm.
        let mut s = vec![0i32; n_lms];
        for (i, &nm) in named.iter().enumerate() {
            s[nm as usize] = i as i32;
        }
        s
    } else {
        sais_main(&named, (cur_name + 1) as usize)
    };

    // ------- Re-place sorted LMS at tails, induce L then S.
    for slot in sa.iter_mut() { *slot = -1; }
    // Reset tails by recomputing in place (cheaper than the alloc).
    let mut sum = 0i64;
    for (c, &cnt) in counts.iter().enumerate() {
        sum += cnt as i64;
        tails_buf[c] = (sum - 1) as usize;
    }
    for i in (0..n_lms).rev() {
        let p = lms_positions[sub_sa[i] as usize];
        let c = t[p] as usize;
        sa[tails_buf[c]] = p as i32;
        if tails_buf[c] > 0 { tails_buf[c] -= 1; }
    }
    induce_sort_l(&mut sa, t, &is_s, &counts);
    induce_sort_s(&mut sa, t, &is_s, &counts);

    sa
}

/// Fast inlined `is_lms_pos` for the hot loops. Same definition as
/// [`is_lms_pos`] but inlined; helpful when the compiler can't be
/// nudged to inline a `pub(crate)` fn through the iterator pipeline.
#[inline(always)]
fn is_lms_pos_inline(is_s: &[bool], i: usize) -> bool {
    i > 0 && is_s[i] && !is_s[i - 1]
}

/// `is_lms_pos(i)` ⇔ position `i` is the start of an LMS substring.
/// Defined as: `i > 0 && is_s[i] && !is_s[i-1]`.
fn is_lms_pos(is_s: &[bool], i: usize) -> bool {
    i > 0 && is_s[i] && !is_s[i - 1]
}

/// Equality of the two LMS substrings starting at `a` and `b`.
/// An LMS substring runs from one LMS position up to (and including)
/// the next LMS position. Two singleton-sentinel LMS substrings
/// compare unequal *unless* `a == b` (sentinels are unique).
fn lms_substring_equal(t: &[i32], is_s: &[bool], a: usize, b: usize) -> bool {
    let n = t.len();
    if a == n - 1 || b == n - 1 {
        return a == b;
    }
    let mut k = 0;
    loop {
        let pa = a + k;
        let pb = b + k;
        if pa >= n || pb >= n {
            return false;
        }
        if t[pa] != t[pb] || is_s[pa] != is_s[pb] {
            return false;
        }
        if k > 0 {
            let a_lms = is_lms_pos(is_s, pa);
            let b_lms = is_lms_pos(is_s, pb);
            if a_lms && b_lms { return true; }
            if a_lms != b_lms { return false; }
        }
        k += 1;
    }
}

/// Per-character bucket sizes (length `k`).
fn bucket_counts(t: &[i32], k: usize) -> Vec<i32> {
    let mut c = vec![0i32; k];
    for &x in t { c[x as usize] += 1; }
    c
}

/// Inclusive end index of each bucket in the SA. Bucket `c` occupies
/// `[head, tail]`.
fn bucket_tails(counts: &[i32]) -> Vec<usize> {
    let mut tails = vec![0usize; counts.len()];
    let mut sum: i64 = 0;
    for (i, &c) in counts.iter().enumerate() {
        sum += c as i64;
        tails[i] = (sum - 1) as usize;
    }
    tails
}

/// Inclusive start index of each bucket.
fn bucket_heads(counts: &[i32]) -> Vec<usize> {
    let mut heads = vec![0usize; counts.len()];
    let mut sum: i64 = 0;
    for (i, &c) in counts.iter().enumerate() {
        heads[i] = sum as usize;
        sum += c as i64;
    }
    heads
}

/// Place L predecessors of every assigned `sa[i]` into bucket heads,
/// scanning left to right.
fn induce_sort_l(sa: &mut [i32], t: &[i32], is_s: &[bool], counts: &[i32]) {
    let mut heads = bucket_heads(counts);
    for i in 0..sa.len() {
        let j = sa[i];
        if j <= 0 { continue; } // -1 (empty) or 0 (no predecessor).
        let pred = (j - 1) as usize;
        if !is_s[pred] {
            let c = t[pred] as usize;
            sa[heads[c]] = pred as i32;
            heads[c] += 1;
        }
    }
}

/// Place S predecessors of every assigned `sa[i]` into bucket tails,
/// scanning right to left.
fn induce_sort_s(sa: &mut [i32], t: &[i32], is_s: &[bool], counts: &[i32]) {
    let mut tails = bucket_tails(counts);
    for i in (0..sa.len()).rev() {
        let j = sa[i];
        if j <= 0 { continue; }
        let pred = (j - 1) as usize;
        if is_s[pred] {
            let c = t[pred] as usize;
            sa[tails[c]] = pred as i32;
            if tails[c] > 0 { tails[c] -= 1; }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: SA via brute lex sort with shorter-suffix-first.
    fn sa_naive(text: &[u8]) -> Vec<usize> {
        let n = text.len();
        let mut sa: Vec<usize> = (0..n).collect();
        sa.sort_by(|&a, &b| {
            let la = n - a;
            let lb = n - b;
            let lmin = la.min(lb);
            for k in 0..lmin {
                let ca = text[a + k];
                let cb = text[b + k];
                if ca != cb { return ca.cmp(&cb); }
            }
            la.cmp(&lb)
        });
        sa
    }

    fn check(text: &[u8]) {
        let want = sa_naive(text);
        let got = sais_u8(text);
        assert_eq!(got, want, "SA-IS differs from naive on {:?}", text);
    }

    #[test] fn empty()        { assert!(sais_u8(b"").is_empty()); }
    #[test] fn single()       { check(b"X"); }
    #[test] fn two_distinct() { check(b"AB"); }
    #[test] fn banana()       { check(b"BANANA"); }
    #[test] fn periodic_abab(){ check(b"ABABABAB"); }
    #[test] fn all_same_64()  { check(&[0xAA; 64]); }
    #[test] fn ascending()    { check(b"ABCDEFGHIJKLMNOP"); }
    #[test] fn pangram()      { check(b"the quick brown fox jumps over the lazy dog"); }

    #[test]
    fn pseudo_random_4k() {
        let mut input = vec![0u8; 4096];
        let mut x: u32 = 0xDEADBEEF;
        for b in input.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        check(&input);
    }

    #[test]
    fn pseudo_random_64k() {
        let mut input = vec![0u8; 1 << 16];
        let mut x: u32 = 0x12345678;
        for b in input.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        check(&input);
    }

    #[test]
    fn very_periodic() {
        // Forces deep recursion: many equal-length LMS substrings.
        let mut input = Vec::new();
        for _ in 0..1024 { input.extend_from_slice(b"abc"); }
        check(&input);
    }

    #[test]
    fn extreme_alphabet() {
        // All 256 distinct bytes.
        let input: Vec<u8> = (0..=255u8).collect();
        check(&input);
    }
}
