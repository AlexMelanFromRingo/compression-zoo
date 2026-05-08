//! Inverse Burrows–Wheeler Transform.
//!
//! Forward BWT (the suffix-array-based encoder) lives in libsais
//! upstream and is large enough to deserve its own port; we punt on it
//! for now. The inverse is a textbook 30-line algorithm and is enough
//! to make `bsc-rs` decode-only end-to-end once the QLFC port lands.
//!
//! Mirrors `bsc_bwt_decode` from
//! `plugins/bsc/upstream/libbsc/bwt/bwt.cpp`, which itself dispatches
//! to `libsais_unbwt`. Our scalar implementation produces the same
//! output bytes for the same `(T, index)` inputs, just slower and
//! without the secondary-index cache-locality optimisation.

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BwtError {
    /// `index` is out of range for `n`.
    BadParameter,
}

/// Inverse BWT for libsais's wire format (the one libbsc uses).
///
/// libsais's `U` array of length `n` plus `primary` is exactly the
/// suffix-array BWT of `T + $` (length n+1) with the sentinel slot at
/// index `primary` deleted. The sentinel `$` is virtual: it sorts
/// before every real byte but never appears in `U`.
///
/// Concretely:
///     BWT5[i] = U[i]      for i < primary
///     BWT5[primary] = $   (virtual)
///     BWT5[i] = U[i - 1]  for i > primary
///
/// Standard cyclic-rotation LF walk fails on periodic inputs (e.g.
/// `"ABAB"`) because cyclic rotations can be equal, breaking the
/// rank-of-c-in-L = rank-of-c-in-F invariant. The sentinel-augmented
/// suffix BWT side-steps this: every suffix of `T + $` is unique by
/// length, so the LF walk recovers `T` cleanly.
///
/// Algorithm:
///   1. Logically expand `U` into the (n+1)-entry sentinel BWT.
///   2. Treat the sentinel as a virtual character ranked below every
///      real byte; build the LF mapping accordingly.
///   3. Walk LF backwards starting at the sentinel row; the n bytes
///      that follow are `T` in reverse.
pub fn unbwt(t: &mut [u8], primary: i32) -> Result<(), BwtError> {
    let n = t.len();
    if n == 0 {
        return Ok(());
    }
    if primary < 1 || (primary as usize) > n {
        return Err(BwtError::BadParameter);
    }
    let sentinel_row = primary as usize;
    let m = n + 1;

    let mut count = [0u32; 256];
    for &b in t.iter() {
        count[b as usize] += 1;
    }
    let mut bucket_start = [0u32; 256];
    {
        // F[0] is the sentinel, real bytes start at F-position 1.
        let mut sum: u32 = 1;
        for c in 0..256 {
            bucket_start[c] = sum;
            sum += count[c];
        }
    }

    // BWT5[j] -> the byte at virtual row j, or None for the sentinel.
    let bwt5_byte_at = |j: usize| -> Option<u8> {
        if j == sentinel_row {
            None
        } else if j < sentinel_row {
            Some(t[j])
        } else {
            Some(t[j - 1])
        }
    };

    let mut lf: Vec<u32> = vec![0; m];
    let mut next_rank = [0u32; 256];
    for j in 0..m {
        match bwt5_byte_at(j) {
            None => { lf[j] = 0; }
            Some(b) => {
                lf[j] = bucket_start[b as usize] + next_rank[b as usize];
                next_rank[b as usize] += 1;
            }
        }
    }

    let mut out = vec![0u8; n];
    // First LF hop consumes the sentinel itself.
    let mut pos = lf[sentinel_row] as usize;
    for k in (0..n).rev() {
        match bwt5_byte_at(pos) {
            Some(b) => {
                out[k] = b;
                pos = lf[pos] as usize;
            }
            None => {
                // Sentinel should appear exactly once; reaching it here
                // means the wire format is malformed.
                return Err(BwtError::BadParameter);
            }
        }
    }

    t.copy_from_slice(&out);
    Ok(())
}

// ---------------------------------------------------------------------
// Forward BWT (libbsc wire format)
// ---------------------------------------------------------------------

/// Forward BWT compatible with libsais's wire format.
///
/// Returns `(U, primary)` where `U.len() == input.len()` and `primary`
/// is what libbsc calls "the BWT primary index" — `internal_index + 1`
/// (the position of the sentinel row in the (n+1)-entry suffix BWT,
/// using libbsc's 1-based convention).
///
/// Wire-format conventions match `libsais_bwt`:
///   * `U[0]` = `T[n-1]` (the cyclic-wrap character).
///   * Slots `U[1..primary]` = the L column of suffixes
///     `0 .. internal_index - 1`.
///   * Slots `U[primary..n]` = the L column of suffixes
///     `internal_index + 1 .. n - 1`.
///   * The slot at suffix-rank `internal_index` (where SA[i] == 0)
///     would have been the sentinel; libbsc squashes it out and
///     records its position via `primary = internal_index + 1`.
///
/// The forward BWT runs in O(n) via the SA-IS suffix array
/// constructor in [`crate::sais`]. The legacy O(n log² n) prefix-
/// doubling implementation lives in [`suffix_array_prefix_doubling`]
/// for cross-validation in tests.
pub fn encode(input: &[u8]) -> (Vec<u8>, i32) {
    let n = input.len();
    if n == 0 { return (Vec::new(), 0); }
    if n == 1 { return (vec![input[0]], 1); }

    let sa = crate::sais::sais_u8(input);

    let mut bwt = vec![0u8; n];
    let mut internal_index = 0usize;
    for (rank, &start) in sa.iter().enumerate() {
        bwt[rank] = input[(start + n - 1) % n];
        if start == 0 { internal_index = rank; }
    }
    let primary = (internal_index + 1) as i32;

    let mut u = vec![0u8; n];
    u[0] = input[n - 1];
    for j in 0..internal_index {
        u[j + 1] = bwt[j];
    }
    for j in (internal_index + 1)..n {
        u[j] = bwt[j];
    }
    (u, primary)
}

/// Suffix-array construction by prefix doubling. Returns indices into
/// `t` sorted lexicographically (with the implicit "shorter suffix
/// is smaller" tie-break that matches libbsc's BWT wire format).
///
/// O(n log² n). Kept around as a reference / fuzzing oracle; the
/// production path is [`crate::sais::sais_u8`].
pub fn suffix_array_prefix_doubling(t: &[u8]) -> Vec<usize> {
    let n = t.len();
    let mut sa: Vec<usize> = (0..n).collect();

    // Rank starts as the byte itself (0..256).
    let mut rank: Vec<i32> = t.iter().map(|&b| b as i32).collect();
    let mut tmp = vec![0i32; n];

    let mut k = 1usize;
    while k < n {
        // Sort by (rank[i], rank[i+k]).
        let key = |i: usize| -> (i32, i32) {
            let r1 = rank[i];
            let r2 = if i + k < n { rank[i + k] } else { -1 };
            (r1, r2)
        };
        sa.sort_unstable_by(|&a, &b| key(a).cmp(&key(b)));

        // Re-rank using sorted order.
        tmp[sa[0]] = 0;
        for i in 1..n {
            let prev = sa[i - 1];
            let curr = sa[i];
            tmp[curr] = tmp[prev] + if key(curr) != key(prev) { 1 } else { 0 };
        }
        rank.copy_from_slice(&tmp);

        if rank[sa[n - 1]] == (n - 1) as i32 {
            break;
        }
        k *= 2;
    }

    sa
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivially-correct (slow) reference forward BWT in libsais's
    /// wire format. Mirrors `libsais_bwt` by sorting **suffixes** of
    /// `T` (lex; shorter-as-prefix comes first because of the
    /// implicit sentinel). Cyclic-rotation sort is wrong here: it
    /// breaks ties differently for periodic inputs and won't match
    /// libsais.
    ///
    ///   * `BWT[i] = T[(SA[i] - 1 + n) % n]`
    ///   * `internal_index = i where SA[i] == 0`
    ///   * `primary = internal_index + 1`
    ///   * `U[0] = T[n-1]`; `U[1..primary] = BWT[0..internal_index]`;
    ///     `U[primary..n] = BWT[internal_index+1..n]`
    fn bwt_naive(input: &[u8]) -> (Vec<u8>, i32) {
        let n = input.len();
        if n == 0 { return (Vec::new(), 0); }
        if n == 1 { return (vec![input[0]], 1); }
        let mut sa: Vec<usize> = (0..n).collect();
        sa.sort_by(|&a, &b| {
            // Compare suffixes T[a..] vs T[b..] lexicographically.
            // The shorter suffix wins ties (acts as if a sentinel
            // smaller than every byte sits at the end).
            let la = n - a;
            let lb = n - b;
            let lmin = la.min(lb);
            for k in 0..lmin {
                let ca = input[a + k];
                let cb = input[b + k];
                if ca != cb { return ca.cmp(&cb); }
            }
            la.cmp(&lb)
        });
        let mut bwt = vec![0u8; n];
        let mut internal_index = 0usize;
        for (rank, &start) in sa.iter().enumerate() {
            bwt[rank] = input[(start + n - 1) % n];
            if start == 0 { internal_index = rank; }
        }
        let primary = internal_index + 1;
        let mut u = vec![0u8; n];
        u[0] = input[n - 1];
        for j in 0..internal_index {
            u[j + 1] = bwt[j];
        }
        for j in (internal_index + 1)..n {
            u[j] = bwt[j];
        }
        (u, primary as i32)
    }

    fn round_trip(input: &[u8]) {
        let (mut t, index) = bwt_naive(input);
        unbwt(&mut t, index).expect("unbwt");
        assert_eq!(&t[..], input, "BWT inverse mismatch");
    }

    /// Forward BWT (`encode`) must agree with the reference, and the
    /// inverse must round-trip.
    fn forward_round_trip(input: &[u8]) {
        let (u_ref, p_ref) = bwt_naive(input);
        let (u_fast, p_fast) = encode(input);
        assert_eq!(u_fast, u_ref, "fast BWT differs from reference");
        assert_eq!(p_fast, p_ref, "primary differs from reference");

        let mut buf = u_fast.clone();
        unbwt(&mut buf, p_fast).expect("unbwt");
        assert_eq!(&buf[..], input, "fwd→inv round-trip mismatch");
    }

    #[test]
    fn fwd_classic_banana() {
        forward_round_trip(b"BANANA");
    }

    #[test]
    fn fwd_periodic_abab() {
        forward_round_trip(b"ABABABAB");
    }

    #[test]
    fn fwd_all_same_byte() {
        forward_round_trip(&[0xAAu8; 64]);
    }

    #[test]
    fn fwd_ascending() {
        forward_round_trip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn fwd_pseudo_random_4k() {
        let mut input = vec![0u8; 4096];
        let mut x: u32 = 0xDEADBEEF;
        for b in input.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        forward_round_trip(&input);
    }

    #[test]
    fn empty_input() {
        let mut t: Vec<u8> = Vec::new();
        unbwt(&mut t, 0).expect("unbwt empty");
        assert!(t.is_empty());
    }

    #[test]
    fn single_byte() {
        round_trip(b"X");
    }

    #[test]
    fn two_distinct_bytes() {
        round_trip(b"AB");
    }

    #[test]
    fn classic_banana() {
        // "BANANA" is the textbook BWT example.
        round_trip(b"BANANA");
    }

    #[test]
    fn ascending() {
        round_trip(b"ABCDEFGHIJKLMNOP");
    }

    #[test]
    fn repetitive() {
        round_trip(b"the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn all_same_byte() {
        round_trip(&[0xAAu8; 64]);
    }

    #[test]
    fn pseudo_random_256() {
        let mut input = vec![0u8; 256];
        let mut x: u32 = 0xDEADBEEF;
        for b in input.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        round_trip(&input);
    }

    #[test]
    fn rejects_bad_index() {
        let mut t = vec![1, 2, 3, 4];
        assert_eq!(unbwt(&mut t, 0),  Err(BwtError::BadParameter));
        assert_eq!(unbwt(&mut t, -1), Err(BwtError::BadParameter));
        assert_eq!(unbwt(&mut t, 100), Err(BwtError::BadParameter));
    }
}
