//! Port of `libbsc/adler32/adler32.cpp` (the portable scalar path).
//!
//! `bsc_adler32(T, n, features)` returns Adler-32 with the standard
//! `BASE = 65521`, processed in chunks of 5504 bytes (`NMAX`) to allow
//! the inner accumulator additions to stay in `u32` range without
//! intermediate `mod` operations.
//!
//! libbsc has SSE2 / SSSE3 / AVX / AVX2 / NEON variants too; we ignore
//! those here and stick to the deterministic scalar path. Output bits are
//! identical to upstream's scalar `bsc_adler32`.

const BASE: u32 = 65521;
const NMAX: usize = 5504;

/// Compute the libbsc Adler-32 of `data`.
pub fn adler32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 1;
    let mut sum2: u32 = 0;

    let mut i = 0usize;
    let n = data.len();

    // NMAX-sized chunks: 344 iterations of the 16-byte unrolled loop.
    while n - i >= NMAX {
        for _ in 0..(NMAX / 16) {
            // Manually unrolled DO16(buf).
            for k in 0..16 {
                sum1 = sum1.wrapping_add(data[i + k] as u32);
                sum2 = sum2.wrapping_add(sum1);
            }
            i += 16;
        }
        sum1 %= BASE;
        sum2 %= BASE;
    }

    while n - i >= 16 {
        for k in 0..16 {
            sum1 = sum1.wrapping_add(data[i + k] as u32);
            sum2 = sum2.wrapping_add(sum1);
        }
        i += 16;
    }

    while i < n {
        sum1 = sum1.wrapping_add(data[i] as u32);
        sum2 = sum2.wrapping_add(sum1);
        i += 1;
    }

    sum1 %= BASE;
    sum2 %= BASE;
    sum1 | (sum2 << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        // Adler-32 seed: sum1 = 1, sum2 = 0. So for empty input the result
        // is 1 | (0 << 16) = 1.
        assert_eq!(adler32(&[]), 1);
    }

    #[test]
    fn one_zero_byte() {
        // sum1 = 1 + 0 = 1, sum2 = 0 + 1 = 1.  result = 1 | (1<<16).
        assert_eq!(adler32(&[0]), (1 << 16) | 1);
    }

    #[test]
    fn ascii_abc() {
        // RFC 1950 example: Adler-32("abc") = 0x024d0127.
        // sum1 = 1+97+98+99 = 295 = 0x0127
        // sum2 = (1+97) + (1+97+98) + (1+97+98+99) = 98 + 196 + 295 = 589 = 0x024d
        assert_eq!(adler32(b"abc"), 0x024d_0127);
    }

    #[test]
    fn larger_input_no_modulo_needed() {
        // 100 bytes of zeros: sum1 = 1, sum2 = 100.
        let v = vec![0u8; 100];
        assert_eq!(adler32(&v), (100 << 16) | 1);
    }

    #[test]
    fn larger_input_crossing_nmax_chunk() {
        // 6000 bytes: > NMAX (5504), so we exercise the chunked modulo
        // path. Compare against a clean independent implementation
        // (scalar reference here).
        let mut v = vec![0u8; 6000];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (i * 37 + 17) as u8;
        }
        assert_eq!(adler32(&v), reference_scalar(&v));
    }

    #[test]
    fn quick_brown_fox() {
        let s = b"The quick brown fox jumps over the lazy dog";
        // Expected Adler-32 from RFC 1950 / standard tables: 0x5bdc0fda.
        assert_eq!(adler32(s), 0x5bdc_0fda);
        assert_eq!(adler32(s), reference_scalar(s));
    }

    /// One-pass scalar Adler-32 with per-byte modulo. Slow but obviously
    /// correct — used as a cross-check for the chunked implementation.
    fn reference_scalar(data: &[u8]) -> u32 {
        let mut sum1: u32 = 1;
        let mut sum2: u32 = 0;
        for &b in data {
            sum1 = (sum1 + b as u32) % BASE;
            sum2 = (sum2 + sum1) % BASE;
        }
        sum1 | (sum2 << 16)
    }
}
