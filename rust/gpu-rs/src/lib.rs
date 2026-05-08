//! CUDA-accelerated kernels for compression-zoo. See
//! [`docs/gpu-acceleration.md`](../../../docs/gpu-acceleration.md)
//! for the design rationale and the full algorithm-by-algorithm
//! survey.
//!
//! The crate is intentionally narrow: it exposes safe Rust wrappers
//! over a small set of `extern "C"` CUDA entry points. When CUDA
//! isn't available at build time the C stub returns errors, the
//! probe [`available`] returns `false`, and callers should fall
//! back to the CPU reference (e.g.
//! [`bsc_rs::sais`](../bsc_rs/sais/index.html)'s `bucket_counts`).
//!
//! Currently shipped:
//!   * [`histogram_u8`] — 256-bin byte histogram.
//!
//! Planned (see design doc):
//!   * libcubwt FFI for big-block BWT.
//!   * CUDA E8E9 forward/inverse for ZPAQ preprocessing.

#![forbid(unsafe_op_in_unsafe_fn)]

extern "C" {
    fn gpu_rs_available() -> i32;
    fn gpu_rs_histogram_u8(
        data: *const u8,
        n: u64,
        out: *mut u32, // [256]
    ) -> i32;
}

/// `true` if a CUDA-capable device is present and the kernels were
/// compiled with `nvcc`. When this returns `false` callers should
/// take the CPU path.
pub fn available() -> bool {
    // SAFETY: gpu_rs_available is a thread-safe C function with no
    // arguments and a trivial return. The stub variant is also safe.
    unsafe { gpu_rs_available() != 0 }
}

/// Errors returned by GPU kernels.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum GpuError {
    /// CUDA wasn't available at runtime (no device, or build was a
    /// CPU stub).
    NotAvailable,
    /// CUDA returned an error code (negative cudaError_t).
    Cuda(i32),
}

/// 256-bin byte histogram on GPU. Returns `[count_of_byte_0, ...,
/// count_of_byte_255]`.
///
/// On hosts without CUDA returns [`GpuError::NotAvailable`]; callers
/// should fall back to a scalar `for b in data { hist[b] += 1; }`
/// (or whatever radix-aware version they already have).
pub fn histogram_u8(data: &[u8]) -> Result<[u32; 256], GpuError> {
    if !available() {
        return Err(GpuError::NotAvailable);
    }
    let mut out = [0u32; 256];
    // SAFETY: data and out point to live, sized buffers; the C
    // contract is documented in `cuda/histogram.cu` (read n bytes
    // from data, write 256 u32s to out).
    let rc = unsafe {
        gpu_rs_histogram_u8(
            data.as_ptr(),
            data.len() as u64,
            out.as_mut_ptr(),
        )
    };
    if rc == 0 {
        Ok(out)
    } else {
        Err(GpuError::Cuda(rc))
    }
}

/// CPU reference for `histogram_u8`, kept here so callers and tests
/// can compare without pulling in another crate.
pub fn histogram_u8_cpu(data: &[u8]) -> [u32; 256] {
    let mut h = [0u32; 256];
    for &b in data { h[b as usize] += 1; }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_matches_cpu_when_available() {
        if !available() {
            // CI / CPU-only host: just exercise the CPU reference.
            let cpu = histogram_u8_cpu(b"abcabcabc");
            assert_eq!(cpu[b'a' as usize], 3);
            assert_eq!(cpu[b'b' as usize], 3);
            assert_eq!(cpu[b'c' as usize], 3);
            return;
        }
        // Pseudo-random 1 MB: every result must match exactly.
        let mut data = vec![0u8; 1 << 20];
        let mut x: u32 = 0xC0FFEE;
        for b in data.iter_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (x >> 24) as u8;
        }
        let gpu = histogram_u8(&data).expect("gpu hist");
        let cpu = histogram_u8_cpu(&data);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn histogram_empty() {
        let cpu = histogram_u8_cpu(&[]);
        assert!(cpu.iter().all(|&c| c == 0));
        if available() {
            let gpu = histogram_u8(&[]).expect("gpu hist empty");
            assert_eq!(gpu, cpu);
        }
    }

    #[test]
    fn histogram_uniform_run() {
        let data = vec![0xAAu8; 1024];
        let cpu = histogram_u8_cpu(&data);
        assert_eq!(cpu[0xAA], 1024);
        if available() {
            let gpu = histogram_u8(&data).expect("gpu hist");
            assert_eq!(gpu[0xAA], 1024);
            for (i, &c) in gpu.iter().enumerate() {
                if i != 0xAA { assert_eq!(c, 0); }
            }
        }
    }
}
