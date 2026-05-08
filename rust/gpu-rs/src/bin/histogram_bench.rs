//! Bench `gpu_rs::histogram_u8` vs the CPU scalar reference across
//! a few input sizes. Skipped (with a stderr note) when CUDA is
//! unavailable.

use std::time::Instant;
use gpu_rs::{available, histogram_u8, histogram_u8_cpu};

fn fill_random(buf: &mut [u8], seed: u32) {
    let mut x = seed;
    for b in buf.iter_mut() {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (x >> 24) as u8;
    }
}

fn bench(n: usize) {
    let mut data = vec![0u8; n];
    fill_random(&mut data, 0xDEADBEEF);

    // Warm CPU.
    let _ = histogram_u8_cpu(&data);
    // Warm GPU (allocations + first-launch cost).
    if available() { let _ = histogram_u8(&data); }

    let trials = 5;
    let mut cpu_us = u128::MAX;
    for _ in 0..trials {
        let t = Instant::now();
        let _ = histogram_u8_cpu(&data);
        let us = t.elapsed().as_micros();
        if us < cpu_us { cpu_us = us; }
    }

    let mut gpu_us = u128::MAX;
    if available() {
        for _ in 0..trials {
            let t = Instant::now();
            let _ = histogram_u8(&data).expect("gpu");
            let us = t.elapsed().as_micros();
            if us < gpu_us { gpu_us = us; }
        }
    }

    let mb = (n as f64) / (1024.0 * 1024.0);
    if available() {
        let speedup = cpu_us as f64 / gpu_us.max(1) as f64;
        println!("size={:>5.1} MiB  CPU={:>8} µs  GPU={:>8} µs  speedup={:.2}×",
            mb, cpu_us, gpu_us, speedup);
    } else {
        println!("size={:>5.1} MiB  CPU={:>8} µs  GPU=unavailable",
            mb, cpu_us);
    }
}

fn main() {
    eprintln!("CUDA available: {}", available());
    for &n in &[1usize<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26] {
        bench(n);
    }
}
