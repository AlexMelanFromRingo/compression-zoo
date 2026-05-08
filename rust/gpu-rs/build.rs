//! Compile `cuda/*.cu` via nvcc into a static lib that the Rust
//! crate links.
//!
//! If nvcc is missing or `GPU_RS_FORCE_CPU=1` is set in the
//! environment, we skip the CUDA build and the crate's runtime
//! probe (`gpu_rs::available()`) returns `false`. The caller is
//! expected to fall back to the CPU reference path.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=cuda/histogram.cu");
    println!("cargo:rerun-if-env-changed=GPU_RS_FORCE_CPU");

    if env::var_os("GPU_RS_FORCE_CPU").is_some() {
        // Compile a tiny C stub providing the same symbols so the
        // Rust crate links cleanly. The stub returns `false` from
        // `gpu_rs_available` and is otherwise a no-op.
        emit_stub();
        return;
    }

    let nvcc = which("nvcc");
    if nvcc.is_none() {
        emit_stub();
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cu = "cuda/histogram.cu";
    let obj = out_dir.join("histogram.o");

    let nvcc = nvcc.unwrap();
    let arch_flags = ["-arch=sm_75"]; // RTX 20-series and newer; safe minimum

    let status = Command::new(&nvcc)
        .args(arch_flags)
        .arg("-O3")
        .arg("-c")
        .arg("--compiler-options=-fPIC")
        .arg("-o").arg(&obj)
        .arg(cu)
        .status()
        .expect("nvcc spawn failed");
    if !status.success() {
        // Don't fail the build — fall back to the stub. The crate's
        // tests skip when CUDA is unavailable.
        eprintln!("warning: nvcc failed; falling back to CPU stub");
        emit_stub();
        return;
    }

    // Bundle into a static archive so cargo's normal link path works.
    let lib = out_dir.join("libgpu_rs_cuda.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crus").arg(&lib).arg(&obj)
        .status().expect("ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_rs_cuda");
    // CUDA runtime.
    if let Some(cuda_lib) = guess_cuda_libdir() {
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=gpu_rs_cuda_built");
}

/// Locate a binary on `PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() { return Some(p); }
    }
    None
}

/// Try `/usr/local/cuda*/lib64` and friends. If nothing matches we
/// trust the system linker to find `libcudart.so` (it usually does
/// when nvcc is installed via the platform package manager).
fn guess_cuda_libdir() -> Option<PathBuf> {
    let candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-13/lib64",
        "/opt/cuda/lib64",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.is_dir() { return Some(p); }
    }
    None
}

/// Emit a tiny C stub returning `false` from `gpu_rs_available`.
fn emit_stub() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src = out_dir.join("stub.c");
    std::fs::write(&src, r#"
#include <stdint.h>
int gpu_rs_available(void) { return 0; }
int gpu_rs_histogram_u8(const uint8_t *data, uint64_t n, uint32_t out[256]) {
    (void)data; (void)n; (void)out; return -1;
}
"#).unwrap();
    let obj = out_dir.join("stub.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .arg("-O2").arg("-fPIC").arg("-c")
        .arg("-o").arg(&obj)
        .arg(&src)
        .status().expect("cc");
    assert!(status.success(), "stub cc failed");
    let lib = out_dir.join("libgpu_rs_cuda.a");
    let _ = std::fs::remove_file(&lib);
    let status = Command::new("ar")
        .arg("crus").arg(&lib).arg(&obj)
        .status().expect("ar");
    assert!(status.success(), "stub ar failed");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_rs_cuda");
}
