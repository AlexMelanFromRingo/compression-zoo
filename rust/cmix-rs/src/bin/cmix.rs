//! `cmix-rs` CLI — minimal port of upstream's `runner.cpp`.
//!
//! Usage:
//!
//! ```text
//! cmix-rs -c <input> <output>     # compress
//! cmix-rs -d <input> <output>     # decompress
//! ```
//!
//! The full upstream `cmix` accepts `-t` (text-mode), `-n` (no
//! preprocessing), `-s` (preprocessing only), and a dictionary
//! argument. Those flags require a fully functional file-type
//! detector and the upstream Predictor orchestrator; not all of that
//! pipeline is ported yet, so this CLI is intentionally minimal.
//! Compression ratio is well below upstream cmix until the full
//! Predictor (paq8 + LSTM + multi-layer mixer tree) lands.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::process::ExitCode;

use cmix_rs::runner::{decode, encode};

fn print_usage(prog: &str) {
    eprintln!("cmix-rs — work-in-progress Rust port of CMIX\n");
    eprintln!("Compress:   {prog} -c <input> <output>");
    eprintln!("Decompress: {prog} -d <input> <output>");
    eprintln!();
    eprintln!(
        "Note: upstream-only flags (-t, -n, -s, dictionary path) \
         not yet supported by this port.",
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let prog = args.get(0).map(|s| s.as_str()).unwrap_or("cmix-rs");
    if args.len() != 4 {
        print_usage(prog);
        return ExitCode::from(1);
    }
    let mode = &args[1];
    let in_path  = Path::new(&args[2]);
    let out_path = Path::new(&args[3]);

    let input = match File::open(in_path) {
        Ok(f) => BufReader::new(f),
        Err(e) => {
            eprintln!("failed to open input {}: {}", in_path.display(), e);
            return ExitCode::from(2);
        }
    };
    let output = match File::create(out_path) {
        Ok(f) => BufWriter::new(f),
        Err(e) => {
            eprintln!("failed to create output {}: {}", out_path.display(), e);
            return ExitCode::from(2);
        }
    };

    let result = match mode.as_str() {
        "-c" => encode(input, output).map(|n| {
            eprintln!("encoded {} bytes", n);
        }),
        "-d" => decode(input, output).map(|n| {
            eprintln!("decoded {} bytes", n);
        }),
        other => {
            eprintln!("unknown mode: {}", other);
            print_usage(prog);
            return ExitCode::from(1);
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("I/O error: {}", e);
            ExitCode::from(3)
        }
    }
}
