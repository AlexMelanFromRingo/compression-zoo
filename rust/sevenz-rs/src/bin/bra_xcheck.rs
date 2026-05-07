//! Mirror of the C `bra_xcheck` harness.

use std::io::{Read, Write};
use sevenz::bra::{self, Direction, X86_STATE_INIT};

fn main() {
    let mut args = std::env::args().skip(1);
    let arch = args.next().expect("usage: bra_xcheck <arch> <e|d> <pc>");
    let op = args.next().expect("usage: bra_xcheck <arch> <e|d> <pc>");
    let pc: u32 = {
        let s = args.next().expect("usage: bra_xcheck <arch> <e|d> <pc>");
        if let Some(stripped) = s.strip_prefix("0x") {
            u32::from_str_radix(stripped, 16).expect("bad pc")
        } else {
            s.parse().expect("bad pc")
        }
    };
    let dir = match op.as_str() {
        "e" => Direction::Encode,
        "d" => Direction::Decode,
        _ => {
            eprintln!("op must be e or d");
            std::process::exit(2);
        }
    };

    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();

    let processed = match arch.as_str() {
        "arm64" => bra::arm64(&mut buf, pc, dir),
        "arm" => bra::arm(&mut buf, pc, dir),
        "armt" => bra::armt(&mut buf, pc, dir),
        "ppc" => bra::ppc(&mut buf, pc, dir),
        "sparc" => bra::sparc(&mut buf, pc, dir),
        "ia64" => bra::ia64(&mut buf, pc, dir),
        "riscv" => bra::riscv(&mut buf, pc, dir),
        "x86" => {
            let mut s = X86_STATE_INIT;
            bra::x86_bcj(&mut buf, pc, &mut s, dir)
        }
        other => {
            eprintln!("bad arch {other}");
            std::process::exit(2);
        }
    };

    std::io::stdout().write_all(&buf).unwrap();
    eprintln!("{processed}");
}
