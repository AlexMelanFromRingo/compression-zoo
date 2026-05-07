//! Mirror of the C `delta_xcheck` harness — applies `delta::encode`/`decode`
//! to stdin and writes the result to stdout. Used to compare bit-exact
//! against the original LZMA SDK implementation.

use std::io::{Read, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let op = args.next().expect("usage: delta_xcheck <e|d> <delta>");
    let delta: usize = args
        .next()
        .expect("usage: delta_xcheck <e|d> <delta>")
        .parse()
        .expect("delta must be an integer");

    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();

    let mut state = [0u8; sevenz::delta::STATE_SIZE];
    match op.as_str() {
        "e" => sevenz::delta::encode(&mut state, delta, &mut buf),
        "d" => sevenz::delta::decode(&mut state, delta, &mut buf),
        _ => {
            eprintln!("op must be e or d");
            std::process::exit(2);
        }
    }

    std::io::stdout().write_all(&buf).unwrap();
}
