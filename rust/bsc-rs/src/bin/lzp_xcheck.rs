use std::io::Read;
use std::io::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().expect("mode e|d");
    let hash: i32 = args.next().expect("hash").parse().unwrap();
    let min: i32 = args.next().expect("min").parse().unwrap();
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let mut out = Vec::with_capacity(buf.len() * 2 + 64);
    // `e`/`d`: wire-format-compatible (with nBlocks header).
    // `eb`/`db`: raw block (no header). Used by the cross-test that
    // also strips libbsc's nBlocks byte.
    match mode.as_str() {
        "e"  => { bsc_rs::lzp::compress(&buf, &mut out, hash, min).expect("compress"); }
        "d"  => { bsc_rs::lzp::decompress(&buf, &mut out, hash, min).expect("decompress"); }
        "eb" => { bsc_rs::lzp::encode_block(&buf, &mut out, hash, min).expect("encode_block"); }
        "db" => { bsc_rs::lzp::decode_block(&buf, &mut out, hash, min).expect("decode_block"); }
        _    => { eprintln!("mode must be e|d|eb|db"); std::process::exit(2); }
    }
    std::io::stdout().write_all(&out).unwrap();
}
