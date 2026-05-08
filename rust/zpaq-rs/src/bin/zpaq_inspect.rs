//! Inspect a ZPAQ archive: dump block + segment headers, body byte
//! ranges, end markers. Doesn't decode the body yet.

use std::io::Read;
use zpaq_rs::format;
use zpaq_rs::io::SliceReader;

fn main() {
    let mut data = Vec::new();
    std::io::stdin().read_to_end(&mut data).unwrap();
    let mut r = SliceReader::new(&data);

    let mut block_idx = 0usize;
    while format::find_block_magic(&mut r).is_ok() {
        let pos_after_magic = r.position();
        let header = match format::read_header(&mut r) {
            Ok(h) => h,
            Err(e) => { eprintln!("header err: {:?}", e); return; }
        };
        println!("block {block_idx} at offset {} after magic:", pos_after_magic);
        println!("  hver={} mtype={} hsize={}",
                 header.hver, header.mtype, header.hsize);
        println!("  hh={} hm={} ph={} pm={} n={}",
                 header.hh, header.hm, header.ph, header.pm, header.n);
        println!("  comp_bytes={} bytes, hcomp={} bytes",
                 header.comp_bytes.len(), header.hcomp.len());

        let mut seg_idx = 0usize;
        loop {
            let seg = match format::read_segment_start(&mut r) {
                Ok(Some(s)) => s,
                Ok(None) => break,
                Err(e) => { eprintln!("segment_start err: {:?}", e); return; }
            };
            let body_start = r.position();
            print!("  segment {seg_idx}: filename=\"{}\" comment=\"{}\"",
                   String::from_utf8_lossy(&seg.filename),
                   String::from_utf8_lossy(&seg.comment));

            // Skip segment body — for store mode (n=0) we can read
            // the 4-byte BE length-prefixed chunks until 4 zero bytes.
            // For other methods, the body is arithmetic-coded — we'd
            // need the predictor + ZPAQL VM. We just bail out here.
            if header.n == 0 {
                let body_len = skip_stored_body(&mut r);
                let end = format::read_segment_end(&mut r).expect("seg end");
                println!(" body={} bytes end={:?}", body_len, end);
            } else {
                println!(" body=??? (modeled, n={} components — not supported yet)", header.n);
                return;
            }
            let _ = body_start;
            seg_idx += 1;
        }
        block_idx += 1;
    }
    println!("done; {} blocks parsed", block_idx);
}

/// Skip a stored-mode segment body (sequence of 4-byte BE length +
/// data, terminated by 4 zero bytes). Returns the number of bytes
/// consumed.
fn skip_stored_body(r: &mut SliceReader) -> usize {
    let start = r.position();
    use zpaq_rs::io::Reader;
    loop {
        let mut len = [0u8; 4];
        for slot in len.iter_mut() {
            *slot = match r.get() { Some(b) => b, None => return r.position() - start };
        }
        let n = u32::from_be_bytes(len) as usize;
        if n == 0 {
            // End of stored stream.
            return r.position() - start;
        }
        for _ in 0..n {
            if r.get().is_none() { return r.position() - start; }
        }
    }
}
