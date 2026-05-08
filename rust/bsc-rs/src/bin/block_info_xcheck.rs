//! Read bytes from stdin (a libbsc-encoded block, header + body) and
//! print the parsed BlockInfo plus a one-line summary. Used to verify
//! that our Rust port of `bsc_block_info` agrees with the C reference
//! on real wire-format output.

use std::io::Read;

fn main() -> std::io::Result<()> {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf)?;

    if buf.len() < bsc_rs::format::LIBBSC_HEADER_SIZE {
        eprintln!("input too short ({} bytes)", buf.len());
        std::process::exit(1);
    }

    match bsc_rs::format::block_info(&buf[..bsc_rs::format::LIBBSC_HEADER_SIZE]) {
        Ok(info) => {
            println!(
                "block_size={}  data_size={}  index={}  sorter={}  coder={}  lzp_min_len={}  lzp_hash={}",
                info.block_size, info.data_size, info.index,
                info.block_sorter, info.coder, info.lzp_min_len, info.lzp_hash_size
            );
            // Sanity: total file size should equal block_size.
            if (info.block_size as usize) != buf.len() {
                eprintln!(
                    "WARNING: file is {} bytes but header reports block_size={}",
                    buf.len(), info.block_size
                );
            }
        }
        Err(e) => {
            eprintln!("block_info error: {:?}", e);
            std::process::exit(1);
        }
    }
    Ok(())
}
