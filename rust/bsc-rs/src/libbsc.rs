//! Top-level libbsc decompress driver. Consumes a libbsc block (header
//! + body) on the input side and produces the original bytes.
//!
//! Mirrors `bsc_decompress` in
//! `plugins/bsc/upstream/libbsc/libbsc/libbsc.cpp`. Supports the common
//! case used by the bsc 7-Zip plugin:
//!   * coder = QLFC_STATIC (the only QLFC variant ported so far).
//!   * blockSorter = BWT (Schindler transforms not yet ported).
//!   * single-threaded encoder output (`num_indexes` == 0).
//!   * Optional LZP preprocessing.
//!
//! Returns `DataCorrupt` for unsupported sorter/coder combos so it
//! refuses adaptive/fast/ST archives cleanly rather than producing
//! garbage.

#![allow(dead_code)]

use crate::adler32::adler32;
use crate::bwt;
use crate::format::{
    self, block_info, BlockInfo, LIBBSC_BLOCKSORTER_BWT, LIBBSC_BLOCKSORTER_NONE,
    LIBBSC_BLOCKSORTER_ST3, LIBBSC_BLOCKSORTER_ST4, LIBBSC_BLOCKSORTER_ST5,
    LIBBSC_BLOCKSORTER_ST6, LIBBSC_BLOCKSORTER_ST7, LIBBSC_BLOCKSORTER_ST8,
    LIBBSC_CODER_NONE, LIBBSC_CODER_QLFC_ADAPTIVE, LIBBSC_CODER_QLFC_FAST,
    LIBBSC_CODER_QLFC_STATIC, LIBBSC_HEADER_SIZE,
};
use crate::lzp;
use crate::qlfc;

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum DecompressError {
    UnexpectedEob,
    DataCorrupt,
    /// The input block uses a coder we haven't ported yet
    /// (QLFC_ADAPTIVE, QLFC_FAST).
    UnsupportedCoder(i32),
    /// The input block uses a Schindler-transform sorter we haven't
    /// ported yet (ST3..ST8).
    UnsupportedSorter(i32),
    /// Reserved (was used briefly when multi-index BWT was thought
    /// to need a different code path; we now just ignore the
    /// `num_indexes` hint and run the standard sequential inverse).
    #[deprecated]
    UnsupportedMultiIndex,
}

/// Decompress a libbsc-format block into `output`. Returns the number
/// of original bytes written.
pub fn decompress(input: &[u8], output: &mut Vec<u8>) -> Result<usize, DecompressError> {
    if input.len() < LIBBSC_HEADER_SIZE {
        return Err(DecompressError::UnexpectedEob);
    }

    let info: BlockInfo = block_info(input).map_err(|e| match e {
        format::BlockInfoError::UnexpectedEob => DecompressError::UnexpectedEob,
        format::BlockInfoError::DataCorrupt => DecompressError::DataCorrupt,
    })?;

    let block_size = info.block_size as usize;
    if block_size > input.len() {
        return Err(DecompressError::UnexpectedEob);
    }

    // Verify body adler (header bytes 20..24 = adler of bytes 28..blockSize).
    let recorded_body_adler = u32::from_le_bytes(input[20..24].try_into().unwrap());
    let body_adler = adler32(&input[LIBBSC_HEADER_SIZE..block_size]);
    if recorded_body_adler != body_adler {
        return Err(DecompressError::DataCorrupt);
    }

    // mode = 0 → store as-is.
    let mode = u32::from_le_bytes(input[8..12].try_into().unwrap());
    let data_size = info.data_size as usize;

    if mode == 0 {
        output.clear();
        output.extend_from_slice(&input[LIBBSC_HEADER_SIZE..LIBBSC_HEADER_SIZE + data_size]);
        return Ok(data_size);
    }

    if info.coder == LIBBSC_CODER_NONE && info.block_sorter == LIBBSC_BLOCKSORTER_NONE {
        // No sorter, no coder → identical to store mode but with the
        // header still present (libbsc's bsc_compress only emits this
        // when input is uncompressible).
        output.clear();
        output.extend_from_slice(&input[LIBBSC_HEADER_SIZE..LIBBSC_HEADER_SIZE + data_size]);
        return Ok(data_size);
    }

    if info.coder != LIBBSC_CODER_QLFC_STATIC
        && info.coder != LIBBSC_CODER_QLFC_ADAPTIVE
        && info.coder != LIBBSC_CODER_QLFC_FAST
    {
        return Err(DecompressError::UnsupportedCoder(info.coder));
    }
    let st_k: Option<i32> = match info.block_sorter {
        LIBBSC_BLOCKSORTER_BWT => None,
        LIBBSC_BLOCKSORTER_ST3 => Some(3),
        LIBBSC_BLOCKSORTER_ST4 => Some(4),
        LIBBSC_BLOCKSORTER_ST5 => Some(5),
        LIBBSC_BLOCKSORTER_ST6 => Some(6),
        LIBBSC_BLOCKSORTER_ST7 => Some(7),
        LIBBSC_BLOCKSORTER_ST8 => Some(8),
        _ => return Err(DecompressError::UnsupportedSorter(info.block_sorter)),
    };

    let recorded_data_adler = u32::from_le_bytes(input[16..20].try_into().unwrap());

    // libbsc records `num_indexes` plus a small index table at the end
    // of the body to enable parallel BWT inverse (libsais's
    // `libsais_unbwt_aux`). The bytes are not consumed by the QLFC
    // coder (which knows its own per-block sizes from the leading
    // table), so we just tolerate them and run the standard sequential
    // BWT inverse.
    let _num_indexes = input[block_size - 1] as usize;

    // Run the coder over the body (header excluded). Coder dispatches
    // on a leading nBlocks byte.
    let coder_input = &input[LIBBSC_HEADER_SIZE..block_size];
    let mut sorted = vec![0u8; data_size]; // upper bound: lzSize <= data_size
    let lz_size = decompress_coder_blocks(coder_input, &mut sorted, info.coder)?;

    // BWT or ST inverse.
    let bwt_buf = &mut sorted[..lz_size];
    if let Some(k) = st_k {
        crate::st::unst(bwt_buf, info.index, k)
            .map_err(|_| DecompressError::DataCorrupt)?;
    } else {
        bwt::unbwt(bwt_buf, info.index).map_err(|_| DecompressError::DataCorrupt)?;
    }

    // LZP inverse (if enabled).
    let lzp_enabled = info.lzp_min_len != 0 || info.lzp_hash_size != 0;
    if lzp_enabled {
        let mut decoded: Vec<u8> = Vec::with_capacity(data_size);
        // `bwt_buf` is the LZP-encoded block. Strip the leading
        // 1-byte nBlocks and decode each sub-block.
        if bwt_buf.is_empty() {
            return Err(DecompressError::DataCorrupt);
        }
        lzp::decompress(bwt_buf, &mut decoded, info.lzp_hash_size, info.lzp_min_len)
            .map_err(|_| DecompressError::DataCorrupt)?;
        if decoded.len() != data_size {
            return Err(DecompressError::DataCorrupt);
        }
        if adler32(&decoded) != recorded_data_adler {
            return Err(DecompressError::DataCorrupt);
        }
        output.clear();
        output.extend_from_slice(&decoded);
    } else {
        if lz_size != data_size {
            return Err(DecompressError::DataCorrupt);
        }
        if adler32(bwt_buf) != recorded_data_adler {
            return Err(DecompressError::DataCorrupt);
        }
        output.clear();
        output.extend_from_slice(bwt_buf);
    }

    Ok(data_size)
}

/// Replicates `bsc_coder_decompress` from upstream `coder.cpp`.
/// Splits the input into `nBlocks` chunks, decodes each (or copies
/// through if `inputSize == outputSize`), and concatenates the
/// results into `output`. Returns the total decoded byte count.
fn decompress_coder_blocks(
    input: &[u8],
    output: &mut [u8],
    coder: i32,
) -> Result<usize, DecompressError> {
    if input.is_empty() {
        return Err(DecompressError::UnexpectedEob);
    }
    let n_blocks = input[0] as usize;
    if n_blocks == 0 {
        return Err(DecompressError::DataCorrupt);
    }
    if n_blocks == 1 {
        return decompress_one_block(&input[1..], output, coder);
    }

    // Per-block size table starts at offset 1; 8 bytes per entry.
    let header_len = 1 + 8 * n_blocks;
    if input.len() < header_len {
        return Err(DecompressError::UnexpectedEob);
    }

    let mut input_ptr = header_len;
    let mut output_ptr = 0;
    let mut total: usize = 0;
    for block_id in 0..n_blocks {
        let off = 1 + 8 * block_id;
        let output_size = i32::from_le_bytes(input[off..off + 4].try_into().unwrap()) as i64;
        let input_size = i32::from_le_bytes(input[off + 4..off + 8].try_into().unwrap()) as i64;
        if input_size < 0 || output_size < 0 {
            return Err(DecompressError::DataCorrupt);
        }
        let input_size = input_size as usize;
        let output_size = output_size as usize;
        if input_ptr + input_size > input.len() || output_ptr + output_size > output.len() {
            return Err(DecompressError::UnexpectedEob);
        }
        if input_size == output_size {
            output[output_ptr..output_ptr + output_size]
                .copy_from_slice(&input[input_ptr..input_ptr + input_size]);
            total += output_size;
        } else {
            let written = decompress_one_block(
                &input[input_ptr..input_ptr + input_size],
                &mut output[output_ptr..output_ptr + output_size],
                coder,
            )?;
            if written != output_size {
                return Err(DecompressError::DataCorrupt);
            }
            total += written;
        }
        input_ptr += input_size;
        output_ptr += output_size;
    }
    Ok(total)
}

fn decompress_one_block(
    input: &[u8],
    output: &mut [u8],
    coder: i32,
) -> Result<usize, DecompressError> {
    if coder == LIBBSC_CODER_QLFC_STATIC {
        qlfc::static_decode(input, output).map_err(|_| DecompressError::DataCorrupt)
    } else if coder == LIBBSC_CODER_QLFC_ADAPTIVE {
        qlfc::adaptive_decode(input, output).map_err(|_| DecompressError::DataCorrupt)
    } else if coder == LIBBSC_CODER_QLFC_FAST {
        qlfc::fast_decode(input, output).map_err(|_| DecompressError::DataCorrupt)
    } else {
        Err(DecompressError::UnsupportedCoder(coder))
    }
}

// ===================================================================
// Forward path: bsc_compress
// ===================================================================

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum CompressError {
    /// `n > 1 GiB` — libbsc's hard cap.
    InputTooLarge,
    /// Bad LZP parameters (hash_size 10..=28, min_len 4..=255).
    BadParameter,
}

/// Compress `input` into `out` using BWT + QLFC + optional LZP.
/// Returns the number of bytes appended to `out` (i.e. the size of
/// the libbsc block).
///
/// `coder` selects the QLFC variant (`LIBBSC_CODER_QLFC_STATIC` or
/// `LIBBSC_CODER_QLFC_ADAPTIVE`). Pass `lzp_hash_size = 0` and
/// `lzp_min_len = 0` to disable LZP.
///
/// Match libbsc level 5 with `(QLFC_STATIC, 15, 72)`; level 7 with
/// `(QLFC_STATIC, 16, 96)`; level 9 with `(QLFC_ADAPTIVE, 16, 128)`.
///
/// If the input is incompressible (final block would be larger than
/// the original) we fall back to `bsc_store` (mode = 0).
pub fn compress(
    input: &[u8],
    out: &mut Vec<u8>,
    lzp_hash_size: i32,
    lzp_min_len: i32,
) -> Result<usize, CompressError> {
    compress_with_coder(input, out, lzp_hash_size, lzp_min_len, LIBBSC_CODER_QLFC_STATIC)
}

/// Same as [`compress`] but parameterised on the QLFC variant.
pub fn compress_with_coder(
    input: &[u8],
    out: &mut Vec<u8>,
    lzp_hash_size: i32,
    lzp_min_len: i32,
    coder: i32,
) -> Result<usize, CompressError> {
    let n = input.len();
    if n > 1_073_741_824 {
        return Err(CompressError::InputTooLarge);
    }
    if n <= LIBBSC_HEADER_SIZE {
        return Ok(store(input, out));
    }
    if coder != LIBBSC_CODER_QLFC_STATIC
        && coder != LIBBSC_CODER_QLFC_ADAPTIVE
        && coder != LIBBSC_CODER_QLFC_FAST
    {
        return Err(CompressError::BadParameter);
    }

    let mut mode: i32 = LIBBSC_BLOCKSORTER_BWT + (coder << 5);

    let lzp_enabled = lzp_min_len != 0 || lzp_hash_size != 0;
    if lzp_enabled {
        if lzp_min_len < 4 || lzp_min_len > 255 {
            return Err(CompressError::BadParameter);
        }
        if lzp_hash_size < 10 || lzp_hash_size > 28 {
            return Err(CompressError::BadParameter);
        }
        mode += lzp_min_len << 8;
        mode += lzp_hash_size << 16;
    }

    // ----- Stage 1: LZP forward (optional) -----------------------
    let mut stage1: Vec<u8> = Vec::with_capacity(n + 1024);
    if lzp_enabled {
        let lz_result = crate::lzp::compress(input, &mut stage1, lzp_hash_size, lzp_min_len);
        if lz_result.is_err() {
            // libbsc treats LZP failure as "fall back to no-LZP".
            stage1.clear();
            stage1.extend_from_slice(input);
            mode &= 0xff;
        }
    } else {
        stage1.extend_from_slice(input);
    }

    let lz_size = stage1.len();
    if lz_size <= LIBBSC_HEADER_SIZE {
        // libbsc forces blockSorter = BWT in this case.
        mode = (mode & 0xffff_ffe0u32 as i32) | LIBBSC_BLOCKSORTER_BWT;
    }

    // ----- Stage 2: BWT forward ----------------------------------
    let (bwt_out, primary) = crate::bwt::encode(&stage1);

    // ----- Stage 3: QLFC encode ---------------------------------
    let mut qlfc_wire: Vec<u8> = Vec::with_capacity(lz_size + 4096);
    let qlfc_cap = lz_size + 4096;
    // Coder framing: leading 1-byte nBlocks header (= 1 for our
    // single-block path).
    qlfc_wire.push(1u8);
    let qlfc_body_start = qlfc_wire.len();
    let qlfc_result = match coder {
        LIBBSC_CODER_QLFC_STATIC =>
            qlfc::static_encode(&bwt_out, &mut qlfc_wire, qlfc_cap),
        LIBBSC_CODER_QLFC_ADAPTIVE =>
            qlfc::adaptive_encode(&bwt_out, &mut qlfc_wire, qlfc_cap),
        LIBBSC_CODER_QLFC_FAST =>
            qlfc::fast_encode(&bwt_out, &mut qlfc_wire, qlfc_cap),
        _ => return Err(CompressError::BadParameter),
    };
    if qlfc_result.is_err() {
        return Ok(store(input, out));
    }

    // Coder body length excludes the leading `1` we wrote.
    let _coder_body_len = qlfc_wire.len() - qlfc_body_start;

    // ----- Stage 4: assemble block -------------------------------
    // num_indexes = 0 (no parallel BWT acceleration on encode).
    let body_size = qlfc_wire.len() + 1; // +1 = num_indexes trailer.
    let block_size = LIBBSC_HEADER_SIZE + body_size;
    if block_size >= n + LIBBSC_HEADER_SIZE {
        // Output would be larger than store mode — fall back.
        return Ok(store(input, out));
    }

    let start = out.len();
    out.resize(start + LIBBSC_HEADER_SIZE, 0);
    out.extend_from_slice(&qlfc_wire);
    out.push(0u8); // num_indexes

    // Adler-32 fields.
    let body_adler = adler32(&out[start + LIBBSC_HEADER_SIZE..start + block_size]);
    let data_adler = adler32(input);

    out[start + 0..start + 4].copy_from_slice(&(block_size as i32).to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(n as i32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&mode.to_le_bytes());
    out[start + 12..start + 16].copy_from_slice(&primary.to_le_bytes());
    out[start + 16..start + 20].copy_from_slice(&data_adler.to_le_bytes());
    out[start + 20..start + 24].copy_from_slice(&body_adler.to_le_bytes());
    let header_adler = adler32(&out[start..start + 24]);
    out[start + 24..start + 28].copy_from_slice(&header_adler.to_le_bytes());

    Ok(block_size)
}

/// `bsc_store` — mode-0 block with the input verbatim.
fn store(input: &[u8], out: &mut Vec<u8>) -> usize {
    let n = input.len();
    let block_size = LIBBSC_HEADER_SIZE + n;
    let start = out.len();
    out.resize(start + LIBBSC_HEADER_SIZE, 0);
    out.extend_from_slice(input);
    let data_adler = adler32(input);
    let body_adler = data_adler;
    out[start + 0..start + 4].copy_from_slice(&(block_size as i32).to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(n as i32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&0i32.to_le_bytes());
    out[start + 12..start + 16].copy_from_slice(&0i32.to_le_bytes());
    out[start + 16..start + 20].copy_from_slice(&data_adler.to_le_bytes());
    out[start + 20..start + 24].copy_from_slice(&body_adler.to_le_bytes());
    let header_adler = adler32(&out[start..start + 24]);
    out[start + 24..start + 28].copy_from_slice(&header_adler.to_le_bytes());
    block_size
}
