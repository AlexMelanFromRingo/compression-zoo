//! Branch converters (a.k.a. "BCJ" filters in LZMA SDK terminology).
//! Port of `7zip/C/Bra.c` and `7zip/C/Bra86.c`.
//!
//! Each converter rewrites relative call/branch offsets as absolute addresses
//! so a downstream entropy coder sees more redundancy.  Shapes:
//!
//! * Stateless (RISC) converters: `arm64`, `arm`, `armt`, `ppc`, `sparc`,
//!   `ia64`, `riscv`. `convert(data, pc, encoding) -> usize` returns the
//!   number of bytes processed in `data`. Input must be aligned to the
//!   architecture's instruction boundary; the function pads `size` down.
//! * Stateful X86 converter: holds a `u32` mask to disambiguate trailing
//!   bytes of `0xE8`/`0xE9` near a buffer boundary.
//!
//! The C "BR_PC_INIT/BR_PC_GET" trick is replaced with a direct
//! `pc + offset_in_data` calculation.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    Encode,
    Decode,
}

#[inline(always)]
fn convert_val(v: u32, c: u32, dir: Direction) -> u32 {
    match dir {
        Direction::Encode => v.wrapping_add(c),
        Direction::Decode => v.wrapping_sub(c),
    }
}

#[inline(always)]
fn read_u32_le(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(data[i..i + 4].try_into().unwrap())
}

#[inline(always)]
fn write_u32_le(data: &mut [u8], i: usize, v: u32) {
    data[i..i + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
fn read_u32_be(data: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(data[i..i + 4].try_into().unwrap())
}

#[inline(always)]
fn write_u32_be(data: &mut [u8], i: usize, v: u32) {
    data[i..i + 4].copy_from_slice(&v.to_be_bytes());
}

#[inline(always)]
fn read_u16_le(data: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(data[i..i + 2].try_into().unwrap())
}

#[inline(always)]
fn write_u16_le(data: &mut [u8], i: usize, v: u16) {
    data[i..i + 2].copy_from_slice(&v.to_le_bytes());
}

// ======================================================================== //
// ARM64
// ======================================================================== //

pub fn arm64(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !3;
    const FLAG: u32 = 1 << (24 - 4); // 0x00100000
    const MASK: u32 = (1 << 24) - (FLAG << 1); // 0x00e00000

    let mut i = 0;
    while i < size {
        let v = read_u32_le(data, i);
        // BL: opcode bits 31:26 == 100101 -> v - 0x94000000 fits in low 26 bits.
        if (v.wrapping_sub(0x9400_0000) & 0xfc00_0000) == 0 {
            let c = pc.wrapping_add(i as u32) >> 2;
            let nv = convert_val(v, c, dir);
            write_u32_le(data, i, (nv & 0x03ff_ffff) | 0x9400_0000);
            i += 4;
            continue;
        }
        // ADRP
        let mut v2 = v.wrapping_sub(0x9000_0000);
        if (v2 & 0x9f00_0000) == 0 {
            v2 = v2.wrapping_add(FLAG);
            if v2 & MASK != 0 {
                i += 4;
                continue;
            }
            let z = (v2 & 0xffff_ffe0) | (v2 >> 26);
            let c = (pc.wrapping_add(i as u32) >> (12 - 3)) & !7u32;
            let z = convert_val(z, c, dir);
            let mut nv = v2 & 0x1f;
            nv |= 0x9000_0000;
            nv |= z << 26;
            nv |= 0x00ff_ffe0 & (z & ((FLAG << 1) - 1)).wrapping_sub(FLAG);
            write_u32_le(data, i, nv);
        }
        i += 4;
    }
    i
}

// ======================================================================== //
// ARM (32-bit, classic)
// ======================================================================== //

pub fn arm(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !3;
    let mut i = 0;
    // pc adjustment: the instruction at offset i has address `pc + i`, but ARM
    // PC is +8 ahead (pipeline) and the C code writes back at `p - 4`.
    // `BR_PC_GET` after `p += 4` and `pc += 8 - 4 = +4` yields `pc + i + 4`,
    // i.e. the next instruction's address — which equals `pc + (i+4)`.
    // We compute `c = (pc + i + 4) >> 2`.
    while i < size {
        if data[i + 3] == 0xeb {
            let v = read_u32_le(data, i);
            let c = pc.wrapping_add(i as u32).wrapping_add(8) >> 2;
            // C writes back at `p-4` after the BR_PC_GET; with our offsets the
            // address used in BR_PC_GET is `pc + i + 4` = next instr.
            // Reconcile: the C uses `pc + 8 - 4` then `BR_PC_GET = pc + p - 4`
            // after `p += 4`, giving `pc + (i+4) - 4 + 8 - 4 = pc + i + 4`.
            let nv = convert_val(v, c, dir);
            write_u32_le(data, i, (nv & 0x00ff_ffff) | 0xeb00_0000);
        }
        i += 4;
    }
    i
}

// ======================================================================== //
// ARM Thumb (ARMT)  — 16-bit instructions, 2-byte alignment
// ======================================================================== //

pub fn armt(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !1;
    if size <= 2 {
        return 0;
    }
    let limit = size - 2;
    let mut i = 0;
    while i < limit {
        // Match BL pattern: high half-word top 5 bits = 11110, low half-word
        // top 5 bits = 11111.  The C check is on bytes [1] and [3]:
        //   (b3 & (b1 ^ 8)) >= 0xf8     (with b1 = data[i+1], b3 = data[i+3])
        let b1 = data[i + 1] as u32;
        let b3 = data[i + 3] as u32;
        if (b3 & (b1 ^ 8)) >= 0xf8 {
            let hi = read_u16_le(data, i) as u32;
            let lo = read_u16_le(data, i + 2) as u32;
            let v = (hi << 11) | (lo & 0x7ff);
            let c = pc.wrapping_add(i as u32).wrapping_add(4) >> 1;
            let v = convert_val(v, c, dir);
            write_u16_le(data, i, (((v >> 11) & 0x7ff) | 0xf000) as u16);
            write_u16_le(data, i + 2, ((v & 0x7ff) | 0xf800) as u16);
            i += 4;
        } else {
            i += 2;
        }
    }
    i
}

// ======================================================================== //
// PowerPC
// ======================================================================== //

pub fn ppc(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !3;
    let mut i = 0;
    while i < size {
        let v = read_u32_be(data, i);
        // BL: top 6 bits = 010010, bottom bits ...01 (LK set).  The mask
        // `0xfc000003` checks both ends.
        if (v.wrapping_sub(0x4800_0001) & 0xfc00_0003) == 0 {
            let c = pc.wrapping_add(i as u32);
            let v = convert_val(v, c, dir);
            write_u32_be(data, i, (v & 0x03ff_ffff) | 0x4800_0000);
        }
        i += 4;
    }
    i
}

// ======================================================================== //
// SPARC
// ======================================================================== //

pub fn sparc(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !3;
    const FLAG: u32 = 1 << 22;
    let mut i = 0;
    while i < size {
        let v_orig = read_u32_be(data, i);
        // Mirrors the `BR_SPARC_USE_ROTATE` branch (used on every CPU with
        // `Z7_CPU_FAST_ROTATE_SUPPORTED`, which is x86/x64/ARM/PPC).  The
        // non-rotate branch is *not* equivalent for non-word-aligned PC
        // values, and the binary 7-Zip ships compiles to the rotate path.
        let v_rotl = v_orig.rotate_left(2);
        let v_check = v_rotl.wrapping_add((FLAG << 2) - 1);
        let mask: u32 = 3u32.wrapping_sub(FLAG << 3); // 0xfe000003
        if v_check & mask == 0 {
            let c = pc.wrapping_add(i as u32);
            let mut x = convert_val(v_check, c, dir);
            x &= (FLAG << 3) - 1;
            x = x.wrapping_sub((FLAG << 2) - 1);
            x = x.rotate_right(2);
            write_u32_be(data, i, x);
        }
        i += 4;
    }
    i
}

// ======================================================================== //
// IA-64 (Itanium)
// ======================================================================== //

pub fn ia64(data: &mut [u8], mut pc: u32, dir: Direction) -> usize {
    let size = data.len() & !15;
    pc = pc.wrapping_sub(1u32 << 4);
    pc >>= 4 - 1;

    let mut i = 0;
    while i < size {
        let m_val = (0x334b_0000u32 >> (data[i] & 0x1e)) as u32;
        i += 16;
        pc = pc.wrapping_add(1u32 << 1);
        let mut m = m_val & 3;
        if m == 0 {
            continue;
        }

        // We backtrack from the start of the next bundle by `5*m - 20`
        // (negative, walks into the current bundle).
        let mut j = (i as isize) + 5 * m as isize - 20;

        loop {
            let pj = j as usize;
            // C reads 32 bits starting at p+0 (subset used: low byte).
            let t = u16::from_le_bytes([data[pj], data[pj + 1]]) as u32;
            let z = u32::from_le_bytes(data[pj + 1..pj + 5].try_into().unwrap()) >> m;
            j += 5;

            if ((t >> m) & (0x70 << 1)) == 0
                && ((z.wrapping_sub(0x500_0000 << 1)) & (0xf00_0000 << 1)) == 0
            {
                let mut v = ((0x8f_ffff << 1) | 1) & z;
                let mut zr = z ^ v;

                let pc_low = pc & ((0x1f_ffff << 1) | 1);
                let pc_high = pc | !((0x1f_ffff << 1) | 1);
                v = match dir {
                    Direction::Encode => v.wrapping_add(pc_low),
                    Direction::Decode => v.wrapping_sub(pc_high),
                };
                v &= !(0x60_0000u32 << 1);
                v = v.wrapping_add(0x70_0000 << 1);
                v &= (0x8f_ffff << 1) | 1;
                zr |= v;
                zr <<= m;
                let dst = (j - 5) as usize + 1;
                data[dst..dst + 4].copy_from_slice(&zr.to_le_bytes());
            }
            m += 1;
            m &= 3;
            if m == 0 {
                break;
            }
        }
    }
    i
}

// ======================================================================== //
// RISC-V
// ======================================================================== //

const RISCV_INSTR_SIZE: usize = 2;
const RISCV_REG_VAL: u32 = 2 << 7;
const RISCV_CMD_VAL: u32 = 3;

#[inline]
fn riscv_load_val(data: &[u8], i: usize) -> u32 {
    read_u16_le(data, i) as u32
}

pub fn riscv(data: &mut [u8], pc: u32, dir: Direction) -> usize {
    let size = data.len() & !1;
    if size <= 6 {
        return 0;
    }
    let limit = size - 6;
    let mut i = 0;

    'outer: while i < limit {
        // Scan loop — find a JAL or AUIPC candidate.
        let mut a;
        loop {
            if i >= limit {
                return i;
            }
            a = (riscv_load_val(data, i) ^ 0x10) + 1;
            if a & 0x77 == 0 {
                break;
            }
            a = (riscv_load_val(data, i + RISCV_INSTR_SIZE) ^ 0x10) + 1;
            i += RISCV_INSTR_SIZE * 2;
            if a & 0x77 == 0 {
                i -= RISCV_INSTR_SIZE;
                if i >= limit {
                    return i;
                }
                break;
            }
        }

        match dir {
            Direction::Encode => {
                let v = a;
                let mut a = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                if v & 8 == 0 {
                    // JAL
                    if (v.wrapping_sub(0x100)) & 0xd80 != 0 {
                        i += RISCV_INSTR_SIZE;
                        continue 'outer;
                    }
                    let mut vv = ((a & (1u32 << 31)) >> 11)
                        | ((a & (0x3ffu32 << 21)) >> 20)
                        | ((a & (1u32 << 20)) >> 9)
                        | (a & (0xffu32 << 12));
                    vv = vv.wrapping_add(pc.wrapping_add(i as u32));
                    data[i + 1] = (((vv >> 13) & 0xf0) | ((a >> 8) & 0xf)) as u8;
                    data[i + 2] = (vv >> 9) as u8;
                    data[i + 3] = (vv >> 1) as u8;
                    i += 4;
                    continue 'outer;
                }
                // AUIPC
                if v & 0xe80 != 0 {
                    let b = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap());
                    let bcheck = ((b.wrapping_sub(RISCV_CMD_VAL)) ^ (v << 8))
                        & (0xf8000 + RISCV_CMD_VAL);
                    if bcheck == 0 {
                        let temp = (b << 12) | (0x17 + RISCV_REG_VAL);
                        data[i..i + 4].copy_from_slice(&temp.to_le_bytes());
                        a &= 0xffff_f000;
                        a = a.wrapping_add(((b as i32) >> 20) as u32);
                        a = a.wrapping_add(pc.wrapping_add(i as u32));
                        write_u32_be(data, i + 4, a);
                        i += 8;
                    } else {
                        i += 4 + RISCV_INSTR_SIZE;
                    }
                } else {
                    let r = a >> 27;
                    let cond = ((v.wrapping_sub((RISCV_CMD_VAL << 12) | RISCV_REG_VAL | 8))
                        << 18)
                        < (r & 0x1d);
                    if cond {
                        let v2 = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap());
                        let r2 = (r << 7) + 0x17 + (v2 & 0xffff_f000);
                        a = (a >> 12) | (v2 << 20);
                        data[i..i + 4].copy_from_slice(&r2.to_le_bytes());
                        data[i + 4..i + 8].copy_from_slice(&a.to_le_bytes());
                        i += 8;
                    } else {
                        i += 4;
                    }
                }
            }
            Direction::Decode => {
                let v = a;
                let mut a32 = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
                if v & 8 == 0 {
                    let av = a.wrapping_sub(0x100 - 0x7f);
                    if av & 0xd80 != 0 {
                        i += RISCV_INSTR_SIZE;
                        continue 'outer;
                    }
                    let a_old = (av.wrapping_add(0xef - 0x7f)) & 0xfff;
                    let mut vv = (data[i + 3] as u32) << 1
                        | (data[i + 2] as u32) << 9
                        | ((av & 0xf000) << 5);
                    vv = vv.wrapping_sub(pc.wrapping_add(i as u32));
                    a32 = a_old
                        | ((vv << 11) & (1u32 << 31))
                        | ((vv << 20) & (0x3ff << 21))
                        | ((vv << 9) & (1u32 << 20))
                        | (vv & (0xffu32 << 12));
                    data[i..i + 4].copy_from_slice(&a32.to_le_bytes());
                    i += 4;
                    continue 'outer;
                }
                // AUIPC decode
                let v_orig = v;
                a32 |= (read_u16_le(data, i + 2) as u32) << 16;
                if v_orig & 0xe80 == 0 {
                    let r = a32 >> 27;
                    let cond = ((v_orig.wrapping_sub((RISCV_CMD_VAL << 12) | RISCV_REG_VAL | 8))
                        << 18)
                        < (r & 0x1d);
                    if cond {
                        let mut b = read_u32_be(data, i + 4);
                        let vv = a32 >> 12;
                        b = b.wrapping_sub(pc.wrapping_add(i as u32));
                        let mut anew = (r << 7) + 0x17;
                        anew = anew.wrapping_add((b.wrapping_add(0x800)) & 0xffff_f000);
                        let v_lo = vv | (b << 20);
                        data[i..i + 4].copy_from_slice(&anew.to_le_bytes());
                        data[i + 4..i + 8].copy_from_slice(&v_lo.to_le_bytes());
                        i += 8;
                    } else {
                        i += 4;
                    }
                } else {
                    let b = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap());
                    let bcheck = ((b.wrapping_sub(RISCV_CMD_VAL)) ^ (v_orig << 8))
                        & (0xf8000 + RISCV_CMD_VAL);
                    if bcheck != 0 {
                        i += 4 + RISCV_INSTR_SIZE;
                    } else {
                        let v_new = (a32 & 0xffff_f000) | (b >> 20);
                        let a_new = (b << 12) | (0x17 + RISCV_REG_VAL);
                        data[i..i + 4].copy_from_slice(&a_new.to_le_bytes());
                        data[i + 4..i + 8].copy_from_slice(&v_new.to_le_bytes());
                        i += 8;
                    }
                }
            }
        }
    }
    i
}

// ======================================================================== //
// X86 BCJ (stateful)
// ======================================================================== //

pub const X86_STATE_INIT: u32 = 0;

#[inline(always)]
fn x86_need_conv(b: u8) -> bool {
    let bb = b as u32;
    ((bb + 1) & 0xfe) == 0
}

/// Inner helper: convert one BCJ instruction at displacement position `i`,
/// returning `Some(())` on success or `None` if the candidate was a false
/// positive (caller should set `mask |= 4` and restart).
#[inline(always)]
fn x86_apply(data: &mut [u8], i: usize, pc: u32, dir: Direction) -> bool {
    let mut v = read_u32_le(data, i);
    v = v.wrapping_add(1u32 << 24);
    if v & 0xfe00_0000 != 0 {
        return false;
    }
    let c = pc.wrapping_add(i as u32 + 4);
    v = convert_val(v, c, dir);
    v &= (1u32 << 25) - 1;
    v = v.wrapping_sub(1u32 << 24);
    write_u32_le(data, i, v);
    true
}

/// Inner helper for the m0/m1/m2 path: caller has positioned `i` at
/// displacement, knows mask != 0, has already verified mask bounds and the
/// MS-byte-is-conv check.  Returns true on successful conversion.
#[inline(always)]
fn x86_apply_with_mask(
    data: &mut [u8],
    i: usize,
    pc: u32,
    mask_shifted: u32,
    dir: Direction,
) -> bool {
    let mut v = read_u32_le(data, i);
    v = v.wrapping_add(1u32 << 24);
    if v & 0xfe00_0000 != 0 {
        return false;
    }
    let c = pc.wrapping_add(i as u32 + 4);
    v = convert_val(v, c, dir);
    let m = mask_shifted << 3;
    if x86_need_conv((v >> m) as u8) {
        let v_xor = ((1u32 << 8) << m).wrapping_sub(1);
        v ^= v_xor;
        let c2 = pc.wrapping_add(i as u32 + 4);
        v = convert_val(v, c2, dir);
    }
    v &= (1u32 << 25) - 1;
    v = v.wrapping_sub(1u32 << 24);
    write_u32_le(data, i, v);
    true
}

/// Encapsulates the m0/m1/m2/a3 dispatch in the start block.
///
/// Returns the next state for the caller:
/// - `Ok((next_i, next_mask, mode))` to continue the outer loop;
/// - `Err(())` to return `disp - 1` with the given mask (fin_p path).
enum NextStep {
    Continue { i: usize, mask: u32, main: bool },
    Return { processed: usize, mask: u32 },
}

#[inline]
fn x86_handle_mn(
    data: &mut [u8],
    disp: usize,
    lim: usize,
    pc: u32,
    mask_in: u32,
    dir: Direction,
) -> NextStep {
    if mask_in == 0 {
        // C: goto a3.  No mask checks.
        if disp > lim {
            return NextStep::Return { processed: disp - 1, mask: 0 };
        }
        if x86_apply(data, disp, pc, dir) {
            NextStep::Continue { i: disp + 4, mask: 0, main: true }
        } else {
            // C's a3 fail: continue outer (mask |= 4).
            NextStep::Continue { i: disp, mask: 4, main: false }
        }
    } else {
        if disp > lim {
            return NextStep::Return { processed: disp - 1, mask: mask_in };
        }
        if mask_in > 4 || mask_in == 3 {
            // mask >>= 1; continue outer (post: mask |= 4)
            return NextStep::Continue {
                i: disp,
                mask: (mask_in >> 1) | 4,
                main: false,
            };
        }
        let m_shift = mask_in >> 1;
        if x86_need_conv(data[disp + m_shift as usize]) {
            return NextStep::Continue { i: disp, mask: m_shift | 4, main: false };
        }
        if x86_apply_with_mask(data, disp, pc, m_shift, dir) {
            NextStep::Continue { i: disp + 4, mask: 0, main: true }
        } else {
            NextStep::Continue { i: disp, mask: m_shift | 4, main: false }
        }
    }
}

#[inline]
fn x86_handle_a(
    data: &mut [u8],
    disp: usize,
    lim: usize,
    pc: u32,
    mask_in: u32,
    dir: Direction,
) -> NextStep {
    if disp > lim {
        return NextStep::Return { processed: disp - 1, mask: mask_in };
    }
    if x86_apply(data, disp, pc, dir) {
        NextStep::Continue { i: disp + 4, mask: 0, main: true }
    } else {
        NextStep::Continue { i: disp, mask: mask_in | 4, main: false }
    }
}

pub fn x86_bcj(data: &mut [u8], pc: u32, state: &mut u32, dir: Direction) -> usize {
    if data.len() < 5 {
        return 0;
    }
    let lim = data.len() - 4;
    let mut i = 0usize;
    let mut mask = *state;
    let mut main_mode = false;

    loop {
        if main_mode {
            // ----- main_loop (mask history is "0", just scan)
            if i >= lim {
                *state = mask;
                return i;
            }
            let b0 = data[i];
            let b1 = data[i + 1];
            let b2 = data[i + 2];
            let b3 = data[i + 3];
            let next = if (b0 ^ 0xe8) & 0xfe == 0 {
                Some(x86_handle_a(data, i + 1, lim, pc, mask, dir))
            } else if (b1 ^ 0xe8) & 0xfe == 0 {
                Some(x86_handle_a(data, i + 2, lim, pc, mask, dir))
            } else if (b2 ^ 0xe8) & 0xfe == 0 {
                Some(x86_handle_a(data, i + 3, lim, pc, mask, dir))
            } else if (b3 ^ 0xe8) & 0xfe == 0 {
                Some(x86_handle_a(data, i + 4, lim, pc, mask, dir))
            } else {
                None
            };
            match next {
                Some(NextStep::Return { processed, mask: m }) => {
                    *state = m;
                    return processed;
                }
                Some(NextStep::Continue { i: ni, mask: nm, main }) => {
                    i = ni;
                    mask = nm;
                    main_mode = main;
                }
                None => {
                    i += 4;
                }
            }
        } else {
            // ----- start (mask history active)
            if i >= lim {
                *state = mask;
                return i;
            }
            let b0 = data[i];
            let b1 = data[i + 1];
            let b2 = data[i + 2];
            let b3 = data[i + 3];

            // Sequential mask updates as we scan bytes 0..3.
            let next = if (b0 ^ 0xe8) & 0xfe == 0 {
                Some(x86_handle_mn(data, i + 1, lim, pc, mask, dir))
            } else {
                let m1 = mask >> 1;
                if (b1 ^ 0xe8) & 0xfe == 0 {
                    Some(x86_handle_mn(data, i + 2, lim, pc, m1, dir))
                } else {
                    let m2 = m1 >> 1;
                    if (b2 ^ 0xe8) & 0xfe == 0 {
                        Some(x86_handle_mn(data, i + 3, lim, pc, m2, dir))
                    } else {
                        // After byte 2 didn't match, C sets mask = 0.
                        if (b3 ^ 0xe8) & 0xfe == 0 {
                            Some(x86_handle_a(data, i + 4, lim, pc, 0, dir))
                        } else {
                            None
                        }
                    }
                }
            };
            match next {
                Some(NextStep::Return { processed, mask: m }) => {
                    *state = m;
                    return processed;
                }
                Some(NextStep::Continue { i: ni, mask: nm, main }) => {
                    i = ni;
                    mask = nm;
                    main_mode = main;
                }
                None => {
                    // No match — advance to main_loop.
                    i += 4;
                    mask = 0;
                    main_mode = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<F>(mut f: F, data: Vec<u8>, pc: u32)
    where
        F: FnMut(&mut [u8], u32, Direction) -> usize,
    {
        let original = data.clone();
        let mut buf = data;
        let processed_e = f(&mut buf, pc, Direction::Encode);
        let processed_d = f(&mut buf, pc, Direction::Decode);
        assert_eq!(processed_e, processed_d, "encode/decode bytes mismatch");
        assert_eq!(buf, original, "round-trip not identity");
    }

    #[test]
    fn arm64_round_trip() {
        let mut data: Vec<u8> = Vec::new();
        // Mix of BL and ADRP and other instructions.
        data.extend_from_slice(&0x9400_0001u32.to_le_bytes()); // BL +1
        data.extend_from_slice(&0x9000_0001u32.to_le_bytes()); // ADRP X1, #1
        data.extend_from_slice(&0xd503_201fu32.to_le_bytes()); // NOP
        data.extend_from_slice(&0x9400_FFF0u32.to_le_bytes());
        round_trip(arm64, data, 0x1000);
    }

    #[test]
    fn arm_round_trip() {
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&0xeb00_0001u32.to_le_bytes()); // BL ...
        data.extend_from_slice(&0xe1a0_0000u32.to_le_bytes()); // MOV
        data.extend_from_slice(&0xeb00_0010u32.to_le_bytes());
        round_trip(arm, data, 0x2000);
    }

    #[test]
    fn ppc_round_trip() {
        let mut data: Vec<u8> = Vec::new();
        // BL with LK set, big-endian
        data.extend_from_slice(&0x4800_0001u32.to_be_bytes());
        data.extend_from_slice(&0x6000_0000u32.to_be_bytes()); // NOP
        data.extend_from_slice(&0x4800_0801u32.to_be_bytes());
        round_trip(ppc, data, 0x4000);
    }

    // X86 BCJ correctness is validated by the cross-check binary against the
    // reference C implementation (see `tests/bra_xcheck.rs`). The boundary
    // semantics make a self-contained round-trip test misleading.
}
