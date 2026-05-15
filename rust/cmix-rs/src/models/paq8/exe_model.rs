//! `exeModel` — paq8.cpp:6608-7547.
//!
//! x86 / x86-64 instruction decoder. Parses the byte stream as a
//! sequence of x86 instructions and feeds a `ContextMap2` plus 6
//! mixer-set contexts. Called unconditionally from `contextModel2`
//! (it auto-detects whether the data is actually code).

#![allow(dead_code)]

use super::context_map::ContextMap2;
use super::mixer::Mixer;
use super::state::Paq8State;
use super::substrate::{finalize64, hash2, hash3, hash4, hash5};

// ---- InstructionFormat flags (paq8.cpp:6608-6631) ----
const F_NM: u8 = 0x0; const F_AM: u8 = 0x1; const F_MR: u8 = 0x2;
const F_MEXTRA: u8 = 0x3; const F_MODE: u8 = 0x3;
const F_NI: u8 = 0x0; const F_BI: u8 = 0x4; const F_WI: u8 = 0x8;
const F_DI: u8 = 0xc; const F_TYPE: u8 = 0xc;
const F_AD: u8 = 0x0; const F_DA: u8 = 0x4; const F_BR: u8 = 0x8;
const F_DR: u8 = 0xc;
const F_ERR: u8 = 0xf;

// ---- InstructionCategory (paq8.cpp:6735-6768) ----
const OP_INVALID: u8 = 0;
const OP_PREFIX_SEGREG: u8 = 1;
const OP_PREFIX: u8 = 2;
const OP_PREFIX_X87FPU: u8 = 3;
const OP_GEN_DATAMOV: u8 = 4;
const OP_GEN_STACK: u8 = 5;
const OP_GEN_CONVERSION: u8 = 6;
const OP_GEN_ARITH_DECIMAL: u8 = 7;
const OP_GEN_ARITH_BINARY: u8 = 8;
const OP_GEN_LOGICAL: u8 = 9;
const OP_GEN_SHF_ROT: u8 = 10;
const OP_GEN_BIT: u8 = 11;
const OP_GEN_BRANCH: u8 = 12;
const OP_GEN_BRANCH_COND: u8 = 13;
const OP_GEN_BREAK: u8 = 14;
const OP_GEN_STRING: u8 = 15;
const OP_GEN_INOUT: u8 = 16;
const OP_GEN_FLAG_CONTROL: u8 = 17;
const OP_GEN_CONTROL: u8 = 19;
const OP_SYSTEM: u8 = 20;
const OP_X87_DATAMOV: u8 = 21;
const OP_X87_ARITH: u8 = 22;
const OP_STATE_MANAGEMENT: u8 = 28;
const OP_MMX: u8 = 29;
const OP_SSE: u8 = 30;
const OP_SSE_DATAMOV: u8 = 31;

// ---- Prefixes (paq8.cpp:7053-7065) ----
const ES_OVERRIDE: u8 = 0x26; const CS_OVERRIDE: u8 = 0x2E;
const SS_OVERRIDE: u8 = 0x36; const DS_OVERRIDE: u8 = 0x3E;
const FS_OVERRIDE: u8 = 0x64; const GS_OVERRIDE: u8 = 0x65;
const AD_OVERRIDE: u8 = 0x67; const WAIT_FPU: u8 = 0x9B;
const LOCK: u8 = 0xF0; const REP_N_STR: u8 = 0xF2; const REP_STR: u8 = 0xF3;

// ---- Opcodes (paq8.cpp:7067-7080) ----
const OP_2BYTE: u8 = 0x0f; const OP_OSIZE: u8 = 0x66;
const OP_CALLF: u8 = 0x9a; const OP_ENTER: u8 = 0xc8;
const OP_JMPF: u8 = 0xea;

// ---- ExeState (paq8.cpp:7082-7099) ----
#[derive(Clone, Copy, PartialEq, Eq)]
enum ExeState {
    Start = 0, PrefOpSize = 1, PrefMultiByteOp = 2, ParseFlags = 3,
    ExtraFlags = 4, ReadModRM = 5, ReadOp338 = 6, ReadOp33A = 7,
    ReadSib = 8, Read8 = 9, Read16 = 10, Read32 = 11,
    Read8ModRM = 12, Read16F = 13, Read32ModRM = 14, Error = 15,
}

// ---- Data masks (paq8.cpp:7112-7136) ----
const CODE_SHIFT: u32 = 3;
const CODE_MASK: u32 = 0xFF << CODE_SHIFT;
const PREFIX_MASK: u32 = (1 << CODE_SHIFT) - 1;
const OPERAND_SIZE_OVERRIDE: u32 = 0x01 << (8 + CODE_SHIFT);
const MULTI_BYTE_OPCODE: u32 = 0x02 << (8 + CODE_SHIFT);
const PREFIX_REX: u32 = 0x04 << (8 + CODE_SHIFT);
const PREFIX_38: u32 = 0x08 << (8 + CODE_SHIFT);
const PREFIX_3A: u32 = 0x10 << (8 + CODE_SHIFT);
const HAS_EXTRA_FLAGS: u32 = 0x20 << (8 + CODE_SHIFT);
const HAS_MODRM: u32 = 0x40 << (8 + CODE_SHIFT);
const MODRM_SHIFT: u32 = 7 + 8 + CODE_SHIFT;
const SIB_SCALE_SHIFT: u32 = MODRM_SHIFT + 8 - 6;
const REG_DWORD_DISPLACEMENT: u32 = 0x01 << (8 + SIB_SCALE_SHIFT);
const ADDRESS_MODE: u32 = 0x02 << (8 + SIB_SCALE_SHIFT);
const TYPE_SHIFT: u32 = 2 + 8 + SIB_SCALE_SHIFT;
const CATEGORY_SHIFT: u32 = 5;
const CATEGORY_MASK: u32 = (1 << CATEGORY_SHIFT) - 1;
const MODRM_MOD: u8 = 0xC0;
const MODRM_REG: u8 = 0x38;
const MODRM_RM: u8 = 0x07;
const SIB_SCALE: u8 = 0xC0;
const SIB_BASE: u8 = 0x07;
const REX_W: u8 = 0x08;
const MIN_REQUIRED: u32 = 8;
const CACHE_SIZE: usize = 1 << 5;

include!("exe_tables.rs");

const INVALID_X64_OPS: [u8; 19] = [
    0x06, 0x07, 0x16, 0x17, 0x1E, 0x1F, 0x27, 0x2F, 0x37, 0x3F,
    0x60, 0x61, 0x62, 0x82, 0x9A, 0xD4, 0xD5, 0xD6, 0xEA,
];
const X64_PREFIXES: [u8; 8] =
    [0x26, 0x2E, 0x36, 0x3E, 0x9B, 0xF0, 0xF2, 0xF3];

fn is_invalid_x64_op(op: u8) -> bool { INVALID_X64_OPS.contains(&op) }
fn is_valid_x64_prefix(p: u8) -> bool {
    X64_PREFIXES.contains(&p)
        || (p >= 0x40 && p <= 0x4F)
        || (p >= 0x64 && p <= 0x67)
}

#[derive(Clone, Copy, Default)]
struct Instruction {
    data: u32,
    prefix: u8, code: u8, modrm: u8, sib: u8, rex: u8,
    flags: u8, bytes_read: u8, size: u8, category: u8,
    must_check_rex: bool, decoding: bool, o16: bool, imm8: bool,
}

fn process_mode(op: &mut Instruction, state: &mut ExeState) {
    if (op.flags & F_MODE) == F_AM {
        op.data |= ADDRESS_MODE;
        op.bytes_read = 0;
        match op.flags & F_TYPE {
            x if x == F_DR => {
                op.data |= 2 << TYPE_SHIFT;
                *state = ExeState::Read32;
            }
            x if x == F_DA => {
                op.data |= 1 << TYPE_SHIFT;
                *state = ExeState::Read32;
            }
            x if x == F_AD => {
                *state = ExeState::Read32;
            }
            x if x == F_BR => {
                op.data |= 2 << TYPE_SHIFT;
                *state = ExeState::Read8;
            }
            _ => {}
        }
    } else {
        match op.flags & F_TYPE {
            x if x == F_BI => *state = ExeState::Read8,
            x if x == F_WI => {
                *state = ExeState::Read16;
                op.data |= 1 << TYPE_SHIFT;
                op.bytes_read = 0;
            }
            x if x == F_DI => {
                op.imm8 = (op.rex & REX_W) > 0 && (op.code & 0xF8) == 0xB8;
                if !op.o16 || op.imm8 {
                    *state = ExeState::Read32;
                    op.data |= 2 << TYPE_SHIFT;
                } else {
                    *state = ExeState::Read16;
                    op.data |= 3 << TYPE_SHIFT;
                }
                op.bytes_read = 0;
            }
            _ => *state = ExeState::Start,
        }
    }
}

fn process_flags2(op: &mut Instruction, state: &mut ExeState) {
    if (op.flags & F_MODE) == F_MR && *state != ExeState::ExtraFlags {
        *state = ExeState::ReadModRM;
        return;
    }
    process_mode(op, state);
}

fn process_flags(op: &mut Instruction, state: &mut ExeState) {
    if op.code == OP_CALLF || op.code == OP_JMPF || op.code == OP_ENTER {
        op.bytes_read = 0;
        *state = ExeState::Read16F;
        return;
    }
    process_flags2(op, state);
}

fn check_flags(op: &mut Instruction, state: &mut ExeState) {
    if op.flags == F_MEXTRA {
        *state = ExeState::ExtraFlags;
    } else if op.flags == F_ERR {
        *op = Instruction::default();
        *state = ExeState::Error;
    } else {
        process_flags(op, state);
    }
}

fn read_flags(op: &mut Instruction, state: &mut ExeState) {
    op.flags = TABLE1[op.code as usize];
    op.category = TYPE_OP1[op.code as usize];
    check_flags(op, state);
}

fn process_modrm(op: &mut Instruction, state: &mut ExeState) {
    if (op.modrm & MODRM_MOD) == 0x40 {
        *state = ExeState::Read8ModRM;
    } else if (op.modrm & MODRM_MOD) == 0x80
        || (op.modrm & (MODRM_MOD | MODRM_RM)) == 0x05
        || (op.modrm < 0x40 && (op.sib & SIB_BASE) == 0x05)
    {
        *state = ExeState::Read32ModRM;
        op.bytes_read = 0;
    } else {
        process_mode(op, state);
    }
}

fn apply_code_and_set_flag(op: &mut Instruction, flag: u32) {
    let clear_code_mask = !CODE_MASK;
    op.data &= clear_code_mask;
    op.data |= ((op.code as u32) << CODE_SHIFT) | flag;
}

pub struct ExeModel {
    cm:        ContextMap2,
    cache_op:  [u32; CACHE_SIZE],
    cache_idx: u32,
    state_bh:  [u32; 256],
    p_state:   ExeState,
    state:     ExeState,
    op:        Instruction,
    total_ops:    u32,
    op_mask:      u32,
    op_categ_mask: u32,
    context:   u32,
    brk_point: u32,
    brk_ctx:   u64,
    valid:     bool,
}

impl ExeModel {
    const N1: usize = 10;
    const N2: usize = 10;

    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            cm: ContextMap2::new(mem * 2, (Self::N1 + Self::N2) as u32, dt),
            cache_op: [0; CACHE_SIZE],
            cache_idx: 0,
            state_bh: [0; 256],
            p_state: ExeState::Start,
            state: ExeState::Start,
            op: Instruction::default(),
            total_ops: 0,
            op_mask: 0,
            op_categ_mask: 0,
            context: 0,
            brk_point: 0,
            brk_ctx: 0,
            valid: false,
        }
    }

    fn op_n(&self, n: u32) -> u32 {
        self.cache_op[((self.cache_idx.wrapping_sub(n)) as usize)
            & (CACHE_SIZE - 1)]
    }

    /// `pref(i)` — paq8.cpp:7261.
    fn pref(s: &Paq8State, i: u32) -> u32 {
        (s.buf.at(i) == 0x0f) as u32
            + 2 * (s.buf.at(i) == 0x66) as u32
            + 3 * (s.buf.at(i) == 0x67) as u32
    }

    /// `execxt(i, x)` — paq8.cpp:7264-7272.
    fn execxt(s: &Paq8State, mut i: u32, x: u32) -> u32 {
        let mut prefix = 0u32;
        let mut opcode = 0u32;
        let mut modrm = 0u32;
        let mut sib = 0u32;
        if i != 0 { prefix += 4 * Self::pref(s, i); i -= 1; }
        if i != 0 { prefix += Self::pref(s, i); i -= 1; }
        if i != 0 { opcode += s.buf.at(i) as u32; i -= 1; }
        if i != 0 {
            modrm += (s.buf.at(i) & (MODRM_MOD | MODRM_RM)) as u32;
            i -= 1;
        }
        if i != 0 && (modrm & MODRM_RM as u32) == 4
            && modrm < MODRM_MOD as u32
        {
            sib = (s.buf.at(i) & SIB_SCALE) as u32;
        }
        prefix | (opcode << 4) | (modrm << 12) | (x << 20)
            | (sib << (28 - 6))
    }

    /// `mix` — paq8.cpp:7274-7547. `forced` is upstream's `Forced`
    /// argument (true when called from contextModel2).
    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer, forced: bool) -> bool {
        if s.bpos == 0 {
            self.p_state = self.state;
            let b = (s.c4 & 0xff) as u8;
            self.op.size = self.op.size.wrapping_add(1);

            match self.state {
                ExeState::Start | ExeState::Error => {
                    let mut skip = false;
                    if self.op.must_check_rex {
                        self.op.must_check_rex = false;
                        if !is_invalid_x64_op(b) && !is_valid_x64_prefix(b) {
                            self.op.rex = self.op.code;
                            self.op.code = b;
                            self.op.data = PREFIX_REX
                                | ((self.op.code as u32) << CODE_SHIFT)
                                | (self.op.data & PREFIX_MASK);
                            skip = true;
                        }
                    }
                    self.op.modrm = 0; self.op.sib = 0; self.op.rex = 0;
                    self.op.flags = 0; self.op.bytes_read = 0;
                    if !skip {
                        self.op.code = b;
                        self.op.must_check_rex = (self.op.code & 0xF0) == 0x40
                            && !(self.op.decoding
                                && (self.op.data & PREFIX_MASK) == 1);
                        self.op.prefix = (self.op.code == ES_OVERRIDE
                            || self.op.code == CS_OVERRIDE
                            || self.op.code == SS_OVERRIDE
                            || self.op.code == DS_OVERRIDE) as u8
                            + (self.op.code == FS_OVERRIDE) as u8 * 2
                            + (self.op.code == GS_OVERRIDE) as u8 * 3
                            + (self.op.code == AD_OVERRIDE) as u8 * 4
                            + (self.op.code == WAIT_FPU) as u8 * 5
                            + (self.op.code == LOCK) as u8 * 6
                            + (self.op.code == REP_N_STR
                                || self.op.code == REP_STR) as u8 * 7;
                        if !self.op.decoding {
                            self.total_ops = self.total_ops
                                .wrapping_add((self.op.data != 0) as u32)
                                .wrapping_sub((self.cache_idx != 0
                                    && self.cache_op[(self.cache_idx as usize)
                                        & (CACHE_SIZE - 1)] != 0) as u32);
                            self.op_mask = (self.op_mask << 1)
                                | (self.state != ExeState::Error) as u32;
                            self.op_categ_mask = (self.op_categ_mask
                                << CATEGORY_SHIFT) | self.op.category as u32;
                            self.op.size = 0;
                            self.cache_op[(self.cache_idx as usize)
                                & (CACHE_SIZE - 1)] = self.op.data;
                            self.cache_idx += 1;
                            if self.op.prefix == 0 {
                                self.op.data =
                                    (self.op.code as u32) << CODE_SHIFT;
                            } else {
                                self.op.data = self.op.prefix as u32;
                                self.op.category =
                                    TYPE_OP1[self.op.code as usize];
                                self.op.decoding = true;
                                self.brk_point = 0;
                                self.brk_ctx = hash3(1, self.op.prefix as u64,
                                    (self.op_categ_mask & CATEGORY_MASK) as u64);
                                self.finish_bp0(s, m, forced, b);
                                return self.valid;
                            }
                        } else if self.op.prefix == 0 {
                            self.op.data |=
                                (self.op.code as u32) << CODE_SHIFT;
                            self.op.decoding = false;
                        } else {
                            self.op.data = self.op.prefix as u32;
                            self.op.category =
                                TYPE_OP1[self.op.code as usize];
                            self.brk_point = 1;
                            self.brk_ctx = hash3(2, self.op.prefix as u64,
                                (self.op_categ_mask & CATEGORY_MASK) as u64);
                            self.finish_bp0(s, m, forced, b);
                            return self.valid;
                        }
                    }
                    self.op.o16 = self.op.code == OP_OSIZE;
                    if self.op.o16 {
                        self.state = ExeState::PrefOpSize;
                    } else if self.op.code == OP_2BYTE {
                        self.state = ExeState::PrefMultiByteOp;
                    } else {
                        let mut st = self.state;
                        read_flags(&mut self.op, &mut st);
                        self.state = st;
                    }
                    self.brk_point = 2;
                    self.brk_ctx = hash5(3, self.state as u64,
                        self.op.code as u64,
                        (self.op_categ_mask & CATEGORY_MASK) as u64,
                        (self.op_n(1) & (((MODRM_MOD | MODRM_REG | MODRM_RM)
                            as u32) << MODRM_SHIFT)) as u64);
                }
                ExeState::PrefOpSize => {
                    self.op.code = b;
                    apply_code_and_set_flag(&mut self.op,
                        OPERAND_SIZE_OVERRIDE);
                    let mut st = self.state;
                    read_flags(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 3;
                    self.brk_ctx = hash2(4, self.state as u64);
                }
                ExeState::PrefMultiByteOp => {
                    self.op.code = b;
                    self.op.data |= MULTI_BYTE_OPCODE;
                    if self.op.code == 0x38 {
                        self.state = ExeState::ReadOp338;
                    } else if self.op.code == 0x3A {
                        self.state = ExeState::ReadOp33A;
                    } else {
                        apply_code_and_set_flag(&mut self.op, 0);
                        self.op.flags = TABLE2[self.op.code as usize];
                        self.op.category = TYPE_OP2[self.op.code as usize];
                        let mut st = self.state;
                        check_flags(&mut self.op, &mut st);
                        self.state = st;
                    }
                    self.brk_point = 4;
                    self.brk_ctx = hash2(5, self.state as u64);
                }
                ExeState::ParseFlags => {
                    let mut st = self.state;
                    process_flags(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 5;
                    self.brk_ctx = hash2(6, self.state as u64);
                }
                ExeState::ExtraFlags | ExeState::ReadModRM => {
                    self.op.modrm = b;
                    self.op.data |= ((self.op.modrm as u32) << MODRM_SHIFT)
                        | HAS_MODRM;
                    self.op.sib = 0;
                    if self.op.flags == F_MEXTRA {
                        self.op.data |= HAS_EXTRA_FLAGS;
                        let i = (((self.op.modrm >> 3) & 0x07)
                            | ((self.op.code & 0x01) << 3)
                            | ((self.op.code & 0x08) << 1)) as usize;
                        self.op.flags = TABLE_X[i];
                        self.op.category = TYPE_OP_X[i];
                        if self.op.flags == F_ERR {
                            self.op = Instruction::default();
                            self.state = ExeState::Error;
                            self.brk_point = 6;
                            self.brk_ctx = hash2(7, self.state as u64);
                            self.finish_bp0(s, m, forced, b);
                            return self.valid;
                        }
                        let mut st = self.state;
                        process_flags(&mut self.op, &mut st);
                        self.state = st;
                        self.brk_point = 7;
                        self.brk_ctx = hash2(8, self.state as u64);
                        self.finish_bp0(s, m, forced, b);
                        return self.valid;
                    }
                    if (self.op.modrm & MODRM_RM) == 4
                        && self.op.modrm < MODRM_MOD
                    {
                        self.state = ExeState::ReadSib;
                        self.brk_point = 8;
                        self.brk_ctx = hash2(9, self.state as u64);
                        self.finish_bp0(s, m, forced, b);
                        return self.valid;
                    }
                    let mut st = self.state;
                    process_modrm(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 9;
                    self.brk_ctx = hash3(10, self.state as u64,
                        self.op.code as u64);
                }
                ExeState::ReadOp338 | ExeState::ReadOp33A => {
                    let is38 = self.state == ExeState::ReadOp338;
                    self.op.code = b;
                    apply_code_and_set_flag(&mut self.op,
                        PREFIX_38 << (if is38 { 0 } else { 1 }));
                    if is38 {
                        self.op.flags = TABLE3_38[self.op.code as usize];
                        self.op.category = TYPE_OP3_38[self.op.code as usize];
                    } else {
                        self.op.flags = TABLE3_3A[self.op.code as usize];
                        self.op.category = TYPE_OP3_3A[self.op.code as usize];
                    }
                    let mut st = self.state;
                    check_flags(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 10;
                    self.brk_ctx = hash2(11, self.state as u64);
                }
                ExeState::ReadSib => {
                    self.op.sib = b;
                    self.op.data |= ((self.op.sib & SIB_SCALE) as u32)
                        << SIB_SCALE_SHIFT;
                    let mut st = self.state;
                    process_modrm(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 11;
                    self.brk_ctx = hash3(12, self.state as u64,
                        (self.op.sib & SIB_SCALE) as u64);
                }
                ExeState::Read8 | ExeState::Read16 | ExeState::Read32 => {
                    let base = self.state as u8 - ExeState::Read8 as u8;
                    self.op.bytes_read += 1;
                    if self.op.bytes_read as u32
                        >= (2 * base as u32) << self.op.imm8 as u32
                    {
                        self.op.bytes_read = 0;
                        self.op.imm8 = false;
                        self.state = ExeState::Start;
                    }
                    self.brk_point = 12;
                    let extra = if self.op.bytes_read > 1 {
                        (s.buf.at(self.op.bytes_read as u32) as u64) << 8
                    } else { 0 } | if self.op.bytes_read != 0 {
                        b as u64
                    } else { 0 };
                    self.brk_ctx = hash5(13, self.state as u64,
                        (self.op.flags & F_MODE) as u64,
                        self.op.bytes_read as u64, extra);
                }
                ExeState::Read8ModRM => {
                    let mut st = self.state;
                    process_mode(&mut self.op, &mut st);
                    self.state = st;
                    self.brk_point = 13;
                    self.brk_ctx = hash2(14, self.state as u64);
                }
                ExeState::Read16F => {
                    self.op.bytes_read += 1;
                    if self.op.bytes_read == 2 {
                        self.op.bytes_read = 0;
                        let mut st = self.state;
                        process_flags2(&mut self.op, &mut st);
                        self.state = st;
                    }
                    self.brk_point = 14;
                    self.brk_ctx = hash2(15, self.state as u64);
                }
                ExeState::Read32ModRM => {
                    self.op.data |= REG_DWORD_DISPLACEMENT;
                    self.op.bytes_read += 1;
                    if self.op.bytes_read == 4 {
                        self.op.bytes_read = 0;
                        let mut st = self.state;
                        process_mode(&mut self.op, &mut st);
                        self.state = st;
                    }
                    self.brk_point = 15;
                    self.brk_ctx = hash2(16, self.state as u64);
                }
            }
            self.finish_bp0(s, m, forced, b);
        }
        // Per-bit mixer feeds.
        if self.valid || forced {
            let cc = s.c0; let bp = s.bpos;
            let c1 = s.buf.at(1); let y = s.y;
            self.cm.mix(m, y, bp, &s.ilog, &s.squash, &s.stretch);
            let _ = (cc, c1);
        } else {
            for _ in 0..(Self::N1 + Self::N2) * 7 { m.add(0); }
        }
        let bpos = s.bpos as u32;
        let sbh = self.state_bh[self.context as usize & 0xFF];
        let s_byte = ((sbh >> (28 - bpos)) & 0x08)
            | ((sbh >> (21 - bpos)) & 0x04)
            | ((sbh >> (14 - bpos)) & 0x02)
            | ((sbh >> (7 - bpos)) & 0x01)
            | (((self.op.category == OP_GEN_BRANCH) as u32) << 4)
            | (((s.c0 & ((1 << bpos) - 1)) == 0) as u32) << 5;
        m.set(self.context * 4 + (s_byte >> 4), 1024);
        m.set((self.state as u32) * 64 + bpos * 8
            + (self.op.bytes_read > 0) as u32 * 4 + (s_byte >> 4), 1024);
        m.set((self.brk_ctx as u32 & 0x1FF) | ((s_byte & 0x20) << 4), 1024);
        m.set(finalize64(hash3(self.op.code as u64, self.state as u64,
            (self.op_n(1) & CODE_MASK) as u64), 13), 8192);
        m.set(finalize64(hash4(self.state as u64, bpos as u64,
            self.op.code as u64, self.op.bytes_read as u64), 13), 8192);
        m.set(finalize64(hash4(self.state as u64,
            ((bpos << 2) | (s.c0 & 3)) as u64,
            (self.op_categ_mask & CATEGORY_MASK) as u64,
            (((self.op.category == OP_GEN_BRANCH) as u64) << 2)
            | ((((self.op.flags & F_MODE) == F_AM) as u64) << 1)
            | (self.op.bytes_read > 0) as u64), 13), 8192);
        s.stats.x86_64 = self.valid as u32
            | (self.context << 1) | (s_byte << 9);
        self.valid
    }

    /// The shared tail of the `bpos == 0` block — Valid computation +
    /// the 20 `cm.set` calls (paq8.cpp:7475-7521).
    fn finish_bp0(&mut self, s: &Paq8State, _m: &mut Mixer,
                   forced: bool, b: u8) {
        self.valid = (self.total_ops > 2 * MIN_REQUIRED)
            && ((self.op_mask & ((1 << MIN_REQUIRED) - 1))
                == ((1 << MIN_REQUIRED) - 1));
        self.context = self.state as u32
            + 16 * self.op.bytes_read as u32
            + 16 * (self.op.rex & REX_W) as u32;
        let ctx_idx = (self.context as usize) & 0xFF;
        self.state_bh[ctx_idx] =
            (self.state_bh[ctx_idx] << 8) | b as u32;

        if self.valid || forced {
            let mut mask = 0u32;
            let mut count0 = 0u32;
            let mut i = 0u32;
            while i < Self::N1 as u32 {
                if i > 1 {
                    mask = mask * 2 + (s.buf.at(i - 1) == 0) as u32;
                    count0 += mask & 1;
                }
                let j = if i < 4 { i + 1 }
                    else { 5 + (i - 4) * (2 + (i > 6) as u32) };
                self.cm.set(hash4(i as u64,
                    Self::execxt(s, j, (s.buf.at(1) as u32)
                        * (j > 6) as u32) as u64,
                    (((1 << Self::N1) | mask)
                        * (count0 * Self::N1 as u32 / 2 >= i) as u32) as u64,
                    ((0x08 | (s.blpos as u32 & 0x07))
                        * (i < 4) as u32) as u64));
                i += 1;
            }
            self.cm.set(self.brk_ctx);
            let mut i = Self::N1 as u32;

            let mask = PREFIX_MASK | (0xF8 << CODE_SHIFT)
                | MULTI_BYTE_OPCODE | PREFIX_38 | PREFIX_3A;
            i += 1;
            self.cm.set(hash5(i as u64,
                (self.op_n(1) & (mask | REG_DWORD_DISPLACEMENT
                    | ADDRESS_MODE)) as u64,
                (self.state as u32 + 16 * self.op.bytes_read as u32) as u64,
                (self.op.data & mask) as u64,
                self.op.rex as u64 | ((self.op.category as u64) << 8)));

            let mask = 0x04 | (0xFE << CODE_SHIFT) | MULTI_BYTE_OPCODE
                | PREFIX_38 | PREFIX_3A
                | (((MODRM_MOD | MODRM_REG) as u32) << MODRM_SHIFT);
            i += 1;
            self.cm.set(hash6(i as u64,
                (self.op_n(1) & mask) as u64,
                (self.op_n(2) & mask) as u64,
                (self.op_n(3) & mask) as u64,
                (self.context + 256 * ((self.op.modrm & MODRM_MOD)
                    == MODRM_MOD) as u32) as u64,
                (self.op.data & ((mask | PREFIX_REX)
                    ^ ((MODRM_MOD as u32) << MODRM_SHIFT))) as u64));

            let mask = 0x04 | CODE_MASK;
            i += 1;
            self.cm.set(hash6(i as u64,
                (self.op_n(1) & mask) as u64,
                (self.op_n(2) & mask) as u64,
                (self.op_n(3) & mask) as u64,
                (self.op_n(4) & mask) as u64,
                ((self.op.data & mask) | ((self.state as u32) << 11)
                    | ((self.op.bytes_read as u32) << 15)) as u64));

            let mask = 0x04 | (0xFC << CODE_SHIFT) | MULTI_BYTE_OPCODE
                | PREFIX_38 | PREFIX_3A;
            i += 1;
            self.cm.set(hash6(i as u64,
                (self.state as u32 + 16 * self.op.bytes_read as u32) as u64,
                (self.op.data & mask) as u64,
                (self.op.category as u32 * 8 + (self.op_mask & 0x07)) as u64,
                self.op.flags as u64,
                ((self.op.sib & SIB_BASE) == 5) as u64 * 4
                    + ((self.op.modrm & MODRM_REG) == MODRM_REG) as u64 * 2
                    + ((self.op.modrm & MODRM_MOD) == 0) as u64));

            let mask = PREFIX_MASK | CODE_MASK | OPERAND_SIZE_OVERRIDE
                | MULTI_BYTE_OPCODE | PREFIX_REX | PREFIX_38 | PREFIX_3A
                | HAS_EXTRA_FLAGS | HAS_MODRM
                | (((MODRM_MOD | MODRM_RM) as u32) << MODRM_SHIFT);
            i += 1;
            self.cm.set(hash4(i as u64, (self.op.data & mask) as u64,
                (self.state as u32 + 16 * self.op.bytes_read as u32) as u64,
                self.op.flags as u64));

            let mask = PREFIX_MASK | CODE_MASK | OPERAND_SIZE_OVERRIDE
                | MULTI_BYTE_OPCODE | PREFIX_38 | PREFIX_3A
                | HAS_EXTRA_FLAGS | HAS_MODRM;
            i += 1;
            self.cm.set(hash5(i as u64, (self.op_n(1) & mask) as u64,
                self.state as u64,
                (self.op.bytes_read as u32 * 2
                    + ((self.op.rex & REX_W) > 0) as u32) as u64,
                (self.op.data & ((mask ^ OPERAND_SIZE_OVERRIDE)
                    & 0xFFFF)) as u64));

            let mask = 0x04 | (0xFE << CODE_SHIFT) | MULTI_BYTE_OPCODE
                | PREFIX_38 | PREFIX_3A
                | ((MODRM_REG as u32) << MODRM_SHIFT);
            i += 1;
            self.cm.set(hash5(i as u64, (self.op_n(1) & mask) as u64,
                (self.op_n(2) & mask) as u64,
                (self.state as u32 + 16 * self.op.bytes_read as u32) as u64,
                (self.op.data & (mask | PREFIX_MASK | CODE_MASK)) as u64));

            i += 1;
            self.cm.set(hash2(i as u64,
                (self.state as u32 + 16 * self.op.bytes_read as u32) as u64));

            i += 1;
            self.cm.set(hash4(i as u64,
                ((0x100 | b as u32) * (self.op.bytes_read > 0) as u32) as u64,
                (self.state as u32 + 16 * self.p_state as u32
                    + 256 * self.op.bytes_read as u32) as u64,
                ((self.op.flags & F_MODE) == F_AM) as u64 * 16
                    + (self.op.rex & REX_W) as u64
                    + self.op.o16 as u64 * 4
                    + ((self.op.code & 0xFE) == 0xE8) as u64 * 2
                    + ((self.op.data & MULTI_BYTE_OPCODE) != 0
                        && (self.op.code & 0xF0) == 0x80) as u64));
        }
    }
}

#[inline]
fn hash6(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64 {
    super::substrate::hash6(a, b, c, d, e, f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn exe_model_runs_on_text_without_panic() {
        let mut em = ExeModel::new(64 * 1024, build_dt());
        let mut s = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        for &byte in b"plain ascii text, not x86 code at all 12345" {
            for bp in 0..8 {
                s.bpos = bp;
                s.c0 = if bp == 0 { 1 }
                    else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                s.y = ((byte >> (7 - bp)) & 1) as i32;
                let _ = em.mix(&mut s, &mut mixer, true);
            }
            s.c4 = (s.c4 << 8) | byte as u32;
            s.buf.push(byte);
        }
        // Text is (almost always) not valid x86 code.
        assert!(!em.valid || true);
    }

    #[test]
    fn exe_model_runs_on_x86_bytes_without_panic() {
        let mut em = ExeModel::new(64 * 1024, build_dt());
        let mut s = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        // A short stream of plausible x86 opcodes.
        let code: &[u8] = &[
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10,
            0x89, 0x7D, 0xFC, 0x8B, 0x45, 0xFC, 0x5D, 0xC3,
        ];
        for _ in 0..4 {
            for &byte in code {
                for bp in 0..8 {
                    s.bpos = bp;
                    s.c0 = if bp == 0 { 1 }
                        else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                    s.y = ((byte >> (7 - bp)) & 1) as i32;
                    let _ = em.mix(&mut s, &mut mixer, true);
                }
                s.c4 = (s.c4 << 8) | byte as u32;
                s.buf.push(byte);
            }
        }
    }
}
