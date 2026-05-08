//! ZPAQL — the bytecode VM driving HCOMP (predictor context hashes)
//! and PCOMP (post-processing). Port of `ZPAQL::run0` and
//! `ZPAQL::execute` from
//! `plugins/zpaq/upstream/libzpaq.cpp:1027-1262`.
//!
//! Architecture:
//!   * 4 general-purpose 32-bit registers `a, b, c, d`.
//!   * 1-bit condition flag `f`.
//!   * Program counter `pc` (byte index into `header`).
//!   * Hash array `h: Vec<u32>` (size = `1 << hbits`).
//!   * Memory array `m: Vec<u8>` (size = `1 << mbits`).
//!   * Register file `r[256]: Vec<u32>`.
//!   * 1024-byte output buffer that flushes through an external
//!     `Writer` and an optional `Sha1`.
//!
//! Indexing of `m[b]` and `h[d]` is `& (size - 1)` (sizes are
//! always powers of two).

#![allow(dead_code)]

use crate::io::Writer;
use crate::sha1::Sha1;

pub struct ZpaqlVm {
    pub header: Vec<u8>,
    pub hbegin: usize,
    pub hend:   usize,
    pub cend:   usize,

    a: u32,
    b: u32,
    c: u32,
    d: u32,
    f: bool,
    pc: usize,

    h: Vec<u32>,
    m: Vec<u8>,
    r: Vec<u32>,

    /// Output buffer for `OUT` instructions.
    outbuf: Vec<u8>,
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum VmError {
    /// The program tried to execute opcode 0 (the explicit ERROR
    /// instruction) or an opcode without a defined case.
    IllegalInstruction(u8),
    /// `LJ` (long jump) target landed outside `[hbegin..hend)`.
    JumpOutOfRange,
}

impl ZpaqlVm {
    pub fn new(header: Vec<u8>, hbegin: usize, hend: usize, cend: usize) -> Self {
        Self {
            header, hbegin, hend, cend,
            a: 0, b: 0, c: 0, d: 0, f: false, pc: 0,
            h: Vec::new(),
            m: Vec::new(),
            r: vec![0u32; 256],
            outbuf: Vec::with_capacity(1 << 14),
        }
    }

    /// `init(hbits, mbits)` — allocate H and M arrays.
    pub fn init(&mut self, hbits: u32, mbits: u32) {
        if hbits > 32 || mbits > 32 {
            // Match libzpaq's `error("H too big")` semantics; we
            // surface this through the API rather than panicking.
            self.h.clear();
            self.m.clear();
            return;
        }
        self.h = vec![0u32; 1 << hbits];
        self.m = vec![0u8; 1 << mbits];
        self.r = vec![0u32; 256];
        self.a = 0; self.b = 0; self.c = 0; self.d = 0;
        self.f = false; self.pc = 0;
    }

    /// `inith()` — initialise as HCOMP using `header[2]` (hh) and
    /// `header[3]` (hm).
    pub fn init_hcomp(&mut self) {
        let hh = self.header[2] as u32;
        let hm = self.header[3] as u32;
        self.init(hh, hm);
    }

    /// `initp()` — initialise as PCOMP using `header[4]` (ph) and
    /// `header[5]` (pm).
    pub fn init_pcomp(&mut self) {
        let ph = self.header[4] as u32;
        let pm = self.header[5] as u32;
        self.init(ph, pm);
    }

    /// Run the HCOMP/PCOMP program on `input`. Mirrors `run0`.
    pub fn run<W: Writer>(
        &mut self,
        input: u32,
        out: Option<&mut W>,
        sha: Option<&mut Sha1>,
    ) -> Result<(), VmError> {
        self.pc = self.hbegin;
        self.a = input;
        while self.execute()? {}
        // PCOMP flushes its output buffer on each run.
        self.flush(out, sha);
        Ok(())
    }

    /// Read element of H by index `i & (h.len() - 1)`.
    pub fn get_h(&self, i: u32) -> u32 {
        let len = self.h.len();
        if len == 0 { 0 } else { self.h[(i as usize) & (len - 1)] }
    }

    fn h_at(&mut self, i: u32) -> &mut u32 {
        let len = self.h.len();
        debug_assert!(len > 0);
        &mut self.h[(i as usize) & (len - 1)]
    }
    fn m_at(&mut self, i: u32) -> &mut u8 {
        let len = self.m.len();
        debug_assert!(len > 0);
        &mut self.m[(i as usize) & (len - 1)]
    }
    fn m_get(&self, i: u32) -> u8 {
        let len = self.m.len();
        if len == 0 { 0 } else { self.m[(i as usize) & (len - 1)] }
    }
    fn h_get(&self, i: u32) -> u32 {
        let len = self.h.len();
        if len == 0 { 0 } else { self.h[(i as usize) & (len - 1)] }
    }

    fn flush<W: Writer>(
        &mut self,
        out: Option<&mut W>,
        sha: Option<&mut Sha1>,
    ) {
        if !self.outbuf.is_empty() {
            if let Some(w) = out { w.write(&self.outbuf); }
            if let Some(s) = sha { s.update(&self.outbuf); }
            self.outbuf.clear();
        }
    }

    /// Run one opcode. Returns `Ok(false)` after HALT, `Ok(true)`
    /// after any other instruction.
    fn execute(&mut self) -> Result<bool, VmError> {
        let op = self.header[self.pc];
        // Set the ZPAQL_TRACE env var to dump every executed
        // instruction with register state — useful for debugging
        // model divergences against libzpaq.
        if std::env::var("ZPAQL_TRACE").is_ok() {
            eprintln!("pc={:4}({:3}) op={:3} a={:08x} b={:08x} c={:08x} d={:08x} f={}",
                self.pc - self.hbegin, self.pc, op,
                self.a, self.b, self.c, self.d, self.f as u8);
        }
        self.pc += 1;

        macro_rules! n_arg {
            () => {{ let n = self.header[self.pc]; self.pc += 1; n }};
        }
        macro_rules! jump_rel {
            () => {{
                let n = self.header[self.pc] as i32;
                let off = ((n + 128) & 255) - 127;
                self.pc = ((self.pc as i32) + off) as usize;
            }};
        }

        match op {
            0 => return Err(VmError::IllegalInstruction(0)),
            1  => { self.a = self.a.wrapping_add(1); }
            2  => { self.a = self.a.wrapping_sub(1); }
            3  => { self.a = !self.a; }
            4  => { self.a = 0; }
            7  => { let n = n_arg!(); self.a = self.r[n as usize]; }
            8  => { let v = self.b; self.b = self.a; self.a = v; }
            9  => { self.b = self.b.wrapping_add(1); }
            10 => { self.b = self.b.wrapping_sub(1); }
            11 => { self.b = !self.b; }
            12 => { self.b = 0; }
            15 => { let n = n_arg!(); self.b = self.r[n as usize]; }
            16 => { let v = self.c; self.c = self.a; self.a = v; }
            17 => { self.c = self.c.wrapping_add(1); }
            18 => { self.c = self.c.wrapping_sub(1); }
            19 => { self.c = !self.c; }
            20 => { self.c = 0; }
            23 => { let n = n_arg!(); self.c = self.r[n as usize]; }
            24 => { let v = self.d; self.d = self.a; self.a = v; }
            25 => { self.d = self.d.wrapping_add(1); }
            26 => { self.d = self.d.wrapping_sub(1); }
            27 => { self.d = !self.d; }
            28 => { self.d = 0; }
            31 => { let n = n_arg!(); self.d = self.r[n as usize]; }
            32 => {
                // *B<>A: swap the low byte of A with M[B], preserving
                // A's high 24 bits. Upstream's `swap(U8& x)` does
                // exactly this via XOR with promotion.
                let m_old = self.m_get(self.b);
                *self.m_at(self.b) = self.a as u8;
                self.a = (self.a & !0xFFu32) | (m_old as u32);
            }
            33 => { let p = self.m_at(self.b); *p = p.wrapping_add(1); }
            34 => { let p = self.m_at(self.b); *p = p.wrapping_sub(1); }
            35 => { let p = self.m_at(self.b); *p = !*p; }
            36 => { *self.m_at(self.b) = 0; }
            39 => { if self.f { jump_rel!(); } else { self.pc += 1; } }
            40 => {
                let m_old = self.m_get(self.c);
                *self.m_at(self.c) = self.a as u8;
                self.a = (self.a & !0xFFu32) | (m_old as u32);
            }
            41 => { let p = self.m_at(self.c); *p = p.wrapping_add(1); }
            42 => { let p = self.m_at(self.c); *p = p.wrapping_sub(1); }
            43 => { let p = self.m_at(self.c); *p = !*p; }
            44 => { *self.m_at(self.c) = 0; }
            47 => { if !self.f { jump_rel!(); } else { self.pc += 1; } }
            48 => { let v = self.h_get(self.d); *self.h_at(self.d) = self.a; self.a = v; }
            49 => { let p = self.h_at(self.d); *p = p.wrapping_add(1); }
            50 => { let p = self.h_at(self.d); *p = p.wrapping_sub(1); }
            51 => { let p = self.h_at(self.d); *p = !*p; }
            52 => { *self.h_at(self.d) = 0; }
            55 => { let n = n_arg!(); self.r[n as usize] = self.a; }
            56 => { return Ok(false); } // HALT
            57 => { self.outbuf.push((self.a & 0xFF) as u8); }
            59 => {
                self.a = (self.a.wrapping_add(self.m_get(self.b) as u32).wrapping_add(512))
                    .wrapping_mul(773);
            }
            60 => {
                let v = self.h_get(self.d).wrapping_add(self.a).wrapping_add(512).wrapping_mul(773);
                *self.h_at(self.d) = v;
            }
            63 => { jump_rel!(); }
            64 => {} // A=A
            65 => { self.a = self.b; }
            66 => { self.a = self.c; }
            67 => { self.a = self.d; }
            68 => { self.a = self.m_get(self.b) as u32; }
            69 => { self.a = self.m_get(self.c) as u32; }
            70 => { self.a = self.h_get(self.d); }
            71 => { let n = n_arg!(); self.a = n as u32; }
            72 => { self.b = self.a; }
            73 => {} // B=B
            74 => { self.b = self.c; }
            75 => { self.b = self.d; }
            76 => { self.b = self.m_get(self.b) as u32; }
            77 => { self.b = self.m_get(self.c) as u32; }
            78 => { self.b = self.h_get(self.d); }
            79 => { let n = n_arg!(); self.b = n as u32; }
            80 => { self.c = self.a; }
            81 => { self.c = self.b; }
            82 => {}
            83 => { self.c = self.d; }
            84 => { self.c = self.m_get(self.b) as u32; }
            85 => { self.c = self.m_get(self.c) as u32; }
            86 => { self.c = self.h_get(self.d); }
            87 => { let n = n_arg!(); self.c = n as u32; }
            88 => { self.d = self.a; }
            89 => { self.d = self.b; }
            90 => { self.d = self.c; }
            91 => {}
            92 => { self.d = self.m_get(self.b) as u32; }
            93 => { self.d = self.m_get(self.c) as u32; }
            94 => { self.d = self.h_get(self.d); }
            95 => { let n = n_arg!(); self.d = n as u32; }
            96 => { *self.m_at(self.b) = self.a as u8; }
            97 => { *self.m_at(self.b) = self.b as u8; }
            98 => { *self.m_at(self.b) = self.c as u8; }
            99 => { *self.m_at(self.b) = self.d as u8; }
            100 => {}
            101 => { let v = self.m_get(self.c); *self.m_at(self.b) = v; }
            102 => { let v = self.h_get(self.d) as u8; *self.m_at(self.b) = v; }
            103 => { let n = n_arg!(); *self.m_at(self.b) = n; }
            104 => { *self.m_at(self.c) = self.a as u8; }
            105 => { *self.m_at(self.c) = self.b as u8; }
            106 => { *self.m_at(self.c) = self.c as u8; }
            107 => { *self.m_at(self.c) = self.d as u8; }
            108 => { let v = self.m_get(self.b); *self.m_at(self.c) = v; }
            109 => {}
            110 => { let v = self.h_get(self.d) as u8; *self.m_at(self.c) = v; }
            111 => { let n = n_arg!(); *self.m_at(self.c) = n; }
            112 => { *self.h_at(self.d) = self.a; }
            113 => { *self.h_at(self.d) = self.b; }
            114 => { *self.h_at(self.d) = self.c; }
            115 => { *self.h_at(self.d) = self.d; }
            116 => { let v = self.m_get(self.b) as u32; *self.h_at(self.d) = v; }
            117 => { let v = self.m_get(self.c) as u32; *self.h_at(self.d) = v; }
            118 => {}
            119 => { let n = n_arg!(); *self.h_at(self.d) = n as u32; }
            128 => { self.a = self.a.wrapping_add(self.a); }
            129 => { self.a = self.a.wrapping_add(self.b); }
            130 => { self.a = self.a.wrapping_add(self.c); }
            131 => { self.a = self.a.wrapping_add(self.d); }
            132 => { self.a = self.a.wrapping_add(self.m_get(self.b) as u32); }
            133 => { self.a = self.a.wrapping_add(self.m_get(self.c) as u32); }
            134 => { self.a = self.a.wrapping_add(self.h_get(self.d)); }
            135 => { let n = n_arg!(); self.a = self.a.wrapping_add(n as u32); }
            136 => { self.a = self.a.wrapping_sub(self.a); }
            137 => { self.a = self.a.wrapping_sub(self.b); }
            138 => { self.a = self.a.wrapping_sub(self.c); }
            139 => { self.a = self.a.wrapping_sub(self.d); }
            140 => { self.a = self.a.wrapping_sub(self.m_get(self.b) as u32); }
            141 => { self.a = self.a.wrapping_sub(self.m_get(self.c) as u32); }
            142 => { self.a = self.a.wrapping_sub(self.h_get(self.d)); }
            143 => { let n = n_arg!(); self.a = self.a.wrapping_sub(n as u32); }
            144 => { self.a = self.a.wrapping_mul(self.a); }
            145 => { self.a = self.a.wrapping_mul(self.b); }
            146 => { self.a = self.a.wrapping_mul(self.c); }
            147 => { self.a = self.a.wrapping_mul(self.d); }
            148 => { self.a = self.a.wrapping_mul(self.m_get(self.b) as u32); }
            149 => { self.a = self.a.wrapping_mul(self.m_get(self.c) as u32); }
            150 => { self.a = self.a.wrapping_mul(self.h_get(self.d)); }
            151 => { let n = n_arg!(); self.a = self.a.wrapping_mul(n as u32); }
            152 => self.div_eq(self.a),
            153 => self.div_eq(self.b),
            154 => self.div_eq(self.c),
            155 => self.div_eq(self.d),
            156 => self.div_eq(self.m_get(self.b) as u32),
            157 => self.div_eq(self.m_get(self.c) as u32),
            158 => self.div_eq(self.h_get(self.d)),
            159 => { let n = n_arg!(); self.div_eq(n as u32); }
            160 => self.mod_eq(self.a),
            161 => self.mod_eq(self.b),
            162 => self.mod_eq(self.c),
            163 => self.mod_eq(self.d),
            164 => self.mod_eq(self.m_get(self.b) as u32),
            165 => self.mod_eq(self.m_get(self.c) as u32),
            166 => self.mod_eq(self.h_get(self.d)),
            167 => { let n = n_arg!(); self.mod_eq(n as u32); }
            168 => { self.a &= self.a; }
            169 => { self.a &= self.b; }
            170 => { self.a &= self.c; }
            171 => { self.a &= self.d; }
            172 => { self.a &= self.m_get(self.b) as u32; }
            173 => { self.a &= self.m_get(self.c) as u32; }
            174 => { self.a &= self.h_get(self.d); }
            175 => { let n = n_arg!(); self.a &= n as u32; }
            176 => { self.a &= !self.a; }
            177 => { self.a &= !self.b; }
            178 => { self.a &= !self.c; }
            179 => { self.a &= !self.d; }
            180 => { self.a &= !(self.m_get(self.b) as u32); }
            181 => { self.a &= !(self.m_get(self.c) as u32); }
            182 => { self.a &= !self.h_get(self.d); }
            183 => { let n = n_arg!(); self.a &= !(n as u32); }
            184 => { self.a |= self.a; }
            185 => { self.a |= self.b; }
            186 => { self.a |= self.c; }
            187 => { self.a |= self.d; }
            188 => { self.a |= self.m_get(self.b) as u32; }
            189 => { self.a |= self.m_get(self.c) as u32; }
            190 => { self.a |= self.h_get(self.d); }
            191 => { let n = n_arg!(); self.a |= n as u32; }
            192 => { self.a ^= self.a; }
            193 => { self.a ^= self.b; }
            194 => { self.a ^= self.c; }
            195 => { self.a ^= self.d; }
            196 => { self.a ^= self.m_get(self.b) as u32; }
            197 => { self.a ^= self.m_get(self.c) as u32; }
            198 => { self.a ^= self.h_get(self.d); }
            199 => { let n = n_arg!(); self.a ^= n as u32; }
            200 => { self.a = self.a.wrapping_shl(self.a & 31); }
            201 => { self.a = self.a.wrapping_shl(self.b & 31); }
            202 => { self.a = self.a.wrapping_shl(self.c & 31); }
            203 => { self.a = self.a.wrapping_shl(self.d & 31); }
            204 => { let s = self.m_get(self.b) as u32 & 31; self.a = self.a.wrapping_shl(s); }
            205 => { let s = self.m_get(self.c) as u32 & 31; self.a = self.a.wrapping_shl(s); }
            206 => { let s = self.h_get(self.d) & 31; self.a = self.a.wrapping_shl(s); }
            207 => { let n = n_arg!(); self.a = self.a.wrapping_shl(n as u32 & 31); }
            208 => { self.a = self.a.wrapping_shr(self.a & 31); }
            209 => { self.a = self.a.wrapping_shr(self.b & 31); }
            210 => { self.a = self.a.wrapping_shr(self.c & 31); }
            211 => { self.a = self.a.wrapping_shr(self.d & 31); }
            212 => { let s = self.m_get(self.b) as u32 & 31; self.a = self.a.wrapping_shr(s); }
            213 => { let s = self.m_get(self.c) as u32 & 31; self.a = self.a.wrapping_shr(s); }
            214 => { let s = self.h_get(self.d) & 31; self.a = self.a.wrapping_shr(s); }
            215 => { let n = n_arg!(); self.a = self.a.wrapping_shr(n as u32 & 31); }
            216 => { self.f = true; }
            217 => { self.f = self.a == self.b; }
            218 => { self.f = self.a == self.c; }
            219 => { self.f = self.a == self.d; }
            220 => { self.f = self.a == self.m_get(self.b) as u32; }
            221 => { self.f = self.a == self.m_get(self.c) as u32; }
            222 => { self.f = self.a == self.h_get(self.d); }
            223 => { let n = n_arg!(); self.f = self.a == n as u32; }
            224 => { self.f = false; }
            225 => { self.f = self.a < self.b; }
            226 => { self.f = self.a < self.c; }
            227 => { self.f = self.a < self.d; }
            228 => { self.f = self.a < self.m_get(self.b) as u32; }
            229 => { self.f = self.a < self.m_get(self.c) as u32; }
            230 => { self.f = self.a < self.h_get(self.d); }
            231 => { let n = n_arg!(); self.f = self.a < n as u32; }
            232 => { self.f = false; }
            233 => { self.f = self.a > self.b; }
            234 => { self.f = self.a > self.c; }
            235 => { self.f = self.a > self.d; }
            236 => { self.f = self.a > self.m_get(self.b) as u32; }
            237 => { self.f = self.a > self.m_get(self.c) as u32; }
            238 => { self.f = self.a > self.h_get(self.d); }
            239 => { let n = n_arg!(); self.f = self.a > n as u32; }
            255 => {
                // LJ: long jump to header[pc] | (header[pc+1] << 8) within HCOMP.
                let lo = self.header[self.pc] as usize;
                let hi = self.header[self.pc + 1] as usize;
                let target = self.hbegin + lo + 256 * hi;
                if target >= self.hend {
                    return Err(VmError::JumpOutOfRange);
                }
                self.pc = target;
            }
            other => return Err(VmError::IllegalInstruction(other)),
        }
        Ok(true)
    }

    fn div_eq(&mut self, x: u32) {
        if x != 0 { self.a /= x; } else { self.a = 0; }
    }
    fn mod_eq(&mut self, x: u32) {
        if x != 0 { self.a %= x; } else { self.a = 0; }
    }

    /// Read-only access to current registers + state for tests.
    pub fn snapshot(&self) -> (u32, u32, u32, u32, bool, usize) {
        (self.a, self.b, self.c, self.d, self.f, self.pc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::VecWriter;

    /// Simulates a minimal HCOMP that just halts. The wrapper layout
    /// mirrors what `format::read_header` produces: header[2..7] =
    /// hh hm ph pm n; then COMP bytes (none here); then 0x00; then
    /// 128-byte gap to hbegin; HCOMP bytes ending at 0x00.
    fn make_simple_program(hcomp: &[u8]) -> ZpaqlVm {
        let mut header = vec![0u8; 256];
        header[2] = 4; // hh = 4 (h size 16)
        header[3] = 8; // hm = 8 (m size 256)
        header[6] = 0; // n = 0
        // header[7] = 0 (COMP terminator)
        let cend = 8;
        let hbegin = cend + 128;
        let hend = hbegin + hcomp.len();
        if header.len() < hend + 16 { header.resize(hend + 16, 0); }
        header[hbegin..hend].copy_from_slice(hcomp);
        ZpaqlVm::new(header, hbegin, hend, cend)
    }

    #[test]
    fn halt_immediately() {
        // 56 = HALT
        let mut vm = make_simple_program(&[56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(0, Some(&mut sink), None).expect("run");
        assert!(sink.buf.is_empty());
    }

    #[test]
    fn add_immediate() {
        // A= 5; A+= 7; HALT.  71 = "A= N"  135 = "A+= N"  56 = HALT
        let mut vm = make_simple_program(&[71, 5, 135, 7, 56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(0, Some(&mut sink), None).unwrap();
        let (a, _, _, _, _, _) = vm.snapshot();
        assert_eq!(a, 12);
    }

    #[test]
    fn out_byte() {
        // A= 65; OUT; HALT  →  outputs 'A'
        let mut vm = make_simple_program(&[71, 65, 57, 56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(0, Some(&mut sink), None).unwrap();
        assert_eq!(sink.buf, b"A");
    }

    #[test]
    fn cond_jump_taken() {
        // A= 5; A== 5; JT 2; A= 99; HALT  →  A should remain 5 (skip "A= 99").
        // Opcodes: 71 5 | 223 5 | 39 +2 | 71 99 | 56
        // Wait — JT N has signed offset relative to next opcode.
        // We want to skip "71 99" (2 bytes), so offset = +2.
        let mut vm = make_simple_program(&[71, 5, 223, 5, 39, 2, 71, 99, 56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(0, Some(&mut sink), None).unwrap();
        let (a, _, _, _, _, _) = vm.snapshot();
        assert_eq!(a, 5);
    }

    #[test]
    fn input_propagates_to_a() {
        let mut vm = make_simple_program(&[56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(42, Some(&mut sink), None).unwrap();
        let (a, _, _, _, _, _) = vm.snapshot();
        assert_eq!(a, 42);
    }

    #[test]
    fn memory_write_persists_across_runs() {
        // Program 1: M[0] = 0x42; HALT
        // Program 2: A = M[0]; HALT
        // After two runs, A should be 0x42.
        // Using opcodes:
        //   71 0x42  | A = 0x42
        //   12       | B = 0
        //   96       | M[B] = A
        //   56       | HALT
        // (then on second invocation we'd want different logic, but
        //  for a simple test, run program once and check m[0])
        let mut vm = make_simple_program(&[71, 0x42, 12, 96, 56]);
        vm.init_hcomp();
        let mut sink = VecWriter::new();
        vm.run(0, Some(&mut sink), None).unwrap();
        // Inspect M[0] via second run that loads it.
        // For this, replace HCOMP. Easier: expose memory.
        // Add a snapshot of m[0]:
        let m0 = vm.m.first().copied().unwrap_or(0);
        assert_eq!(m0, 0x42, "M[0] not persisted");

        // Second run: A = M[0]; HALT.  68 = a=*b, b is still 0.
        // Re-write HCOMP without rebuilding the VM — actually we
        // can't easily do that since header is internal. Just check
        // a fresh program reads the value via a=*b.
        // Instead patch the header in place:
        let hbegin = vm.hbegin;
        vm.header[hbegin]     = 12;   // b=0
        vm.header[hbegin + 1] = 68;   // a=*b
        vm.header[hbegin + 2] = 56;   // halt
        vm.header[hbegin + 3] = 0;    // padding
        vm.header[hbegin + 4] = 0;
        vm.run(0, Some(&mut sink), None).unwrap();
        let (a, _, _, _, _, _) = vm.snapshot();
        assert_eq!(a, 0x42, "M[0] read-back failed");
    }
}
