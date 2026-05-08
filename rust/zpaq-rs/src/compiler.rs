//! ZPAQ config-string compiler. Mirrors `Compiler::*` and the
//! `opcodelist[272]` table from
//! `plugins/zpaq/upstream/libzpaq.cpp:2449-2770`.
//!
//! Input: a libzpaq config-string like
//!
//! ```text
//! comp 1 2 0 0 2
//!   0 icm 16
//!   1 isse 19 0
//! hcomp
//!   ... bytecode ...
//! post 0 end
//! ```
//!
//! Output: a [`CompiledConfig`] containing the wire-format header
//! that [`crate::compress::Compresser::start_block_modeled`]
//! expects, plus the optional PCOMP bytecode for archives that
//! declare a `pcomp ... ; ... end` post-processor.
//!
//! What's intentionally not ported:
//!   * `$N+M` argument substitution (`makeConfig` rewrites them
//!     before we see the string in upstream — callers should do
//!     the same).
//!   * The `pcomp_cmd` writer hook: callers that need the optional
//!     pcomp command string can grab it from the parser via
//!     [`CompiledConfig::pcomp_cmd`].

#![allow(dead_code)]

use crate::format::COMPSIZE;

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum CompileError {
    UnexpectedEnd,
    UnexpectedToken(String),
    NumberOutOfRange { value: i32, low: i32, high: i32 },
    NotANumber(String),
    InvalidComponent(String),
    ProgramTooLarge,
    JumpTooFar,
    UnbalancedIf,
    UnbalancedDo,
}

/// A successfully-parsed config string, ready to be handed to
/// [`crate::compress::Compresser::start_block_modeled`].
#[derive(Debug, Clone)]
pub struct CompiledConfig {
    /// Wire header bytes (`hsize_lo, hsize_hi, hh, hm, ph, pm, n,
    /// COMP..., 0_term, HCOMP..., 0_term`). Drop straight into
    /// `start_block_modeled`.
    pub header: Vec<u8>,
    /// PCOMP program bytecode (without the trailing 0 terminator) if
    /// the config declared a `pcomp ... ; ... end` block. Pass to
    /// [`crate::compress::Compresser::post_process_prog`].
    pub pcomp: Option<Vec<u8>>,
    /// Optional `pcomp_cmd` string captured between `pcomp` and `;`.
    /// upstream forwards this to a `Writer*`; we just keep the raw
    /// bytes for callers that need them.
    pub pcomp_cmd: Vec<u8>,
}

/// Component-name keywords (upstream `compname[]`).
const COMPNAME: &[&str] = &[
    "", "const", "cm", "icm", "match", "avg", "mix2", "mix", "isse", "sse",
];

// Symbolic constants — match upstream `Compiler::CompType`.
const JT:        i32 = 39;
const JF:        i32 = 47;
const JMP:       i32 = 63;
const LJ:        i32 = 255;
const POST:      i32 = 256;
const PCOMP:     i32 = 257;
const END:       i32 = 258;
const IF_OP:     i32 = 259;
const IFNOT_OP:  i32 = 260;
const ELSE_OP:   i32 = 261;
const ENDIF_OP:  i32 = 262;
const DO_OP:     i32 = 263;
const WHILE_OP:  i32 = 264;
const UNTIL_OP:  i32 = 265;
const FOREVER_OP:i32 = 266;
const IFL_OP:    i32 = 267;
const IFNOTL_OP: i32 = 268;
const ELSEL_OP:  i32 = 269;
const SEMICOLON: i32 = 270;

/// `opcodelist[272]` — index = encoded byte (or pseudo-opcode for
/// 256+). Empty strings are unused slots; `next()`/`rtoken` skip
/// them via the `*list[i]` length check, but we keep the layout
/// 1:1 with upstream so that operand emission stays correct.
const OPCODELIST: &[&str] = &[
    // 0..7
    "error","a++","a--","a!","a=0","","","a=r",
    // 8..15
    "b<>a","b++","b--","b!","b=0","","","b=r",
    // 16..23
    "c<>a","c++","c--","c!","c=0","","","c=r",
    // 24..31
    "d<>a","d++","d--","d!","d=0","","","d=r",
    // 32..39
    "*b<>a","*b++","*b--","*b!","*b=0","","","jt",
    // 40..47
    "*c<>a","*c++","*c--","*c!","*c=0","","","jf",
    // 48..55
    "*d<>a","*d++","*d--","*d!","*d=0","","","r=a",
    // 56..63
    "halt","out","","hash","hashd","","","jmp",
    // 64..71
    "a=a","a=b","a=c","a=d","a=*b","a=*c","a=*d","a=",
    // 72..79
    "b=a","b=b","b=c","b=d","b=*b","b=*c","b=*d","b=",
    // 80..87
    "c=a","c=b","c=c","c=d","c=*b","c=*c","c=*d","c=",
    // 88..95
    "d=a","d=b","d=c","d=d","d=*b","d=*c","d=*d","d=",
    // 96..103
    "*b=a","*b=b","*b=c","*b=d","*b=*b","*b=*c","*b=*d","*b=",
    // 104..111
    "*c=a","*c=b","*c=c","*c=d","*c=*b","*c=*c","*c=*d","*c=",
    // 112..119
    "*d=a","*d=b","*d=c","*d=d","*d=*b","*d=*c","*d=*d","*d=",
    // 120..127  (unused)
    "","","","","","","","",
    // 128..135  a+=
    "a+=a","a+=b","a+=c","a+=d","a+=*b","a+=*c","a+=*d","a+=",
    // 136..143  a-=
    "a-=a","a-=b","a-=c","a-=d","a-=*b","a-=*c","a-=*d","a-=",
    // 144..151  a*=
    "a*=a","a*=b","a*=c","a*=d","a*=*b","a*=*c","a*=*d","a*=",
    // 152..159  a/=
    "a/=a","a/=b","a/=c","a/=d","a/=*b","a/=*c","a/=*d","a/=",
    // 160..167  a%=
    "a%=a","a%=b","a%=c","a%=d","a%=*b","a%=*c","a%=*d","a%=",
    // 168..175  a&=
    "a&=a","a&=b","a&=c","a&=d","a&=*b","a&=*c","a&=*d","a&=",
    // 176..183  a&~
    "a&~a","a&~b","a&~c","a&~d","a&~*b","a&~*c","a&~*d","a&~",
    // 184..191  a|=
    "a|=a","a|=b","a|=c","a|=d","a|=*b","a|=*c","a|=*d","a|=",
    // 192..199  a^=
    "a^=a","a^=b","a^=c","a^=d","a^=*b","a^=*c","a^=*d","a^=",
    // 200..207  a<<=
    "a<<=a","a<<=b","a<<=c","a<<=d","a<<=*b","a<<=*c","a<<=*d","a<<=",
    // 208..215  a>>=
    "a>>=a","a>>=b","a>>=c","a>>=d","a>>=*b","a>>=*c","a>>=*d","a>>=",
    // 216..223  a==
    "a==a","a==b","a==c","a==d","a==*b","a==*c","a==*d","a==",
    // 224..231  a<
    "a<a","a<b","a<c","a<d","a<*b","a<*c","a<*d","a<",
    // 232..239  a>
    "a>a","a>b","a>c","a>d","a>*b","a>*c","a>*d","a>",
    // 240..247 unused
    "","","","","","","","",
    // 248..255 (LJ at 255)
    "","","","","","","","lj",
    // 256.. pseudo-opcodes (IF/ELSE/etc)
    "post","pcomp","end","if","ifnot","else","endif","do",
    "while","until","forever","ifl","ifnotl","elsel",";",
];

/// Streaming tokeniser — mirrors upstream `Compiler::next` /
/// `matchToken` / `rtoken`. Keeps a byte slice and current cursor.
struct Tokenizer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src: src.as_bytes(), pos: 0, line: 1 }
    }

    /// Advance to the start of the next token, skipping whitespace
    /// and `(...nested...)` comments. Returns Err at EOF.
    fn next_token_start(&mut self) -> Result<(), CompileError> {
        let mut state: i32 = 0; // 0 = whitespace, >0 = paren nest depth
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if c == b'\n' { self.line += 1; }
            if c == b'(' {
                state += 1;
                self.pos += 1;
                continue;
            }
            if state > 0 && c == b')' {
                state -= 1;
                self.pos += 1;
                continue;
            }
            if state == 0 && c > b' ' {
                return Ok(());
            }
            self.pos += 1;
        }
        Err(CompileError::UnexpectedEnd)
    }

    /// Length of the current word (terminates at whitespace or `(`).
    fn word_len(&self) -> usize {
        let mut k = 0;
        while self.pos + k < self.src.len() {
            let c = self.src[self.pos + k];
            if c <= b' ' || c == b'(' { break; }
            k += 1;
        }
        k
    }

    /// True if the current word matches `tok` case-insensitively.
    fn match_token(&self, tok: &str) -> bool {
        let len = self.word_len();
        if len != tok.len() { return false; }
        let bytes = tok.as_bytes();
        for i in 0..len {
            let a = self.src[self.pos + i];
            let b = bytes[i];
            let al = if a >= b'A' && a <= b'Z' { a + 32 } else { a };
            let bl = if b >= b'A' && b <= b'Z' { b + 32 } else { b };
            if al != bl { return false; }
        }
        true
    }

    /// Read a token, consuming it. Returns the token slice as a
    /// `&str` borrowed from the source.
    fn read_word(&mut self) -> Result<&'a str, CompileError> {
        self.next_token_start()?;
        let len = self.word_len();
        let start = self.pos;
        self.pos += len;
        // Safety: we scanned bytes >' ' so a UTF-8 word is fine.
        Ok(std::str::from_utf8(&self.src[start..start + len])
            .unwrap_or(""))
    }

    /// Match the next token against a list and return its index.
    /// Empty entries in the list are skipped (per upstream).
    fn rtoken_list(&mut self, list: &[&str]) -> Result<usize, CompileError> {
        self.next_token_start()?;
        for (i, &word) in list.iter().enumerate() {
            if !word.is_empty() && self.match_token(word) {
                self.pos += word.len();
                return Ok(i);
            }
        }
        let bad = self.read_word().unwrap_or("").to_string();
        Err(CompileError::UnexpectedToken(bad))
    }

    /// Match a specific keyword.
    fn rtoken_lit(&mut self, expected: &str) -> Result<(), CompileError> {
        self.next_token_start()?;
        if self.match_token(expected) {
            self.pos += expected.len();
            Ok(())
        } else {
            let bad = self.read_word().unwrap_or("").to_string();
            Err(CompileError::UnexpectedToken(bad))
        }
    }

    /// Match a number in `[low, high]`.
    fn rtoken_num(&mut self, low: i32, high: i32) -> Result<i32, CompileError> {
        let word = self.read_word()?;
        let n: i32 = match word.parse() {
            Ok(v) => v,
            Err(_) => return Err(CompileError::NotANumber(word.to_string())),
        };
        if n < low || n > high {
            return Err(CompileError::NumberOutOfRange {
                value: n, low, high,
            });
        }
        Ok(n)
    }
}

/// Compile a config string. The output `header` is exactly what
/// `Compresser::start_block_modeled` wants. PCOMP bytecode and any
/// `pcomp_cmd` text are returned alongside.
pub fn compile(config: &str) -> Result<CompiledConfig, CompileError> {
    let mut t = Tokenizer::new(config);

    // --- COMP section ---------------------------------------------------
    t.rtoken_lit("comp")?;
    let hh = t.rtoken_num(0, 255)? as u8;
    let hm = t.rtoken_num(0, 255)? as u8;
    let ph = t.rtoken_num(0, 255)? as u8;
    let pm = t.rtoken_num(0, 255)? as u8;
    let n  = t.rtoken_num(0, 255)? as u8;

    let mut comp_bytes: Vec<u8> = Vec::new();
    for i in 0..n {
        let _ = t.rtoken_num(i as i32, i as i32)?;
        let ty = t.rtoken_list(COMPNAME)? as u8;
        comp_bytes.push(ty);
        let clen = COMPSIZE[ty as usize] as usize;
        if !(1..=10).contains(&clen) {
            return Err(CompileError::InvalidComponent(format!("type {}", ty)));
        }
        for _ in 1..clen {
            comp_bytes.push(t.rtoken_num(0, 255)? as u8);
        }
    }

    // --- HCOMP section --------------------------------------------------
    t.rtoken_lit("hcomp")?;
    let (hcomp, post_op) = compile_comp(&mut t, comp_bytes.len() + 8)?;
    // upstream: `hsize = cend - 2 + hend - hbegin` where cend - 2
    // is the count of [hh, hm, ph, pm, n, COMP..., 0_term] (= 5 +
    // comp_bytes.len() + 1) and hend - hbegin = hcomp.len() (which
    // already includes the trailing 0 terminator).
    let hsize = 5 + comp_bytes.len() + 1 + hcomp.len();
    if hsize > 65535 {
        return Err(CompileError::ProgramTooLarge);
    }

    // --- Trailing POST 0 END or PCOMP cmd ; ... END ---------------------
    let (pcomp, pcomp_cmd) = match post_op {
        op if op == END => (None, Vec::new()),
        op if op == POST => {
            t.rtoken_num(0, 0)?;
            t.rtoken_lit("end")?;
            (None, Vec::new())
        }
        op if op == PCOMP => {
            // Capture pcomp_cmd up to ';'.
            let mut cmd = Vec::new();
            t.next_token_start()?;
            while t.pos < t.src.len() && t.src[t.pos] != b';' {
                cmd.push(t.src[t.pos]);
                t.pos += 1;
            }
            if t.pos < t.src.len() { t.pos += 1; } // skip ';'
            // Strip trailing whitespace.
            while cmd.last().map_or(false, |&b| b <= b' ') { cmd.pop(); }

            let (pcomp_bytes, end_op) = compile_comp(&mut t, 8)?;
            if end_op != END {
                return Err(CompileError::UnexpectedToken("expected END".into()));
            }
            (Some(pcomp_bytes), cmd)
        }
        _ => return Err(CompileError::UnexpectedToken(
            "expected POST/PCOMP/END".into())),
    };

    // --- Wire-format header --------------------------------------------
    let mut header = Vec::with_capacity(hsize + 2);
    header.push((hsize & 0xFF) as u8);
    header.push((hsize >> 8) as u8);
    header.push(hh); header.push(hm); header.push(ph); header.push(pm);
    header.push(n);
    header.extend_from_slice(&comp_bytes);
    header.push(0); // COMP terminator
    header.extend_from_slice(&hcomp); // HCOMP already ends in 0 from compile_comp

    Ok(CompiledConfig { header, pcomp, pcomp_cmd })
}

/// Compile the body of an HCOMP or PCOMP block, returning the
/// emitted bytecode (with trailing 0 byte) and the terminator
/// pseudo-opcode (POST/PCOMP/END).
///
/// `comp_begin` is the byte offset that upstream calls "comp_begin"
/// — it's the absolute offset of the first emitted byte in the
/// containing ZpaqlVm header. Long jumps (LJ) are encoded relative
/// to this offset.
fn compile_comp(
    t: &mut Tokenizer<'_>,
    comp_begin: usize,
) -> Result<(Vec<u8>, i32), CompileError> {
    let mut buf: Vec<u8> = Vec::new();
    let mut if_stack: Vec<usize> = Vec::new();
    let mut do_stack: Vec<usize> = Vec::new();

    loop {
        let op = t.rtoken_list(OPCODELIST)? as i32;
        if op == POST || op == PCOMP || op == END {
            buf.push(0);
            return Ok((buf, op));
        }

        let mut emit_op = op;
        let mut operand: Option<u8> = None;
        let mut operand2: Option<u8> = None;

        if op == IF_OP {
            emit_op = JF;
            operand = Some(0);
            // Position of the operand byte we'll patch later.
            if_stack.push(buf.len() + 1);
        } else if op == IFNOT_OP {
            emit_op = JT;
            operand = Some(0);
            if_stack.push(buf.len() + 1);
        } else if op == IFL_OP || op == IFNOTL_OP {
            // Long if: emit a short conditional jump skipping the LJ
            // (3 bytes), then a placeholder LJ.
            buf.push(if op == IFL_OP { JT as u8 } else { JF as u8 });
            buf.push(3);
            emit_op = LJ;
            operand = Some(0);
            operand2 = Some(0);
            if_stack.push(buf.len() + 1);
        } else if op == ELSE_OP || op == ELSEL_OP {
            let new_op = if op == ELSE_OP { JMP } else { LJ };
            emit_op = new_op;
            operand = Some(0);
            if new_op == LJ { operand2 = Some(0); }

            let a = if_stack.pop().ok_or(CompileError::UnbalancedIf)?;
            patch_jump(&mut buf, a, comp_begin, new_op == LJ)?;
            if_stack.push(buf.len() + 1);
        } else if op == ENDIF_OP {
            let a = if_stack.pop().ok_or(CompileError::UnbalancedIf)?;
            patch_endif(&mut buf, a, comp_begin)?;
            // No new bytes for ENDIF — the patched jump landed where
            // the next opcode will be.
            continue;
        } else if op == DO_OP {
            do_stack.push(buf.len());
            continue;
        } else if op == WHILE_OP || op == UNTIL_OP || op == FOREVER_OP {
            let a = do_stack.pop().ok_or(CompileError::UnbalancedDo)?;
            // Distance from end-of-jump back to `a`.
            let distance = (a as i32) - (buf.len() as i32) - 2;
            if distance >= -127 {
                emit_op = match op {
                    WHILE_OP => JT,
                    UNTIL_OP => JF,
                    _        => JMP,
                };
                operand = Some((distance & 0xFF) as u8);
            } else {
                // Long backward jump: emit conditional skip + LJ.
                let dest = (a as i32) - (comp_begin as i32);
                if dest < 0 {
                    return Err(CompileError::JumpTooFar);
                }
                if op == WHILE_OP {
                    buf.push(JF as u8); buf.push(3);
                } else if op == UNTIL_OP {
                    buf.push(JT as u8); buf.push(3);
                }
                emit_op = LJ;
                operand = Some((dest & 0xFF) as u8);
                operand2 = Some(((dest >> 8) & 0xFF) as u8);
            }
        } else if (op & 7) == 7 && op <= 255 {
            // 2-byte operand opcodes: read N from the source.
            if op == LJ {
                let v = t.rtoken_num(0, 65535)?;
                operand = Some((v & 0xFF) as u8);
                operand2 = Some(((v >> 8) & 0xFF) as u8);
            } else if op == JT || op == JF || op == JMP {
                let v = t.rtoken_num(-128, 127)?;
                operand = Some((v & 0xFF) as u8);
            } else {
                operand = Some(t.rtoken_num(0, 255)? as u8);
            }
        }

        if (0..=255).contains(&emit_op) {
            buf.push(emit_op as u8);
        }
        if let Some(b) = operand  { buf.push(b); }
        if let Some(b) = operand2 { buf.push(b); }

        if buf.len() + comp_begin - 8 > 65535 {
            return Err(CompileError::ProgramTooLarge);
        }
    }
}

/// Resolve a forward jump at byte offset `a` (the operand byte) in
/// `buf`. `comp_begin` is needed only for long jumps (LJ encodes an
/// absolute offset from comp_begin).
fn patch_jump(
    buf: &mut [u8],
    a: usize,
    comp_begin: usize,
    new_op_is_lj: bool,
) -> Result<(), CompileError> {
    let here = buf.len();
    let opcode = buf[a - 1];
    if opcode != LJ as u8 {
        // Short conditional jump (JT/JF/JMP).
        let j = (here as i32) - (a as i32) + 1 + (new_op_is_lj as i32);
        if j < 0 || j > 127 {
            return Err(CompileError::JumpTooFar);
        }
        buf[a] = j as u8;
    } else {
        // Long jump (LJ).
        let j = (here as i32) - (comp_begin as i32) + 8 + 2 + (new_op_is_lj as i32);
        if j < 0 {
            return Err(CompileError::JumpTooFar);
        }
        buf[a] = (j & 0xFF) as u8;
        buf[a + 1] = ((j >> 8) & 0xFF) as u8;
    }
    Ok(())
}

/// Resolve the jump emitted by IF/IFNOT/IFL/IFNOTL/ELSE/ELSEL when
/// we hit ENDIF.
fn patch_endif(
    buf: &mut [u8],
    a: usize,
    comp_begin: usize,
) -> Result<(), CompileError> {
    let here = buf.len();
    let opcode = buf[a - 1];
    if opcode != LJ as u8 {
        let j = (here as i32) - (a as i32) - 1;
        if j < 0 || j > 127 {
            return Err(CompileError::JumpTooFar);
        }
        buf[a] = j as u8;
    } else {
        let j = (here as i32) - (comp_begin as i32) + 8;
        if j < 0 {
            return Err(CompileError::JumpTooFar);
        }
        buf[a] = (j & 0xFF) as u8;
        buf[a + 1] = ((j >> 8) & 0xFF) as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MIN_CFG, MID_CFG};

    /// Reverse-engineered textual form of `min.cfg` (the bytes are
    /// `models::MIN_CFG`). When compiled it must reproduce that
    /// blob exactly, byte for byte.
    #[test]
    fn compile_min_cfg_matches_canned() {
        let cfg = r#"
            comp 1 2 0 0 2
              0 icm 16
              1 isse 19 0
            hcomp
              *b=a a=0 d=0 hash b-- hash *d=a d++ b-- hash b-- hash *d=a halt
            post 0 end
        "#;
        let cc = compile(cfg).expect("compile min.cfg");
        assert_eq!(cc.header, MIN_CFG,
            "compiled header doesn't match canned MIN_CFG");
        assert!(cc.pcomp.is_none());
        let _ = MID_CFG;
    }

    #[test]
    fn compile_stored_zero() {
        // Trivial config: no model.
        let cfg = "comp 0 0 0 0 0 hcomp end";
        let cc = compile(cfg).expect("compile zero");
        assert_eq!(cc.header[6], 0); // n=0
        assert!(cc.pcomp.is_none());
    }

    #[test]
    fn compile_parses_components() {
        let cfg = "comp 0 0 0 0 1 0 cm 8 0 hcomp end";
        let cc = compile(cfg).expect("compile cm");
        assert_eq!(cc.header[6], 1); // n=1
        assert_eq!(cc.header[7], 2); // ty=cm (=2)
        assert_eq!(cc.header[8], 8); // sizebits=8
        assert_eq!(cc.header[9], 0); // limit=0
    }

    #[test]
    fn rejects_garbage() {
        assert!(compile("not zpaq").is_err());
        assert!(compile("comp 0 0 0 0 0 hcomp halt halt halt").is_err());
    }

    #[test]
    fn comments_are_skipped() {
        let cfg = "comp 0 0 0 0 0 (comment) hcomp (another) end";
        assert!(compile(cfg).is_ok());
    }

    #[test]
    fn case_insensitive_keywords() {
        let cfg = "COMP 0 0 0 0 0 HCOMP END";
        assert!(compile(cfg).is_ok());
    }
}
