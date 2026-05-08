//! `makeConfig` port — turn a libzpaq method string like
//! `"x4,3,ci1"` into the full textual ZPAQ config that
//! [`crate::compiler::compile`] consumes.
//!
//! Mirrors `std::string makeConfig(const char* method, int args[])`
//! in `plugins/zpaq/upstream/libzpaq.cpp:6887`.
//!
//! Supported component specs (the alphabetic letters that may
//! follow the comma-separated args):
//!   * `c[N1[,N2,...]]` — CM (count mode) or ICM context model.
//!   * `i[N1[,N2,...]]` — ISSE chain.
//!   * `m[N1,N2]`        — MIX over all earlier components.
//!   * `t[N1,N2]`        — MIX2 over the last two components.
//!   * `s[N1,N2,N3]`     — SSE on the previous component.
//!   * `a[N1,N2,N3]`     — MATCH model.
//!   * `w[N1,N2,N3,N4,N5,N6]` — word-context ICM-ISSE chain.
//!
//! Out of scope for this port:
//!   * `s`-prefix method strings (streaming — no archive frame).
//!   * `i`-prefix (incremental — sub-block sizing).
//!   * Digit-method expansion (`"1".."5"` choosing between `"x..."`
//!     variants based on input statistics). Callers who want
//!     `compressBlock`-equivalent behaviour for digit methods
//!     should pre-expand using their own type analysis.

#![allow(dead_code)]

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum MakeConfigError {
    UnsupportedMethod,
    BadComponentSpec(String),
    BadNumber(String),
}

/// Parse `method` into a textual config + the 9-element `args`
/// array that `compile_with_args` expects.
pub fn make_config(method: &str) -> Result<(String, [i32; 9]), MakeConfigError> {
    let bytes = method.as_bytes();
    if bytes.is_empty() {
        return Err(MakeConfigError::UnsupportedMethod);
    }

    let kind = bytes[0];
    if kind != b'x' && kind != b'0' {
        return Err(MakeConfigError::UnsupportedMethod);
    }

    // Read args[0..8] from the comma-separated numeric tail.
    let mut args = [0i32; 9];
    let mut p = 1usize;
    let mut idx = 0usize;
    while p < bytes.len() && idx < 9 {
        let c = bytes[p];
        if c.is_ascii_digit() {
            args[idx] = args[idx] * 10 + (c - b'0') as i32;
            p += 1;
        } else if c == b',' || c == b'.' {
            p += 1;
            idx += 1;
        } else {
            break; // start of component spec
        }
    }
    let comp_specs = &bytes[p..];

    if kind == b'0' {
        return Ok(("comp 0 0 0 0 0 hcomp end\n".to_string(), args));
    }

    let level = args[1] & 3;
    let doe8 = args[1] >= 4 && args[1] <= 7;

    // ---- Build hdr (header up to `n`, which we'll fill in later)
    //      and pcomp (post-processor, fixed per level).
    let (hdr_prefix, pcomp): (&str, String) = match level {
        0 => {
            let pcomp = if doe8 { e8e9_pcomp_only() } else { "end\n".to_string() };
            ("comp 9 16 0 0 ", pcomp)
        }
        1 => {
            let pcomp = lz77_var_pcomp(doe8);
            ("comp 9 16 0 $1+20 ", pcomp)
        }
        2 => {
            let pcomp = lz77_byte_pcomp(doe8);
            ("comp 9 16 0 $1+20 ", pcomp)
        }
        3 => {
            let pcomp = bwt_pcomp(doe8);
            ("comp 9 16 $1+20 $1+20 ", pcomp)
        }
        _ => return Err(MakeConfigError::UnsupportedMethod),
    };

    // ---- Standard HCOMP prefix (records the last byte location).
    let mut hcomp = String::from("hcomp\nc-- *c=a a+= 255 d=a *d=c\n");
    if level == 2 {
        // Level-2 LZ77 needs a state machine that tracks the next
        // expected byte type via R1.
        let skip = 111 + 57 * (doe8 as i32);
        hcomp.push_str(
            "  (decode lz77 into M. Codes:\n\
              00xxxxxx = literal length xxxxxx+1\n\
              xx......, xx > 0 = match with xx offset bytes to follow)\n\
            \n\
              a=r 1 a== 0 if (init)\n\
                a= ");
        hcomp.push_str(&skip.to_string());
        hcomp.push_str(
            " (skip post code)\n\
              else a== 1 if  (new code?)\n\
                a=*c r=a 2  (save code in R2)\n\
                a> 63 if a>>= 6 a++ a++  (match)\n\
                else a++ a++ endif  (literal)\n\
              else (read rest of code)\n\
                a--\n\
              endif endif\n\
              r=a 1  (R1 = 1+expected bytes to next code)\n");
    }

    // ---- Walk component specs.
    let mut comp = String::new();
    let mut ncomp = 0i32;
    let mut sb = 5i32;
    let membits = args[0] + 20;

    let mut i = 0usize;
    while i < comp_specs.len() && ncomp < 254 {
        let kind = comp_specs[i];
        i += 1;
        if !matches!(kind, b'c' | b'i' | b'm' | b't' | b's' | b'a' | b'w' | b'f') {
            return Err(MakeConfigError::BadComponentSpec(
                (kind as char).to_string()));
        }

        // Parse `N1,N2,...` until the next non-numeric, non-comma byte.
        let mut v: Vec<i32> = vec![kind as i32];
        if i < comp_specs.len() && comp_specs[i].is_ascii_digit() {
            v.push((comp_specs[i] - b'0') as i32);
            i += 1;
            while i < comp_specs.len() {
                let c = comp_specs[i];
                if c.is_ascii_digit() {
                    let last = v.len() - 1;
                    v[last] = v[last] * 10 + (c - b'0') as i32;
                    i += 1;
                } else if c == b',' || c == b'.' {
                    v.push(0);
                    i += 1;
                } else {
                    break;
                }
            }
        }

        match kind {
            b'c' => emit_c(&v, ncomp, &mut sb, membits, &mut comp, &mut hcomp),
            b'i' if ncomp > 0 => emit_i(&v, &mut ncomp, &mut sb, membits, &mut comp, &mut hcomp),
            b'm' | b't' | b's' if ncomp > (kind == b't') as i32 =>
                emit_mts(kind, &v, ncomp, &mut sb, &mut comp, &mut hcomp),
            b'a' => emit_a(&v, ncomp, &mut sb, membits, &mut comp, &mut hcomp),
            b'w' => emit_w(&v, &mut ncomp, &mut sb, membits, &mut comp, &mut hcomp),
            _ => {} // f / unsupported: silently skip
        }

        // emit_c/i/etc. that *don't* push their own ncomp need it
        // bumped here. For c/m/t/s/a we increment in-line; for i/w
        // the helpers advance ncomp themselves.
        if matches!(kind, b'c' | b'm' | b't' | b's' | b'a') {
            ncomp += 1;
        }
    }

    // ---- Assemble.
    let mut out = String::new();
    out.push_str(hdr_prefix);
    out.push_str(&ncomp.to_string());
    out.push('\n');
    out.push_str(&comp);
    out.push_str(&hcomp);
    out.push_str("halt\n");
    out.push_str(&pcomp);
    Ok((out, args))
}

// ---- Component-spec emitters ------------------------------------------

fn lg(mut x: i32) -> i32 {
    if x <= 0 { return 0; }
    let mut r = 0;
    while x > 0 { r += 1; x >>= 1; }
    r
}

fn nbits(mut x: i32) -> i32 {
    let mut r = 0;
    while x > 0 { r += x & 1; x >>= 1; }
    r
}

fn emit_c(
    v: &[i32], ncomp: i32, sb: &mut i32, membits: i32,
    comp: &mut String, hcomp: &mut String,
) {
    // pad v to length 3 with zeros.
    let mut v = v.to_vec();
    while v.len() < 3 { v.push(0); }

    comp.push_str(&format!("{} ", ncomp));
    *sb = 11;
    if v[2] < 256 { *sb += lg(v[2]); } else { *sb += 6; }
    for &x in &v[3..] {
        if x < 512 { *sb += nbits(x) * 3 / 4; }
    }
    if *sb > membits { *sb = membits; }

    if v[1] % 1000 == 0 {
        comp.push_str(&format!("icm {}\n", *sb - 6 - v[1] / 1000));
    } else {
        comp.push_str(&format!("cm {} {}\n", *sb - 2 - v[1] / 1000, v[1] % 1000 - 1));
    }

    hcomp.push_str(&format!("d= {} *d=0\n", ncomp));
    if v[2] > 1 && v[2] <= 255 {
        if lg(v[2]) != lg(v[2] - 1) {
            hcomp.push_str(&format!("a=c a&= {} hashd\n", v[2] - 1));
        } else {
            hcomp.push_str(&format!("a=c a%= {} hashd\n", v[2]));
        }
    } else if v[2] >= 1000 && v[2] <= 1255 {
        hcomp.push_str(&format!(
            "a= 255 a+= {} d=a a=*d a-=c a> 255 if a= 255 endif d= {} hashd\n",
            v[2] - 1000, ncomp));
    }

    for (i, &mask) in v.iter().enumerate().skip(3) {
        if i == 3 { hcomp.push_str("b=c "); }
        if mask == 255 {
            hcomp.push_str("a=*b hashd\n");
        } else if mask > 0 && mask < 255 {
            hcomp.push_str(&format!("a=*b a&= {} hashd\n", mask));
        } else if mask >= 256 && mask < 512 {
            hcomp.push_str("a=r 1 a> 1 if\n  a=r 2 a< 64 if\n    a=*b ");
            if mask < 511 { hcomp.push_str(&format!("a&= {}", mask - 256)); }
            hcomp.push_str(
                " hashd\n  else\n    a>>= 6 hashd a=r 1 hashd\n  endif\nelse\n\
                  a= 255 hashd a=r 2 hashd\nendif\n");
        } else if mask >= 1256 {
            hcomp.push_str(&format!(
                "a= {} a<<= 8 a+= {} a+=b b=a\n",
                ((mask - 1000) >> 8) & 255, (mask - 1000) & 255));
        } else if mask > 1000 {
            hcomp.push_str(&format!("a= {} a+=b b=a\n", mask - 1000));
        }
        if mask < 512 && i < v.len() - 1 {
            hcomp.push_str("b++ ");
        }
    }
}

fn emit_i(
    v: &[i32], ncomp: &mut i32, sb: &mut i32, membits: i32,
    comp: &mut String, hcomp: &mut String,
) {
    hcomp.push_str(&format!("d= {} b=c a=*d d++\n", *ncomp - 1));
    for (i, &x) in v.iter().enumerate().skip(1) {
        if *ncomp >= 254 { break; }
        for j in 0..(x % 10) {
            hcomp.push_str("hash ");
            if i < v.len() - 1 || j < (x % 10) - 1 {
                hcomp.push_str("b++ ");
            }
            *sb += 6;
        }
        hcomp.push_str("*d=a");
        if i < v.len() - 1 { hcomp.push_str(" d++"); }
        hcomp.push('\n');
        if *sb > membits { *sb = membits; }
        comp.push_str(&format!("{} isse {} {}\n",
            *ncomp, *sb - 6 - x / 10, *ncomp - 1));
        *ncomp += 1;
    }
}

fn emit_mts(
    kind: u8, v: &[i32], ncomp: i32, sb: &mut i32,
    comp: &mut String, hcomp: &mut String,
) {
    let mut v = v.to_vec();
    if v.len() <= 1 { v.push(8); }
    if v.len() <= 2 { v.push(24 + 8 * (kind == b's') as i32); }
    if kind == b's' && v.len() <= 3 { v.push(255); }

    comp.push_str(&ncomp.to_string());
    *sb = 5 + v[1] * 3 / 4;
    match kind {
        b'm' => comp.push_str(&format!(" mix {} 0 {} {} 255\n", v[1], ncomp, v[2])),
        b't' => comp.push_str(&format!(" mix2 {} {} {} {} 255\n",
                                       v[1], ncomp - 1, ncomp - 2, v[2])),
        b's' => comp.push_str(&format!(" sse {} {} {} {}\n",
                                       v[1], ncomp - 1, v[2], v[3])),
        _ => unreachable!(),
    }

    if v[1] > 8 {
        hcomp.push_str(&format!("d= {} *d=0 b=c a=0\n", ncomp));
        let mut bits = v[1];
        while bits >= 16 {
            hcomp.push_str("a<<= 8 a+=*b");
            if bits > 16 { hcomp.push_str(" b++"); }
            hcomp.push('\n');
            bits -= 8;
        }
        if bits > 8 {
            hcomp.push_str(&format!("a<<= 8 a+=*b a>>= {}\n", 16 - bits));
        }
        hcomp.push_str("a<<= 8 *d=a\n");
    }
}

fn emit_a(
    v: &[i32], ncomp: i32, sb: &mut i32, membits: i32,
    comp: &mut String, hcomp: &mut String,
) {
    let mut v = v.to_vec();
    if v.len() <= 1 { v.push(24); }
    while v.len() < 4 { v.push(0); }

    comp.push_str(&format!("{} match {} {}\n",
        ncomp, membits - v[3] - 2, membits - v[2]));
    hcomp.push_str(&format!("d= {} a=*d a*= {} a+=*c a++ *d=a\n",
        ncomp, v[1]));
    *sb = 5 + (membits - v[2]) * 3 / 4;
}

fn emit_w(
    v: &[i32], ncomp: &mut i32, sb: &mut i32, membits: i32,
    comp: &mut String, hcomp: &mut String,
) {
    let mut v = v.to_vec();
    if v.len() <= 1 { v.push(1); }
    if v.len() <= 2 { v.push(65); }
    if v.len() <= 3 { v.push(26); }
    if v.len() <= 4 { v.push(223); }
    if v.len() <= 5 { v.push(20); }
    if v.len() <= 6 { v.push(0); }

    comp.push_str(&format!("{} icm {}\n", *ncomp, membits - 6 - v[6]));
    for i in 1..v[1] {
        comp.push_str(&format!("{} isse {} {}\n",
            *ncomp + i, membits - 6 - v[6], *ncomp + i - 1));
    }
    hcomp.push_str(&format!(
        "a=*c a&= {} a-= {} a&= 255 a< {} if\n",
        v[4], v[2], v[3]));
    for i in 0..v[1] {
        if i == 0 { hcomp.push_str(&format!("  d= {}", *ncomp)); }
        else { hcomp.push_str("  d++"); }
        hcomp.push_str(&format!(" a=*d a*= {} a+=*c a++ *d=a\n", v[5]));
    }
    hcomp.push_str("else\n");
    for i in (1..v[1]).rev() {
        hcomp.push_str(&format!("  d= {} a=*d d++ *d=a\n", *ncomp + i - 1));
    }
    hcomp.push_str(&format!("  d= {} *d=0\nendif\n", *ncomp));
    *ncomp += v[1] - 1;
    *sb = membits - v[6];
    *ncomp += 1;
}

// ---- Per-level PCOMP — compose from existing models constants -------

fn lz77_var_pcomp(doe8: bool) -> String {
    let cfg = if doe8 { crate::models::LZ77_VAR_E8E9_CFG }
              else    { crate::models::LZ77_VAR_CFG };
    extract_pcomp(cfg)
}

fn lz77_byte_pcomp(doe8: bool) -> String {
    let cfg = if doe8 { crate::models::LZ77_BYTE_E8E9_CFG }
              else    { crate::models::LZ77_BYTE_CFG };
    extract_pcomp(cfg)
}

fn bwt_pcomp(doe8: bool) -> String {
    let cfg = if doe8 { crate::models::BWT_E8E9_CFG }
              else    { crate::models::BWT_CFG };
    extract_pcomp(cfg)
}

fn e8e9_pcomp_only() -> String {
    // Level 0 + E8E9 — currently no canned config; return a no-op
    // "end" so the caller's compile_with_args succeeds. Full E8E9
    // post-processor for level-0 archives is a TODO.
    "end\n".to_string()
}

/// Pull just the `pcomp ... end` portion (or `end\n`) out of one of
/// the canned per-level config templates in [`crate::models`].
fn extract_pcomp(cfg: &str) -> String {
    if let Some(start) = cfg.find("pcomp ") {
        cfg[start..].to_string()
    } else if cfg.contains("\nend\n") {
        "end\n".to_string()
    } else {
        "end\n".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `make_config("0")` reproduces upstream's trivial config.
    #[test]
    fn make_config_zero() {
        let (cfg, args) = make_config("0").unwrap();
        assert_eq!(args[0], 0);
        assert_eq!(cfg, "comp 0 0 0 0 0 hcomp end\n");
    }

    /// `make_config("x4,3")` (BWT, no component spec) compiles to a
    /// header with `n = 0` — equivalent to our existing BWT_CFG.
    #[test]
    fn make_config_bwt_no_components() {
        let (cfg, args) = make_config("x4,3").unwrap();
        assert_eq!(args[0], 4);
        assert_eq!(args[1], 3);
        // Compiles without panicking and produces an n=0 header.
        let cc = crate::compiler::compile_with_args(&cfg, args).unwrap();
        assert_eq!(cc.header[6], 0); // n
        assert!(cc.pcomp.is_some());
    }

    /// `make_config("x4,3,ci1")` adds a CM + ISSE on top of BWT.
    #[test]
    fn make_config_bwt_with_ci1() {
        let (cfg, args) = make_config("x4,3,ci1").unwrap();
        let cc = crate::compiler::compile_with_args(&cfg, args).unwrap();
        assert_eq!(cc.header[6], 2); // n=2 (icm + isse)
        assert!(cc.pcomp.is_some());
    }

    /// `make_config("x4,2,4,0,3,15")` (byte LZ77 + secondary-context
    /// args) — same as our LZ77_BYTE path but driven through the
    /// generic builder.
    #[test]
    fn make_config_lz77_byte() {
        let (cfg, args) = make_config("x4,2,4,0,3,15").unwrap();
        assert_eq!(args[1], 2);
        assert_eq!(args[2], 4);
        let cc = crate::compiler::compile_with_args(&cfg, args).unwrap();
        assert_eq!(cc.header[6], 0);
    }
}
