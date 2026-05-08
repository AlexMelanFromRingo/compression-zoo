//! Canned ZPAQ models — bytewise copies of the three headers
//! upstream ships in `Compressor::startBlock(int level)`
//! (`plugins/zpaq/upstream/libzpaq.cpp:2796`):
//!
//!   * [`MIN_CFG`] — level 1 model (min.cfg, 28 bytes).
//!   * [`MID_CFG`] — level 2 model (mid.cfg, 71 bytes).
//!   * [`MAX_CFG`] — level 3 model (max.cfg, 217 bytes).
//!
//! Each blob is the wire format starting at the LE u16 `hsize`
//! prefix, exactly the bytes [`crate::compress::Compresser::start_block_modeled`]
//! expects.

#![allow(dead_code)]

/// Level 1 model — `min.cfg`. ICM context order 2, hashed via HCOMP.
pub const MIN_CFG: &[u8] = &[
    26, 0, 1, 2, 0, 0, 2, 3, 16, 8, 19, 0, 0, 96, 4, 28,
    59, 10, 59, 112, 25, 10, 59, 10, 59, 112, 56, 0,
];

/// Level 2 model — `mid.cfg`. ICM + 5×ISSE + MIX over 7 components.
pub const MID_CFG: &[u8] = &[
    69, 0, 3, 3, 0, 0, 8, 3, 5, 8, 13, 0, 8, 17, 1, 8,
    18, 2, 8, 18, 3, 8, 19, 4, 4, 22, 24, 7, 16, 0, 7, 24,
    255, 0, 17, 104, 74, 4, 95, 1, 59, 112, 10, 25, 59, 112, 10, 25,
    59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 10, 59, 112,
    25, 69, 207, 8, 112, 56, 0,
];

/// Level 3 model — `max.cfg`. 22 components incl. ICM/ISSE/MATCH/MIX/SSE.
pub const MAX_CFG: &[u8] = &[
    196, 0, 5, 9, 0, 0, 22, 1, 160, 3, 5, 8, 13, 1, 8, 16,
    2, 8, 18, 3, 8, 19, 4, 8, 19, 5, 8, 20, 6, 4, 22, 24,
    3, 17, 8, 19, 9, 3, 13, 3, 13, 3, 13, 3, 14, 7, 16, 0,
    15, 24, 255, 7, 8, 0, 16, 10, 255, 6, 0, 15, 16, 24, 0, 9,
    8, 17, 32, 255, 6, 8, 17, 18, 16, 255, 9, 16, 19, 32, 255, 6,
    0, 19, 20, 16, 0, 0, 17, 104, 74, 4, 95, 2, 59, 112, 10, 25,
    59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25, 59, 112, 10, 25,
    59, 10, 59, 112, 10, 25, 59, 112, 10, 25, 69, 183, 32, 239, 64, 47,
    14, 231, 91, 47, 10, 25, 60, 26, 48, 134, 151, 20, 112, 63, 9, 70,
    223, 0, 39, 3, 25, 112, 26, 52, 25, 25, 74, 10, 4, 59, 112, 25,
    10, 4, 59, 112, 25, 10, 4, 59, 112, 25, 65, 143, 212, 72, 4, 59,
    112, 8, 143, 216, 8, 68, 175, 60, 60, 25, 69, 207, 9, 112, 25, 25,
    25, 25, 25, 112, 56, 0,
];

/// Byte-aligned LZ77 (level 2) config template, extracted verbatim
/// from upstream `makeConfig("x4,2,...", args)`. `$1` is the
/// log-block-size and `$3` is the LZ77 minimum match length. The
/// PCOMP runs the inverse LZ77 producing the original byte stream.
pub const LZ77_BYTE_CFG: &str = r#"
comp 9 16 0 $1+20 0
hcomp
c-- *c=a a+= 255 d=a *d=c
  (decode lz77 into M. Codes:
  00xxxxxx = literal length xxxxxx+1
  xx......, xx > 0 = match with xx offset bytes to follow)

  a=r 1 a== 0 if (init)
    a= 111 (skip post code)
  else a== 1 if  (new code?)
    a=*c r=a 2  (save code in R2)
    a> 63 if a>>= 6 a++ a++  (match)
    else a++ a++ endif  (literal)
  else (read rest of code)
    a--
  endif endif
  r=a 1  (R1 = 1+expected bytes to next code)
halt
pcomp lzpre c ;
  (Decode LZ77: d=state, M=output buffer, b=size)
  a> 255 if (at EOF decode e8e9 and output)
    b=0 c=0 d=0 a=0 r=a 1 r=a 2 (reset state)
  halt
  endif

  (in state d==0, expect a new code)
  (put length in r1 and inital part of offset in r2)
  c=a a=d a== 0 if
    a=c a>>= 6 a++ d=a
    a== 1 if (literal?)
      a+=c r=a 1 a=0 r=a 2
    else (3 to 5 byte match)
      d++ a=c a&= 63 a+= $3 r=a 1 a=0 r=a 2
    endif
  else
    a== 1 if (writing literal)
      a=c *b=a b++
 out
      a=r 1 a-- a== 0 if d=0 endif r=a 1 (if (--len==0) state=0)
    else
      a> 2 if (reading offset)
        a=r 2 a<<= 8 a|=c r=a 2 d-- (off=off<<8|c, --state)
      else (state==2, write match)
        a=r 2 a<<= 8 a|=c c=a a=b a-=c a-- c=a (c=i-off-1)
        d=r 1 (d=len)
        do (copy and output d=len bytes)
          a=*c *b=a c++ b++
 out
        d-- a=d a> 0 while
        (d=state=0. off, len don't matter)
      endif
    endif
  endif
  halt
end
"#;

/// BWT (level 3) config template, extracted verbatim from upstream
/// `makeConfig("x4,3", args)`. `$1` substitutes the log-block-size
/// (`args[0]`), so e.g. with `args[0] = 4` the H/M arrays both use
/// `2^(4+20) = 16 MiB`. The PCOMP performs IBWT on the
/// length-prefixed BWT byte stream produced by
/// [`crate::lzbuffer::preprocess`] at `level_flag = 3`.
pub const BWT_CFG: &str = r#"
comp 9 16 $1+20 $1+20 0
hcomp
c-- *c=a a+= 255 d=a *d=c
halt
pcomp bwtrle c ;

  (read BWT, index into M, size in b)
  a> 255 ifnot
    *b=a b++

  (inverse BWT)
  elsel

    (index in last 4 bytes, put in c and R1)
    b-- a=*b
    b-- a<<= 8 a+=*b
    b-- a<<= 8 a+=*b
    b-- a<<= 8 a+=*b c=a r=a 1

    (save size in R2)
    a=b r=a 2

    (count bytes in H[~1..~255, ~0])
    do
      a=b a> 0 if
        b-- a=*b a++ a&= 255 d=a d! *d++
      forever
    endif

    (cumulative counts: H[~i=0..255] = count of bytes before i)
    d=0 d! *d= 1 a=0
    do
      a+=*d *d=a d--
    d<>a a! a> 255 a! d<>a until

    (build first part of linked list in H[0..idx-1])
    b=0 do
      a=c a>b if
        d=*b d! *d++ d=*d d-- *d=b
      b++ forever
    endif

    (rest of list in H[idx+1..n-1])
    b=c b++ c=r 2 do
      a=c a>b if
        d=*b d! *d++ d=*d d-- *d=b
      b++ forever
    endif

    (copy M to low 8 bits of H to reduce cache misses in next loop)
    b=0 do
      a=c a>b if
        d=b a=*d a<<= 8 a+=*b *d=a
      b++ forever
    endif

    (traverse list and output or copy to M)
    d=r 1 b=0 do
      a=d a== 0 ifnot
        a=*d a>>= 8 d=a
 a=*d out
      forever
    endif

  endif
  halt
end
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_sizes_match_hsize_prefix() {
        for cfg in [MIN_CFG, MID_CFG, MAX_CFG] {
            let hsize = (cfg[0] as usize) | ((cfg[1] as usize) << 8);
            // The blob is `hsize + 2` bytes total (hsize prefix +
            // hsize bytes of payload). Verify upstream's invariant.
            assert_eq!(cfg.len(), hsize + 2);
        }
    }
}
