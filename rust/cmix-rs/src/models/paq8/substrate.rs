//! Paq8 substrate primitives (paq8.cpp:153-512, 7900-8095).
//!
//! Includes:
//! * `Random`     — PRNG used by paq8 to seed its hash maps.
//! * `Buf`        — circular byte buffer (history window).
//! * `Ilog`       — base-2 log table.
//! * `Squash`     — `[-2047..2047] → [0..4095]` sigmoid table.
//! * `Stretch`    — inverse: `[0..4095] → [-2047..2047]`.
//! * `dt[1024]`   — `i → 16384/(2i+3)` decay table.
//! * `State_table` — 256-state PAQ history-state transitions.
//! * `dot_product` / `train` — fixed-point i16 vector ops for Mixer.
//! * `BitCount` / `ilog2` / `llog` / `finalize64` / `hash` /
//!                 `combine64` — hash helpers.
//!
//! All tables are computed once at construction; rebuild cost is
//! milliseconds on debug, free on release.

#![allow(dead_code)]

/// Mixer NUM_INPUTS (paq8.cpp:494).
pub const NUM_INPUTS: usize = 1552;
/// Mixer NUM_SETS — count of per-context sets feeding the Mixer.
pub const NUM_SETS: usize = 28;

/// `MEM(level) = 0x10000 << level`. Upstream default is `level = 11`,
/// giving 128 MiB. Calls with `level > 32` are saturated to avoid
/// overflow.
#[inline]
pub fn mem(level: u32) -> u64 {
    let l = level.min(32);
    (0x10000u64) << l
}

// =============================================================
// Random — paq8.cpp:153-169.
// =============================================================

/// Simple 4-LFSR-style PRNG. Upstream uses this to populate its
/// HashTable initial cells.
pub struct Random {
    table: [u32; 64],
    i:     usize,
}

impl Random {
    pub fn new() -> Self {
        let mut table = [0u32; 64];
        table[0] = 123456789;
        table[1] = 987654321;
        for j in 2..64 {
            table[j] = table[j - 1].wrapping_mul(11)
                .wrapping_add(table[j - 2].wrapping_mul(19) / 16);
        }
        Self { table, i: 0 }
    }

    pub fn next(&mut self) -> u32 {
        let lookahead10 = self.table[(self.i + 10) & 63];
        let lookahead24 = self.table[(self.i + 24) & 63];
        self.table[self.i & 63] = self.table[self.i & 63]
            ^ lookahead10 ^ lookahead24;
        let v = self.table[self.i & 63];
        self.i = self.i.wrapping_add(1);
        v
    }
}

impl Default for Random { fn default() -> Self { Self::new() } }

// =============================================================
// Buf — circular history buffer (paq8.cpp:170-204).
// =============================================================

/// Circular byte history. `size` must be a power of two.
pub struct Buf {
    b:   Vec<u8>,
    pub pos: u32,
}

impl Buf {
    /// Construct with default 0-size; call `set_size` before use.
    pub fn new() -> Self { Self { b: Vec::new(), pos: 0 } }

    /// `set_size(n)` — `n` must be a power of two. Resizes the
    /// underlying ring buffer; `pos` is preserved.
    pub fn set_size(&mut self, n: usize) {
        debug_assert!(n == 0 || n.is_power_of_two(),
                      "Buf size must be a power of two");
        self.b = vec![0u8; n];
    }

    pub fn size(&self) -> usize { self.b.len() }

    /// Mutable access to slot `i` modulo size (used to push the most
    /// recent byte).
    pub fn put(&mut self, i: u32, byte: u8) {
        let mask = (self.b.len() as u32).wrapping_sub(1);
        self.b[(i & mask) as usize] = byte;
    }

    /// `buf(i)` — byte at offset `i` positions before `pos`. Upstream
    /// uses `buf(0)` for the most recent (partial) byte, `buf(1)`
    /// for one byte back, etc.
    pub fn at(&self, i: u32) -> u8 {
        if self.b.is_empty() { return 0; }
        let mask = (self.b.len() as u32).wrapping_sub(1);
        self.b[((self.pos.wrapping_sub(i)) & mask) as usize]
    }

    /// `operator[](i)` — byte at absolute position `i` (mod size).
    pub fn abs(&self, i: u32) -> u8 {
        if self.b.is_empty() { return 0; }
        let mask = (self.b.len() as u32).wrapping_sub(1);
        self.b[(i & mask) as usize]
    }

    /// Append a byte at the current `pos` and advance `pos`.
    pub fn push(&mut self, byte: u8) {
        if self.b.is_empty() { return; }
        let mask = (self.b.len() as u32).wrapping_sub(1);
        self.b[(self.pos & mask) as usize] = byte;
        self.pos = self.pos.wrapping_add(1);
    }
}

impl Default for Buf { fn default() -> Self { Self::new() } }

// =============================================================
// Ilog — log lookup (paq8.cpp:254-267).
// =============================================================

/// `ilog(x)` for `x ∈ [0, 65535]` returning approximately
/// `64 * log2(x)`. Built by upstream's recursive `774541002/(i*2-1)`
/// summation.
pub struct Ilog { t: Vec<u8> }

impl Ilog {
    pub fn new() -> Self {
        let mut t = vec![0u8; 65536];
        let mut x: u32 = 14155776;
        for i in 2..65536 {
            x = x.wrapping_add(774541002 / (i as u32 * 2 - 1));
            t[i] = (x >> 24) as u8;
        }
        Self { t }
    }
    pub fn get(&self, x: u16) -> u8 { self.t[x as usize] }
}

impl Default for Ilog { fn default() -> Self { Self::new() } }

/// `llog(x)` — extended-range ilog covering all of `u32`.
#[inline]
pub fn llog(ilog: &Ilog, x: u32) -> u32 {
    if x >= 0x100_0000 { 256 + ilog.get((x >> 16) as u16) as u32 }
    else if x >= 0x10000 { 128 + ilog.get((x >> 8) as u16) as u32 }
    else { ilog.get(x as u16) as u32 }
}

/// Count of set bits in a `u32` (paq8.cpp:239 BitCount).
#[inline]
pub fn bit_count(mut v: u32) -> u32 {
    v -= (v >> 1) & 0x5555_5555;
    v = ((v >> 2) & 0x3333_3333) + (v & 0x3333_3333);
    v = ((v >> 4) + v) & 0x0f0f_0f0f;
    v = ((v >> 8) + v) & 0x00ff_00ff;
    v = ((v >> 16) + v) & 0x0000_ffff;
    v
}

/// Integer log2 (paq8.cpp:248). Returns `0` for `x = 0`.
#[inline]
pub fn ilog2(mut x: u32) -> u32 {
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    bit_count(x >> 1)
}

/// `Clip(Px) = clamp(Px, 0, 255)` — paq8.cpp:4184.
#[inline]
pub fn clip(px: i32) -> u8 { px.clamp(0, 255) as u8 }

/// `Clamp4(Px, n1, n2, n3, n4)` — clamp `Px` to the range
/// `[min(n1,n2,n3,n4), max(n1,n2,n3,n4)]`. paq8.cpp:4187.
#[inline]
pub fn clamp4(px: i32, n1: u8, n2: u8, n3: u8, n4: u8) -> u8 {
    let mx = n1.max(n2).max(n3).max(n4);
    let mn = n1.min(n2).min(n3).min(n4);
    mx.min(mn.max(px.clamp(0, 255) as u8))
}

/// `LogMeanDiffQt(a, b, limit=7)` — paq8.cpp:4191. Quantised log
/// of the relative magnitude difference between two byte values.
#[inline]
pub fn log_mean_diff_qt(a: u8, b: u8, limit: u8) -> u8 {
    if a == b { return 0; }
    let sign = if a > b { 1u8 << 3 } else { 0 };
    let diff = (a as i32 - b as i32).abs() as u32;
    let denom = (2 * diff).max(2);
    let v = (a as u32 + b as u32) / denom + 1;
    let l = ilog2(v) as u8;
    sign | l.min(limit)
}

/// Default `LogMeanDiffQt(a, b)` with limit = 7 (upstream default).
#[inline]
pub fn log_mean_diff(a: u8, b: u8) -> u8 { log_mean_diff_qt(a, b, 7) }

// =============================================================
// Squash / Stretch (paq8.cpp:346-393).
// =============================================================

pub struct Squash { t: Vec<u16> }

impl Squash {
    pub fn new() -> Self {
        let ts: [i32; 33] = [
            1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
            1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
            4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
        ];
        let mut t = vec![0u16; 4096];
        for i in -2047i32..=2047 {
            let w = i & 127;
            let d = ((i >> 7) + 16) as usize;
            let v = (ts[d] * (128 - w) + ts[d + 1] * w + 64) >> 7;
            t[(i + 2048) as usize] = v as u16;
        }
        Self { t }
    }

    pub fn get(&self, p: i32) -> i32 {
        if p > 2047 { return 4095; }
        if p < -2047 { return 0; }
        self.t[(p + 2048) as usize] as i32
    }
}

impl Default for Squash { fn default() -> Self { Self::new() } }

pub struct Stretch { t: Vec<i16> }

impl Stretch {
    pub fn new(squash: &Squash) -> Self {
        let mut t = vec![0i16; 4096];
        let mut pi = 0;
        for x in -2047i32..=2047 {
            let i = squash.get(x);
            for j in pi..=i { t[j as usize] = x as i16; }
            pi = i + 1;
        }
        t[4095] = 2047;
        Self { t }
    }
    pub fn get(&self, p: i32) -> i32 { self.t[p as usize] as i32 }
}

// =============================================================
// dt[1024] (paq8.cpp:203 + 8246 init).
// =============================================================

/// `dt[i] = 16384 / (2i + 3)` decay table (used by StateMap32 etc.).
pub fn build_dt() -> [i32; 1024] {
    let mut dt = [0i32; 1024];
    for i in 0..1024 { dt[i] = 16384 / (i as i32 + i as i32 + 3); }
    dt
}

// =============================================================
// State_table (paq8.cpp:280-345) — 256 × 4 transitions.
// =============================================================

/// `[state][selector]` → next-state. Selectors:
/// `0` = bit-0, `1` = bit-1, `2` = n0 count, `3` = n1 count.
///
/// Upstream declares `State_table[256][4]` but only initialises 253
/// rows; rows 253-255 are C++ zero-initialised. We mirror that with
/// 253 explicit rows + 3 `[0,0,0,0]` rows so `nex()` never goes out
/// of bounds (StateMap init iterates all 256 states).
pub const STATE_TABLE: [[u8; 4]; 256] = [
    [  1,   2,  0,  0],[  3,   5,  1,  0],[  4,   6,  0,  1],[  7,  10,  2,  0],
    [  8,  12,  1,  1],[  9,  13,  1,  1],[ 11,  14,  0,  2],[ 15,  19,  3,  0],
    [ 16,  23,  2,  1],[ 17,  24,  2,  1],[ 18,  25,  2,  1],[ 20,  27,  1,  2],
    [ 21,  28,  1,  2],[ 22,  29,  1,  2],[ 26,  30,  0,  3],[ 31,  33,  4,  0],
    [ 32,  35,  3,  1],[ 32,  35,  3,  1],[ 32,  35,  3,  1],[ 32,  35,  3,  1],
    [ 34,  37,  2,  2],[ 34,  37,  2,  2],[ 34,  37,  2,  2],[ 34,  37,  2,  2],
    [ 34,  37,  2,  2],[ 34,  37,  2,  2],[ 36,  39,  1,  3],[ 36,  39,  1,  3],
    [ 36,  39,  1,  3],[ 36,  39,  1,  3],[ 38,  40,  0,  4],[ 41,  43,  5,  0],
    [ 42,  45,  4,  1],[ 42,  45,  4,  1],[ 44,  47,  3,  2],[ 44,  47,  3,  2],
    [ 46,  49,  2,  3],[ 46,  49,  2,  3],[ 48,  51,  1,  4],[ 48,  51,  1,  4],
    [ 50,  52,  0,  5],[ 53,  43,  6,  0],[ 54,  57,  5,  1],[ 54,  57,  5,  1],
    [ 56,  59,  4,  2],[ 56,  59,  4,  2],[ 58,  61,  3,  3],[ 58,  61,  3,  3],
    [ 60,  63,  2,  4],[ 60,  63,  2,  4],[ 62,  65,  1,  5],[ 62,  65,  1,  5],
    [ 50,  66,  0,  6],[ 67,  55,  7,  0],[ 68,  57,  6,  1],[ 68,  57,  6,  1],
    [ 70,  73,  5,  2],[ 70,  73,  5,  2],[ 72,  75,  4,  3],[ 72,  75,  4,  3],
    [ 74,  77,  3,  4],[ 74,  77,  3,  4],[ 76,  79,  2,  5],[ 76,  79,  2,  5],
    [ 62,  81,  1,  6],[ 62,  81,  1,  6],[ 64,  82,  0,  7],[ 83,  69,  8,  0],
    [ 84,  71,  7,  1],[ 84,  71,  7,  1],[ 86,  73,  6,  2],[ 86,  73,  6,  2],
    [ 44,  59,  5,  3],[ 44,  59,  5,  3],[ 58,  61,  4,  4],[ 58,  61,  4,  4],
    [ 60,  49,  3,  5],[ 60,  49,  3,  5],[ 76,  89,  2,  6],[ 76,  89,  2,  6],
    [ 78,  91,  1,  7],[ 78,  91,  1,  7],[ 80,  92,  0,  8],[ 93,  69,  9,  0],
    [ 94,  87,  8,  1],[ 94,  87,  8,  1],[ 96,  45,  7,  2],[ 96,  45,  7,  2],
    [ 48,  99,  2,  7],[ 48,  99,  2,  7],[ 88, 101,  1,  8],[ 88, 101,  1,  8],
    [ 80, 102,  0,  9],[103,  69, 10,  0],[104,  87,  9,  1],[104,  87,  9,  1],
    [106,  57,  8,  2],[106,  57,  8,  2],[ 62, 109,  2,  8],[ 62, 109,  2,  8],
    [ 88, 111,  1,  9],[ 88, 111,  1,  9],[ 80, 112,  0, 10],[113,  85, 11,  0],
    [114,  87, 10,  1],[114,  87, 10,  1],[116,  57,  9,  2],[116,  57,  9,  2],
    [ 62, 119,  2,  9],[ 62, 119,  2,  9],[ 88, 121,  1, 10],[ 88, 121,  1, 10],
    [ 90, 122,  0, 11],[123,  85, 12,  0],[124,  97, 11,  1],[124,  97, 11,  1],
    [126,  57, 10,  2],[126,  57, 10,  2],[ 62, 129,  2, 10],[ 62, 129,  2, 10],
    [ 98, 131,  1, 11],[ 98, 131,  1, 11],[ 90, 132,  0, 12],[133,  85, 13,  0],
    [134,  97, 12,  1],[134,  97, 12,  1],[136,  57, 11,  2],[136,  57, 11,  2],
    [ 62, 139,  2, 11],[ 62, 139,  2, 11],[ 98, 141,  1, 12],[ 98, 141,  1, 12],
    [ 90, 142,  0, 13],[143,  95, 14,  0],[144,  97, 13,  1],[144,  97, 13,  1],
    [ 68,  57, 12,  2],[ 68,  57, 12,  2],[ 62,  81,  2, 12],[ 62,  81,  2, 12],
    [ 98, 147,  1, 13],[ 98, 147,  1, 13],[100, 148,  0, 14],[149,  95, 15,  0],
    [150, 107, 14,  1],[150, 107, 14,  1],[108, 151,  1, 14],[108, 151,  1, 14],
    [100, 152,  0, 15],[153,  95, 16,  0],[154, 107, 15,  1],[108, 155,  1, 15],
    [100, 156,  0, 16],[157,  95, 17,  0],[158, 107, 16,  1],[108, 159,  1, 16],
    [100, 160,  0, 17],[161, 105, 18,  0],[162, 107, 17,  1],[108, 163,  1, 17],
    [110, 164,  0, 18],[165, 105, 19,  0],[166, 117, 18,  1],[118, 167,  1, 18],
    [110, 168,  0, 19],[169, 105, 20,  0],[170, 117, 19,  1],[118, 171,  1, 19],
    [110, 172,  0, 20],[173, 105, 21,  0],[174, 117, 20,  1],[118, 175,  1, 20],
    [110, 176,  0, 21],[177, 105, 22,  0],[178, 117, 21,  1],[118, 179,  1, 21],
    [110, 180,  0, 22],[181, 115, 23,  0],[182, 117, 22,  1],[118, 183,  1, 22],
    [120, 184,  0, 23],[185, 115, 24,  0],[186, 127, 23,  1],[128, 187,  1, 23],
    [120, 188,  0, 24],[189, 115, 25,  0],[190, 127, 24,  1],[128, 191,  1, 24],
    [120, 192,  0, 25],[193, 115, 26,  0],[194, 127, 25,  1],[128, 195,  1, 25],
    [120, 196,  0, 26],[197, 115, 27,  0],[198, 127, 26,  1],[128, 199,  1, 26],
    [120, 200,  0, 27],[201, 115, 28,  0],[202, 127, 27,  1],[128, 203,  1, 27],
    [120, 204,  0, 28],[205, 115, 29,  0],[206, 127, 28,  1],[128, 207,  1, 28],
    [120, 208,  0, 29],[209, 125, 30,  0],[210, 127, 29,  1],[128, 211,  1, 29],
    [130, 212,  0, 30],[213, 125, 31,  0],[214, 137, 30,  1],[138, 215,  1, 30],
    [130, 216,  0, 31],[217, 125, 32,  0],[218, 137, 31,  1],[138, 219,  1, 31],
    [130, 220,  0, 32],[221, 125, 33,  0],[222, 137, 32,  1],[138, 223,  1, 32],
    [130, 224,  0, 33],[225, 125, 34,  0],[226, 137, 33,  1],[138, 227,  1, 33],
    [130, 228,  0, 34],[229, 125, 35,  0],[230, 137, 34,  1],[138, 231,  1, 34],
    [130, 232,  0, 35],[233, 125, 36,  0],[234, 137, 35,  1],[138, 235,  1, 35],
    [130, 236,  0, 36],[237, 125, 37,  0],[238, 137, 36,  1],[138, 239,  1, 36],
    [130, 240,  0, 37],[241, 125, 38,  0],[242, 137, 37,  1],[138, 243,  1, 37],
    [130, 244,  0, 38],[245, 135, 39,  0],[246, 137, 38,  1],[138, 247,  1, 38],
    [140, 248,  0, 39],[249, 135, 40,  0],[250,  69, 39,  1],[ 80, 251,  1, 39],
    [140, 252,  0, 40],[249, 135, 41,  0],[250,  69, 40,  1],[ 80, 251,  1, 40],
    [140, 252,  0, 41],
    // Rows 253-255: C++ zero-initialised in upstream.
    [  0,   0,  0,  0],[  0,   0,  0,  0],[  0,   0,  0,  0],
];

#[inline]
pub fn nex(state: u8, sel: usize) -> u8 {
    STATE_TABLE[state as usize][sel]
}

// =============================================================
// Hash helpers — verbatim port of paq8.cpp:716-775.
// =============================================================

pub const PHI64:    u64 = 0x9E37_79B9_7F4A_7C15;
pub const MUL64_1:  u64 = 0x993D_DEFF_B146_2949;
pub const MUL64_2:  u64 = 0xE9C9_1DC1_59AB_0D2D;
pub const MUL64_3:  u64 = 0x83D6_A14F_1B0C_ED73;
pub const MUL64_4:  u64 = 0xA14F_1B0C_ED5A_841F;
pub const MUL64_5:  u64 = 0xC0E5_1314_A614_F4EF;
pub const MUL64_6:  u64 = 0xDA9C_C260_0AE4_5A27;
pub const MUL64_7:  u64 = 0x8267_97AA_04A6_5737;
pub const MUL64_8:  u64 = 0x2375_BE54_C41A_08ED;
pub const MUL64_9:  u64 = 0xD391_04E9_5056_4B37;
pub const MUL64_10: u64 = 0x3091_697D_5E68_5623;
pub const MUL64_11: u64 = 0x20EB_84EE_04A3_C7E1;
pub const MUL64_12: u64 = 0xF501_F1D0_944B_2383;
pub const MUL64_13: u64 = 0xE3E4_E8AA_829A_B9B5;

/// Top `hashbits` of a 64-bit hash. Matches upstream's
/// `finalize64(hash, hashbits) = hash >> (64-hashbits)`.
#[inline]
pub fn finalize64(hash: u64, hashbits: u32) -> u32 {
    debug_assert!(hashbits > 0 && hashbits <= 32);
    (hash >> (64 - hashbits as u64)) as u32
}

/// `checksum64(hash, hashbits, checksumbits)` extracts the `csbits`
/// bits BELOW the `finalize64` window — paq8.cpp:735.
#[inline]
pub fn checksum64(hash: u64, hashbits: u32, checksumbits: u32) -> u64 {
    debug_assert!(hashbits + checksumbits <= 64);
    hash >> (64 - hashbits as u64 - checksumbits as u64)
}

/// `combine64(seed, x) = hash(seed + x)` (paq8.cpp:773).
#[inline]
pub fn combine64(seed: u64, x: u64) -> u64 {
    hash1(seed.wrapping_add(x))
}

#[inline]
pub fn hash1(x0: u64) -> u64 { x0.wrapping_add(1).wrapping_mul(PHI64) }

#[inline]
pub fn hash2(x0: u64, x1: u64) -> u64 {
    x0.wrapping_add(1).wrapping_mul(PHI64)
        .wrapping_add(x1.wrapping_add(1).wrapping_mul(MUL64_1))
}

#[inline]
pub fn hash3(x0: u64, x1: u64, x2: u64) -> u64 {
    x0.wrapping_add(1).wrapping_mul(PHI64)
        .wrapping_add(x1.wrapping_add(1).wrapping_mul(MUL64_1))
        .wrapping_add(x2.wrapping_add(1).wrapping_mul(MUL64_2))
}

#[inline]
pub fn hash4(x0: u64, x1: u64, x2: u64, x3: u64) -> u64 {
    hash3(x0, x1, x2)
        .wrapping_add(x3.wrapping_add(1).wrapping_mul(MUL64_3))
}

#[inline]
pub fn hash5(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64) -> u64 {
    hash4(x0, x1, x2, x3)
        .wrapping_add(x4.wrapping_add(1).wrapping_mul(MUL64_4))
}

#[inline]
pub fn hash6(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64, x5: u64) -> u64 {
    hash5(x0, x1, x2, x3, x4)
        .wrapping_add(x5.wrapping_add(1).wrapping_mul(MUL64_5))
}

#[inline]
pub fn hash7(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64,
              x5: u64, x6: u64) -> u64 {
    hash6(x0, x1, x2, x3, x4, x5)
        .wrapping_add(x6.wrapping_add(1).wrapping_mul(MUL64_6))
}

#[inline]
pub fn hash8(x0: u64, x1: u64, x2: u64, x3: u64, x4: u64,
              x5: u64, x6: u64, x7: u64) -> u64 {
    hash7(x0, x1, x2, x3, x4, x5, x6)
        .wrapping_add(x7.wrapping_add(1).wrapping_mul(MUL64_7))
}

// =============================================================
// dot_product / train (paq8.cpp:455-509) — i16 vector ops.
// =============================================================

/// Compute `sum((t[i] * w[i]) >> 8)` over an n-long aligned-up to 16
/// stride. Plain scalar form — matches upstream's non-SIMD fallback.
#[inline]
pub fn dot_product(t: &[i16], w: &[i16], n: usize) -> i32 {
    let n = (n + 15) & !15;
    let mut sum: i32 = 0;
    let mut i = 0;
    while i < n {
        let a = (t[i] as i32 * w[i] as i32) >> 8;
        let b = (t[i + 1] as i32 * w[i + 1] as i32) >> 8;
        sum = sum.wrapping_add(a + b);
        i += 2;
    }
    sum
}

/// Train weights `w` toward target via gradient step. Matches
/// upstream's scalar fallback for portability.
#[inline]
pub fn train(t: &[i16], w: &mut [i16], n: usize, err: i32) {
    let n = (n + 15) & !15;
    for i in 0..n {
        let delta = (((t[i] as i32 * err * 2) >> 16) + 1) >> 1;
        let mut wt = w[i] as i32 + delta;
        if wt < -32768 { wt = -32768; }
        if wt >  32767 { wt =  32767; }
        w[i] = wt as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_is_deterministic_for_same_seed_table() {
        let mut a = Random::new();
        let mut b = Random::new();
        for _ in 0..1000 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn buf_circular_wrap() {
        let mut buf = Buf::new();
        buf.set_size(8);
        buf.put(0, 0xAA); buf.pos = 1;
        buf.put(1, 0xBB); buf.pos = 2;
        assert_eq!(buf.at(1), 0xBB);
        assert_eq!(buf.at(2), 0xAA);
        assert_eq!(buf.at(0), 0); // current slot
    }

    #[test]
    fn ilog_monotone_and_endpoints() {
        let il = Ilog::new();
        assert_eq!(il.get(0), 0);
        for i in 2..65535 {
            assert!(il.get(i) >= il.get(i - 1), "ilog must be monotone");
        }
        // Top u8 value is 255 — paq8's ilog table is normalised to fit
        // `U8` and approximates `64 * log2(x)` rather than the raw log.
        assert!(il.get(65535) >= 200);
    }

    #[test]
    fn bit_count_matches_built_in() {
        for &v in &[0u32, 1, 3, 7, 0xF0F0, 0xFFFF_FFFF, 0x8000_0000] {
            assert_eq!(bit_count(v), v.count_ones());
        }
    }

    #[test]
    fn ilog2_matches_floor_log2() {
        assert_eq!(ilog2(0), 0);
        assert_eq!(ilog2(1), 0);
        assert_eq!(ilog2(2), 1);
        assert_eq!(ilog2(3), 1);
        assert_eq!(ilog2(4), 2);
        assert_eq!(ilog2(255), 7);
        assert_eq!(ilog2(256), 8);
        assert_eq!(ilog2(1 << 31), 31);
    }

    #[test]
    fn squash_stretch_roundtrip_within_one_step() {
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        for &p in &[1i32, 100, 1000, 2048, 3000, 4000, 4094] {
            let d  = st.get(p);
            let p2 = sq.get(d);
            assert!((p - p2).abs() < 50, "p={} → d={} → p2={}", p, d, p2);
        }
    }

    #[test]
    fn squash_endpoints_clamped() {
        let sq = Squash::new();
        assert_eq!(sq.get(-3000), 0);
        assert_eq!(sq.get( 3000), 4095);
    }

    #[test]
    fn dt_table_is_descending() {
        let dt = build_dt();
        for i in 1..1024 { assert!(dt[i] <= dt[i - 1]); }
        assert_eq!(dt[0], 16384 / 3);
    }

    #[test]
    fn state_table_has_256_entries_matching_upstream() {
        assert_eq!(STATE_TABLE.len(), 256);
        for s in 0..=255u8 {
            for sel in 0..4 {
                let _ = nex(s, sel);
            }
        }
        // Spot-check: state 0 → bit-0 → state 1; bit-1 → state 2.
        assert_eq!(nex(0, 0), 1);
        assert_eq!(nex(0, 1), 2);
        // Rows 253-255 are zero-filled (upstream C++ zero-init).
        assert_eq!(nex(253, 0), 0);
        assert_eq!(nex(255, 3), 0);
    }

    #[test]
    fn dot_product_simple() {
        let t = [256i16; 16];
        let w = [256i16; 16];
        // dot = sum_{i=0..16} (256*256)>>8 = sum_{i=0..16} 256 = 4096
        assert_eq!(dot_product(&t, &w, 16), 4096);
    }

    #[test]
    fn train_moves_weights_toward_error() {
        let t = [256i16; 16];
        let mut w = [0i16; 16];
        // Positive error → positive weight delta.
        train(&t, &mut w, 16, 1024);
        for &wi in &w { assert!(wi > 0); }
    }

    #[test]
    fn finalize64_extracts_top_bits() {
        // `0xDEAD_BEEF_CAFE_BABE >> (64-16) = 0xDEAD`.
        assert_eq!(finalize64(0xDEAD_BEEF_CAFE_BABE, 16), 0xDEAD);
        assert_eq!(finalize64(0xFFFF_FFFF_FFFF_FFFF, 1),  1);
        assert_eq!(finalize64(0x0000_0000_0000_0000, 32), 0);
    }

    #[test]
    fn checksum64_extracts_next_window() {
        let h: u64 = 0xDEAD_BEEF_CAFE_BABE;
        // finalize64 takes top 16 (0xDEAD). checksum64(16, 16) takes
        // the next 16 (0xBEEF).
        assert_eq!(checksum64(h, 16, 16) as u32 & 0xffff, 0xBEEF);
    }

    #[test]
    fn hashn_match_upstream_formula() {
        // (1)*PHI64 — explicit base case.
        assert_eq!(hash1(0), PHI64);
        // h(0,0) = (0+1)*PHI64 + (0+1)*MUL64_1
        assert_eq!(hash2(0, 0),
                   PHI64.wrapping_add(MUL64_1));
        // h(1,2,3) per upstream's spelling.
        let expected = 2u64.wrapping_mul(PHI64)
            .wrapping_add(3u64.wrapping_mul(MUL64_1))
            .wrapping_add(4u64.wrapping_mul(MUL64_2));
        assert_eq!(hash3(1, 2, 3), expected);
    }
}
