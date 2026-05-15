//! Shared per-byte state passed between paq8 sub-models (mirror of
//! upstream's `ModelStats` struct, paq8.cpp:205-253).

#![allow(dead_code)]

use crate::preprocess; // for filetype enum, eventually

/// Mirror of upstream `Filetype` enum — kept locally so paq8 can
/// dispatch on it without depending on the runner / preprocessor
/// modules. Order must match `preprocess::Filetype` in upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filetype {
    Default = 0,
    Hdr,
    Jpeg,
    Exe,
    Text,
    Image1,
    Image4,
    Image8,
    Image8Gray,
    Image24,
    Image32,
    Audio,
}

impl Default for Filetype { fn default() -> Self { Filetype::Default } }

#[derive(Default, Clone, Copy, Debug)]
pub struct MatchStats {
    pub length:        u32,
    pub expected_byte: u8,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Pixels {
    pub ww: u8, pub w: u8,
    pub nn: u8, pub n: u8,
    pub wp1: u8, pub np1: u8,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct ImageStats {
    pub pixels: Pixels,
    pub plane:  u8,
    pub ctx:    u8,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct TextStats {
    pub state:        u8,  // 3-bit; unused by SSE today
    pub last_punct:   u8,  // 5-bit
    pub word_length:  u8,  // 4-bit
    pub boolmask:     u8,  // 4-bit
    pub first_letter: u8,
    pub mask:         u8,
}

#[derive(Default, Clone, Debug)]
pub struct ModelStats {
    pub filetype: Filetype,
    pub misses:   u64,
    pub r#match:  MatchStats,
    pub image:    ImageStats,
    pub xml:      u32,
    pub x86_64:   u32,
    pub record:   u32,
    pub text:     TextStats,
}

impl ModelStats {
    pub fn new() -> Self { Self::default() }
}

// Suppress unused-import warning for the preprocess module while the
// enum-conversion plumbing is still in flux.
#[doc(hidden)]
#[allow(dead_code)]
fn _preprocess_anchor(_: &preprocess::PreprocDict) {}
