//! `paq8::Word` + language model scaffolding — paq8.cpp:1548-1726.
//!
//! Distinct from `fxcmv1::Word`: paq8's `Word` carries a 4-element
//! `u64` hash array, a `Type` bitmask and a `Language` id. Used by
//! the stemmers, TextModel, wordModel and nestModel.

#![allow(dead_code)]

use super::substrate::hash3;

pub const MAX_WORD_SIZE: usize = 64;

// =============================================================
// Language flag bitmasks — paq8.cpp:1653-1726.
// =============================================================

/// Shared language flags (`Language::Flags`).
pub mod lang {
    pub const VERB: u64 = 1 << 0;
    pub const NOUN: u64 = 1 << 1;
}

/// English-specific flags (`English::Flags`), paq8.cpp:1675-1699.
pub mod english {
    use super::lang::{NOUN, VERB};
    pub const ADJECTIVE:              u64 = 1 << 2;
    pub const PLURAL:                 u64 = 1 << 3;
    pub const MALE:                   u64 = 1 << 4;
    pub const FEMALE:                 u64 = 1 << 5;
    pub const NEGATION:               u64 = 1 << 6;
    pub const PAST_TENSE:             u64 = (1 << 7) | VERB;
    pub const PRESENT_PARTICIPLE:     u64 = (1 << 8) | VERB;
    pub const ADJECTIVE_SUPERLATIVE:  u64 = (1 << 9) | ADJECTIVE;
    pub const ADJECTIVE_WITHOUT:      u64 = (1 << 10) | ADJECTIVE;
    pub const ADJECTIVE_FULL:         u64 = (1 << 11) | ADJECTIVE;
    pub const ADVERB_OF_MANNER:       u64 = 1 << 12;
    pub const SUFFIX_NESS:            u64 = 1 << 13;
    pub const SUFFIX_ITY:             u64 = (1 << 14) | NOUN;
    pub const SUFFIX_CAPABLE:         u64 = 1 << 15;
    pub const SUFFIX_NCE:             u64 = 1 << 16;
    pub const SUFFIX_NT:              u64 = 1 << 17;
    pub const SUFFIX_ION:             u64 = 1 << 18;
    pub const SUFFIX_AL:              u64 = (1 << 19) | ADJECTIVE;
    pub const SUFFIX_IC:              u64 = (1 << 20) | ADJECTIVE;
    pub const SUFFIX_IVE:             u64 = 1 << 21;
    pub const SUFFIX_OUS:             u64 = (1 << 22) | ADJECTIVE;
    pub const PREFIX_OVER:            u64 = 1 << 23;
    pub const PREFIX_UNDER:           u64 = 1 << 24;
}

/// French-specific flags (`French::Flags`).
pub mod french {
    pub const ADJECTIVE: u64 = 1 << 2;
    pub const PLURAL:    u64 = 1 << 3;
}

/// German-specific flags (`German::Flags`).
pub mod german {
    pub const ADJECTIVE: u64 = 1 << 2;
    pub const PLURAL:    u64 = 1 << 3;
    pub const FEMALE:    u64 = 1 << 4;
}

/// Language ids (`Language::Ids`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageId {
    Unknown = 0,
    English = 1,
    French  = 2,
    German  = 3,
}

// =============================================================
// Word — paq8.cpp:1548-1623.
// =============================================================

#[derive(Clone)]
pub struct Word {
    pub letters: [u8; MAX_WORD_SIZE],
    pub start:   u8,
    pub end:     u8,
    pub hash:    [u64; 4],
    pub r#type:  u64,
    pub language: u64,
}

impl Default for Word { fn default() -> Self { Self::new() } }

impl Word {
    pub fn new() -> Self {
        Self {
            letters: [0; MAX_WORD_SIZE],
            start: 0, end: 0,
            hash: [0; 4], r#type: 0, language: 0,
        }
    }

    /// Reset to the empty word (upstream's default-construct after
    /// `memset`).
    pub fn reset(&mut self) {
        *self = Word::new();
    }

    /// `operator==(const char *s)` — paq8.cpp:1556-1559.
    pub fn equals_str(&self, s: &[u8]) -> bool {
        let extra = if self.letters[self.start as usize] != 0 { 1 } else { 0 };
        let cur_len = (self.end as usize)
            .wrapping_sub(self.start as usize)
            .wrapping_add(extra);
        cur_len == s.len()
            && self.letters[self.start as usize..self.start as usize + s.len()] == *s
    }

    /// `operator+=(const char c)` — paq8.cpp:1563-1568. Appends a
    /// lowercased char.
    pub fn push(&mut self, c: u8) {
        if (self.end as usize) < MAX_WORD_SIZE - 1 {
            if self.letters[self.end as usize] > 0 { self.end += 1; }
            self.letters[self.end as usize] = c.to_ascii_lowercase();
        }
    }

    /// `operator[](U8 i)` — letter at offset `i` from `Start`.
    pub fn at(&self, i: u8) -> u8 {
        if self.end.wrapping_sub(self.start) >= i {
            self.letters[(self.start + i) as usize]
        } else { 0 }
    }

    /// `operator()(U8 i)` — letter at offset `i` from `End`.
    pub fn from_end(&self, i: u8) -> u8 {
        if self.end.wrapping_sub(self.start) >= i {
            self.letters[(self.end - i) as usize]
        } else { 0 }
    }

    pub fn length(&self) -> u32 {
        if self.letters[self.start as usize] != 0 {
            (self.end - self.start + 1) as u32
        } else { 0 }
    }

    pub fn is_empty(&self) -> bool { self.length() == 0 }

    /// `GetHashes()` — paq8.cpp:1580-1594.
    pub fn get_hashes(&mut self) {
        self.hash[0] = 0xc01df;
        self.hash[1] = !self.hash[0];
        for i in self.start..=self.end {
            let l = self.letters[i as usize];
            self.hash[0] ^= hash3(self.hash[0], l as u64, i as u64);
            let folded = if (l & 0x80) == 0 {
                l & 0x5F
            } else if (l & 0xC0) == 0x80 {
                l & 0x3F
            } else if (l & 0xE0) == 0xC0 {
                l & 0x1F
            } else if (l & 0xF0) == 0xE0 {
                l & 0xF
            } else {
                l & 0x7
            };
            self.hash[1] ^= super::substrate::hash2(self.hash[1], folded as u64);
        }
        self.hash[2] = (!self.hash[0]) ^ self.hash[1];
        self.hash[3] = (!self.hash[1]) ^ self.hash[0];
    }

    /// `ChangeSuffix(old, new)` — paq8.cpp:1595-1608.
    pub fn change_suffix(&mut self, old: &[u8], new: &[u8]) -> bool {
        let len = old.len();
        if (self.length() as usize) > len {
            let start = self.end as usize - len + 1;
            if self.letters[start..start + len] == *old {
                if !new.is_empty() {
                    let n = new.len();
                    let new_end = (MAX_WORD_SIZE - 1)
                        .min(self.end as usize + n)
                        .saturating_sub(self.end as usize);
                    let copy_len = new_end;
                    let dst_start = self.end as usize - len + 1;
                    for k in 0..copy_len.min(n) {
                        self.letters[dst_start + k] = new[k];
                    }
                    self.end = ((MAX_WORD_SIZE - 1)
                        .min(self.end as usize - len + n)) as u8;
                } else {
                    self.end -= len as u8;
                }
                return true;
            }
        }
        false
    }

    /// `MatchesAny(a[], count)` — paq8.cpp:1609-1614.
    pub fn matches_any(&self, list: &[&[u8]]) -> bool {
        let len = self.length() as usize;
        for &cand in list {
            if cand.len() == len
                && self.letters[self.start as usize..self.start as usize + len]
                    == *cand
            {
                return true;
            }
        }
        false
    }

    pub fn ends_with(&self, suffix: &[u8]) -> bool {
        let len = suffix.len();
        (self.length() as usize) > len
            && {
                let start = self.end as usize - len + 1;
                self.letters[start..start + len] == *suffix
            }
    }

    pub fn starts_with(&self, prefix: &[u8]) -> bool {
        let len = prefix.len();
        (self.length() as usize) > len
            && self.letters[self.start as usize..self.start as usize + len]
                == *prefix
    }
}

// =============================================================
// Segment / Sentence / Paragraph — paq8.cpp:1625-1651.
// =============================================================

#[derive(Clone, Default)]
pub struct Segment {
    pub first_word: Word,
    pub word_count: u32,
    pub num_count:  u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SentenceType {
    Declarative   = 0,
    Interrogative = 1,
    Exclamative   = 2,
}

pub const SENTENCE_TYPE_COUNT: usize = 3;

#[derive(Clone)]
pub struct Sentence {
    pub segment:       Segment,
    pub r#type:        SentenceType,
    pub segment_count: u32,
    pub verb_index:    u32,
    pub noun_index:    u32,
    pub capital_index: u32,
    pub last_verb:     Word,
    pub last_noun:     Word,
    pub last_capital:  Word,
}

impl Default for Sentence {
    fn default() -> Self {
        Self {
            segment: Segment::default(),
            r#type: SentenceType::Declarative,
            segment_count: 0,
            verb_index: 0,
            noun_index: 0,
            capital_index: 0,
            last_verb: Word::new(),
            last_noun: Word::new(),
            last_capital: Word::new(),
        }
    }
}

#[derive(Clone, Default)]
pub struct Paragraph {
    pub sentence_count: u32,
    pub type_count:     [u32; SENTENCE_TYPE_COUNT],
    pub type_mask:      u32,
}

// =============================================================
// English / French / German abbreviation tables — paq8.cpp.
// =============================================================

pub const ENGLISH_ABBREVIATIONS: [&[u8]; 6] =
    [b"mr", b"mrs", b"ms", b"dr", b"st", b"jr"];
pub const FRENCH_ABBREVIATIONS: [&[u8]; 2] = [b"m", b"mm"];
pub const GERMAN_ABBREVIATIONS: [&[u8]; 3] = [b"fr", b"hr", b"hrn"];

pub fn is_english_abbreviation(w: &Word) -> bool {
    w.matches_any(&ENGLISH_ABBREVIATIONS)
}
pub fn is_french_abbreviation(w: &Word) -> bool {
    w.matches_any(&FRENCH_ABBREVIATIONS)
}
pub fn is_german_abbreviation(w: &Word) -> bool {
    w.matches_any(&GERMAN_ABBREVIATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_push_and_compare() {
        let mut w = Word::new();
        for &c in b"Hello" { w.push(c); }
        assert!(w.equals_str(b"hello"), "push lowercases input");
        assert_eq!(w.length(), 5);
        assert_eq!(w.at(0), b'h');
        assert_eq!(w.from_end(0), b'o');
    }

    #[test]
    fn word_change_suffix_swap() {
        let mut w = Word::new();
        for &c in b"running" { w.push(c); }
        assert!(w.change_suffix(b"ing", b"") , "suffix removed");
        assert!(w.equals_str(b"runn"));
    }

    #[test]
    fn word_ends_starts_with() {
        let mut w = Word::new();
        for &c in b"recompile" { w.push(c); }
        assert!(w.starts_with(b"recom"));
        assert!(w.ends_with(b"pile"));
        assert!(!w.ends_with(b"xyz"));
    }

    #[test]
    fn word_hashes_change_with_letters() {
        let mut a = Word::new();
        let mut b = Word::new();
        for &c in b"alpha" { a.push(c); }
        for &c in b"alphb" { b.push(c); }
        a.get_hashes();
        b.get_hashes();
        assert_ne!(a.hash, b.hash);
    }

    #[test]
    fn english_abbreviation_detection() {
        let mut w = Word::new();
        for &c in b"mr" { w.push(c); }
        assert!(is_english_abbreviation(&w));
        let mut x = Word::new();
        for &c in b"hello" { x.push(c); }
        assert!(!is_english_abbreviation(&x));
    }
}
