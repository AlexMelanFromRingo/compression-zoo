//! `TextModel` — paq8.cpp:3071-3519.
//!
//! The largest paq8 sub-model: a language-aware text model with
//! word/segment/sentence/paragraph caches, a parse state machine,
//! 3-language detection (English/French/German), and a 33-context
//! `ContextMap2`. Drives 8 mixer-set contexts.

#![allow(dead_code)]

use super::context_map::ContextMap2;
use super::mixer::Mixer;
use super::stats::ModelStats;
use super::stemmer::{EnglishStemmer, FrenchStemmer, GermanStemmer};
use super::substrate::{
    finalize64, hash2, hash3, hash4, hash5, ilog2, llog, Buf, Ilog,
    Squash, Stretch,
};
use super::util::Cache;
use super::word::{
    is_english_abbreviation, is_french_abbreviation, is_german_abbreviation,
    lang, Paragraph, Segment, Sentence, SentenceType, Word,
};

const MIN_RECOGNIZED_WORDS: u32 = 4;
/// Language slots: 0 = Unknown, 1 = English, 2 = French, 3 = German.
const LANG_COUNT: usize = 4;

const TAB: u8 = 0x09;
const NEW_LINE: u8 = 0x0A;
const CARRIAGE_RETURN: u8 = 0x0D;
const SPACE: u8 = 0x20;

/// AsciiGroup table — paq8.cpp:3052-3069 (used by `Update`'s
/// `g = AsciiGroup[c]`).
const ASCII_GROUP: [u8; 128] = [
    0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 7, 8, 17, 17, 9, 17, 10, 11, 12, 17, 17, 13, 14, 15, 16,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 18, 19, 20, 23, 21, 22,
    23, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 24, 27, 25, 27, 26,
    27, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 28, 30, 29, 30, 30,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Parse {
    Unknown = 0,
    ReadingWord,
    PossibleHyphenation,
    WasAbbreviation,
    AfterComma,
    AfterQuote,
    AfterAbbreviation,
    ExpectDigit,
}

#[derive(Clone, Copy, Default)]
struct LangState {
    count: [u32; LANG_COUNT - 1],
    mask:  [u64; LANG_COUNT - 1],
    id:    usize,  // 0 = Unknown
    p_id:  usize,
}

#[derive(Clone, Default)]
struct Info {
    numbers:    [u64; 2],
    num_hashes: [u64; 2],
    num_length: [u8; 2],
    num_mask:   u32,
    num_diff:   u32,
    last_upper: u32,
    mask_upper: u32,
    last_letter: u32,
    last_digit:  u32,
    last_punct:  u32,
    last_new_line: u32,
    prev_new_line: u32,
    word_gap:    u32,
    spaces:      u32,
    space_count: u32,
    commas:      u32,
    quote_length: u32,
    mask_punct:  u32,
    nest_hash:   u32,
    last_nest:   u32,
    ascii_mask:  u64,
    masks:       [u32; 5],
    word_length: [u32; 2],
    utf8_remaining: i32,
    first_letter: u8,
    first_char:   u8,
    expected_digit: u8,
    prev_punct:   u8,
    topic_descriptor: Word,
}

pub struct TextModel {
    map:        ContextMap2,
    /// Per-language word ring caches (Unknown / English / French / German).
    words:      Vec<Cache<Word>>,
    segments:   Cache<Segment>,
    sentences:  Cache<Sentence>,
    paragraphs: Cache<Paragraph>,
    word_pos:   Vec<u32>,
    byte_pos:   [u32; 256],
    /// (lang_index, cache_offset) of the current / previous word.
    c_word: (usize, u32),
    p_word: (usize, u32),
    state:  Parse,
    p_state: Parse,
    l:      LangState,
    info:   Info,
    parse_ctx: u64,
}

impl TextModel {
    pub fn new(size: u64, dt: [i32; 1024]) -> Self {
        let words = (0..LANG_COUNT).map(|_| Cache::<Word>::new(8)).collect();
        Self {
            map: ContextMap2::new(size, 33, dt),
            words,
            segments:   Cache::new(4),
            sentences:  Cache::new(4),
            paragraphs: Cache::new(2),
            word_pos:   vec![0u32; 0x10000],
            byte_pos:   [0u32; 256],
            c_word: (0, 0),
            p_word: (0, 1),
            state:  Parse::Unknown,
            p_state: Parse::Unknown,
            l:      LangState::default(),
            info:   Info::default(),
            parse_ctx: 0,
        }
    }

    /// Stem `Words[lang](0)` with the matching language stemmer;
    /// returns `true` if the stemmer recognised the word.
    fn stem(lang: usize, w: &mut Word) -> bool {
        match lang {
            1 => EnglishStemmer::stem(w),
            2 => FrenchStemmer::stem(w),
            3 => GermanStemmer::stem(w),
            _ => false,
        }
    }

    fn is_abbreviation(lang: usize, w: &Word) -> bool {
        match lang {
            1 => is_english_abbreviation(w),
            2 => is_french_abbreviation(w),
            3 => is_german_abbreviation(w),
            _ => false,
        }
    }

    fn is_vowel(lang: usize, c: u8) -> bool {
        match lang {
            1 => EnglishStemmer::is_vowel(c),
            2 => FrenchStemmer::is_vowel(c),
            3 => GermanStemmer::is_vowel(c),
            _ => false,
        }
    }

    #[inline]
    fn cword(&self) -> &Word { self.words[self.c_word.0].at(self.c_word.1) }
    #[inline]
    fn cword_mut(&mut self) -> &mut Word {
        self.words[self.c_word.0].at_mut(self.c_word.1)
    }
    #[inline]
    fn pword(&self) -> &Word { self.words[self.p_word.0].at(self.p_word.1) }

    fn update(&mut self, buffer: &Buf, pos: u32, c0: u32, stats: &mut ModelStats) {
        let info = &mut self.info;
        info.last_upper  = 0xFF.min(info.last_upper + 1);
        info.mask_upper <<= 1;
        info.last_letter = 0x1F.min(info.last_letter + 1);
        info.last_digit  = 0xFF.min(info.last_digit + 1);
        info.last_punct  = 0x3F.min(info.last_punct + 1);
        info.last_new_line += 1;
        info.prev_new_line += 1;
        info.last_nest     += 1;
        info.space_count -= info.spaces >> 31;
        info.spaces <<= 1;
        info.masks[0] <<= 2;
        info.masks[1] <<= 2;
        info.masks[2] <<= 4;
        info.masks[3] <<= 3;
        self.p_state = self.state;

        let mut c = buffer.at(1);
        let p_c_lower = c.to_ascii_lowercase();
        let g = if c < 0x80 { ASCII_GROUP[c as usize] } else { 31 };
        if !((g <= 4) && (g as u64) == (self.info.ascii_mask & 0x1f)) {
            self.info.ascii_mask =
                ((self.info.ascii_mask << 5) | g as u64) & ((1u64 << 60) - 1);
        }
        self.info.masks[4] = (self.info.ascii_mask & ((1 << 30) - 1)) as u32;
        self.byte_pos[c as usize] = pos;
        if c != p_c_lower {
            c = p_c_lower;
            self.info.last_upper = 0;
            self.info.mask_upper |= 1;
        }
        let p_c = buffer.at(2);
        self.state = Parse::Unknown;
        self.parse_ctx = hash5(
            self.state as u64,
            self.pword().hash[1],
            c as u64,
            ((ilog2(self.info.last_new_line) + 1)
                * ((self.info.last_new_line * 3 > self.info.prev_new_line) as u32))
                as u64,
            (self.info.masks[1] & 0xFC) as u64,
        );

        let is_word_char = (c >= b'a' && c <= b'z')
            || c == b'\'' || c == b'-' || c > 0x7F;
        if is_word_char {
            if self.info.word_length[0] == 0 {
                // Hyphenation with "+" (rare branch).
                let hyphen = (self.info.last_letter == 3
                    && buffer.at(3) == b'+')
                    || (self.info.last_letter == 4
                        && buffer.at(3) == CARRIAGE_RETURN
                        && buffer.at(4) == b'+');
                if p_c == NEW_LINE && hyphen {
                    self.info.word_length[0] = self.info.word_length[1];
                    for i in 0..LANG_COUNT {
                        self.words[i].retreat();
                    }
                    // cWord = pWord; pWord = &Words[Lang.pId](1)
                    self.c_word = self.p_word;
                    self.p_word = (self.l.p_id, 1);
                    *self.cword_mut() = Word::new();
                    let wl = self.info.word_length[0];
                    let ll = self.info.last_letter;
                    for k in 0..wl {
                        let ch = buffer.at(wl - k + ll);
                        self.cword_mut().push(ch);
                    }
                    self.info.word_length[1] = self.pword().length();
                    let sc = self.segments.at_mut(0);
                    sc.word_count = sc.word_count.saturating_sub(1);
                    let stc = self.sentences.at_mut(0);
                    stc.segment.word_count =
                        stc.segment.word_count.saturating_sub(1);
                } else {
                    self.info.word_gap = self.info.last_letter;
                    self.info.first_letter = c;
                }
            }
            self.info.last_letter = 0;
            self.info.word_length[0] += 1;
            let vow_bonus = if self.l.id != 0 {
                1 + Self::is_vowel(self.l.id, c) as u32
            } else { 1 };
            self.info.masks[0] += vow_bonus;
            self.info.masks[1] += 1;
            self.info.masks[3] += self.info.masks[0] & 3;
            if c == b'\'' {
                self.info.masks[2] += 12;
                if self.info.word_length[0] == 1 {
                    if self.info.quote_length == 0 && p_c == SPACE {
                        self.info.quote_length = 1;
                    } else if self.info.quote_length > 0
                        && self.info.last_punct == 1
                    {
                        self.info.quote_length = 0;
                        self.state = Parse::AfterQuote;
                        self.parse_ctx = hash2(self.state as u64, p_c as u64);
                    }
                }
            }
            self.cword_mut().push(c);
            self.cword_mut().get_hashes();
            self.state = Parse::ReadingWord;
            self.parse_ctx = hash2(self.state as u64, self.cword().hash[1]);
        } else {
            if self.cword().length() > 0 {
                // Copy current word into the Unknown cache slot 0.
                if self.l.id != 0 {
                    let cw = self.cword().clone();
                    *self.words[0].at_mut(0) = cw;
                }
                // Try each non-Unknown language stemmer.
                for i in (1..LANG_COUNT).rev() {
                    self.l.count[i - 1] -= (self.l.mask[i - 1] >> 63) as u32;
                    self.l.mask[i - 1] <<= 1;
                    if i != self.l.id {
                        let cw = self.cword().clone();
                        *self.words[i].at_mut(0) = cw;
                    }
                    let mut w = self.words[i].at(0).clone();
                    let recognised = Self::stem(i, &mut w);
                    *self.words[i].at_mut(0) = w;
                    if recognised {
                        self.l.count[i - 1] += 1;
                        self.l.mask[i - 1] |= 1;
                    }
                }
                self.l.id = 0;
                let mut best = MIN_RECOGNIZED_WORDS;
                for i in (1..LANG_COUNT).rev() {
                    if self.l.count[i - 1] >= best {
                        best = self.l.count[i - 1]
                            + (i == self.l.p_id) as u32;
                        self.l.id = i;
                    }
                    self.words[i].advance();
                }
                self.words[0].advance();
                self.l.p_id = self.l.id;
                self.p_word = (self.l.id, 1);
                self.c_word = (self.l.id, 0);
                *self.cword_mut() = Word::new();
                let pw_hash1 = self.pword().hash[1];
                let wp_mask = self.word_pos.len() - 1;
                self.word_pos[(pw_hash1 as usize) & wp_mask] = pos;
                if self.segments.at(0).word_count == 0 {
                    let pw = self.pword().clone();
                    self.segments.at_mut(0).first_word = pw;
                }
                self.segments.at_mut(0).word_count += 1;
                if self.sentences.at(0).segment.word_count == 0 {
                    let pw = self.pword().clone();
                    self.sentences.at_mut(0).segment.first_word = pw;
                }
                self.sentences.at_mut(0).segment.word_count += 1;
                self.info.word_length[1] = self.info.word_length[0];
                self.info.word_length[0] = 0;
                self.info.quote_length += (self.info.quote_length > 0) as u32;
                if self.info.quote_length > 0x1F {
                    self.info.quote_length = 0;
                }
                {
                    let st = self.sentences.at_mut(0);
                    st.verb_index += 1;
                    st.noun_index += 1;
                    st.capital_index += 1;
                }
                let pw_type = self.pword().r#type;
                if pw_type & lang::VERB != 0 {
                    let pw = self.pword().clone();
                    let st = self.sentences.at_mut(0);
                    st.verb_index = 0;
                    st.last_verb = pw;
                }
                if pw_type & lang::NOUN != 0 {
                    let pw = self.pword().clone();
                    let st = self.sentences.at_mut(0);
                    st.noun_index = 0;
                    st.last_noun = pw;
                }
                if self.sentences.at(0).segment.word_count > 1
                    && self.info.last_upper < self.info.word_length[1]
                {
                    let pw = self.pword().clone();
                    let st = self.sentences.at_mut(0);
                    st.capital_index = 0;
                    st.last_capital = pw;
                }
            }
            self.punctuation_branch(buffer, c, p_c, p_c_lower);
        }
        if self.info.last_new_line == 1 {
            self.info.first_char = if self.l.id != 0 { c } else { c.min(96) };
        }
        if self.info.last_nest > 512 {
            self.info.nest_hash = 0;
        }
        // UTF-8 tracking.
        let mut leading_bits_set = 0;
        while ((c >> (7 - leading_bits_set)) & 1) != 0 && leading_bits_set < 8 {
            leading_bits_set += 1;
        }
        if self.info.utf8_remaining > 0 && leading_bits_set == 1 {
            self.info.utf8_remaining -= 1;
        } else {
            self.info.utf8_remaining = if leading_bits_set != 1 {
                if c != 0xC0 && c != 0xC1 && c < 0xF5 {
                    (leading_bits_set as i32) - (leading_bits_set > 0) as i32
                } else { -1 }
            } else { 0 };
        }
        self.info.mask_punct =
            ((self.byte_pos[b',' as usize] > self.byte_pos[b'.' as usize]) as u32)
            | (((self.byte_pos[b',' as usize] > self.byte_pos[b'!' as usize]) as u32) << 1)
            | (((self.byte_pos[b',' as usize] > self.byte_pos[b'?' as usize]) as u32) << 2)
            | (((self.byte_pos[b',' as usize] > self.byte_pos[b':' as usize]) as u32) << 3)
            | (((self.byte_pos[b',' as usize] > self.byte_pos[b';' as usize]) as u32) << 4);

        stats.text.state = self.state as u8;
        stats.text.last_punct = 0x1F.min(self.info.last_punct) as u8;
        stats.text.word_length = 0xF.min(self.info.word_length[0]) as u8;
        stats.text.boolmask =
            ((self.info.last_digit < self.info.word_length[0] + self.info.word_gap) as u8)
            | (((self.info.last_upper < self.info.last_letter + self.info.word_length[1]) as u8) << 1)
            | (((self.info.last_punct < self.info.word_length[0] + self.info.word_gap) as u8) << 2)
            | (((self.info.last_upper < self.info.word_length[0]) as u8) << 3);
        stats.text.first_letter = self.info.first_letter;
        stats.text.mask = (self.info.masks[1] & 0xFF) as u8;
        let _ = c0;
    }

    /// The punctuation / whitespace / bracket / digit switch
    /// (paq8.cpp:3299-3401).
    fn punctuation_branch(&mut self, buffer: &Buf, c: u8, p_c: u8,
                            _p_c_lower: u8) {
        let mut skip = false;
        // `.` abbreviation special case.
        if c == b'.' && self.l.id != 0
            && self.info.last_upper == self.info.word_length[1]
            && Self::is_abbreviation(self.l.id, self.pword())
        {
            self.state = Parse::WasAbbreviation;
            self.parse_ctx = hash2(self.state as u64, self.pword().hash[1]);
        } else {
            // `.`/`?`/`!` (sentence terminators) fall through into
            // `,`/`;`/`:` (segment separators).
            let mut handled_sentence = false;
            if c == b'.' || c == b'?' || c == b'!' {
                let stype = if c == b'.' {
                    SentenceType::Declarative
                } else if c == b'?' {
                    SentenceType::Interrogative
                } else {
                    SentenceType::Exclamative
                };
                self.sentences.at_mut(0).r#type = stype;
                self.sentences.at_mut(0).segment_count += 1;
                self.paragraphs.at_mut(0).sentence_count += 1;
                let ti = stype as usize;
                self.paragraphs.at_mut(0).type_count[ti] += 1;
                self.paragraphs.at_mut(0).type_mask <<= 2;
                self.paragraphs.at_mut(0).type_mask |= ti as u32;
                *self.sentences.next() = Sentence::default();
                self.info.masks[3] += 3;
                skip = true;
                handled_sentence = true;
            }
            if handled_sentence || c == b',' || c == b';' || c == b':' {
                if c == b',' {
                    self.info.commas += 1;
                    self.state = Parse::AfterComma;
                    self.parse_ctx = hash4(
                        self.state as u64,
                        ilog2(self.info.quote_length + 1) as u64,
                        ilog2(self.info.last_new_line) as u64,
                        (self.info.last_upper
                            < self.info.last_letter + self.info.word_length[1])
                            as u64,
                    );
                } else if c == b':' {
                    self.info.topic_descriptor = self.pword().clone();
                }
                if !skip {
                    self.sentences.at_mut(0).segment_count += 1;
                    self.info.masks[3] += 4;
                }
                self.info.last_punct = 0;
                self.info.prev_punct = c;
                self.info.masks[0] += 3;
                self.info.masks[1] += 2;
                self.info.masks[2] += 15;
                *self.segments.next() = Segment::default();
            } else {
                // Whitespace + brackets + the rest.
                let mut handled_nl = false;
                if c == NEW_LINE {
                    self.info.prev_new_line = self.info.last_new_line;
                    self.info.last_new_line = 0;
                    self.info.commas = 0;
                    if self.info.prev_new_line == 1
                        || (self.info.prev_new_line == 2 && p_c == CARRIAGE_RETURN)
                    {
                        *self.paragraphs.next() = Paragraph::default();
                    } else if (self.info.last_letter == 2 && p_c == b'+')
                        || (self.info.last_letter == 3 && p_c == CARRIAGE_RETURN
                            && buffer.at(3) == b'+')
                    {
                        self.parse_ctx = hash2(Parse::ReadingWord as u64,
                            self.pword().hash[1]);
                        self.state = Parse::PossibleHyphenation;
                    }
                    handled_nl = true;
                }
                if handled_nl || c == TAB || c == CARRIAGE_RETURN || c == SPACE {
                    self.info.space_count += 1;
                    self.info.spaces |= 1;
                    self.info.masks[1] += 3;
                    self.info.masks[3] += 5;
                    if c == SPACE && self.p_state == Parse::WasAbbreviation {
                        self.state = Parse::AfterAbbreviation;
                        self.parse_ctx = hash2(self.state as u64,
                            self.pword().hash[1]);
                    }
                } else {
                    match c {
                        b'(' => { self.info.masks[2] += 1; self.info.masks[3] += 6;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_add(31);
                                   self.info.last_nest = 0; }
                        b'[' => { self.info.masks[2] += 2;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_add(11);
                                   self.info.last_nest = 0; }
                        b'{' => { self.info.masks[2] += 3;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_add(17);
                                   self.info.last_nest = 0; }
                        b'<' => { self.info.masks[2] += 4;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_add(23);
                                   self.info.last_nest = 0; }
                        0xAB => { self.info.masks[2] += 5; }
                        b')' => { self.info.masks[2] += 6;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_sub(31);
                                   self.info.last_nest = 0; }
                        b']' => { self.info.masks[2] += 7;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_sub(11);
                                   self.info.last_nest = 0; }
                        b'}' => { self.info.masks[2] += 8;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_sub(17);
                                   self.info.last_nest = 0; }
                        b'>' => { self.info.masks[2] += 9;
                                   self.info.nest_hash = self.info.nest_hash.wrapping_sub(23);
                                   self.info.last_nest = 0; }
                        0xBB => { self.info.masks[2] += 10; }
                        b'"' => {
                            self.info.masks[2] += 11;
                            if self.info.quote_length == 0 {
                                self.info.quote_length = 1;
                            } else {
                                self.info.quote_length = 0;
                                self.state = Parse::AfterQuote;
                                self.parse_ctx = hash2(self.state as u64,
                                    0x100 | p_c as u64);
                            }
                        }
                        b'/' | b'-' | b'+' | b'*' | b'=' | b'%' => {
                            self.info.masks[2] += 13;
                        }
                        b'\\' | b'|' | b'_' | b'@' | b'&' | b'^' => {
                            self.info.masks[2] += 14;
                        }
                        _ => {}
                    }
                }
            }
        }
        // Digit handling.
        if c >= b'0' && c <= b'9' {
            self.info.numbers[0] = self.info.numbers[0] * 10 + (c & 0xF) as u64;
            self.info.num_length[0] = 19.min(self.info.num_length[0] + 1);
            self.info.num_hashes[0] = super::substrate::combine64(
                self.info.num_hashes[0], c as u64);
            self.info.expected_digit = 0xFF; // upstream's -1 as U8
            if self.info.num_length[0] < self.info.num_length[1]
                && (self.p_state == Parse::ExpectDigit
                    || ((self.info.num_diff & 3) == 0
                        && self.info.num_length[0] <= 1))
            {
                let expected_num = self.info.numbers[1]
                    .wrapping_add((self.info.num_mask & 3) as u64)
                    .wrapping_sub(2);
                let mut place_divisor: u64 = 1;
                for _ in 0..(self.info.num_length[1] - self.info.num_length[0]) {
                    place_divisor *= 10;
                }
                if place_divisor != 0
                    && expected_num / place_divisor == self.info.numbers[0]
                {
                    place_divisor /= 10;
                    if place_divisor != 0 {
                        self.info.expected_digit =
                            ((expected_num / place_divisor) % 10) as u8;
                    }
                    self.state = Parse::ExpectDigit;
                }
            } else {
                let d = buffer.at(self.info.num_length[0] as u32 + 2);
                if self.info.num_length[0] < 3
                    && buffer.at(self.info.num_length[0] as u32 + 1) == b','
                    && (b'0'..=b'9').contains(&d)
                {
                    self.state = Parse::ExpectDigit;
                }
            }
            self.info.last_digit = 0;
            self.info.masks[3] += 7;
        } else if self.info.numbers[0] > 0 {
            self.info.num_mask <<= 2;
            self.info.num_mask |= 1
                + (self.info.numbers[0] >= self.info.numbers[1]) as u32
                + (self.info.numbers[0] > self.info.numbers[1]) as u32;
            let diff = (self.info.numbers[0] as i64
                - self.info.numbers[1] as i64).unsigned_abs() as u32;
            self.info.num_diff <<= 2;
            self.info.num_diff |= 3.min(ilog2(diff));
            self.info.numbers[1] = self.info.numbers[0];
            self.info.numbers[0] = 0;
            self.info.num_hashes[1] = self.info.num_hashes[0];
            self.info.num_hashes[0] = 0;
            self.info.num_length[1] = self.info.num_length[0];
            self.info.num_length[0] = 0;
            self.segments.at_mut(0).num_count += 1;
            self.sentences.at_mut(0).segment.num_count += 1;
        }
    }

    /// `SetContexts` — paq8.cpp:3428-3519.
    fn set_contexts(&mut self, buffer: &Buf, pos: u32, ilog: &Ilog) {
        let c = buffer.at(1);
        let lc = c.to_ascii_lowercase();
        let m2 = self.info.masks[2] & 0xF;
        let column = 0xFF.min(self.info.last_new_line);
        let w = (if self.state == Parse::ReadingWord {
            self.cword().hash[1]
        } else {
            self.pword().hash[1]
        }) & 0xFFFF;
        let h = (if self.state == Parse::ReadingWord {
            self.cword().hash[1]
        } else {
            self.pword().hash[2]
        }).wrapping_mul(271).wrapping_add(c as u64);
        let mut i = (self.state as u64) << 6;

        let cw = self.cword().clone();
        let pw = self.pword().clone();
        let info = self.info.clone();
        let lang_p = self.l.p_id;

        macro_rules! wp { ($off:expr) => { self.words[lang_p].at($off).clone() } }

        self.map.set(self.parse_ctx);
        self.map.set(hash4(i, cw.hash[0], pw.hash[0],
            ((info.last_upper < info.word_length[0]) as u64)
            | (((info.last_digit < info.word_length[0] + info.word_gap) as u64) << 1)));
        i += 1;
        self.map.set(hash5(i, cw.hash[1], wp!(2).hash[1],
            10.min(ilog2(info.numbers[0] as u32)) as u64,
            ((info.last_upper < info.last_letter + info.word_length[1]) as u64)
            | (((info.last_letter > 3) as u64) << 1)
            | (((info.last_letter > 0 && info.word_length[1] < 3) as u64) << 2)));
        i += 1;
        self.map.set(hash5(i, cw.hash[1] & 0xFFF,
            (info.masks[1] & 0x3FF) as u64, wp!(3).hash[2],
            ((info.last_digit < info.word_length[0] + info.word_gap) as u64)
            | (((info.last_upper < info.last_letter + info.word_length[1]) as u64) << 1)
            | (((info.spaces & 0x7F) as u64) << 2)));
        i += 1;
        self.map.set(hash4(i, cw.hash[1], pw.hash[3], wp!(2).hash[3]));
        i += 1;
        self.map.set(hash4(i, h & 0x7FFF,
            wp!(2).hash[1] & 0xFFF, wp!(3).hash[1] & 0xFFF));
        i += 1;
        let st0_lastverb = {
            let st = self.sentences.at(0);
            if st.verb_index < st.segment.word_count {
                st.last_verb.hash[1]
            } else { 0 }
        };
        self.map.set(hash4(i, cw.hash[1], c as u64, st0_lastverb));
        i += 1;
        self.map.set(hash5(i, pw.hash[2], (info.masks[1] & 0xFC) as u64,
            lc as u64, info.word_gap as u64));
        i += 1;
        let seg0_fw_h2 = self.segments.at(0).first_word.hash[2];
        let seg0_wc = self.segments.at(0).word_count;
        self.map.set(hash5(i,
            if info.last_letter == 0 { cw.hash[1] } else { pw.hash[1] },
            c as u64, seg0_fw_h2, 3.min(ilog2(seg0_wc + 1)) as u64));
        i += 1;
        let seg1_fw_h3 = self.segments.at(1).first_word.hash[3];
        self.map.set(hash4(i, cw.hash[1], c as u64, seg1_fw_h3));
        i += 1;
        self.map.set(hash5(i, 31u8.max(lc) as u64,
            (info.masks[1] & 0xFFC) as u64,
            ((info.spaces & 0xFE) | (info.last_punct < info.last_letter) as u32) as u64,
            ((info.mask_upper & 0xFF)
                | (((0x100 | info.first_letter as u32)
                    * (info.word_length[0] > 1) as u32) << 8)) as u64));
        i += 1;
        self.map.set(hash4(i, column as u64,
            7.min(ilog2(info.last_upper + 1)) as u64,
            ilog2(info.last_punct + 1) as u64));
        i += 1;
        self.map.set(
            ((column & 0xF8)
            | (info.masks[1] & 3)
            | (((info.prev_new_line.wrapping_sub(info.last_new_line) > 63) as u32) << 2)
            | (3.min(info.last_letter) << 8)
            | ((info.first_char as u32) << 10)
            | (((info.commas > 4) as u32) << 18)
            | (((m2 >= 1 && m2 <= 5) as u32) << 19)
            | (((m2 >= 6 && m2 <= 10) as u32) << 20)
            | (((m2 == 11 || m2 == 12) as u32) << 21)
            | (((info.last_upper < column) as u32) << 22)
            | (((info.last_digit < column) as u32) << 23)
            | (((column < info.prev_new_line.wrapping_sub(info.last_new_line)) as u32) << 24))
            as u64);
        self.map.set(hash5(
            ((2 * column) / 3) as u64,
            (13.min(info.last_punct) + (info.last_punct > 16) as u32
                + (info.last_punct > 32) as u32 + info.mask_punct * 16) as u64,
            ilog2(info.last_upper + 1) as u64,
            ilog2(info.prev_new_line.wrapping_sub(info.last_new_line)) as u64,
            (((info.masks[1] & 3) == 0) as u64)
            | (((m2 < 6) as u64) << 1)
            | (((m2 < 11) as u64) << 2)));
        self.map.set(hash3(i, (column >> 1) as u64,
            (info.spaces & 0xF) as u64));
        i += 1;
        self.map.set(hash5(
            (info.masks[3] & 0x3F) as u64,
            (((info.word_length[0].max(3) - 2) * (info.word_length[0] < 8) as u32)
                .min(3)) as u64,
            (info.first_letter as u32 * (info.word_length[0] < 5) as u32) as u64,
            (w & 0x3FF) as u64,
            ((c == buffer.at(2)) as u64)
            | (((info.masks[2] > 0) as u64) << 1)
            | (((info.last_punct < info.word_length[0] + info.word_gap) as u64) << 2)
            | (((info.last_upper < info.word_length[0]) as u64) << 3)
            | (((info.last_digit < info.word_length[0] + info.word_gap) as u64) << 4)
            | (((info.last_punct < 2 + info.word_length[0] + info.word_gap
                + info.word_length[1]) as u64) << 5)));
        self.map.set(hash4(i, w, c as u64, info.num_hashes[1]));
        i += 1;
        let wp_dist = llog(ilog, pos.wrapping_sub(
            self.word_pos[(w as usize) & (self.word_pos.len() - 1)])) >> 1;
        self.map.set(hash4(i, w, c as u64, wp_dist as u64));
        i += 1;
        self.map.set(hash4(i, w, c as u64,
            info.topic_descriptor.hash[1] & 0x7FFF));
        i += 1;
        self.map.set(hash4(i, info.num_length[0] as u64, c as u64,
            info.topic_descriptor.hash[1] & 0x7FFF));
        i += 1;
        self.map.set(hash4(i,
            if info.last_letter > 0 { c as u64 } else { 0x100 },
            (info.masks[1] & 0xFFC) as u64,
            (info.nest_hash & 0x7FF) as u64));
        i += 1;
        let (st0_vi, st0_lv_len, st0_sc, st0_wc) = {
            let st = self.sentences.at(0);
            (st.verb_index, st.last_verb.length(),
             st.segment_count, st.segment.word_count)
        };
        self.map.set(hash3(i, (w as u64) * 17 + c as u64,
            ((info.masks[3] & 0x1FF) as u64)
            | (((st0_vi == 0 && st0_lv_len > 0) as u64) << 6)
            | (((info.word_length[1] > 3) as u64) << 5)
            | (((seg0_wc == 0) as u64) << 4)
            | (((st0_sc == 0 && st0_wc < 2) as u64) << 3)
            | (((info.last_punct >= info.last_letter + info.word_length[1]
                + info.word_gap) as u64) << 2)
            | (((info.last_upper < info.last_letter + info.word_length[1]) as u64) << 1)
            | ((info.last_upper < info.word_length[0] + info.word_gap
                + info.word_length[1]) as u64)));
        i += 1;
        self.map.set(hash4(i, c as u64, pw.hash[2],
            (info.first_letter as u32 * (info.word_length[0] < 6) as u32) as u64
            | ((((info.last_punct < info.word_length[0] + info.word_gap) as u64) << 1)
                | ((info.last_punct >= info.last_letter + info.word_length[1]
                    + info.word_gap) as u64))));
        i += 1;
        let wp_idx = 1 + (info.word_length[0] == 0) as u32;
        let wp_first = {
            let wword = self.words[lang_p].at(wp_idx);
            wword.letters[wword.start as usize]
        };
        self.map.set(hash4(i, (w as u64) * 23 + c as u64, wp_first as u64,
            (info.first_letter as u32 * (info.word_length[0] < 7) as u32) as u64));
        i += 1;
        self.map.set(hash3(i, column as u64,
            ((info.spaces & 7) as u64)
            | (((info.nest_hash & 0x7FF) as u64) << 3)));
        i += 1;
        self.map.set(hash4(i, cw.hash[1],
            (info.last_upper < column) as u64
            | (((info.last_upper < info.word_length[0]) as u64) << 1),
            5.min(info.word_length[0]) as u64));
        i += 1;
        self.map.set(info.masks[4] as u64);
        self.map.set(hash2(info.ascii_mask as u32 as u64,
            (info.ascii_mask >> 32) as u64));
        self.map.set(info.ascii_mask & ((1 << 20) - 1));
        self.map.set(info.ascii_mask & ((1 << 10) - 1));
        self.map.set(hash2((info.ascii_mask >> 5) & ((1 << 30) - 1),
            buffer.at(1) as u64));
        self.map.set(hash3((info.ascii_mask >> 10) & ((1 << 30) - 1),
            buffer.at(1) as u64, buffer.at(2) as u64));
        self.map.set(hash4((info.ascii_mask >> 15) & ((1 << 30) - 1),
            buffer.at(1) as u64, buffer.at(2) as u64, buffer.at(3) as u64));
        let _ = i;
    }

    /// `Predict` — paq8.cpp:3158-3186.
    pub fn predict(&mut self, m: &mut Mixer, buffer: &Buf,
                   pos: u32, c0: u32, bpos: i32, y: i32, grp0: u8,
                   ilog: &Ilog, squash: &Squash, stretch: &Stretch,
                   stats: &mut ModelStats) {
        if bpos == 0 {
            self.update(buffer, pos, c0, stats);
            self.set_contexts(buffer, pos, ilog);
        }
        self.map.mix(m, y, bpos, ilog, squash, stretch);

        let info = &self.info;
        let vow = if self.l.id != 0 {
            1 + Self::is_vowel(self.l.id, buffer.at(1)) as u32
        } else { 0 };
        m.set(finalize64(hash3(vow as u64,
            (info.masks[1] & 0xFF) as u64, c0 as u64), 11), 2048);
        m.set(finalize64(hash3(ilog2(info.word_length[0] + 1) as u64, c0 as u64,
            ((info.last_digit < info.word_length[0] + info.word_gap) as u64)
            | (((info.last_upper < info.last_letter + info.word_length[1]) as u64) << 1)
            | (((info.last_punct < info.word_length[0] + info.word_gap) as u64) << 2)
            | (((info.last_upper < info.word_length[0]) as u64) << 3)), 11), 2048);
        m.set(finalize64(hash4((info.masks[1] & 0x3FF) as u64, grp0 as u64,
            (info.last_upper < info.word_length[0]) as u64,
            (info.last_upper < info.last_letter + info.word_length[1]) as u64),
            12), 4096);
        m.set(finalize64(hash3((info.spaces & 0x1FF) as u64, grp0 as u64,
            ((info.last_upper < info.word_length[0]) as u64)
            | (((info.last_upper < info.last_letter + info.word_length[1]) as u64) << 1)
            | (((info.last_punct < info.last_letter) as u64) << 2)
            | (((info.last_punct < info.word_length[0] + info.word_gap) as u64) << 3)
            | (((info.last_punct < info.last_letter + info.word_length[1]
                + info.word_gap) as u64) << 4)), 12), 4096);
        m.set(finalize64(hash3(
            (info.first_letter as u32 * (info.word_length[0] < 4) as u32) as u64,
            6.min(info.word_length[0]) as u64, c0 as u64), 11), 2048);
        let pw = self.pword();
        m.set(finalize64(hash4(pw.at(0) as u64, pw.from_end(0) as u64,
            4.min(info.word_length[0]) as u64,
            (info.last_punct < info.last_letter) as u64), 11), 2048);
        m.set(finalize64(hash3(4.min(info.word_length[0]) as u64, grp0 as u64,
            (info.last_upper < info.word_length[0]) as u64), 12), 4096);
        // last context (paq8.cpp:3185).
        m.set(finalize64(hash3(grp0 as u64, (info.masks[4] & 0x1F) as u64,
            ((info.masks[4] >> 5) & 0x1F) as u64), 13), 8192);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn text_model_runs_through_english_text_without_panic() {
        let dt = build_dt();
        let il = Ilog::new();
        let sq = Squash::new();
        let st = Stretch::new(&sq);
        // 256 KiB / 64-byte buckets ContextMap2 — cheap for a test.
        let mut tm = TextModel::new(256 * 1024, dt);
        let mut buf = Buf::new();
        buf.set_size(1 << 16);
        let mut stats = ModelStats::new();
        let mut mixer = Mixer::new(2048, 28, 0);

        let text = b"The quick brown fox jumps over the lazy dog. \
                     Sentences end here! And here? Yes.\n\
                     A new paragraph with numbers 12, 34, and 567.";
        let mut pos = 0u32;
        for &byte in text {
            for bp in 0..8 {
                let bit = ((byte >> (7 - bp)) & 1) as i32;
                let c0 = if bp == 0 { 1u32 }
                    else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                let grp0 = if bp > 0 { c0 as u8 } else { 0 };
                tm.predict(&mut mixer, &buf, pos, c0, bp, bit, grp0,
                    &il, &sq, &st, &mut stats);
            }
            buf.push(byte);
            pos += 1;
        }
        // After a sentence terminator the parser should have moved
        // out of the initial Unknown state at least once.
        assert!(stats.text.mask != 0 || stats.text.word_length != 0
            || stats.text.first_letter != 0,
            "TextModel should have populated some text stats");
    }
}
