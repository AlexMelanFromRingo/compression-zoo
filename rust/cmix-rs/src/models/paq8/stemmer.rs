//! English / French / German stemmers — paq8.cpp:1730-3070.
//!
//! The English stemmer is a faithful port of Márcio Pais's modified
//! Porter2 stemmer (paq8.cpp:1765-2423). French and German stemmers
//! follow upstream's Porter-based variants.

#![allow(dead_code)]

use super::word::{english, lang, LanguageId, Word, MAX_WORD_SIZE};

#[inline]
fn char_in(c: u8, arr: &[u8]) -> bool { arr.contains(&c) }

// =============================================================
// Shared region helpers (`Stemmer`, paq8.cpp:1730-1752).
// =============================================================

/// `GetRegion(W, From)` — paq8.cpp:1732-1742.
fn get_region(w: &Word, from: u32, is_vowel: impl Fn(u8) -> bool) -> u32 {
    let mut has_vowel = false;
    let start = w.start as u32 + from;
    for i in start..=w.end as u32 {
        let c = w.letters[i as usize];
        if is_vowel(c) {
            has_vowel = true;
            continue;
        } else if has_vowel {
            return i - w.start as u32 + 1;
        }
    }
    w.start as u32 + w.length()
}

/// `SuffixInRn(W, Rn, Suffix)` — paq8.cpp:1744-1746.
fn suffix_in_rn(w: &Word, rn: u32, suffix_len: usize) -> bool {
    w.start != w.end && rn <= w.length().saturating_sub(suffix_len as u32)
}

// =============================================================
// EnglishStemmer — paq8.cpp:1765-2423.
// =============================================================

const VOWELS: &[u8] = b"aeiouy";
const DOUBLES: &[u8] = b"bdfgmnprt";
const LI_ENDINGS: &[u8] = b"cdeghkmnrt";
const NON_SHORT_CONSONANTS: &[u8] = b"wxY";

const MALE_WORDS: [&[u8]; 9] = [
    b"he", b"him", b"his", b"himself", b"man", b"men", b"boy",
    b"husband", b"actor",
];
const FEMALE_WORDS: [&[u8]; 8] = [
    b"she", b"her", b"herself", b"woman", b"women", b"girl",
    b"wife", b"actress",
];
const COMMON_WORDS: [&[u8]; 12] = [
    b"the", b"be", b"to", b"of", b"and", b"in", b"that", b"you",
    b"have", b"with", b"from", b"but",
];

const SUFFIXES_STEP0: [&[u8]; 3] = [b"'s'", b"'s", b"'"];

const SUFFIXES_STEP1B: [&[u8]; 6] =
    [b"eedly", b"eed", b"ed", b"edly", b"ing", b"ingly"];
const TYPES_STEP1B: [u64; 6] = [
    english::ADVERB_OF_MANNER,
    0,
    english::PAST_TENSE,
    english::ADVERB_OF_MANNER | english::PAST_TENSE,
    english::PRESENT_PARTICIPLE,
    english::ADVERB_OF_MANNER | english::PRESENT_PARTICIPLE,
];

const SUFFIXES_STEP2: [(&[u8], &[u8]); 22] = [
    (b"ization", b"ize"), (b"ational", b"ate"), (b"ousness", b"ous"),
    (b"iveness", b"ive"), (b"fulness", b"ful"), (b"tional", b"tion"),
    (b"lessli", b"less"), (b"biliti", b"ble"), (b"entli", b"ent"),
    (b"ation", b"ate"), (b"alism", b"al"), (b"aliti", b"al"),
    (b"fulli", b"ful"), (b"ousli", b"ous"), (b"iviti", b"ive"),
    (b"enci", b"ence"), (b"anci", b"ance"), (b"abli", b"able"),
    (b"izer", b"ize"), (b"ator", b"ate"), (b"alli", b"al"),
    (b"bli", b"ble"),
];
const TYPES_STEP2: [u64; 22] = [
    english::SUFFIX_ION,
    english::SUFFIX_ION | english::SUFFIX_AL,
    english::SUFFIX_NESS,
    english::SUFFIX_NESS,
    english::SUFFIX_NESS,
    english::SUFFIX_ION | english::SUFFIX_AL,
    english::ADVERB_OF_MANNER,
    english::ADVERB_OF_MANNER | english::SUFFIX_ITY,
    english::ADVERB_OF_MANNER,
    english::SUFFIX_ION,
    0,
    english::SUFFIX_ITY,
    english::ADVERB_OF_MANNER,
    english::ADVERB_OF_MANNER,
    english::SUFFIX_ITY,
    0,
    0,
    english::ADVERB_OF_MANNER,
    0,
    0,
    english::ADVERB_OF_MANNER,
    english::ADVERB_OF_MANNER,
];

const SUFFIXES_STEP3: [(&[u8], &[u8]); 8] = [
    (b"ational", b"ate"), (b"tional", b"tion"), (b"alize", b"al"),
    (b"icate", b"ic"), (b"iciti", b"ic"), (b"ical", b"ic"),
    (b"ful", b""), (b"ness", b""),
];
const TYPES_STEP3: [u64; 8] = [
    english::SUFFIX_ION | english::SUFFIX_AL,
    english::SUFFIX_ION | english::SUFFIX_AL,
    0,
    0,
    english::SUFFIX_ITY,
    english::SUFFIX_AL,
    english::ADJECTIVE_FULL,
    english::SUFFIX_NESS,
];

const SUFFIXES_STEP4: [&[u8]; 20] = [
    b"al", b"ance", b"ence", b"er", b"ic", b"able", b"ible", b"ant",
    b"ement", b"ment", b"ent", b"ou", b"ism", b"ate", b"iti", b"ous",
    b"ive", b"ize", b"sion", b"tion",
];
const TYPES_STEP4: [u64; 20] = [
    english::SUFFIX_AL,
    english::SUFFIX_NCE,
    english::SUFFIX_NCE,
    0,
    english::SUFFIX_IC,
    english::SUFFIX_CAPABLE,
    english::SUFFIX_CAPABLE,
    english::SUFFIX_NT,
    0,
    0,
    english::SUFFIX_NT,
    0,
    0,
    0,
    english::SUFFIX_ITY,
    english::SUFFIX_OUS,
    english::SUFFIX_IVE,
    0,
    english::SUFFIX_ION,
    english::SUFFIX_ION,
];

const EXCEPTIONS_REGION1: [&[u8]; 3] = [b"gener", b"arsen", b"commun"];

const EXCEPTIONS1: [(&[u8], &[u8]); 18] = [
    (b"skis", b"ski"), (b"skies", b"sky"), (b"dying", b"die"),
    (b"lying", b"lie"), (b"tying", b"tie"), (b"idly", b"idle"),
    (b"gently", b"gentle"), (b"ugly", b"ugli"), (b"early", b"earli"),
    (b"only", b"onli"), (b"singly", b"singl"), (b"sky", b"sky"),
    (b"news", b"news"), (b"howe", b"howe"), (b"atlas", b"atlas"),
    (b"cosmos", b"cosmos"), (b"bias", b"bias"), (b"andes", b"andes"),
];
const TYPES_EXCEPTIONS1: [u64; 18] = [
    lang::NOUN | english::PLURAL,
    english::PLURAL,
    english::PRESENT_PARTICIPLE,
    english::PRESENT_PARTICIPLE,
    english::PRESENT_PARTICIPLE,
    english::ADVERB_OF_MANNER,
    english::ADVERB_OF_MANNER,
    english::ADJECTIVE,
    english::ADJECTIVE | english::ADVERB_OF_MANNER,
    0,
    english::ADVERB_OF_MANNER,
    lang::NOUN,
    lang::NOUN,
    0,
    lang::NOUN,
    lang::NOUN,
    lang::NOUN,
    0,
];

const EXCEPTIONS2: [&[u8]; 8] = [
    b"inning", b"outing", b"canning", b"herring", b"earring",
    b"proceed", b"exceed", b"succeed",
];
const TYPES_EXCEPTIONS2: [u64; 8] = [
    lang::NOUN, lang::NOUN, lang::NOUN, lang::NOUN, lang::NOUN,
    lang::VERB, lang::VERB, lang::VERB,
];

/// English Porter2 stemmer — paq8.cpp:1765-2423.
pub struct EnglishStemmer;

impl EnglishStemmer {
    #[inline]
    pub fn is_vowel(c: u8) -> bool { char_in(c, VOWELS) }
    #[inline]
    fn is_consonant(c: u8) -> bool { !Self::is_vowel(c) }
    #[inline]
    fn is_short_consonant(c: u8) -> bool {
        !char_in(c, NON_SHORT_CONSONANTS)
    }
    #[inline]
    fn is_double(c: u8) -> bool { char_in(c, DOUBLES) }
    #[inline]
    fn is_li_ending(c: u8) -> bool { char_in(c, LI_ENDINGS) }

    fn get_region1(w: &Word) -> u32 {
        for ex in EXCEPTIONS_REGION1 {
            if w.starts_with(ex) { return ex.len() as u32; }
        }
        get_region(w, 0, Self::is_vowel)
    }

    fn ends_in_short_syllable(w: &Word) -> bool {
        if w.end == w.start {
            false
        } else if w.end == w.start + 1 {
            Self::is_vowel(w.from_end(1)) && Self::is_consonant(w.from_end(0))
        } else {
            Self::is_consonant(w.from_end(2))
                && Self::is_vowel(w.from_end(1))
                && Self::is_consonant(w.from_end(0))
                && Self::is_short_consonant(w.from_end(0))
        }
    }

    fn is_short_word(w: &Word) -> bool {
        Self::ends_in_short_syllable(w) && Self::get_region1(w) == w.length()
    }

    fn has_vowels(w: &Word) -> bool {
        for i in w.start..=w.end {
            if Self::is_vowel(w.letters[i as usize]) { return true; }
        }
        false
    }

    fn trim_starting_apostrophe(w: &mut Word) -> bool {
        let r = w.start != w.end && w.at(0) == b'\'';
        w.start += r as u8;
        r
    }

    fn mark_ys_as_consonants(w: &mut Word) {
        if w.at(0) == b'y' {
            w.letters[w.start as usize] = b'Y';
        }
        for i in (w.start + 1)..=w.end {
            if Self::is_vowel(w.letters[(i - 1) as usize])
                && w.letters[i as usize] == b'y'
            {
                w.letters[i as usize] = b'Y';
            }
        }
    }

    fn process_prefixes(w: &mut Word) -> bool {
        if w.starts_with(b"irr") && w.length() > 5
            && (w.at(3) == b'a' || w.at(3) == b'e')
        {
            w.start += 2; w.r#type |= english::NEGATION;
        } else if w.starts_with(b"over") && w.length() > 5 {
            w.start += 4; w.r#type |= english::PREFIX_OVER;
        } else if w.starts_with(b"under") && w.length() > 6 {
            w.start += 5; w.r#type |= english::PREFIX_UNDER;
        } else if w.starts_with(b"unn") && w.length() > 5 {
            w.start += 2; w.r#type |= english::NEGATION;
        } else if w.starts_with(b"non")
            && w.length() > (5 + (w.at(3) == b'-') as u32)
        {
            w.start += 2 + (w.at(3) == b'-') as u8;
            w.r#type |= english::NEGATION;
        } else {
            return false;
        }
        true
    }

    fn process_superlatives(w: &mut Word) -> bool {
        if w.ends_with(b"est") && w.length() > 4 {
            let i = w.end;
            w.end -= 3;
            w.r#type |= english::ADJECTIVE_SUPERLATIVE;

            let sugg = w.length() >= 4
                && &w.letters[w.end as usize - 3..w.end as usize + 1] == b"sugg";
            if w.from_end(0) == w.from_end(1) && w.from_end(0) != b'r' && !sugg {
                let cond_a = (w.from_end(0) != b'f' && w.from_end(0) != b'l'
                    && w.from_end(0) != b's')
                    || (w.length() > 4 && w.from_end(1) == b'l'
                        && (w.from_end(2) == b'u' || w.from_end(3) == b'u'
                            || w.from_end(3) == b'v'));
                let cond_b = !(w.length() == 3 && w.from_end(1) == b'd'
                    && w.from_end(2) == b'o');
                w.end -= (cond_a && cond_b) as u8;
                if w.length() == 2 && (w.at(0) != b'i' || w.at(1) != b'n') {
                    w.end = i;
                    w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                }
            } else {
                match w.from_end(0) {
                    b'd' | b'k' | b'm' | b'y' => {}
                    b'g' => {
                        let cong = w.length() > 3
                            && (w.from_end(1) == b'n' || w.from_end(1) == b'r')
                            && &w.letters[w.end as usize - 3..w.end as usize + 1] != b"cong";
                        if !cong {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        } else {
                            w.end += (w.from_end(2) == b'a') as u8;
                        }
                    }
                    b'i' => { w.letters[w.end as usize] = b'y'; }
                    b'l' => {
                        if w.end == w.start + 1
                            || &w.letters[w.end as usize - 2..w.end as usize] == b"mo"
                        {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        } else {
                            w.end += Self::is_consonant(w.from_end(1)) as u8;
                        }
                    }
                    b'n' => {
                        if w.length() < 3 || Self::is_consonant(w.from_end(1))
                            || Self::is_consonant(w.from_end(2))
                        {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b'r' => {
                        if w.length() > 3 && Self::is_vowel(w.from_end(1))
                            && Self::is_vowel(w.from_end(2))
                        {
                            w.end += ((w.from_end(2) == b'u')
                                && (w.from_end(1) == b'a' || w.from_end(1) == b'i'))
                                as u8;
                        } else {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b's' => { w.end += 1; }
                    b'w' => {
                        if !(w.length() > 2 && Self::is_vowel(w.from_end(1))) {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    b'h' => {
                        if !(w.length() > 2 && Self::is_consonant(w.from_end(1))) {
                            w.end = i;
                            w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                        }
                    }
                    _ => {
                        w.end += 3;
                        w.r#type &= !english::ADJECTIVE_SUPERLATIVE;
                    }
                }
            }
        }
        (w.r#type & english::ADJECTIVE_SUPERLATIVE) > 0
    }

    fn step0(w: &mut Word) -> bool {
        for s in SUFFIXES_STEP0 {
            if w.ends_with(s) {
                w.end -= s.len() as u8;
                w.r#type |= english::PLURAL;
                return true;
            }
        }
        false
    }

    fn step1a(w: &mut Word) -> bool {
        if w.ends_with(b"sses") {
            w.end -= 2;
            w.r#type |= english::PLURAL;
            return true;
        }
        if w.ends_with(b"ied") || w.ends_with(b"ies") {
            w.r#type |= if w.from_end(0) == b'd' {
                english::PAST_TENSE
            } else {
                english::PLURAL
            };
            w.end -= 1 + (w.length() > 4) as u8;
            return true;
        }
        if w.ends_with(b"us") || w.ends_with(b"ss") {
            return false;
        }
        if w.from_end(0) == b's' && w.length() > 2 {
            for i in w.start..=w.end - 2 {
                if Self::is_vowel(w.letters[i as usize]) {
                    w.end -= 1;
                    w.r#type |= english::PLURAL;
                    return true;
                }
            }
        }
        if w.ends_with(b"n't") && w.length() > 4 {
            match w.from_end(3) {
                b'a' => {
                    if w.from_end(4) == b'c' {
                        w.end -= 2;
                    } else {
                        w.change_suffix(b"n't", b"ll");
                    }
                }
                b'i' => { w.change_suffix(b"in't", b"m"); }
                b'o' => {
                    if w.from_end(4) == b'w' {
                        w.change_suffix(b"on't", b"ill");
                    } else {
                        w.end -= 3;
                    }
                }
                _ => { w.end -= 3; }
            }
            w.r#type |= english::NEGATION;
            return true;
        }
        if w.ends_with(b"hood") && w.length() > 7 {
            w.end -= 4;
            return true;
        }
        false
    }

    fn step1b(w: &mut Word, r1: u32) -> bool {
        for i in 0..SUFFIXES_STEP1B.len() {
            let suf = SUFFIXES_STEP1B[i];
            if w.ends_with(suf) {
                match i {
                    0 | 1 => {
                        if suffix_in_rn(w, r1, suf.len()) {
                            w.end -= 1 + (i as u8) * 2;
                        }
                    }
                    _ => {
                        let j = w.end;
                        w.end -= suf.len() as u8;
                        if Self::has_vowels(w) {
                            if w.ends_with(b"at") || w.ends_with(b"bl")
                                || w.ends_with(b"iz") || Self::is_short_word(w)
                            {
                                w.push(b'e');
                            } else if w.length() > 2 {
                                if w.from_end(0) == w.from_end(1)
                                    && Self::is_double(w.from_end(0))
                                {
                                    w.end -= 1;
                                } else if i == 2 || i == 3 {
                                    match w.from_end(0) {
                                        b'c' | b's' | b'v' => {
                                            w.end += !(w.ends_with(b"ss")
                                                || w.ends_with(b"ias")) as u8;
                                        }
                                        b'd' => {
                                            let n_allowed = b"aeio";
                                            w.end += (Self::is_vowel(w.from_end(1))
                                                && !char_in(w.from_end(2), n_allowed))
                                                as u8;
                                        }
                                        b'k' => {
                                            w.end += w.ends_with(b"uak") as u8;
                                        }
                                        b'l' => {
                                            let allowed1 = b"bcdfgkptyz";
                                            let allowed2 = b"aiou";
                                            w.end += (char_in(w.from_end(1), allowed1)
                                                || (char_in(w.from_end(1), allowed2)
                                                    && Self::is_consonant(w.from_end(2))))
                                                as u8;
                                        }
                                        _ => {}
                                    }
                                } else if i >= 4 {
                                    Self::step1b_i_ge4(w);
                                }
                            }
                        } else {
                            w.end = j;
                            return false;
                        }
                    }
                }
                w.r#type |= TYPES_STEP1B[i];
                return true;
            }
        }
        false
    }

    /// The deeply-nested `i>=4` block of Step1b — paq8.cpp:2140-2210.
    fn step1b_i_ge4(w: &mut Word) {
        match w.from_end(0) {
            b'd' => {
                if Self::is_vowel(w.from_end(1)) && w.from_end(2) != b'a'
                    && w.from_end(2) != b'e' && w.from_end(2) != b'o'
                {
                    w.push(b'e');
                }
            }
            b'g' => {
                let allowed = b"adeilru";
                let cond = char_in(w.from_end(1), allowed)
                    || (w.from_end(1) == b'n'
                        && (w.from_end(2) == b'e'
                            || (w.from_end(2) == b'u' && w.from_end(3) != b'b'
                                && w.from_end(3) != b'd')
                            || (w.from_end(2) == b'a'
                                && (w.from_end(3) == b'r'
                                    || (w.from_end(3) == b'h' && w.from_end(4) == b'c')))
                            || (w.ends_with(b"ring")
                                && (w.from_end(4) == b'c' || w.from_end(4) == b'f'))));
                if cond { w.push(b'e'); }
            }
            b'l' => {
                if !(w.from_end(1) == b'l' || w.from_end(1) == b'r'
                    || w.from_end(1) == b'w'
                    || (Self::is_vowel(w.from_end(1)) && Self::is_vowel(w.from_end(2))))
                {
                    w.push(b'e');
                }
                if w.ends_with(b"uell") && w.length() > 4 && w.from_end(4) != b'q' {
                    w.end -= 1;
                }
            }
            b'r' => {
                let cond = ((w.from_end(1) == b'i' && w.from_end(2) != b'a'
                    && w.from_end(2) != b'e' && w.from_end(2) != b'o')
                    || (w.from_end(1) == b'a'
                        && !(w.from_end(2) == b'e' || w.from_end(2) == b'o'
                            || (w.from_end(2) == b'l' && w.from_end(3) == b'l')))
                    || (w.from_end(1) == b'o'
                        && !(w.from_end(2) == b'o'
                            || (w.from_end(2) == b't' && w.from_end(3) != b's')))
                    || w.from_end(1) == b'c' || w.from_end(1) == b't')
                    && !w.ends_with(b"str");
                if cond { w.push(b'e'); }
            }
            b't' => {
                if w.from_end(1) == b'o' && w.from_end(2) != b'g'
                    && w.from_end(2) != b'l' && w.from_end(2) != b'i'
                    && w.from_end(2) != b'o'
                {
                    w.push(b'e');
                }
            }
            b'u' => {
                if !(w.length() > 3 && Self::is_vowel(w.from_end(1))
                    && Self::is_vowel(w.from_end(2)))
                {
                    w.push(b'e');
                }
            }
            b'z' => {
                if w.ends_with(b"izz") && w.length() > 3
                    && (w.from_end(3) == b'h' || w.from_end(3) == b'u')
                {
                    w.end -= 1;
                } else if w.from_end(1) != b't' && w.from_end(1) != b'z' {
                    w.push(b'e');
                }
            }
            b'k' => {
                if w.ends_with(b"uak") { w.push(b'e'); }
            }
            b'b' | b'c' | b's' | b'v' => {
                let zinc = w.equals_str(b"zinc");
                if !((w.from_end(0) == b'b'
                        && (w.from_end(1) == b'm' || w.from_end(1) == b'r'))
                    || w.ends_with(b"ss") || w.ends_with(b"ias") || zinc)
                {
                    w.push(b'e');
                }
            }
            _ => {}
        }
    }

    fn step1c(w: &mut Word) -> bool {
        if w.length() > 2
            && w.from_end(0).to_ascii_lowercase() == b'y'
            && Self::is_consonant(w.from_end(1))
        {
            w.letters[w.end as usize] = b'i';
            return true;
        }
        false
    }

    fn step2(w: &mut Word, r1: u32) -> bool {
        for i in 0..SUFFIXES_STEP2.len() {
            let (suf, repl) = SUFFIXES_STEP2[i];
            if w.ends_with(suf) && suffix_in_rn(w, r1, suf.len()) {
                w.change_suffix(suf, repl);
                w.r#type |= TYPES_STEP2[i];
                return true;
            }
        }
        if w.ends_with(b"logi") && suffix_in_rn(w, r1, 3) {
            w.end -= 1;
            return true;
        } else if w.ends_with(b"li") {
            if suffix_in_rn(w, r1, 2) && Self::is_li_ending(w.from_end(2)) {
                w.end -= 2;
                w.r#type |= english::ADVERB_OF_MANNER;
                return true;
            } else if w.length() > 3 {
                match w.from_end(2) {
                    b'b' => {
                        w.letters[w.end as usize] = b'e';
                        w.r#type |= english::ADVERB_OF_MANNER;
                        return true;
                    }
                    b'i' => {
                        if w.length() > 4 {
                            w.end -= 2;
                            w.r#type |= english::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    b'l' => {
                        if w.length() > 5
                            && (w.from_end(3) == b'a' || w.from_end(3) == b'u')
                        {
                            w.end -= 2;
                            w.r#type |= english::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    b's' => {
                        w.end -= 2;
                        w.r#type |= english::ADVERB_OF_MANNER;
                        return true;
                    }
                    b'e' | b'g' | b'm' | b'n' | b'r' | b'w' => {
                        if w.length() > (4 + (w.from_end(2) == b'r') as u32) {
                            w.end -= 2;
                            w.r#type |= english::ADVERB_OF_MANNER;
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    fn step3(w: &mut Word, r1: u32, r2: u32) -> bool {
        let mut res = false;
        for i in 0..SUFFIXES_STEP3.len() {
            let (suf, repl) = SUFFIXES_STEP3[i];
            if w.ends_with(suf) && suffix_in_rn(w, r1, suf.len()) {
                w.change_suffix(suf, repl);
                w.r#type |= TYPES_STEP3[i];
                res = true;
                break;
            }
        }
        if w.ends_with(b"ative") && suffix_in_rn(w, r2, 5) {
            w.end -= 5;
            w.r#type |= english::SUFFIX_IVE;
            return true;
        }
        if w.length() > 5 && w.ends_with(b"less") {
            w.end -= 4;
            w.r#type |= english::ADJECTIVE_WITHOUT;
            return true;
        }
        res
    }

    fn step4(w: &mut Word, r2: u32) -> bool {
        let mut res = false;
        for i in 0..SUFFIXES_STEP4.len() {
            let suf = SUFFIXES_STEP4[i];
            if w.ends_with(suf) && suffix_in_rn(w, r2, suf.len()) {
                w.end -= (suf.len() - (i > 17) as usize) as u8;
                if i != 10 || w.from_end(0) != b'm' {
                    w.r#type |= TYPES_STEP4[i];
                }
                if i == 0 && w.ends_with(b"nti") {
                    w.end -= 1;
                    res = true;
                    continue;
                }
                return true;
            }
        }
        res
    }

    fn step5(w: &mut Word, r1: u32, r2: u32) -> bool {
        if w.from_end(0) == b'e' && !w.equals_str(b"here") {
            if suffix_in_rn(w, r2, 1) {
                w.end -= 1;
            } else if suffix_in_rn(w, r1, 1) {
                w.end -= 1;
                w.end += Self::ends_in_short_syllable(w) as u8;
            } else {
                return false;
            }
            true
        } else if w.length() > 1 && w.from_end(0) == b'l'
            && suffix_in_rn(w, r2, 1) && w.from_end(1) == b'l'
        {
            w.end -= 1;
            true
        } else {
            false
        }
    }

    /// `Hash(W)` — paq8.cpp:2351-2363. Populates `hash[2]` / `hash[3]`.
    pub fn hash(w: &mut Word) {
        w.hash[2] = 0xb0a710ad;
        w.hash[3] = 0xb0a710ad;
        for i in w.start..=w.end {
            let l = w.letters[i as usize];
            w.hash[2] = w.hash[2].wrapping_mul(263 * 32).wrapping_add(l as u64);
            if Self::is_vowel(l) {
                w.hash[3] = w.hash[3].wrapping_mul(997 * 8)
                    .wrapping_add((l / 4).wrapping_sub(22) as u64);
            } else if (b'b'..=b'z').contains(&l) {
                w.hash[3] = w.hash[3].wrapping_mul(271 * 32)
                    .wrapping_add((l - 97) as u64);
            } else {
                w.hash[3] = w.hash[3].wrapping_mul(11 * 32)
                    .wrapping_add(l as u64);
            }
        }
    }

    /// `Stem(W)` — paq8.cpp:2364-2422. Returns `true` if a meaningful
    /// stem / classification was applied.
    pub fn stem(w: &mut Word) -> bool {
        if w.length() < 2 {
            Self::hash(w);
            return false;
        }
        let mut res = Self::trim_starting_apostrophe(w);
        res |= Self::process_prefixes(w);
        res |= Self::process_superlatives(w);

        for i in 0..EXCEPTIONS1.len() {
            let (from, to) = EXCEPTIONS1[i];
            if w.equals_str(from) {
                if i < 11 {
                    let len = to.len();
                    for k in 0..len {
                        w.letters[w.start as usize + k] = to[k];
                    }
                    w.end = w.start + (len - 1) as u8;
                }
                Self::hash(w);
                w.r#type |= TYPES_EXCEPTIONS1[i];
                w.language = LanguageId::English as u64;
                return i < 11;
            }
        }

        // Modified Porter2 stemmer.
        Self::mark_ys_as_consonants(w);
        let r1 = Self::get_region1(w);
        let r2 = get_region(w, r1, Self::is_vowel);
        res |= Self::step0(w);
        res |= Self::step1a(w);

        for i in 0..EXCEPTIONS2.len() {
            if w.equals_str(EXCEPTIONS2[i]) {
                Self::hash(w);
                w.r#type |= TYPES_EXCEPTIONS2[i];
                w.language = LanguageId::English as u64;
                return res;
            }
        }

        res |= Self::step1b(w, r1);
        res |= Self::step1c(w);
        res |= Self::step2(w, r1);
        res |= Self::step3(w, r1, r2);
        res |= Self::step4(w, r2);
        res |= Self::step5(w, r1, r2);

        for i in w.start..=w.end {
            if w.letters[i as usize] == b'Y' {
                w.letters[i as usize] = b'y';
            }
        }
        if w.r#type == 0 || w.r#type == english::PLURAL {
            if w.matches_any(&MALE_WORDS) {
                res = true;
                w.r#type |= english::MALE;
            } else if w.matches_any(&FEMALE_WORDS) {
                res = true;
                w.r#type |= english::FEMALE;
            }
        }
        if !res {
            res = w.matches_any(&COMMON_WORDS);
        }
        Self::hash(w);
        if res {
            w.language = LanguageId::English as u64;
        }
        res
    }
}

// =============================================================
// FrenchStemmer — paq8.cpp:2434-2823.
// =============================================================

const FR_VOWELS: &[u8] = &[
    b'a', b'e', b'i', b'o', b'u', b'y',
    0xE2, 0xE0, 0xEB, 0xE9, 0xEA, 0xE8, 0xEF, 0xEE, 0xF4, 0xFB, 0xF9,
];
const FR_COMMON_WORDS: [&[u8]; 10] = [
    b"de", b"la", b"le", b"et", b"en", b"un", b"une", b"du", b"que", b"pas",
];
const FR_EXCEPTIONS: [(&[u8], &[u8]); 3] = [
    (b"monument", b"monument"),
    (b"yeux", b"oeil"),
    (b"travaux", b"travail"),
];
const FR_TYPES_EXCEPTIONS: [u64; 3] = [
    lang::NOUN,
    lang::NOUN | super::word::french::PLURAL,
    lang::NOUN | super::word::french::PLURAL,
];

/// French Porter stemmer — paq8.cpp:2434-2823. Suffix list indices
/// match upstream exactly (the loop boundaries 11/17/25/27/29/31/
/// 35/37 depend on them).
pub struct FrenchStemmer;

impl FrenchStemmer {
    #[inline]
    pub fn is_vowel(c: u8) -> bool { char_in(c, FR_VOWELS) }
    #[inline]
    fn is_consonant(c: u8) -> bool { !Self::is_vowel(c) }

    const STEP1: [&'static [u8]; 39] = [
        b"ance", b"iqUe", b"isme", b"able", b"iste", b"eux",
        b"ances", b"iqUes", b"ismes", b"ables", b"istes",
        b"atrice", b"ateur", b"ation", b"atrices", b"ateurs", b"ations",
        b"logie", b"logies",
        b"usion", b"ution", b"usions", b"utions",
        b"ence", b"ences",
        b"issement", b"issements",
        b"ement", b"ements",
        &[b'i', b't', 0xE9], &[b'i', b't', 0xE9, b's'],
        b"if", b"ive", b"ifs", b"ives",
        b"euse", b"euses",
        b"ment", b"ments",
    ];
    const STEP2A: [&'static [u8]; 35] = [
        b"issaIent", b"issantes", b"iraIent", b"issante",
        b"issants", b"issions", b"irions", b"issais",
        b"issait", b"issant", b"issent", b"issiez", b"issons",
        b"irais", b"irait", b"irent", b"iriez", b"irons",
        b"iront", b"isses", b"issez", &[0xEE, b'm', b'e', b's'],
        &[0xEE, b't', b'e', b's'], b"irai", b"iras", b"irez", b"isse",
        b"ies", b"ira", &[0xEE, b't'], b"ie", b"ir", b"is",
        b"it", b"i",
    ];
    const STEP2B: [&'static [u8]; 38] = [
        b"eraIent", b"assions", b"erions", b"assent",
        b"assiez", &[0xE8, b'r', b'e', b'n', b't'], b"erais", b"erait",
        b"eriez", b"erons", b"eront", b"aIent", b"antes",
        b"asses", b"ions", b"erai", b"eras", b"erez",
        &[0xE2, b'm', b'e', b's'], &[0xE2, b't', b'e', b's'], b"ante", b"ants",
        b"asse", &[0xE9, b'e', b's'], b"era", b"iez", b"ais",
        b"ait", b"ant", &[0xE9, b'e'], &[0xE9, b's'], b"er",
        b"ez", &[0xE2, b't'], b"ai", b"as", &[0xE9], b"a",
    ];
    const SET_STEP4: &'static [u8] = &[b'a', b'i', b'o', b'u', 0xE8, b's'];
    const STEP4: [&'static [u8]; 7] = [
        &[b'i', 0xE8, b'r', b'e'], &[b'I', 0xE8, b'r', b'e'],
        b"ion", b"ier", b"Ier", b"e", &[0xEB],
    ];
    const STEP5: [&'static [u8]; 5] = [b"enn", b"onn", b"ett", b"ell", b"eill"];

    fn convert_utf8(w: &mut Word) {
        let mut i = w.start;
        while i < w.end {
            let next = w.letters[(i + 1) as usize];
            let c = next.wrapping_add(if next < 0xA0 { 0x60 } else { 0x40 });
            if w.letters[i as usize] == 0xC3
                && (Self::is_vowel(c) || (next & 0xDF) == 0x87)
            {
                w.letters[i as usize] = c;
                if i + 1 < w.end {
                    let span = (w.end - i - 1) as usize;
                    for k in 0..span {
                        w.letters[(i + 1) as usize + k] =
                            w.letters[(i + 2) as usize + k];
                    }
                }
                w.end -= 1;
            }
            i += 1;
        }
    }

    fn mark_vowels_as_consonants(w: &mut Word) {
        for i in w.start..=w.end {
            match w.letters[i as usize] {
                b'i' | b'u' => {
                    if i > w.start && i < w.end
                        && (Self::is_vowel(w.letters[(i - 1) as usize])
                            || (w.letters[(i - 1) as usize] == b'q'
                                && w.letters[i as usize] == b'u'))
                        && Self::is_vowel(w.letters[(i + 1) as usize])
                    {
                        w.letters[i as usize] =
                            w.letters[i as usize].to_ascii_uppercase();
                    }
                }
                b'y' => {
                    if (i > w.start
                        && Self::is_vowel(w.letters[(i - 1) as usize]))
                        || (i < w.end
                            && Self::is_vowel(w.letters[(i + 1) as usize]))
                    {
                        w.letters[i as usize] =
                            w.letters[i as usize].to_ascii_uppercase();
                    }
                }
                _ => {}
            }
        }
    }

    fn get_rv(w: &Word) -> u32 {
        let len = w.length();
        let res = w.start as u32 + len;
        if len >= 3
            && ((Self::is_vowel(w.letters[w.start as usize])
                && Self::is_vowel(w.letters[w.start as usize + 1]))
                || w.starts_with(b"par")
                || w.starts_with(b"col")
                || w.starts_with(b"tap"))
        {
            return w.start as u32 + 3;
        }
        for i in (w.start as u32 + 1)..=w.end as u32 {
            if Self::is_vowel(w.letters[i as usize]) {
                return i + 1;
            }
        }
        res
    }

    fn step1(w: &mut Word, rv: u32, r1: u32, r2: u32,
            force_step2a: &mut bool) -> bool {
        let s = &Self::STEP1;
        // i in 0..11 — R2.
        for i in 0..11 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if i == 3 { w.r#type |= super::word::french::ADJECTIVE; }
                return true;
            }
        }
        // 11..17 — R2.
        for i in 11..17 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if w.ends_with(b"ic") { w.change_suffix(b"c", b"qU"); }
                return true;
            }
        }
        // 17..25 — R2.
        for i in 17..25 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= (s[i].len() as i32 - 1 - (i < 19) as i32 * 2) as u8;
                if i > 22 {
                    w.end += 2;
                    w.letters[w.end as usize] = b't';
                }
                return true;
            }
        }
        // 25..27 — R1, consonant check.
        for i in 25..27 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r1, s[i].len())
                && Self::is_consonant(w.from_end(s[i].len() as u8))
            {
                w.end -= s[i].len() as u8;
                return true;
            }
        }
        // 27..29 — RV, nested.
        for i in 27..29 {
            if w.ends_with(s[i]) && suffix_in_rn(w, rv, s[i].len()) {
                w.end -= s[i].len() as u8;
                if w.ends_with(b"iv") && suffix_in_rn(w, r2, 2) {
                    w.end -= 2;
                    if w.ends_with(b"at") && suffix_in_rn(w, r2, 2) {
                        w.end -= 2;
                    }
                } else if w.ends_with(b"eus") {
                    if suffix_in_rn(w, r2, 3) {
                        w.end -= 3;
                    } else if suffix_in_rn(w, r1, 3) {
                        w.letters[w.end as usize] = b'x';
                    }
                } else if (w.ends_with(b"abl") && suffix_in_rn(w, r2, 3))
                    || (w.ends_with(b"iqU") && suffix_in_rn(w, r2, 3))
                {
                    w.end -= 3;
                } else if (w.ends_with(&[b'i', 0xE8, b'r'])
                    && suffix_in_rn(w, rv, 3))
                    || (w.ends_with(&[b'I', 0xE8, b'r'])
                        && suffix_in_rn(w, rv, 3))
                {
                    w.end -= 2;
                    w.letters[w.end as usize] = b'i';
                }
                return true;
            }
        }
        // 29..31 — R2.
        for i in 29..31 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if w.ends_with(b"abil") {
                    if suffix_in_rn(w, r2, 4) {
                        w.end -= 4;
                    } else {
                        w.end -= 1;
                        w.letters[w.end as usize] = b'l';
                    }
                } else if w.ends_with(b"ic") {
                    if suffix_in_rn(w, r2, 2) {
                        w.end -= 2;
                    } else {
                        w.change_suffix(b"c", b"qU");
                    }
                } else if w.ends_with(b"iv") && suffix_in_rn(w, r2, 2) {
                    w.end -= 2;
                }
                return true;
            }
        }
        // 31..35 — R2.
        for i in 31..35 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if w.ends_with(b"at") && suffix_in_rn(w, r2, 2) {
                    w.end -= 2;
                    if w.ends_with(b"ic") {
                        if suffix_in_rn(w, r2, 2) {
                            w.end -= 2;
                        } else {
                            w.change_suffix(b"c", b"qU");
                        }
                    }
                }
                return true;
            }
        }
        // 35..37 — R2 or R1.
        for i in 35..37 {
            if w.ends_with(s[i]) {
                if suffix_in_rn(w, r2, s[i].len()) {
                    w.end -= s[i].len() as u8;
                    return true;
                } else if suffix_in_rn(w, r1, s[i].len()) {
                    w.change_suffix(s[i], b"eux");
                    return true;
                }
            }
        }
        // 37..39 — RV+1, vowel check.
        for i in 37..39 {
            if w.ends_with(s[i]) && suffix_in_rn(w, rv + 1, s[i].len())
                && Self::is_vowel(w.from_end(s[i].len() as u8))
            {
                w.end -= s[i].len() as u8;
                *force_step2a = true;
                return true;
            }
        }
        // Trailing special cases.
        if w.ends_with(b"eaux") || w.equals_str(b"eaux") {
            w.end -= 1;
            w.r#type |= super::word::french::PLURAL;
            true
        } else if w.ends_with(b"aux") && suffix_in_rn(w, r1, 3) {
            w.end -= 1;
            w.letters[w.end as usize] = b'l';
            w.r#type |= super::word::french::PLURAL;
            true
        } else if w.ends_with(b"amment") && suffix_in_rn(w, rv, 6) {
            w.change_suffix(b"amment", b"ant");
            *force_step2a = true;
            true
        } else if w.ends_with(b"emment") && suffix_in_rn(w, rv, 6) {
            w.change_suffix(b"emment", b"ent");
            *force_step2a = true;
            true
        } else {
            false
        }
    }

    fn step2a(w: &mut Word, rv: u32) -> bool {
        for i in 0..Self::STEP2A.len() {
            let suf = Self::STEP2A[i];
            if w.ends_with(suf) && suffix_in_rn(w, rv + 1, suf.len())
                && Self::is_consonant(w.from_end(suf.len() as u8))
            {
                w.end -= suf.len() as u8;
                if i == 31 { w.r#type |= lang::VERB; }
                return true;
            }
        }
        false
    }

    fn step2b(w: &mut Word, rv: u32, r2: u32) -> bool {
        for i in 0..Self::STEP2B.len() {
            let suf = Self::STEP2B[i];
            if w.ends_with(suf) && suffix_in_rn(w, rv, suf.len()) {
                match suf[0] {
                    b'a' | 0xE2 => {
                        w.end -= suf.len() as u8;
                        if w.ends_with(b"e") && suffix_in_rn(w, rv, 1) {
                            w.end -= 1;
                        }
                        return true;
                    }
                    _ => {
                        if i != 14 || suffix_in_rn(w, r2, suf.len()) {
                            w.end -= suf.len() as u8;
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn step3(w: &mut Word) {
        let f = w.letters[w.end as usize];
        if f == b'Y' {
            w.letters[w.end as usize] = b'i';
        } else if f == 0xE7 {
            w.letters[w.end as usize] = b'c';
        }
    }

    fn step4(w: &mut Word, rv: u32, r2: u32) -> bool {
        let mut res = false;
        if w.length() >= 2 && w.letters[w.end as usize] == b's'
            && !char_in(w.from_end(1), Self::SET_STEP4)
        {
            w.end -= 1;
            res = true;
        }
        for i in 0..Self::STEP4.len() {
            let suf = Self::STEP4[i];
            if w.ends_with(suf) && suffix_in_rn(w, rv, suf.len()) {
                match i {
                    2 => {
                        let prec = w.from_end(3);
                        if suffix_in_rn(w, r2, suf.len())
                            && suffix_in_rn(w, rv + 1, suf.len())
                            && (prec == b's' || prec == b't')
                        {
                            w.end -= 3;
                            return true;
                        }
                    }
                    5 => { w.end -= 1; return true; }
                    6 => {
                        if w.ends_with(&[b'g', b'u', 0xEB]) {
                            w.end -= 1;
                            return true;
                        }
                    }
                    _ => {
                        w.change_suffix(suf, b"i");
                        return true;
                    }
                }
            }
        }
        res
    }

    fn step5(w: &mut Word) -> bool {
        for suf in Self::STEP5 {
            if w.ends_with(suf) {
                w.end -= 1;
                return true;
            }
        }
        false
    }

    fn step6(w: &mut Word) -> bool {
        let mut i = w.end as i32;
        while i >= w.start as i32 {
            if Self::is_vowel(w.letters[i as usize]) {
                if (i as u8) < w.end && (w.letters[i as usize] & 0xFE) == 0xE8 {
                    w.letters[i as usize] = b'e';
                    return true;
                }
                return false;
            }
            i -= 1;
        }
        false
    }

    pub fn hash(w: &mut Word) {
        w.hash[2] = !0xeff1cace_u64;
        w.hash[3] = !0xeff1cace_u64;
        for i in w.start..=w.end {
            let l = w.letters[i as usize];
            w.hash[2] = w.hash[2].wrapping_mul(251 * 32).wrapping_add(l as u64);
            if Self::is_vowel(l) {
                w.hash[3] = w.hash[3].wrapping_mul(997 * 16).wrapping_add(l as u64);
            } else if (b'b'..=b'z').contains(&l) {
                w.hash[3] = w.hash[3].wrapping_mul(271 * 32)
                    .wrapping_add((l - 97) as u64);
            } else {
                w.hash[3] = w.hash[3].wrapping_mul(11 * 32).wrapping_add(l as u64);
            }
        }
    }

    pub fn stem(w: &mut Word) -> bool {
        Self::convert_utf8(w);
        if w.length() < 2 {
            Self::hash(w);
            return false;
        }
        for i in 0..FR_EXCEPTIONS.len() {
            let (from, to) = FR_EXCEPTIONS[i];
            if w.equals_str(from) {
                for k in 0..to.len() {
                    w.letters[w.start as usize + k] = to[k];
                }
                w.end = w.start + (to.len() - 1) as u8;
                Self::hash(w);
                w.r#type |= FR_TYPES_EXCEPTIONS[i];
                w.language = LanguageId::French as u64;
                return true;
            }
        }
        Self::mark_vowels_as_consonants(w);
        let rv = Self::get_rv(w);
        let r1 = get_region(w, 0, Self::is_vowel);
        let r2 = get_region(w, r1, Self::is_vowel);
        let mut do_next = false;
        let mut res = Self::step1(w, rv, r1, r2, &mut do_next);
        do_next |= !res;
        if do_next {
            do_next = !Self::step2a(w, rv);
            res |= !do_next;
            if do_next {
                res |= Self::step2b(w, rv, r2);
            }
        }
        if res {
            Self::step3(w);
        } else {
            res |= Self::step4(w, rv, r2);
        }
        res |= Self::step5(w);
        res |= Self::step6(w);
        for i in w.start..=w.end {
            w.letters[i as usize] = w.letters[i as usize].to_ascii_lowercase();
        }
        if !res {
            res = w.matches_any(&FR_COMMON_WORDS);
        }
        Self::hash(w);
        if res {
            w.language = LanguageId::French as u64;
        }
        res
    }
}

// =============================================================
// GermanStemmer — paq8.cpp:2832-2998.
// =============================================================

const DE_VOWELS: &[u8] = &[
    b'a', b'e', b'i', b'o', b'u', b'y', 0xE4, 0xF6, 0xFC,
];
const DE_COMMON_WORDS: [&[u8]; 10] = [
    b"der", b"die", b"das", b"und", b"sie", b"ich", b"mit", b"sich",
    b"auf", b"nicht",
];
const DE_ENDINGS: &[u8] =
    &[b'b', b'd', b'f', b'g', b'h', b'k', b'l', b'm', b'n', b't'];

/// German Porter stemmer — paq8.cpp:2832-2998.
pub struct GermanStemmer;

impl GermanStemmer {
    #[inline]
    pub fn is_vowel(c: u8) -> bool { char_in(c, DE_VOWELS) }

    const STEP1: [&'static [u8]; 6] =
        [b"em", b"ern", b"er", b"e", b"en", b"es"];
    const STEP2: [&'static [u8]; 3] = [b"en", b"er", b"est"];
    const STEP3: [&'static [u8]; 7] =
        [b"end", b"ung", b"ik", b"ig", b"isch", b"lich", b"heit"];

    fn convert_utf8(w: &mut Word) {
        let mut i = w.start;
        while i < w.end {
            let next = w.letters[(i + 1) as usize];
            let c = next.wrapping_add(if next < 0x9F { 0x60 } else { 0x40 });
            if w.letters[i as usize] == 0xC3 && (Self::is_vowel(c) || c == 0xDF) {
                w.letters[i as usize] = c;
                if i + 1 < w.end {
                    let span = (w.end - i - 1) as usize;
                    for k in 0..span {
                        w.letters[(i + 1) as usize + k] =
                            w.letters[(i + 2) as usize + k];
                    }
                }
                w.end -= 1;
            }
            i += 1;
        }
    }

    fn replace_sharp_s(w: &mut Word) {
        let mut i = w.start;
        while i <= w.end {
            if w.letters[i as usize] == 0xDF {
                w.letters[i as usize] = b's';
                if (i as usize + 1) < MAX_WORD_SIZE {
                    let mut k = MAX_WORD_SIZE - 1;
                    while k > (i as usize + 1) {
                        w.letters[k] = w.letters[k - 1];
                        k -= 1;
                    }
                    w.letters[(i + 1) as usize] = b's';
                    if (w.end as usize) < MAX_WORD_SIZE - 1 { w.end += 1; }
                }
            }
            i += 1;
        }
    }

    fn mark_vowels_as_consonants(w: &mut Word) {
        if w.end <= w.start { return; }
        for i in (w.start + 1)..w.end {
            let c = w.letters[i as usize];
            if (c == b'u' || c == b'y')
                && Self::is_vowel(w.letters[(i - 1) as usize])
                && Self::is_vowel(w.letters[(i + 1) as usize])
            {
                w.letters[i as usize] = c.to_ascii_uppercase();
            }
        }
    }

    fn is_valid_ending(c: u8, include_r: bool) -> bool {
        char_in(c, DE_ENDINGS) || (include_r && c == b'r')
    }

    fn step1(w: &mut Word, r1: u32) -> bool {
        for i in 0..3 {
            let suf = Self::STEP1[i];
            if w.ends_with(suf) && suffix_in_rn(w, r1, suf.len()) {
                w.end -= suf.len() as u8;
                return true;
            }
        }
        for i in 3..6 {
            let suf = Self::STEP1[i];
            if w.ends_with(suf) && suffix_in_rn(w, r1, suf.len()) {
                w.end -= suf.len() as u8;
                w.end -= w.ends_with(b"niss") as u8;
                return true;
            }
        }
        if w.ends_with(b"s") && suffix_in_rn(w, r1, 1)
            && Self::is_valid_ending(w.from_end(1), true)
        {
            w.end -= 1;
            return true;
        }
        false
    }

    fn step2(w: &mut Word, r1: u32) -> bool {
        for suf in Self::STEP2 {
            if w.ends_with(suf) && suffix_in_rn(w, r1, suf.len()) {
                w.end -= suf.len() as u8;
                return true;
            }
        }
        if w.ends_with(b"st") && suffix_in_rn(w, r1, 2) && w.length() > 5
            && Self::is_valid_ending(w.from_end(2), false)
        {
            w.end -= 2;
            return true;
        }
        false
    }

    fn step3(w: &mut Word, r1: u32, r2: u32) -> bool {
        let s = &Self::STEP3;
        for i in 0..2 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if w.ends_with(b"ig") && w.from_end(2) != b'e'
                    && suffix_in_rn(w, r2, 2)
                {
                    w.end -= 2;
                }
                if i != 0 { w.r#type |= lang::NOUN; }
                return true;
            }
        }
        for i in 2..5 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len())
                && w.from_end(s[i].len() as u8) != b'e'
            {
                w.end -= s[i].len() as u8;
                if i > 2 { w.r#type |= super::word::german::ADJECTIVE; }
                return true;
            }
        }
        for i in 5..7 {
            if w.ends_with(s[i]) && suffix_in_rn(w, r2, s[i].len()) {
                w.end -= s[i].len() as u8;
                if (w.ends_with(b"er") || w.ends_with(b"en"))
                    && suffix_in_rn(w, r1, 2)
                {
                    w.end -= 2;
                }
                if i > 5 {
                    w.r#type |= lang::NOUN | super::word::german::FEMALE;
                }
                return true;
            }
        }
        if w.ends_with(b"keit") && suffix_in_rn(w, r2, 4) {
            w.end -= 4;
            if w.ends_with(b"lich") && suffix_in_rn(w, r2, 4) {
                w.end -= 4;
            } else if w.ends_with(b"ig") && suffix_in_rn(w, r2, 2) {
                w.end -= 2;
            }
            w.r#type |= lang::NOUN | super::word::german::FEMALE;
            return true;
        }
        false
    }

    pub fn hash(w: &mut Word) {
        w.hash[2] = !0xbea7ab1e_u64;
        w.hash[3] = !0xbea7ab1e_u64;
        for i in w.start..=w.end {
            let l = w.letters[i as usize];
            w.hash[2] = w.hash[2].wrapping_mul(263 * 32).wrapping_add(l as u64);
            if Self::is_vowel(l) {
                w.hash[3] = w.hash[3].wrapping_mul(997 * 16).wrapping_add(l as u64);
            } else if (b'b'..=b'z').contains(&l) {
                w.hash[3] = w.hash[3].wrapping_mul(251 * 32)
                    .wrapping_add((l - 97) as u64);
            } else {
                w.hash[3] = w.hash[3].wrapping_mul(11 * 32).wrapping_add(l as u64);
            }
        }
    }

    pub fn stem(w: &mut Word) -> bool {
        Self::convert_utf8(w);
        if w.length() < 2 {
            Self::hash(w);
            return false;
        }
        Self::replace_sharp_s(w);
        Self::mark_vowels_as_consonants(w);
        let r1 = get_region(w, 0, Self::is_vowel).min(3);
        let r2 = get_region(w, get_region(w, 0, Self::is_vowel), Self::is_vowel);
        let mut res = Self::step1(w, r1);
        res |= Self::step2(w, r1);
        res |= Self::step3(w, r1, r2);
        for i in w.start..=w.end {
            match w.letters[i as usize] {
                0xE4 => w.letters[i as usize] = b'a',
                0xF6 | 0xFC => {
                    w.letters[i as usize] -= 0x87;
                }
                _ => {
                    w.letters[i as usize] =
                        w.letters[i as usize].to_ascii_lowercase();
                }
            }
        }
        if !res {
            res = w.matches_any(&DE_COMMON_WORDS);
        }
        Self::hash(w);
        if res {
            w.language = LanguageId::German as u64;
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(s: &[u8]) -> Word {
        let mut w = Word::new();
        for &c in s { w.push(c); }
        w
    }

    #[test]
    fn stems_basic_plural() {
        let mut w = mk(b"cats");
        EnglishStemmer::stem(&mut w);
        assert!(w.equals_str(b"cat"), "got {:?}",
            &w.letters[w.start as usize..=w.end as usize]);
        assert!(w.r#type & english::PLURAL != 0);
    }

    #[test]
    fn stems_present_participle() {
        let mut w = mk(b"running");
        EnglishStemmer::stem(&mut w);
        // "running" → "run" via Step1b (ing removal + double drop).
        assert!(w.equals_str(b"run"), "got {:?}",
            &w.letters[w.start as usize..=w.end as usize]);
    }

    #[test]
    fn recognises_male_pronoun() {
        let mut w = mk(b"he");
        EnglishStemmer::stem(&mut w);
        assert!(w.r#type & english::MALE != 0);
    }

    #[test]
    fn exception1_skies_becomes_sky() {
        let mut w = mk(b"skies");
        EnglishStemmer::stem(&mut w);
        assert!(w.equals_str(b"sky"));
    }

    #[test]
    fn short_word_does_not_panic() {
        let mut w = mk(b"a");
        assert!(!EnglishStemmer::stem(&mut w));
    }

    #[test]
    fn common_word_tagged_english() {
        let mut w = mk(b"the");
        let res = EnglishStemmer::stem(&mut w);
        assert!(res);
        assert_eq!(w.language, LanguageId::English as u64);
    }

    #[test]
    fn french_common_word_tagged_french() {
        let mut w = mk(b"que");
        let res = FrenchStemmer::stem(&mut w);
        assert!(res);
        assert_eq!(w.language, LanguageId::French as u64);
    }

    #[test]
    fn french_strips_ment_suffix() {
        let mut w = mk(b"rapidement");
        FrenchStemmer::stem(&mut w);
        // "ment" suffix removed by Step1 (i=37/38, RV+1, vowel check).
        assert!(w.length() < 10);
    }

    #[test]
    fn german_common_word_tagged_german() {
        let mut w = mk(b"nicht");
        let res = GermanStemmer::stem(&mut w);
        assert!(res);
        assert_eq!(w.language, LanguageId::German as u64);
    }

    #[test]
    fn german_strips_plural_suffix() {
        let mut w = mk(b"hunde");
        GermanStemmer::stem(&mut w);
        // "e" suffix from Step1 — "hunde" → "hund".
        assert!(w.length() <= 5);
    }

    #[test]
    fn stemmers_never_panic_on_short_or_weird_input() {
        for input in [&b"x"[..], b"ab", b"qqqq", b"'", b"a'b'c"] {
            let mut a = mk(input);
            let _ = EnglishStemmer::stem(&mut a);
            let mut b = mk(input);
            let _ = FrenchStemmer::stem(&mut b);
            let mut c = mk(input);
            let _ = GermanStemmer::stem(&mut c);
        }
    }
}
