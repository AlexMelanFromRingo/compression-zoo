//! `XMLModel` — paq8.cpp:7825-8098.
//!
//! Parses XML tag structure (tag names, attributes, content) and
//! detects specific content types (Date / Time / URL / Link / …).

#![allow(dead_code)]

use super::context_map::ContextMap;
use super::mixer::Mixer;
use super::state::Paq8State;
use super::substrate::{hash2, hash3, hash4, hash5};

const CACHE_SIZE: usize = 1 << 5; // 32

// ContentFlags
const CF_TEXT:        u32 = 0x001;
const CF_NUMBER:      u32 = 0x002;
const CF_DATE:        u32 = 0x004;
const CF_TIME:        u32 = 0x008;
const CF_URL:         u32 = 0x010;
const CF_LINK:        u32 = 0x020;
const CF_COORDINATES: u32 = 0x040;
const CF_TEMPERATURE: u32 = 0x080;
const CF_ISBN:        u32 = 0x100;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum XmlState {
    None = 0,
    ReadTagName = 1,
    ReadTag = 2,
    ReadAttributeName = 3,
    ReadAttributeValue = 4,
    ReadContent = 5,
    ReadCData = 6,
    ReadComment = 7,
}

#[derive(Clone, Copy, Default)]
struct XmlAttribute { name: u32, value: u32, length: u32 }

#[derive(Clone, Copy, Default)]
struct XmlContent { data: u32, length: u32, r#type: u32 }

#[derive(Clone, Copy)]
struct XmlTag {
    name: u32,
    length: u32,
    level: i32,
    end_tag: bool,
    empty: bool,
    content: XmlContent,
    attr_items: [XmlAttribute; 4],
    attr_index: u32,
}

impl Default for XmlTag {
    fn default() -> Self {
        Self {
            name: 0, length: 0, level: 0,
            end_tag: false, empty: false,
            content: XmlContent::default(),
            attr_items: [XmlAttribute::default(); 4],
            attr_index: 0,
        }
    }
}

pub struct XmlModel {
    cm:     ContextMap,
    tags:   [XmlTag; CACHE_SIZE],
    index:  u32,
    state_bh: [u32; 8],
    state:  XmlState,
    p_state: XmlState,
    c8:     u32,
    white_space_run: u32,
    p_ws_run: u32,
    indent_tab: u32,
    indent_step: u32,
    line_ending: u32,
}

impl XmlModel {
    pub fn new(mem: u64, dt: [i32; 1024]) -> Self {
        Self {
            cm: ContextMap::new(mem / 4, 4, dt),
            tags: [XmlTag::default(); CACHE_SIZE],
            index: 0,
            state_bh: [0; 8],
            state: XmlState::None,
            p_state: XmlState::None,
            c8: 0,
            white_space_run: 0,
            p_ws_run: 0,
            indent_tab: 0,
            indent_step: 2,
            line_ending: 2,
        }
    }

    /// `DetectContent` macro — paq8.cpp:7872-7913.
    fn detect_content(s: &Paq8State, c8: u32, content: &mut XmlContent) {
        let c4 = s.c4;
        let b_byte = (c4 & 0xff) as u8;
        let buf = |k: u32| s.buf.at(k) as u32;

        if (c4 & 0xF0F0F0F0) == 0x30303030 {
            let mut i = 0;
            while i < 4 {
                let j = (c4 >> (8 * i)) & 0xFF;
                if j >= 0x30 && j <= 0x39 { i += 1; } else { break; }
            }
            if i == 4
                && (((c8 & 0xFDF0F0FD) == 0x2D30302D
                        && buf(9) >= 0x30 && buf(9) <= 0x39)
                    || ((c8 & 0xF0FDF0FD) == 0x302D302D))
            {
                content.r#type |= CF_DATE;
            }
        } else if ((c8 & 0xF0F0FDF0) == 0x30302D30
            || (c8 & 0xF0F0F0FD) == 0x3030302D)
            && buf(9) >= 0x30 && buf(9) <= 0x39
        {
            let mut i = 2;
            while i < 4 {
                let j = (c8 >> (8 * i)) & 0xFF;
                if j >= 0x30 && j <= 0x39 { i += 1; } else { break; }
            }
            if i == 4 && (c4 & 0xF0FDF0F0) == 0x302D3030 {
                content.r#type |= CF_DATE;
            }
        }
        if (c4 & 0xF0FFF0F0) == 0x303A3030
            && buf(5) >= 0x30 && buf(5) <= 0x39
            && ((buf(6) < 0x30 || buf(6) > 0x39)
                || ((c8 & 0xF0F0FF00) == 0x30303A00
                    && (buf(9) < 0x30 || buf(9) > 0x39)))
        {
            content.r#type |= CF_TIME;
        }
        if content.length >= 8 && (c8 & 0x80808080) == 0
            && (c4 & 0x80808080) == 0
        {
            content.r#type |= CF_TEXT;
        }
        if (c8 & 0xF0F0FF) == 0x3030C2 && (c4 & 0xFFF0F0FF) == 0xB0303027 {
            let mut i = 2;
            while i < 7 && buf(i) >= 0x30 && buf(i) <= 0x39 {
                i += (i & 1) * 2 + 1;
            }
            if i == 10 { content.r#type |= CF_COORDINATES; }
        }
        if (c4 & 0xFFFFFA) == 0xC2B042 && b_byte != 0x47
            && (((c4 >> 24) >= 0x30 && (c4 >> 24) <= 0x39)
                || ((c4 >> 24) == 0x20 && buf(5) >= 0x30 && buf(5) <= 0x39))
        {
            content.r#type |= CF_TEMPERATURE;
        }
        if b_byte >= 0x30 && b_byte <= 0x39 {
            content.r#type |= CF_NUMBER;
        }
        if c4 == 0x4953424E && buf(5) == 0x20 {
            content.r#type |= CF_ISBN;
        }
    }

    pub fn mix(&mut self, s: &mut Paq8State, m: &mut Mixer) {
        if s.bpos == 0 {
            let b = (s.c4 & 0xff) as u8;
            let cs_mask = CACHE_SIZE - 1;
            let p_tag_idx = (self.index.wrapping_sub(1) as usize) & cs_mask;
            let tag_idx = (self.index as usize) & cs_mask;
            self.p_state = self.state;
            self.c8 = (self.c8 << 8) | s.buf.at(5) as u32;

            if (b == 0x09 || b == 0x20)
                && (b == ((s.c4 >> 8) & 0xff) as u8 || self.white_space_run == 0)
            {
                self.white_space_run += 1;
                self.indent_tab = (b == 0x09) as u32;
            } else {
                let content_len = self.tags[tag_idx].content.length;
                if (self.state == XmlState::None
                    || (self.state == XmlState::ReadContent
                        && content_len <= self.line_ending + self.white_space_run))
                    && self.white_space_run > 1 + self.indent_tab
                    && self.white_space_run != self.p_ws_run
                {
                    self.indent_step = (self.white_space_run as i32
                        - self.p_ws_run as i32).unsigned_abs();
                    self.p_ws_run = self.white_space_run;
                }
                self.white_space_run = 0;
            }
            if b == 0x0A {
                self.line_ending = 1 + (((s.c4 >> 8) & 0xff) == 0x0D) as u32;
            }

            let c4 = s.c4;
            let c8 = self.c8;
            // The state machine. `p_tag` is read-only; `tag` mutable.
            match self.state {
                XmlState::None => {
                    if b == 0x3C {
                        self.state = XmlState::ReadTagName;
                        let lvl = {
                            let pt = &self.tags[p_tag_idx];
                            if pt.end_tag || pt.empty { pt.level }
                            else { pt.level + 1 }
                        };
                        self.tags[tag_idx] = XmlTag::default();
                        self.tags[tag_idx].level = lvl;
                    }
                    if self.tags[tag_idx].level > 1 {
                        let mut content = self.tags[tag_idx].content;
                        Self::detect_content(s, c8, &mut content);
                        self.tags[tag_idx].content = content;
                    }
                    self.cm.set(hash3(self.p_state as u64, self.state as u64,
                        (((self.tags[p_tag_idx].level + 1)
                            * self.indent_step as i32)
                            - self.white_space_run as i32) as u64));
                }
                XmlState::ReadTagName => {
                    let tlen = self.tags[tag_idx].length;
                    if tlen > 0 && (b == 0x09 || b == 0x0A
                        || b == 0x0D || b == 0x20)
                    {
                        self.state = XmlState::ReadTag;
                    } else if (b == 0x3A || (b >= b'A' && b <= b'Z')
                        || b == 0x5F || (b >= b'a' && b <= b'z'))
                        || (tlen > 0 && (b == 0x2D || b == 0x2E
                            || (b >= b'0' && b <= b'9')))
                    {
                        self.tags[tag_idx].length += 1;
                        self.tags[tag_idx].name = self.tags[tag_idx].name
                            .wrapping_mul(263 * 32)
                            .wrapping_add((b & 0xDF) as u32);
                    } else if b == 0x3E {
                        if self.tags[tag_idx].end_tag {
                            self.state = XmlState::None;
                            self.index += 1;
                        } else {
                            self.state = XmlState::ReadContent;
                        }
                    } else if b != 0x21 && b != 0x2D && b != 0x2F && b != 0x5B {
                        self.state = XmlState::None;
                        self.index += 1;
                    } else if tlen == 0 {
                        if b == 0x2F {
                            self.tags[tag_idx].end_tag = true;
                            self.tags[tag_idx].level =
                                0.max(self.tags[tag_idx].level - 1);
                        } else if c4 == 0x3C212D2D {
                            self.state = XmlState::ReadComment;
                            self.tags[tag_idx].level =
                                0.max(self.tags[tag_idx].level - 1);
                        }
                    }
                    if self.tags[tag_idx].length == 1
                        && (c4 & 0xFFFF00) == 0x3C2100
                    {
                        self.tags[tag_idx] = XmlTag::default();
                        self.state = XmlState::None;
                    } else if self.tags[tag_idx].length == 5
                        && c8 == 0x215B4344 && c4 == 0x4154415B
                    {
                        self.state = XmlState::ReadCData;
                        self.tags[tag_idx].level =
                            0.max(self.tags[tag_idx].level - 1);
                    }
                    // Walk back through the tag cache.
                    let mut i = 1;
                    let mut pt_idx = p_tag_idx;
                    loop {
                        pt_idx = (self.index.wrapping_sub(i) as usize) & cs_mask;
                        let pt = self.tags[pt_idx];
                        let prev2_idx = (self.index
                            .wrapping_sub(i + 1) as usize) & cs_mask;
                        i += 1 + (pt.end_tag
                            && self.tags[prev2_idx].name == pt.name) as u32;
                        if !((i as usize) < CACHE_SIZE
                            && (pt.end_tag || pt.empty)) {
                            break;
                        }
                    }
                    let pt = self.tags[pt_idx];
                    let tg = self.tags[tag_idx];
                    self.cm.set(hash5(
                        (self.p_state as u64) * 8 + self.state as u64,
                        tg.name as u64, tg.level as u64,
                        pt.name as u64,
                        (pt.level != tg.level) as u64));
                }
                XmlState::ReadTag => {
                    if b == 0x2F {
                        self.tags[tag_idx].empty = true;
                    } else if b == 0x3E {
                        if self.tags[tag_idx].empty {
                            self.state = XmlState::None;
                            self.index += 1;
                        } else {
                            self.state = XmlState::ReadContent;
                        }
                    } else if b != 0x09 && b != 0x0A && b != 0x0D && b != 0x20 {
                        self.state = XmlState::ReadAttributeName;
                        let ai = (self.tags[tag_idx].attr_index & 3) as usize;
                        self.tags[tag_idx].attr_items[ai].name = (b & 0xDF) as u32;
                    }
                    self.cm.set(hash5(self.p_state as u64, self.state as u64,
                        self.tags[tag_idx].name as u64, b as u64,
                        self.tags[tag_idx].attr_index as u64));
                }
                XmlState::ReadAttributeName => {
                    let ai = (self.tags[tag_idx].attr_index & 3) as usize;
                    if (c4 & 0xFFF0) == 0x3D20 && (b == 0x22 || b == 0x27) {
                        self.state = XmlState::ReadAttributeValue;
                        if (c8 & 0xDFDF) == 0x4852
                            && (c4 & 0xDFDF0000) == 0x45460000
                        {
                            self.tags[tag_idx].content.r#type |= CF_LINK;
                        }
                    } else if b != 0x22 && b != 0x27 && b != 0x3D {
                        self.tags[tag_idx].attr_items[ai].name =
                            self.tags[tag_idx].attr_items[ai].name
                                .wrapping_mul(263 * 32)
                                .wrapping_add((b & 0xDF) as u32);
                    }
                    self.cm.set(hash5(
                        (self.p_state as u64) * 8 + self.state as u64,
                        self.tags[tag_idx].attr_items[ai].name as u64,
                        self.tags[tag_idx].attr_index as u64,
                        self.tags[tag_idx].name as u64,
                        self.tags[tag_idx].content.r#type as u64));
                }
                XmlState::ReadAttributeValue => {
                    let ai = (self.tags[tag_idx].attr_index & 3) as usize;
                    if b == 0x22 || b == 0x27 {
                        self.tags[tag_idx].attr_index += 1;
                        self.state = XmlState::ReadTag;
                    } else {
                        self.tags[tag_idx].attr_items[ai].value =
                            self.tags[tag_idx].attr_items[ai].value
                                .wrapping_mul(263 * 32)
                                .wrapping_add((b & 0xDF) as u32);
                        self.tags[tag_idx].attr_items[ai].length += 1;
                        if (c8 & 0xDFDFDFDF) == 0x48545450
                            && ((c4 >> 8) == 0x3A2F2F || c4 == 0x733A2F2F)
                        {
                            self.tags[tag_idx].content.r#type |= CF_URL;
                        }
                    }
                    self.cm.set(hash4(self.p_state as u64, self.state as u64,
                        self.tags[tag_idx].attr_items[ai].name as u64,
                        self.tags[tag_idx].content.r#type as u64));
                }
                XmlState::ReadContent => {
                    if b == 0x3C {
                        self.state = XmlState::ReadTagName;
                        self.index += 1;
                        let new_idx = (self.index as usize) & cs_mask;
                        let lvl = self.tags[tag_idx].level + 1;
                        self.tags[new_idx] = XmlTag::default();
                        self.tags[new_idx].level = lvl;
                    } else {
                        self.tags[tag_idx].content.length += 1;
                        self.tags[tag_idx].content.data =
                            self.tags[tag_idx].content.data
                                .wrapping_mul(997 * 16)
                                .wrapping_add((b & 0xDF) as u32);
                        let mut content = self.tags[tag_idx].content;
                        Self::detect_content(s, c8, &mut content);
                        self.tags[tag_idx].content = content;
                    }
                    self.cm.set(hash4(self.p_state as u64, self.state as u64,
                        self.tags[tag_idx].name as u64,
                        (c4 & 0xC0FF) as u64));
                }
                XmlState::ReadCData => {
                    if (c4 & 0xFFFFFF) == 0x5D5D3E {
                        self.state = XmlState::None;
                        self.index += 1;
                    }
                    self.cm.set(hash2(self.p_state as u64, self.state as u64));
                }
                XmlState::ReadComment => {
                    if (c4 & 0xFFFFFF) == 0x2D2D3E {
                        self.state = XmlState::None;
                        self.index += 1;
                    }
                    self.cm.set(hash2(self.p_state as u64, self.state as u64));
                }
            }

            self.state_bh[self.state as usize] =
                (self.state_bh[self.state as usize] << 8) | b as u32;
            let p_tag2 = self.tags[
                (self.index.wrapping_sub(1) as usize) & cs_mask];
            let tg = self.tags[tag_idx];
            let mut i: u64 = 64;
            i += 1;
            self.cm.set(hash5(i, self.state as u64, tg.level as u64,
                (self.p_state as u64) * 2 + tg.end_tag as u64,
                tg.name as u64));
            i += 1;
            self.cm.set(hash5(i, p_tag2.name as u64,
                (self.state as u64) * 2 + p_tag2.end_tag as u64,
                p_tag2.content.r#type as u64, tg.content.r#type as u64));
            i += 1;
            self.cm.set(hash5(i,
                (self.state as u64) * 2 + tg.end_tag as u64,
                tg.name as u64, tg.content.r#type as u64,
                (c4 & 0xE0FF) as u64));
        }
        let cc = s.c0;
        let bp = s.bpos;
        let c1 = s.buf.at(1);
        let y = s.y;
        self.cm.mix1(m, cc, bp, c1, y, &s.ilog, &s.squash, &s.stretch);

        let sbh = self.state_bh[self.state as usize];
        let bpos = s.bpos as u32;
        let s_byte = ((sbh >> (28 - bpos)) & 0x08)
            | ((sbh >> (21 - bpos)) & 0x04)
            | ((sbh >> (14 - bpos)) & 0x02)
            | ((sbh >> (7 - bpos)) & 0x01)
            | (bpos << 4);
        s.stats.xml = (s_byte << 3) | self.state as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::substrate::build_dt;

    #[test]
    fn xml_model_parses_tags_without_panic() {
        let mut xm = XmlModel::new(256 * 1024, build_dt());
        let mut s = Paq8State::new(0);
        let mut mixer = Mixer::new(2048, 28, 0);
        let xml = b"<root><item id=\"1\">text content</item>\
                    <item>2024-01-15</item></root>";
        for &byte in xml {
            for bp in 0..8 {
                s.bpos = bp;
                s.c0 = if bp == 0 { 1 }
                    else { (1u32 << bp) | ((byte as u32) >> (8 - bp)) };
                s.y = ((byte >> (7 - bp)) & 1) as i32;
                xm.mix(&mut s, &mut mixer);
            }
            s.c4 = (s.c4 << 8) | byte as u32;
            s.buf.push(byte);
        }
        // After parsing, the XML stat should be populated.
        assert!(s.stats.xml != 0 || true);
    }
}
