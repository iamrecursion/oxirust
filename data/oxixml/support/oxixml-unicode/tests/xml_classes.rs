//! XML character-class production tests, checked against the grammar ranges.

use oxixml_unicode::xml::{
    XmlVersion, is_char, is_char_v10, is_char_v11, is_name_char, is_name_start_char,
    is_restricted_char, is_xml_whitespace,
};

/// Reference implementation of the XML 1.0 (Fifth Edition) `Char` production,
/// written straight from the grammar for cross-checking.
fn ref_char_v10(cp: u32) -> bool {
    cp == 0x9
        || cp == 0xA
        || cp == 0xD
        || (0x20..=0xD7FF).contains(&cp)
        || (0xE000..=0xFFFD).contains(&cp)
        || (0x10000..=0x10FFFF).contains(&cp)
}

/// Reference implementation of the XML 1.1 `Char` production.
fn ref_char_v11(cp: u32) -> bool {
    (0x1..=0xD7FF).contains(&cp)
        || (0xE000..=0xFFFD).contains(&cp)
        || (0x10000..=0x10FFFF).contains(&cp)
}

#[test]
fn char_productions_match_reference() {
    for cp in 0u32..=0x11000 {
        if let Some(c) = char::from_u32(cp) {
            assert_eq!(is_char_v10(c), ref_char_v10(cp), "v10 U+{cp:04X}");
            assert_eq!(is_char_v11(c), ref_char_v11(cp), "v11 U+{cp:04X}");
            assert_eq!(is_char(c, XmlVersion::V10), is_char_v10(c));
            assert_eq!(is_char(c, XmlVersion::V11), is_char_v11(c));
        }
    }
    // A supplementary-plane spot check.
    assert!(is_char_v10('\u{10FFFF}'));
    assert!(is_char_v11('\u{10FFFF}'));
}

#[test]
fn nul_and_controls() {
    assert!(!is_char_v10('\u{0}'));
    assert!(!is_char_v11('\u{0}'));
    // XML 1.0 forbids C0 controls other than tab/lf/cr.
    for cp in [0x1u32, 0x8, 0xB, 0xC, 0xE, 0x1F] {
        let c = char::from_u32(cp).expect("valid scalar");
        assert!(!is_char_v10(c), "v10 must reject U+{cp:04X}");
        assert!(is_char_v11(c), "v11 must accept U+{cp:04X}");
    }
    // Noncharacters FFFE/FFFF are never Char.
    assert!(!is_char_v10('\u{FFFE}'));
    assert!(!is_char_v11('\u{FFFF}'));
}

#[test]
fn restricted_chars() {
    fn ref_restricted(cp: u32) -> bool {
        (0x1..=0x8).contains(&cp)
            || (0xB..=0xC).contains(&cp)
            || (0xE..=0x1F).contains(&cp)
            || (0x7F..=0x84).contains(&cp)
            || (0x86..=0x9F).contains(&cp)
    }
    for cp in 0u32..=0x200 {
        let c = char::from_u32(cp).expect("valid scalar");
        assert_eq!(is_restricted_char(c), ref_restricted(cp), "U+{cp:04X}");
    }
    // Every restricted char is a valid XML 1.1 Char.
    for cp in 0u32..=0x9F {
        let c = char::from_u32(cp).expect("valid scalar");
        if is_restricted_char(c) {
            assert!(is_char_v11(c));
        }
    }
}

#[test]
fn name_start_char_boundaries() {
    // Explicit members.
    for c in [':', '_', 'A', 'Z', 'a', 'z', 'é', '\u{2070}', '\u{10000}'] {
        assert!(is_name_start_char(c), "{c:?} should start a name");
    }
    // Explicit non-members near the boundaries.
    for c in ['-', '.', '0', '9', ' ', '\u{B7}', '\u{2000}', '\u{37E}'] {
        assert!(!is_name_start_char(c), "{c:?} should not start a name");
    }
    // Every NameStartChar is also a NameChar.
    for cp in 0u32..=0x3FFF {
        let c = char::from_u32(cp).expect("valid scalar");
        if is_name_start_char(c) {
            assert!(is_name_char(c));
        }
    }
}

#[test]
fn name_char_extras() {
    for c in [
        '-', '.', '0', '9', '\u{B7}', '\u{300}', '\u{36F}', '\u{203F}', '\u{2040}',
    ] {
        assert!(is_name_char(c), "{c:?} should be a name char");
    }
    for c in [' ', '\t', '\u{37E}', '\u{2041}'] {
        assert!(!is_name_char(c), "{c:?} should not be a name char");
    }
}

#[test]
fn whitespace() {
    for c in [' ', '\t', '\r', '\n'] {
        assert!(is_xml_whitespace(c));
    }
    for c in ['\u{A0}', '\u{2028}', '\u{0C}', 'x'] {
        assert!(!is_xml_whitespace(c));
    }
}
