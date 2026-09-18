//! Tests for general categories, category groups, blocks, scripts, and case.

use oxixml_unicode::UNICODE_VERSION;
use oxixml_unicode::blocks::{block, block_range, in_block};
use oxixml_unicode::case::{
    fold_str, simple_fold, simple_lowercase, simple_titlecase, simple_uppercase,
};
use oxixml_unicode::gc::{GeneralCategory, general_category, matches_group};
use oxixml_unicode::scripts::{Script, script};

#[test]
fn general_categories() {
    assert_eq!(general_category('A'), GeneralCategory::UppercaseLetter);
    assert_eq!(general_category('z'), GeneralCategory::LowercaseLetter);
    assert_eq!(general_category('0'), GeneralCategory::DecimalNumber);
    assert_eq!(general_category(' '), GeneralCategory::SpaceSeparator);
    assert_eq!(general_category('\u{2028}'), GeneralCategory::LineSeparator);
    assert_eq!(
        general_category('\u{2029}'),
        GeneralCategory::ParagraphSeparator
    );
    assert_eq!(general_category('_'), GeneralCategory::ConnectorPunctuation);
    assert_eq!(general_category('\u{0}'), GeneralCategory::Control);
    assert_eq!(
        general_category('\u{0301}'),
        GeneralCategory::NonspacingMark
    );
    assert_eq!(
        general_category('\u{20AC}'),
        GeneralCategory::CurrencySymbol
    ); // €
    // A precomposed Hangul syllable is an OtherLetter.
    assert_eq!(general_category('\u{D799}'), GeneralCategory::OtherLetter);
    // Unassigned / noncharacter.
    assert_eq!(general_category('\u{10FFFF}'), GeneralCategory::Unassigned);
    // Abbreviations round-trip category identity.
    assert_eq!(GeneralCategory::MathSymbol.abbreviation(), "Sm");
}

#[test]
fn category_groups() {
    assert!(matches_group('A', 'L'));
    assert!(matches_group('a', 'L'));
    assert!(matches_group('0', 'N'));
    assert!(matches_group('\u{0301}', 'M'));
    assert!(matches_group('.', 'P'));
    assert!(matches_group('+', 'S'));
    assert!(matches_group(' ', 'Z'));
    assert!(matches_group('\u{0}', 'C'));
    // Unassigned code points fall in group C via Cn.
    assert!(matches_group('\u{10FFFF}', 'C'));
    // Wrong group and non-group letters.
    assert!(!matches_group('A', 'N'));
    assert!(!matches_group('A', 'x'));
    assert!(!matches_group('A', 'l'));
}

#[test]
fn blocks() {
    assert_eq!(block('A'), Some("Basic Latin"));
    assert_eq!(block('\u{00E9}'), Some("Latin-1 Supplement"));
    assert_eq!(block('\u{3042}'), Some("Hiragana"));
    assert_eq!(block('\u{1F600}'), Some("Emoticons"));

    // Loose name matching (UAX44-LM3-lite).
    assert_eq!(block_range("Basic Latin"), Some(('\u{0}', '\u{7F}')));
    assert_eq!(block_range("basiclatin"), Some(('\u{0}', '\u{7F}')));
    assert_eq!(block_range("BASIC-LATIN"), Some(('\u{0}', '\u{7F}')));
    assert_eq!(block_range("basic_latin"), Some(('\u{0}', '\u{7F}')));
    assert_eq!(
        block_range("Latin_1_Supplement"),
        Some(('\u{80}', '\u{FF}'))
    );
    assert_eq!(block_range("does not exist"), None);

    assert!(in_block('A', "BasicLatin"));
    assert!(in_block('\u{00E9}', "Latin-1 Supplement"));
    assert!(!in_block('A', "Hiragana"));
    assert!(!in_block('A', "No Such Block"));
}

#[test]
fn scripts() {
    assert_eq!(script('A'), Some(Script::Latin));
    assert_eq!(script('\u{0391}'), Some(Script::Greek));
    assert_eq!(script('\u{0410}'), Some(Script::Cyrillic));
    assert_eq!(script('\u{3042}'), Some(Script::Hiragana));
    assert_eq!(script('\u{4E00}'), Some(Script::Han));
    assert_eq!(script(' '), Some(Script::Common));
    assert_eq!(script('\u{0301}'), Some(Script::Inherited));
    assert_eq!(Script::Latin.name(), "Latin");
    // Unassigned high code points have no script.
    assert_eq!(script('\u{10FFFF}'), None);
}

#[test]
fn simple_case_mappings() {
    assert_eq!(simple_uppercase('a'), 'A');
    assert_eq!(simple_lowercase('A'), 'a');
    assert_eq!(simple_uppercase('\u{00E9}'), '\u{00C9}'); // é -> É
    assert_eq!(simple_lowercase('\u{00C9}'), '\u{00E9}');
    // Titlecase differs from uppercase for digraphs.
    assert_eq!(simple_titlecase('\u{01F3}'), '\u{01F2}'); // ǳ -> ǲ
    assert_eq!(simple_uppercase('\u{01F3}'), '\u{01F1}'); // ǳ -> Ǳ
    // Characters without a mapping map to themselves.
    assert_eq!(simple_uppercase('5'), '5');
    assert_eq!(simple_lowercase('中'), '中');
}

#[test]
fn case_folding() {
    // Simple folding is one-to-one.
    assert_eq!(simple_fold('A'), 'a');
    assert_eq!(simple_fold('\u{03A3}'), '\u{03C3}'); // Σ -> σ
    assert_eq!(simple_fold('5'), '5');

    // Full folding may expand.
    assert_eq!(fold_str("HELLO"), "hello");
    assert_eq!(fold_str("Stra\u{00DF}e"), "strasse"); // ß -> ss
    assert_eq!(fold_str("\u{FB01}le"), "file"); // ﬁ ligature
    assert_eq!(fold_str("MASSE"), fold_str("Ma\u{00DF}e")); // caseless-equal
    // Idempotence on already-folded text.
    assert_eq!(fold_str(&fold_str("Grüße")), fold_str("Grüße"));
}

#[test]
fn unicode_version_is_recorded() {
    assert!(!UNICODE_VERSION.is_empty());
    assert!(UNICODE_VERSION.split('.').count() == 3);
}
