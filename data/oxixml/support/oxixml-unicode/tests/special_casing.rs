//! `SpecialCasing.txt` full case mapping conformance tests.
//!
//! The embedded fixture drives every unconditional row of the crate's
//! Unicode version; the remaining tests target the context- and
//! language-sensitive rules individually, since each depends on different
//! surrounding characters.

use oxixml_unicode::case::{
    CaseLocale, full_lowercase, full_lowercase_locale, full_titlecase, full_uppercase,
    full_uppercase_locale,
};

// Embedded fixture: `SPECIAL_CASING_UNCONDITIONAL: &[(&str, &str, &str, &str)]`.
include!("ucd/special_casing_cases.rs");

#[test]
fn unconditional_rows_match_special_casing_txt() {
    for &(code, lower, title, upper) in SPECIAL_CASING_UNCONDITIONAL {
        assert_eq!(full_lowercase(code), lower, "lower({code:?})");
        assert_eq!(full_titlecase(code), title, "title({code:?})");
        assert_eq!(full_uppercase(code), upper, "upper({code:?})");
    }
}

// ---------------------------------------------------------------------------
// Final_Sigma (Greek capital sigma lowercasing)
// ---------------------------------------------------------------------------

#[test]
fn final_sigma_at_word_end() {
    // Capital sigma preceded by a cased letter and followed by nothing (or a
    // non-cased character) lowercases to the final form ς, not σ.
    assert_eq!(full_lowercase("\u{391}\u{3A3}"), "\u{3B1}\u{3C2}"); // ΑΣ -> ας
    assert_eq!(
        full_lowercase("\u{39F}\u{394}\u{3A5}\u{3A3}\u{3A3}\u{395}\u{3A5}\u{3A3}"),
        "\u{3BF}\u{3B4}\u{3C5}\u{3C3}\u{3C3}\u{3B5}\u{3C5}\u{3C2}"
    );
    // A trailing case-ignorable character (here, a period) does not count as
    // "followed by a cased letter", so the sigma is still final.
    assert_eq!(full_lowercase("\u{391}\u{3A3}."), "\u{3B1}\u{3C2}.");
}

#[test]
fn final_sigma_interior_stays_regular_sigma() {
    // Sigma at the start of a word has no preceding cased letter, so
    // Final_Sigma does not apply.
    assert_eq!(full_lowercase("\u{3A3}\u{391}"), "\u{3C3}\u{3B1}"); // ΣΑ -> σα
    // Sigma followed (after skipping case-ignorable characters) by another
    // cased letter is not final either.
    assert_eq!(
        full_lowercase("\u{391}\u{3A3}.\u{391}"),
        "\u{3B1}\u{3C3}.\u{3B1}"
    );
}

// ---------------------------------------------------------------------------
// German ß and the ligature expansions (unconditional multi-char mappings)
// ---------------------------------------------------------------------------

#[test]
fn sharp_s_uppercases_to_ss() {
    assert_eq!(full_uppercase("stra\u{DF}e"), "STRASSE");
    // ß itself is unaffected by lowercasing (it is already lowercase), but
    // its title mapping is also "Ss" (SpecialCasing.txt notes this case
    // should never arise in practice: it equals titlecase(uppercase(ß))).
    assert_eq!(full_lowercase("\u{DF}"), "\u{DF}");
    assert_eq!(full_titlecase("\u{DF}"), "Ss");
}

#[test]
fn ligature_expansions_change_length() {
    // ﬀ, ﬁ, ﬂ, ﬃ, ﬄ, ﬅ, ﬆ each expand to two or three characters when cased.
    let before = "\u{FB01}le"; // "ﬁle"
    let after = full_uppercase(before);
    assert_eq!(after, "FILE");
    assert!(after.chars().count() > before.chars().count());
}

#[test]
fn idempotence_of_multi_char_expansions() {
    // Re-applying the same full case operation to an already-cased result
    // is a no-op.
    for input in ["stra\u{DF}e", "\u{FB01}le", "STRASSE", "FILE"] {
        let upper = full_uppercase(input);
        assert_eq!(full_uppercase(&upper), upper, "idempotent upper({input:?})");
        let lower = full_lowercase(input);
        assert_eq!(full_lowercase(&lower), lower, "idempotent lower({input:?})");
    }
}

// ---------------------------------------------------------------------------
// Turkish / Azeri dotted-and-dotless I
// ---------------------------------------------------------------------------

#[test]
fn turkish_i_round_trips() {
    // i <-> İ (dotted) and I <-> ı (dotless) are the two Turkish case pairs.
    assert_eq!(full_uppercase_locale("i", CaseLocale::Turkish), "\u{130}");
    assert_eq!(full_lowercase_locale("\u{130}", CaseLocale::Turkish), "i");
    assert_eq!(full_lowercase_locale("I", CaseLocale::Turkish), "\u{131}");
    assert_eq!(full_uppercase_locale("\u{131}", CaseLocale::Turkish), "I");

    // Root behavior differs: the plain ASCII pair I/i, and İ decomposes.
    assert_eq!(full_uppercase("i"), "I");
    assert_eq!(full_lowercase("\u{130}"), "i\u{307}");
    assert_eq!(full_lowercase("I"), "i");
}

#[test]
fn azeri_matches_turkish() {
    assert_eq!(full_uppercase_locale("i", CaseLocale::Azeri), "\u{130}");
    assert_eq!(full_lowercase_locale("\u{130}", CaseLocale::Azeri), "i");
    assert_eq!(full_lowercase_locale("I", CaseLocale::Azeri), "\u{131}");
}

#[test]
fn turkish_capital_i_before_dot_above_stays_undotted() {
    // A capital I directly followed by a combining dot above lowercases the
    // same way as its canonically equivalent precomposed İ: to plain i, with
    // the dot above deleted (Not_Before_Dot / After_I).
    assert_eq!(full_lowercase_locale("I\u{307}", CaseLocale::Turkish), "i");
    // Root behavior keeps both characters, unaffected by the Turkish rule.
    assert_eq!(full_lowercase("I\u{307}"), "i\u{307}");
}

// ---------------------------------------------------------------------------
// Lithuanian dot-above retention
// ---------------------------------------------------------------------------

#[test]
fn lithuanian_keeps_dot_above_before_further_accents() {
    // A capital I, J, or I-with-ogonek followed by an accent that has
    // combining class Above (230) gets an explicit combining dot above when
    // lowercased, so the dot is not lost under the accent.
    assert_eq!(
        full_lowercase_locale("I\u{300}", CaseLocale::Lithuanian),
        "i\u{307}\u{300}"
    );
    assert_eq!(
        full_lowercase_locale("J\u{301}", CaseLocale::Lithuanian),
        "j\u{307}\u{301}"
    );
    assert_eq!(
        full_lowercase_locale("\u{12E}\u{303}", CaseLocale::Lithuanian),
        "\u{12F}\u{307}\u{303}"
    );

    // Without a following accent, no dot is introduced (More_Above fails).
    assert_eq!(full_lowercase_locale("I", CaseLocale::Lithuanian), "i");

    // The precomposed I-grave/acute/tilde always get the explicit dot in
    // Lithuanian, unconditionally (no More_Above test needed).
    assert_eq!(
        full_lowercase_locale("\u{CC}", CaseLocale::Lithuanian),
        "i\u{307}\u{300}"
    );
    assert_eq!(
        full_lowercase_locale("\u{CD}", CaseLocale::Lithuanian),
        "i\u{307}\u{301}"
    );
    assert_eq!(
        full_lowercase_locale("\u{128}", CaseLocale::Lithuanian),
        "i\u{307}\u{303}"
    );

    // Root behavior never introduces the extra dot above.
    assert_eq!(full_lowercase("I\u{300}"), "i\u{300}");
    assert_eq!(full_lowercase("\u{CC}"), "\u{EC}");
}

#[test]
fn lithuanian_drops_redundant_dot_above_when_uppercasing() {
    // A soft-dotted lowercase i followed by an explicit combining dot above
    // uppercases to plain I: the dot is redundant once the letter is
    // uppercase, so After_Soft_Dotted deletes it.
    assert_eq!(
        full_uppercase_locale("i\u{307}", CaseLocale::Lithuanian),
        "I"
    );
    // Root behavior keeps the combining dot above (no special deletion).
    assert_eq!(full_uppercase("i\u{307}"), "I\u{307}");
}

// ---------------------------------------------------------------------------
// Regression: no panics on empty input or lone combining marks
// ---------------------------------------------------------------------------

#[test]
fn empty_and_context_free_input_do_not_panic() {
    assert_eq!(full_lowercase(""), "");
    assert_eq!(full_uppercase(""), "");
    assert_eq!(full_titlecase(""), "");
    assert_eq!(
        full_lowercase_locale("\u{307}", CaseLocale::Turkish),
        "\u{307}"
    );
    assert_eq!(
        full_uppercase_locale("\u{307}", CaseLocale::Lithuanian),
        "\u{307}"
    );
}
