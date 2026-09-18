use super::*;

fn collation(parameters: &str) -> UcaCollation {
    UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?{parameters}"))
        .expect("a valid UCA URI")
        .expect("a UCA collation")
}

#[test]
fn a_non_uca_uri_is_not_recognised() -> FnResult<()> {
    assert!(UcaCollation::parse("urn:example:other")?.is_none());
    Ok(())
}

/// The registry every collation-sensitive function goes through builds a UCA collation on demand,
/// and leaves the collations that answer to a URI of their own alone.
#[test]
fn the_registry_resolves_the_uca_family_without_disturbing_the_built_ins() -> FnResult<()> {
    use crate::collation::{CASEBLIND_COLLATION, CODEPOINT_COLLATION, Collations};

    let collations = Collations::new();
    let uca = collations.resolve(&format!("{UCA_COLLATION_BASE}?lang=de;strength=primary"))?;
    assert_eq!(uca.compare("Strasse", "Straße"), Ordering::Equal);
    assert_eq!(
        collations.resolve(UCA_COLLATION_BASE)?.uri(),
        UCA_COLLATION_BASE,
        "the bare URI is a UCA collation with every parameter at its default"
    );

    let caseblind = collations.resolve(CASEBLIND_COLLATION)?;
    assert_eq!(caseblind.uri(), CASEBLIND_COLLATION);
    assert!(caseblind.equal("Red", "rED"));
    assert_eq!(caseblind.compare("green", "ORanGE"), Ordering::Less);
    assert_eq!(
        collations
            .resolve(CODEPOINT_COLLATION)?
            .compare("ABC", "abc"),
        Ordering::Less,
        "the code point collation still compares code points"
    );
    Ok(())
}

/// What `collations-1006` of the XSLT 3.0 suite asks of
/// `UCA?strength=secondary;fallback=no`: it resolves, it is case-blind, and it sorts
/// `('Ab', 'aBcD', 'aBC')` as `Ab aBC aBcD`.
#[test]
fn a_secondary_strength_collation_without_fallback_is_case_blind() -> FnResult<()> {
    use crate::collation::{Collations, collation_starts_with};

    let uri = format!("{UCA_COLLATION_BASE}?strength=secondary;fallback=no");
    let uca = Collations::new().resolve(&uri)?;
    assert!(uca.equal("abc", "ABC"));
    assert!(collation_starts_with(uca.as_ref(), "abc", "AB")?);
    assert_eq!(uca.compare("abc", "ABc"), Ordering::Equal);

    let mut words = ["Ab", "aBcD", "aBC"];
    words.sort_by(|left, right| uca.compare(left, right));
    assert_eq!(words, ["Ab", "aBC", "aBcD"]);
    Ok(())
}

/// XSLT 3.0 §5.3.4: an undefined parameter, or an undefined value for a defined one, is
/// `FOCH0002` under `fallback=no` and *ignored* under the default `fallback=yes`.
#[test]
fn an_undefined_parameter_is_foch0002_only_without_fallback() {
    for parameters in ["nonsense=1", "strength=nope", "caseFirst=sideways"] {
        let tolerated = UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?{parameters}"))
            .expect("fallback=yes ignores what it cannot honour");
        assert!(tolerated.is_some());
        let error = UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?{parameters};fallback=no"))
            .expect_err("rejected");
        assert!(error.is(FnErrorCode::Foch0002), "for `{parameters}`");
    }
    // `fallback` may follow the parameter it governs, and still governs it.
    let error = UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?fallback=no;reorder=digit"))
        .expect_err("rejected");
    assert!(error.is(FnErrorCode::Foch0002));
    // The DUCET version and the reordering groups both need weights beyond the root table this
    // collation carries, so `fallback=no` rejects them too.
    for parameters in ["version=9.0", "reorder=Z,digit"] {
        assert!(
            UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?{parameters}"))
                .expect("fallback=yes ignores them")
                .is_some()
        );
        assert!(
            UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?{parameters};fallback=no")).is_err(),
            "for `{parameters}`"
        );
    }
}

/// A language this collation has no tailoring for is the root order under `fallback=yes` and
/// `FOCH0002` under `fallback=no`; one it *does* have is neither.
#[test]
fn an_untailored_language_is_foch0002_only_without_fallback() {
    for tag in ["ja", "th", "zh"] {
        let tolerated = collation(&format!("lang={tag}"));
        assert!(tolerated.is_fallback(), "`{tag}` has no tailoring here");
        let error = UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?lang={tag};fallback=no"))
            .expect_err("rejected");
        assert!(error.is(FnErrorCode::Foch0002), "for `{tag}`");
    }
    for tag in ["en", "en-GB", "de", "de-AT", "sv", "und"] {
        assert!(
            !collation(&format!("lang={tag}")).is_fallback(),
            "`{tag}` is supported"
        );
        assert!(
            UcaCollation::parse(&format!("{UCA_COLLATION_BASE}?lang={tag};fallback=no")).is_ok(),
            "`{tag}` is supported even without fallback"
        );
    }
    assert!(
        !collation("strength=secondary").is_fallback(),
        "a URI with no `lang` at all asks for no tailoring"
    );
}

// --- the level ladder ---------------------------------------------------------------------------

#[test]
fn primary_strength_ignores_case_and_accents() {
    let uca = collation("lang=en;strength=primary");
    assert_eq!(uca.compare("database", "DATABASE"), Ordering::Equal);
    assert_eq!(uca.compare("database", "DÃTABASE"), Ordering::Equal);
    assert_eq!(uca.compare("database", "Databases"), Ordering::Less);
    assert_eq!(uca.compare("databases", "Database"), Ordering::Greater);
}

#[test]
fn secondary_strength_sees_accents_but_not_case() {
    let uca = collation("lang=en;strength=secondary");
    assert_eq!(uca.compare("database", "DATABASE"), Ordering::Equal);
    assert_eq!(uca.compare("database", "DÃTABASE"), Ordering::Less);
    assert_eq!(uca.compare("database", "Databases"), Ordering::Less);
}

#[test]
fn tertiary_strength_sees_case_in_the_order_case_first_asks_for() {
    let lower = collation("lang=en;strength=tertiary;caseFirst=lower");
    let upper = collation("lang=en;strength=tertiary;caseFirst=upper");
    assert_eq!(lower.compare("database", "Database"), Ordering::Less);
    assert_eq!(upper.compare("database", "Database"), Ordering::Greater);
    assert_eq!(lower.compare("databases", "Database"), Ordering::Greater);
    assert_eq!(lower.compare("dâtabases", "Database"), Ordering::Greater);
    assert_ne!(
        collation("lang=en;strength=tertiary").compare("database", "DATABASE"),
        Ordering::Equal
    );
}

/// The ordering `sort-018` of the XSLT 3.0 suite pins for the two `case-order` settings of
/// `xsl:sort`, over the fourteen letters of its source document.
///
/// `case-order="lower-first"`/`"upper-first"` is `caseFirst=lower`/`upper` on the collation the
/// sort runs under: at tertiary strength the collation itself decides the order of `b` and `B`,
/// and `case-order` is not a tie-break the sort could apply afterwards, because the collation
/// never calls the two equal.
#[test]
fn case_first_orders_the_sort_018_letters() {
    let input = [
        "X", "c", "t", "U", "V", "b", "K", "x", "C", "r", "u", "V", "B", "a",
    ];
    let sorted = |parameters: &str| {
        let uca = collation(parameters);
        let mut letters = input;
        // `xsl:sort` is stable, which is what decides the two `V`s.
        letters.sort_by(|left, right| uca.compare(left, right));
        letters.join(",")
    };
    assert_eq!(
        sorted("lang=en-GB;strength=tertiary;caseFirst=lower"),
        "a,b,B,c,C,K,r,t,u,U,V,V,x,X"
    );
    assert_eq!(
        sorted("lang=en-GB;strength=tertiary;caseFirst=upper"),
        "a,B,b,C,c,K,r,t,U,u,V,V,X,x"
    );
}

#[test]
fn the_case_level_can_be_asked_for_on_its_own() {
    let uca = collation("lang=en;strength=primary;caseLevel=yes");
    assert_eq!(uca.compare("DATABASE", "DÃTABASE"), Ordering::Equal);
    let lower = collation("lang=en;strength=primary;caseLevel=yes;caseFirst=lower");
    assert_eq!(lower.compare("DATABASE", "DÃTAbase"), Ordering::Greater);
}

#[test]
fn backwards_compares_accents_from_the_right() {
    let forwards = collation("lang=en;strength=tertiary");
    let backwards = collation("lang=en;strength=tertiary;backwards=yes");
    assert_eq!(forwards.compare("DATABÃSE", "DÃTABASE"), Ordering::Less);
    assert_eq!(backwards.compare("DATABÃSE", "DÃTABASE"), Ordering::Greater);
}

/// The root table places whitespace, then punctuation, then the symbols, then the digits, then
/// the letters — an order code point comparison does not have.
#[test]
fn the_root_order_puts_punctuation_below_the_digits_and_the_letters() {
    let uca = collation("lang=en;strength=tertiary");
    assert_eq!(uca.compare("_a", "0a"), Ordering::Less);
    assert_eq!(uca.compare("~a", "0a"), Ordering::Less);
    assert_eq!(uca.compare("0a", "aa"), Ordering::Less);
    assert_eq!(uca.compare("za", "\u{3042}a"), Ordering::Less);
}

/// Beyond the root table's own repertoire the implicit weights of UTS #10 §10 take over, which
/// order a script's characters by code point.
#[test]
fn implicit_weights_order_the_han_ideographs_by_code_point() {
    let uca = collation("lang=en;strength=tertiary");
    assert_eq!(uca.compare("\u{4e00}", "\u{4e09}"), Ordering::Less);
    assert_eq!(uca.compare("\u{4e09}", "\u{4e8c}"), Ordering::Less);
    assert_eq!(uca.compare("\u{03b1}", "\u{0430}"), Ordering::Less);
    assert_eq!(uca.compare("\u{4e00}", "\u{03b1}"), Ordering::Greater);
}

// --- expansions ---------------------------------------------------------------------------------

/// The three `fn:compare` cases of the conformance suite's *F&O spec examples* set, verbatim.
///
/// `fo-test-fn-compare-002` and `-003` are `fn:compare('Strasse', 'Straße')` under
/// `lang=de;strength=primary`, and `-004` is `fn:compare('Strassen', 'Straße')` under a bare
/// `lang=de`, which is tertiary strength.
#[test]
fn the_sharp_s_weighs_as_two_letters() {
    let primary = collation("lang=de;strength=primary");
    assert_eq!(primary.compare("Strasse", "Straße"), Ordering::Equal);
    assert_eq!(
        collation("lang=de").compare("Strassen", "Straße"),
        Ordering::Greater
    );

    // Secondary strength is still an equality; tertiary is where the two spellings part.
    let secondary = collation("lang=de;strength=secondary");
    assert_eq!(secondary.compare("Strasse", "Straße"), Ordering::Equal);
    let tertiary = collation("lang=de;strength=tertiary");
    assert_eq!(tertiary.compare("strasse", "straße"), Ordering::Less);

    // The expansion is not a German rule: it is the root table's, so every language has it.
    assert_eq!(
        collation("strength=primary").compare("Strasse", "Straße"),
        Ordering::Equal
    );
    // And it is a real expansion, not a fold of `ß` to a single `s`.
    assert_eq!(primary.compare("Strase", "Straße"), Ordering::Less);
}

#[test]
fn the_capital_sharp_s_and_the_ligatures_weigh_as_their_letters() {
    let primary = collation("lang=en;strength=primary");
    assert_eq!(primary.compare("STRASSE", "STRA\u{1e9e}E"), Ordering::Equal);
    assert_eq!(primary.compare("office", "o\u{fb03}ce"), Ordering::Equal);
    assert_eq!(primary.compare("fjord", "\u{fb01}ord"), Ordering::Greater);
}

// --- tailorings ---------------------------------------------------------------------------------

/// German needs no tailoring of its own for the ordering DIN 5007-1 defines: `ä` sorts as `a`
/// with a secondary difference, which the canonical decomposition already gives it.
#[test]
fn german_reads_the_root_order() {
    let uca = collation("lang=de;strength=tertiary");
    assert_eq!(uca.compare("Bar", "Bär"), Ordering::Less);
    assert_eq!(uca.compare("Bär", "Bat"), Ordering::Less);
    assert_eq!(
        uca.compare("Bär", "Bar"),
        collation("lang=en;strength=tertiary").compare("Bär", "Bar")
    );
}

/// Swedish places `å`, `ä` and `ö` after `z`, where the root order puts them with `a` and `o`.
#[test]
fn swedish_places_three_letters_after_z() {
    let swedish = collation("lang=sv;strength=tertiary");
    let root = collation("lang=en;strength=tertiary");

    assert_eq!(swedish.compare("Zebra", "Åke"), Ordering::Less);
    assert_eq!(swedish.compare("Åke", "Ärt"), Ordering::Less);
    assert_eq!(swedish.compare("Ärt", "Öl"), Ordering::Less);
    assert_eq!(root.compare("Åke", "Zebra"), Ordering::Less);

    let mut words = ["Öl", "Zebra", "Åke", "Ärt", "Bok"];
    words.sort_by(|left, right| swedish.compare(left, right));
    assert_eq!(words, ["Bok", "Zebra", "Åke", "Ärt", "Öl"]);
}

/// SS 613621 sorts `æ` with `ä`, `ø` with `ö` and `ü` with `y`, each a secondary difference apart.
#[test]
fn swedish_equates_the_neighbouring_spellings() {
    let swedish = collation("lang=sv;strength=primary");
    assert_eq!(swedish.compare("ær", "är"), Ordering::Equal);
    assert_eq!(swedish.compare("øl", "öl"), Ordering::Equal);
    assert_eq!(swedish.compare("ük", "yk"), Ordering::Equal);

    let secondary = collation("lang=sv;strength=secondary");
    assert_eq!(secondary.compare("är", "ær"), Ordering::Less);
    assert_eq!(secondary.compare("öl", "øl"), Ordering::Less);
}

// --- variable weighting -------------------------------------------------------------------------

#[test]
fn the_alternate_parameter_decides_what_a_space_costs() {
    let blanked_t = collation("lang=en;alternate=blanked;strength=tertiary");
    let blanked_q = collation("lang=en;alternate=blanked;strength=quaternary");
    let blanked_i = collation("lang=en;alternate=blanked;strength=identical");
    assert_eq!(blanked_t.compare("database", "data base"), Ordering::Equal);
    assert_eq!(blanked_q.compare("database", "data base"), Ordering::Equal);
    assert_ne!(blanked_i.compare("database", "data base"), Ordering::Equal);

    let shifted_t = collation("lang=en;alternate=shifted;strength=tertiary");
    let shifted_q = collation("lang=en;alternate=shifted;strength=quaternary");
    assert_eq!(shifted_t.compare("database", "data base"), Ordering::Equal);
    assert_ne!(shifted_q.compare("database", "data base"), Ordering::Equal);
}

/// UTS #10 §3.4: an ignorable element that follows a variable one is set aside with it, so a
/// blanked hyphen takes its own accent with it.
#[test]
fn a_blanked_character_takes_its_accents_with_it() {
    let blanked = collation("lang=en;alternate=blanked;strength=secondary");
    assert_eq!(
        blanked.compare("data\u{2010}\u{0301}base", "database"),
        Ordering::Equal
    );
    let plain = collation("lang=en;alternate=non-ignorable;strength=secondary");
    assert_ne!(
        plain.compare("data\u{2010}base", "database"),
        Ordering::Equal
    );
}

/// UTS #10 §3.2.2: under `alternate=shifted` a *non*-variable element weighs the maximum at
/// the fourth level, so a variable one placed against it loses.
#[test]
fn shifted_quaternary_weights_put_a_variable_before_an_ordinary_character() {
    let quaternary = collation("lang=en;strength=quaternary;alternate=shifted");
    assert_eq!(quaternary.compare("a-bc", "abc"), Ordering::Less);
    assert_eq!(quaternary.compare("abc", "a-bc"), Ordering::Greater);
    assert_eq!(
        collation("lang=en;strength=tertiary;alternate=shifted").compare("a-bc", "abc"),
        Ordering::Equal,
        "below the fourth level the hyphen is invisible"
    );
    let space = collation("lang=en;strength=quaternary;alternate=shifted;maxVariable=space");
    assert_eq!(space.compare("database", "data base"), Ordering::Greater);
}

/// The reordering groups nest `space` ⊂ `punct` ⊂ `symbol` ⊂ `currency`.
#[test]
fn max_variable_names_the_widest_group_that_is_variable() {
    let symbol = collation("lang=en;alternate=shifted;maxVariable=symbol");
    let currency = collation("lang=en;alternate=shifted;maxVariable=currency");
    assert_eq!(
        symbol.compare("data=base", "data$base"),
        Ordering::Greater,
        "`=` is variable and ignored, `$` is not, and `$` sorts before `b`"
    );
    assert_eq!(
        currency.compare("data=base", "data$base"),
        Ordering::Equal,
        "naming currency makes both variable"
    );
}

/// The ordering `sort-079` of the XSLT 3.0 suite pins for the three `alternate` settings, over
/// the same six strings and at all three strengths it uses.
#[test]
fn the_alternate_settings_order_the_deluge_words_as_the_suite_expects() {
    let input = ["deluge", "Deluge", "delug", "delu-ge", "de-luge", "deluge-"];
    let sorted = |parameters: &str| {
        let uca = collation(parameters);
        let mut words = input;
        // `xsl:perform-sort` is stable, which is what decides the words this collation calls
        // equal.
        words.sort_by(|left, right| uca.compare(left, right));
        words.join(" ")
    };
    for strength in ["primary", "secondary"] {
        assert_eq!(
            sorted(&format!(
                "lang=en;strength={strength};alternate=non-ignorable"
            )),
            "de-luge delu-ge delug deluge Deluge deluge-"
        );
        for alternate in ["shifted", "blanked"] {
            assert_eq!(
                sorted(&format!(
                    "lang=en;strength={strength};alternate={alternate}"
                )),
                "delug deluge Deluge delu-ge de-luge deluge-",
                "at {strength} strength with alternate={alternate}"
            );
        }
    }
    assert_eq!(
        sorted("lang=en;strength=tertiary;alternate=non-ignorable"),
        "de-luge delu-ge delug deluge Deluge deluge-"
    );
    for alternate in ["shifted", "blanked"] {
        assert_eq!(
            sorted(&format!("lang=en;strength=tertiary;alternate={alternate}")),
            "delug deluge delu-ge de-luge deluge- Deluge",
            "at tertiary strength with alternate={alternate}"
        );
    }
}

// --- numbers ------------------------------------------------------------------------------------

#[test]
fn numeric_compares_digit_runs_by_value() {
    let uca = collation("lang=en;numeric=yes");
    assert_eq!(uca.compare("Chap2", "Chap10"), Ordering::Less);
    assert_eq!(uca.compare("Chap2-b", "Chap10-b"), Ordering::Less);
}

/// A digit run is weighed by its length and then its digits, so a number wider than any machine
/// integer still orders by magnitude.
#[test]
fn numeric_orders_runs_no_integer_type_could_hold() {
    let uca = collation("lang=en;numeric=yes");
    assert_eq!(uca.compare("4294967296", "4294967297"), Ordering::Less);
    assert_eq!(
        uca.compare(
            "123456789012345678901234567890",
            "123456789012345678901234567891"
        ),
        Ordering::Less
    );
    assert_eq!(
        uca.compare("007", "7"),
        Ordering::Equal,
        "leading zeros do not count"
    );
    assert_eq!(uca.compare("0", "00"), Ordering::Equal);
    assert_eq!(uca.compare("9", "10"), Ordering::Less);
    // A run is read digit by digit, whatever script the digits are written in.
    assert_eq!(uca.compare("\u{0663}", "3"), Ordering::Equal);
    assert_eq!(uca.compare("\u{0661}\u{0660}", "9"), Ordering::Greater);
}

/// Without `numeric=yes` the digits are ordinary characters, and the root table still puts them
/// below the letters and above the punctuation.
#[test]
fn the_digits_of_every_script_weigh_by_value() {
    let uca = collation("lang=en;strength=primary");
    assert_eq!(uca.compare("\u{0663}", "3"), Ordering::Equal);
    assert_eq!(uca.compare("3", "4"), Ordering::Less);
    assert_eq!(uca.compare("9", "10"), Ordering::Greater);
}

// --- collation units and sort keys ----------------------------------------------------------------

#[test]
fn substring_matching_works_at_primary_strength() {
    let uca = collation("lang=en;strength=primary");
    assert_eq!(uca.collation_units("Ábc"), Some(vec!['a', 'b', 'c']));
}

/// A collation unit has to carry exactly the differences the collation's strength can see,
/// or `fn:contains` finds matches `fn:compare` would call unequal.
#[test]
fn collation_units_carry_the_differences_the_strength_sees() {
    let primary = collation("lang=en;strength=primary");
    let secondary = collation("lang=en;strength=secondary");
    let tertiary = collation("lang=en;strength=tertiary");
    assert_eq!(
        primary.collation_units("dÂta"),
        Some("data".chars().collect())
    );
    assert_eq!(
        secondary.collation_units("dÂta"),
        Some("dâta".chars().collect()),
        "an accent is a secondary difference, so it survives; case does not"
    );
    assert_eq!(
        tertiary.collation_units("dÂta"),
        Some("dÂta".chars().collect()),
        "case is a tertiary difference"
    );
    assert_eq!(
        collation("lang=en;strength=primary;caseLevel=yes").collation_units("dAta"),
        Some("dAta".chars().collect()),
        "caseLevel=yes makes case visible even at primary strength"
    );
}

/// F&O 3.1 §5.3.8: the ordering of collation keys reflects the ordering of the strings.
#[test]
fn a_sort_key_orders_the_way_the_collation_does() {
    for parameters in [
        "lang=en;caseFirst=lower",
        "lang=en;caseFirst=upper",
        "lang=en;strength=primary",
        "lang=en;strength=secondary",
        "lang=en;strength=identical",
        "lang=en;strength=primary;caseLevel=yes",
        "lang=en;strength=quaternary;alternate=shifted",
        "lang=en;alternate=blanked",
        "lang=en;numeric=yes",
        "lang=de;strength=primary",
        "lang=sv;strength=tertiary",
    ] {
        let uca = collation(parameters);
        for (left, right) in [
            ("abc", "ABC"),
            ("dâta", "data"),
            ("a", "b"),
            ("", "a"),
            ("ab", "abc"),
            ("Strasse", "Straße"),
            ("a-bc", "abc"),
            ("Chap2", "Chap10"),
            ("Åke", "Zebra"),
            ("_a", "0a"),
        ] {
            let keys = uca
                .sort_key(left)
                .expect("a key")
                .cmp(&uca.sort_key(right).expect("a key"));
            assert_eq!(
                keys,
                uca.compare(left, right),
                "`{left}` vs `{right}` under `{parameters}`"
            );
            assert_eq!(
                uca.sort_key(right)
                    .expect("a key")
                    .cmp(&uca.sort_key(left).expect("a key")),
                uca.compare(right, left),
                "`{right}` vs `{left}` under `{parameters}`"
            );
        }
    }
}

/// A shorter level must compare below the longer one it is a prefix of, which is only true while
/// no byte of a weight can be mistaken for the separator between two levels.
#[test]
fn no_weight_of_a_sort_key_can_be_read_as_a_level_separator() {
    // Every level this collation can have: secondary, case, tertiary, quaternary and identical.
    let uca = collation("lang=en;strength=identical;numeric=yes;caseLevel=yes;alternate=shifted");
    for value in ["", "a", "Straße", "data base", "\u{4e00}", "Chap10", "_"] {
        let key = uca.sort_key(value).expect("a key");
        let separators = key.iter().filter(|byte| **byte == LEVEL_SEPARATOR).count();
        assert_eq!(
            separators, 5,
            "`{value}` should carry one separator per level after the first"
        );
    }
}
