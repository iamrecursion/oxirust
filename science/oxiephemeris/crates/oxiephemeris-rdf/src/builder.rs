//! Small helpers for assembling triples, shared by the ontology, SKOS,
//! chart, and provenance builders.
//!
//! # Non-finite guard
//!
//! `NaN` and `±Infinity` have **no** `xsd:double` lexical form that a
//! conformant parser will accept as a number (`"NaN"` is legal in
//! `xsd:double` but Rust's `Display` prints `NaN`/`inf`, and `inf` is not
//! the XSD form `INF`). Rather than emit a literal that would be silently
//! wrong, [`add_double`] **omits the triple entirely** for a non-finite
//! value. Chart data never contains one; this is a correctness backstop,
//! not an expected path.

use oxrdf::vocab::xsd;
use oxrdf::{Graph, Literal, NamedNode, NamedOrBlankNode, Term, Triple};

/// Inserts one triple.
pub(crate) fn add(
    graph: &mut Graph,
    subject: impl Into<NamedOrBlankNode>,
    predicate: impl Into<NamedNode>,
    object: impl Into<Term>,
) {
    graph.insert(&Triple::new(subject, predicate, object));
}

/// An `xsd:double` literal, or `None` for a non-finite value.
///
/// Rust's `Display` for `f64` never uses scientific notation, so the
/// lexical form is always a valid `xsd:double` decimal.
#[must_use]
pub(crate) fn double(value: f64) -> Option<Literal> {
    value
        .is_finite()
        .then(|| Literal::new_typed_literal(value.to_string(), xsd::DOUBLE))
}

/// Inserts an `xsd:double`-valued triple, skipping non-finite values.
pub(crate) fn add_double(
    graph: &mut Graph,
    subject: impl Into<NamedOrBlankNode>,
    predicate: impl Into<NamedNode>,
    value: f64,
) {
    if let Some(literal) = double(value) {
        add(graph, subject, predicate, literal);
    }
}

/// Decimal places kept when canonicalizing a *definitional* angle.
const DEFINITIONAL_DEG_PLACES: f64 = 1e9;

/// Canonicalizes a definitional angle in degrees.
///
/// A trine is exactly 120°, but the engine stores it as `2π/3` radians and
/// `(2π/3).to_degrees()` is `120.00000000000001`. Publishing that as the
/// *definition* of a trine would be wrong — it is float noise, not a
/// measurement. Definitional angles (aspect exact angles and default
/// orbs) are therefore snapped to `1e-9` degrees, which is `3.6` µas:
/// eleven orders of magnitude finer than this workspace's tightest
/// accuracy claim, so nothing real can hide under it.
///
/// Chart *data* is never rounded — those longitudes are computed
/// quantities and keep full `f64` precision.
#[must_use]
pub(crate) fn round_definitional_deg(value: f64) -> f64 {
    if value.is_finite() {
        (value * DEFINITIONAL_DEG_PLACES).round() / DEFINITIONAL_DEG_PLACES
    } else {
        value
    }
}

/// An `xsd:integer` literal.
#[must_use]
pub(crate) fn integer(value: i64) -> Literal {
    Literal::new_typed_literal(value.to_string(), xsd::INTEGER)
}

/// An `xsd:boolean` literal.
#[must_use]
pub(crate) fn boolean(value: bool) -> Literal {
    Literal::new_typed_literal(if value { "true" } else { "false" }, xsd::BOOLEAN)
}

/// An `xsd:dateTimeStamp` literal — `xsd:dateTime` restricted to values
/// that carry a timezone. Every epoch this crate emits is a UTC instant
/// ending in `Z`, so the stricter datatype is the honest one, and it
/// matches the range of `time:inXSDDateTimeStamp`.
#[must_use]
pub(crate) fn date_time_stamp(value: &str) -> Literal {
    Literal::new_typed_literal(value.to_owned(), xsd::DATE_TIME_STAMP)
}

/// A plain `xsd:string` literal.
#[must_use]
pub(crate) fn plain(value: &str) -> Literal {
    Literal::new_simple_literal(value.to_owned())
}

/// The only two language tags this crate emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lang {
    /// English.
    En,
    /// Japanese.
    Ja,
}

impl Lang {
    pub(crate) const fn tag(self) -> &'static str {
        match self {
            Self::En => "en",
            Self::Ja => "ja",
        }
    }
}

/// A language-tagged literal. `Lang` restricts the tag to two known-valid
/// values, which is what makes the `_unchecked` constructor sound here;
/// `tests::language_tags_are_valid` pins that.
#[must_use]
pub(crate) fn lang(value: &str, language: Lang) -> Literal {
    Literal::new_language_tagged_literal_unchecked(value.to_owned(), language.tag())
}

/// Adds `skos:prefLabel` in both languages when a Japanese label exists.
pub(crate) fn add_labels(
    graph: &mut Graph,
    subject: &NamedNode,
    predicate: impl Into<NamedNode> + Clone,
    english: &str,
    japanese: Option<&str>,
) {
    add(
        graph,
        subject.clone(),
        predicate.clone(),
        lang(english, Lang::En),
    );
    if let Some(ja) = japanese {
        add(graph, subject.clone(), predicate, lang(ja, Lang::Ja));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_tags_are_valid() {
        for l in [Lang::En, Lang::Ja] {
            assert!(
                Literal::new_language_tagged_literal("x", l.tag()).is_ok(),
                "invalid language tag {}",
                l.tag()
            );
        }
    }

    #[test]
    fn non_finite_doubles_are_refused() {
        assert!(double(f64::NAN).is_none());
        assert!(double(f64::INFINITY).is_none());
        assert!(double(f64::NEG_INFINITY).is_none());
        assert!(double(0.0).is_some());
    }

    #[test]
    fn add_double_skips_non_finite_triples() {
        let mut graph = Graph::new();
        let s = NamedNode::new_unchecked("https://example.org/s");
        let p = NamedNode::new_unchecked("https://example.org/p");
        add_double(&mut graph, s.clone(), p.clone(), f64::NAN);
        assert_eq!(graph.len(), 0, "NaN must not produce a triple");
        add_double(&mut graph, s, p, 1.5);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn double_lexical_form_is_plain_decimal() {
        let Some(lit) = double(2_440_587.500_465_196) else {
            panic!("finite value must produce a literal");
        };
        assert_eq!(lit.value(), "2440587.500465196");
        assert_eq!(lit.datatype(), xsd::DOUBLE);
        // Rust never emits scientific notation for f64 Display.
        assert!(!lit.value().contains(['e', 'E']));
    }

    #[test]
    // Exact equality is precisely what this test asserts: the snap must
    // land on the literal `120.0`, not merely near it.
    #[allow(clippy::float_cmp)]
    fn definitional_angles_lose_float_noise_but_keep_real_precision() {
        // (2π/3).to_degrees() is 120.00000000000001 — a trine is exactly 120°.
        let trine_rad = 2.0 * std::f64::consts::PI / 3.0;
        assert_eq!(round_definitional_deg(trine_rad.to_degrees()), 120.0);
        // A genuine 1e-8 deg (36 µas) difference survives the snap.
        assert_ne!(round_definitional_deg(120.000_000_01), 120.0);
        // Non-finite passes through untouched (add_double then drops it).
        assert!(round_definitional_deg(f64::NAN).is_nan());
    }

    #[test]
    fn integer_and_boolean_forms() {
        assert_eq!(integer(-4).value(), "-4");
        assert_eq!(integer(-4).datatype(), xsd::INTEGER);
        assert_eq!(boolean(true).value(), "true");
        assert_eq!(boolean(false).datatype(), xsd::BOOLEAN);
    }

    #[test]
    fn plain_and_datetime_literals() {
        assert_eq!(plain("x").value(), "x");
        let dt = date_time_stamp("1970-01-01T00:00:00Z");
        assert_eq!(dt.datatype(), xsd::DATE_TIME_STAMP);
    }
}
