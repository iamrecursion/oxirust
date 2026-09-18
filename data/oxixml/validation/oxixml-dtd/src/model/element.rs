//! Element type declarations and content models.
//!
//! Mirrors the `elementdecl` production of XML 1.0 §3.2 `[45]` and the
//! `contentspec` `[46]`, `Mixed` `[51]` and `children` `[47]` productions it
//! delegates to.

use core::fmt;
use std::collections::HashSet;

/// The repetition operator that may follow a content particle
/// (`cp ::= (Name | choice | seq) ('?' | '*' | '+')?`, XML 1.0 §3.2.1 `[48]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Occurrence {
    /// No operator: exactly once.
    #[default]
    One,
    /// `?` — zero or one.
    Optional,
    /// `*` — zero or more.
    ZeroOrMore,
    /// `+` — one or more.
    OneOrMore,
}

impl Occurrence {
    /// Returns `true` when zero repetitions are allowed (`?` and `*`).
    #[must_use]
    pub const fn allows_zero(self) -> bool {
        matches!(self, Self::Optional | Self::ZeroOrMore)
    }

    /// Returns `true` when more than one repetition is allowed (`*` and `+`).
    #[must_use]
    pub const fn allows_many(self) -> bool {
        matches!(self, Self::ZeroOrMore | Self::OneOrMore)
    }

    /// The character that spells this operator, or `None` for [`One`](Self::One).
    #[must_use]
    pub const fn as_char(self) -> Option<char> {
        match self {
            Self::One => None,
            Self::Optional => Some('?'),
            Self::ZeroOrMore => Some('*'),
            Self::OneOrMore => Some('+'),
        }
    }
}

impl fmt::Display for Occurrence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_char() {
            Some(c) => f.write_str(match c {
                '?' => "?",
                '*' => "*",
                _ => "+",
            }),
            None => Ok(()),
        }
    }
}

/// A content particle: a child element type, or a bracketed choice or sequence,
/// optionally repeated.
///
/// # Examples
///
/// ```
/// use oxixml_dtd::{ContentParticle, ContentSpec, DtdParser, Occurrence};
///
/// let dtd = DtdParser::new().parse_internal_subset("<!ELEMENT a (b, (c | d)*)>")?;
/// let ContentSpec::Children(particle) = dtd.element("a").expect("declared").content() else {
///     panic!("a sequence content model");
/// };
/// let ContentParticle::Sequence { particles, .. } = particle else {
///     panic!("a sequence")
/// };
/// assert_eq!(particles.len(), 2);
/// assert!(matches!(&particles[0], ContentParticle::Name { name, .. } if name == "b"));
/// assert!(matches!(
///     &particles[1],
///     ContentParticle::Choice { occurrence: Occurrence::ZeroOrMore, .. }
/// ));
/// # Ok::<(), oxixml_dtd::DtdError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentParticle {
    /// A child element type, e.g. `b+`.
    Name {
        /// The element type name.
        name: String,
        /// Its repetition operator.
        occurrence: Occurrence,
    },
    /// An alternation, e.g. `(a | b)?`.
    Choice {
        /// The alternatives, in declaration order. Always at least two.
        particles: Vec<ContentParticle>,
        /// The repetition operator applied to the whole group.
        occurrence: Occurrence,
    },
    /// A concatenation, e.g. `(a, b)*`.
    Sequence {
        /// The members, in declaration order. Always at least one.
        particles: Vec<ContentParticle>,
        /// The repetition operator applied to the whole group.
        occurrence: Occurrence,
    },
}

impl ContentParticle {
    /// The repetition operator of this particle.
    #[must_use]
    pub const fn occurrence(&self) -> Occurrence {
        match self {
            Self::Name { occurrence, .. }
            | Self::Choice { occurrence, .. }
            | Self::Sequence { occurrence, .. } => *occurrence,
        }
    }

    /// Appends every element type named anywhere inside this particle to `out`.
    pub fn collect_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::Name { name, .. } => out.push(name),
            Self::Choice { particles, .. } | Self::Sequence { particles, .. } => {
                for particle in particles {
                    particle.collect_names(out);
                }
            }
        }
    }

    /// The number of leaf ([`Name`](Self::Name)) particles anywhere inside this content
    /// particle — one per child-element occurrence, counting repeated names once per
    /// occurrence in the declaration (not once per distinct name).
    ///
    /// This is exactly the number of *positions* the Glushkov automaton of
    /// [`crate::automaton`] allocates for the model, which is what
    /// [`MAX_CONTENT_MODEL_POSITIONS`](crate::MAX_CONTENT_MODEL_POSITIONS) bounds — computing
    /// it costs only a single linear pass, letting the caller reject an oversized model before
    /// paying for automaton construction at all.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Name { .. } => 1,
            Self::Choice { particles, .. } | Self::Sequence { particles, .. } => {
                particles.iter().map(Self::leaf_count).sum()
            }
        }
    }
}

impl fmt::Display for ContentParticle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name { name, occurrence } => write!(f, "{name}{occurrence}"),
            Self::Choice {
                particles,
                occurrence,
            } => {
                write!(f, "(")?;
                for (index, particle) in particles.iter().enumerate() {
                    if index > 0 {
                        write!(f, "|")?;
                    }
                    write!(f, "{particle}")?;
                }
                write!(f, "){occurrence}")
            }
            Self::Sequence {
                particles,
                occurrence,
            } => {
                write!(f, "(")?;
                for (index, particle) in particles.iter().enumerate() {
                    if index > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{particle}")?;
                }
                write!(f, "){occurrence}")
            }
        }
    }
}

/// The `contentspec` of an element type declaration (XML 1.0 §3.2 `[46]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentSpec {
    /// `EMPTY` — no content at all.
    Empty,
    /// `ANY` — any declared element type, interleaved with character data.
    Any,
    /// `(#PCDATA)` or `(#PCDATA | a | b)*` — mixed content.
    Mixed(MixedContent),
    /// A `children` model: element content, no character data.
    Children(ContentParticle),
}

impl ContentSpec {
    /// Returns `true` for [`ContentSpec::Children`], the only model in which
    /// non-whitespace character data is forbidden outright.
    #[must_use]
    pub const fn is_element_content(&self) -> bool {
        matches!(self, Self::Children(_))
    }
}

impl fmt::Display for ContentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("EMPTY"),
            Self::Any => f.write_str("ANY"),
            Self::Mixed(mixed) => write!(f, "{mixed}"),
            Self::Children(particle) => write!(f, "{particle}"),
        }
    }
}

/// A mixed content model: `#PCDATA` plus an unordered, repeatable set of child
/// element types.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MixedContent {
    names: Vec<String>,
}

impl MixedContent {
    /// The bare `(#PCDATA)` model, which permits no child elements at all.
    #[must_use]
    pub fn pcdata_only() -> Self {
        Self { names: Vec::new() }
    }

    /// A `(#PCDATA | …)*` model over the given child element types.
    #[must_use]
    pub fn new(names: Vec<String>) -> Self {
        Self { names }
    }

    /// The permitted child element types, in declaration order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Returns `true` when no child elements are permitted (`(#PCDATA)`).
    #[must_use]
    pub fn is_pcdata_only(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns the first element type listed twice, if any — a violation of
    /// VC: No Duplicate Types (XML 1.0 §3.2.2).
    #[must_use]
    pub fn first_duplicate(&self) -> Option<&str> {
        // A `HashSet` built in one pass keeps this linear in the number of child types; the
        // nested scan it replaces was quadratic, which mattered for a mixed-content model
        // naming many thousands of alternatives.
        let mut seen: HashSet<&str> = HashSet::with_capacity(self.names.len());
        self.names
            .iter()
            .find(|name| !seen.insert(name.as_str()))
            .map(String::as_str)
    }
}

impl fmt::Display for MixedContent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.names.is_empty() {
            return f.write_str("(#PCDATA)");
        }
        f.write_str("(#PCDATA")?;
        for name in &self.names {
            write!(f, "|{name}")?;
        }
        f.write_str(")*")
    }
}

/// An element type declaration, `<!ELEMENT name contentspec>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementDecl {
    name: String,
    content: ContentSpec,
    external: bool,
}

impl ElementDecl {
    /// Builds a declaration. `external` records whether it came from the
    /// external subset or a parameter entity, which the standalone-document
    /// validity constraint of XML 1.0 §2.9 needs to know.
    #[must_use]
    pub fn new(name: impl Into<String>, content: ContentSpec, external: bool) -> Self {
        Self {
            name: name.into(),
            content,
            external,
        }
    }

    /// The declared element type.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The content model.
    #[must_use]
    pub fn content(&self) -> &ContentSpec {
        &self.content
    }

    /// Whether the declaration came from outside the internal subset.
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.external
    }
}

impl fmt::Display for ElementDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<!ELEMENT {} {}>", self.name, self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn particles_round_trip_through_display() {
        let particle = ContentParticle::Sequence {
            particles: vec![
                ContentParticle::Name {
                    name: "a".to_owned(),
                    occurrence: Occurrence::OneOrMore,
                },
                ContentParticle::Choice {
                    particles: vec![
                        ContentParticle::Name {
                            name: "b".to_owned(),
                            occurrence: Occurrence::One,
                        },
                        ContentParticle::Name {
                            name: "c".to_owned(),
                            occurrence: Occurrence::One,
                        },
                    ],
                    occurrence: Occurrence::Optional,
                },
            ],
            occurrence: Occurrence::One,
        };
        assert_eq!(particle.to_string(), "(a+,(b|c)?)");
    }

    #[test]
    fn mixed_content_detects_duplicate_types() {
        let mixed = MixedContent::new(vec!["a".to_owned(), "b".to_owned(), "a".to_owned()]);
        assert_eq!(mixed.first_duplicate(), Some("a"));
        assert_eq!(mixed.to_string(), "(#PCDATA|a|b|a)*");
        assert!(MixedContent::pcdata_only().is_pcdata_only());
    }

    #[test]
    fn a_huge_mixed_content_model_with_no_duplicates_checks_quickly_not_quadratically() {
        // `first_duplicate` used to scan every prior name for each name in the list — quadratic
        // in the number of alternatives, the same defect class as the attribute-declaration
        // checks in `model::attribute`. A `HashSet` built in one pass keeps it linear; 200 000
        // alternatives (`(#PCDATA|a0|a1|...|a199999)*`, far beyond anything a real DTD declares)
        // would take on the order of half a minute the old way and a small fraction of a second
        // this way, so a generous bound cleanly tells the two apart without being sensitive to
        // machine speed.
        let names: Vec<String> = (0..200_000).map(|i| format!("a{i}")).collect();
        let mixed = MixedContent::new(names);
        let started = std::time::Instant::now();
        let duplicate = mixed.first_duplicate();
        let elapsed = started.elapsed();
        assert_eq!(duplicate, None);
        assert!(
            elapsed < std::time::Duration::from_secs(15),
            "first_duplicate took {elapsed:?} for 200 000 distinct names"
        );
    }

    #[test]
    fn collect_names_walks_the_whole_particle() {
        let particle = ContentParticle::Choice {
            particles: vec![
                ContentParticle::Name {
                    name: "a".to_owned(),
                    occurrence: Occurrence::One,
                },
                ContentParticle::Sequence {
                    particles: vec![ContentParticle::Name {
                        name: "b".to_owned(),
                        occurrence: Occurrence::One,
                    }],
                    occurrence: Occurrence::One,
                },
            ],
            occurrence: Occurrence::One,
        };
        let mut names = Vec::new();
        particle.collect_names(&mut names);
        assert_eq!(names, ["a", "b"]);
    }
}
