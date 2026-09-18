//! [`XPathVersion`] — the language level a compilation targets.

use std::fmt;

/// Which revision of XPath an expression is compiled and evaluated against.
///
/// The version is not decoration: it changes both what parses and what the parsed expression
/// *means*.
///
/// # What changes at the grammar
///
/// [XPath 1.0](https://www.w3.org/TR/xpath-10/) has no sequence constructs at all. Selecting
/// [`XPathVersion::XPath10`] rejects, as a static error, every operator and expression form
/// introduced later: `to`, `idiv`, `intersect`, `except`, `eq`/`ne`/`lt`/`le`/`gt`/`ge`, `is`,
/// `<<`, `>>`, `||`, `!`, `=>`, `for`, `let`, `some`, `every`, `if`, `instance of`, `treat as`,
/// `cast`/`castable as`, kind tests other than `node()`, `text()`, `comment()` and
/// `processing-instruction()`, `Q{uri}local` names, `*:local` wildcards, inline functions,
/// named function references, dynamic function calls, maps, arrays and lookups.
///
/// # What changes at evaluation
///
/// | | XPath 1.0 | XPath 2.0 and later |
/// |---|---|---|
/// | Values | four types: node-set, boolean, number (an IEEE double), string | XDM sequences of typed items |
/// | `+ - * div mod` | both operands go through `number()`, so `"3" + 4` is `7` | operands are atomized, `xs:untypedAtomic` becomes `xs:double`, and `xs:string` is a type error |
/// | `= != < <= > >=` | node-set/node-set, node-set/number, node-set/string, node-set/boolean and scalar/scalar each have their own rule | one rule: atomize, then look for a pair that satisfies the corresponding value comparison |
/// | `number()` of a node-set | the first node in document order | every atomized item must be castable, and a sequence of more than one item is a type error |
/// | division by zero | `INF`, `-INF` or `NaN` — never an error | `FOAR0001` for `xs:decimal` and `xs:integer` |
/// | `last()`, `position()` | 1-based within the step's node-list | unchanged |
///
/// [`XPathVersion::XPath20`], [`XPathVersion::XPath30`] and [`XPathVersion::XPath31`] share the
/// XDM semantics and differ only in which grammar productions are accepted: 3.0 adds inline
/// functions, named function references, dynamic function calls, `let`, `||` and `=>`; 3.1 adds
/// maps, arrays, lookup (`?`) and the arrow operator's map/array targets.
///
/// # Examples
///
/// ```
/// use oxixml_xpath::{StaticContext, XPath, XPathVersion};
///
/// // `"3" + 4` is 7 in XPath 1.0 …
/// let one = XPath::compile("'3' + 4", XPathVersion::XPath10, StaticContext::new())?;
/// assert_eq!(one.evaluate_to_number_without_context()?, 7.0);
///
/// // … and a type error in XPath 2.0 and later.
/// let two = XPath::compile("'3' + 4", XPathVersion::XPath31, StaticContext::new())?;
/// assert!(two.evaluate_without_context().is_err());
///
/// // `to` does not even parse under XPath 1.0.
/// assert!(XPath::compile("1 to 3", XPathVersion::XPath10, StaticContext::new()).is_err());
/// assert!(XPath::compile("1 to 3", XPathVersion::XPath20, StaticContext::new()).is_ok());
/// # Ok::<(), oxixml_xpath::XPathError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum XPathVersion {
    /// [XPath 1.0](https://www.w3.org/TR/xpath-10/): node-sets, booleans, numbers and strings.
    XPath10,
    /// [XPath 2.0](https://www.w3.org/TR/xpath20/): sequences, typed values and the type
    /// operators.
    XPath20,
    /// [XPath 3.0](https://www.w3.org/TR/xpath-30/): function items, `let`, `||` and `=>`.
    XPath30,
    /// [XPath 3.1](https://www.w3.org/TR/xpath-31/): maps, arrays and lookup. The default.
    #[default]
    XPath31,
}

impl XPathVersion {
    /// Every version, oldest first.
    pub const ALL: &'static [Self] = &[Self::XPath10, Self::XPath20, Self::XPath30, Self::XPath31];

    /// The version number as it is written in a specification title — `"1.0"`, `"2.0"`,
    /// `"3.0"`, `"3.1"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::XPath10 => "1.0",
            Self::XPath20 => "2.0",
            Self::XPath30 => "3.0",
            Self::XPath31 => "3.1",
        }
    }

    /// Parses a version number, accepting the spellings [`XPathVersion::as_str`] produces.
    #[must_use]
    pub fn from_str_lossy(text: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|version| version.as_str() == text.trim())
    }

    /// Whether this is XPath 1.0, whose values are node-sets rather than sequences.
    #[must_use]
    pub const fn is_xpath_10(self) -> bool {
        matches!(self, Self::XPath10)
    }

    /// Whether this version has the XDM sequence semantics — everything but 1.0.
    #[must_use]
    pub const fn has_sequences(self) -> bool {
        !self.is_xpath_10()
    }

    /// Whether this version has function items: inline functions, named function references,
    /// dynamic calls and partial application. XPath 3.0 and later.
    #[must_use]
    pub const fn has_function_items(self) -> bool {
        matches!(self, Self::XPath30 | Self::XPath31)
    }

    /// Whether this version has maps, arrays and the lookup operator. XPath 3.1 only.
    #[must_use]
    pub const fn has_maps_and_arrays(self) -> bool {
        matches!(self, Self::XPath31)
    }

    /// Whether `self` is at least `other`.
    #[must_use]
    pub const fn at_least(self, other: Self) -> bool {
        self as u8 >= other as u8
    }
}

impl fmt::Display for XPathVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "XPath {}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_matches_the_release_order() {
        assert!(XPathVersion::XPath31.at_least(XPathVersion::XPath10));
        assert!(!XPathVersion::XPath10.at_least(XPathVersion::XPath20));
        assert!(XPathVersion::XPath30.at_least(XPathVersion::XPath30));
    }

    #[test]
    fn capability_predicates() {
        assert!(XPathVersion::XPath10.is_xpath_10());
        assert!(!XPathVersion::XPath10.has_sequences());
        assert!(XPathVersion::XPath20.has_sequences());
        assert!(!XPathVersion::XPath20.has_function_items());
        assert!(XPathVersion::XPath30.has_function_items());
        assert!(!XPathVersion::XPath30.has_maps_and_arrays());
        assert!(XPathVersion::XPath31.has_maps_and_arrays());
        assert_eq!(XPathVersion::default(), XPathVersion::XPath31);
    }

    #[test]
    fn spellings_round_trip() {
        for version in XPathVersion::ALL {
            assert_eq!(
                XPathVersion::from_str_lossy(version.as_str()),
                Some(*version)
            );
        }
        assert_eq!(XPathVersion::from_str_lossy("9.9"), None);
        assert_eq!(XPathVersion::XPath31.to_string(), "XPath 3.1");
    }
}
