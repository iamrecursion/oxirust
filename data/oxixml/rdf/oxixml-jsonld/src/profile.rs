//! JSON-LD [profiles](https://www.w3.org/TR/json-ld11/#iana-considerations) and
//! [processing modes](https://www.w3.org/TR/json-ld11/#dfn-processing-mode).

use std::array;
use std::ops::{BitOr, BitOrAssign};

/// A JSON-LD document-form profile.
///
/// This enumeration is non-exhaustive; new profiles may be added in the future.
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
#[non_exhaustive]
pub enum JsonLdProfile {
    /// [Expanded document form](https://www.w3.org/TR/json-ld11/#dfn-expanded-document-form).
    Expanded,
    /// [Compacted document form](https://www.w3.org/TR/json-ld11/#dfn-compacted-document-form).
    Compacted,
    /// [Context document](https://www.w3.org/TR/json-ld11/#dfn-context-document).
    Context,
    /// [Flattened document form](https://www.w3.org/TR/json-ld11/#dfn-flattened-document-form).
    Flattened,
    /// [Frame document](https://www.w3.org/TR/json-ld11-framing/#dfn-frame).
    Frame,
    /// [Framed document form](https://www.w3.org/TR/json-ld11/#dfn-framed-document-form).
    Framed,
    /// [Streaming document form](https://www.w3.org/TR/json-ld11-streaming/#dfn-streaming-document).
    Streaming,
}

impl JsonLdProfile {
    /// The profile's canonical IRI.
    ///
    /// ```
    /// use oxixml_jsonld::JsonLdProfile;
    ///
    /// assert_eq!(
    ///     JsonLdProfile::Expanded.iri(),
    ///     "http://www.w3.org/ns/json-ld#expanded"
    /// );
    /// ```
    #[must_use]
    pub const fn iri(self) -> &'static str {
        match self {
            Self::Expanded => "http://www.w3.org/ns/json-ld#expanded",
            Self::Compacted => "http://www.w3.org/ns/json-ld#compacted",
            Self::Context => "http://www.w3.org/ns/json-ld#context",
            Self::Flattened => "http://www.w3.org/ns/json-ld#flattened",
            Self::Frame => "http://www.w3.org/ns/json-ld#frame",
            Self::Framed => "http://www.w3.org/ns/json-ld#framed",
            Self::Streaming => "http://www.w3.org/ns/json-ld#streaming",
        }
    }

    /// Looks up a known profile from its IRI.
    ///
    /// ```
    /// use oxixml_jsonld::JsonLdProfile;
    ///
    /// assert_eq!(
    ///     JsonLdProfile::from_iri("http://www.w3.org/ns/json-ld#expanded"),
    ///     Some(JsonLdProfile::Expanded)
    /// );
    /// ```
    #[must_use]
    pub fn from_iri(iri: &str) -> Option<Self> {
        match iri {
            "http://www.w3.org/ns/json-ld#expanded" => Some(Self::Expanded),
            "http://www.w3.org/ns/json-ld#compacted" => Some(Self::Compacted),
            "http://www.w3.org/ns/json-ld#context" => Some(Self::Context),
            "http://www.w3.org/ns/json-ld#flattened" => Some(Self::Flattened),
            "http://www.w3.org/ns/json-ld#frame" => Some(Self::Frame),
            "http://www.w3.org/ns/json-ld#framed" => Some(Self::Framed),
            "http://www.w3.org/ns/json-ld#streaming" => Some(Self::Streaming),
            _ => None,
        }
    }

    const fn to_bit(self) -> u64 {
        match self {
            Self::Expanded => 1,
            Self::Compacted => 2,
            Self::Context => 4,
            Self::Flattened => 8,
            Self::Frame => 16,
            Self::Framed => 32,
            Self::Streaming => 64,
        }
    }
}

/// A set of JSON-LD [`JsonLdProfile`]s.
///
/// ```
/// use oxixml_jsonld::{JsonLdProfile, JsonLdProfileSet};
///
/// let mut profiles = JsonLdProfileSet::empty();
/// profiles |= JsonLdProfile::Expanded;
/// profiles |= JsonLdProfile::Streaming;
/// assert!(profiles.contains(JsonLdProfile::Streaming));
/// assert_eq!(
///     profiles.into_iter().collect::<Vec<_>>(),
///     vec![JsonLdProfile::Expanded, JsonLdProfile::Streaming]
/// );
/// ```
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash, Default)]
pub struct JsonLdProfileSet {
    value: u64,
}

impl JsonLdProfileSet {
    /// The empty set.
    #[must_use]
    pub const fn empty() -> Self {
        Self { value: 0 }
    }

    /// A set containing exactly `profile`.
    #[must_use]
    pub const fn from_profile(profile: JsonLdProfile) -> Self {
        Self {
            value: profile.to_bit(),
        }
    }

    /// Whether the set contains `profile`.
    #[must_use]
    pub const fn contains(self, profile: JsonLdProfile) -> bool {
        self.value & profile.to_bit() != 0
    }

    /// Whether the set is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.value == 0
    }
}

impl From<JsonLdProfile> for JsonLdProfileSet {
    fn from(profile: JsonLdProfile) -> Self {
        Self::from_profile(profile)
    }
}

impl IntoIterator for JsonLdProfileSet {
    type Item = JsonLdProfile;
    type IntoIter = JsonLdProfileBagIter;

    fn into_iter(self) -> JsonLdProfileBagIter {
        JsonLdProfileBagIter {
            set: self,
            possible_values: [
                JsonLdProfile::Expanded,
                JsonLdProfile::Compacted,
                JsonLdProfile::Context,
                JsonLdProfile::Flattened,
                JsonLdProfile::Frame,
                JsonLdProfile::Framed,
                JsonLdProfile::Streaming,
            ]
            .into_iter(),
        }
    }
}

/// Iterator over the [`JsonLdProfile`]s in a [`JsonLdProfileSet`].
pub struct JsonLdProfileBagIter {
    set: JsonLdProfileSet,
    possible_values: array::IntoIter<JsonLdProfile, 7>,
}

impl Iterator for JsonLdProfileBagIter {
    type Item = JsonLdProfile;

    fn next(&mut self) -> Option<JsonLdProfile> {
        loop {
            let candidate = self.possible_values.next()?;
            if self.set.contains(candidate) {
                return Some(candidate);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = usize::try_from(self.set.value.count_ones()).unwrap_or(usize::MAX);
        (size, Some(size))
    }
}

impl BitOr for JsonLdProfileSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            value: self.value | rhs.value,
        }
    }
}

impl BitOr<JsonLdProfile> for JsonLdProfileSet {
    type Output = JsonLdProfileSet;

    fn bitor(self, rhs: JsonLdProfile) -> JsonLdProfileSet {
        self | JsonLdProfileSet::from(rhs)
    }
}

impl BitOr<JsonLdProfileSet> for JsonLdProfile {
    type Output = JsonLdProfileSet;

    fn bitor(self, rhs: JsonLdProfileSet) -> JsonLdProfileSet {
        JsonLdProfileSet::from(self) | rhs
    }
}

impl BitOr for JsonLdProfile {
    type Output = JsonLdProfileSet;

    fn bitor(self, rhs: Self) -> JsonLdProfileSet {
        JsonLdProfileSet::from(self) | JsonLdProfileSet::from(rhs)
    }
}

impl BitOrAssign for JsonLdProfileSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.value |= rhs.value;
    }
}

impl BitOrAssign<JsonLdProfile> for JsonLdProfileSet {
    fn bitor_assign(&mut self, rhs: JsonLdProfile) {
        *self |= JsonLdProfileSet::from(rhs);
    }
}

/// The JSON-LD [processing mode](https://www.w3.org/TR/json-ld11/#dfn-processing-mode).
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash, Default)]
pub enum JsonLdProcessingMode {
    /// JSON-LD 1.0.
    JsonLd1_0,
    /// JSON-LD 1.1 (the default).
    #[default]
    JsonLd1_1,
}

impl JsonLdProcessingMode {
    /// The string identifier, e.g. `"json-ld-1.1"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::JsonLd1_0 => "json-ld-1.0",
            Self::JsonLd1_1 => "json-ld-1.1",
        }
    }

    /// Parses a processing mode from its string identifier.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "json-ld-1.0" => Some(Self::JsonLd1_0),
            "json-ld-1.1" => Some(Self::JsonLd1_1),
            _ => None,
        }
    }
}
