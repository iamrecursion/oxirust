//! [`RdfFormat`]: RDF serialization format identification, detection, and metadata.

use oxixml_jsonld::{JsonLdProfile, JsonLdProfileSet};
use std::fmt;

/// RDF serialization formats supported by [`RdfParser`](crate::RdfParser) and
/// [`RdfSerializer`](crate::RdfSerializer).
///
/// This mirrors [`oxrdfio::RdfFormat`](https://docs.rs/oxrdfio/latest/oxrdfio/enum.RdfFormat.html),
/// with one **OxiXML extension**: [`RdfFormat::TriX`]. TriX is not part of the `oxrdfio` format
/// set nor the W3C [Unique URIs for file formats](https://www.w3.org/ns/formats/) registry, but
/// it is a simple, long-standing XML syntax for RDF datasets, implemented by
/// [`oxixml-trix`](oxixml_trix).
///
/// This enumeration is non-exhaustive; new formats might be added in the future.
#[derive(Eq, PartialEq, Debug, Clone, Copy, Hash)]
#[non_exhaustive]
pub enum RdfFormat {
    /// [N3](https://w3c.github.io/N3/spec/).
    N3,
    /// [N-Quads](https://www.w3.org/TR/n-quads/).
    NQuads,
    /// [N-Triples](https://www.w3.org/TR/n-triples/).
    NTriples,
    /// [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/).
    RdfXml,
    /// [TriG](https://www.w3.org/TR/trig/).
    TriG,
    /// [Turtle](https://www.w3.org/TR/turtle/).
    Turtle,
    /// [TriX](https://en.wikipedia.org/wiki/TriX_(syntax)) (Triples in XML).
    ///
    /// An **OxiXML extension** beyond the `oxrdfio` format set: see the enum-level
    /// documentation.
    TriX,
    /// [JSON-LD](https://www.w3.org/TR/json-ld/).
    JsonLd {
        /// The document-form profile(s) in effect (e.g. [`JsonLdProfile::Streaming`]).
        profile: JsonLdProfileSet,
    },
}

impl RdfFormat {
    /// The format's canonical IRI according to the
    /// [Unique URIs for file formats registry](https://www.w3.org/ns/formats/), where one exists.
    ///
    /// TriX has no entry in that registry; its own XML namespace IRI is used instead (see
    /// [`TRIX_NAMESPACE`](oxixml_trix::TRIX_NAMESPACE)).
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(
    ///     RdfFormat::NTriples.iri(),
    ///     "http://www.w3.org/ns/formats/N-Triples"
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub const fn iri(self) -> &'static str {
        match self {
            Self::JsonLd { .. } => "https://www.w3.org/ns/formats/data/JSON-LD",
            Self::N3 => "http://www.w3.org/ns/formats/N3",
            Self::NQuads => "http://www.w3.org/ns/formats/N-Quads",
            Self::NTriples => "http://www.w3.org/ns/formats/N-Triples",
            Self::RdfXml => "http://www.w3.org/ns/formats/RDF_XML",
            Self::TriG => "http://www.w3.org/ns/formats/TriG",
            Self::Turtle => "http://www.w3.org/ns/formats/Turtle",
            Self::TriX => oxixml_trix::TRIX_NAMESPACE,
        }
    }

    /// The format's [IANA media type](https://tools.ietf.org/html/rfc2046).
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(RdfFormat::NTriples.media_type(), "application/n-triples");
    /// assert_eq!(RdfFormat::TriX.media_type(), "application/trix+xml");
    /// ```
    #[inline]
    #[must_use]
    pub const fn media_type(self) -> &'static str {
        match self {
            Self::JsonLd { profile } => {
                if profile.contains(JsonLdProfile::Streaming) {
                    "application/ld+json;profile=http://www.w3.org/ns/json-ld#streaming"
                } else {
                    "application/ld+json"
                }
            }
            Self::N3 => "text/n3",
            Self::NQuads => "application/n-quads",
            Self::NTriples => "application/n-triples",
            Self::RdfXml => "application/rdf+xml",
            Self::TriG => "application/trig",
            Self::Turtle => "text/turtle",
            Self::TriX => "application/trix+xml",
        }
    }

    /// The format's [IANA-registered](https://tools.ietf.org/html/rfc2046) file extension.
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(RdfFormat::NTriples.file_extension(), "nt");
    /// assert_eq!(RdfFormat::TriX.file_extension(), "trix");
    /// ```
    #[inline]
    #[must_use]
    pub const fn file_extension(self) -> &'static str {
        match self {
            Self::JsonLd { .. } => "jsonld",
            Self::N3 => "n3",
            Self::NQuads => "nq",
            Self::NTriples => "nt",
            Self::RdfXml => "rdf",
            Self::TriG => "trig",
            Self::Turtle => "ttl",
            Self::TriX => "trix",
        }
    }

    /// The format's human-readable name.
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(RdfFormat::NTriples.name(), "N-Triples");
    /// ```
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::JsonLd { profile } => {
                if profile.contains(JsonLdProfile::Streaming) {
                    "Streaming JSON-LD"
                } else {
                    "JSON-LD"
                }
            }
            Self::N3 => "N3",
            Self::NQuads => "N-Quads",
            Self::NTriples => "N-Triples",
            Self::RdfXml => "RDF/XML",
            Self::TriG => "TriG",
            Self::Turtle => "Turtle",
            Self::TriX => "TriX",
        }
    }

    /// Whether the format supports [RDF datasets](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-dataset)
    /// (multiple named graphs) rather than only a single
    /// [RDF graph](https://www.w3.org/TR/rdf11-concepts/#dfn-rdf-graph).
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(RdfFormat::NTriples.supports_datasets(), false);
    /// assert_eq!(RdfFormat::NQuads.supports_datasets(), true);
    /// assert_eq!(RdfFormat::TriX.supports_datasets(), true);
    /// ```
    #[inline]
    #[must_use]
    pub const fn supports_datasets(self) -> bool {
        matches!(
            self,
            Self::JsonLd { .. } | Self::NQuads | Self::TriG | Self::TriX
        )
    }

    /// Looks up a known format from a media type.
    ///
    /// Supports parameters (e.g. `;charset=utf-8`) and a handful of common media-type aliases
    /// (for example `application/x-turtle` for [`RdfFormat::Turtle`]).
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(
    ///     RdfFormat::from_media_type("text/turtle; charset=utf-8"),
    ///     Some(RdfFormat::Turtle)
    /// );
    /// assert_eq!(
    ///     RdfFormat::from_media_type("application/trix+xml"),
    ///     Some(RdfFormat::TriX)
    /// );
    /// ```
    #[must_use]
    pub fn from_media_type(media_type: &str) -> Option<Self> {
        const MEDIA_SUBTYPES: [(&str, RdfFormat); 16] = [
            (
                "activity+json",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            (
                "json",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            (
                "ld+json",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            (
                "jsonld",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            ("n-quads", RdfFormat::NQuads),
            ("n-triples", RdfFormat::NTriples),
            ("n3", RdfFormat::N3),
            ("nquads", RdfFormat::NQuads),
            ("ntriples", RdfFormat::NTriples),
            ("plain", RdfFormat::NTriples),
            ("rdf+xml", RdfFormat::RdfXml),
            ("trig", RdfFormat::TriG),
            ("trix", RdfFormat::TriX),
            ("trix+xml", RdfFormat::TriX),
            ("turtle", RdfFormat::Turtle),
            ("xml", RdfFormat::RdfXml),
        ];
        const UTF8_CHARSETS: [&str; 3] = ["ascii", "utf8", "utf-8"];

        let (type_subtype, parameters) = media_type.split_once(';').unwrap_or((media_type, ""));

        let (kind, subtype) = type_subtype.split_once('/')?;
        let kind = kind.trim();
        if !kind.eq_ignore_ascii_case("application") && !kind.eq_ignore_ascii_case("text") {
            return None;
        }
        let subtype = subtype.trim();
        let subtype = subtype.strip_prefix("x-").unwrap_or(subtype);

        let parameters = parameters.trim();
        let parameters = if parameters.is_empty() {
            Vec::new()
        } else {
            parameters
                .split(';')
                .map(|parameter| {
                    let (key, value) = parameter.split_once('=')?;
                    Some((key.trim(), value.trim()))
                })
                .collect::<Option<Vec<_>>>()?
        };

        for (candidate_subtype, mut candidate_id) in MEDIA_SUBTYPES {
            if !candidate_subtype.eq_ignore_ascii_case(subtype) {
                continue;
            }
            // Moves `parameters`: safe because this branch always returns immediately after.
            for (key, mut value) in parameters {
                match key {
                    "charset" if !UTF8_CHARSETS.iter().any(|c| c.eq_ignore_ascii_case(value)) => {
                        return None; // No other charset than UTF-8 is supported.
                    }
                    "profile" => {
                        if value.starts_with('"') && value.ends_with('"') && value.len() >= 2 {
                            value = &value[1..value.len() - 1];
                        }
                        if let RdfFormat::JsonLd { profile } = &mut candidate_id {
                            for value in value.split(' ') {
                                if let Some(value) = JsonLdProfile::from_iri(value.trim()) {
                                    *profile |= value;
                                }
                            }
                        }
                    }
                    _ => (), // Ignored.
                }
            }
            return Some(candidate_id);
        }
        None
    }

    /// Looks up a known format from a file extension (without the leading dot).
    ///
    /// Supports a handful of common extension aliases (for example `.txt` for
    /// [`RdfFormat::NTriples`]).
    ///
    /// ```
    /// use oxixml_io::RdfFormat;
    ///
    /// assert_eq!(RdfFormat::from_extension("nt"), Some(RdfFormat::NTriples));
    /// assert_eq!(RdfFormat::from_extension("trix"), Some(RdfFormat::TriX));
    /// ```
    #[must_use]
    pub fn from_extension(extension: &str) -> Option<Self> {
        const EXTENSIONS: [(&str, RdfFormat); 11] = [
            (
                "json",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            (
                "jsonld",
                RdfFormat::JsonLd {
                    profile: JsonLdProfileSet::empty(),
                },
            ),
            ("n3", RdfFormat::N3),
            ("nq", RdfFormat::NQuads),
            ("nt", RdfFormat::NTriples),
            ("rdf", RdfFormat::RdfXml),
            ("trig", RdfFormat::TriG),
            ("trix", RdfFormat::TriX),
            ("ttl", RdfFormat::Turtle),
            ("txt", RdfFormat::NTriples),
            ("xml", RdfFormat::RdfXml),
        ];
        for (candidate_extension, candidate_id) in EXTENSIONS {
            if candidate_extension.eq_ignore_ascii_case(extension) {
                return Some(candidate_id);
            }
        }
        None
    }
}

impl fmt::Display for RdfFormat {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_media_type_basics() {
        assert_eq!(RdfFormat::from_media_type("foo/bar"), None);
        assert_eq!(RdfFormat::from_media_type("text/csv"), None);
        assert_eq!(
            RdfFormat::from_media_type("text/turtle"),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            RdfFormat::from_media_type("application/x-turtle"),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            RdfFormat::from_media_type("text/turtle;charset=utf-8"),
            Some(RdfFormat::Turtle)
        );
        assert_eq!(
            RdfFormat::from_media_type("text/turtle;charset=latin1"),
            None
        );
        assert_eq!(
            RdfFormat::from_media_type("application/ld+json"),
            Some(RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty()
            })
        );
        assert_eq!(
            RdfFormat::from_media_type(
                "application/ld+json;profile=http://www.w3.org/ns/json-ld#streaming"
            ),
            Some(RdfFormat::JsonLd {
                profile: JsonLdProfile::Streaming.into()
            })
        );
        assert_eq!(
            RdfFormat::from_media_type("application/trix+xml"),
            Some(RdfFormat::TriX)
        );
    }

    #[test]
    fn from_extension_basics() {
        assert_eq!(RdfFormat::from_extension("nt"), Some(RdfFormat::NTriples));
        assert_eq!(RdfFormat::from_extension("NT"), Some(RdfFormat::NTriples));
        assert_eq!(RdfFormat::from_extension("unknown"), None);
        assert_eq!(RdfFormat::from_extension("trix"), Some(RdfFormat::TriX));
    }

    #[test]
    fn round_trip_metadata() {
        for format in [
            RdfFormat::N3,
            RdfFormat::NQuads,
            RdfFormat::NTriples,
            RdfFormat::RdfXml,
            RdfFormat::TriG,
            RdfFormat::Turtle,
            RdfFormat::TriX,
        ] {
            assert_eq!(
                RdfFormat::from_extension(format.file_extension()),
                Some(format)
            );
            assert_eq!(format.to_string(), format.name());
        }
    }

    #[test]
    fn supports_datasets_matrix() {
        assert!(!RdfFormat::NTriples.supports_datasets());
        assert!(!RdfFormat::Turtle.supports_datasets());
        assert!(!RdfFormat::RdfXml.supports_datasets());
        assert!(!RdfFormat::N3.supports_datasets());
        assert!(RdfFormat::NQuads.supports_datasets());
        assert!(RdfFormat::TriG.supports_datasets());
        assert!(RdfFormat::TriX.supports_datasets());
        assert!(
            RdfFormat::JsonLd {
                profile: JsonLdProfileSet::empty()
            }
            .supports_datasets()
        );
    }
}
