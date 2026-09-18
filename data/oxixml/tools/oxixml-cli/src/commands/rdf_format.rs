//! Mapping the RDF format names `canon`, `convert` and `hdt build` accept on the command line to
//! [`RdfFormat`], with a few common aliases (`ttl`, `nt`, `nq`, ...) alongside the full names.

use oxixml_io::RdfFormat;

/// Resolves a format name given on the command line, or the failure message to report when it
/// is not recognized.
///
/// # Errors
///
/// A human-readable message listing the accepted names.
pub fn parse(name: &str) -> Result<RdfFormat, String> {
    match name.to_ascii_lowercase().as_str() {
        "turtle" | "ttl" => Ok(RdfFormat::Turtle),
        "trig" => Ok(RdfFormat::TriG),
        "ntriples" | "n-triples" | "nt" => Ok(RdfFormat::NTriples),
        "nquads" | "n-quads" | "nq" => Ok(RdfFormat::NQuads),
        "rdfxml" | "rdf-xml" | "rdf" | "xml" => Ok(RdfFormat::RdfXml),
        "jsonld" | "json-ld" | "json" => Ok(RdfFormat::JsonLd {
            profile: oxixml_io::JsonLdProfileSet::empty(),
        }),
        "trix" => Ok(RdfFormat::TriX),
        "n3" => Ok(RdfFormat::N3),
        other => Err(format!(
            "unknown RDF format `{other}` (expected one of: turtle, trig, ntriples, nquads, \
             rdfxml, jsonld, trix, n3)"
        )),
    }
}

/// Resolves the format named by a file's extension, for auto-detection when no explicit format
/// was given on the command line.
#[must_use]
pub fn from_path(path: &str) -> Option<RdfFormat> {
    let extension = std::path::Path::new(path).extension()?.to_str()?;
    RdfFormat::from_extension(extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_aliases_resolve() {
        assert_eq!(parse("ttl"), Ok(RdfFormat::Turtle));
        assert_eq!(parse("NT"), Ok(RdfFormat::NTriples));
        assert_eq!(parse("nq"), Ok(RdfFormat::NQuads));
    }

    #[test]
    fn unknown_formats_are_rejected() {
        assert!(parse("bogus").is_err());
    }

    #[test]
    fn from_path_uses_the_extension() {
        assert_eq!(from_path("data.ttl"), Some(RdfFormat::Turtle));
        assert_eq!(from_path("data"), None);
    }
}
