//! Output-format selection and RDF-emission helpers shared by the
//! chart-family subcommands and `oxieph vocab`.

use std::io::Write;

use clap::ValueEnum;

use oxiephemeris_rdf::oxrdf::Graph;
use oxiephemeris_rdf::serialize::{write_ntriples, write_turtle};

use crate::errors::CliError;

/// How a command renders its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text.
    Text,
    /// The command's stable JSON schema.
    Json,
    /// RDF 1.1 Turtle.
    Turtle,
    /// RDF 1.1 N-Triples.
    ///
    /// Clap would derive the value name `n-triples` from the variant; the
    /// spelling users actually type is `ntriples`, so that is the primary
    /// name and the hyphenated and short forms are aliases.
    #[value(name = "ntriples", alias = "n-triples", alias = "nt")]
    NTriples,
}

impl OutputFormat {
    /// Whether this format is one of the RDF serializations.
    #[must_use]
    pub const fn is_rdf(self) -> bool {
        matches!(self, Self::Turtle | Self::NTriples)
    }

    /// The `oxiephemeris_chart` facade RDF format this corresponds to, or
    /// `None` for the non-RDF (`Text`/`Json`) formats.
    #[must_use]
    pub const fn as_rdf_format(self) -> Option<oxiephemeris_chart::RdfFormat> {
        match self {
            Self::Turtle => Some(oxiephemeris_chart::RdfFormat::Turtle),
            Self::NTriples => Some(oxiephemeris_chart::RdfFormat::NTriples),
            Self::Text | Self::Json => None,
        }
    }
}

/// Resolves the effective format from a `--format` value and the legacy
/// `--json` flag.
///
/// `--json` predates `--format` and is kept working: when it is set and
/// `--format` was left at its default, the result is
/// [`OutputFormat::Json`]. An explicit `--format` always wins, so
/// `--json --format turtle` emits Turtle rather than silently ignoring the
/// newer flag.
#[must_use]
pub const fn resolve_format(format: Option<OutputFormat>, json_flag: bool) -> OutputFormat {
    match format {
        Some(explicit) => explicit,
        None if json_flag => OutputFormat::Json,
        None => OutputFormat::Text,
    }
}

/// Writes `graph` to stdout in the requested RDF format.
///
/// # Errors
///
/// [`CliError::Rdf`] if serialization or writing fails.
pub fn print_graph(graph: &Graph, format: OutputFormat) -> Result<(), CliError> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    match format {
        OutputFormat::Turtle => write_turtle(graph, &mut handle)?,
        OutputFormat::NTriples => write_ntriples(graph, &mut handle)?,
        OutputFormat::Text | OutputFormat::Json => {
            return Err(CliError::Arg(
                "print_graph called with a non-RDF format".to_owned(),
            ))
        }
    }
    handle.flush().map_err(|e| CliError::Rdf(e.into()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_flag_is_a_default_only_alias() {
        assert_eq!(resolve_format(None, false), OutputFormat::Text);
        assert_eq!(resolve_format(None, true), OutputFormat::Json);
        // An explicit --format wins over the legacy flag.
        assert_eq!(
            resolve_format(Some(OutputFormat::Turtle), true),
            OutputFormat::Turtle
        );
    }

    #[test]
    fn rdf_formats_are_classified() {
        assert!(OutputFormat::Turtle.is_rdf());
        assert!(OutputFormat::NTriples.is_rdf());
        assert!(!OutputFormat::Json.is_rdf());
        assert!(!OutputFormat::Text.is_rdf());
    }

    #[test]
    fn print_graph_rejects_non_rdf_formats() {
        let graph = Graph::new();
        assert!(print_graph(&graph, OutputFormat::Json).is_err());
    }
}
