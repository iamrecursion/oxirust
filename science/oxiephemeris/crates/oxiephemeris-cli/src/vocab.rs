//! `oxieph vocab`: dump the `OxiEphemeris` astrology vocabulary as RDF.
//!
//! Two documents live behind the published namespaces:
//!
//! * the **ontology** (`oxa:` — classes, properties, ranges, QUDT units),
//! * the **SKOS concept schemes** (`oxc:`/`oxs:` — the twelve signs, ten
//!   planets, eleven aspects, and the nine smaller schemes), generated
//!   from the engine's own enums and dignity tables so they cannot drift
//!   from the code that computes charts.
//!
//! By default both are emitted into one document, which is what a
//! consumer usually wants. `--part ontology|concepts` narrows it.
//!
//! The output is byte-stable: triples are emitted in a canonical sorted
//! order, so re-running this command never produces a spurious diff.

use clap::{Args, ValueEnum};

use oxiephemeris_rdf::oxrdf::Graph;
use oxiephemeris_rdf::{concept_scheme_graph, ontology_graph};

use crate::errors::CliError;
use crate::rdf_args::{print_graph, OutputFormat};

/// Which document to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VocabPart {
    /// Both the ontology and the concept schemes.
    All,
    /// The `oxa:` classes and properties only.
    Ontology,
    /// The SKOS concept schemes only.
    Concepts,
}

/// `oxieph vocab` arguments.
#[derive(Debug, Args)]
pub struct VocabArgs {
    /// Which part of the vocabulary to emit.
    #[arg(long, value_enum, default_value_t = VocabPart::All)]
    pub part: VocabPart,
    /// RDF serialization to emit.
    #[arg(long, value_enum, default_value_t = OutputFormat::Turtle)]
    pub format: OutputFormat,
}

/// Runs `oxieph vocab`.
///
/// # Errors
///
/// [`CliError::Arg`] if a non-RDF `--format` is requested (the vocabulary
/// has no text or JSON rendering), or [`CliError::Rdf`] on a write
/// failure.
pub fn run(args: &VocabArgs) -> Result<(), CliError> {
    if !args.format.is_rdf() {
        return Err(CliError::Arg(
            "vocab emits RDF only; use --format turtle or --format ntriples".to_owned(),
        ));
    }
    let graph = match args.part {
        VocabPart::Ontology => ontology_graph(),
        VocabPart::Concepts => concept_scheme_graph(),
        VocabPart::All => {
            let mut merged: Graph = ontology_graph();
            for triple in &concept_scheme_graph() {
                merged.insert(triple);
            }
            merged
        }
    };
    print_graph(&graph, args.format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiephemeris_rdf::to_turtle_string;

    #[test]
    fn all_is_the_union_of_the_two_documents() {
        let ontology = ontology_graph();
        let concepts = concept_scheme_graph();
        let mut merged = ontology_graph();
        for triple in &concepts {
            merged.insert(triple);
        }
        // The two documents share no triples, so the union is their sum.
        assert_eq!(merged.len(), ontology.len() + concepts.len());
    }

    #[test]
    fn text_and_json_formats_are_rejected() {
        for format in [OutputFormat::Text, OutputFormat::Json] {
            let args = VocabArgs {
                part: VocabPart::All,
                format,
            };
            assert!(run(&args).is_err(), "{format:?} must be rejected");
        }
    }

    #[test]
    fn concepts_document_mentions_scorpio_and_its_ruler() {
        let Ok(turtle) = to_turtle_string(&concept_scheme_graph()) else {
            panic!("serialization must succeed");
        };
        assert!(turtle.contains("sign:Scorpio"));
        assert!(turtle.contains("traditionalRuler"));
        assert!(turtle.contains("\"蠍座\"@ja"), "Japanese label missing");
    }
}
