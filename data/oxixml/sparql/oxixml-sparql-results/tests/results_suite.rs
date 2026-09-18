//! W3C `rdf-tests/sparql` results-format corpus harness for `oxixml-sparql-results`.
//!
//! The `rdf-tests/sparql` suite ships hundreds of `.srx` (XML) and `.srj` (JSON) fixture files
//! used across many `sparql11`/`sparql12` manifests as expected query results. They are not a
//! results-format conformance suite in their own right (there is no per-file pass/fail
//! manifest for *this* crate), so this harness uses them as a large, real-world corpus for
//! round-trip fidelity: every fixture that our parser accepts must, after being re-serialized
//! and re-parsed, produce the exact same set of solutions (or boolean).
//!
//! The corpus root is discovered from the `OXIXML_TESTSUITE_DIR` environment variable and the
//! whole harness silently succeeds (skips) when it is unset, so the crate builds and tests
//! cleanly without the data checked out.

use oxixml_sparql_results::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};
use std::path::{Path, PathBuf};

fn testsuite_root() -> Option<PathBuf> {
    let dir = std::env::var_os("OXIXML_TESTSUITE_DIR")?;
    let root = PathBuf::from(dir).join("rdf-tests").join("sparql");
    root.is_dir().then_some(root)
}

fn find_files(dir: &Path, extension: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            find_files(&path, extension, out);
        } else if path.extension().is_some_and(|e| e == extension) {
            out.push(path);
        }
    }
}

/// Parses `bytes` in `format`, fully draining any solutions, and returns either the collected
/// solutions or the boolean result. Returns `Err` if any part of the document fails to parse.
fn parse_all(
    format: QueryResultsFormat,
    bytes: &[u8],
) -> Result<ParsedResult, Box<dyn std::error::Error>> {
    Ok(
        match QueryResultsParser::from_format(format).for_reader(bytes)? {
            ReaderQueryResultsParserOutput::Boolean(v) => ParsedResult::Boolean(v),
            ReaderQueryResultsParserOutput::Solutions(reader) => {
                let variables = reader.variables().to_vec();
                let solutions = reader.collect::<Result<Vec<_>, _>>()?;
                ParsedResult::Solutions {
                    variables,
                    solutions,
                }
            }
        },
    )
}

enum ParsedResult {
    Solutions {
        variables: Vec<oxixml_model::Variable>,
        solutions: Vec<QuerySolution>,
    },
    Boolean(bool),
}

/// Runs the round-trip fidelity check on every fixture with the given `extension` under `root`,
/// returning `(attempted, round_tripped)` file counts. A file that fails to parse at all is
/// simply not counted as "attempted" (the corpus is not a conformance suite for this crate: not
/// every fixture is necessarily valid standalone SPARQL results, and some require the `rdf-12`
/// feature to be enabled).
fn run_corpus(root: &Path, format: QueryResultsFormat, extension: &str) -> (usize, usize) {
    let mut files = Vec::new();
    find_files(root, extension, &mut files);
    assert!(
        !files.is_empty(),
        "expected to find *.{extension} fixtures under {root:?}"
    );

    let mut attempted = 0;
    let mut round_tripped = 0;
    for file in files {
        let Ok(bytes) = std::fs::read(&file) else {
            continue;
        };
        let Ok(first) = parse_all(format, &bytes) else {
            continue;
        };
        attempted += 1;

        let mut buffer = Vec::new();
        let ok = match &first {
            ParsedResult::Boolean(v) => QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_writer(&mut buffer, *v)
                .is_ok(),
            ParsedResult::Solutions {
                variables,
                solutions,
            } => (|| -> Option<()> {
                let mut writer = QueryResultsSerializer::from_format(format)
                    .serialize_solutions_to_writer(&mut buffer, variables.clone())
                    .ok()?;
                for solution in solutions {
                    writer.serialize(solution.iter()).ok()?;
                }
                writer.finish().ok()?;
                Some(())
            })()
            .is_some(),
        };
        if !ok {
            eprintln!(
                "oxixml-sparql-results: re-serialization failed for {}",
                file.display()
            );
            continue;
        }

        let Ok(second) = parse_all(format, &buffer) else {
            eprintln!(
                "oxixml-sparql-results: re-parse failed for {}",
                file.display()
            );
            continue;
        };

        let matches = match (&first, &second) {
            (ParsedResult::Boolean(a), ParsedResult::Boolean(b)) => a == b,
            (
                ParsedResult::Solutions {
                    variables: v1,
                    solutions: s1,
                },
                ParsedResult::Solutions {
                    variables: v2,
                    solutions: s2,
                },
            ) => v1 == v2 && s1 == s2,
            _ => false,
        };
        if matches {
            round_tripped += 1;
        } else {
            eprintln!(
                "oxixml-sparql-results: round-trip mismatch for {}",
                file.display()
            );
        }
    }
    (attempted, round_tripped)
}

#[test]
fn srx_round_trip_corpus() {
    let Some(root) = testsuite_root() else {
        eprintln!("OXIXML_TESTSUITE_DIR is unset: skipping the rdf-tests/sparql SRX corpus");
        return;
    };
    let (attempted, round_tripped) = run_corpus(&root, QueryResultsFormat::Xml, "srx");
    assert!(attempted > 0, "no .srx fixture parsed successfully");
    assert_eq!(
        attempted,
        round_tripped,
        "{}/{} SRX fixtures failed to round-trip",
        attempted - round_tripped,
        attempted
    );
}

#[test]
fn srj_round_trip_corpus() {
    let Some(root) = testsuite_root() else {
        eprintln!("OXIXML_TESTSUITE_DIR is unset: skipping the rdf-tests/sparql SRJ corpus");
        return;
    };
    let (attempted, round_tripped) = run_corpus(&root, QueryResultsFormat::Json, "srj");
    assert!(attempted > 0, "no .srj fixture parsed successfully");
    assert_eq!(
        attempted,
        round_tripped,
        "{}/{} SRJ fixtures failed to round-trip",
        attempted - round_tripped,
        attempted
    );
}
