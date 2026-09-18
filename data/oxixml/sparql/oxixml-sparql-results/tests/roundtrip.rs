//! Self-contained round-trip tests: parse -> serialize -> parse must produce identical
//! solutions for the three lossless formats (XML, JSON, TSV). CSV is intentionally lossy and
//! is checked separately for its documented escaping behavior only.

use oxixml_model::{BlankNode, Literal, NamedNode, Variable};
use oxixml_sparql_results::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, QuerySolution,
    ReaderQueryResultsParserOutput,
};
use std::error::Error;

fn sample_variables() -> Vec<Variable> {
    vec![
        Variable::new("s").expect("valid variable"),
        Variable::new("p").expect("valid variable"),
        Variable::new("o").expect("valid variable"),
    ]
}

fn sample_solutions(variables: &[Variable]) -> Vec<QuerySolution> {
    vec![
        QuerySolution::from((
            variables.to_vec(),
            vec![
                Some(NamedNode::new_unchecked("http://example.com/s1").into()),
                Some(NamedNode::new_unchecked("http://example.com/p1").into()),
                // Note: XML SPARQL results trims leading/trailing ASCII whitespace from decoded
                // `<literal>` text content (matching `sparesults`' exact behavior), so this
                // fixture keeps whitespace mid-string rather than at either boundary.
                Some(Literal::new_simple_literal("hello\n\"world\"\tend").into()),
            ],
        )),
        QuerySolution::from((
            variables.to_vec(),
            vec![
                Some(BlankNode::new_unchecked("b0").into()),
                Some(NamedNode::new_unchecked("http://example.com/p2").into()),
                Some(
                    Literal::new_language_tagged_literal("bonjour", "fr")
                        .expect("valid lang tag")
                        .into(),
                ),
            ],
        )),
        QuerySolution::from((
            variables.to_vec(),
            vec![
                Some(NamedNode::new_unchecked("http://example.com/s3").into()),
                None,
                Some(Literal::from(42).into()),
            ],
        )),
        QuerySolution::from((variables.to_vec(), vec![None, None, None])),
    ]
}

fn round_trip(format: QueryResultsFormat) -> Result<(), Box<dyn Error>> {
    let variables = sample_variables();
    let solutions = sample_solutions(&variables);

    let mut buffer = Vec::new();
    let mut writer = QueryResultsSerializer::from_format(format)
        .serialize_solutions_to_writer(&mut buffer, variables.clone())?;
    for solution in &solutions {
        writer.serialize(solution.iter())?;
    }
    writer.finish()?;

    let parsed = match QueryResultsParser::from_format(format).for_reader(buffer.as_slice())? {
        ReaderQueryResultsParserOutput::Solutions(reader) => {
            assert_eq!(
                reader.variables(),
                variables.as_slice(),
                "format {format:?} variables"
            );
            reader.collect::<Result<Vec<_>, _>>()?
        }
        ReaderQueryResultsParserOutput::Boolean(_) => panic!("expected solutions, found a boolean"),
    };

    assert_eq!(
        parsed.len(),
        solutions.len(),
        "format {format:?} solution count"
    );
    for (expected, actual) in solutions.iter().zip(&parsed) {
        assert_eq!(expected, actual, "format {format:?} solution mismatch");
    }
    Ok(())
}

#[test]
fn xml_round_trip() -> Result<(), Box<dyn Error>> {
    round_trip(QueryResultsFormat::Xml)
}

#[test]
fn json_round_trip() -> Result<(), Box<dyn Error>> {
    round_trip(QueryResultsFormat::Json)
}

#[test]
fn tsv_round_trip() -> Result<(), Box<dyn Error>> {
    round_trip(QueryResultsFormat::Tsv)
}

#[test]
fn boolean_round_trip_all_formats() -> Result<(), Box<dyn Error>> {
    for format in [
        QueryResultsFormat::Xml,
        QueryResultsFormat::Json,
        QueryResultsFormat::Csv,
        QueryResultsFormat::Tsv,
    ] {
        for value in [true, false] {
            let mut buffer = Vec::new();
            QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_writer(&mut buffer, value)?;
            if format == QueryResultsFormat::Csv {
                // CSV/TSV booleans share the bare `true`/`false` serialization and are not
                // format-tagged, so only round-trip XML/JSON/TSV through the parser.
                continue;
            }
            match QueryResultsParser::from_format(format).for_reader(buffer.as_slice())? {
                ReaderQueryResultsParserOutput::Boolean(v) => {
                    assert_eq!(v, value, "format {format:?}")
                }
                ReaderQueryResultsParserOutput::Solutions(_) => panic!("expected a boolean"),
            }
        }
    }
    Ok(())
}

#[test]
fn csv_is_write_only() {
    let mut buffer = Vec::new();
    let mut writer = QueryResultsSerializer::from_format(QueryResultsFormat::Csv)
        .serialize_solutions_to_writer(&mut buffer, sample_variables())
        .expect("csv writer starts");
    writer
        .serialize(std::iter::empty())
        .expect("csv row writes");
    writer.finish().expect("csv writer finishes");
    assert!(
        QueryResultsParser::from_format(QueryResultsFormat::Csv)
            .for_slice(&buffer)
            .is_err(),
        "CSV parsing must be rejected: the format is lossy"
    );
}
