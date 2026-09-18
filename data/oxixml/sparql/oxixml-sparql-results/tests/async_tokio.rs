//! Round-trip tests for the `async-tokio` feature of `oxixml-sparql-results`.
//!
//! Every supported format is driven through both the blocking and the asynchronous API and the
//! two are asserted to agree, byte for byte on serialization and solution for solution on
//! parsing.

#![cfg(feature = "async-tokio")]

use oxixml_model::{Literal, Term, Variable};
use oxixml_sparql_results::{
    QueryResultsFormat, QueryResultsParser, QueryResultsSerializer, ReaderQueryResultsParserOutput,
    TokioAsyncReaderQueryResultsParserOutput,
};

/// The formats that can be both written and read back.
const ROUND_TRIP_FORMATS: [QueryResultsFormat; 3] = [
    QueryResultsFormat::Xml,
    QueryResultsFormat::Json,
    QueryResultsFormat::Tsv,
];

/// Every format the serializer supports (CSV is write-only: it is a lossy serialization).
const ALL_FORMATS: [QueryResultsFormat; 4] = [
    QueryResultsFormat::Xml,
    QueryResultsFormat::Json,
    QueryResultsFormat::Csv,
    QueryResultsFormat::Tsv,
];

/// One binding per solution: the variable it binds and the term bound to it.
type Binding = (Variable, Term);

/// The declared variables and the bindings each solution carries, shared by every test.
type Sample = (Vec<Variable>, Vec<Binding>);

fn sample() -> Result<Sample, Box<dyn std::error::Error>> {
    let first = Variable::new("first")?;
    let second = Variable::new("second")?;
    let bindings = vec![
        (first.clone(), Literal::new_simple_literal("test").into()),
        (second.clone(), Literal::new_simple_literal("other").into()),
    ];
    Ok((vec![first, second], bindings))
}

async fn serialize_async(
    format: QueryResultsFormat,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (variables, bindings) = sample()?;
    let mut serializer = QueryResultsSerializer::from_format(format)
        .serialize_solutions_to_tokio_async_write(Vec::new(), variables)
        .await?;
    for (variable, term) in &bindings {
        serializer.serialize([(variable, term)]).await?;
    }
    Ok(serializer.finish().await?)
}

fn serialize_blocking(format: QueryResultsFormat) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (variables, bindings) = sample()?;
    let mut serializer = QueryResultsSerializer::from_format(format)
        .serialize_solutions_to_writer(Vec::new(), variables)?;
    for (variable, term) in &bindings {
        serializer.serialize([(variable, term)])?;
    }
    Ok(serializer.finish()?)
}

#[tokio::test]
async fn async_solutions_serializer_matches_blocking() -> Result<(), Box<dyn std::error::Error>> {
    for format in ALL_FORMATS {
        assert_eq!(
            serialize_async(format).await?,
            serialize_blocking(format)?,
            "serialized bytes differ for {format:?}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn async_boolean_serializer_matches_blocking() -> Result<(), Box<dyn std::error::Error>> {
    for format in ALL_FORMATS {
        for value in [true, false] {
            let asynchronous = QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_tokio_async_write(Vec::new(), value)
                .await?;
            let blocking = QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_writer(Vec::new(), value)?;
            assert_eq!(
                asynchronous, blocking,
                "boolean bytes differ for {format:?}"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn async_solutions_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let (variables, _) = sample()?;
    for format in ROUND_TRIP_FORMATS {
        let bytes = serialize_async(format).await?;

        let output = QueryResultsParser::from_format(format)
            .for_tokio_async_reader(bytes.as_slice())
            .await?;
        let TokioAsyncReaderQueryResultsParserOutput::Solutions(mut solutions) = output else {
            panic!("{format:?} should have produced solutions");
        };
        assert_eq!(solutions.variables(), variables.as_slice());

        let mut parsed = Vec::new();
        while let Some(solution) = solutions.next().await {
            let solution = solution?;
            parsed.push(
                solution
                    .iter()
                    .map(|(variable, term)| (variable.clone(), term.clone()))
                    .collect::<Vec<_>>(),
            );
        }

        // The blocking parser over the same bytes must agree exactly.
        let ReaderQueryResultsParserOutput::Solutions(blocking) =
            QueryResultsParser::from_format(format).for_reader(bytes.as_slice())?
        else {
            panic!("{format:?} should have produced solutions");
        };
        let expected = blocking
            .map(|solution| {
                solution.map(|solution| {
                    solution
                        .iter()
                        .map(|(variable, term)| (variable.clone(), term.clone()))
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(parsed, expected, "parsed solutions differ for {format:?}");
        assert_eq!(parsed.len(), 2);
    }
    Ok(())
}

#[tokio::test]
async fn async_boolean_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    for format in ROUND_TRIP_FORMATS {
        for value in [true, false] {
            let bytes = QueryResultsSerializer::from_format(format)
                .serialize_boolean_to_tokio_async_write(Vec::new(), value)
                .await?;
            let output = QueryResultsParser::from_format(format)
                .for_tokio_async_reader(bytes.as_slice())
                .await?;
            let TokioAsyncReaderQueryResultsParserOutput::Boolean(parsed) = output else {
                panic!("{format:?} should have produced a boolean");
            };
            assert_eq!(parsed, value, "boolean differs for {format:?}");
        }
    }
    Ok(())
}

#[tokio::test]
async fn async_parser_rejects_csv() -> Result<(), Box<dyn std::error::Error>> {
    // CSV is lossy and, exactly as in the blocking API, cannot be parsed back.
    let error = QueryResultsParser::from_format(QueryResultsFormat::Csv)
        .for_tokio_async_reader(b"first\n".as_slice())
        .await
        .err()
        .ok_or("CSV parsing should be rejected")?;
    assert!(error.to_string().contains("lossy"), "unexpected: {error}");
    Ok(())
}

/// A reader that yields one byte per poll, proving the async parsers really feed their
/// incremental cores rather than slurping the whole input.
#[tokio::test]
async fn async_parser_handles_chunked_reads() -> Result<(), Box<dyn std::error::Error>> {
    struct OneByteAtATime<'a> {
        data: &'a [u8],
    }

    impl tokio::io::AsyncRead for OneByteAtATime<'_> {
        fn poll_read(
            mut self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
            buf: &mut tokio::io::ReadBuf<'_>,
        ) -> std::task::Poll<std::io::Result<()>> {
            if let Some((first, rest)) = self.data.split_first() {
                buf.put_slice(std::slice::from_ref(first));
                self.data = rest;
            }
            std::task::Poll::Ready(Ok(()))
        }
    }

    for format in ROUND_TRIP_FORMATS {
        let bytes = serialize_blocking(format)?;
        let output = QueryResultsParser::from_format(format)
            .for_tokio_async_reader(OneByteAtATime { data: &bytes })
            .await?;
        let TokioAsyncReaderQueryResultsParserOutput::Solutions(mut solutions) = output else {
            panic!("{format:?} should have produced solutions");
        };
        let mut count = 0;
        while let Some(solution) = solutions.next().await {
            solution?;
            count += 1;
        }
        assert_eq!(count, 2, "solution count differs for {format:?}");
    }
    Ok(())
}

/// The JSON solutions reader must behave like a fused iterator after an error on the
/// asynchronous path too: an undeclared-variable binding must fail on its own call, and every
/// call after that must return `None` rather than feeding the next raw token into state left
/// over from the error (which used to panic on the following, otherwise well-formed, solution).
///
/// The blocking `for_reader`/`for_slice` paths are covered in `src/json.rs`'s own unit tests;
/// this is the same repro through `for_tokio_async_reader`, which shares the same inner reader.
#[tokio::test]
async fn async_json_solutions_are_fused_after_an_undeclared_variable_error()
-> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{"head":{"vars":["x"]},"results":{"bindings":[
        {"x":{"type":"uri","value":"http://example.com/1"}},
        {"y":{"type":"uri","value":"http://example.com/bad"}},
        {"x":{"type":"uri","value":"http://example.com/2"}}
    ]}}"#;

    let output = QueryResultsParser::from_format(QueryResultsFormat::Json)
        .for_tokio_async_reader(json.as_bytes())
        .await?;
    let TokioAsyncReaderQueryResultsParserOutput::Solutions(mut solutions) = output else {
        panic!("expected solutions");
    };

    let first = solutions.next().await.expect("a first item");
    assert!(first.is_ok(), "the first, well-formed solution: {first:?}");
    let second = solutions.next().await.expect("a second item");
    assert!(second.is_err(), "the malformed solution must fail");
    // Fused: every call from here on returns `None`, never `Some(Ok(_))` (recovering into the
    // third, otherwise well-formed solution) nor a panic.
    assert!(
        solutions.next().await.is_none(),
        "must be fused after the error"
    );
    assert!(solutions.next().await.is_none(), "must stay fused");
    Ok(())
}
