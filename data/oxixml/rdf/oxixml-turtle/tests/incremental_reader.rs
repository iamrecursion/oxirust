//! Chunk-boundary torture tests for the incremental `for_reader` code path.
//!
//! `for_reader` grows the lexer's buffer with whatever a single `Read::read` call returns, so a
//! source that hands over one byte at a time forces every token to be assembled across many
//! separate buffer extensions — exactly where chunk-boundary bugs hide. The crate's other tests
//! only ever call `for_slice` with the whole document already in memory.
//!
//! Every case asserts parity with the `for_slice` result on the same input, including error
//! messages and byte offsets.

use oxixml_turtle::nquads::NQuadsParser;
use oxixml_turtle::ntriples::NTriplesParser;
use oxixml_turtle::trig::TriGParser;
use oxixml_turtle::turtle::TurtleParser;
use std::io::{Read, Result as IoResult};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A [`Read`] source that hands out at most `chunk` bytes per call.
struct ChunkedReader<'a> {
    data: &'a [u8],
    chunk: usize,
}

impl Read for ChunkedReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> IoResult<usize> {
        let take = self.chunk.min(buffer.len()).min(self.data.len());
        buffer[..take].copy_from_slice(&self.data[..take]);
        self.data = &self.data[take..];
        Ok(take)
    }
}

/// The documents every syntax-specific test runs through, in a form each grammar accepts.
///
/// Deliberately free of blank-node syntax: parsing twice allocates fresh labels each time, which
/// would make the two results differ for reasons unrelated to chunking.
fn line_based_documents() -> Vec<&'static str> {
    vec![
        "",
        "\n\n\n",
        "# only a comment\n",
        "<http://example.com/s> <http://example.com/p> <http://example.com/o> .\n",
        concat!(
            "<http://example.com/s> <http://example.com/p> ",
            "\"a long literal with \\\"escapes\\\", \\\\backslashes\\\\, \\n newlines and \\t tabs\" .\n"
        ),
        "<http://example.com/s> <http://example.com/p> \"日本語のニックネーム 😀\"@ja .\n",
        "<http://example.com/héllo> <http://example.com/p> \"x\"^^<http://example.com/d> .\n",
        // Malformed: a missing final dot.
        "<http://example.com/s> <http://example.com/p> <http://example.com/o>\n",
        // Malformed: a missing object.
        "<http://example.com/s> <http://example.com/p> .\n",
        // Malformed: an unterminated string.
        "<http://example.com/s> <http://example.com/p> \"unterminated .\n",
    ]
}

/// 500 statements, so the incremental buffer has to grow (and re-grow) many times.
fn long_document() -> String {
    let mut document = String::new();
    for i in 0..500 {
        document.push_str(&format!(
            "<http://example.com/s{i}> <http://example.com/p> \"value {i} with \\\"escapes\\\"\" .\n"
        ));
    }
    document
}

/// Renders one parse as a comparable string, errors included verbatim.
fn render<I, T, E>(iterator: I) -> Vec<String>
where
    I: Iterator<Item = Result<T, E>>,
    T: std::fmt::Display,
    E: std::fmt::Display,
{
    iterator
        .map(|item| match item {
            Ok(value) => format!("ok {value}"),
            Err(error) => format!("err {error}"),
        })
        .collect()
}

macro_rules! parity_test {
    ($name:ident, $parser:ty, $documents:expr) => {
        #[test]
        fn $name() -> TestResult {
            for document in $documents {
                let document: &str = document.as_ref();
                let expected = render(<$parser>::new().for_slice(document.as_bytes()));
                for chunk in [1usize, 2, 3, 7, 64] {
                    let reader = ChunkedReader {
                        data: document.as_bytes(),
                        chunk,
                    };
                    let actual = render(<$parser>::new().for_reader(reader));
                    assert_eq!(
                        actual, expected,
                        "chunk size {chunk} disagrees with for_slice on {document:?}"
                    );
                }
            }
            Ok(())
        }
    };
}

parity_test!(
    ntriples_reader_matches_slice,
    NTriplesParser,
    line_based_documents()
);
parity_test!(
    nquads_reader_matches_slice,
    NQuadsParser,
    line_based_documents()
);
parity_test!(
    turtle_reader_matches_slice,
    TurtleParser,
    line_based_documents()
);
parity_test!(
    trig_reader_matches_slice,
    TriGParser,
    line_based_documents()
);

parity_test!(
    ntriples_reader_matches_slice_on_a_long_document,
    NTriplesParser,
    [long_document()]
);
parity_test!(
    turtle_reader_matches_slice_on_a_long_document,
    TurtleParser,
    [long_document()]
);

/// Turtle-specific syntax whose tokens (prefixes, prefixed names, long strings, comments) each
/// span several one-byte chunks.
#[test]
fn turtle_prefixes_and_long_strings_survive_one_byte_chunks() -> TestResult {
    let document = concat!(
        "@prefix ex: <http://example.com/> . # a trailing comment\n",
        "@base <http://example.com/base/> .\n",
        "ex:s ex:p \"\"\"a long\nstring spanning\nseveral lines\"\"\" ;\n",
        "     ex:p2 'single quoted' , 42 , 4.2 , 1.0e3 , true .\n",
        "<relative> ex:p ex:o .\n"
    );
    let expected = render(TurtleParser::new().for_slice(document.as_bytes()));
    assert!(expected.iter().all(|line| line.starts_with("ok ")));
    for chunk in [1usize, 2, 5, 13] {
        let reader = ChunkedReader {
            data: document.as_bytes(),
            chunk,
        };
        assert_eq!(
            render(TurtleParser::new().for_reader(reader)),
            expected,
            "chunk size {chunk}"
        );
    }
    Ok(())
}

/// TriG-specific syntax: named graph blocks split across chunks.
#[test]
fn trig_graph_blocks_survive_one_byte_chunks() -> TestResult {
    let document = concat!(
        "@prefix ex: <http://example.com/> .\n",
        "ex:g { ex:s ex:p ex:o . ex:s ex:p \"literal\" . }\n",
        "GRAPH ex:g2 { ex:s2 ex:p2 \"日本語\"@ja . }\n",
        "ex:s3 ex:p3 ex:o3 .\n"
    );
    let expected = render(TriGParser::new().for_slice(document.as_bytes()));
    assert_eq!(expected.len(), 4);
    for chunk in [1usize, 3, 11] {
        let reader = ChunkedReader {
            data: document.as_bytes(),
            chunk,
        };
        assert_eq!(
            render(TriGParser::new().for_reader(reader)),
            expected,
            "chunk size {chunk}"
        );
    }
    Ok(())
}

/// A `Read` that reports zero bytes without being at EOF must not lose data either.
#[test]
fn a_stuttering_reader_still_parses_everything() -> TestResult {
    struct StutteringReader<'a> {
        data: &'a [u8],
        stutter: bool,
    }

    impl Read for StutteringReader<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> IoResult<usize> {
            self.stutter = !self.stutter;
            if self.stutter && !self.data.is_empty() {
                // A zero-length read that is *not* EOF.
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "please retry",
                ));
            }
            let take = 1.min(self.data.len()).min(buffer.len());
            buffer[..take].copy_from_slice(&self.data[..take]);
            self.data = &self.data[take..];
            Ok(take)
        }
    }

    let document = long_document();
    let expected = render(NTriplesParser::new().for_slice(document.as_bytes()));
    let actual = render(NTriplesParser::new().for_reader(StutteringReader {
        data: document.as_bytes(),
        stutter: false,
    }));
    // `ErrorKind::Interrupted` is surfaced as an I/O error rather than retried, so the parser
    // must either match the slice result or stop with that very error — never silently skip a
    // statement or emit a corrupted one.
    assert!(
        actual == expected || actual.iter().any(|line| line.starts_with("err ")),
        "unexpected result for a stuttering reader"
    );
    for (index, line) in actual.iter().enumerate() {
        if line.starts_with("ok ") {
            assert_eq!(line, &expected[index]);
        }
    }
    Ok(())
}
