# oxixml-testsuite

The workspace-wide W3C/oxigraph conformance harness for
[OxiXML](https://github.com/cool-japan/oxixml). Every crate in the workspace already carries
its own focused test suite; this crate answers the question a per-crate suite can't: *across
the whole workspace*, which W3C/oxigraph conformance suites can we run right now, and what is
the pass rate of each? It is dev-only (`publish = false`) — nothing here is meant to be
depended on by another crate, and it is never published to crates.io.

## Install

Not published — run it from a workspace checkout instead. No suite path is ever hardcoded;
three environment variables name the checkout roots to drive against, and any left unset just
contribute an `unavailable` row rather than failing a build:

```sh
export OXIXML_XML_SUITES_DIR=/path/to/w3c-xml-suites    # xmlconf, xsts, qt3, xslt30, relaxng,
                                                          # schematron, c14n, xmldsig
export OXIXML_TESTSUITE_DIR=/path/to/oxigraph/testsuite  # rdf-tests, rdf-canon, json-ld, sparql
export OXIXML_RDFA_SUITE_DIR=/path/to/rdfa-suite         # optional

cargo run -p oxixml-testsuite --bin oxixml-conformance-report
```

## Usage

The shared report type needs no suite checkout to demonstrate:

```rust
use oxixml_testsuite::report::{ConformanceReport, SuiteReport};

let mut report = SuiteReport::new("demo");
report.record_pass();
report.record_pass();
report.record_fail("case-1", "mismatch");

assert_eq!(report.total, 3);
assert!((report.pass_rate().expect("cases were attempted") - 66.666_666_666).abs() < 1e-6);

let table: ConformanceReport = [report].into_iter().collect();
assert!(table.to_string().contains("demo"));
```

Driving a real suite needs the environment above; `all_suites()` is the entry point both the
`oxixml-conformance-report` binary and `tests/conformance.rs` (behind `#[ignore]`, since a full
run can take several minutes) use.

## Suites covered

`xmlconf`, `xsts` (XML Schema), `qt3` (XPath/XQuery), `xslt30`, `relaxng`, `schematron`, `c14n`,
`xmldsig`, `rdf-tests`, `rdf-canon`, `json-ld`, `sparql`, and `rdfa` — one module per suite
under `src/`, each exposing `run() -> Vec<SuiteReport>`.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
