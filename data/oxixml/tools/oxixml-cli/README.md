# oxixml-cli

The `oxixml` command-line tool — a single `xmllint`/`xsltproc`/`rapper`-class binary over the
whole [OxiXML](https://github.com/cool-japan/oxixml) workspace. It wires ten subcommands to
the sibling library crates that already implement them; the CLI itself contributes no XML or
RDF parsing logic of its own beyond command-line ergonomics — see the
[`oxixml`](https://docs.rs/oxixml) crate for that.

## Install

```sh
cargo install oxixml-cli
```

This installs a binary named `oxixml` (see `oxixml --help` for the full reference).

## Usage

```sh
$ echo '<a><b/></a>' | oxixml fmt -
<a>
  <b/>
</a>
```

The library surface behind the binary is public too — `run_status` takes the same arguments as
a real invocation and returns the raw exit status, which is what the crate's own tests assert on:

```rust
use oxixml_cli::run_status;

let args = vec!["fmt".to_owned(), "-".to_owned()];
let mut stdin: &[u8] = b"<a><b/></a>";
let mut stdout = Vec::new();
let mut stderr = Vec::new();

let status = run_status(&args, &mut stdin, &mut stdout, &mut stderr);
assert_eq!(status, 0);
assert_eq!(String::from_utf8(stdout).expect("utf8 output"), "<a>\n  <b/>\n</a>\n");
```

## Subcommands

| Subcommand | What it does | Backed by |
|---|---|---|
| `fmt` | Pretty-print / canonically re-indent an XML document | `oxixml-dom` |
| `lint` | Well-formedness check, `line:col` diagnostics | `oxixml-sax` |
| `validate` | Validate against a DTD, XSD, RELAX NG or Schematron schema | `oxixml-dtd`, `oxixml-schema`, `oxixml-relaxng`, `oxixml-schematron` |
| `transform` | Apply an XSLT stylesheet | `oxixml-xslt` |
| `query` | Evaluate an XPath expression or an XQuery query | `oxixml-xpath`, `oxixml-xquery` |
| `canon` | XML canonicalization, or RDF dataset canonicalization (RDFC-1.0) | `oxixml-c14n`, `oxixml-canon` |
| `convert` | RDF format conversion, or SPARQL results format conversion | `oxixml-io`, `oxixml-sparql-results` |
| `hdt` | Build, dump, or report statistics about an HDT file | `oxixml-hdt` |
| `sparql` | Parse a SPARQL query and print its algebra (SSE) | `oxixml-sparql-syntax` |
| `svg` | SVG-aware lint, statistics, and pretty/minified re-serialization | `oxixml-svg` |

Exit codes: `0` success, `1` the command line itself was invalid, `2` the input/schema/
stylesheet could not be read or processed, `3` the check completed and found the input invalid
or not well-formed. Run `oxixml <subcommand> --help` for a subcommand's full reference.

## What it replaces

A Pure-Rust stand-in for the `xmllint` / `xsltproc` / `rapper`-class tools built on libxml2,
libxslt and raptor.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](../../LICENSE) or
<http://www.apache.org/licenses/LICENSE-2.0>).
