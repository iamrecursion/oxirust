# oxixml-xsd

[XML Schema 1.1 Part 2](https://www.w3.org/TR/xmlschema11-2/) datatype value spaces in Pure
Rust — the numeric and temporal atomic types that RDF literals, SPARQL, and XPath/XQuery in
[OxiXML](https://github.com/cool-japan/oxixml) are built on.

The numeric tower is `Boolean`, `Float` (`f32`), `Double` (`f64`), `Decimal` (an `i128`
fixed-point value with 18 fractional digits) and `Integer` (`i64`). The temporal tower is
`DateTime`, `Date`, `Time`, `GYearMonth`, `GYear`, `GMonthDay`, `GMonth`, `GDay`,
`TimezoneOffset`, `Duration`, `YearMonthDuration` and `DayTimeDuration`. Every type provides the
exact XSD lexical mapping via `FromStr` (no implicit whitespace trimming — collapse happens
upstream), the XSD canonical form via `Display`, spec-exact equality/ordering (`NaN` unordered
for `float`/`double`; date/time values without a timezone form a partial order against those with
one, per the ±14:00 rule), and checked arithmetic that returns `None` rather than panicking or
wrapping on overflow.

## Install

```sh
cargo add oxixml-xsd@0.1.2
```

## Usage

```rust
use oxixml_xsd::{DateTime, Decimal, Duration};
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Lexical variants map to the same canonical form.
    assert_eq!(Decimal::from_str("+1.2500")?.to_string(), "1.25");

    // XML Schema Appendix E date arithmetic, with end-of-month clamping.
    let start = DateTime::from_str("2000-01-31T00:00:00Z")?;
    let shifted = start
        .checked_add_duration(Duration::from_str("P1M")?)
        .expect("no overflow");
    assert_eq!(shifted.to_string(), "2000-02-29T00:00:00Z");
    Ok(())
}
```

## Replaces

Pure-Rust drop-in replacement for [`oxsdatatypes`](https://crates.io/crates/oxsdatatypes),
mirroring its public API shape.

## Conformance

Every datatype (`Boolean`, `Decimal`, `Double`, `Float`, `Integer`, `Duration`, and the
`date_time` family) carries its own lexical-mapping, canonical-form, ordering, and checked-
arithmetic unit tests, including XML Schema Appendix E's end-of-month-clamping date arithmetic.

## License

Licensed under the Apache License, Version 2.0 (<http://www.apache.org/licenses/LICENSE-2.0>).
See the `LICENSE` file at the root of the [OxiXML](https://github.com/cool-japan/oxixml)
workspace.
