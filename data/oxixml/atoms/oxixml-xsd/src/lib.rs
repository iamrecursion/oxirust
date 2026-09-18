//! `oxixml-xsd` — the [XML Schema 1.1 Part 2](https://www.w3.org/TR/xmlschema11-2/)
//! datatype value spaces in Pure Rust.
//!
//! Part of the [OxiXML](https://github.com/cool-japan/oxixml) workspace —
//! the COOLJAPAN Pure-Rust foundation for all XML and RDF processing. This crate
//! is the definitive implementation of the XSD atomic datatypes used throughout
//! the stack (RDF literals, SPARQL, XPath/XQuery), and it is the drop-in
//! replacement for `oxsdatatypes`, mirroring its public API shape.
//!
//! Every datatype provides:
//!
//! * [`FromStr`](std::str::FromStr) implementing the exact XSD *lexical mapping*
//!   (leading/trailing whitespace is **not** trimmed here — collapse happens
//!   upstream);
//! * [`Display`](std::fmt::Display) producing the XSD **canonical** representation;
//! * spec-exact equality and ordering: `double`/`float` treat `NaN` as unordered
//!   and unequal, and date/time values follow the *F&O* relations — a value with
//!   no explicit timezone is read in an [implicit
//!   timezone](https://www.w3.org/TR/xpath-functions-31/#dt-implicit-timezone),
//!   which [`PartialEq`]/[`PartialOrd`] take to be UTC. Pass a different one with
//!   `equal_with_implicit_timezone` / `compare_with_implicit_timezone` /
//!   `checked_sub_with_implicit_timezone`, or ask for the XSD 1.1 *partial* order
//!   (the ±14:00 rule, with its indeterminate cases) via `xsd_eq` /
//!   `xsd_partial_cmp`;
//! * checked arithmetic that returns [`None`] (rather than panicking or wrapping)
//!   on overflow, matching the XPath error semantics.
//!
//! # Reading the clock
//!
//! [`DateTime::now`], [`Date::now`] and [`Time::now`] back `fn:current-dateTime` and
//! friends. Targets without a host clock — `wasm32-unknown-unknown` above all, where
//! [`std::time::SystemTime::now`] traps — must register one with [`set_now_provider`];
//! the `checked_now()` constructors report its absence as [`None`] rather than panicking.
//!
//! The numeric tower is: [`Boolean`], [`Float`] (an [`f32`] wrapper), [`Double`]
//! (an [`f64`] wrapper), [`Decimal`] (an [`i128`] fixed-point value with 18
//! fractional digits) and [`Integer`] (an [`i64`]). The temporal tower is:
//! [`DateTime`], [`Date`], [`Time`], [`GYearMonth`], [`GYear`], [`GMonthDay`],
//! [`GMonth`], [`GDay`], [`TimezoneOffset`], [`Duration`], [`YearMonthDuration`]
//! and [`DayTimeDuration`].
//!
//! # Example
//!
//! ```
//! use oxixml_xsd::{DateTime, Decimal, Duration};
//! use std::str::FromStr;
//!
//! // Lexical variants map to the same canonical form.
//! assert_eq!(Decimal::from_str("+1.2500")?.to_string(), "1.25");
//!
//! // XML Schema Appendix E date arithmetic, with end-of-month clamping.
//! let start = DateTime::from_str("2000-01-31T00:00:00Z")?;
//! let shifted = start
//!     .checked_add_duration(Duration::from_str("P1M")?)
//!     .expect("no overflow");
//! assert_eq!(shifted.to_string(), "2000-02-29T00:00:00Z");
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
#![deny(missing_docs)]

mod boolean;
mod date_time;
mod decimal;
mod double;
mod duration;
mod float;
mod integer;

pub use self::boolean::Boolean;
pub use self::date_time::{
    Date, DateTime, DateTimeOverflowError, GDay, GMonth, GMonthDay, GYear, GYearMonth,
    InvalidTimezoneError, MAX_YEAR, MIN_YEAR, NowProvider, NowProviderAlreadySetError,
    ParseDateTimeError, Time, TimezoneOffset, set_now_provider,
};
pub use self::decimal::{Decimal, ParseDecimalError, TooLargeForDecimalError};
pub use self::double::Double;
pub use self::duration::{
    DayTimeDuration, Duration, DurationOverflowError, OppositeSignInDurationComponentsError,
    ParseDurationError, YearMonthDuration,
};
pub use self::float::Float;
pub use self::integer::{Integer, TooLargeForIntegerError};
