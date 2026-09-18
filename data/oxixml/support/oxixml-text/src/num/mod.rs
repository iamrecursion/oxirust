//! Floating-point-to-string formatting for two different specifications.
//!
//! * [`ecmascript_double`]/[`ecmascript_float`] implement the ECMAScript
//!   `Number::toString(10)` algorithm (as used, for example, by `JSON.stringify` and
//!   template-literal interpolation of numbers).
//! * [`xsd_canonical_double`]/[`xsd_canonical_float`] implement the XSD 1.1 canonical mapping
//!   for `double`/`float`, always using `M.NNNEexp` exponential form.
//!
//! Both families share `digits::decompose` to obtain the shortest round-tripping decimal
//! digits of a finite, non-zero magnitude (via the standard library's own `{:e}`
//! formatting), then re-assemble those digits according to their own grammar. This
//! guarantees both formatters agree with Rust's `f64`/`f32` `PartialEq` on which values are
//! "the same number" — there is exactly one shortest digit string per value, so there is no
//! risk of the two formatters disagreeing about, say, whether `0.1` and `0.10000000000000001`
//! are the same `f64`.
//!
//! `xsd:decimal`'s canonical mapping is arbitrary-precision and lives in `oxixml-xsd`, not
//! here — this module only covers the binary floating-point datatypes.

mod digits;
mod ecmascript;
mod xsd;

pub use ecmascript::{ecmascript_double, ecmascript_float};
pub use xsd::{xsd_canonical_double, xsd_canonical_float};
