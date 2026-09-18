//! Parsers and serializers for the SVG attribute microsyntaxes.

pub mod aspect;
pub mod clock;
pub mod color;
pub mod iri;
pub mod length;
pub mod lex;
pub mod paint;
pub mod path;
pub mod points;
pub mod transform;

// Convenience re-exports of the main value types, so a caller who does not
// need the full crate-root curation can still write
// `oxixml_svg::values::TransformList` instead of spelling out the submodule.
// Free-function parsers with an obvious associated type (`Length::parse`,
// `Color::parse`, ...) are not duplicated here; the handful without one
// (`points`, list/lexer helpers) stay reachable through their own submodule.
pub use aspect::{Align, MeetOrSlice, PreserveAspectRatio, ViewBox};
pub use clock::{
    ClockDuration, ClockValue, RepeatCount, SignedClockValue, SyncbaseEdge, TimingSpecifier,
};
pub use color::Color;
pub use iri::FuncIri;
pub use length::{Angle, AngleUnit, Length, LengthContext, LengthUnit, PercentBasis};
pub use paint::{Paint, PaintFallback};
pub use path::{PathData, PathSegment};
pub use transform::{Transform, TransformList};
