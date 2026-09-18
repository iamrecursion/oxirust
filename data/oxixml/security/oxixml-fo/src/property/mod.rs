//! The XSL-FO 1.1 property subsystem.
//!
//! The whole formatting property vocabulary of the Recommendation lives here:
//!
//! * [`PropertyId`] names all 272 properties of
//!   [Appendix D](https://www.w3.org/TR/xsl11/#property-index), and
//!   [`PropertyKey`] additionally addresses one component of a compound
//!   datatype (`space-before.optimum`);
//! * [`PropertySpec`] carries the static metadata of each property — whether it
//!   is inherited, its initial value, the datatypes and keywords it accepts,
//!   and what a percentage refers to;
//! * [`datatype`] holds the value datatypes of §5.11 and [`Value`] the computed
//!   values built from them;
//! * [`shorthand`] expands the twenty-four shorthands of §7.31 in the
//!   precedence order of §5.2;
//! * [`WritingMode`] and the corresponding-property tables of §5.3 relate
//!   `start`/`end`/`before`/`after` to `left`/`right`/`top`/`bottom`;
//! * [`RefineOptions`] configures the refinement engine that turns specified
//!   values into computed ones.
//!
//! ```
//! use oxixml_fo::property::{PropertyId, PropertyKey, Value};
//!
//! // Metadata comes straight from the Recommendation's property table.
//! assert!(PropertyId::StartIndent.is_inherited());
//! assert!(!PropertyId::PaddingStart.is_inherited());
//! assert_eq!(PropertyId::parse("space-before"), Some(PropertyId::SpaceBefore));
//!
//! // Compound properties are addressed component by component.
//! let key = PropertyKey::parse("space-before.optimum").expect("a component");
//! assert_eq!(key.id, PropertyId::SpaceBefore);
//! assert_eq!(key.component, Some("optimum"));
//!
//! // Computed values are a closed enumeration.
//! assert!(matches!(Value::keyword("auto"), Value::Keyword(_)));
//! ```

pub(crate) mod applicability;
pub mod corresponding;
pub mod datatype;
mod expr;
mod id;
mod key;
mod lexer;
mod parse;
mod resolve;
pub mod shorthand;
mod spec;
mod table;
mod value;

pub use crate::property::corresponding::{AbsoluteEdge, Direction, RelativeEdge, WritingMode};
pub use crate::property::datatype::{
    Color, Conditionality, Keep, KeepValue, Length, LengthBpIp, LengthConditional, LengthRange,
    Measure, Precedence, Shape, Space, Unit,
};
pub use crate::property::id::{PROPERTY_COUNT, PropertyId};
pub use crate::property::key::PropertyKey;
pub use crate::property::spec::{Compound, PercentBase, PropertySpec, forms};
pub use crate::property::value::Value;

pub use crate::property::resolve::RefineOptions;

pub(crate) use crate::property::resolve::refine;
