//! The functions that convert between XML text and the data model:
//! [`fn:parse-xml`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml),
//! [`fn:parse-xml-fragment`](https://www.w3.org/TR/xpath-functions-31/#func-parse-xml-fragment)
//! and [`fn:serialize`](https://www.w3.org/TR/xpath-functions-31/#func-serialize).

pub mod parse;
pub mod serialize;

pub use parse::{parse_xml, parse_xml_fragment};
pub use serialize::{
    OutputMethod, SERIALIZATION_NAMESPACE, SerializationParameters, parameters_from_element,
    parameters_from_element_in_context, parameters_from_element_with_defaults, qnames_from_option,
    serialize,
};
