//! Documentation generation for preservation

pub mod descriptive;
pub mod generate;
pub mod technical;

pub use descriptive::DescriptiveDocGenerator;
pub use generate::{DocumentationGenerator, DocumentationPackage};
pub use technical::TechnicalDocGenerator;
