//! Conforming module for proxy-to-original workflows.

pub mod edl;
pub mod engine;
pub mod mapper;
pub mod relink;
pub mod timeline;
pub mod xml;

pub use edl::EdlConformer;
pub use engine::{ConformEngine, ConformResult};
pub use mapper::{AutoPathMapper, MappingResult, PathMapper};
pub use relink::{RelinkResult, Relinker};
pub use timeline::{
    MediaReference, TimelineConformResult, TimelineConformer, TimelineFormat,
    TimelineFormatDetector, TimelineValidation,
};
pub use xml::{XmlConformer, XmlFormat};
