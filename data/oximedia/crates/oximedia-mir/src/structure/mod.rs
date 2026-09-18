//! Structural segmentation and similarity analysis.

pub mod labels;
pub mod segment;
pub mod segmentation;
pub mod similarity;

pub use labels::SectionLabeler;
pub use segment::StructureAnalyzer;
pub use segmentation::{
    MusicSegment, SegmentLabel, SelfSimilarityMatrix, StructureAnalyzer as SegmentationAnalyzer,
    StructureReport,
};
pub use similarity::SimilarityMatrix;
