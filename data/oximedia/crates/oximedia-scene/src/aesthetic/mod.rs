//! Aesthetic quality assessment.

pub mod features;
pub mod score;

pub use features::{AestheticFeatures, FeatureExtractor};
pub use score::{AestheticScore, AestheticScorer, ContentType, ContentTypeScorer};
