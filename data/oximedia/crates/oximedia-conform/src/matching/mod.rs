//! Matching strategies for conforming media files to clips.

pub mod bloom;
pub mod content;
pub mod filename;
pub mod strategies;
pub mod timecode;

pub use bloom::{CandidateBloom, CandidatePreFilter, MatchingBloom};
pub use content::{perceptual_hash_match, PerceptualHashMatcher};
pub use strategies::{
    MatchConfidence, MatchStrategy, MatchStrategyKind, WeightedMultiStrategyMatcher,
};
