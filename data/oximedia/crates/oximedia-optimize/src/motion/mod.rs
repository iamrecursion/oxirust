//! Motion estimation optimization module.
//!
//! Advanced motion search algorithms for encoder optimization.

pub mod bidirectional;
pub mod predictor;
pub mod search;
pub mod subpel;

pub use bidirectional::{BiPredResult, BidirectionalOptimizer};
pub use predictor::{MvPredictor, MvPredictorList, TemporalMvPredictor};
pub use search::{
    parallel_motion_search, MotionOptimizer, MotionSearchResult, MotionVector,
    ParallelSearchResult, SearchAlgorithm,
};
pub use subpel::{SubpelOptimizer, SubpelPrecision};
