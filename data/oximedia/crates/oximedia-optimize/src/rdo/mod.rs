//! Rate-distortion optimization module.
//!
//! Provides the core RDO engine for encoder optimization.

pub mod cost;
pub mod engine;
pub mod lambda;
pub mod partition_rdo;
pub mod trellis;

pub use cost::{BitCost, CostEstimate, CostMetric};
pub use engine::{ModeCandidate, RdoEngine, RdoResult};
pub use lambda::{LambdaCalculator, LambdaParams};
pub use partition_rdo::{
    rdo_with_early_termination, CachedRdoEngine, EarlyTermConfig, PartitionRdo, PartitionType,
    RdoCacheEntry, RdoConfig,
};
pub use trellis::{RdoqOptimizer, RdoqResult, RunLengthTrellis, TrellisQuantizer};
