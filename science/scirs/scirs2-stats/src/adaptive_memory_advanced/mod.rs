//! Auto-generated module structure

pub mod accesspatternconfig_traits;
pub mod adaptivememoryconfig_traits;
pub mod adaptivememorymanager_traits;
pub mod alertcondition_traits;
pub mod cacheoptimizationconfig_traits;
pub mod compressionconfig_traits;
pub mod emergencyresponseconfig_traits;
pub mod featureextractionconfig_traits;
pub mod filesystemconfig_traits;
pub mod functions;
pub mod garbagecollectionconfig_traits;
pub mod gcperformancetuning_traits;
pub mod gctriggerconditions_traits;
pub mod gcworkloadawareness_traits;
pub mod memoryperformancemetrics_traits;
pub mod memorypressureconfig_traits;
pub mod numaconfig_traits;
pub mod outofcoreconfig_traits;
pub mod predictiveconfig_traits;
pub mod prefetchconfig_traits;
pub mod pressurethresholds_traits;
pub mod responsestrategies_traits;
pub mod storageconfig_traits;
pub mod types;

// Re-export all types
pub use accesspatternconfig_traits::*;
pub use adaptivememoryconfig_traits::*;
pub use adaptivememorymanager_traits::*;
pub use alertcondition_traits::*;
pub use cacheoptimizationconfig_traits::*;
pub use compressionconfig_traits::*;
pub use emergencyresponseconfig_traits::*;
pub use featureextractionconfig_traits::*;
pub use filesystemconfig_traits::*;
pub use functions::*;
pub use garbagecollectionconfig_traits::*;
pub use gcperformancetuning_traits::*;
pub use gctriggerconditions_traits::*;
pub use gcworkloadawareness_traits::*;
pub use memoryperformancemetrics_traits::*;
pub use memorypressureconfig_traits::*;
pub use numaconfig_traits::*;
pub use outofcoreconfig_traits::*;
pub use predictiveconfig_traits::*;
pub use prefetchconfig_traits::*;
pub use pressurethresholds_traits::*;
pub use responsestrategies_traits::*;
pub use storageconfig_traits::*;
pub use types::*;

#[cfg(test)]
#[path = "../adaptive_memory_advanced_tests.rs"]
mod tests;
