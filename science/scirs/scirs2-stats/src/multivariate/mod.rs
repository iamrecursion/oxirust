//! Multivariate statistical analysis methods
//!
//! This module provides implementations of multivariate analysis techniques including:
//! - Principal Component Analysis (PCA)
//! - Factor Analysis (EFA with EM algorithm, varimax/promax rotation)
//! - Linear and Quadratic Discriminant Analysis (LDA/QDA)
//! - Canonical Correlation Analysis (CCA)
//! - Partial Least Squares (PLS)

mod canonical_correlation;
mod discriminant_analysis;
mod enhanced_analysis;
mod factor_analysis;
mod pca;

pub use canonical_correlation::*;
pub use discriminant_analysis::*;
// Import enhanced analysis with aliases to avoid conflicts
pub use enhanced_analysis::{
    enhanced_factor_analysis, enhanced_pca, EnhancedFactorAnalysis, EnhancedPCA,
    FactorAnalysisConfig as EnhancedFactorAnalysisConfig,
    FactorAnalysisResult as EnhancedFactorAnalysisResult, PCAAlgorithm as EnhancedPCAAlgorithm,
    PCAConfig as EnhancedPCAConfig, PCAResult as EnhancedPCAResult,
    RotationMethod as EnhancedRotationMethod,
};
// Import standard analysis - this is the main API
pub use factor_analysis::*;
pub use pca::*;

#[allow(unused_imports)]
use crate::error::StatsResult as Result;
#[allow(unused_imports)]
use scirs2_core::{simd_ops::SimdUnifiedOps, validation::*};
