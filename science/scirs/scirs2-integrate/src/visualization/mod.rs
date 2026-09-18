//! Visualization utilities for numerical integration and specialized solvers
//!
//! This module provides tools for visualizing results from various solvers,
//! including phase space plots, bifurcation diagrams, and field visualizations.

pub mod advanced;
pub mod engine;
pub mod error_viz;
pub mod interactive;
pub mod specialized;
pub mod types;
pub mod utils;

// Re-export all public types for backward compatibility
pub use types::{
    AnimationSettings, AttractorInfo, AttractorStability, BifurcationDiagram, ClusteringMethod,
    ColorScheme, ConvergenceCurve, ConvergencePlot, DimensionReductionMethod,
    ErrorDistributionPlot, ErrorStatistics, ErrorType, ErrorVisualizationOptions,
    ExplorationMethod, ExplorationMetrics, FluidState, FluidState3D, HeatMapPlot,
    HighDimensionalPlot, InteractivePlotControls, MultiMetricConvergencePlot, OutputFormat,
    ParameterExplorationPlot, ParameterExplorationResult, ParameterRegion, PhaseDensityPlot,
    PhaseSpace3D, PhaseSpacePlot, PlotMetadata, PlotStatistics, RealTimeBifurcationPlot,
    SensitivityPlot, StepSizeAnalysisPlot, SurfacePlot, VectorFieldPlot,
};

// Re-export from engine module
pub use engine::VisualizationEngine;

// Re-export from utils module
pub use utils::{generate_colormap, optimal_grid_resolution, plot_statistics};

// Re-export from interactive module
pub use interactive::{BifurcationDiagramGenerator, InteractiveParameterExplorer};

// Re-export from advanced module
pub use advanced::{
    advanced_interactive_3d, advanced_visualization, AnimatedVisualizer, MultiDimensionalVisualizer,
};

// Re-export from specialized module
pub use specialized::{
    specialized_visualizations, FinanceVisualizer, FluidVisualizer, QuantumVisualizer,
};

// Re-export from error_viz module
pub use error_viz::{
    ConvergenceInfo, ConvergenceVisualizationEngine, ConvergenceVisualizer,
    ErrorVisualizationEngine, MetricStatistics, PerformanceTracker,
};
