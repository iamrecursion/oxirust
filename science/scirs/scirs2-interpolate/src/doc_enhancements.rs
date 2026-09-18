//! Documentation enhancements and complexity analysis for interpolation methods
//!
//! This module provides comprehensive documentation utilities including:
//! - Computational complexity analysis
//! - Memory complexity analysis  
//! - Performance characteristics
//! - Usage recommendations
//! - Method comparison guides

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Computational complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputationalComplexity {
    /// O(1) - Constant time
    Constant,
    /// O(log n) - Logarithmic time
    Logarithmic,
    /// O(n) - Linear time
    Linear,
    /// O(n log n) - Linearithmic time
    Linearithmic,
    /// O(n²) - Quadratic time
    Quadratic,
    /// O(n³) - Cubic time
    Cubic,
    /// O(2^n) - Exponential time
    Exponential,
}

impl Display for ComputationalComplexity {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ComputationalComplexity::Constant => write!(f, "O(1)"),
            ComputationalComplexity::Logarithmic => write!(f, "O(log n)"),
            ComputationalComplexity::Linear => write!(f, "O(n)"),
            ComputationalComplexity::Linearithmic => write!(f, "O(n log n)"),
            ComputationalComplexity::Quadratic => write!(f, "O(n²)"),
            ComputationalComplexity::Cubic => write!(f, "O(n³)"),
            ComputationalComplexity::Exponential => write!(f, "O(2^n)"),
        }
    }
}

/// Memory complexity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryComplexity {
    /// O(1) - Constant space
    Constant,
    /// O(n) - Linear space
    Linear,
    /// O(n²) - Quadratic space
    Quadratic,
    /// O(n³) - Cubic space
    Cubic,
}

impl Display for MemoryComplexity {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            MemoryComplexity::Constant => write!(f, "O(1)"),
            MemoryComplexity::Linear => write!(f, "O(n)"),
            MemoryComplexity::Quadratic => write!(f, "O(n²)"),
            MemoryComplexity::Cubic => write!(f, "O(n³)"),
        }
    }
}

/// Performance characteristics of an interpolation method
#[derive(Debug, Clone)]
pub struct PerformanceCharacteristics {
    /// Time complexity for construction/fitting
    pub construction_complexity: ComputationalComplexity,
    /// Time complexity for evaluation at a single point
    pub evaluation_complexity: ComputationalComplexity,
    /// Time complexity for batch evaluation at m points
    pub batch_evaluation_complexity: ComputationalComplexity,
    /// Memory complexity
    pub memory_complexity: MemoryComplexity,
    /// Whether the method is suitable for real-time applications
    pub real_time_suitable: bool,
    /// Whether the method supports parallel evaluation
    pub parallel_evaluation: bool,
    /// Whether the method is numerically stable
    pub numerically_stable: bool,
    /// Typical accuracy level (relative error)
    pub typical_accuracy: f64,
}

/// Usage recommendation for interpolation methods
#[derive(Debug, Clone)]
pub struct UsageRecommendation {
    /// Recommended data size range
    pub recommended_data_size: (usize, Option<usize>),
    /// Best use cases
    pub best_use_cases: Vec<String>,
    /// When to avoid this method
    pub avoid_when: Vec<String>,
    /// Alternative methods to consider
    pub alternatives: Vec<String>,
    /// Configuration tips
    pub configuration_tips: Vec<String>,
}

/// Comprehensive method documentation
#[derive(Debug, Clone)]
pub struct MethodDocumentation {
    /// Method name
    pub method_name: String,
    /// Brief description
    pub description: String,
    /// Performance characteristics
    pub performance: PerformanceCharacteristics,
    /// Usage recommendations
    pub usage: UsageRecommendation,
    /// Mathematical background
    pub mathematical_background: String,
    /// Implementation notes
    pub implementation_notes: Vec<String>,
    /// References to papers/algorithms
    pub references: Vec<String>,
}

/// Generate documentation for common interpolation methods
#[allow(dead_code)]
pub fn get_method_documentation(method_name: &str) -> Option<MethodDocumentation> {
    match method_name {
        "linear" => Some(linear_interpolation_docs()),
        "cubic" => Some(cubic_interpolation_docs()),
        "pchip" => Some(pchip_interpolation_docs()),
        "bspline" => Some(bspline_interpolation_docs()),
        "rbf" => Some(rbf_interpolation_docs()),
        "kriging" => Some(kriging_interpolation_docs()),
        "akima" => Some(akima_interpolation_docs()),
        "hermite" => Some(hermite_interpolation_docs()),
        _ => None,
    }
}

/// Linear interpolation documentation
#[allow(dead_code)]
fn linear_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Linear Interpolation".to_string(),
        description: "Piecewise linear interpolation connecting adjacent data points with straight lines".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Linear,
            evaluation_complexity: ComputationalComplexity::Logarithmic,
            batch_evaluation_complexity: ComputationalComplexity::Linearithmic,
            memory_complexity: MemoryComplexity::Linear,
            real_time_suitable: true,
            parallel_evaluation: true,
            numerically_stable: true,
            typical_accuracy: 1e-2,
        },
        usage: UsageRecommendation {
            recommended_data_size: (2, Some(1_000_000)),
            best_use_cases: vec![
                "Real-time applications".to_string(),
                "Large datasets".to_string(),
                "When simplicity is required".to_string(),
                "Preserving monotonicity".to_string(),
            ],
            avoid_when: vec![
                "High accuracy requirements".to_string(),
                "Smooth derivatives needed".to_string(),
                "Data has high curvature".to_string(),
            ],
            alternatives: vec![
                "Cubic splines for smoothness".to_string(),
                "PCHIP for shape preservation".to_string(),
                "Akima for outlier robustness".to_string(),
            ],
            configuration_tips: vec![
                "Ensure input data is sorted".to_string(),
                "Consider preprocessing for noise reduction".to_string(),
            ],
        },
        mathematical_background: "Given points (x₀,y₀)...(xₙ,yₙ), interpolates using f(x) = y₀ + (y₁-y₀)*(x-x₀)/(x₁-x₀) for x ∈ [x₀,x₁]".to_string(),
        implementation_notes: vec![
            "Uses binary search for interval location".to_string(),
            "Supports extrapolation with constant or linear modes".to_string(),
            "Vectorized implementation available for batch evaluation".to_string(),
        ],
        references: vec![
            "Burden, R.L., Faires, J.D. (2010). Numerical Analysis, 9th Edition".to_string(),
        ],
    }
}

/// Cubic spline interpolation documentation
#[allow(dead_code)]
fn cubic_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Cubic Spline Interpolation".to_string(),
        description: "Smooth piecewise cubic polynomial interpolation with continuous first and second derivatives".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Linear,
            evaluation_complexity: ComputationalComplexity::Logarithmic,
            batch_evaluation_complexity: ComputationalComplexity::Linearithmic,
            memory_complexity: MemoryComplexity::Linear,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: true,
            typical_accuracy: 1e-6,
        },
        usage: UsageRecommendation {
            recommended_data_size: (4, Some(100_000)),
            best_use_cases: vec![
                "Smooth curve fitting".to_string(),
                "Derivative estimation".to_string(),
                "Integration applications".to_string(),
                "Animation and graphics".to_string(),
            ],
            avoid_when: vec![
                "Real-time constraints".to_string(),
                "Non-smooth underlying functions".to_string(),
                "Presence of outliers".to_string(),
            ],
            alternatives: vec![
                "Linear for speed".to_string(),
                "PCHIP for shape preservation".to_string(),
                "B-splines for more control".to_string(),
            ],
            configuration_tips: vec![
                "Choose appropriate boundary conditions".to_string(),
                "Natural boundaries for unknown derivatives".to_string(),
                "Clamped boundaries when derivatives are known".to_string(),
            ],
        },
        mathematical_background: "Constructs cubic polynomials S(x) = aᵢ + bᵢ(x-xᵢ) + cᵢ(x-xᵢ)² + dᵢ(x-xᵢ)³ with C² continuity".to_string(),
        implementation_notes: vec![
            "Solves tridiagonal system for coefficients".to_string(),
            "Supports multiple boundary conditions".to_string(),
            "Provides derivative and integral methods".to_string(),
        ],
        references: vec![
            "de Boor, C. (1978). A Practical Guide to Splines".to_string(),
            "Burden, R.L., Faires, J.D. (2010). Numerical Analysis, 9th Edition".to_string(),
        ],
    }
}

/// PCHIP interpolation documentation
#[allow(dead_code)]
fn pchip_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)".to_string(),
        description: "Shape-preserving cubic interpolation that maintains monotonicity and avoids overshooting".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Linear,
            evaluation_complexity: ComputationalComplexity::Logarithmic,
            batch_evaluation_complexity: ComputationalComplexity::Linearithmic,
            memory_complexity: MemoryComplexity::Linear,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: true,
            typical_accuracy: 1e-4,
        },
        usage: UsageRecommendation {
            recommended_data_size: (3, Some(50_000)),
            best_use_cases: vec![
                "Monotonic data".to_string(),
                "Avoiding oscillations".to_string(),
                "Financial data interpolation".to_string(),
                "Physical measurements".to_string(),
            ],
            avoid_when: vec![
                "High-frequency oscillations needed".to_string(),
                "Maximum smoothness required".to_string(),
            ],
            alternatives: vec![
                "Cubic splines for smoothness".to_string(),
                "Akima for outlier robustness".to_string(),
                "Linear for simplicity".to_string(),
            ],
            configuration_tips: vec![
                "Works best with monotonic data".to_string(),
                "No additional parameters needed".to_string(),
            ],
        },
        mathematical_background: "Uses Fritsch-Carlson method to compute derivatives that preserve shape, then builds cubic Hermite polynomials".to_string(),
        implementation_notes: vec![
            "Automatically computes shape-preserving derivatives".to_string(),
            "Handles flat regions gracefully".to_string(),
            "No oscillations near extrema".to_string(),
        ],
        references: vec![
            "Fritsch, F.N., Carlson, R.E. (1980). Monotone Piecewise Cubic Interpolation".to_string(),
        ],
    }
}

/// B-spline interpolation documentation
#[allow(dead_code)]
fn bspline_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "B-spline Interpolation".to_string(),
        description: "Flexible spline interpolation using basis functions with adjustable degree and knot placement".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Linear,
            evaluation_complexity: ComputationalComplexity::Constant,
            batch_evaluation_complexity: ComputationalComplexity::Linear,
            memory_complexity: MemoryComplexity::Linear,
            real_time_suitable: true,
            parallel_evaluation: true,
            numerically_stable: true,
            typical_accuracy: 1e-8,
        },
        usage: UsageRecommendation {
            recommended_data_size: (5, Some(1_000_000)),
            best_use_cases: vec![
                "CAD/CAM applications".to_string(),
                "Computer graphics".to_string(),
                "Signal processing".to_string(),
                "Large datasets".to_string(),
            ],
            avoid_when: vec![
                "Simple linear relationships".to_string(),
                "When knot placement is unclear".to_string(),
            ],
            alternatives: vec![
                "Cubic splines for automatic knot placement".to_string(),
                "NURBS for rational curves".to_string(),
            ],
            configuration_tips: vec![
                "Choose degree based on smoothness needs".to_string(),
                "Use uniform knots for regular data".to_string(),
                "Consider least-squares fitting for noisy data".to_string(),
            ],
        },
        mathematical_background: "Constructs splines as linear combinations of B-spline basis functions: S(x) = Σ cᵢ Nᵢ,ₖ(x)".to_string(),
        implementation_notes: vec![
            "Uses de Boor's algorithm for evaluation".to_string(),
            "Supports arbitrary degree and knot vectors".to_string(),
            "Efficient derivative computation".to_string(),
        ],
        references: vec![
            "de Boor, C. (1978). A Practical Guide to Splines".to_string(),
            "Piegl, L., Tiller, W. (1997). The NURBS Book".to_string(),
        ],
    }
}

/// RBF interpolation documentation
#[allow(dead_code)]
fn rbf_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Radial Basis Function (RBF) Interpolation".to_string(),
        description: "Meshfree interpolation method using radially symmetric basis functions centered at data points".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Cubic,
            evaluation_complexity: ComputationalComplexity::Linear,
            batch_evaluation_complexity: ComputationalComplexity::Quadratic,
            memory_complexity: MemoryComplexity::Quadratic,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: false, // Can be ill-conditioned
            typical_accuracy: 1e-10,
        },
        usage: UsageRecommendation {
            recommended_data_size: (5, Some(5_000)),
            best_use_cases: vec![
                "Scattered data interpolation".to_string(),
                "Multivariate interpolation".to_string(),
                "Irregular grids".to_string(),
                "High accuracy requirements".to_string(),
            ],
            avoid_when: vec![
                "Large datasets (>10,000 points)".to_string(),
                "Real-time applications".to_string(),
                "Ill-conditioned data".to_string(),
            ],
            alternatives: vec![
                "Fast RBF for large datasets".to_string(),
                "Kriging for uncertainty quantification".to_string(),
                "Moving least squares for local approximation".to_string(),
            ],
            configuration_tips: vec![
                "Choose shape parameter carefully".to_string(),
                "Use regularization for stability".to_string(),
                "Consider conditioning for large datasets".to_string(),
            ],
        },
        mathematical_background: "Approximates f(x) = Σ λᵢ φ(||x - xᵢ||) where φ is the RBF and λᵢ are coefficients".to_string(),
        implementation_notes: vec![
            "Solves dense linear system Aλ = f".to_string(),
            "Multiple kernel options available".to_string(),
            "Supports polynomial augmentation".to_string(),
        ],
        references: vec![
            "Buhmann, M.D. (2003). Radial Basis Functions: Theory and Implementations".to_string(),
            "Fasshauer, G.E. (2007). Meshfree Approximation Methods with MATLAB".to_string(),
        ],
    }
}

/// Kriging interpolation documentation
#[allow(dead_code)]
fn kriging_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Kriging (Gaussian Process Regression)".to_string(),
        description: "Statistical interpolation method that provides uncertainty estimates along with predictions".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Cubic,
            evaluation_complexity: ComputationalComplexity::Linear,
            batch_evaluation_complexity: ComputationalComplexity::Quadratic,
            memory_complexity: MemoryComplexity::Quadratic,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: false, // Can be ill-conditioned
            typical_accuracy: 1e-8,
        },
        usage: UsageRecommendation {
            recommended_data_size: (5, Some(3_000)),
            best_use_cases: vec![
                "Uncertainty quantification".to_string(),
                "Noisy data".to_string(),
                "Spatial statistics".to_string(),
                "Optimization (Bayesian)".to_string(),
            ],
            avoid_when: vec![
                "Large datasets".to_string(),
                "Deterministic data".to_string(),
                "Real-time requirements".to_string(),
            ],
            alternatives: vec![
                "Fast kriging for larger datasets".to_string(),
                "RBF for deterministic interpolation".to_string(),
                "Enhanced kriging for better conditioning".to_string(),
            ],
            configuration_tips: vec![
                "Choose covariance function based on data characteristics".to_string(),
                "Use nugget effect for noisy data".to_string(),
                "Consider anisotropy for directional data".to_string(),
            ],
        },
        mathematical_background: "Models data as realization of Gaussian process: f(x) ~ GP(μ(x), k(x,x')) with covariance function k".to_string(),
        implementation_notes: vec![
            "Provides prediction variance".to_string(),
            "Multiple covariance functions supported".to_string(),
            "Can handle trend functions".to_string(),
        ],
        references: vec![
            "Cressie, N. (1993). Statistics for Spatial Data".to_string(),
            "Rasmussen, C.E., Williams, C.K.I. (2006). Gaussian Processes for Machine Learning".to_string(),
        ],
    }
}

/// Akima interpolation documentation
#[allow(dead_code)]
fn akima_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Akima Spline Interpolation".to_string(),
        description: "Robust piecewise cubic interpolation that reduces oscillations and is less sensitive to outliers".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Linear,
            evaluation_complexity: ComputationalComplexity::Logarithmic,
            batch_evaluation_complexity: ComputationalComplexity::Linearithmic,
            memory_complexity: MemoryComplexity::Linear,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: true,
            typical_accuracy: 1e-5,
        },
        usage: UsageRecommendation {
            recommended_data_size: (5, Some(50_000)),
            best_use_cases: vec![
                "Data with outliers".to_string(),
                "Irregular spacing".to_string(),
                "Avoiding oscillations".to_string(),
                "Geophysical data".to_string(),
            ],
            avoid_when: vec![
                "Maximum smoothness required".to_string(),
                "High-frequency content".to_string(),
            ],
            alternatives: vec![
                "Cubic splines for smoothness".to_string(),
                "PCHIP for monotonicity".to_string(),
                "Robust splines for heavy outliers".to_string(),
            ],
            configuration_tips: vec![
                "No parameters to tune".to_string(),
                "Works well with default settings".to_string(),
            ],
        },
        mathematical_background: "Uses local polynomial fitting with weighted averages to estimate derivatives, reducing oscillations".to_string(),
        implementation_notes: vec![
            "Estimates derivatives using Akima's method".to_string(),
            "Less sensitive to isolated outliers".to_string(),
            "Maintains local shape characteristics".to_string(),
        ],
        references: vec![
            "Akima, H. (1970). A New Method of Interpolation and Smooth Curve Fitting".to_string(),
        ],
    }
}

/// Hermite interpolation documentation
#[allow(dead_code)]
fn hermite_interpolation_docs() -> MethodDocumentation {
    MethodDocumentation {
        method_name: "Hermite Interpolation".to_string(),
        description: "Polynomial interpolation that matches both function values and derivatives at specified points".to_string(),
        performance: PerformanceCharacteristics {
            construction_complexity: ComputationalComplexity::Quadratic,
            evaluation_complexity: ComputationalComplexity::Linear,
            batch_evaluation_complexity: ComputationalComplexity::Quadratic,
            memory_complexity: MemoryComplexity::Quadratic,
            real_time_suitable: false,
            parallel_evaluation: true,
            numerically_stable: false, // High-degree polynomials
            typical_accuracy: 1e-12,
        },
        usage: UsageRecommendation {
            recommended_data_size: (2, Some(1_000)),
            best_use_cases: vec![
                "Known derivatives".to_string(),
                "Boundary value problems".to_string(),
                "Animation interpolation".to_string(),
                "Small datasets".to_string(),
            ],
            avoid_when: vec![
                "Large datasets".to_string(),
                "Unknown derivatives".to_string(),
                "Risk of oscillations".to_string(),
            ],
            alternatives: vec![
                "Cubic splines for large datasets".to_string(),
                "PCHIP for automatic derivatives".to_string(),
                "B-splines for better conditioning".to_string(),
            ],
            configuration_tips: vec![
                "Provide accurate derivatives".to_string(),
                "Consider piecewise approach for large intervals".to_string(),
            ],
        },
        mathematical_background: "Constructs polynomial of degree 2n-1 to interpolate n points with specified derivatives".to_string(),
        implementation_notes: vec![
            "Supports cubic and quintic variants".to_string(),
            "Multiple derivative specification methods".to_string(),
            "Direct polynomial evaluation".to_string(),
        ],
        references: vec![
            "Burden, R.L., Faires, J.D. (2010). Numerical Analysis, 9th Edition".to_string(),
        ],
    }
}

/// Print comprehensive method comparison
#[allow(dead_code)]
pub fn print_method_comparison() {
    let methods = [
        "linear", "cubic", "pchip", "bspline", "rbf", "kriging", "akima", "hermite",
    ];

    println!("# Interpolation Methods Comparison\n");

    for method in &methods {
        if let Some(doc) = get_method_documentation(method) {
            println!("## {}\n", doc.method_name);
            println!("{}\n", doc.description);

            println!("**Performance Characteristics:**");
            println!(
                "- Construction: {}",
                doc.performance.construction_complexity
            );
            println!("- Evaluation: {}", doc.performance.evaluation_complexity);
            println!("- Memory: {}", doc.performance.memory_complexity);
            println!(
                "- Real-time suitable: {}",
                doc.performance.real_time_suitable
            );
            println!(
                "- Typical accuracy: {:.0e}\n",
                doc.performance.typical_accuracy
            );

            println!("**Best for:** {}\n", doc.usage.best_use_cases.join(", "));
            println!("**Avoid when:** {}\n", doc.usage.avoid_when.join(", "));
            println!("---\n");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_display() {
        assert_eq!(ComputationalComplexity::Linear.to_string(), "O(n)");
        assert_eq!(ComputationalComplexity::Quadratic.to_string(), "O(n²)");
        assert_eq!(MemoryComplexity::Linear.to_string(), "O(n)");
    }

    #[test]
    fn test_method_documentation_available() {
        assert!(get_method_documentation("linear").is_some());
        assert!(get_method_documentation("cubic").is_some());
        assert!(get_method_documentation("unknown").is_none());
    }
}
