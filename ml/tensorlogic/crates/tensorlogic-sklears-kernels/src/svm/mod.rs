//! Support Vector Machines via Sequential Minimal Optimization.
//!
//! Provides binary/multi-class C-SVM classification ([`svc`]) and
//! ε-insensitive SVR regression ([`svr`]), both using the SMO solver ([`smo`]).
//!
//! ## Algorithms
//!
//! ### SMO (Sequential Minimal Optimization)
//!
//! Platt (1998) decomposes the SVM QP into a sequence of two-variable
//! sub-problems that have closed-form analytic solutions.  Each iteration picks
//! two "violating" dual variables and updates them jointly while holding the
//! rest fixed, maintaining feasibility by design.
//!
//! Keerthi et al. (2001) introduced a more effective working-set selection
//! heuristic (maximise |E_i - E_j|) and improved KKT tolerance handling.
//!
//! ### C-SVC
//!
//! Binary classification via the C-SVM dual.  Multi-class support uses the
//! One-vs-Rest (OvR) strategy: one binary SVC per class, argmax at prediction.
//!
//! ### ε-SVR
//!
//! Regression with an ε-insensitive loss.  The dual is reformulated as a 2N
//! binary SVM problem on an augmented dataset (Smola & Schölkopf 1998),
//! enabling reuse of the SMO machinery.
//!
//! ## References
//!
//! - Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for
//!   Training Support Vector Machines. MSR-TR-98-14.
//! - Keerthi, S.S. et al. (2001). Improvements to Platt's SMO algorithm for
//!   SVM classifier design. Neural Computation 13(3), 637–649.
//! - Smola, A.J., Schölkopf, B. (1998). A tutorial on support vector
//!   regression. NeuroCOLT2 Technical Report NC2-TR-1998-030.
//! - Schölkopf, B., Smola, A.J. (2002). Learning with Kernels. MIT Press.

pub mod smo;
pub mod svc;
pub mod svr;

#[cfg(test)]
mod tests;

pub use smo::SmoConfig;
pub use svc::{SvcFitted, SvcModel};
pub use svr::{SvrFitted, SvrModel};
