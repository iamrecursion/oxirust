//! Auto-generated module structure

pub mod chebyshevdistance_traits;
pub mod euclideandistance_traits;
pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod functions_4;
pub mod manhattandistance_traits;
pub mod minkowskidistance_traits;
pub mod types;

// Re-export all types
pub use chebyshevdistance_traits::*;
pub use euclideandistance_traits::*;
pub use functions::*;
pub use functions_2::*;
pub use functions_3::*;
pub use functions_4::*;
pub use manhattandistance_traits::*;
pub use minkowskidistance_traits::*;
pub use types::*;

#[cfg(test)]
#[path = "../distance_tests.rs"]
mod tests;
