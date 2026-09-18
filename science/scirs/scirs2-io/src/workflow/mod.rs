//! Auto-generated module structure

#![allow(missing_docs)]

pub mod executorconfig_traits;
pub mod functions;
pub mod functions_2;
pub mod functions_3;
pub mod functions_4;
pub mod notificationconfig_traits;
pub mod retrypolicy_traits;
pub mod types;
pub mod workflowconfig_traits;

// Re-export all types
pub use executorconfig_traits::*;
pub use functions::*;
pub use functions_2::*;
pub use functions_3::*;
pub use functions_4::*;
pub use notificationconfig_traits::*;
pub use retrypolicy_traits::*;
pub use types::*;
pub use workflowconfig_traits::*;

#[cfg(test)]
#[path = "../workflow_tests.rs"]
mod tests;
