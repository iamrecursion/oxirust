use super::*;

/// This module contains tests where size of the list have a compile-time size
#[path = "fixed_name/fixed_size.rs"]
mod fixed_size;

/// This module contains tests where size of the list have an unspecified size
#[path = "fixed_name/variable_size.rs"]
mod variable_size;
