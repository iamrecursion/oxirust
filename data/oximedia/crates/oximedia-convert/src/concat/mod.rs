// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! File concatenation and validation.

pub mod join;
pub mod validate;

pub use join::FileJoiner;
pub use validate::CompatibilityValidator;
