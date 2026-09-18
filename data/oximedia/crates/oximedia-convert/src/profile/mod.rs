// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion profiles for common use cases.

pub mod custom;
pub mod presets;
pub mod system;

pub use custom::ProfileBuilder;
pub use presets::ProfilePresets;
pub use system::{Profile, ProfileSystem};
