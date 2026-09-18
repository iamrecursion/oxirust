//! Exporters for conformed sequences and reports.

pub mod project;
pub mod report;
pub mod sequence;

use crate::error::ConformResult;
use crate::types::OutputFormat;
use std::path::Path;

/// Trait for timeline exporters.
pub trait Exporter {
    /// Export to the specified format.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    fn export<P: AsRef<Path>>(&self, output_path: P, format: OutputFormat) -> ConformResult<()>;
}
