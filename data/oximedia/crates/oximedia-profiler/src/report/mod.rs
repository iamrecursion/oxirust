//! Reporting modules.

pub mod generate;
pub mod html;
pub mod json;
pub mod streaming;

pub use generate::{Report, ReportGenerator};
pub use html::HtmlReporter;
pub use json::JsonReporter;
pub use streaming::{ProfilingEvent, ProfilingEventType, StreamingReporter};
