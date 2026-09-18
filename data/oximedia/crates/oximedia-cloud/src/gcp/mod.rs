//! Google Cloud Platform services integration

mod gcs;
mod media;

pub use gcs::GcsStorage;
pub use media::GcpMediaServices;
