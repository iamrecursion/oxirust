//! AWS cloud services integration

mod media;
mod s3;

pub use media::AwsMediaServices;
pub use s3::S3Storage;
