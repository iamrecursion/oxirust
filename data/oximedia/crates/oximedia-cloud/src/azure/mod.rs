//! Azure cloud services integration

mod blob;
mod media;

pub use blob::AzureBlobStorage;
pub use media::AzureMediaServices;
