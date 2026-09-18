//! RTMP ingest module for live streaming.
//!
//! This module orchestrates RTMP stream ingestion, integrating with
//! the low-level RTMP server from `oximedia-net`.

mod publish;
mod server;
mod session;

pub use publish::{PublishContext, PublishHandler};
pub use server::{RtmpIngestConfig, RtmpIngestServer};
pub use session::{IngestSession, SessionManager};
