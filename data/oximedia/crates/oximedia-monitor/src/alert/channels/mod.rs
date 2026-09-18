//! Alert notification channels.

#[cfg(not(target_arch = "wasm32"))]
pub mod discord;
#[cfg(all(not(target_arch = "wasm32"), feature = "email"))]
pub mod email;
#[cfg(not(target_arch = "wasm32"))]
pub mod file;
#[cfg(not(target_arch = "wasm32"))]
pub mod slack;
#[cfg(not(target_arch = "wasm32"))]
pub mod sms;
#[cfg(not(target_arch = "wasm32"))]
pub mod webhook;

#[cfg(not(target_arch = "wasm32"))]
pub use discord::DiscordChannel;
#[cfg(all(not(target_arch = "wasm32"), feature = "email"))]
pub use email::EmailChannel;
#[cfg(not(target_arch = "wasm32"))]
pub use file::FileChannel;
#[cfg(not(target_arch = "wasm32"))]
pub use slack::SlackChannel;
#[cfg(not(target_arch = "wasm32"))]
pub use sms::SmsChannel;
#[cfg(not(target_arch = "wasm32"))]
pub use webhook::WebhookChannel;

use crate::alert::Alert;
use async_trait::async_trait;

/// Alert channel trait.
#[async_trait]
pub trait AlertChannel: Send + Sync {
    /// Send an alert through this channel.
    async fn send(&self, alert: &Alert) -> crate::error::MonitorResult<()>;
}
