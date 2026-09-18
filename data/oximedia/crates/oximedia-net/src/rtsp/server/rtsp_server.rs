//! RTSP 1.0 server — TCP accept loop and connection dispatch.

use std::sync::Arc;

use tokio::net::TcpListener;

use super::connection::ServerConnection;
use super::registry::MountPointRegistry;
use super::state::RtspServerConfig;
use crate::error::NetError;

/// RTSP 1.0 server.
///
/// Binds a `TcpListener`, accepts connections up to `max_connections`, and
/// spawns a [`ServerConnection`] task for each one.
///
/// # Example
///
/// ```no_run
/// use oximedia_net::rtsp::server::{RtspServer, RtspServerConfig};
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let server = RtspServer::with_default_config();
/// server.run().await?;
/// # Ok(()) }
/// ```
pub struct RtspServer {
    config: Arc<RtspServerConfig>,
    registry: MountPointRegistry,
}

impl RtspServer {
    /// Create a server with explicit configuration.
    #[must_use]
    pub fn new(config: RtspServerConfig) -> Self {
        Self {
            config: Arc::new(config),
            registry: MountPointRegistry::new(),
        }
    }

    /// Create a server with default configuration (`0.0.0.0:554`).
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(RtspServerConfig::default())
    }

    /// Access the mount-point registry to register stream sources.
    #[must_use]
    pub fn registry(&self) -> &MountPointRegistry {
        &self.registry
    }

    /// Run the server, accepting connections indefinitely.
    ///
    /// Returns only when the underlying `TcpListener` fails to `accept()`.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::Connection`] if the listener fails to bind or
    /// an `accept()` error is considered fatal.
    pub async fn run(self) -> Result<(), NetError> {
        let listener = TcpListener::bind(&self.config.bind_address)
            .await
            .map_err(|e| NetError::Connection(format!("bind failed: {e}")))?;
        self.run_with_listener(listener).await
    }

    /// Run the server using an already-bound listener.
    ///
    /// Useful for tests: bind to port 0, obtain the actual port from
    /// `listener.local_addr()?.port()`, then pass the listener here.
    pub async fn run_with_listener(self, listener: TcpListener) -> Result<(), NetError> {
        let mut connection_count: usize = 0;

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    if connection_count >= self.config.max_connections {
                        // Silently drop the stream — connection limit reached.
                        drop(stream);
                        continue;
                    }
                    connection_count += 1;
                    let config = Arc::clone(&self.config);
                    let registry = self.registry.clone();
                    tokio::spawn(async move {
                        ServerConnection::new(stream, config, registry).run().await;
                    });
                }
                Err(e) => {
                    return Err(NetError::Connection(format!("accept failed: {e}")));
                }
            }
        }
    }
}
