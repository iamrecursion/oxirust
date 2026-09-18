use super::*;

/// RTMP server.
pub struct RtmpServer {
    /// Server configuration.
    pub config: RtmpServerConfig,
    /// Active connections.
    connections: Arc<RwLock<HashMap<u64, mpsc::UnboundedSender<OutgoingMessage>>>>,
    /// Next connection ID.
    next_connection_id: Arc<RwLock<u64>>,
    /// Stream registry.
    stream_registry: Arc<StreamRegistry>,
    /// Authentication handler.
    auth_handler: Arc<dyn AuthHandler>,
}

impl RtmpServer {
    /// Creates a new RTMP server.
    #[must_use]
    pub fn new(config: RtmpServerConfig, auth_handler: Arc<dyn AuthHandler>) -> Self {
        Self {
            config,
            connections: Arc::new(RwLock::new(HashMap::new())),
            next_connection_id: Arc::new(RwLock::new(1)),
            stream_registry: Arc::new(StreamRegistry::new()),
            auth_handler,
        }
    }

    /// Creates a new server with default configuration.
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(RtmpServerConfig::default(), Arc::new(AllowAllAuth))
    }

    /// Returns the stream registry.
    #[must_use]
    pub fn stream_registry(&self) -> &Arc<StreamRegistry> {
        &self.stream_registry
    }

    /// Runs the server.
    ///
    /// # Errors
    ///
    /// Returns an error if the server fails to start or bind.
    pub async fn run(&self) -> NetResult<()> {
        let listener = TcpListener::bind(&self.config.bind_address)
            .await
            .map_err(|e| {
                NetError::connection(format!(
                    "Failed to bind to {}: {e}",
                    self.config.bind_address
                ))
            })?;

        loop {
            let (stream, addr) = listener
                .accept()
                .await
                .map_err(|e| NetError::connection(format!("Accept failed: {e}")))?;

            // Check connection limit
            {
                let connections = self.connections.read().await;
                if connections.len() >= self.config.max_connections {
                    continue;
                }
            }

            // Allocate connection ID
            let connection_id = {
                let mut next_id = self.next_connection_id.write().await;
                let id = *next_id;
                *next_id += 1;
                id
            };

            // Spawn connection handler
            let connections = self.connections.clone();
            let config = self.config.clone();
            let stream_registry = Arc::clone(&self.stream_registry);
            let auth_handler = Arc::clone(&self.auth_handler);

            tokio::spawn(async move {
                let conn = ServerConnection::new(
                    connection_id,
                    stream,
                    addr,
                    config,
                    stream_registry,
                    auth_handler,
                );
                let sender = conn.message_sender();

                // Register connection
                {
                    let mut conns = connections.write().await;
                    conns.insert(connection_id, sender);
                }

                // Run connection
                let result = conn.run().await;

                // Unregister connection
                {
                    let mut conns = connections.write().await;
                    conns.remove(&connection_id);
                }

                if let Err(e) = result {
                    tracing::warn!(connection_id = %connection_id, error = %e, "RTMP connection error");
                }
            });
        }
    }

    /// Broadcasts a message to all connections.
    pub async fn broadcast(&self, message: RtmpMessage, csid: u32) -> NetResult<()> {
        let connections = self.connections.read().await;

        for sender in connections.values() {
            let _ = sender.send(OutgoingMessage {
                message: message.clone(),
                chunk_stream_id: csid,
            });
        }

        Ok(())
    }

    /// Sends a message to a specific connection.
    pub async fn send_to_connection(
        &self,
        connection_id: u64,
        message: RtmpMessage,
        csid: u32,
    ) -> NetResult<()> {
        let connections = self.connections.read().await;

        if let Some(sender) = connections.get(&connection_id) {
            sender
                .send(OutgoingMessage {
                    message,
                    chunk_stream_id: csid,
                })
                .map_err(|e| NetError::connection(format!("Failed to send message: {e}")))?;
        }

        Ok(())
    }

    /// Returns the number of active connections.
    pub async fn connection_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }
}

/// RTMP server builder.
pub struct RtmpServerBuilder {
    config: RtmpServerConfig,
    auth_handler: Option<Arc<dyn AuthHandler>>,
}

impl RtmpServerBuilder {
    /// Creates a new server builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: RtmpServerConfig::default(),
            auth_handler: None,
        }
    }

    /// Sets the bind address.
    #[must_use]
    pub fn bind_address(mut self, address: impl Into<String>) -> Self {
        self.config.bind_address = address.into();
        self
    }

    /// Sets the read timeout.
    #[must_use]
    pub const fn read_timeout(mut self, timeout: Duration) -> Self {
        self.config.read_timeout = timeout;
        self
    }

    /// Sets the write timeout.
    #[must_use]
    pub const fn write_timeout(mut self, timeout: Duration) -> Self {
        self.config.write_timeout = timeout;
        self
    }

    /// Sets the chunk size.
    #[must_use]
    pub const fn chunk_size(mut self, size: u32) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Sets the window acknowledgement size.
    #[must_use]
    pub const fn window_ack_size(mut self, size: u32) -> Self {
        self.config.window_ack_size = size;
        self
    }

    /// Sets the maximum number of connections.
    #[must_use]
    pub const fn max_connections(mut self, max: usize) -> Self {
        self.config.max_connections = max;
        self
    }

    /// Sets the authentication handler.
    #[must_use]
    pub fn auth_handler(mut self, handler: Arc<dyn AuthHandler>) -> Self {
        self.auth_handler = Some(handler);
        self
    }

    /// Builds the server.
    #[must_use]
    pub fn build(self) -> RtmpServer {
        let auth_handler = self.auth_handler.unwrap_or_else(|| Arc::new(AllowAllAuth));
        RtmpServer::new(self.config, auth_handler)
    }
}

impl Default for RtmpServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
