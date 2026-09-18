//! WebSocket connection handler.

use super::{Message, WebSocketManager};
use crate::{auth::AuthUser, AppState};
use axum::{
    extract::{
        ws::{WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tokio::time::{interval, Duration};

/// Handles WebSocket upgrade requests.
pub async fn handle_websocket(
    ws: WebSocketUpgrade,
    State(_state): State<Arc<AppState>>,
    auth_user: AuthUser,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, auth_user))
}

/// Handles an active WebSocket connection.
async fn handle_socket(socket: WebSocket, auth_user: AuthUser) {
    let (mut sender, mut receiver) = socket.split();
    let ws_manager = WebSocketManager::new();

    // Register connection
    let mut rx = ws_manager.register(auth_user.user_id.clone());

    // Spawn task to receive messages from the channel and send to client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            if sender
                .send(axum::extract::ws::Message::Text(json.into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    // Spawn task to receive messages from client
    let user_id = auth_user.user_id.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let axum::extract::ws::Message::Text(text) = msg {
                // Handle incoming messages
                if let Ok(message) = serde_json::from_str::<Message>(&text) {
                    match message {
                        Message::Ping => {
                            // Respond with pong
                            tracing::debug!("Received ping from {}", user_id);
                        }
                        _ => {
                            // Ignore other message types from client
                        }
                    }
                }
            } else if let axum::extract::ws::Message::Close(_) = msg {
                break;
            }
        }
    });

    // Spawn keep-alive task
    let user_id = auth_user.user_id.clone();
    let mut keepalive_task = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            tracing::trace!("Sending keepalive to {}", user_id);
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = &mut send_task => {
            tracing::debug!("Send task completed");
            recv_task.abort();
            keepalive_task.abort();
        }
        _ = &mut recv_task => {
            tracing::debug!("Receive task completed");
            send_task.abort();
            keepalive_task.abort();
        }
        _ = &mut keepalive_task => {
            tracing::debug!("Keepalive task completed");
            send_task.abort();
            recv_task.abort();
        }
    }

    // Unregister connection
    ws_manager.unregister(&auth_user.user_id);
    tracing::info!("WebSocket connection closed for user {}", auth_user.user_id);
}
