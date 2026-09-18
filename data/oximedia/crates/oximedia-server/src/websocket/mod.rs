//! WebSocket support for real-time updates.

pub mod handler;
pub mod manager;

pub use handler::handle_websocket;
pub use manager::{Message, WebSocketManager};
