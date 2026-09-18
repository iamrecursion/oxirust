//! User cursor tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// User cursor position.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CursorPosition {
    /// X coordinate (normalized 0.0-1.0).
    pub x: f32,
    /// Y coordinate (normalized 0.0-1.0).
    pub y: f32,
}

impl CursorPosition {
    /// Create a new cursor position.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
        }
    }

    /// Calculate distance to another position.
    #[must_use]
    pub fn distance_to(&self, other: &CursorPosition) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// User cursor information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCursor {
    /// User ID.
    pub user_id: String,
    /// User name (for display).
    pub user_name: String,
    /// Current frame.
    pub frame: i64,
    /// Cursor position on the frame.
    pub position: CursorPosition,
    /// Last updated timestamp.
    pub updated_at: DateTime<Utc>,
    /// Cursor color (for display).
    pub color: String,
}

impl UserCursor {
    /// Create a new user cursor.
    #[must_use]
    pub fn new(user_id: String, user_name: String, frame: i64, position: CursorPosition) -> Self {
        Self {
            user_id,
            user_name,
            frame,
            position,
            updated_at: Utc::now(),
            color: "#FF0000".to_string(),
        }
    }

    /// Update cursor position.
    pub fn update_position(&mut self, position: CursorPosition) {
        self.position = position;
        self.updated_at = Utc::now();
    }

    /// Update frame.
    pub fn update_frame(&mut self, frame: i64) {
        self.frame = frame;
        self.updated_at = Utc::now();
    }

    /// Set cursor color.
    pub fn set_color(&mut self, color: String) {
        self.color = color;
    }

    /// Check if cursor is stale (not updated recently).
    #[must_use]
    pub fn is_stale(&self, threshold_seconds: i64) -> bool {
        let elapsed = Utc::now() - self.updated_at;
        elapsed.num_seconds() > threshold_seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_position_creation() {
        let pos = CursorPosition::new(0.5, 0.5);
        assert!((pos.x - 0.5).abs() < 0.001);
        assert!((pos.y - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cursor_position_clamping() {
        let pos = CursorPosition::new(1.5, -0.5);
        assert!((pos.x - 1.0).abs() < 0.001);
        assert!((pos.y - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cursor_position_distance() {
        let pos1 = CursorPosition::new(0.0, 0.0);
        let pos2 = CursorPosition::new(0.3, 0.4);
        let distance = pos1.distance_to(&pos2);
        assert!((distance - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_user_cursor_creation() {
        let position = CursorPosition::new(0.5, 0.5);
        let cursor = UserCursor::new("user-1".to_string(), "User One".to_string(), 100, position);

        assert_eq!(cursor.user_id, "user-1");
        assert_eq!(cursor.frame, 100);
    }

    #[test]
    fn test_user_cursor_update_position() {
        let position = CursorPosition::new(0.5, 0.5);
        let mut cursor =
            UserCursor::new("user-1".to_string(), "User One".to_string(), 100, position);

        let new_position = CursorPosition::new(0.6, 0.7);
        cursor.update_position(new_position);

        assert!((cursor.position.x - 0.6).abs() < 0.001);
        assert!((cursor.position.y - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_user_cursor_update_frame() {
        let position = CursorPosition::new(0.5, 0.5);
        let mut cursor =
            UserCursor::new("user-1".to_string(), "User One".to_string(), 100, position);

        cursor.update_frame(200);
        assert_eq!(cursor.frame, 200);
    }

    #[test]
    fn test_user_cursor_is_stale() {
        let position = CursorPosition::new(0.5, 0.5);
        let cursor = UserCursor::new("user-1".to_string(), "User One".to_string(), 100, position);

        assert!(!cursor.is_stale(60)); // Not stale within 60 seconds
    }
}
