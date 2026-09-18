//! User presence tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// User presence information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPresence {
    /// User ID.
    pub user_id: String,
    /// Presence status.
    pub status: PresenceStatus,
    /// Current frame being viewed.
    pub current_frame: Option<i64>,
    /// Last active timestamp.
    pub last_active: DateTime<Utc>,
    /// Custom status message.
    pub status_message: Option<String>,
}

/// Presence status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresenceStatus {
    /// User is online and active.
    Online,
    /// User is idle (inactive for a while).
    Idle,
    /// User is offline.
    Offline,
    /// User is busy/in focus mode.
    Busy,
}

impl UserPresence {
    /// Create a new user presence.
    #[must_use]
    pub fn new(user_id: String) -> Self {
        Self {
            user_id,
            status: PresenceStatus::Online,
            current_frame: None,
            last_active: Utc::now(),
            status_message: None,
        }
    }

    /// Update last active timestamp.
    pub fn update_activity(&mut self) {
        self.last_active = Utc::now();
        if self.status == PresenceStatus::Idle {
            self.status = PresenceStatus::Online;
        }
    }

    /// Set the current frame.
    pub fn set_frame(&mut self, frame: i64) {
        self.current_frame = Some(frame);
        self.update_activity();
    }

    /// Set status.
    pub fn set_status(&mut self, status: PresenceStatus) {
        self.status = status;
        self.last_active = Utc::now();
    }

    /// Check if user is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self.status, PresenceStatus::Online | PresenceStatus::Busy)
    }

    /// Get time since last activity.
    #[must_use]
    pub fn time_since_active(&self) -> chrono::Duration {
        Utc::now() - self.last_active
    }
}

impl PresenceStatus {
    /// Get status name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Online => "Online",
            Self::Idle => "Idle",
            Self::Offline => "Offline",
            Self::Busy => "Busy",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_presence_creation() {
        let presence = UserPresence::new("user-1".to_string());
        assert_eq!(presence.status, PresenceStatus::Online);
        assert!(presence.current_frame.is_none());
    }

    #[test]
    fn test_user_presence_update_activity() {
        let mut presence = UserPresence::new("user-1".to_string());
        presence.status = PresenceStatus::Idle;

        presence.update_activity();
        assert_eq!(presence.status, PresenceStatus::Online);
    }

    #[test]
    fn test_user_presence_set_frame() {
        let mut presence = UserPresence::new("user-1".to_string());
        presence.set_frame(100);

        assert_eq!(presence.current_frame, Some(100));
    }

    #[test]
    fn test_user_presence_is_active() {
        let mut presence = UserPresence::new("user-1".to_string());
        assert!(presence.is_active());

        presence.set_status(PresenceStatus::Idle);
        assert!(!presence.is_active());

        presence.set_status(PresenceStatus::Busy);
        assert!(presence.is_active());
    }

    #[test]
    fn test_presence_status_name() {
        assert_eq!(PresenceStatus::Online.name(), "Online");
        assert_eq!(PresenceStatus::Idle.name(), "Idle");
        assert_eq!(PresenceStatus::Offline.name(), "Offline");
        assert_eq!(PresenceStatus::Busy.name(), "Busy");
    }
}
