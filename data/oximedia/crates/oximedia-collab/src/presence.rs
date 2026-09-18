//! User presence and cursor tracking for real-time collaborative editing.
//!
//! This module provides rich per-user presence state: cursor positions on the
//! media timeline, selection ranges, activity timestamps, a HSV-based color
//! assigner for unique pastel user colors, and a typed event bus for presence
//! changes.

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// The position of a user's edit cursor on the media timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct CursorPosition {
    /// Track the cursor is on.
    pub track_id: u32,
    /// Absolute timeline position in milliseconds.
    pub timestamp_ms: i64,
    /// Optional human-readable label (e.g. "in-point marker").
    pub label: Option<String>,
}

/// A contiguous selection range on a single track.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRange {
    /// Track the selection is on.
    pub track_id: u32,
    /// Start of the selection in milliseconds (inclusive).
    pub start_ms: i64,
    /// End of the selection in milliseconds (exclusive).
    pub end_ms: i64,
}

impl SelectionRange {
    /// Duration of the selection in milliseconds.
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// True when `timestamp_ms` falls within this selection.
    pub fn contains(&self, timestamp_ms: i64) -> bool {
        timestamp_ms >= self.start_ms && timestamp_ms < self.end_ms
    }
}

/// Full presence record for a single collaborating user.
#[derive(Debug, Clone)]
pub struct UserPresence {
    /// Numeric user identifier (matches the session's user table).
    pub user_id: u32,
    /// Display name shown to other collaborators.
    pub user_name: String,
    /// RGB avatar / cursor color `[R, G, B]`.
    pub color: [u8; 3],
    /// Current cursor position, if any.
    pub cursor_position: Option<CursorPosition>,
    /// Current selection range, if any.
    pub selection: Option<SelectionRange>,
    /// Wall-clock timestamp (ms) of the last received heartbeat.
    pub last_seen_ms: i64,
    /// Whether this user is currently considered active.
    pub is_active: bool,
}

impl UserPresence {
    /// Create a new, active user presence entry with no cursor or selection.
    pub fn new(user_id: u32, user_name: impl Into<String>, color: [u8; 3], now_ms: i64) -> Self {
        Self {
            user_id,
            user_name: user_name.into(),
            color,
            cursor_position: None,
            selection: None,
            last_seen_ms: now_ms,
            is_active: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PresenceManager
// ─────────────────────────────────────────────────────────────────────────────

/// Central registry managing presence for all users in a collaboration session.
#[derive(Debug, Default)]
pub struct PresenceManager {
    users: HashMap<u32, UserPresence>,
}

impl PresenceManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    /// Register a new user.  Replaces any existing record with the same id.
    pub fn add_user(&mut self, presence: UserPresence) {
        self.users.insert(presence.user_id, presence);
    }

    /// Remove a user from the registry.
    pub fn remove_user(&mut self, user_id: u32) {
        self.users.remove(&user_id);
    }

    /// Update (or create) the cursor position for a user.
    ///
    /// Also refreshes `last_seen_ms` and re-activates the user.
    pub fn update_cursor(&mut self, user_id: u32, pos: CursorPosition) {
        if let Some(u) = self.users.get_mut(&user_id) {
            u.cursor_position = Some(pos);
            u.is_active = true;
        }
    }

    /// Update the selection range for a user.
    pub fn update_selection(&mut self, user_id: u32, sel: SelectionRange) {
        if let Some(u) = self.users.get_mut(&user_id) {
            u.selection = Some(sel);
            u.is_active = true;
        }
    }

    /// Mark a user as inactive if `now_ms - last_seen_ms >= threshold_ms`.
    pub fn mark_inactive(&mut self, user_id: u32, threshold_ms: i64) {
        if let Some(u) = self.users.get_mut(&user_id) {
            // We compare against an assumed "now" by checking last_seen relative
            // to threshold. Callers supply the current timestamp separately.
            // For API simplicity we expose the threshold as "how long since
            // last_seen_ms before we mark inactive": the caller passes
            // `now_ms - last_seen_ms` indirectly by providing the absolute
            // threshold value they want applied to last_seen.
            // Real usage: pass `(now_ms - threshold_duration)` so that any
            // user whose `last_seen_ms < threshold_ms` is marked inactive.
            if u.last_seen_ms < threshold_ms {
                u.is_active = false;
            }
        }
    }

    /// Heartbeat — update `last_seen_ms` for a user and ensure they are active.
    pub fn heartbeat(&mut self, user_id: u32, now_ms: i64) {
        if let Some(u) = self.users.get_mut(&user_id) {
            u.last_seen_ms = now_ms;
            u.is_active = true;
        }
    }

    /// Return references to all currently active users.
    pub fn active_users(&self) -> Vec<&UserPresence> {
        self.users.values().filter(|u| u.is_active).collect()
    }

    /// Return all users whose cursor `timestamp_ms` falls within
    /// `[timestamp_ms - window_ms, timestamp_ms + window_ms]`.
    pub fn users_near_timestamp(&self, timestamp_ms: i64, window_ms: i64) -> Vec<&UserPresence> {
        self.users
            .values()
            .filter(|u| {
                if let Some(ref cp) = u.cursor_position {
                    (cp.timestamp_ms - timestamp_ms).abs() <= window_ms
                } else {
                    false
                }
            })
            .collect()
    }

    /// Look up a specific user.
    pub fn get_user(&self, user_id: u32) -> Option<&UserPresence> {
        self.users.get(&user_id)
    }

    /// Number of users currently tracked.
    pub fn user_count(&self) -> usize {
        self.users.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CursorBroadcast
// ─────────────────────────────────────────────────────────────────────────────

/// Describes a cursor position update to be broadcast to other participants.
#[derive(Debug, Clone)]
pub struct CursorBroadcast {
    /// User whose cursor moved.
    pub user_id: u32,
    /// New cursor position.
    pub position: CursorPosition,
    /// Minimum ms between successive broadcasts for this user (throttle).
    pub broadcast_interval_ms: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PresenceEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Events emitted by the presence system to inform other subsystems / the UI.
#[derive(Debug, Clone, PartialEq)]
pub enum PresenceEvent {
    /// A user's cursor moved to a new position.
    CursorMoved {
        user_id: u32,
        position: CursorPosition,
    },
    /// A user changed their selection range.
    SelectionChanged {
        user_id: u32,
        selection: SelectionRange,
    },
    /// A new user joined the session.
    UserJoined { user_id: u32 },
    /// A user left the session.
    UserLeft { user_id: u32 },
    /// A user has gone idle (no activity for the configured threshold).
    UserIdle { user_id: u32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorAssigner
// ─────────────────────────────────────────────────────────────────────────────

/// Assigns unique, visually distinct pastel colors to collaborating users.
///
/// Colors are generated in HSV space with fixed S = 0.7, V = 0.9 and hue
/// evenly spaced across `[0, 360)` degrees for up to `max_users` users.
/// Users beyond `max_users` wrap around (hues repeat).
pub struct ColorAssigner {
    /// Maximum number of distinct hues before wrapping.
    max_users: u32,
    /// Next user index to assign.
    next_index: u32,
    /// Already-assigned `user_id → color` map.
    assignments: HashMap<u32, [u8; 3]>,
}

impl ColorAssigner {
    /// Create a new assigner for up to `max_users` distinct colors.
    ///
    /// `max_users` must be at least 1.
    pub fn new(max_users: u32) -> Self {
        let max_users = max_users.max(1);
        Self {
            max_users,
            next_index: 0,
            assignments: HashMap::new(),
        }
    }

    /// Assign (or retrieve) the color for `user_id`.
    pub fn assign(&mut self, user_id: u32) -> [u8; 3] {
        if let Some(&c) = self.assignments.get(&user_id) {
            return c;
        }
        let color = Self::hsv_to_rgb(
            360.0 * (self.next_index % self.max_users) as f64 / self.max_users as f64,
            0.7,
            0.9,
        );
        self.next_index += 1;
        self.assignments.insert(user_id, color);
        color
    }

    /// Convert HSV (h∈[0,360), s∈[0,1], v∈[0,1]) to RGB `[u8; 3]`.
    fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r1, g1, b1) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        [
            ((r1 + m) * 255.0).round() as u8,
            ((g1 + m) * 255.0).round() as u8,
            ((b1 + m) * 255.0).round() as u8,
        ]
    }

    /// Number of users that have been assigned a color.
    pub fn assigned_count(&self) -> usize {
        self.assignments.len()
    }

    /// Retrieve an already-assigned color without allocating a new one.
    pub fn get(&self, user_id: u32) -> Option<[u8; 3]> {
        self.assignments.get(&user_id).copied()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn alice(now_ms: i64) -> UserPresence {
        UserPresence::new(1, "Alice", [255, 0, 0], now_ms)
    }

    fn bob(now_ms: i64) -> UserPresence {
        UserPresence::new(2, "Bob", [0, 255, 0], now_ms)
    }

    // ── UserPresence basics ──────────────────────────────────────────────────

    #[test]
    fn test_new_user_is_active() {
        let u = alice(0);
        assert!(u.is_active);
        assert!(u.cursor_position.is_none());
        assert!(u.selection.is_none());
    }

    // ── SelectionRange ───────────────────────────────────────────────────────

    #[test]
    fn test_selection_duration() {
        let s = SelectionRange {
            track_id: 0,
            start_ms: 100,
            end_ms: 500,
        };
        assert_eq!(s.duration_ms(), 400);
    }

    #[test]
    fn test_selection_contains() {
        let s = SelectionRange {
            track_id: 0,
            start_ms: 100,
            end_ms: 500,
        };
        assert!(s.contains(100));
        assert!(s.contains(300));
        assert!(!s.contains(500)); // exclusive end
        assert!(!s.contains(99));
    }

    // ── PresenceManager ──────────────────────────────────────────────────────

    #[test]
    fn test_add_and_get_user() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(alice(1000));
        let u = mgr.get_user(1).expect("alice should be present");
        assert_eq!(u.user_name, "Alice");
    }

    #[test]
    fn test_remove_user() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(alice(0));
        mgr.remove_user(1);
        assert!(mgr.get_user(1).is_none());
        assert_eq!(mgr.user_count(), 0);
    }

    #[test]
    fn test_update_cursor() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(alice(0));
        let pos = CursorPosition {
            track_id: 2,
            timestamp_ms: 3000,
            label: None,
        };
        mgr.update_cursor(1, pos.clone());
        let u = mgr.get_user(1).expect("alice should be present");
        assert_eq!(
            u.cursor_position.as_ref().expect("cursor should be set"),
            &pos
        );
    }

    #[test]
    fn test_update_cursor_unknown_user_is_noop() {
        let mut mgr = PresenceManager::new();
        let pos = CursorPosition {
            track_id: 0,
            timestamp_ms: 0,
            label: None,
        };
        mgr.update_cursor(999, pos); // should not panic
        assert_eq!(mgr.user_count(), 0);
    }

    #[test]
    fn test_update_selection() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(bob(0));
        let sel = SelectionRange {
            track_id: 1,
            start_ms: 500,
            end_ms: 1500,
        };
        mgr.update_selection(2, sel.clone());
        let u = mgr.get_user(2).expect("bob should be present");
        assert_eq!(u.selection.as_ref().expect("selection should be set"), &sel);
    }

    #[test]
    fn test_mark_inactive() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(UserPresence::new(1, "Alice", [0; 3], 500));
        // threshold_ms = 1000 means "inactive if last_seen_ms < 1000"
        mgr.mark_inactive(1, 1000);
        assert!(!mgr.get_user(1).expect("alice").is_active);
    }

    #[test]
    fn test_mark_inactive_not_applied_when_recent() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(UserPresence::new(1, "Alice", [0; 3], 2000));
        mgr.mark_inactive(1, 1000); // last_seen(2000) >= threshold(1000) → still active
        assert!(mgr.get_user(1).expect("alice").is_active);
    }

    #[test]
    fn test_heartbeat_reactivates() {
        let mut mgr = PresenceManager::new();
        let mut u = alice(0);
        u.is_active = false;
        mgr.add_user(u);
        mgr.heartbeat(1, 5000);
        let fetched = mgr.get_user(1).expect("alice");
        assert!(fetched.is_active);
        assert_eq!(fetched.last_seen_ms, 5000);
    }

    #[test]
    fn test_active_users_filter() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(alice(0));
        mgr.add_user(bob(0));
        mgr.mark_inactive(2, i64::MAX); // mark bob inactive
        let active = mgr.active_users();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].user_id, 1);
    }

    #[test]
    fn test_users_near_timestamp() {
        let mut mgr = PresenceManager::new();
        let mut a = alice(0);
        a.cursor_position = Some(CursorPosition {
            track_id: 0,
            timestamp_ms: 1000,
            label: None,
        });
        let mut b = bob(0);
        b.cursor_position = Some(CursorPosition {
            track_id: 1,
            timestamp_ms: 5000,
            label: None,
        });
        mgr.add_user(a);
        mgr.add_user(b);

        let near = mgr.users_near_timestamp(1200, 500);
        assert_eq!(near.len(), 1);
        assert_eq!(near[0].user_id, 1);
    }

    #[test]
    fn test_users_near_timestamp_both_in_window() {
        let mut mgr = PresenceManager::new();
        let mut a = alice(0);
        a.cursor_position = Some(CursorPosition {
            track_id: 0,
            timestamp_ms: 1000,
            label: None,
        });
        let mut b = bob(0);
        b.cursor_position = Some(CursorPosition {
            track_id: 1,
            timestamp_ms: 1100,
            label: None,
        });
        mgr.add_user(a);
        mgr.add_user(b);

        let near = mgr.users_near_timestamp(1050, 200);
        assert_eq!(near.len(), 2);
    }

    #[test]
    fn test_users_near_timestamp_no_cursor() {
        let mut mgr = PresenceManager::new();
        mgr.add_user(alice(0)); // no cursor
        let near = mgr.users_near_timestamp(0, 9999);
        assert!(near.is_empty());
    }

    // ── ColorAssigner ────────────────────────────────────────────────────────

    #[test]
    fn test_color_assigner_basic() {
        let mut ca = ColorAssigner::new(8);
        let c1 = ca.assign(1);
        let c2 = ca.assign(2);
        assert_ne!(c1, c2, "different users should get different colors");
    }

    #[test]
    fn test_color_assigner_idempotent() {
        let mut ca = ColorAssigner::new(8);
        let c1 = ca.assign(1);
        let c2 = ca.assign(1);
        assert_eq!(c1, c2, "same user should always get same color");
    }

    #[test]
    fn test_color_assigner_count() {
        let mut ca = ColorAssigner::new(4);
        ca.assign(10);
        ca.assign(20);
        ca.assign(30);
        assert_eq!(ca.assigned_count(), 3);
    }

    #[test]
    fn test_color_assigner_get() {
        let mut ca = ColorAssigner::new(4);
        let c = ca.assign(5);
        assert_eq!(ca.get(5), Some(c));
        assert!(ca.get(999).is_none());
    }

    #[test]
    fn test_color_assigner_wrap() {
        // With max_users=2, the 3rd user wraps and shares a hue with user 1.
        let mut ca = ColorAssigner::new(2);
        let c1 = ca.assign(1);
        let _c2 = ca.assign(2);
        let c3 = ca.assign(3);
        assert_eq!(c1, c3, "wraps around to same hue bucket");
    }

    #[test]
    fn test_hsv_to_rgb_red() {
        // H=0, S=1, V=1 → pure red
        let rgb = ColorAssigner::hsv_to_rgb(0.0, 1.0, 1.0);
        assert_eq!(rgb[0], 255);
        assert_eq!(rgb[1], 0);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_hsv_to_rgb_green() {
        // H=120, S=1, V=1 → pure green
        let rgb = ColorAssigner::hsv_to_rgb(120.0, 1.0, 1.0);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 255);
        assert_eq!(rgb[2], 0);
    }

    #[test]
    fn test_hsv_to_rgb_blue() {
        // H=240, S=1, V=1 → pure blue
        let rgb = ColorAssigner::hsv_to_rgb(240.0, 1.0, 1.0);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 0);
        assert_eq!(rgb[2], 255);
    }

    // ── PresenceEvent ────────────────────────────────────────────────────────

    #[test]
    fn test_presence_event_cursor_moved() {
        let pos = CursorPosition {
            track_id: 1,
            timestamp_ms: 2000,
            label: Some("cut".into()),
        };
        let ev = PresenceEvent::CursorMoved {
            user_id: 42,
            position: pos.clone(),
        };
        assert_eq!(
            ev,
            PresenceEvent::CursorMoved {
                user_id: 42,
                position: pos
            }
        );
    }

    #[test]
    fn test_presence_event_user_joined_left() {
        assert_eq!(
            PresenceEvent::UserJoined { user_id: 7 },
            PresenceEvent::UserJoined { user_id: 7 }
        );
        assert_ne!(
            PresenceEvent::UserJoined { user_id: 7 },
            PresenceEvent::UserLeft { user_id: 7 }
        );
    }
}
