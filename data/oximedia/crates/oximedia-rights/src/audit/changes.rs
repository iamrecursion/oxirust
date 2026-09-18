//! Change tracking

use crate::audit::AuditEntry;
use std::collections::HashMap;

/// Change tracker for entities
pub struct ChangeTracker {
    entity_type: String,
    entity_id: String,
    changes: HashMap<String, (String, String)>, // field -> (old_value, new_value)
}

impl ChangeTracker {
    /// Create a new change tracker
    pub fn new(entity_type: impl Into<String>, entity_id: impl Into<String>) -> Self {
        Self {
            entity_type: entity_type.into(),
            entity_id: entity_id.into(),
            changes: HashMap::new(),
        }
    }

    /// Track a field change
    pub fn track_change(
        &mut self,
        field: impl Into<String>,
        old_value: impl Into<String>,
        new_value: impl Into<String>,
    ) {
        self.changes
            .insert(field.into(), (old_value.into(), new_value.into()));
    }

    /// Create an audit entry from tracked changes
    pub fn to_audit_entry(self, action: impl Into<String>) -> AuditEntry {
        let mut entry = AuditEntry::new(&self.entity_type, &self.entity_id, action);

        for (field, (old_val, new_val)) in self.changes {
            entry = entry.add_change(format!("{field}_old"), old_val);
            entry = entry.add_change(format!("{field}_new"), new_val);
        }

        entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_tracker() {
        let mut tracker = ChangeTracker::new("grant", "grant123");
        tracker.track_change("status", "pending", "active");
        tracker.track_change("amount", "100", "200");

        let entry = tracker.to_audit_entry("update");
        assert_eq!(entry.entity_type, "grant");
        assert_eq!(entry.changes.len(), 4); // 2 fields * 2 (old/new)
    }
}
