//! Archive retention schedule management.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// RetentionClass
// ---------------------------------------------------------------------------

/// Classification of a retention requirement for an asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RetentionClass {
    /// Temporary — may be deleted after a short window (e.g. proxy files).
    Temporary,
    /// Standard operational retention (e.g. 5 years).
    Standard,
    /// Long-term archival retention (e.g. 10 years).
    LongTerm,
    /// Permanent preservation — never to be deleted.
    Permanent,
}

impl RetentionClass {
    /// Default retention period in years.  `None` = forever (Permanent).
    #[must_use]
    pub fn default_years(&self) -> Option<u32> {
        match self {
            RetentionClass::Temporary => Some(1),
            RetentionClass::Standard => Some(5),
            RetentionClass::LongTerm => Some(10),
            RetentionClass::Permanent => None,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            RetentionClass::Temporary => "Temporary",
            RetentionClass::Standard => "Standard",
            RetentionClass::LongTerm => "Long-Term",
            RetentionClass::Permanent => "Permanent",
        }
    }
}

// ---------------------------------------------------------------------------
// RetentionEntry
// ---------------------------------------------------------------------------

/// Associates an asset with a retention class and optional expiry timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetentionEntry {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Retention class assigned to this asset.
    pub class: RetentionClass,
    /// Unix timestamp (milliseconds) when the asset was ingested.
    pub ingested_at_ms: u64,
    /// Optional explicit expiry timestamp in milliseconds.  Overrides default duration.
    pub expires_at_ms: Option<u64>,
    /// Whether a legal hold is placed on the asset (prevents deletion).
    pub legal_hold: bool,
}

impl RetentionEntry {
    /// Create a new retention entry.
    #[must_use]
    pub fn new(
        asset_id: impl Into<String>,
        class: RetentionClass,
        ingested_at_ms: u64,
        expires_at_ms: Option<u64>,
        legal_hold: bool,
    ) -> Self {
        Self {
            asset_id: asset_id.into(),
            class,
            ingested_at_ms,
            expires_at_ms,
            legal_hold,
        }
    }

    /// Returns `true` if the asset has passed its expiry date *and* no legal hold is active.
    #[must_use]
    pub fn is_eligible_for_deletion(&self, now_ms: u64) -> bool {
        if self.legal_hold {
            return false;
        }
        if self.class == RetentionClass::Permanent {
            return false;
        }
        if let Some(expiry) = self.expires_at_ms {
            return now_ms >= expiry;
        }
        // Fall back to default retention period.
        if let Some(years) = self.class.default_years() {
            let duration_ms = u64::from(years) * 365 * 24 * 3_600_000;
            return now_ms >= self.ingested_at_ms.saturating_add(duration_ms);
        }
        false
    }
}

// ---------------------------------------------------------------------------
// RetentionSchedule
// ---------------------------------------------------------------------------

/// Manages a collection of retention entries.
#[derive(Debug, Default)]
pub struct RetentionSchedule {
    pub(crate) entries: Vec<RetentionEntry>,
}

impl RetentionSchedule {
    /// Create a new, empty schedule.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a retention entry.
    pub fn add(&mut self, entry: RetentionEntry) {
        self.entries.push(entry);
    }

    /// Number of entries in the schedule.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the schedule contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries eligible for deletion at `now_ms`.
    #[must_use]
    pub fn eligible_for_deletion(&self, now_ms: u64) -> Vec<&RetentionEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_eligible_for_deletion(now_ms))
            .collect()
    }

    /// Look up a retention entry by asset ID.
    #[must_use]
    pub fn lookup(&self, asset_id: &str) -> Option<&RetentionEntry> {
        self.entries.iter().find(|e| e.asset_id == asset_id)
    }

    /// All entries with an active legal hold.
    #[must_use]
    pub fn legal_holds(&self) -> Vec<&RetentionEntry> {
        self.entries.iter().filter(|e| e.legal_hold).collect()
    }

    /// All entries belonging to a given class.
    #[must_use]
    pub fn by_class(&self, class: RetentionClass) -> Vec<&RetentionEntry> {
        self.entries.iter().filter(|e| e.class == class).collect()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MS_PER_YEAR: u64 = 365 * 24 * 3_600_000;

    fn make_entry(
        id: &str,
        class: RetentionClass,
        ingested: u64,
        expires: Option<u64>,
        hold: bool,
    ) -> RetentionEntry {
        RetentionEntry::new(id, class, ingested, expires, hold)
    }

    // --- RetentionClass ---

    #[test]
    fn test_retention_class_default_years_temporary() {
        assert_eq!(RetentionClass::Temporary.default_years(), Some(1));
    }

    #[test]
    fn test_retention_class_default_years_standard() {
        assert_eq!(RetentionClass::Standard.default_years(), Some(5));
    }

    #[test]
    fn test_retention_class_default_years_longterm() {
        assert_eq!(RetentionClass::LongTerm.default_years(), Some(10));
    }

    #[test]
    fn test_retention_class_default_years_permanent() {
        assert_eq!(RetentionClass::Permanent.default_years(), None);
    }

    #[test]
    fn test_retention_class_labels() {
        assert_eq!(RetentionClass::Temporary.label(), "Temporary");
        assert_eq!(RetentionClass::Standard.label(), "Standard");
        assert_eq!(RetentionClass::LongTerm.label(), "Long-Term");
        assert_eq!(RetentionClass::Permanent.label(), "Permanent");
    }

    // --- RetentionEntry ---

    #[test]
    fn test_entry_eligible_explicit_expiry_past() {
        let e = make_entry("a1", RetentionClass::Standard, 0, Some(1000), false);
        assert!(e.is_eligible_for_deletion(2000));
    }

    #[test]
    fn test_entry_not_eligible_explicit_expiry_future() {
        let e = make_entry("a2", RetentionClass::Standard, 0, Some(5000), false);
        assert!(!e.is_eligible_for_deletion(2000));
    }

    #[test]
    fn test_entry_legal_hold_prevents_deletion() {
        let e = make_entry("a3", RetentionClass::Standard, 0, Some(1000), true);
        assert!(!e.is_eligible_for_deletion(2000));
    }

    #[test]
    fn test_entry_permanent_never_eligible() {
        let e = make_entry("a4", RetentionClass::Permanent, 0, None, false);
        assert!(!e.is_eligible_for_deletion(u64::MAX));
    }

    #[test]
    fn test_entry_default_duration_expired() {
        // Standard = 5 years; ingest = 0; now = 6 years
        let e = make_entry("a5", RetentionClass::Standard, 0, None, false);
        assert!(e.is_eligible_for_deletion(6 * MS_PER_YEAR));
    }

    #[test]
    fn test_entry_default_duration_not_expired() {
        let e = make_entry("a6", RetentionClass::Standard, 0, None, false);
        assert!(!e.is_eligible_for_deletion(3 * MS_PER_YEAR));
    }

    // --- RetentionSchedule ---

    #[test]
    fn test_schedule_empty() {
        let sched = RetentionSchedule::new();
        assert!(sched.is_empty());
    }

    #[test]
    fn test_schedule_len_after_add() {
        let mut sched = RetentionSchedule::new();
        sched.add(make_entry(
            "x",
            RetentionClass::Standard,
            0,
            Some(1000),
            false,
        ));
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn test_schedule_eligible_for_deletion() {
        let mut sched = RetentionSchedule::new();
        sched.add(make_entry(
            "del-me",
            RetentionClass::Temporary,
            0,
            Some(100),
            false,
        ));
        sched.add(make_entry(
            "keep-me",
            RetentionClass::Standard,
            0,
            Some(9999),
            false,
        ));
        let eligible = sched.eligible_for_deletion(200);
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].asset_id, "del-me");
    }

    #[test]
    fn test_schedule_lookup_found() {
        let mut sched = RetentionSchedule::new();
        sched.add(make_entry(
            "find-me",
            RetentionClass::LongTerm,
            0,
            None,
            false,
        ));
        assert!(sched.lookup("find-me").is_some());
    }

    #[test]
    fn test_schedule_lookup_not_found() {
        let sched = RetentionSchedule::new();
        assert!(sched.lookup("ghost").is_none());
    }

    #[test]
    fn test_schedule_legal_holds() {
        let mut sched = RetentionSchedule::new();
        sched.add(make_entry("held", RetentionClass::Standard, 0, None, true));
        sched.add(make_entry("free", RetentionClass::Standard, 0, None, false));
        assert_eq!(sched.legal_holds().len(), 1);
    }

    #[test]
    fn test_schedule_by_class() {
        let mut sched = RetentionSchedule::new();
        sched.add(make_entry("p1", RetentionClass::Permanent, 0, None, false));
        sched.add(make_entry("s1", RetentionClass::Standard, 0, None, false));
        sched.add(make_entry("p2", RetentionClass::Permanent, 0, None, false));
        assert_eq!(sched.by_class(RetentionClass::Permanent).len(), 2);
        assert_eq!(sched.by_class(RetentionClass::Standard).len(), 1);
    }
}
