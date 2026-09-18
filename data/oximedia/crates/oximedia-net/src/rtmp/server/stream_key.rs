// ─────────────────────────────────────────────────────────────────────────────
// Stream key validation
// ─────────────────────────────────────────────────────────────────────────────

/// Stream key format requirements.
#[derive(Debug, Clone)]
pub struct StreamKeyPolicy {
    /// Minimum key length.
    pub min_length: usize,
    /// Maximum key length.
    pub max_length: usize,
    /// Allowed characters (None = any printable ASCII).
    pub allowed_chars: Option<String>,
    /// Whether numeric-only keys are rejected.
    pub reject_numeric_only: bool,
    /// Whether empty keys are rejected.
    pub reject_empty: bool,
}

impl Default for StreamKeyPolicy {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: 256,
            allowed_chars: None,
            reject_numeric_only: false,
            reject_empty: true,
        }
    }
}

/// Stream key validator.
#[derive(Debug, Clone, Default)]
pub struct StreamKeyValidator {
    /// Validation policy.
    policy: StreamKeyPolicy,
    /// Known-valid keys (allowlist, empty = allow all).
    allowlist: Vec<String>,
    /// Known-bad key prefixes (denylist).
    denylist_prefixes: Vec<String>,
}

impl StreamKeyValidator {
    /// Creates a new validator with the given policy.
    #[must_use]
    pub fn new(policy: StreamKeyPolicy) -> Self {
        Self {
            policy,
            allowlist: Vec::new(),
            denylist_prefixes: Vec::new(),
        }
    }

    /// Adds a key to the allowlist.  When at least one key is in the
    /// allowlist only explicitly allowed keys are accepted.
    pub fn add_allowed_key(&mut self, key: impl Into<String>) {
        self.allowlist.push(key.into());
    }

    /// Adds a prefix to the denylist.
    pub fn add_denied_prefix(&mut self, prefix: impl Into<String>) {
        self.denylist_prefixes.push(prefix.into());
    }

    /// Validates a stream key.  Returns `Ok(())` or the first validation
    /// failure as an error string.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error if the key violates any policy rule.
    pub fn validate(&self, key: &str) -> Result<(), String> {
        // Empty check
        if self.policy.reject_empty && key.is_empty() {
            return Err("stream key must not be empty".to_string());
        }

        // Length checks
        if key.len() < self.policy.min_length {
            return Err(format!(
                "stream key too short: {} < {}",
                key.len(),
                self.policy.min_length
            ));
        }
        if key.len() > self.policy.max_length {
            return Err(format!(
                "stream key too long: {} > {}",
                key.len(),
                self.policy.max_length
            ));
        }

        // Character allowlist
        if let Some(allowed) = &self.policy.allowed_chars {
            for ch in key.chars() {
                if !allowed.contains(ch) {
                    return Err(format!("stream key contains disallowed character: '{ch}'"));
                }
            }
        } else {
            // Default: printable ASCII only
            for ch in key.chars() {
                if !ch.is_ascii() || ch.is_ascii_control() {
                    return Err(format!("stream key must be printable ASCII; found '{ch}'"));
                }
            }
        }

        // Numeric-only rejection
        if self.policy.reject_numeric_only && key.chars().all(|c| c.is_ascii_digit()) {
            return Err("stream key must not be numeric-only".to_string());
        }

        // Denylist prefixes
        for prefix in &self.denylist_prefixes {
            if key.starts_with(prefix.as_str()) {
                return Err(format!("stream key starts with denied prefix '{prefix}'"));
            }
        }

        // Allowlist (if non-empty)
        if !self.allowlist.is_empty() && !self.allowlist.iter().any(|k| k == key) {
            return Err("stream key not in allowlist".to_string());
        }

        Ok(())
    }

    /// Returns `true` when the key passes all validation rules.
    #[must_use]
    pub fn is_valid(&self, key: &str) -> bool {
        self.validate(key).is_ok()
    }
}
