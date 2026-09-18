//! Server-side session management for the OxiMedia server.
//!
//! Provides an in-memory session store with automatic expiry and a simple
//! key-value data bag per session.  Session IDs are randomly generated
//! 128-bit hex strings.

// ── Session ───────────────────────────────────────────────────────────────────

/// A single server-side session.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// Authenticated user ID, if the session is associated with a user.
    pub user_id: Option<u64>,
    /// Unix timestamp (seconds) when the session was created.
    pub created_at: u64,
    /// Unix timestamp (seconds) of the last activity on this session.
    pub last_active: u64,
    /// Arbitrary string key-value pairs stored in the session.
    pub data: std::collections::HashMap<String, String>,
}

impl Session {
    /// Creates a new session with the given ID.
    ///
    /// Both `created_at` and `last_active` are set to `0`; callers should
    /// pass the real "now" timestamp via `touch` immediately after creation.
    #[allow(dead_code)]
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            user_id: None,
            created_at: 0,
            last_active: 0,
            data: std::collections::HashMap::new(),
        }
    }

    /// Stores a key-value pair in the session data.
    #[allow(dead_code)]
    pub fn set(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), value.to_string());
    }

    /// Retrieves the value for `key` from the session data.
    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }

    /// Returns `true` if the session has not been active within `ttl_s` seconds.
    ///
    /// * `now`   – current Unix timestamp in seconds
    /// * `ttl_s` – session time-to-live in seconds
    #[allow(dead_code)]
    pub fn is_expired(&self, now: u64, ttl_s: u64) -> bool {
        now.saturating_sub(self.last_active) >= ttl_s
    }

    /// Updates `last_active` to `now`, keeping the session alive.
    #[allow(dead_code)]
    pub fn touch(&mut self, now: u64) {
        self.last_active = now;
    }

    /// Returns the age of the session in seconds relative to `now`.
    #[allow(dead_code)]
    pub fn age_s(&self, now: u64) -> u64 {
        now.saturating_sub(self.created_at)
    }

    /// Returns `true` if any data is stored in the session.
    #[allow(dead_code)]
    pub fn has_data(&self) -> bool {
        !self.data.is_empty()
    }

    /// Removes a key from the session data.  Returns the old value if present.
    #[allow(dead_code)]
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Clears all key-value pairs from the session.
    #[allow(dead_code)]
    pub fn clear_data(&mut self) {
        self.data.clear();
    }
}

// ── SessionStore ──────────────────────────────────────────────────────────────

/// In-memory store for [`Session`]s with automatic TTL-based expiry.
#[allow(dead_code)]
#[derive(Debug)]
pub struct SessionStore {
    /// All active sessions indexed by session ID.
    sessions: std::collections::HashMap<String, Session>,
    /// Session time-to-live in seconds.
    ttl_s: u64,
    /// Monotonically increasing counter used for generating unique IDs when
    /// a proper random source is unavailable in tests.
    id_counter: u64,
}

impl SessionStore {
    /// Creates a new session store with the specified TTL.
    #[allow(dead_code)]
    pub fn new(ttl_s: u64) -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
            ttl_s,
            id_counter: 0,
        }
    }

    /// Generates a pseudo-unique session ID.
    ///
    /// In production you would replace this with a cryptographically secure
    /// random ID (e.g., via the `uuid` crate).  Here we combine a counter with
    /// `now` to stay dependency-free.
    fn generate_id(&mut self, now: u64) -> String {
        self.id_counter += 1;
        format!("{:016x}{:016x}", now, self.id_counter)
    }

    /// Creates a new session, stores it, and returns the session ID.
    #[allow(dead_code)]
    pub fn create(&mut self, now: u64) -> String {
        let id = self.generate_id(now);
        let mut session = Session::new(&id);
        session.created_at = now;
        session.last_active = now;
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Retrieves an immutable reference to a session if it exists and is
    /// not expired.  Also touches the session on success.
    #[allow(dead_code)]
    pub fn get(&mut self, id: &str, now: u64) -> Option<&Session> {
        if let Some(session) = self.sessions.get_mut(id) {
            if session.is_expired(now, self.ttl_s) {
                return None;
            }
            session.touch(now);
            // Re-borrow as immutable
            self.sessions.get(id)
        } else {
            None
        }
    }

    /// Retrieves a mutable reference to a session if it exists and is not expired.
    /// Also touches the session on success.
    #[allow(dead_code)]
    pub fn get_mut(&mut self, id: &str, now: u64) -> Option<&mut Session> {
        if let Some(session) = self.sessions.get_mut(id) {
            if session.is_expired(now, self.ttl_s) {
                return None;
            }
            session.touch(now);
            self.sessions.get_mut(id)
        } else {
            None
        }
    }

    /// Removes the session with the given ID.  Returns `true` if it existed.
    #[allow(dead_code)]
    pub fn remove(&mut self, id: &str) -> bool {
        self.sessions.remove(id).is_some()
    }

    /// Removes all expired sessions and returns the count of removed sessions.
    #[allow(dead_code)]
    pub fn purge_expired(&mut self, now: u64) -> usize {
        let ttl = self.ttl_s;
        let before = self.sessions.len();
        self.sessions.retain(|_, s| !s.is_expired(now, ttl));
        before - self.sessions.len()
    }

    /// Returns the number of sessions currently in the store (including expired).
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns `true` if no sessions are stored.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Returns the configured TTL in seconds.
    #[allow(dead_code)]
    pub fn ttl_s(&self) -> u64 {
        self.ttl_s
    }
}

// ── SessionToken ──────────────────────────────────────────────────────────────

/// A session authentication token backed by a hex string.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionToken {
    /// Raw token value (hex-encoded).
    pub value: String,
}

impl SessionToken {
    /// Generates a token derived from `seed` using FNV-1a mixing.
    ///
    /// The result is a 32-hex-character string (128 bits).
    #[allow(dead_code)]
    pub fn generate(seed: u64) -> Self {
        // Mix seed with two FNV-1a passes to get two 64-bit halves.
        let fnv_prime: u64 = 1_099_511_628_211;
        let fnv_offset: u64 = 14_695_981_039_346_656_037;

        let mut h1 = fnv_offset;
        for byte in seed.to_le_bytes() {
            h1 ^= u64::from(byte);
            h1 = h1.wrapping_mul(fnv_prime);
        }
        let mut h2 = h1 ^ 0xDEAD_BEEF_CAFE_BABE_u64;
        for byte in (!seed).to_le_bytes() {
            h2 ^= u64::from(byte);
            h2 = h2.wrapping_mul(fnv_prime);
        }

        Self {
            value: format!("{h1:016x}{h2:016x}"),
        }
    }

    /// Returns `true` if the token value looks like a valid 32-char hex string.
    #[allow(dead_code)]
    pub fn is_valid_format(&self) -> bool {
        self.value.len() == 32 && self.value.chars().all(|c| c.is_ascii_hexdigit())
    }
}

// ── TokenSession ──────────────────────────────────────────────────────────────

/// A richer session that carries a [`SessionToken`] and per-key-value data.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TokenSession {
    /// Authentication token associated with this session.
    pub token: SessionToken,
    /// Identifier of the authenticated user.
    pub user_id: String,
    /// Timestamp (ms) when the session was created.
    pub created_ms: u64,
    /// Timestamp (ms) of the most recent activity.
    pub last_seen_ms: u64,
    /// Arbitrary key-value pairs stored in the session.
    pub data: Vec<(String, String)>,
}

impl TokenSession {
    /// Creates a new token session.
    #[allow(dead_code)]
    pub fn new(token: SessionToken, user_id: &str, now_ms: u64) -> Self {
        Self {
            token,
            user_id: user_id.to_string(),
            created_ms: now_ms,
            last_seen_ms: now_ms,
            data: Vec::new(),
        }
    }

    /// Returns `true` if the session has exceeded `ttl_ms` since `last_seen_ms`.
    #[allow(dead_code)]
    pub fn is_expired(&self, now_ms: u64, ttl_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_seen_ms) >= ttl_ms
    }

    /// Updates `last_seen_ms` to `now_ms`.
    #[allow(dead_code)]
    pub fn touch(&mut self, now_ms: u64) {
        self.last_seen_ms = now_ms;
    }

    /// Sets or updates `key` in the session data bag.
    #[allow(dead_code)]
    pub fn set_data(&mut self, key: &str, value: &str) {
        if let Some(entry) = self.data.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value.to_string();
        } else {
            self.data.push((key.to_string(), value.to_string()));
        }
    }

    /// Returns the value for `key`, or `None` if absent.
    #[allow(dead_code)]
    pub fn get_data(&self, key: &str) -> Option<&str> {
        self.data
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Token-based session store.
#[allow(dead_code)]
#[derive(Debug)]
pub struct TokenSessionStore {
    /// Active sessions.
    sessions: Vec<TokenSession>,
    /// Maximum number of sessions to retain.
    max_sessions: usize,
    /// Counter for generating unique seeds when creating new tokens.
    seed_counter: u64,
}

impl TokenSessionStore {
    /// Creates a new store with the given session cap.
    #[allow(dead_code)]
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: Vec::new(),
            max_sessions,
            seed_counter: 0,
        }
    }

    /// Creates a new session for `user_id` and returns a reference to it.
    ///
    /// If the store is full, the oldest session is evicted first.
    #[allow(dead_code)]
    pub fn create(&mut self, user_id: &str, now_ms: u64) -> &TokenSession {
        if self.sessions.len() >= self.max_sessions && !self.sessions.is_empty() {
            self.sessions.remove(0);
        }
        self.seed_counter += 1;
        let seed = now_ms ^ (self.seed_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let token = SessionToken::generate(seed);
        let session = TokenSession::new(token, user_id, now_ms);
        self.sessions.push(session);
        // Indexing the element we just pushed rather than `.last().expect(..)`
        // avoids a panicking API while remaining equally infallible: the
        // vector's length grew by exactly one on the line above.
        let last_idx = self.sessions.len() - 1;
        &self.sessions[last_idx]
    }

    /// Returns a reference to the session with the given token value.
    #[allow(dead_code)]
    pub fn get(&self, token: &str) -> Option<&TokenSession> {
        self.sessions.iter().find(|s| s.token.value == token)
    }

    /// Touches the session with the given token.  Returns `true` if found.
    #[allow(dead_code)]
    pub fn touch(&mut self, token: &str, now_ms: u64) -> bool {
        if let Some(s) = self.sessions.iter_mut().find(|s| s.token.value == token) {
            s.touch(now_ms);
            true
        } else {
            false
        }
    }

    /// Removes the session with the given token.  Returns `true` if found.
    #[allow(dead_code)]
    pub fn remove(&mut self, token: &str) -> bool {
        let before = self.sessions.len();
        self.sessions.retain(|s| s.token.value != token);
        self.sessions.len() < before
    }

    /// Removes all sessions expired relative to `now_ms` with TTL `ttl_ms`.
    ///
    /// Returns the number of removed sessions.
    #[allow(dead_code)]
    pub fn cleanup_expired(&mut self, now_ms: u64, ttl_ms: u64) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|s| !s.is_expired(now_ms, ttl_ms));
        before - self.sessions.len()
    }

    /// Returns the number of active sessions.
    #[allow(dead_code)]
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── Session ──────────────────────────────────────────────────────────────

    #[test]
    fn session_new_has_no_user() {
        let s = Session::new("abc123");
        assert_eq!(s.id, "abc123");
        assert!(s.user_id.is_none());
        assert!(!s.has_data());
    }

    #[test]
    fn session_set_and_get() {
        let mut s = Session::new("s1");
        s.set("lang", "en");
        assert_eq!(s.get("lang"), Some("en"));
        assert_eq!(s.get("missing"), None);
    }

    #[test]
    fn session_remove_key() {
        let mut s = Session::new("s1");
        s.set("k", "v");
        assert_eq!(s.remove("k"), Some("v".to_string()));
        assert_eq!(s.get("k"), None);
    }

    #[test]
    fn session_clear_data() {
        let mut s = Session::new("s1");
        s.set("a", "1");
        s.set("b", "2");
        s.clear_data();
        assert!(!s.has_data());
    }

    #[test]
    fn session_is_expired_when_inactive() {
        let mut s = Session::new("s1");
        s.touch(1_000); // last active at t=1000
        assert!(!s.is_expired(1_500, 1_000)); // 500s elapsed < 1000s TTL
        assert!(s.is_expired(2_001, 1_000)); // 1001s elapsed >= 1000s TTL
    }

    #[test]
    fn session_touch_resets_expiry() {
        let mut s = Session::new("s1");
        s.touch(0);
        assert!(s.is_expired(1_001, 1_000));
        s.touch(1_001); // renew
        assert!(!s.is_expired(1_001, 1_000));
    }

    #[test]
    fn session_age_s() {
        let mut s = Session::new("s1");
        s.created_at = 500;
        assert_eq!(s.age_s(1_500), 1_000);
        assert_eq!(s.age_s(499), 0); // saturating sub
    }

    // ── SessionStore ─────────────────────────────────────────────────────────

    #[test]
    fn store_create_and_count() {
        let mut store = SessionStore::new(3_600);
        let id = store.create(0);
        assert!(!id.is_empty());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn store_get_existing_session() {
        let mut store = SessionStore::new(3_600);
        let id = store.create(0);
        assert!(store.get(&id, 1_000).is_some());
    }

    #[test]
    fn store_get_expired_returns_none() {
        let mut store = SessionStore::new(100); // 100s TTL
        let id = store.create(0);
        assert!(store.get(&id, 101).is_none()); // expired
    }

    #[test]
    fn store_remove_session() {
        let mut store = SessionStore::new(3_600);
        let id = store.create(0);
        assert!(store.remove(&id));
        assert!(!store.remove(&id)); // already gone
        assert!(store.is_empty());
    }

    #[test]
    fn store_purge_expired() {
        let mut store = SessionStore::new(100);
        store.create(0);
        store.create(0);
        store.create(50);
        // Two sessions expired after 101s, one created at 50 expires at 150
        let purged = store.purge_expired(101);
        assert_eq!(purged, 2);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn store_get_mut_modifies_session() {
        let mut store = SessionStore::new(3_600);
        let id = store.create(0);
        if let Some(s) = store.get_mut(&id, 1) {
            s.set("role", "admin");
        }
        // Verify via get
        let val = store
            .get(&id, 2)
            .and_then(|s| s.get("role"))
            .map(str::to_string);
        assert_eq!(val.as_deref(), Some("admin"));
    }

    #[test]
    fn store_unique_ids_for_multiple_sessions() {
        let mut store = SessionStore::new(3_600);
        let id1 = store.create(1_000);
        let id2 = store.create(1_000);
        assert_ne!(id1, id2);
        assert_eq!(store.count(), 2);
    }

    // ── SessionToken ─────────────────────────────────────────────────────────

    #[test]
    fn token_generate_is_valid_format() {
        let t = SessionToken::generate(12345);
        assert!(t.is_valid_format(), "token '{}' is not valid", t.value);
    }

    #[test]
    fn token_generate_different_seeds_give_different_tokens() {
        let t1 = SessionToken::generate(1);
        let t2 = SessionToken::generate(2);
        assert_ne!(t1.value, t2.value);
    }

    #[test]
    fn token_generate_same_seed_is_deterministic() {
        let t1 = SessionToken::generate(999);
        let t2 = SessionToken::generate(999);
        assert_eq!(t1.value, t2.value);
    }

    #[test]
    fn token_value_is_32_chars() {
        let t = SessionToken::generate(0);
        assert_eq!(t.value.len(), 32);
    }

    // ── TokenSession ─────────────────────────────────────────────────────────

    #[test]
    fn token_session_is_not_expired_immediately() {
        let tok = SessionToken::generate(1);
        let s = TokenSession::new(tok, "alice", 1_000);
        assert!(!s.is_expired(1_000, 3_600_000));
    }

    #[test]
    fn token_session_expires_after_ttl() {
        let tok = SessionToken::generate(2);
        let s = TokenSession::new(tok, "bob", 0);
        assert!(s.is_expired(3_601_000, 3_600_000));
    }

    #[test]
    fn token_session_touch_extends_life() {
        let tok = SessionToken::generate(3);
        let mut s = TokenSession::new(tok, "carol", 0);
        s.touch(5_000);
        assert!(!s.is_expired(5_000, 3_600_000));
    }

    #[test]
    fn token_session_set_and_get_data() {
        let tok = SessionToken::generate(4);
        let mut s = TokenSession::new(tok, "dave", 0);
        s.set_data("theme", "dark");
        assert_eq!(s.get_data("theme"), Some("dark"));
        assert_eq!(s.get_data("missing"), None);
    }

    #[test]
    fn token_session_update_data() {
        let tok = SessionToken::generate(5);
        let mut s = TokenSession::new(tok, "eve", 0);
        s.set_data("lang", "en");
        s.set_data("lang", "fr");
        assert_eq!(s.get_data("lang"), Some("fr"));
        // Only one entry per key.
        assert_eq!(s.data.iter().filter(|(k, _)| k == "lang").count(), 1);
    }

    // ── TokenSessionStore ────────────────────────────────────────────────────

    #[test]
    fn token_store_create_and_find() {
        let mut store = TokenSessionStore::new(10);
        let token_val = store.create("user1", 0).token.value.clone();
        assert!(store.get(&token_val).is_some());
    }

    #[test]
    fn token_store_remove() {
        let mut store = TokenSessionStore::new(10);
        let token_val = store.create("user1", 0).token.value.clone();
        assert!(store.remove(&token_val));
        assert!(store.get(&token_val).is_none());
    }

    #[test]
    fn token_store_cleanup_expired() {
        let mut store = TokenSessionStore::new(10);
        store.create("u1", 0);
        store.create("u2", 0);
        store.create("u3", 160_000); // created later, not expired
        let removed = store.cleanup_expired(200_000, 50_000); // TTL=50s, t=200s
        assert_eq!(removed, 2);
        assert_eq!(store.active_count(), 1);
    }

    #[test]
    fn token_store_evicts_oldest_when_full() {
        let mut store = TokenSessionStore::new(2);
        let t1 = store.create("u1", 0).token.value.clone();
        store.create("u2", 1);
        store.create("u3", 2); // should evict u1
        assert!(store.get(&t1).is_none());
        assert_eq!(store.active_count(), 2);
    }

    #[test]
    fn token_store_touch_updates_last_seen() {
        let mut store = TokenSessionStore::new(10);
        let tv = store.create("u1", 0).token.value.clone();
        assert!(store.touch(&tv, 999_000));
        let s = store.get(&tv).expect("should succeed in test");
        assert_eq!(s.last_seen_ms, 999_000);
    }
}
