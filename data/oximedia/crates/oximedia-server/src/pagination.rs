//! Cursor-based pagination for stable, scalable API responses.
//!
//! Provides a cursor-based pagination system that avoids the offset/skip
//! instability of traditional page-number pagination. Cursors encode an
//! opaque position marker that is stable even as new items are inserted or
//! deleted between requests.
//!
//! # Cursor format
//!
//! Cursors are base64-encoded JSON containing the sort field value and a
//! unique tie-breaker (typically the item ID). This makes them opaque to
//! clients while remaining debuggable.

#![allow(dead_code)]

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── Cursor ───────────────────────────────────────────────────────────────────

/// Decoded cursor payload containing sort position and tie-breaker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CursorPayload {
    /// Value of the sort field at the cursor position (e.g., timestamp, name).
    pub sort_value: String,
    /// Unique item identifier for tie-breaking when sort values collide.
    pub id: String,
    /// Sort direction used when this cursor was created.
    pub direction: SortDirection,
}

impl CursorPayload {
    /// Creates a new cursor payload.
    pub fn new(
        sort_value: impl Into<String>,
        id: impl Into<String>,
        direction: SortDirection,
    ) -> Self {
        Self {
            sort_value: sort_value.into(),
            id: id.into(),
            direction,
        }
    }

    /// Encodes the payload into an opaque cursor string.
    pub fn encode(&self) -> Result<String, PaginationError> {
        let json = serde_json::to_string(self).map_err(|e| {
            PaginationError::EncodingFailed(format!("cursor serialization failed: {}", e))
        })?;
        Ok(URL_SAFE_NO_PAD.encode(json.as_bytes()))
    }

    /// Decodes an opaque cursor string back into a payload.
    pub fn decode(cursor: &str) -> Result<Self, PaginationError> {
        let bytes = URL_SAFE_NO_PAD
            .decode(cursor.as_bytes())
            .map_err(|e| PaginationError::InvalidCursor(format!("base64 decode failed: {}", e)))?;
        let json = String::from_utf8(bytes).map_err(|e| {
            PaginationError::InvalidCursor(format!("cursor is not valid UTF-8: {}", e))
        })?;
        serde_json::from_str(&json)
            .map_err(|e| PaginationError::InvalidCursor(format!("cursor JSON parse failed: {}", e)))
    }
}

// ── Sort Direction ───────────────────────────────────────────────────────────

/// Sort direction for cursor-based pagination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum SortDirection {
    /// Ascending order (oldest first, A→Z).
    Ascending,
    /// Descending order (newest first, Z→A).
    Descending,
}

impl SortDirection {
    /// Returns the SQL operator for the "after cursor" condition.
    pub fn sql_operator(&self) -> &'static str {
        match self {
            SortDirection::Ascending => ">",
            SortDirection::Descending => "<",
        }
    }

    /// Returns the SQL ORDER BY direction keyword.
    pub fn sql_order(&self) -> &'static str {
        match self {
            SortDirection::Ascending => "ASC",
            SortDirection::Descending => "DESC",
        }
    }

    /// Reverses the direction.
    pub fn reverse(&self) -> Self {
        match self {
            SortDirection::Ascending => SortDirection::Descending,
            SortDirection::Descending => SortDirection::Ascending,
        }
    }
}

impl fmt::Display for SortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortDirection::Ascending => write!(f, "asc"),
            SortDirection::Descending => write!(f, "desc"),
        }
    }
}

impl Default for SortDirection {
    fn default() -> Self {
        SortDirection::Descending
    }
}

// ── Page Request ─────────────────────────────────────────────────────────────

/// A pagination request parsed from query parameters.
#[derive(Debug, Clone)]
pub struct PageRequest {
    /// Number of items to return (clamped to `max_page_size`).
    pub page_size: usize,
    /// Cursor to resume from (None for the first page).
    pub after: Option<CursorPayload>,
    /// Cursor for backward navigation (None when paging forward).
    pub before: Option<CursorPayload>,
    /// Field to sort by.
    pub sort_by: String,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Default page size.
const DEFAULT_PAGE_SIZE: usize = 25;
/// Maximum page size to prevent abuse.
const MAX_PAGE_SIZE: usize = 100;

impl PageRequest {
    /// Creates a new page request from raw query parameters.
    ///
    /// # Arguments
    ///
    /// * `page_size` - Requested page size (clamped to 1..=100).
    /// * `after_cursor` - Opaque cursor string for forward pagination.
    /// * `before_cursor` - Opaque cursor string for backward pagination.
    /// * `sort_by` - Field name to sort by.
    /// * `direction` - Sort direction.
    pub fn new(
        page_size: Option<usize>,
        after_cursor: Option<&str>,
        before_cursor: Option<&str>,
        sort_by: Option<&str>,
        direction: Option<SortDirection>,
    ) -> Result<Self, PaginationError> {
        let size = page_size
            .unwrap_or(DEFAULT_PAGE_SIZE)
            .clamp(1, MAX_PAGE_SIZE);

        let after = after_cursor.map(CursorPayload::decode).transpose()?;
        let before = before_cursor.map(CursorPayload::decode).transpose()?;

        if after.is_some() && before.is_some() {
            return Err(PaginationError::InvalidCursor(
                "cannot specify both 'after' and 'before' cursors".to_string(),
            ));
        }

        Ok(Self {
            page_size: size,
            after,
            before,
            sort_by: sort_by.unwrap_or("created_at").to_string(),
            direction: direction.unwrap_or_default(),
        })
    }

    /// Generates the SQL WHERE clause fragment and ORDER BY for this page request.
    ///
    /// The returned `WhereClause` contains placeholders `?1` and `?2` for the
    /// sort value and ID tie-breaker respectively.
    pub fn sql_clause(&self, sort_column: &str, id_column: &str) -> SqlClause {
        let effective_direction = if self.before.is_some() {
            self.direction.reverse()
        } else {
            self.direction
        };

        let cursor = self.after.as_ref().or(self.before.as_ref());

        let where_fragment = cursor.map(|_| {
            let op = effective_direction.sql_operator();
            format!("({sort_column} {op} ?1 OR ({sort_column} = ?1 AND {id_column} {op} ?2))")
        });

        let order_by = format!(
            "{sort_column} {dir}, {id_column} {dir}",
            dir = effective_direction.sql_order()
        );

        SqlClause {
            where_fragment,
            order_by,
            limit: self.page_size + 1, // fetch one extra to detect has_next
            cursor_sort_value: cursor.map(|c| c.sort_value.clone()),
            cursor_id: cursor.map(|c| c.id.clone()),
        }
    }
}

/// SQL fragments generated by a [`PageRequest`].
#[derive(Debug, Clone)]
pub struct SqlClause {
    /// Optional WHERE clause fragment (e.g., `(created_at < ?1 OR ...)`).
    pub where_fragment: Option<String>,
    /// ORDER BY clause (e.g., `created_at DESC, id DESC`).
    pub order_by: String,
    /// LIMIT value (page_size + 1 to detect has_next).
    pub limit: usize,
    /// Bind value for the sort field (position 1).
    pub cursor_sort_value: Option<String>,
    /// Bind value for the ID tie-breaker (position 2).
    pub cursor_id: Option<String>,
}

// ── Page Response ────────────────────────────────────────────────────────────

/// A paginated response containing items and navigation cursors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageResponse<T> {
    /// The page of items.
    pub items: Vec<T>,
    /// Total number of items returned in this page.
    pub count: usize,
    /// Cursor to pass as `after` to get the next page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Cursor to pass as `before` to get the previous page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prev_cursor: Option<String>,
    /// Whether there are more items after this page.
    pub has_next: bool,
    /// Whether there are items before this page.
    pub has_prev: bool,
}

impl<T> PageResponse<T> {
    /// Builds a page response from a raw result set.
    ///
    /// The `items` vec should contain at most `page_size + 1` elements;
    /// if it has more than `page_size`, `has_next` is `true` and the extra
    /// item is dropped.
    ///
    /// `sort_value_fn` extracts the sort field value from each item.
    /// `id_fn` extracts the unique ID from each item.
    pub fn build<F, G>(
        mut items: Vec<T>,
        page_size: usize,
        direction: SortDirection,
        has_prev: bool,
        sort_value_fn: F,
        id_fn: G,
    ) -> Result<Self, PaginationError>
    where
        F: Fn(&T) -> String,
        G: Fn(&T) -> String,
    {
        let has_next = items.len() > page_size;
        if has_next {
            items.truncate(page_size);
        }

        let next_cursor = if has_next {
            if let Some(last) = items.last() {
                let payload = CursorPayload::new(sort_value_fn(last), id_fn(last), direction);
                Some(payload.encode()?)
            } else {
                None
            }
        } else {
            None
        };

        let prev_cursor = if has_prev {
            if let Some(first) = items.first() {
                let payload = CursorPayload::new(sort_value_fn(first), id_fn(first), direction);
                Some(payload.encode()?)
            } else {
                None
            }
        } else {
            None
        };

        let count = items.len();

        Ok(Self {
            items,
            count,
            next_cursor,
            prev_cursor,
            has_next,
            has_prev,
        })
    }
}

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors from the pagination module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PaginationError {
    /// The cursor string could not be decoded.
    #[error("Invalid cursor: {0}")]
    InvalidCursor(String),
    /// Cursor encoding failed (serialization error).
    #[error("Cursor encoding failed: {0}")]
    EncodingFailed(String),
    /// The requested sort field is not supported.
    #[error("Unsupported sort field: {0}")]
    UnsupportedSortField(String),
}

// ── Sort field validation ────────────────────────────────────────────────────

/// A registry of allowed sort fields for a given resource type.
#[derive(Debug, Clone)]
pub struct SortFieldRegistry {
    /// Map from user-facing field name to the SQL column name.
    fields: std::collections::HashMap<String, String>,
    /// The default field to sort by if none is specified.
    default_field: String,
}

impl SortFieldRegistry {
    /// Creates a new registry with the given default field.
    pub fn new(default_field: impl Into<String>) -> Self {
        let default = default_field.into();
        let mut fields = std::collections::HashMap::new();
        fields.insert(default.clone(), default.clone());
        Self {
            fields,
            default_field: default,
        }
    }

    /// Registers an allowed sort field with its SQL column mapping.
    #[must_use]
    pub fn with_field(
        mut self,
        field_name: impl Into<String>,
        column_name: impl Into<String>,
    ) -> Self {
        self.fields.insert(field_name.into(), column_name.into());
        self
    }

    /// Resolves a user-facing sort field name to the SQL column name.
    ///
    /// Returns the default if `field_name` is `None`.
    pub fn resolve(&self, field_name: Option<&str>) -> Result<&str, PaginationError> {
        let name = field_name.unwrap_or(&self.default_field);
        self.fields
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| PaginationError::UnsupportedSortField(name.to_string()))
    }

    /// Returns all registered field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.keys().map(String::as_str).collect()
    }

    /// Returns the default sort field name.
    pub fn default_field(&self) -> &str {
        &self.default_field
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CursorPayload ───────────────────────────────────────────────────────

    #[test]
    fn test_cursor_roundtrip() {
        let payload =
            CursorPayload::new("2025-01-15T10:30:00Z", "abc-123", SortDirection::Descending);
        let encoded = payload.encode().expect("encode");
        let decoded = CursorPayload::decode(&encoded).expect("decode");
        assert_eq!(payload, decoded);
    }

    #[test]
    fn test_cursor_encode_is_url_safe() {
        let payload =
            CursorPayload::new("value/with+special=chars", "id-1", SortDirection::Ascending);
        let encoded = payload.encode().expect("encode");
        // URL_SAFE_NO_PAD uses - and _ instead of + and /
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_cursor_decode_invalid_base64() {
        let result = CursorPayload::decode("!!!not-base64!!!");
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, PaginationError::InvalidCursor(_)));
    }

    #[test]
    fn test_cursor_decode_invalid_json() {
        let bad = URL_SAFE_NO_PAD.encode(b"not json");
        let result = CursorPayload::decode(&bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_cursor_decode_valid_base64_wrong_structure() {
        let bad = URL_SAFE_NO_PAD.encode(br#"{"wrong":"fields"}"#);
        let result = CursorPayload::decode(&bad);
        assert!(result.is_err());
    }

    // ── SortDirection ───────────────────────────────────────────────────────

    #[test]
    fn test_sort_direction_sql_operator() {
        assert_eq!(SortDirection::Ascending.sql_operator(), ">");
        assert_eq!(SortDirection::Descending.sql_operator(), "<");
    }

    #[test]
    fn test_sort_direction_sql_order() {
        assert_eq!(SortDirection::Ascending.sql_order(), "ASC");
        assert_eq!(SortDirection::Descending.sql_order(), "DESC");
    }

    #[test]
    fn test_sort_direction_reverse() {
        assert_eq!(
            SortDirection::Ascending.reverse(),
            SortDirection::Descending
        );
        assert_eq!(
            SortDirection::Descending.reverse(),
            SortDirection::Ascending
        );
    }

    #[test]
    fn test_sort_direction_display() {
        assert_eq!(SortDirection::Ascending.to_string(), "asc");
        assert_eq!(SortDirection::Descending.to_string(), "desc");
    }

    #[test]
    fn test_sort_direction_default_is_descending() {
        assert_eq!(SortDirection::default(), SortDirection::Descending);
    }

    // ── PageRequest ─────────────────────────────────────────────────────────

    #[test]
    fn test_page_request_defaults() {
        let req = PageRequest::new(None, None, None, None, None).expect("ok");
        assert_eq!(req.page_size, DEFAULT_PAGE_SIZE);
        assert_eq!(req.sort_by, "created_at");
        assert_eq!(req.direction, SortDirection::Descending);
        assert!(req.after.is_none());
        assert!(req.before.is_none());
    }

    #[test]
    fn test_page_request_clamps_page_size_min() {
        let req = PageRequest::new(Some(0), None, None, None, None).expect("ok");
        assert_eq!(req.page_size, 1);
    }

    #[test]
    fn test_page_request_clamps_page_size_max() {
        let req = PageRequest::new(Some(500), None, None, None, None).expect("ok");
        assert_eq!(req.page_size, MAX_PAGE_SIZE);
    }

    #[test]
    fn test_page_request_with_after_cursor() {
        let payload = CursorPayload::new("2025-01-15", "id-1", SortDirection::Descending);
        let cursor = payload.encode().expect("encode");
        let req = PageRequest::new(Some(10), Some(&cursor), None, None, None).expect("ok");
        assert!(req.after.is_some());
        assert_eq!(req.after.as_ref().map(|c| c.id.as_str()), Some("id-1"));
    }

    #[test]
    fn test_page_request_rejects_both_cursors() {
        let payload_a = CursorPayload::new("a", "1", SortDirection::Descending);
        let payload_b = CursorPayload::new("b", "2", SortDirection::Descending);
        let ca = payload_a.encode().expect("encode");
        let cb = payload_b.encode().expect("encode");
        let result = PageRequest::new(None, Some(&ca), Some(&cb), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_page_request_custom_sort() {
        let req = PageRequest::new(
            Some(50),
            None,
            None,
            Some("filename"),
            Some(SortDirection::Ascending),
        )
        .expect("ok");
        assert_eq!(req.sort_by, "filename");
        assert_eq!(req.direction, SortDirection::Ascending);
    }

    // ── SqlClause ───────────────────────────────────────────────────────────

    #[test]
    fn test_sql_clause_no_cursor() {
        let req = PageRequest::new(Some(20), None, None, None, None).expect("ok");
        let clause = req.sql_clause("created_at", "id");
        assert!(clause.where_fragment.is_none());
        assert_eq!(clause.order_by, "created_at DESC, id DESC");
        assert_eq!(clause.limit, 21); // page_size + 1
    }

    #[test]
    fn test_sql_clause_with_after_cursor_desc() {
        let payload = CursorPayload::new("2025-01-15", "item-42", SortDirection::Descending);
        let cursor = payload.encode().expect("encode");
        let req = PageRequest::new(
            Some(10),
            Some(&cursor),
            None,
            None,
            Some(SortDirection::Descending),
        )
        .expect("ok");
        let clause = req.sql_clause("created_at", "id");
        assert!(clause.where_fragment.is_some());
        let frag = clause.where_fragment.as_ref().expect("has fragment");
        assert!(frag.contains("<")); // DESC uses <
        assert_eq!(clause.cursor_sort_value.as_deref(), Some("2025-01-15"));
        assert_eq!(clause.cursor_id.as_deref(), Some("item-42"));
    }

    #[test]
    fn test_sql_clause_with_after_cursor_asc() {
        let payload = CursorPayload::new("alpha", "id-1", SortDirection::Ascending);
        let cursor = payload.encode().expect("encode");
        let req = PageRequest::new(
            Some(10),
            Some(&cursor),
            None,
            None,
            Some(SortDirection::Ascending),
        )
        .expect("ok");
        let clause = req.sql_clause("name", "id");
        let frag = clause.where_fragment.as_ref().expect("has fragment");
        assert!(frag.contains(">")); // ASC uses >
        assert_eq!(clause.order_by, "name ASC, id ASC");
    }

    #[test]
    fn test_sql_clause_before_cursor_reverses_direction() {
        let payload = CursorPayload::new("2025-01-10", "id-5", SortDirection::Descending);
        let cursor = payload.encode().expect("encode");
        let req = PageRequest::new(
            Some(10),
            None,
            Some(&cursor),
            None,
            Some(SortDirection::Descending),
        )
        .expect("ok");
        let clause = req.sql_clause("created_at", "id");
        // Before cursor reverses: DESC -> ASC
        assert_eq!(clause.order_by, "created_at ASC, id ASC");
        let frag = clause.where_fragment.as_ref().expect("has fragment");
        assert!(frag.contains(">")); // reversed to ASC uses >
    }

    // ── PageResponse ────────────────────────────────────────────────────────

    #[test]
    fn test_page_response_build_with_next() {
        let items: Vec<(String, String)> = (0..11)
            .map(|i| (format!("2025-01-{:02}", i + 1), format!("id-{}", i)))
            .collect();

        let resp = PageResponse::build(
            items,
            10,
            SortDirection::Descending,
            false,
            |item| item.0.clone(),
            |item| item.1.clone(),
        )
        .expect("build");

        assert_eq!(resp.count, 10); // truncated from 11
        assert!(resp.has_next);
        assert!(!resp.has_prev);
        assert!(resp.next_cursor.is_some());
        assert!(resp.prev_cursor.is_none());
    }

    #[test]
    fn test_page_response_build_last_page() {
        let items: Vec<(String, String)> = (0..5)
            .map(|i| (format!("val-{}", i), format!("id-{}", i)))
            .collect();

        let resp = PageResponse::build(
            items,
            10,
            SortDirection::Ascending,
            true,
            |item| item.0.clone(),
            |item| item.1.clone(),
        )
        .expect("build");

        assert_eq!(resp.count, 5);
        assert!(!resp.has_next);
        assert!(resp.has_prev);
        assert!(resp.next_cursor.is_none());
        assert!(resp.prev_cursor.is_some());
    }

    #[test]
    fn test_page_response_build_empty() {
        let items: Vec<(String, String)> = vec![];

        let resp = PageResponse::build(
            items,
            10,
            SortDirection::Descending,
            false,
            |item: &(String, String)| item.0.clone(),
            |item: &(String, String)| item.1.clone(),
        )
        .expect("build");

        assert_eq!(resp.count, 0);
        assert!(!resp.has_next);
        assert!(!resp.has_prev);
        assert!(resp.items.is_empty());
    }

    #[test]
    fn test_page_response_cursor_roundtrip() {
        let items: Vec<(String, String)> = (0..11)
            .map(|i| (format!("ts-{}", i), format!("id-{}", i)))
            .collect();

        let resp = PageResponse::build(
            items,
            10,
            SortDirection::Descending,
            false,
            |item| item.0.clone(),
            |item| item.1.clone(),
        )
        .expect("build");

        // The next cursor should decode back to the last item's values
        let cursor = resp.next_cursor.as_ref().expect("has next");
        let decoded = CursorPayload::decode(cursor).expect("decode");
        assert_eq!(decoded.sort_value, "ts-9"); // last item in truncated list
        assert_eq!(decoded.id, "id-9");
    }

    #[test]
    fn test_page_response_serializes_json() {
        let items = vec!["a".to_string(), "b".to_string()];
        let resp = PageResponse {
            items,
            count: 2,
            next_cursor: Some("abc123".to_string()),
            prev_cursor: None,
            has_next: true,
            has_prev: false,
        };
        let json = serde_json::to_value(&resp).expect("serialize");
        assert_eq!(json["count"], 2);
        assert_eq!(json["has_next"], true);
        assert_eq!(json["next_cursor"], "abc123");
        // prev_cursor should be omitted (skip_serializing_if)
        assert!(json.get("prev_cursor").is_none());
    }

    // ── SortFieldRegistry ───────────────────────────────────────────────────

    #[test]
    fn test_sort_field_registry_default() {
        let reg = SortFieldRegistry::new("created_at");
        assert_eq!(reg.resolve(None).expect("ok"), "created_at");
    }

    #[test]
    fn test_sort_field_registry_custom_mapping() {
        let reg = SortFieldRegistry::new("created_at")
            .with_field("name", "filename")
            .with_field("size", "file_size_bytes");
        assert_eq!(reg.resolve(Some("name")).expect("ok"), "filename");
        assert_eq!(reg.resolve(Some("size")).expect("ok"), "file_size_bytes");
    }

    #[test]
    fn test_sort_field_registry_unknown_field() {
        let reg = SortFieldRegistry::new("created_at");
        let result = reg.resolve(Some("nonexistent"));
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            PaginationError::UnsupportedSortField(_)
        ));
    }

    #[test]
    fn test_sort_field_registry_field_names() {
        let reg = SortFieldRegistry::new("created_at")
            .with_field("name", "filename")
            .with_field("size", "file_size_bytes");
        let names = reg.field_names();
        assert!(names.contains(&"created_at"));
        assert!(names.contains(&"name"));
        assert!(names.contains(&"size"));
    }

    // ── PaginationError ─────────────────────────────────────────────────────

    #[test]
    fn test_pagination_error_display() {
        let err = PaginationError::InvalidCursor("bad cursor".to_string());
        assert_eq!(err.to_string(), "Invalid cursor: bad cursor");
    }

    #[test]
    fn test_pagination_error_unsupported_field() {
        let err = PaginationError::UnsupportedSortField("invalid_col".to_string());
        assert!(err.to_string().contains("invalid_col"));
    }
}
