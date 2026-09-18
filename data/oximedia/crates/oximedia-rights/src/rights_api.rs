//! REST-style rights query API.
//!
//! Provides request/response types for a rights management HTTP API.
//! This module is transport-agnostic: it defines the data structures and
//! handler logic but does not bind to any specific HTTP framework, making it
//! usable from both native servers and wasm environments.
//!
//! The core handler is [`RightsApiHandler`] which processes `ApiRequest`
//! values and returns [`ApiResponse`] values, operating entirely against
//! an in-memory [`crate::rights_check::RightsChecker`].

#![allow(dead_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::rights_check::{ActionKind, CheckRequest, RightsChecker, RightsGrant};
use crate::{Result, RightsError};

// ── ApiRequest ───────────────────────────────────────────────────────────────

/// The kind of API operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "params")]
pub enum ApiOp {
    /// Check whether an action is permitted.
    CheckRights {
        /// Asset identifier.
        asset_id: String,
        /// Desired action (serialised as the variant name).
        action: String,
        /// ISO-3166-1 alpha-2 territory code.
        territory: String,
        /// Platform name.
        platform: String,
        /// Current Unix timestamp (seconds).
        now: u64,
    },
    /// Retrieve all grants registered for an asset.
    ListGrants {
        /// Asset identifier.
        asset_id: String,
    },
    /// Register a new rights grant.
    AddGrant {
        /// The grant to add (serialised inline).
        grant: GrantSpec,
    },
    /// Revoke a grant by ID.
    RevokeGrant {
        /// Grant identifier.
        grant_id: String,
    },
    /// Return the total number of registered grants.
    GrantCount,
}

/// Specification for creating a new grant via the API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrantSpec {
    /// Unique grant identifier.
    pub id: String,
    /// Asset this grant applies to.
    pub asset_id: String,
    /// Permitted actions (as string names matching `ActionKind` variants).
    pub actions: Vec<String>,
    /// Territory codes (empty = worldwide).
    pub territories: Vec<String>,
    /// Platform names (empty = all).
    pub platforms: Vec<String>,
    /// Validity start (Unix seconds).
    pub valid_from: u64,
    /// Validity end (Unix seconds). `0` is interpreted as `u64::MAX` (no expiry).
    pub valid_until: u64,
}

impl GrantSpec {
    /// Parse the action strings into [`ActionKind`] values.
    ///
    /// Returns `Err` if any action string is unrecognised.
    pub fn parse_actions(&self) -> std::result::Result<Vec<ActionKind>, String> {
        self.actions.iter().map(|a| parse_action_kind(a)).collect()
    }

    /// Convert this spec into a [`RightsGrant`].
    ///
    /// Returns `Err` if any action name is unrecognised.
    pub fn into_grant(self) -> std::result::Result<RightsGrant, String> {
        let actions = self.parse_actions()?;
        let valid_until = if self.valid_until == 0 {
            u64::MAX
        } else {
            self.valid_until
        };

        let mut grant =
            RightsGrant::new(&self.id, &self.asset_id).with_window(self.valid_from, valid_until);
        for action in actions {
            grant = grant.with_action(action);
        }
        for t in &self.territories {
            grant = grant.with_territory(t);
        }
        for p in &self.platforms {
            grant = grant.with_platform(p);
        }
        Ok(grant)
    }
}

/// Parse an action-kind string into an [`ActionKind`].
fn parse_action_kind(s: &str) -> std::result::Result<ActionKind, String> {
    match s.to_lowercase().as_str() {
        "stream" => Ok(ActionKind::Stream),
        "download" => Ok(ActionKind::Download),
        "embed" => Ok(ActionKind::Embed),
        "broadcast" => Ok(ActionKind::Broadcast),
        "advertising" | "advertise" => Ok(ActionKind::Advertising),
        "derivative" => Ok(ActionKind::Derivative),
        "distribute" => Ok(ActionKind::Distribute),
        "archive" => Ok(ActionKind::Archive),
        other => Err(format!("Unknown action kind: {other}")),
    }
}

// ── ApiResponse ──────────────────────────────────────────────────────────────

/// HTTP-style status code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatusCode {
    /// 200 OK.
    Ok,
    /// 201 Created.
    Created,
    /// 400 Bad Request.
    BadRequest,
    /// 403 Forbidden.
    Forbidden,
    /// 404 Not Found.
    NotFound,
    /// 409 Conflict.
    Conflict,
    /// 500 Internal Server Error.
    InternalServerError,
}

impl StatusCode {
    /// Whether the status indicates success.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::Ok | Self::Created)
    }

    /// Numeric HTTP status code.
    #[must_use]
    pub fn as_u16(self) -> u16 {
        match self {
            Self::Ok => 200,
            Self::Created => 201,
            Self::BadRequest => 400,
            Self::Forbidden => 403,
            Self::NotFound => 404,
            Self::Conflict => 409,
            Self::InternalServerError => 500,
        }
    }
}

/// Response from the API handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    /// HTTP-style status.
    pub status: StatusCode,
    /// Machine-readable body.
    pub body: ApiResponseBody,
}

impl ApiResponse {
    fn ok(body: ApiResponseBody) -> Self {
        Self {
            status: StatusCode::Ok,
            body,
        }
    }

    fn created(body: ApiResponseBody) -> Self {
        Self {
            status: StatusCode::Created,
            body,
        }
    }

    fn error(status: StatusCode, message: &str) -> Self {
        Self {
            status,
            body: ApiResponseBody::Error {
                message: message.to_string(),
            },
        }
    }

    /// Whether the response indicates success.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }
}

/// The payload of an API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ApiResponseBody {
    /// Plain text message.
    Message {
        /// The message text.
        message: String,
    },
    /// Error description.
    Error {
        /// The error description text.
        message: String,
    },
    /// Rights check result.
    CheckResult {
        /// Whether the action is allowed.
        allowed: bool,
        /// The grant ID that permitted the action, or the denial reason.
        reason: String,
    },
    /// List of grants (minimal representation).
    Grants {
        /// The grants list.
        grants: Vec<GrantInfo>,
    },
    /// Total grant count.
    Count {
        /// The count value.
        count: usize,
    },
}

/// Minimal serialisable summary of a rights grant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrantInfo {
    /// Grant ID.
    pub id: String,
    /// Asset ID.
    pub asset_id: String,
    /// Whether the grant is currently revoked.
    pub revoked: bool,
    /// Validity start.
    pub valid_from: u64,
    /// Validity end.
    pub valid_until: u64,
}

impl From<&RightsGrant> for GrantInfo {
    fn from(g: &RightsGrant) -> Self {
        Self {
            id: g.id.clone(),
            asset_id: g.asset_id.clone(),
            revoked: g.revoked,
            valid_from: g.valid_from,
            valid_until: g.valid_until,
        }
    }
}

// ── RightsApiHandler ─────────────────────────────────────────────────────────

/// Transport-agnostic handler for rights API operations.
///
/// Wraps an in-memory [`RightsChecker`] and processes [`ApiOp`] requests.
#[derive(Debug, Default)]
pub struct RightsApiHandler {
    checker: RightsChecker,
    /// Grant store for retrieval (mirrors checker's internal grants).
    grants: HashMap<String, RightsGrant>,
}

impl RightsApiHandler {
    /// Create a new handler with an empty rights store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Process an [`ApiOp`] and return an [`ApiResponse`].
    pub fn handle(&mut self, op: ApiOp) -> ApiResponse {
        match op {
            ApiOp::CheckRights {
                asset_id,
                action,
                territory,
                platform,
                now,
            } => {
                let action_kind = match parse_action_kind(&action) {
                    Ok(a) => a,
                    Err(msg) => return ApiResponse::error(StatusCode::BadRequest, &msg),
                };
                let req = CheckRequest::new(&asset_id, action_kind, &territory, &platform, now);
                let result = self.checker.check(&req);
                let (allowed, reason) = match result {
                    crate::rights_check::CheckResult::Allowed(ref grant_id) => {
                        (true, grant_id.clone())
                    }
                    crate::rights_check::CheckResult::Denied(ref msg) => (false, msg.clone()),
                };
                if allowed {
                    ApiResponse::ok(ApiResponseBody::CheckResult { allowed, reason })
                } else {
                    ApiResponse {
                        status: StatusCode::Forbidden,
                        body: ApiResponseBody::CheckResult { allowed, reason },
                    }
                }
            }

            ApiOp::ListGrants { asset_id } => {
                let infos: Vec<GrantInfo> = self
                    .grants
                    .values()
                    .filter(|g| g.asset_id == asset_id)
                    .map(GrantInfo::from)
                    .collect();
                ApiResponse::ok(ApiResponseBody::Grants { grants: infos })
            }

            ApiOp::AddGrant { grant: spec } => {
                let id = spec.id.clone();
                if self.grants.contains_key(&id) {
                    return ApiResponse::error(
                        StatusCode::Conflict,
                        &format!("Grant {id} already exists"),
                    );
                }
                match spec.into_grant() {
                    Ok(grant) => {
                        self.checker.add_grant(grant.clone());
                        self.grants.insert(grant.id.clone(), grant);
                        ApiResponse::created(ApiResponseBody::Message {
                            message: format!("Grant {id} created"),
                        })
                    }
                    Err(msg) => ApiResponse::error(StatusCode::BadRequest, &msg),
                }
            }

            ApiOp::RevokeGrant { grant_id } => {
                match self.grants.get_mut(&grant_id) {
                    Some(g) => {
                        g.revoked = true;
                        // The checker holds its own copy; re-add revoked state.
                        // Simplest approach: rebuild checker from grants.
                        self.rebuild_checker();
                        ApiResponse::ok(ApiResponseBody::Message {
                            message: format!("Grant {grant_id} revoked"),
                        })
                    }
                    None => ApiResponse::error(
                        StatusCode::NotFound,
                        &format!("Grant {grant_id} not found"),
                    ),
                }
            }

            ApiOp::GrantCount => ApiResponse::ok(ApiResponseBody::Count {
                count: self.grants.len(),
            }),
        }
    }

    /// Process a JSON-encoded operation string.
    ///
    /// Returns the JSON-encoded [`ApiResponse`].
    pub fn handle_json(&mut self, json: &str) -> Result<String> {
        let op: ApiOp =
            serde_json::from_str(json).map_err(|e| RightsError::Serialization(e.to_string()))?;
        let response = self.handle(op);
        serde_json::to_string_pretty(&response)
            .map_err(|e| RightsError::Serialization(e.to_string()))
    }

    /// Rebuild the internal checker from the current grants map.
    fn rebuild_checker(&mut self) {
        let mut checker = RightsChecker::new();
        for grant in self.grants.values() {
            checker.add_grant(grant.clone());
        }
        self.checker = checker;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn stream_grant_spec(id: &str, asset: &str) -> GrantSpec {
        GrantSpec {
            id: id.to_string(),
            asset_id: asset.to_string(),
            actions: vec!["stream".to_string()],
            territories: vec![],
            platforms: vec![],
            valid_from: 0,
            valid_until: 0, // no expiry
        }
    }

    #[test]
    fn test_add_grant_and_count() {
        let mut handler = RightsApiHandler::new();
        let resp = handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-1"),
        });
        assert!(resp.is_success());
        assert_eq!(resp.status, StatusCode::Created);

        let count_resp = handler.handle(ApiOp::GrantCount);
        assert!(count_resp.is_success());
        if let ApiResponseBody::Count { count } = count_resp.body {
            assert_eq!(count, 1);
        } else {
            panic!("Expected Count body");
        }
    }

    #[test]
    fn test_add_duplicate_grant_conflict() {
        let mut handler = RightsApiHandler::new();
        handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-1"),
        });
        let resp = handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-1"),
        });
        assert_eq!(resp.status, StatusCode::Conflict);
    }

    #[test]
    fn test_check_rights_allowed() {
        let mut handler = RightsApiHandler::new();
        handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-A"),
        });
        let resp = handler.handle(ApiOp::CheckRights {
            asset_id: "asset-A".into(),
            action: "stream".into(),
            territory: "US".into(),
            platform: "web".into(),
            now: 100,
        });
        assert_eq!(resp.status, StatusCode::Ok);
        if let ApiResponseBody::CheckResult { allowed, .. } = resp.body {
            assert!(allowed);
        } else {
            panic!("Expected CheckResult body");
        }
    }

    #[test]
    fn test_check_rights_denied_no_grant() {
        let mut handler = RightsApiHandler::new();
        let resp = handler.handle(ApiOp::CheckRights {
            asset_id: "unknown".into(),
            action: "stream".into(),
            territory: "US".into(),
            platform: "web".into(),
            now: 100,
        });
        assert_eq!(resp.status, StatusCode::Forbidden);
    }

    #[test]
    fn test_check_rights_bad_action() {
        let mut handler = RightsApiHandler::new();
        let resp = handler.handle(ApiOp::CheckRights {
            asset_id: "asset-A".into(),
            action: "fly_to_mars".into(),
            territory: "US".into(),
            platform: "web".into(),
            now: 100,
        });
        assert_eq!(resp.status, StatusCode::BadRequest);
    }

    #[test]
    fn test_list_grants() {
        let mut handler = RightsApiHandler::new();
        handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-X"),
        });
        handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g2", "asset-X"),
        });
        let resp = handler.handle(ApiOp::ListGrants {
            asset_id: "asset-X".into(),
        });
        assert!(resp.is_success());
        if let ApiResponseBody::Grants { grants } = resp.body {
            assert_eq!(grants.len(), 2);
        } else {
            panic!("Expected Grants body");
        }
    }

    #[test]
    fn test_revoke_grant() {
        let mut handler = RightsApiHandler::new();
        handler.handle(ApiOp::AddGrant {
            grant: stream_grant_spec("g1", "asset-R"),
        });
        let rev_resp = handler.handle(ApiOp::RevokeGrant {
            grant_id: "g1".into(),
        });
        assert!(rev_resp.is_success());

        // Check is now denied
        let check = handler.handle(ApiOp::CheckRights {
            asset_id: "asset-R".into(),
            action: "stream".into(),
            territory: "US".into(),
            platform: "web".into(),
            now: 100,
        });
        assert_eq!(check.status, StatusCode::Forbidden);
    }

    #[test]
    fn test_revoke_nonexistent_grant() {
        let mut handler = RightsApiHandler::new();
        let resp = handler.handle(ApiOp::RevokeGrant {
            grant_id: "ghost".into(),
        });
        assert_eq!(resp.status, StatusCode::NotFound);
    }

    #[test]
    fn test_handle_json_round_trip() {
        let mut handler = RightsApiHandler::new();
        let json = r#"{"op":"GrantCount","params":null}"#;
        let result = handler.handle_json(json);
        assert!(result.is_ok());
        let s = result.expect("json round trip");
        assert!(s.contains("count"));
    }

    #[test]
    fn test_status_code_numeric() {
        assert_eq!(StatusCode::Ok.as_u16(), 200);
        assert_eq!(StatusCode::Created.as_u16(), 201);
        assert_eq!(StatusCode::Forbidden.as_u16(), 403);
        assert_eq!(StatusCode::NotFound.as_u16(), 404);
        assert_eq!(StatusCode::Conflict.as_u16(), 409);
        assert_eq!(StatusCode::InternalServerError.as_u16(), 500);
    }

    #[test]
    fn test_status_code_is_success() {
        assert!(StatusCode::Ok.is_success());
        assert!(StatusCode::Created.is_success());
        assert!(!StatusCode::Forbidden.is_success());
        assert!(!StatusCode::NotFound.is_success());
    }

    #[test]
    fn test_add_grant_bad_action() {
        let mut handler = RightsApiHandler::new();
        let spec = GrantSpec {
            id: "bad-g".into(),
            asset_id: "a".into(),
            actions: vec!["not_an_action".into()],
            territories: vec![],
            platforms: vec![],
            valid_from: 0,
            valid_until: 0,
        };
        let resp = handler.handle(ApiOp::AddGrant { grant: spec });
        assert_eq!(resp.status, StatusCode::BadRequest);
    }

    #[test]
    fn test_parse_action_kind_variants() {
        assert!(parse_action_kind("stream").is_ok());
        assert!(parse_action_kind("download").is_ok());
        assert!(parse_action_kind("embed").is_ok());
        assert!(parse_action_kind("broadcast").is_ok());
        assert!(parse_action_kind("advertising").is_ok());
        assert!(parse_action_kind("advertise").is_ok());
        assert!(parse_action_kind("derivative").is_ok());
        assert!(parse_action_kind("distribute").is_ok());
        assert!(parse_action_kind("archive").is_ok());
        assert!(parse_action_kind("unknown_op").is_err());
    }
}
