//! IS-05 Connection API HTTP handlers.
//!
//! Extracted from `http.rs` to keep file sizes under 2000 lines.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, Response};
use serde_json::{json, Value};

use super::{RtpTransportParams, TransportConstraints};

use super::http::{
    json_response, NmosHttpError, ServerState, StagedReceiverParams, StagedSenderParams,
};

// ============================================================================
// IS-05 handlers
// ============================================================================

pub(super) fn handle_connection_root() -> Response<Full<Bytes>> {
    let body = json!(["single/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_connection_sender_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let ids: Vec<String> = reg.all_senders().iter().map(|s| s.id.clone()).collect();
    json_response(200, json!(ids).to_string())
}

pub(super) async fn handle_connection_sender_root(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    if reg.get_sender(id).is_none() {
        let err = NmosHttpError::NotFound(format!("sender {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = json!(["staged/", "active/", "constraints/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_get_sender_staged(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    // Verify sender exists.
    {
        let reg = state.registry.read().await;
        if reg.get_sender(id).is_none() {
            let err = NmosHttpError::NotFound(format!("sender {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }
    let staged = state.sender_staged.read().await;
    let params = staged.get(id).cloned().unwrap_or_default();
    json_response(200, params.to_json().to_string())
}

pub(super) async fn handle_patch_sender_staged(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    // Verify sender exists.
    {
        let reg = state.registry.read().await;
        if reg.get_sender(id).is_none() {
            let err = NmosHttpError::NotFound(format!("sender {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }

    // Collect body bytes.
    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            let err = NmosHttpError::Body(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    // Parse JSON patch.
    let patch: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    // Validate transport_params against stored sender constraints (if any).
    {
        let reg = state.registry.read().await;
        if let Some(tc) = reg.get_sender_constraints(id) {
            let proposed = RtpTransportParams::from_json_patch(&patch);
            if let Err(violation) = tc.validate(&proposed) {
                let err = NmosHttpError::ConstraintViolation(violation.to_string());
                let mut body = err.to_json_body();
                // Inject the IS-05 structured violation detail
                if let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(&body) {
                    parsed["detail"] = violation.to_json();
                    body = parsed.to_string();
                }
                return json_response(err.status_code().as_u16(), body);
            }
        }
    }

    let mut staged = state.sender_staged.write().await;
    let params = staged
        .entry(id.to_string())
        .or_insert_with(StagedSenderParams::default);

    // Apply patch fields.
    if let Some(v) = patch.get("master_enable").and_then(|v| v.as_bool()) {
        params.master_enable = v;
    }
    if let Some(v) = patch.get("receiver_id") {
        params.receiver_id = v.as_str().map(str::to_string);
    }
    if let Some(tp) = patch.get("transport_params").and_then(|v| v.as_array()) {
        if let Some(first) = tp.first() {
            if let Some(v) = first.get("destination_ip").and_then(|v| v.as_str()) {
                params.destination_ip = Some(v.to_string());
            }
            if let Some(v) = first.get("destination_port").and_then(|v| v.as_u64()) {
                params.destination_port = Some(v as u16);
            }
            if let Some(v) = first.get("source_ip").and_then(|v| v.as_str()) {
                params.source_ip = Some(v.to_string());
            }
        }
    }

    let result = params.to_json().to_string();
    json_response(200, result)
}

pub(super) async fn handle_get_sender_active(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    // Verify sender exists.
    {
        let reg = state.registry.read().await;
        if reg.get_sender(id).is_none() {
            let err = NmosHttpError::NotFound(format!("sender {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }
    // Active state: reflect IS-05 active connections involving this sender.
    let cm = state.connection_manager.read().await;
    let active_conns: Vec<&str> = cm
        .active_connections()
        .iter()
        .filter(|c| c.sender_id == id)
        .map(|c| c.receiver_id.as_str())
        .collect();

    let body = json!({
        "master_enable": !active_conns.is_empty(),
        "receiver_id": active_conns.first().copied(),
        "transport_params": [{}],
        "activation": {
            "mode": null,
            "requested_time": null,
            "activation_time": null
        }
    })
    .to_string();
    json_response(200, body)
}

pub(super) async fn handle_connection_receiver_list(state: &ServerState) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    let ids: Vec<String> = reg.all_receivers().iter().map(|r| r.id.clone()).collect();
    json_response(200, json!(ids).to_string())
}

pub(super) async fn handle_connection_receiver_root(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    if reg.get_receiver(id).is_none() {
        let err = NmosHttpError::NotFound(format!("receiver {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = json!(["staged/", "active/", "constraints/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_get_receiver_staged(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    {
        let reg = state.registry.read().await;
        if reg.get_receiver(id).is_none() {
            let err = NmosHttpError::NotFound(format!("receiver {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }
    let staged = state.receiver_staged.read().await;
    let params = staged.get(id).cloned().unwrap_or_default();
    json_response(200, params.to_json().to_string())
}

pub(super) async fn handle_patch_receiver_staged(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    {
        let reg = state.registry.read().await;
        if reg.get_receiver(id).is_none() {
            let err = NmosHttpError::NotFound(format!("receiver {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }

    let body_bytes = match req.into_body().collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            let err = NmosHttpError::Body(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let patch: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    // Validate transport_params against stored receiver constraints (if any).
    {
        let reg = state.registry.read().await;
        if let Some(tc) = reg.get_receiver_constraints(id) {
            let proposed = RtpTransportParams::from_json_patch(&patch);
            if let Err(violation) = tc.validate(&proposed) {
                let err = NmosHttpError::ConstraintViolation(violation.to_string());
                let mut body = err.to_json_body();
                if let Ok(mut parsed) = serde_json::from_str::<serde_json::Value>(&body) {
                    parsed["detail"] = violation.to_json();
                    body = parsed.to_string();
                }
                return json_response(err.status_code().as_u16(), body);
            }
        }
    }

    let mut staged = state.receiver_staged.write().await;
    let params = staged
        .entry(id.to_string())
        .or_insert_with(StagedReceiverParams::default);

    if let Some(v) = patch.get("master_enable").and_then(|v| v.as_bool()) {
        params.master_enable = v;
    }
    if let Some(v) = patch.get("sender_id") {
        params.sender_id = v.as_str().map(str::to_string);
    }
    if let Some(tp) = patch.get("transport_params").and_then(|v| v.as_array()) {
        if let Some(first) = tp.first() {
            if let Some(v) = first.get("interface_ip").and_then(|v| v.as_str()) {
                params.interface_ip = Some(v.to_string());
            }
            if let Some(v) = first.get("multicast_ip").and_then(|v| v.as_str()) {
                params.multicast_ip = Some(v.to_string());
            }
            if let Some(v) = first.get("source_port").and_then(|v| v.as_u64()) {
                params.source_port = Some(v as u16);
            }
        }
    }

    // If sender_id is being set, also reflect in connection manager.
    if let Some(sender_id) = &params.sender_id.clone() {
        let mut cm = state.connection_manager.write().await;
        if params.master_enable {
            cm.connect(sender_id.clone(), id.to_string());
        } else {
            cm.disconnect(sender_id, id);
        }
    }

    let result = params.to_json().to_string();
    json_response(200, result)
}

pub(super) async fn handle_get_receiver_active(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    {
        let reg = state.registry.read().await;
        if reg.get_receiver(id).is_none() {
            let err = NmosHttpError::NotFound(format!("receiver {id}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }
    let cm = state.connection_manager.read().await;
    let active_sender: Option<&str> = cm
        .active_connections()
        .iter()
        .find(|c| c.receiver_id == id)
        .map(|c| c.sender_id.as_str());

    let body = json!({
        "master_enable": active_sender.is_some(),
        "sender_id": active_sender,
        "transport_params": [{}],
        "activation": {
            "mode": null,
            "requested_time": null,
            "activation_time": null
        }
    })
    .to_string();
    json_response(200, body)
}

// ============================================================================
// IS-05 constraint handlers
// ============================================================================

/// Serialize a `TransportConstraints` to the IS-05 JSON array format.
fn constraints_to_json(tc: &TransportConstraints) -> serde_json::Value {
    tc.to_json()
}

pub(super) async fn handle_get_sender_constraints(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    if reg.get_sender(id).is_none() {
        let err = NmosHttpError::NotFound(format!("sender {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = match reg.get_sender_constraints(id) {
        Some(tc) => constraints_to_json(tc).to_string(),
        // IS-05: empty array means unconstrained
        None => serde_json::json!([{}]).to_string(),
    };
    json_response(200, body)
}

pub(super) async fn handle_get_receiver_constraints(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let reg = state.registry.read().await;
    if reg.get_receiver(id).is_none() {
        let err = NmosHttpError::NotFound(format!("receiver {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = match reg.get_receiver_constraints(id) {
        Some(tc) => constraints_to_json(tc).to_string(),
        None => serde_json::json!([{}]).to_string(),
    };
    json_response(200, body)
}
