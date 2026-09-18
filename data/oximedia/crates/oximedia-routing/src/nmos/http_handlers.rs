//! IS-08 Channel Mapping, IS-09 System, and IS-11 Stream Compatibility HTTP handlers.
//!
//! These handlers are separated from the main `http` module for size management.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, Response};
use serde_json::{json, Value};

use super::channel_mapping::{
    active_to_json, io_summary_to_json, staged_to_json, ChannelMappingEntry,
};
use super::compatibility::{
    compatibility_state_to_json, constraints_to_json as is11_constraints_to_json, Is11ConstraintSet,
};
use super::http::{json_response, NmosHttpError, ServerState};

// ============================================================================
// IS-08 Channel Mapping handlers
// ============================================================================

pub(super) fn handle_channel_mapping_root() -> Response<Full<Bytes>> {
    let body = json!(["map/", "io/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_channel_mapping_activation_list(
    state: &ServerState,
) -> Response<Full<Bytes>> {
    let cm = state.channel_mapping.read().await;
    let device_ids: Vec<&str> = cm.list_devices();
    json_response(200, json!(device_ids).to_string())
}

pub(super) async fn handle_channel_mapping_activation_device(
    state: &ServerState,
    device_id: &str,
) -> Response<Full<Bytes>> {
    let cm = state.channel_mapping.read().await;
    match cm.get_table(device_id) {
        Some(_) => {
            let body = json!(["active/", "staged/"]).to_string();
            json_response(200, body)
        }
        None => {
            let err = NmosHttpError::NotFound(format!("channel mapping device {device_id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

pub(super) async fn handle_channel_mapping_activation_active(
    state: &ServerState,
    device_id: &str,
) -> Response<Full<Bytes>> {
    let cm = state.channel_mapping.read().await;
    match cm.get_table(device_id) {
        Some(table) => {
            let v = active_to_json(table);
            json_response(200, v.to_string())
        }
        None => {
            let err = NmosHttpError::NotFound(format!("channel mapping device {device_id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

pub(super) async fn handle_get_channel_mapping_staged(
    state: &ServerState,
    device_id: &str,
) -> Response<Full<Bytes>> {
    let cm = state.channel_mapping.read().await;
    match cm.get_table(device_id) {
        Some(table) => {
            let v = staged_to_json(table);
            json_response(200, v.to_string())
        }
        None => {
            let err = NmosHttpError::NotFound(format!("channel mapping device {device_id}"));
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

/// POST /x-nmos/channelmapping/v1.0/map/activations/{device_id}/staged
pub(super) async fn handle_post_channel_mapping_staged(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    device_id: &str,
) -> Response<Full<Bytes>> {
    {
        let cm = state.channel_mapping.read().await;
        if cm.get_table(device_id).is_none() {
            let err = NmosHttpError::NotFound(format!("channel mapping device {device_id}"));
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

    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let entries_val = payload
        .get("entries")
        .cloned()
        .unwrap_or(Value::Array(vec![]));
    let entries: Vec<ChannelMappingEntry> = match serde_json::from_value(entries_val) {
        Ok(e) => e,
        Err(e) => {
            let err = NmosHttpError::BadRequest(format!("invalid entries: {e}"));
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let action = payload
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("stage");

    let mut cm = state.channel_mapping.write().await;

    if let Err(e) = cm.stage_mapping(device_id, entries) {
        let err = NmosHttpError::BadRequest(e.to_string());
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }

    if action == "activate" {
        if let Err(e) = cm.activate(device_id) {
            let err = NmosHttpError::Internal(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    }

    let table = match cm.get_table(device_id) {
        Some(t) => t,
        None => {
            let err = NmosHttpError::Internal("table disappeared after staging".to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let staged_v = staged_to_json(table);
    let active_v = active_to_json(table);
    let body = json!({
        "device_id": device_id,
        "staged": staged_v,
        "active": active_v
    })
    .to_string();
    json_response(200, body)
}

pub(super) async fn handle_channel_mapping_io(state: &ServerState) -> Response<Full<Bytes>> {
    let cm = state.channel_mapping.read().await;
    let summaries: Vec<Value> = cm
        .list_devices()
        .iter()
        .filter_map(|id| cm.get_table(id).map(|t| io_summary_to_json(id, t)))
        .collect();
    json_response(200, json!(summaries).to_string())
}

// ============================================================================
// IS-09 System API handlers
// ============================================================================

pub(super) fn handle_system_root() -> Response<Full<Bytes>> {
    let body = json!(["global/", "health/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_system_global(state: &ServerState) -> Response<Full<Bytes>> {
    let api = state.system_api.read().await;
    match api.to_global_json() {
        Ok(val) => json_response(200, val.to_string()),
        Err(e) => {
            let err = NmosHttpError::Internal(e.to_string());
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

pub(super) async fn handle_system_health(state: &ServerState) -> Response<Full<Bytes>> {
    let api = state.system_api.read().await;
    let registry = state.registry.read().await;
    let conn_mgr = state.connection_manager.read().await;
    let health = api.health(&registry, &conn_mgr);
    match serde_json::to_string(&health) {
        Ok(body) => json_response(200, body),
        Err(e) => {
            let err = NmosHttpError::Internal(e.to_string());
            json_response(err.status_code().as_u16(), err.to_json_body())
        }
    }
}

// ============================================================================
// IS-11 Stream Compatibility Management handlers
// ============================================================================

pub(super) fn handle_stream_compat_root() -> Response<Full<Bytes>> {
    let body = json!(["senders/", "receivers/"]).to_string();
    json_response(200, body)
}

pub(super) async fn handle_stream_compat_sender_list(state: &ServerState) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    let ids: Vec<&str> = compat.all_sender_ids();
    json_response(200, json!(ids).to_string())
}

pub(super) async fn handle_stream_compat_receiver_list(
    state: &ServerState,
) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    let ids: Vec<&str> = compat.all_receiver_ids();
    json_response(200, json!(ids).to_string())
}

pub(super) async fn handle_get_stream_compat_sender_constraints(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    if compat.get_sender_cap(id).is_none() {
        let err = NmosHttpError::NotFound(format!("IS-11 sender {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = match compat.get_sender_active_constraints(id) {
        Some(sets) => is11_constraints_to_json(sets).to_string(),
        None => json!([]).to_string(),
    };
    json_response(200, body)
}

pub(super) async fn handle_put_stream_compat_sender_constraints(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    {
        let compat = state.compatibility.read().await;
        if compat.get_sender_cap(id).is_none() {
            let err = NmosHttpError::NotFound(format!("IS-11 sender {id}"));
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

    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let sets: Vec<Is11ConstraintSet> = match payload.as_array() {
        Some(arr) => arr
            .iter()
            .map(|item| {
                let map = item
                    .as_object()
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                Is11ConstraintSet(map)
            })
            .collect(),
        None => {
            let err = NmosHttpError::BadRequest(
                "IS-11 active_constraints must be a JSON array".to_string(),
            );
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let mut compat = state.compatibility.write().await;
    if let Err(e) = compat.set_sender_active_constraints(id, sets) {
        let err = NmosHttpError::BadRequest(e.to_string());
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }

    let body = match compat.get_sender_active_constraints(id) {
        Some(s) => is11_constraints_to_json(s).to_string(),
        None => json!([]).to_string(),
    };
    json_response(200, body)
}

pub(super) async fn handle_stream_compat_sender_status(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    if compat.get_sender_cap(id).is_none() {
        let err = NmosHttpError::NotFound(format!("IS-11 sender {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let receiver_ids: Vec<String> = compat
        .all_receiver_ids()
        .into_iter()
        .map(str::to_string)
        .collect();
    let statuses: Vec<Value> = receiver_ids
        .iter()
        .map(|rid| {
            let state_val = compat.check_compatibility(id, rid);
            json!({
                "receiver_id": rid,
                "status": compatibility_state_to_json(&state_val),
            })
        })
        .collect();
    let body = json!({
        "sender_id": id,
        "receivers": statuses,
    })
    .to_string();
    json_response(200, body)
}

pub(super) async fn handle_get_stream_compat_receiver_constraints(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    if compat.get_receiver_cap(id).is_none() {
        let err = NmosHttpError::NotFound(format!("IS-11 receiver {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let body = match compat.get_receiver_active_constraints(id) {
        Some(sets) => is11_constraints_to_json(sets).to_string(),
        None => json!([]).to_string(),
    };
    json_response(200, body)
}

pub(super) async fn handle_put_stream_compat_receiver_constraints(
    req: Request<hyper::body::Incoming>,
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    {
        let compat = state.compatibility.read().await;
        if compat.get_receiver_cap(id).is_none() {
            let err = NmosHttpError::NotFound(format!("IS-11 receiver {id}"));
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

    let payload: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            let err = NmosHttpError::BadRequest(e.to_string());
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let sets: Vec<Is11ConstraintSet> = match payload.as_array() {
        Some(arr) => arr
            .iter()
            .map(|item| {
                let map = item
                    .as_object()
                    .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();
                Is11ConstraintSet(map)
            })
            .collect(),
        None => {
            let err = NmosHttpError::BadRequest(
                "IS-11 active_constraints must be a JSON array".to_string(),
            );
            return json_response(err.status_code().as_u16(), err.to_json_body());
        }
    };

    let mut compat = state.compatibility.write().await;
    if let Err(e) = compat.set_receiver_active_constraints(id, sets) {
        let err = NmosHttpError::BadRequest(e.to_string());
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }

    let body = match compat.get_receiver_active_constraints(id) {
        Some(s) => is11_constraints_to_json(s).to_string(),
        None => json!([]).to_string(),
    };
    json_response(200, body)
}

pub(super) async fn handle_stream_compat_receiver_status(
    state: &ServerState,
    id: &str,
) -> Response<Full<Bytes>> {
    let compat = state.compatibility.read().await;
    if compat.get_receiver_cap(id).is_none() {
        let err = NmosHttpError::NotFound(format!("IS-11 receiver {id}"));
        return json_response(err.status_code().as_u16(), err.to_json_body());
    }
    let sender_ids: Vec<String> = compat
        .all_sender_ids()
        .into_iter()
        .map(str::to_string)
        .collect();
    let statuses: Vec<Value> = sender_ids
        .iter()
        .map(|sid| {
            let state_val = compat.check_compatibility(sid, id);
            json!({
                "sender_id": sid,
                "status": compatibility_state_to_json(&state_val),
            })
        })
        .collect();
    let body = json!({
        "receiver_id": id,
        "senders": statuses,
    })
    .to_string();
    json_response(200, body)
}
