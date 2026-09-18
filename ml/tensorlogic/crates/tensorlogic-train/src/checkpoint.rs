//! Optimizer checkpointing: save/load optimizer state (momentum buffers, step counts, etc.)
//!
//! Provides [`OptimizerCheckpoint`], [`CheckpointManager`], and [`LossTracker`] for
//! persisting and restoring optimizer state during training.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Per-parameter optimizer state (moment vectors, step counter, shape).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParamState {
    pub name: String,
    /// First moment estimate: velocity for SGD-momentum, m for Adam.
    pub first_moment: Vec<f64>,
    /// Second moment estimate: v for Adam; empty for SGD.
    pub second_moment: Vec<f64>,
    pub step: u64,
    pub shape: Vec<usize>,
}

impl ParamState {
    /// Construct a new [`ParamState`].
    pub fn new(
        name: impl Into<String>,
        first_moment: Vec<f64>,
        second_moment: Vec<f64>,
        step: u64,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            name: name.into(),
            first_moment,
            second_moment,
            step,
            shape,
        }
    }
}

/// Metadata attached to a checkpoint (loss values, extra annotations).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMetadata {
    pub created_at_step: u64,
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub extra: HashMap<String, String>,
}

impl CheckpointMetadata {
    /// Construct metadata for a given step.
    pub fn new(created_at_step: u64) -> Self {
        Self {
            created_at_step,
            loss: None,
            val_loss: None,
            extra: HashMap::new(),
        }
    }
}

/// Serialisable snapshot of an optimizer's full training state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizerCheckpoint {
    pub step: u64,
    pub epoch: u32,
    pub optimizer_name: String,
    /// Parameter name → per-parameter state.
    pub param_states: HashMap<String, ParamState>,
    pub hyperparams: HashMap<String, f64>,
    pub metadata: CheckpointMetadata,
}

impl OptimizerCheckpoint {
    /// Create an empty checkpoint for the given optimizer at `step`/`epoch`.
    pub fn new(optimizer_name: impl Into<String>, step: u64, epoch: u32) -> Self {
        Self {
            step,
            epoch,
            optimizer_name: optimizer_name.into(),
            param_states: HashMap::new(),
            hyperparams: HashMap::new(),
            metadata: CheckpointMetadata::new(step),
        }
    }

    /// Insert or replace the state for one parameter.
    pub fn add_param_state(&mut self, name: impl Into<String>, state: ParamState) {
        self.param_states.insert(name.into(), state);
    }

    /// Record a scalar hyper-parameter (learning rate, beta1, etc.).
    pub fn set_hyperparam(&mut self, key: impl Into<String>, value: f64) {
        self.hyperparams.insert(key.into(), value);
    }

    /// Retrieve a previously recorded hyper-parameter value.
    pub fn get_hyperparam(&self, key: &str) -> Option<f64> {
        self.hyperparams.get(key).copied()
    }

    /// Number of parameters stored in this checkpoint.
    pub fn num_params(&self) -> usize {
        self.param_states.len()
    }

    /// Total number of scalar elements across all first-moment vectors.
    pub fn total_elements(&self) -> usize {
        self.param_states
            .values()
            .map(|ps| ps.first_moment.len())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Serialization format
// ---------------------------------------------------------------------------

/// Wire format used when writing checkpoints to disk.
#[derive(Debug, Clone)]
pub enum CheckpointFormat {
    /// Custom binary envelope: magic bytes `TLCK` + u32 version + JSON payload.
    Binary,
    /// Human-readable `key=value` text with `\n---\n` section separators.
    Text,
}

impl CheckpointFormat {
    fn file_extension(&self) -> &'static str {
        match self {
            CheckpointFormat::Binary => "tlck",
            CheckpointFormat::Text => "tlckt",
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointError
// ---------------------------------------------------------------------------

/// Errors that can occur while managing checkpoints.
#[derive(Debug, Clone)]
pub enum CheckpointError {
    IoError(String),
    SerializationError(String),
    DeserializationError(String),
    CheckpointNotFound { step: u64 },
    NoCheckpointsAvailable,
    InvalidFormat(String),
    DirectoryCreationFailed(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::IoError(msg) => write!(f, "IO error: {msg}"),
            CheckpointError::SerializationError(msg) => {
                write!(f, "Serialization error: {msg}")
            }
            CheckpointError::DeserializationError(msg) => {
                write!(f, "Deserialization error: {msg}")
            }
            CheckpointError::CheckpointNotFound { step } => {
                write!(f, "Checkpoint not found for step {step}")
            }
            CheckpointError::NoCheckpointsAvailable => {
                write!(f, "No checkpoints are available")
            }
            CheckpointError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            CheckpointError::DirectoryCreationFailed(msg) => {
                write!(f, "Directory creation failed: {msg}")
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

// ---------------------------------------------------------------------------
// Serialization helpers (Text format)
// ---------------------------------------------------------------------------

/// Encode a `f64` slice as a comma-separated string.
fn encode_f64_slice(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Decode a comma-separated string into `Vec<f64>`.
fn decode_f64_slice(s: &str) -> Result<Vec<f64>, CheckpointError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|tok| {
            tok.trim()
                .parse::<f64>()
                .map_err(|e| CheckpointError::DeserializationError(format!("f64 parse: {e}")))
        })
        .collect()
}

/// Encode a `usize` slice as a comma-separated string.
fn encode_usize_slice(values: &[usize]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Decode a comma-separated string into `Vec<usize>`.
fn decode_usize_slice(s: &str) -> Result<Vec<usize>, CheckpointError> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|tok| {
            tok.trim()
                .parse::<usize>()
                .map_err(|e| CheckpointError::DeserializationError(format!("usize parse: {e}")))
        })
        .collect()
}

/// Serialize [`OptimizerCheckpoint`] to the `Text` format.
fn serialize_text(ckpt: &OptimizerCheckpoint) -> Vec<u8> {
    let mut out = String::new();

    // --- header section ---
    out.push_str("section=header\n");
    out.push_str(&format!("step={}\n", ckpt.step));
    out.push_str(&format!("epoch={}\n", ckpt.epoch));
    out.push_str(&format!("optimizer_name={}\n", ckpt.optimizer_name));
    out.push_str(&format!(
        "created_at_step={}\n",
        ckpt.metadata.created_at_step
    ));
    if let Some(loss) = ckpt.metadata.loss {
        out.push_str(&format!("loss={loss}\n"));
    }
    if let Some(val_loss) = ckpt.metadata.val_loss {
        out.push_str(&format!("val_loss={val_loss}\n"));
    }
    for (k, v) in &ckpt.metadata.extra {
        out.push_str(&format!("extra.{k}={v}\n"));
    }

    out.push_str("\n---\n");

    // --- hyperparams section ---
    out.push_str("section=hyperparams\n");
    for (k, v) in &ckpt.hyperparams {
        out.push_str(&format!("hp.{k}={v}\n"));
    }

    out.push_str("\n---\n");

    // --- param_states section ---
    out.push_str("section=param_states\n");
    for (param_name, ps) in &ckpt.param_states {
        out.push_str(&format!("param.name={param_name}\n"));
        out.push_str(&format!(
            "param.first_moment={}\n",
            encode_f64_slice(&ps.first_moment)
        ));
        out.push_str(&format!(
            "param.second_moment={}\n",
            encode_f64_slice(&ps.second_moment)
        ));
        out.push_str(&format!("param.step={}\n", ps.step));
        out.push_str(&format!("param.shape={}\n", encode_usize_slice(&ps.shape)));
        out.push_str("param.end\n");
    }

    out.into_bytes()
}

/// Deserialize [`OptimizerCheckpoint`] from the `Text` format.
fn deserialize_text(bytes: &[u8]) -> Result<OptimizerCheckpoint, CheckpointError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|e| CheckpointError::DeserializationError(format!("UTF-8: {e}")))?;

    let mut step: Option<u64> = None;
    let mut epoch: Option<u32> = None;
    let mut optimizer_name: Option<String> = None;
    let mut created_at_step: u64 = 0;
    let mut loss: Option<f64> = None;
    let mut val_loss: Option<f64> = None;
    let mut extra: HashMap<String, String> = HashMap::new();
    let mut hyperparams: HashMap<String, f64> = HashMap::new();
    let mut param_states: HashMap<String, ParamState> = HashMap::new();

    // Working state for the currently-open param block.
    let mut cur_name: Option<String> = None;
    let mut cur_first: Vec<f64> = Vec::new();
    let mut cur_second: Vec<f64> = Vec::new();
    let mut cur_step: u64 = 0;
    let mut cur_shape: Vec<usize> = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line == "---" {
            continue;
        }
        if line.starts_with("section=") {
            continue;
        }
        if line == "param.end" {
            if let Some(name) = cur_name.take() {
                param_states.insert(
                    name.clone(),
                    ParamState {
                        name,
                        first_moment: std::mem::take(&mut cur_first),
                        second_moment: std::mem::take(&mut cur_second),
                        step: cur_step,
                        shape: std::mem::take(&mut cur_shape),
                    },
                );
            }
            cur_step = 0;
            continue;
        }

        let (key, value) = line.split_once('=').ok_or_else(|| {
            CheckpointError::DeserializationError(format!("Missing '=' in line: {line}"))
        })?;

        match key {
            "step" => {
                step =
                    Some(value.parse::<u64>().map_err(|e| {
                        CheckpointError::DeserializationError(format!("step: {e}"))
                    })?);
            }
            "epoch" => {
                epoch =
                    Some(value.parse::<u32>().map_err(|e| {
                        CheckpointError::DeserializationError(format!("epoch: {e}"))
                    })?);
            }
            "optimizer_name" => {
                optimizer_name = Some(value.to_owned());
            }
            "created_at_step" => {
                created_at_step = value.parse::<u64>().map_err(|e| {
                    CheckpointError::DeserializationError(format!("created_at_step: {e}"))
                })?;
            }
            "loss" => {
                loss =
                    Some(value.parse::<f64>().map_err(|e| {
                        CheckpointError::DeserializationError(format!("loss: {e}"))
                    })?);
            }
            "val_loss" => {
                val_loss = Some(value.parse::<f64>().map_err(|e| {
                    CheckpointError::DeserializationError(format!("val_loss: {e}"))
                })?);
            }
            "param.name" => {
                cur_name = Some(value.to_owned());
            }
            "param.first_moment" => {
                cur_first = decode_f64_slice(value)?;
            }
            "param.second_moment" => {
                cur_second = decode_f64_slice(value)?;
            }
            "param.step" => {
                cur_step = value.parse::<u64>().map_err(|e| {
                    CheckpointError::DeserializationError(format!("param.step: {e}"))
                })?;
            }
            "param.shape" => {
                cur_shape = decode_usize_slice(value)?;
            }
            other if other.starts_with("hp.") => {
                let hp_key = other.trim_start_matches("hp.");
                let hp_val = value.parse::<f64>().map_err(|e| {
                    CheckpointError::DeserializationError(format!("hyperparam {hp_key}: {e}"))
                })?;
                hyperparams.insert(hp_key.to_owned(), hp_val);
            }
            other if other.starts_with("extra.") => {
                let ex_key = other.trim_start_matches("extra.");
                extra.insert(ex_key.to_owned(), value.to_owned());
            }
            _ => {} // tolerate unknown fields for forward-compat
        }
    }

    let step =
        step.ok_or_else(|| CheckpointError::DeserializationError("missing field: step".into()))?;
    let epoch = epoch
        .ok_or_else(|| CheckpointError::DeserializationError("missing field: epoch".into()))?;
    let optimizer_name = optimizer_name.ok_or_else(|| {
        CheckpointError::DeserializationError("missing field: optimizer_name".into())
    })?;

    Ok(OptimizerCheckpoint {
        step,
        epoch,
        optimizer_name,
        param_states,
        hyperparams,
        metadata: CheckpointMetadata {
            created_at_step,
            loss,
            val_loss,
            extra,
        },
    })
}

// Magic bytes that begin every Binary checkpoint file.
const BINARY_MAGIC: [u8; 4] = [0x54, 0x4C, 0x43, 0x4B]; // "TLCK"
const BINARY_VERSION: u32 = 1;

/// Serialize an [`OptimizerCheckpoint`] to bytes using the chosen format.
pub fn serialize_checkpoint(
    ckpt: &OptimizerCheckpoint,
    format: CheckpointFormat,
) -> Result<Vec<u8>, CheckpointError> {
    match format {
        CheckpointFormat::Text => Ok(serialize_text(ckpt)),
        CheckpointFormat::Binary => {
            // Payload: serde_json → UTF-8 bytes.
            let json = serde_json::to_vec(ckpt)
                .map_err(|e| CheckpointError::SerializationError(format!("JSON: {e}")))?;

            // Envelope: magic(4) + version(4, BE) + payload_len(4, BE) + payload.
            let payload_len = json.len() as u32;
            let mut out = Vec::with_capacity(12 + json.len());
            out.extend_from_slice(&BINARY_MAGIC);
            out.extend_from_slice(&BINARY_VERSION.to_be_bytes());
            out.extend_from_slice(&payload_len.to_be_bytes());
            out.extend_from_slice(&json);
            Ok(out)
        }
    }
}

/// Deserialize an [`OptimizerCheckpoint`] from bytes using the chosen format.
pub fn deserialize_checkpoint(
    bytes: &[u8],
    format: CheckpointFormat,
) -> Result<OptimizerCheckpoint, CheckpointError> {
    match format {
        CheckpointFormat::Text => deserialize_text(bytes),
        CheckpointFormat::Binary => {
            // Validate magic bytes.
            if bytes.len() < 12 {
                return Err(CheckpointError::InvalidFormat(
                    "binary checkpoint too short".into(),
                ));
            }
            if bytes[..4] != BINARY_MAGIC {
                return Err(CheckpointError::InvalidFormat(
                    "bad magic bytes — not a TLCK checkpoint".into(),
                ));
            }
            let version = u32::from_be_bytes(
                bytes[4..8]
                    .try_into()
                    .map_err(|_| CheckpointError::InvalidFormat("version bytes".into()))?,
            );
            if version != BINARY_VERSION {
                return Err(CheckpointError::InvalidFormat(format!(
                    "unsupported version {version}"
                )));
            }
            let payload_len = u32::from_be_bytes(
                bytes[8..12]
                    .try_into()
                    .map_err(|_| CheckpointError::InvalidFormat("length bytes".into()))?,
            ) as usize;
            let payload_end = 12 + payload_len;
            if bytes.len() < payload_end {
                return Err(CheckpointError::InvalidFormat(
                    "truncated binary checkpoint".into(),
                ));
            }
            let json = &bytes[12..payload_end];
            serde_json::from_slice(json)
                .map_err(|e| CheckpointError::DeserializationError(format!("JSON: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages writing and reading checkpoints under a directory.
///
/// Keeps a rolling window of the most-recent `max_to_keep` checkpoints, deleting
/// older files automatically after each save.
pub struct CheckpointManager {
    pub dir: PathBuf,
    pub max_to_keep: usize,
    pub format: CheckpointFormat,
    /// Ordered list of saved checkpoint paths (oldest first).
    saved: Vec<PathBuf>,
}

impl CheckpointManager {
    /// Create a new manager, creating `dir` if it does not already exist.
    pub fn new(
        dir: impl AsRef<Path>,
        max_to_keep: usize,
        format: CheckpointFormat,
    ) -> Result<Self, CheckpointError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| {
            CheckpointError::DirectoryCreationFailed(format!("{}: {e}", dir.display()))
        })?;
        Ok(Self {
            dir,
            max_to_keep,
            format,
            saved: Vec::new(),
        })
    }

    /// Compute the filename for a checkpoint at `step` with the given format.
    fn checkpoint_filename(step: u64, format: &CheckpointFormat) -> String {
        format!("ckpt-step-{:012}.{}", step, format.file_extension())
    }

    /// Save `ckpt` to disk and prune old checkpoints. Returns the saved path.
    pub fn save(&mut self, ckpt: &OptimizerCheckpoint) -> Result<PathBuf, CheckpointError> {
        let filename = Self::checkpoint_filename(ckpt.step, &self.format);
        let path = self.dir.join(&filename);

        let bytes = serialize_checkpoint(ckpt, self.format.clone())?;
        std::fs::write(&path, &bytes)
            .map_err(|e| CheckpointError::IoError(format!("write {}: {e}", path.display())))?;

        self.saved.push(path.clone());
        self.prune_old()?;
        Ok(path)
    }

    /// Load the most recently saved checkpoint.
    pub fn load_latest(&self) -> Result<OptimizerCheckpoint, CheckpointError> {
        let path = self
            .saved
            .last()
            .ok_or(CheckpointError::NoCheckpointsAvailable)?;
        self.load_from_path(path)
    }

    /// Load the checkpoint saved at a specific training step.
    pub fn load_at_step(&self, step: u64) -> Result<OptimizerCheckpoint, CheckpointError> {
        let filename = Self::checkpoint_filename(step, &self.format);
        let path = self.dir.join(&filename);
        if !self.saved.iter().any(|p| p == &path) {
            return Err(CheckpointError::CheckpointNotFound { step });
        }
        self.load_from_path(&path)
    }

    /// Return `(step, path)` pairs for all retained checkpoints.
    pub fn list(&self) -> Vec<(u64, &Path)> {
        self.saved
            .iter()
            .filter_map(|p| {
                // Extract step from filename "ckpt-step-<12digits>.<ext>"
                let stem = p.file_stem()?.to_str()?;
                let step_str = stem.strip_prefix("ckpt-step-")?;
                let step = step_str.parse::<u64>().ok()?;
                Some((step, p.as_path()))
            })
            .collect()
    }

    /// Number of checkpoints currently retained.
    pub fn count(&self) -> usize {
        self.saved.len()
    }

    // --- private helpers ---

    fn load_from_path(&self, path: &Path) -> Result<OptimizerCheckpoint, CheckpointError> {
        let bytes = std::fs::read(path)
            .map_err(|e| CheckpointError::IoError(format!("read {}: {e}", path.display())))?;
        deserialize_checkpoint(&bytes, self.format.clone())
    }

    /// Delete checkpoints that exceed the `max_to_keep` rolling window.
    fn prune_old(&mut self) -> Result<(), CheckpointError> {
        while self.saved.len() > self.max_to_keep {
            let oldest = self.saved.remove(0);
            if oldest.exists() {
                std::fs::remove_file(&oldest).map_err(|e| {
                    CheckpointError::IoError(format!("delete {}: {e}", oldest.display()))
                })?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LossTracker
// ---------------------------------------------------------------------------

/// Rolling-window tracker for scalar loss values recorded during training.
///
/// Provides moving average, min/max, and a simple improvement check useful for
/// early stopping decisions.
#[derive(Debug, Clone)]
pub struct LossTracker {
    pub window_size: usize,
    history: VecDeque<f64>,
}

impl LossTracker {
    /// Create a new tracker with the given sliding window capacity.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
        }
    }

    /// Record a new loss value, evicting the oldest if the window is full.
    pub fn push(&mut self, loss: f64) {
        if self.history.len() == self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(loss);
    }

    /// Arithmetic mean over the current window; `None` if empty.
    pub fn moving_average(&self) -> Option<f64> {
        if self.history.is_empty() {
            return None;
        }
        let sum: f64 = self.history.iter().sum();
        Some(sum / self.history.len() as f64)
    }

    /// Minimum value in the current window; `None` if empty.
    pub fn min(&self) -> Option<f64> {
        self.history.iter().copied().reduce(f64::min)
    }

    /// Maximum value in the current window; `None` if empty.
    pub fn max(&self) -> Option<f64> {
        self.history.iter().copied().reduce(f64::max)
    }

    /// Returns `true` when the minimum loss seen in the most-recent `patience`
    /// values is strictly less than the minimum over the full window *excluding*
    /// those recent values.  This captures "the model has improved recently".
    ///
    /// Returns `false` when there are not enough data points to compare.
    pub fn is_improving(&self, patience: usize) -> bool {
        if self.history.len() <= patience {
            return false;
        }
        let split = self.history.len() - patience;
        let older_min = self.history.iter().take(split).copied().reduce(f64::min);
        let recent_min = self.history.iter().skip(split).copied().reduce(f64::min);
        match (older_min, recent_min) {
            (Some(old), Some(new)) => new < old,
            _ => false,
        }
    }

    /// Number of values currently held.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// `true` when no values have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- helper: build a small checkpoint ---
    fn make_ckpt(step: u64, epoch: u32) -> OptimizerCheckpoint {
        let mut ckpt = OptimizerCheckpoint::new("adam", step, epoch);
        ckpt.set_hyperparam("lr", 0.001);
        ckpt.set_hyperparam("beta1", 0.9);
        let ps = ParamState::new(
            "layer0.weight",
            vec![0.1, 0.2, 0.3],
            vec![0.01, 0.02, 0.03],
            step,
            vec![3],
        );
        ckpt.add_param_state("layer0.weight", ps);
        ckpt
    }

    // --- OptimizerCheckpoint ---

    #[test]
    fn test_optimizer_checkpoint_new() {
        let ckpt = OptimizerCheckpoint::new("sgd", 42, 3);
        assert_eq!(ckpt.step, 42);
        assert_eq!(ckpt.epoch, 3);
        assert_eq!(ckpt.optimizer_name, "sgd");
    }

    #[test]
    fn test_add_param_state() {
        let mut ckpt = OptimizerCheckpoint::new("adam", 0, 0);
        assert_eq!(ckpt.num_params(), 0);
        let ps = ParamState::new("w", vec![1.0], vec![], 0, vec![1]);
        ckpt.add_param_state("w", ps);
        assert_eq!(ckpt.num_params(), 1);
    }

    #[test]
    fn test_set_get_hyperparam() {
        let mut ckpt = OptimizerCheckpoint::new("adam", 0, 0);
        ckpt.set_hyperparam("lr", 3e-4);
        let retrieved = ckpt.get_hyperparam("lr");
        assert!(retrieved.is_some());
        let diff = (retrieved.unwrap_or(0.0) - 3e-4).abs();
        assert!(diff < 1e-12, "hyperparam roundtrip mismatch");
        assert!(ckpt.get_hyperparam("missing").is_none());
    }

    #[test]
    fn test_total_elements() {
        let mut ckpt = OptimizerCheckpoint::new("adam", 0, 0);
        ckpt.add_param_state(
            "a",
            ParamState::new("a", vec![1.0, 2.0], vec![], 0, vec![2]),
        );
        ckpt.add_param_state(
            "b",
            ParamState::new("b", vec![3.0, 4.0, 5.0], vec![], 0, vec![3]),
        );
        assert_eq!(ckpt.total_elements(), 5);
    }

    // --- Text format serialization ---

    #[test]
    fn test_serialize_text_roundtrip() {
        let ckpt = make_ckpt(100, 2);
        let bytes = serialize_checkpoint(&ckpt, CheckpointFormat::Text).expect("serialize text");
        let loaded =
            deserialize_checkpoint(&bytes, CheckpointFormat::Text).expect("deserialize text");
        assert_eq!(loaded.step, 100);
        assert_eq!(loaded.epoch, 2);
        assert_eq!(loaded.optimizer_name, "adam");
    }

    #[test]
    fn test_serialize_text_param_states() {
        let ckpt = make_ckpt(50, 1);
        let bytes = serialize_checkpoint(&ckpt, CheckpointFormat::Text).expect("serialize");
        let loaded = deserialize_checkpoint(&bytes, CheckpointFormat::Text).expect("deserialize");
        assert_eq!(loaded.num_params(), 1);
        let ps = loaded
            .param_states
            .get("layer0.weight")
            .expect("param not found");
        assert_eq!(ps.first_moment, vec![0.1, 0.2, 0.3]);
        assert_eq!(ps.second_moment, vec![0.01, 0.02, 0.03]);
        assert_eq!(ps.shape, vec![3]);
    }

    // --- Binary format serialization ---

    #[test]
    fn test_serialize_binary_roundtrip() {
        let ckpt = make_ckpt(200, 5);
        let bytes =
            serialize_checkpoint(&ckpt, CheckpointFormat::Binary).expect("serialize binary");
        // Verify magic header.
        assert_eq!(&bytes[..4], &BINARY_MAGIC);
        let loaded =
            deserialize_checkpoint(&bytes, CheckpointFormat::Binary).expect("deserialize binary");
        assert_eq!(loaded.step, 200);
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.optimizer_name, "adam");
    }

    #[test]
    fn test_serialize_hyperparams_roundtrip() {
        let mut ckpt = OptimizerCheckpoint::new("rmsprop", 10, 0);
        ckpt.set_hyperparam("alpha", 0.99);
        ckpt.set_hyperparam("eps", 1e-8);

        for format in [CheckpointFormat::Text, CheckpointFormat::Binary] {
            let bytes = serialize_checkpoint(&ckpt, format.clone()).expect("serialize");
            let loaded = deserialize_checkpoint(&bytes, format).expect("deserialize");
            let alpha = loaded.get_hyperparam("alpha").expect("alpha");
            let eps = loaded.get_hyperparam("eps").expect("eps");
            assert!((alpha - 0.99).abs() < 1e-12);
            assert!((eps - 1e-8).abs() < 1e-20);
        }
    }

    // --- CheckpointManager ---

    fn tmp_dir(suffix: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("tl_ckpt_test_{suffix}_{}", std::process::id()));
        p
    }

    #[test]
    fn test_checkpoint_manager_new_creates_dir() {
        let dir = tmp_dir("new_creates_dir");
        let _mgr =
            CheckpointManager::new(&dir, 3, CheckpointFormat::Text).expect("manager creation");
        assert!(dir.exists(), "directory should have been created");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_save_creates_file() {
        let dir = tmp_dir("save_creates_file");
        let mut mgr = CheckpointManager::new(&dir, 5, CheckpointFormat::Text).expect("manager");
        let ckpt = make_ckpt(1, 0);
        let path = mgr.save(&ckpt).expect("save");
        assert!(path.exists(), "saved file should exist");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_load_latest() {
        let dir = tmp_dir("load_latest");
        let mut mgr = CheckpointManager::new(&dir, 5, CheckpointFormat::Text).expect("manager");
        let ckpt = make_ckpt(7, 1);
        mgr.save(&ckpt).expect("save");
        let loaded = mgr.load_latest().expect("load_latest");
        assert_eq!(loaded.step, 7);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_list() {
        let dir = tmp_dir("list");
        let mut mgr = CheckpointManager::new(&dir, 5, CheckpointFormat::Text).expect("manager");
        mgr.save(&make_ckpt(10, 0)).expect("save 1");
        mgr.save(&make_ckpt(20, 1)).expect("save 2");
        let list = mgr.list();
        assert_eq!(list.len(), 2);
        let steps: Vec<u64> = list.iter().map(|(s, _)| *s).collect();
        assert!(steps.contains(&10));
        assert!(steps.contains(&20));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_max_to_keep() {
        let dir = tmp_dir("max_to_keep");
        let mut mgr = CheckpointManager::new(&dir, 3, CheckpointFormat::Text).expect("manager");
        for step in 0..5_u64 {
            mgr.save(&make_ckpt(step * 10, step as u32)).expect("save");
        }
        assert_eq!(mgr.count(), 3, "only last 3 should be retained");
        let steps: Vec<u64> = mgr.list().iter().map(|(s, _)| *s).collect();
        assert!(steps.contains(&20));
        assert!(steps.contains(&30));
        assert!(steps.contains(&40));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_load_at_step() {
        let dir = tmp_dir("load_at_step");
        let mut mgr = CheckpointManager::new(&dir, 5, CheckpointFormat::Binary).expect("manager");
        mgr.save(&make_ckpt(5, 0)).expect("save");
        mgr.save(&make_ckpt(10, 1)).expect("save");
        let loaded = mgr.load_at_step(5).expect("load step 5");
        assert_eq!(loaded.step, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_manager_no_checkpoints() {
        let dir = tmp_dir("no_checkpoints");
        let mgr = CheckpointManager::new(&dir, 3, CheckpointFormat::Text).expect("manager");
        let result = mgr.load_latest();
        assert!(
            matches!(result, Err(CheckpointError::NoCheckpointsAvailable)),
            "expected NoCheckpointsAvailable, got {result:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- LossTracker ---

    #[test]
    fn test_loss_tracker_moving_average() {
        let mut tracker = LossTracker::new(5);
        tracker.push(1.0);
        tracker.push(2.0);
        tracker.push(3.0);
        let avg = tracker.moving_average().expect("average");
        let diff = (avg - 2.0).abs();
        assert!(diff < 1e-12, "expected 2.0, got {avg}");
    }

    #[test]
    fn test_loss_tracker_min_max() {
        let mut tracker = LossTracker::new(10);
        for v in [5.0, 1.0, 8.0, 3.0_f64] {
            tracker.push(v);
        }
        assert!((tracker.min().expect("min") - 1.0).abs() < 1e-12);
        assert!((tracker.max().expect("max") - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_loss_tracker_is_improving_true() {
        let mut tracker = LossTracker::new(10);
        // Older values are high; recent values are lower → improving.
        for v in [5.0, 4.8, 4.7, 4.9_f64] {
            tracker.push(v);
        }
        // patience=2 → recent = [4.7, 4.9], older = [5.0, 4.8].
        // recent min = 4.7 < older min = 4.8 → true.
        assert!(
            tracker.is_improving(2),
            "expected improving with decreasing loss"
        );
    }

    #[test]
    fn test_loss_tracker_is_improving_false() {
        let mut tracker = LossTracker::new(10);
        // Loss is not decreasing.
        for v in [1.0, 2.0, 3.0, 4.0_f64] {
            tracker.push(v);
        }
        // patience=2 → recent = [3.0, 4.0], older = [1.0, 2.0].
        // recent min = 3.0 > older min = 1.0 → false.
        assert!(
            !tracker.is_improving(2),
            "expected not improving with increasing loss"
        );
    }

    // --- CheckpointError Display ---

    #[test]
    fn test_checkpoint_error_display() {
        let variants: Vec<CheckpointError> = vec![
            CheckpointError::IoError("test io".into()),
            CheckpointError::SerializationError("test ser".into()),
            CheckpointError::DeserializationError("test deser".into()),
            CheckpointError::CheckpointNotFound { step: 42 },
            CheckpointError::NoCheckpointsAvailable,
            CheckpointError::InvalidFormat("bad".into()),
            CheckpointError::DirectoryCreationFailed("dir".into()),
        ];
        for err in &variants {
            let s = err.to_string();
            assert!(
                !s.is_empty(),
                "display output should not be empty for {err:?}"
            );
        }
    }
}
