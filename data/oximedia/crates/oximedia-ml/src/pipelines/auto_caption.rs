//! Whisper-compatible auto-caption pipeline.
//!
//! [`AutoCaptionPipeline`] wires together:
//!
//! 1. **Audio preprocessing** — mono PCM → log-mel spectrogram via
//!    [`oximedia_audio::spectrum::fft::compute_log_mel_spectrogram`].
//! 2. **Encoder ONNX model** — accepts the spectrogram and emits hidden-state
//!    embeddings for the decoder.
//! 3. **Decoder ONNX model** — auto-regressive token generation: at each step
//!    the decoder receives the encoder output and the last emitted token id,
//!    and returns logits over the full vocabulary.
//! 4. **Greedy decode loop** — argmax of logits at each step, stopping at the
//!    EOS token or `max_decode_steps`.
//!
//! Both models must be available as ONNX files; this module is gated behind the
//! `onnx` Cargo feature.  When that feature is absent the types still compile
//! but every constructor returns [`crate::error::MlError::FeatureDisabled`].
//!
//! ## Whisper-compatible defaults
//!
//! [`AutoCaptionConfig::default`] mirrors the Whisper base/small conventions:
//!
//! | Knob             | Value  | Notes                                     |
//! |------------------|--------|-------------------------------------------|
//! | `sample_rate`    | 16 000 | Hz                                        |
//! | `n_mels`         | 80     | Mel filterbank channels                   |
//! | `n_fft`          | 400    | FFT window (25 ms @ 16 kHz)               |
//! | `hop_length`     | 160    | Hop (10 ms @ 16 kHz)                      |
//! | `max_decode_steps`| 448   | Maximum output tokens                     |
//! | `vocab_size`     | 51 865 | Whisper multilingual vocabulary           |
//! | `bos_token`      | 50 258 | `<|startoftranscript|>`                   |
//! | `eos_token`      | 50 257 | `<|endoftext|>`                           |
//!
//! ## Example (no-run, requires ONNX files)
//!
//! ```no_run
//! # #[cfg(feature = "onnx")]
//! # fn demo() -> oximedia_ml::MlResult<()> {
//! use oximedia_ml::pipelines::auto_caption::{AutoCaptionConfig, AutoCaptionPipeline};
//!
//! let cfg = AutoCaptionConfig::default();
//! let pipeline = AutoCaptionPipeline::new(
//!     "whisper-encoder.onnx".as_ref(),
//!     "whisper-decoder.onnx".as_ref(),
//!     cfg,
//! )?;
//!
//! // 3 seconds of silence at 16 kHz.
//! let samples = vec![0.0_f32; 48_000];
//! let tokens = pipeline.caption(&samples)?;
//! println!("decoded {} tokens", tokens.len());
//! # Ok(()) }
//! ```

use std::path::Path;

use crate::device::DeviceType;
use crate::error::{MlError, MlResult};
use crate::model::OnnxModel;

/// Configuration for [`AutoCaptionPipeline`].
///
/// The defaults mirror the Whisper base/small model family.  Override any
/// field when connecting a different encoder-decoder architecture.
#[derive(Clone, Debug)]
pub struct AutoCaptionConfig {
    /// Sample rate of the input PCM audio (Hz). Default: 16 000.
    pub sample_rate: u32,
    /// Number of mel filterbank channels. Default: 80.
    pub n_mels: usize,
    /// FFT window size in samples. Default: 400.
    pub n_fft: usize,
    /// Hop between consecutive spectrogram frames. Default: 160.
    pub hop_length: usize,
    /// Maximum number of decoder steps (tokens generated). Default: 448.
    pub max_decode_steps: usize,
    /// Vocabulary size (number of decoder output classes). Default: 51 865.
    pub vocab_size: usize,
    /// Beginning-of-sequence token id. Default: 50 258 (`<|startoftranscript|>`).
    pub bos_token: u32,
    /// End-of-sequence token id; generation stops when this token is emitted.
    /// Default: 50 257 (`<|endoftext|>`).
    pub eos_token: u32,
    /// Name of the encoder ONNX input tensor. `None` → use the model's first input.
    pub encoder_input_name: Option<String>,
    /// Name of the encoder ONNX output tensor. `None` → use the model's first output.
    pub encoder_output_name: Option<String>,
    /// Name of the decoder token input tensor. `None` → `"token"`.
    pub decoder_token_input_name: Option<String>,
    /// Name of the decoder encoder-state input tensor. `None` → `"encoder_output"`.
    pub decoder_state_input_name: Option<String>,
    /// Name of the decoder logits output tensor. `None` → use the model's first output.
    pub decoder_logits_output_name: Option<String>,
}

impl Default for AutoCaptionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            max_decode_steps: 448,
            vocab_size: 51_865,
            bos_token: 50_258,
            eos_token: 50_257,
            encoder_input_name: None,
            encoder_output_name: None,
            decoder_token_input_name: None,
            decoder_state_input_name: None,
            decoder_logits_output_name: None,
        }
    }
}

/// End-to-end audio caption pipeline.
///
/// Combines a spectrogram encoder and an auto-regressive decoder ONNX model
/// to produce a sequence of token ids from raw PCM audio samples.
///
/// Gate this type behind `#[cfg(feature = "onnx")]` in downstream code, or
/// handle the [`MlError::FeatureDisabled`] error returned by [`Self::new`]
/// when the feature is absent.
pub struct AutoCaptionPipeline {
    encoder: OnnxModel,
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    decoder: OnnxModel,
    config: AutoCaptionConfig,
}

impl std::fmt::Debug for AutoCaptionPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoCaptionPipeline")
            .field("n_mels", &self.config.n_mels)
            .field("vocab_size", &self.config.vocab_size)
            .field("max_decode_steps", &self.config.max_decode_steps)
            .finish()
    }
}

impl AutoCaptionPipeline {
    /// Load encoder and decoder ONNX models from disk.
    ///
    /// Both models are loaded onto the CPU device so the pipeline works
    /// everywhere including WASM.  Pass explicit [`DeviceType`] values to
    /// [`Self::new_with_device`] when GPU execution is desired.
    ///
    /// # Errors
    ///
    /// * [`MlError::FeatureDisabled`] when the `onnx` feature is not compiled in.
    /// * [`MlError::ModelLoad`] when either ONNX file cannot be opened or parsed.
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        config: AutoCaptionConfig,
    ) -> MlResult<Self> {
        Self::new_with_device(
            encoder_path,
            decoder_path,
            config,
            DeviceType::Cpu,
            DeviceType::Cpu,
        )
    }

    /// Load encoder and decoder ONNX models with explicit execution devices.
    ///
    /// # Errors
    ///
    /// * [`MlError::FeatureDisabled`] when the `onnx` feature is not compiled in.
    /// * [`MlError::ModelLoad`] when either ONNX file cannot be opened or parsed.
    /// * [`MlError::DeviceUnavailable`] when a requested device is not present.
    pub fn new_with_device(
        encoder_path: &Path,
        decoder_path: &Path,
        config: AutoCaptionConfig,
        encoder_device: DeviceType,
        decoder_device: DeviceType,
    ) -> MlResult<Self> {
        let encoder = OnnxModel::load(encoder_path, encoder_device)?;
        let decoder = OnnxModel::load(decoder_path, decoder_device)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Encode raw PCM samples into a flat embedding vector.
    ///
    /// Runs the full preprocessing pipeline (log-mel spectrogram) and the
    /// encoder ONNX model's forward pass.  The returned vector is the raw
    /// `f32` buffer of the encoder's first (or configured) output tensor;
    /// callers may inspect its shape via the encoder model info if needed.
    ///
    /// # Errors
    ///
    /// * [`MlError::InvalidInput`] when `samples` is empty.
    /// * [`MlError::Pipeline`] when spectrogram computation produces no output.
    /// * [`MlError::OnnxRuntime`] on inference failure.
    pub fn encode_audio(&self, samples: &[f32]) -> MlResult<Vec<f32>> {
        if samples.is_empty() {
            return Err(MlError::invalid_input(
                "encode_audio: samples must not be empty",
            ));
        }

        let log_mel = oximedia_audio::spectrum::fft::compute_log_mel_spectrogram(
            samples,
            self.config.sample_rate,
            self.config.n_mels,
            self.config.n_fft,
            self.config.hop_length,
        );

        if log_mel.is_empty() {
            return Err(MlError::pipeline(
                "auto-caption-encode",
                "compute_log_mel_spectrogram returned an empty tensor",
            ));
        }

        // Determine the number of frames from the log-mel output.
        let n_frames = log_mel.len() / self.config.n_mels;

        // Input tensor name: config override or model's first input.
        let input_name = self
            .config
            .encoder_input_name
            .clone()
            .or_else(|| self.encoder.info().inputs.first().map(|s| s.name.clone()))
            .unwrap_or_else(|| "input".to_string());

        // Shape: [batch=1, n_mels, n_frames] — standard Whisper encoder convention.
        let shape = vec![1, self.config.n_mels, n_frames];
        let outputs = self
            .encoder
            .run_single(input_name.as_str(), log_mel, shape)?;

        // Output tensor name: config override or model's first output.
        let output_name = self
            .config
            .encoder_output_name
            .clone()
            .or_else(|| self.encoder.info().outputs.first().map(|s| s.name.clone()))
            .unwrap_or_else(|| "output".to_string());

        outputs.get(&output_name).cloned().ok_or_else(|| {
            MlError::postprocess(format!(
                "encode_audio: encoder output '{output_name}' not found in model run results"
            ))
        })
    }

    /// Run one decoder step.
    ///
    /// Takes the last generated `token_id` and the `encoder_output` produced by
    /// [`Self::encode_audio`]; returns a logits vector of length `vocab_size`.
    ///
    /// The decoder receives two inputs:
    /// * A scalar token input (shape `[1, 1]`) with the current token id.
    /// * The encoder hidden state (shape determined by the encoder output).
    ///
    /// # Errors
    ///
    /// * [`MlError::OnnxRuntime`] on inference failure.
    /// * [`MlError::Postprocess`] when the expected output tensor is absent.
    #[cfg(feature = "onnx")]
    pub fn step_decode(&self, token_id: u32, encoder_output: &[f32]) -> MlResult<Vec<f32>> {
        use oxionnx::Tensor;
        use std::collections::HashMap;

        let token_input_name = self
            .config
            .decoder_token_input_name
            .as_deref()
            .unwrap_or("token");
        let state_input_name = self
            .config
            .decoder_state_input_name
            .as_deref()
            .unwrap_or("encoder_output");

        // Token tensor: shape [1, 1], single scalar cast to f32.
        let token_tensor = Tensor {
            data: vec![token_id as f32],
            shape: vec![1, 1],
        };

        // Encoder-output tensor: shape determined by encoder run.
        let enc_len = encoder_output.len();
        let state_tensor = Tensor {
            data: encoder_output.to_vec(),
            shape: vec![1, enc_len],
        };

        let mut inputs: HashMap<&str, Tensor> = HashMap::with_capacity(2);
        inputs.insert(token_input_name, token_tensor);
        inputs.insert(state_input_name, state_tensor);

        let outputs = self.decoder.run(&inputs)?;

        let logits_name = self
            .config
            .decoder_logits_output_name
            .clone()
            .or_else(|| self.decoder.info().outputs.first().map(|s| s.name.clone()))
            .unwrap_or_else(|| "logits".to_string());

        outputs
            .get(&logits_name)
            .map(|t| t.data.clone())
            .ok_or_else(|| {
                MlError::postprocess(format!(
                    "step_decode: decoder output '{logits_name}' not found in model run results"
                ))
            })
    }

    /// Stub for non-`onnx` builds.
    ///
    /// Always returns [`MlError::FeatureDisabled`].
    #[cfg(not(feature = "onnx"))]
    pub fn step_decode(&self, _token_id: u32, _encoder_output: &[f32]) -> MlResult<Vec<f32>> {
        Err(MlError::FeatureDisabled("onnx"))
    }

    /// Greedy-decode a caption from raw PCM audio.
    ///
    /// Performs:
    /// 1. `encode_audio(samples)` → encoder hidden states.
    /// 2. Prime the decoder with the BOS token.
    /// 3. Loop: `step_decode(last_token, encoder_output)` → logits → argmax → push token.
    /// 4. Stop when the EOS token is generated or `max_decode_steps` is reached.
    ///
    /// Returns the generated token id sequence (excluding the initial BOS token).
    ///
    /// # Errors
    ///
    /// * Any error propagated from [`Self::encode_audio`] or [`Self::step_decode`].
    /// * [`MlError::Postprocess`] when logits for a step are too short for the
    ///   configured `vocab_size`.
    pub fn caption(&self, samples: &[f32]) -> MlResult<Vec<u32>> {
        let encoder_output = self.encode_audio(samples)?;

        let mut tokens: Vec<u32> = Vec::with_capacity(self.config.max_decode_steps);
        let mut last_token = self.config.bos_token;

        for _step in 0..self.config.max_decode_steps {
            let logits = self.step_decode(last_token, &encoder_output)?;

            if logits.len() < self.config.vocab_size {
                return Err(MlError::postprocess(format!(
                    "caption: logits length {} is less than vocab_size {}",
                    logits.len(),
                    self.config.vocab_size,
                )));
            }

            // Greedy: argmax over the first vocab_size entries.
            let vocab_logits = &logits[..self.config.vocab_size];
            let next_token = crate::postprocess::argmax(vocab_logits)
                .map_err(|e| MlError::postprocess(format!("caption: argmax failed: {e:?}")))?;

            let next_token_u32 = u32::try_from(next_token).map_err(|_| {
                MlError::postprocess(format!(
                    "caption: token index {next_token} exceeds u32::MAX"
                ))
            })?;

            tokens.push(next_token_u32);

            if next_token_u32 == self.config.eos_token {
                break;
            }

            last_token = next_token_u32;
        }

        Ok(tokens)
    }

    /// Read-only view of the pipeline configuration.
    #[must_use]
    pub fn config(&self) -> &AutoCaptionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_whisper_values() {
        let cfg = AutoCaptionConfig::default();
        assert_eq!(cfg.sample_rate, 16_000, "sample_rate");
        assert_eq!(cfg.n_mels, 80, "n_mels");
        assert_eq!(cfg.n_fft, 400, "n_fft");
        assert_eq!(cfg.hop_length, 160, "hop_length");
        assert_eq!(cfg.max_decode_steps, 448, "max_decode_steps");
        assert_eq!(cfg.vocab_size, 51_865, "vocab_size");
        assert_eq!(cfg.bos_token, 50_258, "bos_token");
        assert_eq!(cfg.eos_token, 50_257, "eos_token");
    }

    /// `AutoCaptionPipeline::new` must return `Err` when given non-existent
    /// model paths.  The error kind depends on feature availability:
    /// * `onnx` enabled → `MlError::ModelLoad` (file not found).
    /// * `onnx` disabled → `MlError::FeatureDisabled`.
    #[test]
    fn new_with_missing_paths_returns_err() {
        let tmp = std::env::temp_dir();
        let enc = tmp.join("oximedia-ml-autocaption-nonexistent-enc.onnx");
        let dec = tmp.join("oximedia-ml-autocaption-nonexistent-dec.onnx");
        // Ensure they really do not exist.
        let _ = std::fs::remove_file(&enc);
        let _ = std::fs::remove_file(&dec);

        let result = AutoCaptionPipeline::new(&enc, &dec, AutoCaptionConfig::default());
        assert!(
            result.is_err(),
            "expected Err for missing ONNX paths, got Ok"
        );
    }

    #[test]
    fn config_clone_and_debug_do_not_panic() {
        let cfg = AutoCaptionConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.sample_rate, cfg2.sample_rate);
        // Debug impl exercised via format string.
        let _ = format!("{cfg:?}");
    }
}
