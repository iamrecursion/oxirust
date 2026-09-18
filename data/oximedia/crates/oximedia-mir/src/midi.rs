//! Audio-to-MIDI basic transcription.
//!
//! Converts a mono audio signal into a sequence of MIDI note events by:
//!
//! 1. **Onset detection** — spectral-flux based detection of note attacks.
//! 2. **Pitch estimation** — per-frame autocorrelation F0 tracking between
//!    detected onset and offset boundaries.
//! 3. **Note grouping** — consecutive frames with the same (or harmonically
//!    similar) pitch are merged into a single [`MidiNote`].
//! 4. **Velocity mapping** — frame RMS energy is mapped to MIDI velocity
//!    (1–127).
//!
//! # Limitations
//!
//! This is a *monophonic* transcriber: at any given time only one note is
//! active.  Polyphonic signals will be converted by tracking the dominant F0.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_mir::midi::{AudioToMidi, AudioToMidiConfig};
//!
//! let config = AudioToMidiConfig::default();
//! let transcriber = AudioToMidi::new(config);
//! let samples = vec![0.0_f32; 44100];
//! let notes = transcriber.transcribe(&samples, 44100.0).unwrap();
//! println!("Detected {} notes", notes.notes.len());
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use crate::{MirError, MirResult};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single MIDI note event produced by the transcriber.
#[derive(Debug, Clone, PartialEq)]
pub struct MidiNote {
    /// MIDI note number (0–127, A4 = 69).
    pub note: u8,
    /// MIDI velocity (1–127).
    pub velocity: u8,
    /// Note-on time in seconds.
    pub start_secs: f32,
    /// Note-off time in seconds.
    pub end_secs: f32,
    /// Duration of the note in seconds.
    pub duration_secs: f32,
    /// Fundamental frequency in Hz (before rounding to MIDI).
    pub frequency_hz: f32,
    /// Pitch confidence (0–1) averaged over the note's frames.
    pub confidence: f32,
    /// MIDI channel (0-indexed, always 0 for monophonic output).
    pub channel: u8,
}

impl MidiNote {
    /// Return the note name (e.g., `"A4"`, `"C#3"`).
    #[must_use]
    pub fn name(&self) -> String {
        const NAMES: [&str; 12] = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        let octave = (self.note as i32 / 12) - 1;
        let pc = (self.note % 12) as usize;
        format!("{}{octave}", NAMES[pc])
    }
}

/// Tempo metadata accompanying the MIDI event list.
#[derive(Debug, Clone)]
pub struct MidiTempo {
    /// Tempo in beats-per-minute.
    pub bpm: f32,
    /// Microseconds per quarter note (standard MIDI representation).
    pub us_per_quarter: u32,
}

impl MidiTempo {
    /// Create from BPM.
    #[must_use]
    pub fn from_bpm(bpm: f32) -> Self {
        let us = if bpm > 0.0 {
            (60_000_000.0 / bpm).round() as u32
        } else {
            500_000 // 120 BPM fallback
        };
        Self {
            bpm,
            us_per_quarter: us,
        }
    }
}

/// A complete MIDI transcription result.
#[derive(Debug, Clone)]
pub struct MidiTranscription {
    /// Detected note events, sorted by `start_secs`.
    pub notes: Vec<MidiNote>,
    /// Tempo estimate (may be 0 BPM if undetected).
    pub tempo: MidiTempo,
    /// Number of MIDI ticks per quarter note (resolution).
    pub ticks_per_quarter: u16,
    /// Total duration of the audio in seconds.
    pub total_duration_secs: f32,
}

impl MidiTranscription {
    /// Serialise to the standard MIDI file byte format (SMF type 0).
    ///
    /// The output is a minimal, valid Type-0 (single-track) MIDI file.
    ///
    /// # Errors
    ///
    /// Returns error if note data is malformed (e.g., note number out of range).
    pub fn to_midi_bytes(&self) -> MirResult<Vec<u8>> {
        let tpq = self.ticks_per_quarter;
        let us_per_q = self.tempo.us_per_quarter;

        // ── Collect & sort all note-on / note-off events ────────────────────
        #[derive(Debug)]
        struct RawEvent {
            tick: u32,
            // true = note-on, false = note-off
            is_on: bool,
            note: u8,
            velocity: u8,
        }

        let mut events: Vec<RawEvent> = Vec::with_capacity(self.notes.len() * 2);
        for note in &self.notes {
            if note.note > 127 {
                return Err(MirError::InvalidInput(format!(
                    "MIDI note number {} out of range [0, 127]",
                    note.note
                )));
            }
            // Convert seconds → ticks
            let bpm = self.tempo.bpm.max(1.0);
            let ticks_per_sec = bpm / 60.0 * tpq as f32;
            let on_tick = (note.start_secs * ticks_per_sec).round() as u32;
            let off_tick = (note.end_secs * ticks_per_sec).round() as u32;
            events.push(RawEvent {
                tick: on_tick,
                is_on: true,
                note: note.note,
                velocity: note.velocity,
            });
            events.push(RawEvent {
                tick: off_tick.max(on_tick + 1),
                is_on: false,
                note: note.note,
                velocity: 0,
            });
        }
        events.sort_by_key(|e| e.tick);

        // ── Build track chunk ────────────────────────────────────────────────
        let mut track: Vec<u8> = Vec::new();

        // Tempo meta-event: FF 51 03 tt tt tt
        track.extend_from_slice(&[0x00, 0xFF, 0x51, 0x03]);
        track.push(((us_per_q >> 16) & 0xFF) as u8);
        track.push(((us_per_q >> 8) & 0xFF) as u8);
        track.push((us_per_q & 0xFF) as u8);

        let mut current_tick = 0u32;
        for ev in &events {
            let delta = ev.tick.saturating_sub(current_tick);
            current_tick = ev.tick;
            // Variable-length quantity encoding
            write_vlq(&mut track, delta);
            let status = if ev.is_on { 0x90 } else { 0x80 };
            track.push(status); // channel 0
            track.push(ev.note);
            track.push(ev.velocity);
        }

        // End-of-track: delta=0, FF 2F 00
        track.extend_from_slice(&[0x00, 0xFF, 0x2F, 0x00]);

        // ── Assemble SMF ─────────────────────────────────────────────────────
        let mut smf: Vec<u8> = Vec::with_capacity(14 + 8 + track.len());

        // MThd
        smf.extend_from_slice(b"MThd");
        smf.extend_from_slice(&6u32.to_be_bytes()); // chunk length
        smf.extend_from_slice(&0u16.to_be_bytes()); // format 0
        smf.extend_from_slice(&1u16.to_be_bytes()); // 1 track
        smf.extend_from_slice(&tpq.to_be_bytes()); // ticks per quarter note

        // MTrk
        smf.extend_from_slice(b"MTrk");
        smf.extend_from_slice(&(track.len() as u32).to_be_bytes());
        smf.extend_from_slice(&track);

        Ok(smf)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the audio-to-MIDI transcriber.
#[derive(Debug, Clone)]
pub struct AudioToMidiConfig {
    /// FFT window size for spectral analysis.
    pub window_size: usize,
    /// Hop size between frames.
    pub hop_size: usize,
    /// Minimum detectable pitch frequency in Hz.
    pub min_freq: f32,
    /// Maximum detectable pitch frequency in Hz.
    pub max_freq: f32,
    /// Voicing confidence threshold (0–1); frames below this are treated as
    /// rests.
    pub voicing_threshold: f32,
    /// Minimum note duration in seconds; shorter notes are discarded.
    pub min_note_duration_secs: f32,
    /// Maximum pitch deviation (in semitones) for merging adjacent frames
    /// into a single note.
    pub pitch_merge_semitones: f32,
    /// Onset detection sensitivity (higher → more onsets detected).
    pub onset_sensitivity: f64,
    /// Sliding-median window length (in frames) for adaptive onset threshold.
    pub onset_median_window: usize,
    /// MIDI ticks per quarter note (resolution).
    pub ticks_per_quarter: u16,
}

impl Default for AudioToMidiConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            hop_size: 512,
            min_freq: 50.0,
            max_freq: 4000.0,
            voicing_threshold: 0.25,
            min_note_duration_secs: 0.05,
            pitch_merge_semitones: 0.7,
            onset_sensitivity: 1.2,
            onset_median_window: 11,
            ticks_per_quarter: 480,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio-to-MIDI transcriber
// ---------------------------------------------------------------------------

/// Monophonic audio-to-MIDI transcriber.
pub struct AudioToMidi {
    config: AudioToMidiConfig,
}

impl AudioToMidi {
    /// Create a new transcriber with the given configuration.
    #[must_use]
    pub fn new(config: AudioToMidiConfig) -> Self {
        Self { config }
    }

    /// Create a transcriber with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(AudioToMidiConfig::default())
    }

    /// Transcribe a mono audio signal into MIDI notes.
    ///
    /// # Arguments
    ///
    /// * `samples` — Mono f32 audio samples (amplitude in [-1, 1]).
    /// * `sample_rate` — Sample rate in Hz.
    ///
    /// # Errors
    ///
    /// Returns [`MirError::InsufficientData`] if the signal is shorter than
    /// one analysis window.
    pub fn transcribe(&self, samples: &[f32], sample_rate: f32) -> MirResult<MidiTranscription> {
        if samples.len() < self.config.window_size {
            return Err(MirError::InsufficientData(format!(
                "Signal too short for MIDI transcription: need ≥{} samples, got {}",
                self.config.window_size,
                samples.len()
            )));
        }

        let win = self.config.window_size;
        let hop = self.config.hop_size;
        let sr = sample_rate;

        // ── Step 1: Per-frame analysis ────────────────────────────────────
        let frames = self.analyse_frames(samples, sr)?;

        // ── Step 2: Onset detection over frame energies ───────────────────
        let onset_flags = self.detect_onsets(&frames);

        // ── Step 3: Segment frames into note regions ──────────────────────
        let segments = segment_into_notes(&frames, &onset_flags, &self.config, sr, hop, win);

        // ── Step 4: Estimate global tempo from onset intervals ────────────
        let bpm = estimate_bpm_from_onsets(&frames, &onset_flags, sr, hop);
        let tempo = MidiTempo::from_bpm(bpm);

        let total_duration_secs = samples.len() as f32 / sr;

        Ok(MidiTranscription {
            notes: segments,
            tempo,
            ticks_per_quarter: self.config.ticks_per_quarter,
            total_duration_secs,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Compute per-frame pitch + energy features.
    fn analyse_frames(&self, samples: &[f32], sample_rate: f32) -> MirResult<Vec<FrameData>> {
        let win = self.config.window_size;
        let hop = self.config.hop_size;
        let min_lag = (sample_rate / self.config.max_freq).floor() as usize;
        let max_lag = (sample_rate / self.config.min_freq).ceil() as usize;

        if min_lag == 0 || max_lag < min_lag {
            return Err(MirError::InvalidInput(
                "Invalid frequency bounds for MIDI transcription".to_string(),
            ));
        }

        let n_frames = (samples.len().saturating_sub(win)) / hop + 1;
        let mut frames = Vec::with_capacity(n_frames);

        let window = crate::utils::hann_window(win);

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop;
            let end = (start + win).min(samples.len());
            if end <= start {
                break;
            }
            let frame = &samples[start..end];

            // RMS energy
            let rms = {
                let sq: f32 = frame.iter().map(|&s| s * s).sum();
                (sq / frame.len() as f32).sqrt()
            };

            // Windowed autocorrelation pitch estimate
            let (freq_hz, pitch_conf) = Self::estimate_pitch_autocorr(
                frame,
                &window[..frame.len()],
                min_lag,
                max_lag,
                sample_rate,
            );

            let time_secs = (start + win / 2) as f32 / sample_rate;
            frames.push(FrameData {
                time_secs,
                rms,
                frequency_hz: freq_hz,
                pitch_confidence: pitch_conf,
            });
        }

        Ok(frames)
    }

    /// Autocorrelation-based pitch estimation for a single frame.
    ///
    /// Returns `(frequency_hz, confidence)`.
    fn estimate_pitch_autocorr(
        frame: &[f32],
        window: &[f32],
        min_lag: usize,
        max_lag: usize,
        sample_rate: f32,
    ) -> (f32, f32) {
        let n = frame.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        // Apply Hann window and compute energy
        let mut windowed = Vec::with_capacity(n);
        let mut energy = 0.0_f32;
        for i in 0..n {
            let w = if i < window.len() { window[i] } else { 1.0 };
            let v = frame[i] * w;
            windowed.push(v);
            energy += v * v;
        }
        if energy < 1e-12 {
            return (0.0, 0.0);
        }

        let max_lag_clamped = max_lag.min(n - 1);
        if min_lag >= n {
            return (0.0, 0.0);
        }

        // Normalised autocorrelation (AMDF-free, Pearson-style)
        let mut best_corr = -1.0_f32;
        let mut best_lag = 0usize;

        for lag in min_lag..=max_lag_clamped {
            let m = n - lag;
            let (mut num, mut da, mut db) = (0.0_f32, 0.0_f32, 0.0_f32);
            for j in 0..m {
                num += windowed[j] * windowed[j + lag];
                da += windowed[j] * windowed[j];
                db += windowed[j + lag] * windowed[j + lag];
            }
            let denom = (da * db).sqrt();
            let corr = if denom > 1e-12 { num / denom } else { 0.0 };
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        let confidence = best_corr.max(0.0);
        if best_lag == 0 {
            return (0.0, confidence);
        }
        let freq = sample_rate / best_lag as f32;
        (freq, confidence)
    }

    /// Detect onset frames using spectral flux (energy-based) with adaptive threshold.
    fn detect_onsets(&self, frames: &[FrameData]) -> Vec<bool> {
        if frames.is_empty() {
            return Vec::new();
        }

        // Spectral-flux proxy: positive first-difference of frame RMS energy
        let n = frames.len();
        let mut flux = vec![0.0_f64; n];
        flux[0] = frames[0].rms as f64;
        for i in 1..n {
            let diff = frames[i].rms as f64 - frames[i - 1].rms as f64;
            flux[i] = if diff > 0.0 { diff } else { 0.0 };
        }

        // Sliding-median adaptive threshold
        let half = self.config.onset_median_window / 2;
        let mut threshold = vec![0.0_f64; n];
        for i in 0..n {
            let lo = if i >= half { i - half } else { 0 };
            let hi = (i + half + 1).min(n);
            let mut window: Vec<f64> = flux[lo..hi].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = window[window.len() / 2];
            threshold[i] = median * self.config.onset_sensitivity;
        }

        // Mark onsets with minimum-distance suppression (4 frames)
        const MIN_DIST: usize = 4;
        let mut onsets = vec![false; n];
        let mut last_onset = 0usize;
        for i in 1..n.saturating_sub(1) {
            if flux[i] > flux[i - 1]
                && flux[i] >= flux[i + 1]
                && flux[i] > threshold[i]
                && (i == 0 || i - last_onset >= MIN_DIST)
            {
                onsets[i] = true;
                last_onset = i;
            }
        }
        onsets
    }
}

// ---------------------------------------------------------------------------
// Internal frame data
// ---------------------------------------------------------------------------

/// Per-frame analysis data (pitch + energy).
#[derive(Debug, Clone)]
struct FrameData {
    /// Centre time of the frame in seconds.
    time_secs: f32,
    /// RMS amplitude.
    rms: f32,
    /// Dominant frequency in Hz (0 = unvoiced / silence).
    frequency_hz: f32,
    /// Pitch detection confidence (0–1).
    pitch_confidence: f32,
}

// ---------------------------------------------------------------------------
// Note segmentation
// ---------------------------------------------------------------------------

/// Group frames into MIDI notes.
///
/// A new note begins at each detected onset, or whenever the pitch changes
/// by more than `config.pitch_merge_semitones`.
#[allow(clippy::cast_precision_loss)]
fn segment_into_notes(
    frames: &[FrameData],
    onset_flags: &[bool],
    config: &AudioToMidiConfig,
    sample_rate: f32,
    hop: usize,
    win: usize,
) -> Vec<MidiNote> {
    if frames.is_empty() {
        return Vec::new();
    }

    let frame_dur = hop as f32 / sample_rate;
    let mut notes: Vec<MidiNote> = Vec::new();

    // State machine: accumulate frames for the current note candidate
    let mut note_start = 0usize; // frame index of note start
    let mut note_freq_sum = 0.0_f32;
    let mut note_conf_sum = 0.0_f32;
    let mut note_rms_sum = 0.0_f32;
    let mut note_frame_count = 0usize;
    let mut current_midi: Option<u8> = None;
    let mut in_note = false;

    let flush = |start_idx: usize,
                 n_frames: usize,
                 freq_sum: f32,
                 conf_sum: f32,
                 rms_sum: f32,
                 frames: &[FrameData],
                 notes: &mut Vec<MidiNote>,
                 config: &AudioToMidiConfig,
                 frame_dur: f32,
                 _hop: usize,
                 win: usize,
                 sample_rate: f32| {
        if n_frames == 0 {
            return;
        }
        let avg_freq = freq_sum / n_frames as f32;
        let avg_conf = conf_sum / n_frames as f32;
        let avg_rms = rms_sum / n_frames as f32;

        if avg_conf < config.voicing_threshold || avg_freq <= 0.0 {
            return;
        }

        let start_secs = frames[start_idx].time_secs - win as f32 / (2.0 * sample_rate);
        let end_secs = start_secs + n_frames as f32 * frame_dur;
        let dur = end_secs - start_secs;

        if dur < config.min_note_duration_secs {
            return;
        }

        let midi_float = 69.0 + 12.0 * (avg_freq / 440.0).log2();
        let midi_note_num = midi_float.round() as i32;
        if !(0..=127).contains(&midi_note_num) {
            return;
        }

        // Map RMS → velocity (1–127)
        let velocity = (avg_rms * 127.0 * 8.0).clamp(1.0, 127.0) as u8;

        notes.push(MidiNote {
            note: midi_note_num as u8,
            velocity,
            start_secs: start_secs.max(0.0),
            end_secs,
            duration_secs: dur,
            frequency_hz: avg_freq,
            confidence: avg_conf,
            channel: 0,
        });
    };

    for (i, frame) in frames.iter().enumerate() {
        let is_voiced =
            frame.pitch_confidence >= config.voicing_threshold && frame.frequency_hz > 0.0;
        let is_onset = onset_flags.get(i).copied().unwrap_or(false);

        // Compute MIDI note number for current frame
        let frame_midi: Option<u8> = if is_voiced {
            let m = (69.0 + 12.0 * (frame.frequency_hz / 440.0).log2()).round() as i32;
            if (0..=127).contains(&m) {
                Some(m as u8)
            } else {
                None
            }
        } else {
            None
        };

        // Pitch-change boundary
        let pitch_changed = match (current_midi, frame_midi) {
            (Some(cur), Some(new_m)) => {
                (cur as f32 - new_m as f32).abs() > config.pitch_merge_semitones
            }
            (Some(_), None) | (None, Some(_)) => true,
            (None, None) => false,
        };

        // Start a new note on onset or significant pitch change
        if (is_onset || pitch_changed) && in_note {
            flush(
                note_start,
                note_frame_count,
                note_freq_sum,
                note_conf_sum,
                note_rms_sum,
                frames,
                &mut notes,
                config,
                frame_dur,
                hop,
                win,
                sample_rate,
            );
            note_start = i;
            note_freq_sum = 0.0;
            note_conf_sum = 0.0;
            note_rms_sum = 0.0;
            note_frame_count = 0;
            in_note = false;
            current_midi = None;
        }

        if is_voiced {
            if !in_note {
                note_start = i;
                in_note = true;
                current_midi = frame_midi;
            }
            note_freq_sum += frame.frequency_hz;
            note_conf_sum += frame.pitch_confidence;
            note_rms_sum += frame.rms;
            note_frame_count += 1;
        } else if in_note {
            // Unvoiced frame: flush current note
            flush(
                note_start,
                note_frame_count,
                note_freq_sum,
                note_conf_sum,
                note_rms_sum,
                frames,
                &mut notes,
                config,
                frame_dur,
                hop,
                win,
                sample_rate,
            );
            in_note = false;
            note_frame_count = 0;
            note_freq_sum = 0.0;
            note_conf_sum = 0.0;
            note_rms_sum = 0.0;
            current_midi = None;
        }
    }

    // Flush the last note
    if in_note {
        flush(
            note_start,
            note_frame_count,
            note_freq_sum,
            note_conf_sum,
            note_rms_sum,
            frames,
            &mut notes,
            config,
            frame_dur,
            hop,
            win,
            sample_rate,
        );
    }

    notes.sort_by(|a, b| {
        a.start_secs
            .partial_cmp(&b.start_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    notes
}

// ---------------------------------------------------------------------------
// Tempo estimation from onset intervals
// ---------------------------------------------------------------------------

/// Estimate global BPM from inter-onset intervals via a histogram vote.
fn estimate_bpm_from_onsets(
    frames: &[FrameData],
    onset_flags: &[bool],
    sample_rate: f32,
    hop: usize,
) -> f32 {
    let frame_dur = hop as f32 / sample_rate;
    let onset_times: Vec<f32> = onset_flags
        .iter()
        .enumerate()
        .filter(|(_, &on)| on)
        .map(|(i, _)| frames[i].time_secs.max(0.0) + frame_dur * 0.5)
        .collect();

    if onset_times.len() < 2 {
        return 120.0; // Default
    }

    // Inter-onset intervals (seconds)
    let iois: Vec<f32> = onset_times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d > 0.1 && d < 4.0) // plausible beat range
        .collect();

    if iois.is_empty() {
        return 120.0;
    }

    // Histogram vote over BPM bins (40–240 BPM, 1-BPM resolution)
    let mut histogram = vec![0u32; 201]; // index 0 → 40 BPM, 200 → 240 BPM
    for &ioi in &iois {
        let bpm = 60.0 / ioi;
        if bpm < 40.0 || bpm > 240.0 {
            continue;
        }
        let bin = (bpm - 40.0).round() as usize;
        if bin < histogram.len() {
            histogram[bin] += 1;
        }
        // Also vote for half- and double-time
        let half = bpm / 2.0;
        if half >= 40.0 && half <= 240.0 {
            let hb = (half - 40.0).round() as usize;
            if hb < histogram.len() {
                histogram[hb] += 1;
            }
        }
        let dbl = bpm * 2.0;
        if dbl >= 40.0 && dbl <= 240.0 {
            let db = (dbl - 40.0).round() as usize;
            if db < histogram.len() {
                histogram[db] += 1;
            }
        }
    }

    let best_bin = histogram
        .iter()
        .enumerate()
        .max_by_key(|&(_, &v)| v)
        .map_or(80, |(i, _)| i); // default 120 BPM

    40.0 + best_bin as f32
}

// ---------------------------------------------------------------------------
// VLQ helper
// ---------------------------------------------------------------------------

/// Write a MIDI variable-length quantity into `buf`.
fn write_vlq(buf: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        buf.push(0);
        return;
    }
    let mut bytes = [0u8; 5];
    let mut len = 0usize;
    while value > 0 {
        bytes[len] = (value & 0x7F) as u8;
        value >>= 7;
        len += 1;
    }
    for i in (0..len).rev() {
        let b = if i > 0 { bytes[i] | 0x80 } else { bytes[i] };
        buf.push(b);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn sine(freq: f32, sr: f32, secs: f32) -> Vec<f32> {
        let n = (sr * secs) as usize;
        (0..n).map(|i| (TAU * freq * i as f32 / sr).sin()).collect()
    }

    fn silence(n: usize) -> Vec<f32> {
        vec![0.0; n]
    }

    // ── MidiNote ─────────────────────────────────────────────────────────

    #[test]
    fn test_midi_note_name_a4() {
        let note = MidiNote {
            note: 69,
            velocity: 80,
            start_secs: 0.0,
            end_secs: 0.5,
            duration_secs: 0.5,
            frequency_hz: 440.0,
            confidence: 0.9,
            channel: 0,
        };
        assert_eq!(note.name(), "A4");
    }

    #[test]
    fn test_midi_note_name_c3() {
        let note = MidiNote {
            note: 48,
            velocity: 80,
            start_secs: 0.0,
            end_secs: 0.5,
            duration_secs: 0.5,
            frequency_hz: 130.81,
            confidence: 0.9,
            channel: 0,
        };
        assert_eq!(note.name(), "C3");
    }

    #[test]
    fn test_midi_tempo_from_bpm() {
        let t = MidiTempo::from_bpm(120.0);
        assert_eq!(t.us_per_quarter, 500_000);
    }

    #[test]
    fn test_midi_tempo_from_bpm_zero() {
        let t = MidiTempo::from_bpm(0.0);
        // Fallback is 500_000 (120 BPM)
        assert_eq!(t.us_per_quarter, 500_000);
    }

    // ── AudioToMidi ───────────────────────────────────────────────────────

    #[test]
    fn test_transcribe_silence() {
        let t = AudioToMidi::default_config();
        let sig = silence(44100 * 2);
        let result = t.transcribe(&sig, 44100.0).expect("should succeed");
        // Silence → no voiced frames → no notes
        assert!(result.notes.is_empty());
    }

    #[test]
    fn test_transcribe_short_signal_error() {
        let t = AudioToMidi::default_config();
        let sig = silence(100);
        let result = t.transcribe(&sig, 44100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_transcribe_sine_produces_note() {
        // A 440 Hz sine should produce at least one voiced MIDI note
        let sr = 22050.0;
        let cfg = AudioToMidiConfig {
            window_size: 1024,
            hop_size: 256,
            min_freq: 200.0,
            max_freq: 1000.0,
            voicing_threshold: 0.3,
            min_note_duration_secs: 0.02,
            ..AudioToMidiConfig::default()
        };
        let t = AudioToMidi::new(cfg);
        let sig = sine(440.0, sr, 1.0);
        let result = t.transcribe(&sig, sr).expect("should succeed");
        assert!(
            !result.notes.is_empty(),
            "Expected at least one note for 440 Hz sine, got none"
        );
        // The note should be approximately A4 (MIDI 69)
        let note = &result.notes[0];
        assert!(
            (note.note as i32 - 69).abs() <= 2,
            "Expected MIDI ~69, got {}",
            note.note
        );
        assert!(note.duration_secs >= 0.02);
    }

    #[test]
    fn test_transcribe_result_has_tempo() {
        let t = AudioToMidi::default_config();
        let sig = sine(440.0, 44100.0, 1.0);
        let result = t.transcribe(&sig, 44100.0).expect("should succeed");
        assert!(result.tempo.bpm > 0.0);
        assert!(result.tempo.us_per_quarter > 0);
    }

    #[test]
    fn test_transcribe_total_duration() {
        let t = AudioToMidi::default_config();
        let sr = 44100.0;
        let sig = sine(440.0, sr, 2.0);
        let result = t.transcribe(&sig, sr).expect("should succeed");
        assert!((result.total_duration_secs - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_notes_sorted_by_start() {
        let sr = 22050.0;
        let cfg = AudioToMidiConfig {
            window_size: 512,
            hop_size: 128,
            min_freq: 80.0,
            max_freq: 2000.0,
            voicing_threshold: 0.3,
            min_note_duration_secs: 0.01,
            ..AudioToMidiConfig::default()
        };
        let t = AudioToMidi::new(cfg);
        // Concatenate two sines at different frequencies
        let mut sig = sine(220.0, sr, 0.5);
        sig.extend(sine(440.0, sr, 0.5));
        let result = t.transcribe(&sig, sr).expect("should succeed");
        for w in result.notes.windows(2) {
            assert!(
                w[1].start_secs >= w[0].start_secs,
                "Notes not sorted: {} > {}",
                w[0].start_secs,
                w[1].start_secs
            );
        }
    }

    // ── MIDI byte serialisation ───────────────────────────────────────────

    #[test]
    fn test_to_midi_bytes_empty_notes() {
        let txn = MidiTranscription {
            notes: Vec::new(),
            tempo: MidiTempo::from_bpm(120.0),
            ticks_per_quarter: 480,
            total_duration_secs: 1.0,
        };
        let bytes = txn.to_midi_bytes().expect("should succeed");
        // Minimal valid SMF: MThd (14 bytes) + MTrk header + tempo meta + EOT
        assert!(bytes.len() >= 14, "SMF too short: {} bytes", bytes.len());
        assert_eq!(&bytes[..4], b"MThd");
        assert_eq!(&bytes[14..18], b"MTrk");
    }

    #[test]
    fn test_to_midi_bytes_with_note() {
        let note = MidiNote {
            note: 60,
            velocity: 80,
            start_secs: 0.0,
            end_secs: 0.5,
            duration_secs: 0.5,
            frequency_hz: 261.63,
            confidence: 0.9,
            channel: 0,
        };
        let txn = MidiTranscription {
            notes: vec![note],
            tempo: MidiTempo::from_bpm(120.0),
            ticks_per_quarter: 480,
            total_duration_secs: 2.0,
        };
        let bytes = txn.to_midi_bytes().expect("should succeed");
        assert!(bytes.len() > 30);
        assert_eq!(&bytes[..4], b"MThd");
    }

    #[test]
    fn test_to_midi_bytes_out_of_range_note_error() {
        let note = MidiNote {
            note: 128, // invalid!
            velocity: 80,
            start_secs: 0.0,
            end_secs: 0.5,
            duration_secs: 0.5,
            frequency_hz: 440.0,
            confidence: 0.9,
            channel: 0,
        };
        let txn = MidiTranscription {
            notes: vec![note],
            tempo: MidiTempo::from_bpm(120.0),
            ticks_per_quarter: 480,
            total_duration_secs: 1.0,
        };
        let result = txn.to_midi_bytes();
        assert!(result.is_err());
    }

    // ── VLQ encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_vlq_zero() {
        let mut buf = Vec::new();
        write_vlq(&mut buf, 0);
        assert_eq!(buf, &[0x00]);
    }

    #[test]
    fn test_vlq_127() {
        let mut buf = Vec::new();
        write_vlq(&mut buf, 127);
        assert_eq!(buf, &[0x7F]);
    }

    #[test]
    fn test_vlq_128() {
        let mut buf = Vec::new();
        write_vlq(&mut buf, 128);
        assert_eq!(buf, &[0x81, 0x00]);
    }

    #[test]
    fn test_vlq_268435455() {
        // Maximum 4-byte VLQ
        let mut buf = Vec::new();
        write_vlq(&mut buf, 0x0FFF_FFFF);
        assert_eq!(buf, &[0xFF, 0xFF, 0xFF, 0x7F]);
    }

    // ── Onset detection helper ────────────────────────────────────────────

    #[test]
    fn test_detect_onsets_silence_no_onsets() {
        let t = AudioToMidi::default_config();
        let frames: Vec<FrameData> = (0..100)
            .map(|i| FrameData {
                time_secs: i as f32 * 0.01,
                rms: 0.0,
                frequency_hz: 0.0,
                pitch_confidence: 0.0,
            })
            .collect();
        let onsets = t.detect_onsets(&frames);
        assert_eq!(onsets.len(), frames.len());
        assert!(onsets.iter().all(|&o| !o));
    }

    #[test]
    fn test_detect_onsets_impulse() {
        let t = AudioToMidi::default_config();
        let mut frames: Vec<FrameData> = (0..50)
            .map(|i| FrameData {
                time_secs: i as f32 * 0.01,
                rms: 0.001,
                frequency_hz: 0.0,
                pitch_confidence: 0.0,
            })
            .collect();
        // Large spike at frame 20
        frames[20].rms = 1.0;
        let onsets = t.detect_onsets(&frames);
        // At least one onset should be detected around frame 20
        let any_onset = onsets[17..24].iter().any(|&o| o);
        assert!(any_onset, "Expected onset near spike at frame 20");
    }
}
