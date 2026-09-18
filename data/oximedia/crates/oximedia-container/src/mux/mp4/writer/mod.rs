//! MP4 muxer writer implementation.
//!
//! Generates ISOBMFF-compliant MP4 files with moov/mdat layout for
//! progressive MP4, or moov/moof+mdat for fragmented MP4 (fMP4).
//!
//! # Fragmented MP4 box layout
//!
//! ```text
//! [ftyp][moov[mvhd][mvex[trex…]][trak(empty stbl)…]]
//!   ([sidx][moof[mfhd][traf[tfhd][tfdt][trun]]][mdat])…
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

mod boxes;
mod fragment;
mod sample_table;
mod types;

// ─── Public re-exports ────────────────────────────────────────────────────────

pub use fragment::build_sidx;
pub use types::{Mp4Config, Mp4FragmentMode, Mp4Mode, Mp4SampleEntry, Mp4TrackState};

use oximedia_core::{CodecId, MediaType, OxiError, OxiResult, Rational};

use crate::mux::cmaf::{
    build_empty_stco, build_empty_stsc, build_empty_stsz, build_empty_stts, build_mfhd, build_traf,
    write_box, write_full_box, write_u32_be, FragSampleEntry, FragTrackRun,
};
use boxes::{
    build_dinf, build_hdlr, build_mdhd, build_smhd, build_stco, build_stsd, build_stss, build_tkhd,
    build_trex, build_vmhd, IDENTITY_MATRIX,
};
use fragment::{compute_fragment_boundaries, dts_at, select_time_range, wrap_box};
use sample_table::{build_ctts, build_stsc, build_stsz, build_stts};
use types::{MAX_TRACKS, MOVIE_TIMESCALE};

use crate::{Packet, StreamInfo};

// ─── MP4 Muxer ───────────────────────────────────────────────────────────────

/// MP4/ISOBMFF muxer.
///
/// Builds a valid ISOBMFF file in memory with proper box hierarchy.
/// Supports both progressive and fragmented modes.
#[derive(Debug)]
pub struct Mp4Muxer {
    /// Configuration.
    config: Mp4Config,
    /// Per-track states.
    tracks: Vec<Mp4TrackState>,
    /// Whether the header has been written.
    header_written: bool,
    /// Fragment sequence number (for fragmented mode).
    fragment_sequence: u32,
    /// Completed fragments (for fragmented mode).
    fragments: Vec<Vec<u8>>,
}

impl Mp4Muxer {
    /// Creates a new MP4 muxer.
    #[must_use]
    pub fn new(config: Mp4Config) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            header_written: false,
            fragment_sequence: 1,
            fragments: Vec::new(),
        }
    }

    /// Adds a stream/track to the muxer. Returns the track index.
    ///
    /// # Errors
    ///
    /// Returns an error if the codec is not supported or the maximum number
    /// of tracks has been exceeded.
    pub fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::parse(
                0,
                "Cannot add streams after header is written",
            ));
        }
        if self.tracks.len() >= MAX_TRACKS {
            return Err(OxiError::parse(0, "Maximum number of tracks exceeded"));
        }

        // Validate codec is patent-free
        validate_codec(info.codec)?;

        let track_id = (self.tracks.len() + 1) as u32;
        self.tracks.push(Mp4TrackState::new(info, track_id));
        Ok(self.tracks.len() - 1)
    }

    /// Writes the file type box. Call after adding all streams.
    ///
    /// # Errors
    ///
    /// Returns an error if no streams have been added.
    pub fn write_header(&mut self) -> OxiResult<()> {
        if self.tracks.is_empty() {
            return Err(OxiError::parse(0, "No streams added"));
        }
        self.header_written = true;
        Ok(())
    }

    /// Writes a packet to the appropriate track.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream index is invalid or the header
    /// has not been written.
    pub fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::parse(0, "Header not written yet"));
        }

        let track_idx = packet.stream_index;
        if track_idx >= self.tracks.len() {
            return Err(OxiError::parse(
                0,
                format!("Invalid stream index {track_idx}"),
            ));
        }

        let track = &self.tracks[track_idx];
        let timescale = track.timescale;

        // Calculate sample duration from packet
        let duration = packet
            .duration()
            .map(|d| convert_duration(d, &track.stream_info.timebase, timescale))
            .unwrap_or(1);

        // Composition offset: PTS - DTS
        let comp_offset = if let Some(dts) = packet.dts() {
            let pts_ticks = convert_duration(packet.pts(), &track.stream_info.timebase, timescale);
            let dts_ticks = convert_duration(dts, &track.stream_info.timebase, timescale);
            (pts_ticks as i64 - dts_ticks as i64) as i32
        } else {
            0
        };

        let is_sync = packet.is_keyframe();

        let track = &mut self.tracks[track_idx];
        track.add_sample(&packet.data, duration, comp_offset, is_sync);

        Ok(())
    }

    /// Finalizes the MP4 file and returns the complete byte output.
    ///
    /// For progressive mode, this assembles ftyp + moov + mdat.
    /// For fragmented mode, this assembles ftyp + moov (with mvex) + fragments.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    pub fn finalize(&self) -> OxiResult<Vec<u8>> {
        if !self.header_written {
            return Err(OxiError::parse(0, "Header not written"));
        }

        let mut output = Vec::new();

        // 1. ftyp box
        output.extend(self.build_ftyp());

        match self.config.mode {
            Mp4FragmentMode::Progressive => {
                // 2. Collect all mdat data with offsets
                let (mdat_box, chunk_offsets) = self.build_mdat_progressive();

                // 3. moov box (needs chunk offsets, which depend on ftyp + moov size)
                // We need to do a two-pass: first estimate moov size, then recalc offsets
                let ftyp_size = output.len() as u64;
                let moov_estimate = self.build_moov_progressive(&chunk_offsets, ftyp_size);
                let moov_size = moov_estimate.len() as u64;

                // Recalculate offsets with correct moov size
                let mdat_start = ftyp_size + moov_size + 8; // +8 for mdat header
                let adjusted_offsets = adjust_chunk_offsets(&chunk_offsets, mdat_start);

                let moov_final = self.build_moov_progressive(&adjusted_offsets, ftyp_size);

                output.extend(moov_final);
                output.extend(mdat_box);
            }
            Mp4FragmentMode::Fragmented { .. } => {
                // moov with mvex
                let moov = self.build_moov_fragmented();
                output.extend(moov);

                // Append all cached fragments
                for frag in &self.fragments {
                    output.extend(frag);
                }

                // Build fragments from pending samples
                let frags = self.build_fragments_from_tracks();
                for frag in frags {
                    output.extend(frag);
                }
            }
        }

        Ok(output)
    }

    /// Returns the number of tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Returns a reference to a track state by index.
    #[must_use]
    pub fn track(&self, index: usize) -> Option<&Mp4TrackState> {
        self.tracks.get(index)
    }

    /// Returns the total number of samples across all tracks.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.tracks.iter().map(|t| t.samples.len()).sum()
    }

    // ─── Box builders ───────────────────────────────────────────────────

    fn build_ftyp(&self) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(&self.config.major_brand);
        content.extend_from_slice(&write_u32_be(self.config.minor_version));
        for brand in &self.config.compatible_brands {
            content.extend_from_slice(brand);
        }
        write_box(b"ftyp", &content)
    }

    fn build_mdat_progressive(&self) -> (Vec<u8>, Vec<Vec<u64>>) {
        let mut mdat_payload = Vec::new();
        let mut all_chunk_offsets: Vec<Vec<u64>> = Vec::new();

        for track in &self.tracks {
            let mut track_offsets = Vec::new();
            let mut sample_idx: u32 = 0;
            let mut current_offset = mdat_payload.len() as u64;

            // Process each chunk
            for (chunk_idx, &chunk_start) in track.chunk_boundaries.iter().enumerate() {
                let chunk_end = track
                    .chunk_boundaries
                    .get(chunk_idx + 1)
                    .copied()
                    .unwrap_or(track.samples.len() as u32);

                track_offsets.push(current_offset);

                for si in chunk_start..chunk_end {
                    if let Some(sample) = track.samples.get(si as usize) {
                        let start = sample_data_offset(track, sample_idx);
                        let end = start + sample.size as usize;
                        if end <= track.mdat_data.len() {
                            mdat_payload.extend_from_slice(&track.mdat_data[start..end]);
                            current_offset += u64::from(sample.size);
                        }
                    }
                    sample_idx += 1;
                }
            }

            all_chunk_offsets.push(track_offsets);
        }

        let mdat = write_box(b"mdat", &mdat_payload);
        (mdat, all_chunk_offsets)
    }

    fn build_moov_progressive(&self, chunk_offsets: &[Vec<u64>], _ftyp_size: u64) -> Vec<u8> {
        let mut content = Vec::new();

        // mvhd
        content.extend(self.build_mvhd());

        // trak boxes
        for (i, track) in self.tracks.iter().enumerate() {
            let offsets = chunk_offsets.get(i).map_or(&[][..], |v| v.as_slice());
            content.extend(self.build_trak(track, offsets));
        }

        write_box(b"moov", &content)
    }

    fn build_moov_fragmented(&self) -> Vec<u8> {
        let mut content = Vec::new();
        // mvhd with duration=0 (indeterminate for fragmented)
        content.extend(self.build_mvhd_fragmented());

        // mvex with trex for each track
        let mut mvex_content = Vec::new();
        for track in &self.tracks {
            mvex_content.extend(build_trex(track.track_id));
        }
        content.extend(write_box(b"mvex", &mvex_content));

        // trak boxes with EMPTY stbl tables (no sample data in init segment)
        for track in &self.tracks {
            content.extend(self.build_trak_fragmented(track));
        }

        write_box(b"moov", &content)
    }

    /// Builds an `mvhd` with duration=0 (indeterminate, for fragmented mode).
    fn build_mvhd_fragmented(&self) -> Vec<u8> {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(self.config.creation_time as u32));
        c.extend_from_slice(&write_u32_be(self.config.modification_time as u32));
        c.extend_from_slice(&write_u32_be(MOVIE_TIMESCALE));
        // duration = 0 for fragmented (unknown at init time)
        c.extend_from_slice(&write_u32_be(0));
        c.extend_from_slice(&write_u32_be(0x0001_0000)); // rate 1.0
        c.extend_from_slice(&[0x01, 0x00]); // volume 1.0
        c.extend_from_slice(&[0u8; 10]); // reserved
        c.extend_from_slice(&IDENTITY_MATRIX);
        c.extend_from_slice(&[0u8; 24]); // pre_defined
        c.extend_from_slice(&write_u32_be((self.tracks.len() + 1) as u32)); // next_track_id
        write_full_box(b"mvhd", 0, 0, &c)
    }

    /// Builds a `trak` box suitable for a fragmented init segment.
    ///
    /// The `stbl` contains only an `stsd` (sample description) plus the four
    /// mandatory empty sub-boxes — no sample timing, size, or chunk-offset data.
    fn build_trak_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_tkhd(track));
        content.extend(self.build_mdia_fragmented(track));
        write_box(b"trak", &content)
    }

    fn build_mdia_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_mdhd(track));
        content.extend(build_hdlr(track));
        content.extend(self.build_minf_fragmented(track));
        write_box(b"mdia", &content)
    }

    fn build_minf_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        match track.stream_info.media_type {
            MediaType::Video => content.extend(build_vmhd()),
            MediaType::Audio => content.extend(build_smhd()),
            _ => content.extend(write_full_box(b"nmhd", 0, 0, &[])),
        }
        content.extend(build_dinf());
        content.extend(self.build_stbl_fragmented(track));
        write_box(b"minf", &content)
    }

    fn build_stbl_fragmented(&self, track: &Mp4TrackState) -> Vec<u8> {
        let mut content = Vec::new();
        // stsd must be present with proper sample entry
        content.extend(build_stsd(track));
        // Four mandatory empty tables (required by ISO 14496-12 §8.6.1 for fragmented)
        content.extend(build_empty_stts());
        content.extend(build_empty_stsc());
        content.extend(build_empty_stsz());
        content.extend(build_empty_stco());
        write_box(b"stbl", &content)
    }

    fn build_mvhd(&self) -> Vec<u8> {
        let mut c = Vec::new();
        // creation_time
        c.extend_from_slice(&write_u32_be(self.config.creation_time as u32));
        // modification_time
        c.extend_from_slice(&write_u32_be(self.config.modification_time as u32));
        // timescale
        c.extend_from_slice(&write_u32_be(MOVIE_TIMESCALE));
        // duration
        let max_dur = self
            .tracks
            .iter()
            .map(|t| {
                if t.timescale == 0 {
                    0
                } else {
                    t.total_duration * u64::from(MOVIE_TIMESCALE) / u64::from(t.timescale)
                }
            })
            .max()
            .unwrap_or(0);
        c.extend_from_slice(&write_u32_be(max_dur as u32));
        // rate (1.0 as fixed-point 16.16)
        c.extend_from_slice(&write_u32_be(0x0001_0000));
        // volume (1.0 as fixed-point 8.8)
        c.extend_from_slice(&[0x01, 0x00]);
        // reserved (10 bytes)
        c.extend_from_slice(&[0u8; 10]);
        // matrix (identity, 36 bytes)
        c.extend_from_slice(&IDENTITY_MATRIX);
        // pre_defined (24 bytes)
        c.extend_from_slice(&[0u8; 24]);
        // next_track_id
        c.extend_from_slice(&write_u32_be((self.tracks.len() + 1) as u32));

        write_full_box(b"mvhd", 0, 0, &c)
    }

    fn build_trak(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_tkhd(track));
        content.extend(self.build_mdia(track, chunk_offsets));
        write_box(b"trak", &content)
    }

    fn build_mdia(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend(build_mdhd(track));
        content.extend(build_hdlr(track));
        content.extend(self.build_minf(track, chunk_offsets));
        write_box(b"mdia", &content)
    }

    fn build_minf(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();

        // vmhd or smhd
        match track.stream_info.media_type {
            MediaType::Video => {
                content.extend(build_vmhd());
            }
            MediaType::Audio => {
                content.extend(build_smhd());
            }
            _ => {
                // nmhd for other types
                content.extend(write_full_box(b"nmhd", 0, 0, &[]));
            }
        }

        // dinf + dref
        content.extend(build_dinf());

        // stbl
        content.extend(self.build_stbl(track, chunk_offsets));

        write_box(b"minf", &content)
    }

    fn build_stbl(&self, track: &Mp4TrackState, chunk_offsets: &[u64]) -> Vec<u8> {
        let mut content = Vec::new();

        // stsd (sample description)
        content.extend(build_stsd(track));

        // stts (time-to-sample)
        content.extend(build_stts(track));

        // ctts (composition time offsets, if needed)
        if track.samples.iter().any(|s| s.composition_offset != 0) {
            content.extend(build_ctts(track));
        }

        // stsc (sample-to-chunk)
        content.extend(build_stsc(track));

        // stsz (sample sizes)
        content.extend(build_stsz(track));

        // stco (chunk offsets)
        content.extend(build_stco(chunk_offsets));

        // stss (sync sample table, for video with non-all-sync)
        if track.stream_info.media_type == MediaType::Video
            && track.samples.iter().any(|s| !s.is_sync)
        {
            content.extend(build_stss(track));
        }

        write_box(b"stbl", &content)
    }

    /// Splits all track samples into timed fragments and builds `sidx`+`moof`+`mdat` for each.
    ///
    /// Fragment boundaries are determined by the `fragment_duration_ms` setting:
    /// a new fragment begins at each keyframe whose decode timestamp is at or beyond
    /// the target duration since the previous fragment start.
    fn build_fragments_from_tracks(&self) -> Vec<Vec<u8>> {
        // Collect fragments from the first non-empty track; use it as the reference timeline.
        let ref_track = match self.tracks.iter().find(|t| !t.samples.is_empty()) {
            Some(t) => t,
            None => return Vec::new(),
        };

        // Compute fragment boundaries (sample indices) based on the reference track.
        let config_frag_ms = match self.config.mode {
            Mp4FragmentMode::Fragmented {
                fragment_duration_ms,
            } => fragment_duration_ms,
            Mp4FragmentMode::Progressive => 0,
        };
        let frag_duration_ticks = if ref_track.timescale > 0 && config_frag_ms > 0 {
            u64::from(config_frag_ms) * u64::from(ref_track.timescale) / 1000
        } else {
            0
        };
        let boundaries = compute_fragment_boundaries(ref_track, frag_duration_ticks);

        let mut result = Vec::new();
        let mut seq = self.fragment_sequence;

        for &(frag_start, frag_end) in &boundaries {
            let fragment = self.build_one_fragment(seq, frag_start, frag_end);
            result.push(fragment);
            seq += 1;
        }

        result
    }

    /// Builds one `sidx`+`moof`+`mdat` fragment for samples `[frag_start, frag_end)` on the
    /// reference (video) track and matching time-range samples on every other track.
    fn build_one_fragment(&self, seq: u32, frag_start: usize, frag_end: usize) -> Vec<u8> {
        // ── Determine per-track sample slices ────────────────────────────────
        let ref_track = match self.tracks.first() {
            Some(t) => t,
            None => return Vec::new(),
        };

        // Compute the DTS range covered by this fragment on the reference track.
        let frag_base_dts = dts_at(ref_track, frag_start);
        let frag_end_dts = dts_at(ref_track, frag_end);

        // ── Collect mdat payload and traf descriptors ─────────────────────────
        let mut mdat_payload: Vec<u8> = Vec::new();
        let mut track_runs: Vec<FragTrackRun> = Vec::new();

        for track in &self.tracks {
            if track.samples.is_empty() {
                continue;
            }

            // For the reference track use explicit indices;
            // for other tracks select samples whose DTS falls in [frag_base_dts, frag_end_dts).
            let (slice_start, slice_end) = if track.track_id == ref_track.track_id {
                (frag_start, frag_end)
            } else {
                select_time_range(track, frag_base_dts, frag_end_dts)
            };

            if slice_start >= slice_end {
                continue;
            }

            let base_dts = dts_at(track, slice_start);
            let mdat_offset_start = mdat_payload.len() as u32;
            let mut entries: Vec<FragSampleEntry> = Vec::new();

            let mut byte_offset = 0usize;
            for s_idx in 0..slice_start {
                byte_offset += track.samples.get(s_idx).map_or(0, |s| s.size as usize);
            }

            for s_idx in slice_start..slice_end {
                let s = match track.samples.get(s_idx) {
                    Some(s) => s,
                    None => break,
                };
                let end = byte_offset + s.size as usize;
                if end <= track.mdat_data.len() {
                    mdat_payload.extend_from_slice(&track.mdat_data[byte_offset..end]);
                }
                byte_offset = end;

                let flags: u32 = if s.is_sync { 0x0200_0000 } else { 0x0101_0000 };
                #[allow(clippy::cast_possible_wrap)]
                let pts_offset = s.composition_offset;
                entries.push(FragSampleEntry {
                    duration: s.duration,
                    size: s.size,
                    flags,
                    pts_offset,
                });
            }

            track_runs.push(FragTrackRun {
                track_id: track.track_id,
                base_dts,
                mdat_offset_start,
                entries,
            });
        }

        // ── Two-pass moof build to fixup data_offset ─────────────────────────
        let build_moof = |data_offset_base: i32, runs: &[FragTrackRun]| -> Vec<u8> {
            let mut moof_content = Vec::new();
            moof_content.extend(build_mfhd(seq));
            for tr in runs {
                moof_content.extend(build_traf(
                    tr.track_id,
                    tr.base_dts,
                    data_offset_base + tr.mdat_offset_start as i32,
                    &tr.entries,
                ));
            }
            wrap_box(b"moof", &moof_content)
        };

        let moof_placeholder = build_moof(0, &track_runs);
        let moof_size = moof_placeholder.len() as i32;
        // data_offset is relative to start of moof; mdat header is 8 bytes.
        let moof = build_moof(moof_size + 8, &track_runs);
        let mdat = write_box(b"mdat", &mdat_payload);

        // ── sidx (segment index) ──────────────────────────────────────────────
        // Compute subsegment_duration in the reference track's timescale.
        let subseg_dur = track_runs
            .iter()
            .find(|tr| tr.track_id == ref_track.track_id)
            .map(|tr| {
                tr.entries
                    .iter()
                    .map(|e| u64::from(e.duration))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        let is_sap = track_runs
            .iter()
            .find(|tr| tr.track_id == ref_track.track_id)
            .and_then(|tr| tr.entries.first())
            .is_some_and(|e| e.flags == 0x0200_0000);
        let referenced_size = (moof.len() + mdat.len()) as u32;
        let sidx = build_sidx(
            ref_track.track_id,
            ref_track.timescale,
            frag_base_dts,
            referenced_size,
            subseg_dur as u32,
            is_sap,
        );

        let mut out = sidx;
        out.extend(moof);
        out.extend(mdat);
        out
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn validate_codec(codec: CodecId) -> OxiResult<()> {
    match codec {
        CodecId::Av1
        | CodecId::Vp9
        | CodecId::Vp8
        | CodecId::Opus
        | CodecId::Flac
        | CodecId::Vorbis
        | CodecId::Apv   // ISO/IEC 23009-13 — royalty-free
        | CodecId::Mjpeg // JPEG patents expired — royalty-free
        => Ok(()),
        _ => Err(OxiError::PatentViolation(format!(
            "Codec {:?} is not supported in MP4 muxer (patent-free codecs only)",
            codec
        ))),
    }
}

#[allow(clippy::cast_precision_loss)]
fn convert_duration(ticks: i64, timebase: &Rational, target_timescale: u32) -> u32 {
    if timebase.den == 0 || target_timescale == 0 {
        return 1;
    }
    let seconds = (ticks as f64 * timebase.num as f64) / timebase.den as f64;
    let result = seconds * target_timescale as f64;
    if result <= 0.0 {
        1
    } else {
        result as u32
    }
}

fn sample_data_offset(track: &Mp4TrackState, sample_idx: u32) -> usize {
    let mut offset = 0usize;
    for i in 0..sample_idx as usize {
        if let Some(s) = track.samples.get(i) {
            offset += s.size as usize;
        }
    }
    offset
}

fn adjust_chunk_offsets(offsets: &[Vec<u64>], mdat_data_start: u64) -> Vec<Vec<u64>> {
    offsets
        .iter()
        .map(|track_offsets| track_offsets.iter().map(|&o| o + mdat_data_start).collect())
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PacketFlags;
    use bytes::Bytes;
    use oximedia_core::{CodecId, Rational, Timestamp};

    fn make_video_stream() -> StreamInfo {
        let mut info = StreamInfo::new(0, CodecId::Av1, Rational::new(1, 90000));
        info.codec_params = crate::CodecParams::video(1920, 1080);
        info
    }

    fn make_audio_stream() -> StreamInfo {
        let mut info = StreamInfo::new(1, CodecId::Opus, Rational::new(1, 48000));
        info.codec_params = crate::CodecParams::audio(48000, 2);
        info
    }

    fn make_video_packet(stream_index: usize, pts: i64, keyframe: bool) -> Packet {
        let mut ts = Timestamp::new(pts, Rational::new(1, 90000));
        ts.duration = Some(3000); // 1 frame at 30fps in 90kHz
        Packet::new(
            stream_index,
            Bytes::from(vec![0xAA; 100]),
            ts,
            if keyframe {
                PacketFlags::KEYFRAME
            } else {
                PacketFlags::empty()
            },
        )
    }

    fn make_audio_packet(stream_index: usize, pts: i64) -> Packet {
        let mut ts = Timestamp::new(pts, Rational::new(1, 48000));
        ts.duration = Some(960); // 20ms of audio at 48kHz
        Packet::new(
            stream_index,
            Bytes::from(vec![0xBB; 50]),
            ts,
            PacketFlags::KEYFRAME,
        )
    }

    // ── Config tests ────────────────────────────────────────────────────

    #[test]
    fn test_mp4_config_default() {
        let config = Mp4Config::new();
        assert_eq!(config.mode, Mp4FragmentMode::Progressive);
        assert_eq!(config.major_brand, *b"isom");
    }

    #[test]
    fn test_mp4_config_builder() {
        let config = Mp4Config::new()
            .with_fragmented(4000)
            .with_major_brand(*b"av01")
            .with_minor_version(0x100)
            .with_compatible_brand(*b"dash");

        assert_eq!(
            config.mode,
            Mp4FragmentMode::Fragmented {
                fragment_duration_ms: 4000
            }
        );
        assert_eq!(config.major_brand, *b"av01");
        assert_eq!(config.minor_version, 0x100);
        assert!(config.compatible_brands.contains(b"dash"));
    }

    // ── Muxer lifecycle tests ───────────────────────────────────────────

    #[test]
    fn test_add_stream_returns_index() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let idx0 = muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        let idx1 = muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(muxer.track_count(), 2);
    }

    #[test]
    fn test_add_stream_after_header_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let result = muxer.add_stream(make_audio_stream());
        assert!(result.is_err());
    }

    #[test]
    fn test_write_header_no_streams_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let result = muxer.write_header();
        assert!(result.is_err());
    }

    #[test]
    fn test_write_packet_before_header_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        let pkt = make_video_packet(0, 0, true);
        let result = muxer.write_packet(&pkt);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_packet_invalid_index_fails() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let pkt = make_video_packet(99, 0, true);
        let result = muxer.write_packet(&pkt);
        assert!(result.is_err());
    }

    #[test]
    fn test_patent_encumbered_codec_rejected() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        let info = StreamInfo::new(0, CodecId::Pcm, Rational::new(1, 44100));
        let result = muxer.add_stream(info);
        assert!(result.is_err());
    }

    // ── Progressive MP4 output tests ────────────────────────────────────

    #[test]
    fn test_progressive_ftyp_present() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        let pkt = make_video_packet(0, 0, true);
        muxer.write_packet(&pkt).expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        // ftyp box should be first
        assert_eq!(&output[4..8], b"ftyp");
        // major brand should be "isom"
        assert_eq!(&output[8..12], b"isom");
    }

    #[test]
    fn test_progressive_contains_moov_and_mdat() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 3000, false))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_moov = output.windows(4).any(|w| w == b"moov");
        let has_mdat = output.windows(4).any(|w| w == b"mdat");
        assert!(has_moov, "output must contain moov box");
        assert!(has_mdat, "output must contain mdat box");
    }

    #[test]
    fn test_progressive_contains_trak_boxes() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_audio_packet(1, 0))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let trak_count = output.windows(4).filter(|w| *w == b"trak").count();
        assert_eq!(trak_count, 2, "output must contain 2 trak boxes");
    }

    #[test]
    fn test_progressive_contains_stbl_tables() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        for i in 0..5 {
            muxer
                .write_packet(&make_video_packet(0, i * 3000, i == 0))
                .expect("should succeed");
        }

        let output = muxer.finalize().expect("should succeed");

        // Check for essential sample table boxes
        let has_stsd = output.windows(4).any(|w| w == b"stsd");
        let has_stts = output.windows(4).any(|w| w == b"stts");
        let has_stsz = output.windows(4).any(|w| w == b"stsz");
        let has_stco = output.windows(4).any(|w| w == b"stco");
        let has_stss = output.windows(4).any(|w| w == b"stss");

        assert!(has_stsd, "must have stsd");
        assert!(has_stts, "must have stts");
        assert!(has_stsz, "must have stsz");
        assert!(has_stco, "must have stco");
        assert!(has_stss, "must have stss (video with non-sync frames)");
    }

    #[test]
    fn test_progressive_mdat_contains_sample_data() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");

        // Find mdat and verify it contains our 0xAA payload
        let mdat_pos = output
            .windows(4)
            .position(|w| w == b"mdat")
            .expect("mdat must exist");
        // mdat data starts 4 bytes after the fourcc
        let mdat_start = mdat_pos + 4;
        assert!(
            output[mdat_start..].windows(4).any(|w| w == [0xAA; 4]),
            "mdat must contain sample payload"
        );
    }

    #[test]
    fn test_total_samples_count() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 3000, false))
            .expect("should succeed");
        muxer
            .write_packet(&make_audio_packet(1, 0))
            .expect("should succeed");

        assert_eq!(muxer.total_samples(), 3);
    }

    // ── Fragmented MP4 tests ────────────────────────────────────────────

    #[test]
    fn test_fragmented_contains_mvex() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_mvex = output.windows(4).any(|w| w == b"mvex");
        let has_trex = output.windows(4).any(|w| w == b"trex");
        assert!(has_mvex, "fragmented must contain mvex");
        assert!(has_trex, "fragmented must contain trex");
    }

    #[test]
    fn test_fragmented_contains_moof_mdat() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_moof = output.windows(4).any(|w| w == b"moof");
        let has_mdat = output.windows(4).any(|w| w == b"mdat");
        assert!(has_moof, "fragmented must contain moof");
        assert!(has_mdat, "fragmented must contain mdat");
    }

    #[test]
    fn test_fragmented_contains_traf_trun() {
        let config = Mp4Config::new().with_fragmented(2000);
        let mut muxer = Mp4Muxer::new(config);
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        muxer
            .write_packet(&make_video_packet(0, 0, true))
            .expect("should succeed");

        let output = muxer.finalize().expect("should succeed");
        let has_traf = output.windows(4).any(|w| w == b"traf");
        let has_trun = output.windows(4).any(|w| w == b"trun");
        assert!(has_traf, "fragmented must contain traf");
        assert!(has_trun, "fragmented must contain trun");
    }

    // ── Helper function tests ───────────────────────────────────────────

    #[test]
    fn test_validate_codec_accepted() {
        assert!(validate_codec(CodecId::Av1).is_ok());
        assert!(validate_codec(CodecId::Vp9).is_ok());
        assert!(validate_codec(CodecId::Opus).is_ok());
        assert!(validate_codec(CodecId::Flac).is_ok());
    }

    #[test]
    fn test_validate_codec_rejected() {
        assert!(validate_codec(CodecId::Pcm).is_err());
    }

    #[test]
    fn test_codec_to_fourcc() {
        assert_eq!(boxes::codec_to_fourcc(CodecId::Av1), *b"av01");
        assert_eq!(boxes::codec_to_fourcc(CodecId::Vp9), *b"vp09");
        assert_eq!(boxes::codec_to_fourcc(CodecId::Opus), *b"Opus");
    }

    #[test]
    fn test_convert_duration_basic() {
        let tb = Rational::new(1, 90000);
        let result = convert_duration(90000, &tb, 1000);
        assert_eq!(result, 1000); // 1 second
    }

    #[test]
    fn test_convert_duration_zero_target() {
        let tb = Rational::new(1, 90000);
        let result = convert_duration(100, &tb, 0);
        assert_eq!(result, 1); // fallback
    }

    #[test]
    fn test_sample_data_offset() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0xAA; 100], 3000, 0, true);
        track.add_sample(&[0xBB; 200], 3000, 0, false);
        track.add_sample(&[0xCC; 150], 3000, 0, false);

        assert_eq!(sample_data_offset(&track, 0), 0);
        assert_eq!(sample_data_offset(&track, 1), 100);
        assert_eq!(sample_data_offset(&track, 2), 300);
    }

    #[test]
    fn test_adjust_chunk_offsets() {
        let offsets = vec![vec![0, 100, 200], vec![0, 50]];
        let adjusted = adjust_chunk_offsets(&offsets, 1000);
        assert_eq!(adjusted[0], vec![1000, 1100, 1200]);
        assert_eq!(adjusted[1], vec![1000, 1050]);
    }

    #[test]
    fn test_build_stts_run_length() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        // 5 samples all with duration 3000
        for _ in 0..5 {
            track.add_sample(&[0u8; 10], 3000, 0, true);
        }
        let stts = build_stts(&track);
        // Should have entry_count=1 (all same duration)
        // fullbox header (12) + entry_count (4) + one entry (8) = 24
        assert_eq!(stts.len(), 24);
    }

    #[test]
    fn test_build_stsz_uniform() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        for _ in 0..3 {
            track.add_sample(&[0u8; 42], 3000, 0, true);
        }
        let stsz = build_stsz(&track);
        // With uniform sizes: fullbox header (12) + sample_size (4) + sample_count (4) = 20
        assert_eq!(stsz.len(), 20);
    }

    #[test]
    fn test_build_stsz_variable() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0u8; 100], 3000, 0, true);
        track.add_sample(&[0u8; 200], 3000, 0, false);
        let stsz = build_stsz(&track);
        // Variable: fullbox header (12) + sample_size=0 (4) + sample_count (4) + 2*4 = 28
        assert_eq!(stsz.len(), 28);
    }

    #[test]
    fn test_build_stss_sync_samples() {
        let mut track = Mp4TrackState::new(make_video_stream(), 1);
        track.add_sample(&[0u8; 10], 3000, 0, true);
        track.add_sample(&[0u8; 10], 3000, 0, false);
        track.add_sample(&[0u8; 10], 3000, 0, false);
        track.add_sample(&[0u8; 10], 3000, 0, true);
        let stss = build_stss(&track);
        // 2 sync samples: fullbox header (12) + entry_count (4) + 2*4 = 24
        assert_eq!(stss.len(), 24);
    }

    #[test]
    fn test_track_state_handler_types() {
        let video_track = Mp4TrackState::new(make_video_stream(), 1);
        assert_eq!(video_track.handler_type, *b"vide");

        let audio_track = Mp4TrackState::new(make_audio_stream(), 2);
        assert_eq!(audio_track.handler_type, *b"soun");
    }

    #[test]
    fn test_progressive_multi_packet_output_valid() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer
            .add_stream(make_audio_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");

        // Interleaved packets
        for i in 0..10 {
            muxer
                .write_packet(&make_video_packet(0, i * 3000, i == 0 || i == 5))
                .expect("should succeed");
            muxer
                .write_packet(&make_audio_packet(1, i * 960))
                .expect("should succeed");
        }

        let output = muxer.finalize().expect("should succeed");
        assert!(!output.is_empty());

        // Verify box structure: ftyp must come first
        assert_eq!(&output[4..8], b"ftyp");

        // Must contain all essential boxes
        let has_mvhd = output.windows(4).any(|w| w == b"mvhd");
        let has_tkhd = output.windows(4).any(|w| w == b"tkhd");
        let has_mdhd = output.windows(4).any(|w| w == b"mdhd");
        let has_hdlr = output.windows(4).any(|w| w == b"hdlr");
        assert!(has_mvhd);
        assert!(has_tkhd);
        assert!(has_mdhd);
        assert!(has_hdlr);
    }

    #[test]
    fn test_finalize_before_header_fails() {
        let muxer = Mp4Muxer::new(Mp4Config::new());
        assert!(muxer.finalize().is_err());
    }

    #[test]
    fn test_empty_tracks_finalize() {
        let mut muxer = Mp4Muxer::new(Mp4Config::new());
        muxer
            .add_stream(make_video_stream())
            .expect("should succeed");
        muxer.write_header().expect("should succeed");
        // No packets written
        let output = muxer.finalize().expect("should succeed");
        assert!(!output.is_empty());
        assert_eq!(&output[4..8], b"ftyp");
    }
}
