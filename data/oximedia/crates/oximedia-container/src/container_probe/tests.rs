//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    #[test]
    fn test_has_video_true() {
        let mut r = ContainerProbeResult::new("mkv");
        r.video_present = true;
        assert!(r.has_video());
    }
    #[test]
    fn test_has_video_false() {
        let r = ContainerProbeResult::new("flac");
        assert!(!r.has_video());
    }
    #[test]
    fn test_has_audio_true() {
        let mut r = ContainerProbeResult::new("ogg");
        r.audio_present = true;
        assert!(r.has_audio());
    }
    #[test]
    fn test_is_av_both() {
        let mut r = ContainerProbeResult::new("mp4");
        r.video_present = true;
        r.audio_present = true;
        assert!(r.is_av());
    }
    #[test]
    fn test_is_av_audio_only() {
        let mut r = ContainerProbeResult::new("wav");
        r.audio_present = true;
        assert!(!r.is_av());
    }
    #[test]
    fn test_is_confident() {
        let r = ContainerProbeResult::new("matroska");
        assert!(r.is_confident(0.9));
        assert!(!r.is_confident(1.1));
    }
    #[test]
    fn test_container_info_format_name() {
        let info = ContainerInfo::new("matroska");
        assert_eq!(info.format_name(), "matroska");
    }
    #[test]
    fn test_container_info_track_count() {
        let info = ContainerInfo::new("mp4").with_tracks(1, 2);
        assert_eq!(info.track_count(), 3);
    }
    #[test]
    fn test_container_info_video_count() {
        let info = ContainerInfo::new("mkv").with_tracks(2, 4);
        assert_eq!(info.video_count(), 2);
        assert_eq!(info.audio_count(), 4);
    }
    #[test]
    fn test_estimated_bitrate_kbps() {
        let info = ContainerInfo::new("mp4")
            .with_file_size(1_000_000)
            .with_duration_ms(1000);
        let kbps = info
            .estimated_bitrate_kbps()
            .expect("operation should succeed");
        assert!((kbps - 8000.0).abs() < 1.0);
    }
    #[test]
    fn test_estimated_bitrate_kbps_no_duration() {
        let info = ContainerInfo::new("mkv").with_file_size(1_000_000);
        assert!(info.estimated_bitrate_kbps().is_none());
    }
    #[test]
    fn test_probe_matroska() {
        let mut p = ContainerProber::new();
        let magic = [0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00];
        let r = p.probe_header(&magic);
        assert_eq!(r.format_label, "matroska");
        assert!(r.has_video());
        assert!(r.has_audio());
    }
    #[test]
    fn test_probe_flac() {
        let mut p = ContainerProber::new();
        let r = p.probe_header(b"fLaC\x00\x00\x00\x22");
        assert_eq!(r.format_label, "flac");
        assert!(!r.has_video());
        assert!(r.has_audio());
    }
    #[test]
    fn test_probe_mp4() {
        let mut p = ContainerProber::new();
        let header = b"\x00\x00\x00\x18ftyp\x69\x73\x6f\x6d";
        let r = p.probe_header(header);
        assert_eq!(r.format_label, "mp4");
        assert!(r.has_video());
        assert_eq!(p.probed_count(), 1);
    }
    #[test]
    fn test_probe_unknown() {
        let mut p = ContainerProber::new();
        let r = p.probe_header(b"\xFF\xFF\xFF\xFF");
        assert_eq!(r.format_label, "unknown");
        assert_eq!(r.confidence, 0.0);
    }
    #[test]
    fn test_multiformat_probe_empty() {
        let info = MultiFormatProber::probe(&[]);
        assert_eq!(info.format, "unknown");
        assert!(info.streams.is_empty());
    }
    #[test]
    fn test_multiformat_probe_random() {
        let info = MultiFormatProber::probe(&[0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(info.format, "unknown");
    }
    #[test]
    fn test_multiformat_probe_flac_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"fLaC");
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        data.push(0x22);
        data.extend_from_slice(&[0u8; 10]);
        data.push(0xAC);
        data.push(0x44);
        data.push(0x42);
        data.push(0xF0);
        data.extend_from_slice(&[0u8; 20]);
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "flac");
        assert!(!info.streams.is_empty());
        assert_eq!(info.streams[0].codec, "flac");
        assert_eq!(info.streams[0].stream_type, "audio");
    }
    #[test]
    fn test_multiformat_probe_wav() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        let total_size: u32 = 36;
        data.extend_from_slice(&total_size.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&44100u32.to_le_bytes());
        data.extend_from_slice(&(44100 * 2 * 2u32).to_le_bytes());
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "wav");
        assert!(!info.streams.is_empty());
        let s = &info.streams[0];
        assert_eq!(s.codec, "pcm");
        assert_eq!(s.sample_rate, Some(44100));
        assert_eq!(s.channels, Some(2));
    }
    #[test]
    fn test_multiformat_probe_ogg() {
        let mut data = vec![0u8; 300];
        data[0..4].copy_from_slice(b"OggS");
        data[4] = 0;
        data[5] = 0x02;
        data[6..14].fill(0);
        data[14..18].fill(0);
        data[18..22].fill(0);
        data[22..26].fill(0);
        data[26] = 1;
        data[27] = 19;
        data[28..36].copy_from_slice(b"OpusHead");
        data[36] = 1;
        data[37] = 2;
        data[38..40].fill(0);
        data[40..44].copy_from_slice(&48000u32.to_le_bytes());
        data[44..46].fill(0);
        data[46] = 0;
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "ogg");
    }
    #[test]
    fn test_multiformat_probe_mp4_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(&20u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"iso5");
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"iso5");
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "mp4");
    }
    #[test]
    fn test_multiformat_probe_mkv_magic() {
        let data = [
            0x1A, 0x45, 0xDF, 0xA3, 0x84, 0x42, 0x82, 0x84, 0x77, 0x65, 0x62, 0x6D, 0x00,
        ];
        let info = MultiFormatProber::probe(&data);
        assert!(
            info.format == "mkv" || info.format == "webm",
            "got format: {}",
            info.format
        );
    }
    #[test]
    fn test_probe_streams_only() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&36u32.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&22050u32.to_le_bytes());
        data.extend_from_slice(&(22050u32 * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        let streams = MultiFormatProber::probe_streams_only(&data);
        assert!(!streams.is_empty());
        assert_eq!(streams[0].stream_type, "audio");
    }
    #[test]
    fn test_multiformat_file_size() {
        let data = b"not a real container at all, just some bytes";
        let info = MultiFormatProber::probe(data);
        assert_eq!(info.file_size_bytes, data.len() as u64);
    }
    #[test]
    fn test_multiformat_wav_duration() {
        let mut data = Vec::new();
        let pcm_bytes: u32 = 44100 * 2;
        let total: u32 = 36 + pcm_bytes;
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&total.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&44100u32.to_le_bytes());
        data.extend_from_slice(&(44100u32 * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&pcm_bytes.to_le_bytes());
        data.extend(vec![0u8; pcm_bytes as usize]);
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "wav");
        assert_eq!(info.duration_ms, Some(1000));
    }
    #[test]
    fn test_detailed_stream_info_default() {
        let s = DetailedStreamInfo::default();
        assert!(s.codec.is_empty());
        assert!(s.stream_type.is_empty());
        assert!(s.duration_ms.is_none());
    }
    #[test]
    fn test_detailed_container_info_metadata() {
        let info = DetailedContainerInfo::default();
        assert!(info.metadata.is_empty());
        assert!(info.streams.is_empty());
        assert_eq!(info.file_size_bytes, 0);
    }
    #[test]
    fn test_multiformat_probe_caf() {
        let mut data = Vec::new();
        data.extend_from_slice(b"caff");
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        data.extend_from_slice(b"desc");
        data.extend_from_slice(&32u64.to_be_bytes());
        data.extend_from_slice(&44100.0_f64.to_be_bytes());
        data.extend_from_slice(b"lpcm");
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&4u32.to_be_bytes());
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&2u32.to_be_bytes());
        data.extend_from_slice(&16u32.to_be_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "caf");
        assert!(!info.streams.is_empty());
        assert_eq!(info.streams[0].stream_type, "audio");
        assert_eq!(info.streams[0].sample_rate, Some(44100));
        assert_eq!(info.streams[0].channels, Some(2));
    }
    #[test]
    fn test_caf_short_data() {
        let mut data = Vec::new();
        data.extend_from_slice(b"caff");
        data.extend_from_slice(&1u16.to_be_bytes());
        data.extend_from_slice(&0u16.to_be_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "caf");
        assert!(info.streams.is_empty());
    }
    #[test]
    fn test_probe_tiff_le() {
        let mut data = vec![0u8; 128];
        data[0] = 0x49;
        data[1] = 0x49;
        data[2] = 0x2A;
        data[3] = 0x00;
        data[4..8].copy_from_slice(&8u32.to_le_bytes());
        data[8..10].copy_from_slice(&0u16.to_le_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "tiff");
    }
    #[test]
    fn test_probe_dng() {
        let mut data = vec![0u8; 128];
        data[0] = 0x49;
        data[1] = 0x49;
        data[2] = 0x2A;
        data[3] = 0x00;
        data[4..8].copy_from_slice(&8u32.to_le_bytes());
        data[8..10].copy_from_slice(&2u16.to_le_bytes());
        data[10..12].copy_from_slice(&0x0100u16.to_le_bytes());
        data[12..14].copy_from_slice(&3u16.to_le_bytes());
        data[14..18].copy_from_slice(&1u32.to_le_bytes());
        data[18..22].copy_from_slice(&4000u32.to_le_bytes());
        data[22..24].copy_from_slice(&0xC612u16.to_le_bytes());
        data[24..26].copy_from_slice(&1u16.to_le_bytes());
        data[26..30].copy_from_slice(&4u32.to_le_bytes());
        data[30..34].copy_from_slice(&1u32.to_le_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "dng");
        assert!(!info.streams.is_empty());
        assert_eq!(info.streams[0].stream_type, "video");
        assert_eq!(info.streams[0].codec, "raw");
        assert_eq!(info.streams[0].width, Some(4000));
    }
    #[test]
    fn test_probe_tiff_be() {
        let mut data = vec![0u8; 64];
        data[0] = 0x4D;
        data[1] = 0x4D;
        data[2] = 0x00;
        data[3] = 0x2A;
        data[4..8].copy_from_slice(&8u32.to_be_bytes());
        data[8..10].copy_from_slice(&0u16.to_be_bytes());
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "tiff");
    }
    #[test]
    fn test_probe_mxf() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&[0x06, 0x0E, 0x2B, 0x34]);
        data[4..8].copy_from_slice(&[0x02, 0x05, 0x01, 0x01]);
        data[8..12].copy_from_slice(&[0x0D, 0x01, 0x02, 0x01]);
        data[12..16].copy_from_slice(&[0x01, 0x02, 0x04, 0x00]);
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "mxf");
        assert!(info.metadata.contains_key("mxf_partition_type"));
        assert_eq!(
            info.metadata.get("mxf_partition_type"),
            Some(&"header_partition".to_string())
        );
    }
    #[test]
    fn test_probe_mxf_streams() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(&[0x06, 0x0E, 0x2B, 0x34]);
        data[4..8].copy_from_slice(&[0x02, 0x05, 0x01, 0x01]);
        data[8..12].copy_from_slice(&[0x0D, 0x01, 0x02, 0x01]);
        data[12..16].copy_from_slice(&[0x01, 0x03, 0x04, 0x00]);
        let info = MultiFormatProber::probe(&data);
        assert_eq!(info.format, "mxf");
        assert!(!info.streams.is_empty());
        assert_eq!(info.streams[0].codec, "mxf_essence");
    }
    #[test]
    fn test_integrity_empty() {
        let result = check_container_integrity(&[]);
        assert!(!result.valid);
        assert!(!result.issues.is_empty());
    }
    #[test]
    fn test_integrity_too_short() {
        let result = check_container_integrity(&[0x00, 0x01, 0x02]);
        assert!(!result.valid);
    }
    #[test]
    fn test_integrity_valid_mp4() {
        let mut data = Vec::new();
        data.extend_from_slice(&20u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"iso5");
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"iso5");
        let result = check_container_integrity(&data);
        assert!(result.valid);
        assert!(result.score > 0.5);
    }
    #[test]
    fn test_integrity_mp4_bad_box() {
        let mut data = Vec::new();
        data.extend_from_slice(&200u32.to_be_bytes());
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(&[0u8; 12]);
        let result = check_container_integrity(&data);
        assert!(result.score < 1.0);
    }
    #[test]
    fn test_integrity_valid_flac() {
        let mut data = vec![0u8; 50];
        data[0..4].copy_from_slice(b"fLaC");
        data[4] = 0x00;
        let result = check_container_integrity(&data);
        assert!(result.valid);
    }
    #[test]
    fn test_integrity_flac_short() {
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"fLaC");
        let result = check_container_integrity(&data);
        assert!(result.score < 1.0);
    }
    #[test]
    fn test_integrity_valid_wav() {
        let data_size: u32 = 36;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&data_size.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 16]);
        data.extend_from_slice(b"data");
        data.extend_from_slice(&0u32.to_le_bytes());
        let result = check_container_integrity(&data);
        assert!(result.valid);
    }
    #[test]
    fn test_integrity_riff_size_mismatch() {
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&100_000u32.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(&[0u8; 8]);
        let result = check_container_integrity(&data);
        assert!(result.score < 1.0);
    }
    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 0.5), 0.0);
    }
    #[test]
    fn test_percentile_single() {
        assert_eq!(percentile(&[42.0], 0.0), 42.0);
        assert_eq!(percentile(&[42.0], 1.0), 42.0);
    }
    #[test]
    fn test_percentile_five_elements() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 1.0), 5.0);
        let p50 = percentile(&data, 0.5);
        assert!((p50 - 3.0).abs() < 1e-9, "p50={p50}");
    }
    #[test]
    fn test_percentile_interpolation() {
        let data = [0.0, 10.0];
        let p25 = percentile(&data, 0.25);
        assert!((p25 - 2.5).abs() < 1e-9, "p25={p25}");
        let p75 = percentile(&data, 0.75);
        assert!((p75 - 7.5).abs() < 1e-9, "p75={p75}");
    }
    #[test]
    fn test_probe_detailed_empty_error() {
        let result = probe_detailed(&[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_probe_detailed_wav_single_stream() {
        let pcm_bytes: u32 = 44100 * 2;
        let total: u32 = 36 + pcm_bytes;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&total.to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&44100u32.to_le_bytes());
        data.extend_from_slice(&(44100u32 * 2).to_le_bytes());
        data.extend_from_slice(&2u16.to_le_bytes());
        data.extend_from_slice(&16u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&pcm_bytes.to_le_bytes());
        data.extend(vec![0u8; pcm_bytes as usize]);
        let stats = probe_detailed(&data).expect("probe_detailed should succeed on WAV");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].stream_index, 0);
        assert_eq!(stats[0].codec_id, "pcm");
        assert!(stats[0].keyframe_intervals_s.is_none());
        assert!((stats[0].bitrate_window_s - 1.0).abs() < f64::EPSILON);
    }
    #[test]
    fn test_probe_detailed_unknown_format_empty_streams() {
        let data = [0xFF_u8; 64];
        let stats = probe_detailed(&data).expect("probe_detailed should succeed on unknown data");
        assert!(stats.is_empty());
    }
    #[test]
    fn test_probe_detailed_flac_codec() {
        let mut data = Vec::new();
        data.extend_from_slice(b"fLaC");
        data.push(0x00);
        data.push(0x00);
        data.push(0x00);
        data.push(0x22);
        data.extend_from_slice(&[0u8; 10]);
        data.push(0xAC);
        data.push(0x44);
        data.push(0x42);
        data.push(0xF0);
        data.extend_from_slice(&[0u8; 20]);
        let stats = probe_detailed(&data).expect("probe_detailed should succeed on FLAC");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].codec_id, "flac");
    }
    #[test]
    fn test_detailed_stream_stats_fields() {
        let s = DetailedStreamStats {
            stream_index: 2,
            codec_id: "av1".into(),
            duration_s: 10.5,
            bitrate_window_s: 1.0,
            bitrate_histogram: vec![1_000_000, 2_000_000],
            bitrate_mean: 1_500_000.0,
            bitrate_p50: 1_500_000.0,
            bitrate_p95: 1_900_000.0,
            bitrate_max: 2_000_000.0,
            keyframe_intervals_s: Some(vec![2.0, 2.0, 2.0]),
            keyframe_interval_mean: Some(2.0),
            keyframe_interval_p50: Some(2.0),
            keyframe_interval_p95: Some(2.0),
            keyframe_interval_max: Some(2.0),
        };
        assert_eq!(s.stream_index, 2);
        assert_eq!(s.codec_id, "av1");
        assert!((s.duration_s - 10.5).abs() < f64::EPSILON);
        assert_eq!(s.bitrate_histogram.len(), 2);
        assert!(s.keyframe_intervals_s.is_some());
    }
    #[test]
    fn test_probe_detailed_mpegts_synthetic() {
        let mut data = Vec::new();
        let make_ts_pkt = |pid: u16, pusi: bool, payload: &[u8]| -> [u8; 188] {
            let mut pkt = [0u8; 188];
            pkt[0] = 0x47;
            pkt[1] = (if pusi { 0x40 } else { 0x00 }) | ((pid >> 8) as u8 & 0x1F);
            pkt[2] = (pid & 0xFF) as u8;
            pkt[3] = 0x10;
            let copy_len = payload.len().min(184);
            pkt[4..4 + copy_len].copy_from_slice(&payload[..copy_len]);
            pkt
        };
        let mut pat_payload = vec![0u8; 184];
        pat_payload[0] = 0x00;
        pat_payload[1] = 0x00;
        pat_payload[2] = 0xB0;
        pat_payload[3] = 0x0D;
        pat_payload[4] = 0x00;
        pat_payload[5] = 0x01;
        pat_payload[6] = 0xC1;
        pat_payload[7] = 0x00;
        pat_payload[8] = 0x00;
        pat_payload[9] = 0x00;
        pat_payload[10] = 0x01;
        pat_payload[11] = 0xE0 | 0x00;
        pat_payload[12] = 0x10;
        data.extend_from_slice(&make_ts_pkt(0x0000, true, &pat_payload));
        let mut pmt_payload = vec![0u8; 184];
        pmt_payload[0] = 0x00;
        pmt_payload[1] = 0x02;
        pmt_payload[2] = 0xB0;
        pmt_payload[3] = 0x12;
        pmt_payload[4] = 0x00;
        pmt_payload[5] = 0x01;
        pmt_payload[6] = 0xC1;
        pmt_payload[7] = 0x00;
        pmt_payload[8] = 0x00;
        pmt_payload[9] = 0xE1;
        pmt_payload[10] = 0x00;
        pmt_payload[11] = 0xF0;
        pmt_payload[12] = 0x00;
        pmt_payload[13] = 0x85;
        pmt_payload[14] = 0xE1;
        pmt_payload[15] = 0x00;
        pmt_payload[16] = 0xF0;
        pmt_payload[17] = 0x00;
        data.extend_from_slice(&make_ts_pkt(0x0010, true, &pmt_payload));
        let mut pes1 = vec![0u8; 184];
        pes1[0] = 0x00;
        pes1[1] = 0x00;
        pes1[2] = 0x01;
        pes1[3] = 0xE0;
        pes1[4] = 0x00;
        pes1[5] = 0x00;
        pes1[6] = 0x80;
        pes1[7] = 0x80;
        pes1[8] = 0x05;
        let pts1: u64 = 90_000;
        pes1[9] = 0x21 | (((pts1 >> 29) & 0x0E) as u8);
        pes1[10] = ((pts1 >> 22) & 0xFF) as u8;
        pes1[11] = (((pts1 >> 14) & 0xFE) as u8) | 0x01;
        pes1[12] = ((pts1 >> 7) & 0xFF) as u8;
        pes1[13] = (((pts1 & 0x7F) << 1) as u8) | 0x01;
        data.extend_from_slice(&make_ts_pkt(0x0100, true, &pes1));
        let mut pes2 = pes1.clone();
        let pts2: u64 = 270_000;
        pes2[9] = 0x21 | (((pts2 >> 29) & 0x0E) as u8);
        pes2[10] = ((pts2 >> 22) & 0xFF) as u8;
        pes2[11] = (((pts2 >> 14) & 0xFE) as u8) | 0x01;
        pes2[12] = ((pts2 >> 7) & 0xFF) as u8;
        pes2[13] = (((pts2 & 0x7F) << 1) as u8) | 0x01;
        data.extend_from_slice(&make_ts_pkt(0x0100, true, &pes2));
        let stats = probe_detailed(&data).expect("probe_detailed should succeed on TS");
        assert!(!stats.is_empty(), "Expected at least one stream");
        let video_stat = stats.iter().find(|s| !s.bitrate_histogram.is_empty());
        assert!(
            video_stat.is_some(),
            "Expected at least one stream with non-empty bitrate histogram"
        );
    }
}
