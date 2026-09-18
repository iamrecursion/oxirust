//! Proxy preset definitions for common workflows.

use super::settings::ProxyGenerationSettings;

/// Standard proxy presets for common use cases.
pub struct ProxyPresets;

impl ProxyPresets {
    /// Quarter resolution H.264 preset - optimized for remote editing.
    ///
    /// - 25% scale factor (e.g., 1920x1080 -> 480x270)
    /// - H.264 codec with fast encoding preset
    /// - 2 Mbps video bitrate
    /// - AAC 128 kbps audio
    /// - MP4 container
    ///
    /// Best for: Remote editing over slow connections, low-powered systems.
    #[must_use]
    pub fn quarter_res_h264_remote() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.25,
            codec: "h264".to_string(),
            bitrate: 1_500_000, // 1.5 Mbps for remote
            audio_codec: "aac".to_string(),
            audio_bitrate: 96_000, // Lower for remote
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "veryfast".to_string(),
        }
    }

    /// Quarter resolution H.264 preset - balanced quality.
    ///
    /// - 25% scale factor
    /// - H.264 codec with medium encoding preset
    /// - 2.5 Mbps video bitrate
    /// - AAC 128 kbps audio
    /// - MP4 container
    ///
    /// Best for: Standard offline editing workflows.
    #[must_use]
    pub fn quarter_res_h264_balanced() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.25,
            codec: "h264".to_string(),
            bitrate: 2_500_000,
            audio_codec: "aac".to_string(),
            audio_bitrate: 128_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "medium".to_string(),
        }
    }

    /// Half resolution H.264 preset - high quality proxy.
    ///
    /// - 50% scale factor (e.g., 1920x1080 -> 960x540)
    /// - H.264 codec with medium encoding preset
    /// - 5 Mbps video bitrate
    /// - AAC 192 kbps audio
    /// - MP4 container
    ///
    /// Best for: Color grading preview, high-quality offline edit.
    #[must_use]
    pub fn half_res_h264_high_quality() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.5,
            codec: "h264".to_string(),
            bitrate: 5_000_000,
            audio_codec: "aac".to_string(),
            audio_bitrate: 192_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "medium".to_string(),
        }
    }

    /// Full resolution H.264 preset - editing proxy.
    ///
    /// - 100% scale factor (same resolution as original)
    /// - H.264 codec with fast encoding
    /// - 15 Mbps video bitrate
    /// - AAC 256 kbps audio
    /// - MP4 container
    ///
    /// Best for: Maintaining resolution while reducing file size.
    #[must_use]
    pub fn full_res_h264_editing() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 1.0,
            codec: "h264".to_string(),
            bitrate: 15_000_000,
            audio_codec: "aac".to_string(),
            audio_bitrate: 256_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "fast".to_string(),
        }
    }

    /// Quarter resolution VP9 preset - maximum compression.
    ///
    /// - 25% scale factor
    /// - VP9 codec with good quality
    /// - 1 Mbps video bitrate
    /// - Opus 96 kbps audio
    /// - WebM container
    ///
    /// Best for: Minimal file size, cloud storage.
    #[must_use]
    pub fn quarter_res_vp9_compressed() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.25,
            codec: "vp9".to_string(),
            bitrate: 1_000_000,
            audio_codec: "opus".to_string(),
            audio_bitrate: 96_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "webm".to_string(),
            use_hw_accel: false, // VP9 HW accel not always available
            threads: 4,
            quality_preset: "good".to_string(),
        }
    }

    /// Half resolution VP9 preset - balanced compression.
    ///
    /// - 50% scale factor
    /// - VP9 codec with good quality
    /// - 3 Mbps video bitrate
    /// - Opus 128 kbps audio
    /// - WebM container
    ///
    /// Best for: Good quality with small files.
    #[must_use]
    pub fn half_res_vp9_balanced() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.5,
            codec: "vp9".to_string(),
            bitrate: 3_000_000,
            audio_codec: "opus".to_string(),
            audio_bitrate: 128_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "webm".to_string(),
            use_hw_accel: false,
            threads: 4,
            quality_preset: "good".to_string(),
        }
    }

    /// DNxHD proxy preset (when patent-free).
    ///
    /// - 50% scale factor
    /// - DNxHD 36 Mbps
    /// - PCM audio
    /// - MOV container
    ///
    /// Best for: Avid Media Composer workflows.
    #[must_use]
    pub fn dnxhd_proxy() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.5,
            codec: "dnxhd".to_string(),
            bitrate: 36_000_000,
            audio_codec: "pcm".to_string(),
            audio_bitrate: 1_536_000, // 48kHz 16-bit stereo
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mov".to_string(),
            use_hw_accel: false,
            threads: 0,
            quality_preset: "dnxhr_lb".to_string(),
        }
    }

    /// ProRes Proxy preset (when patent-free).
    ///
    /// - 50% scale factor
    /// - ProRes Proxy
    /// - PCM audio
    /// - MOV container
    ///
    /// Best for: Final Cut Pro / DaVinci Resolve workflows.
    #[must_use]
    pub fn prores_proxy() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.5,
            codec: "prores".to_string(),
            bitrate: 45_000_000,
            audio_codec: "pcm".to_string(),
            audio_bitrate: 1_536_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mov".to_string(),
            use_hw_accel: false,
            threads: 0,
            quality_preset: "proxy".to_string(),
        }
    }

    /// Social media preview preset.
    ///
    /// - 33% scale factor
    /// - H.264 with baseline profile
    /// - 3 Mbps video bitrate
    /// - AAC 128 kbps audio
    /// - MP4 container
    ///
    /// Best for: Quick previews for social media review.
    #[must_use]
    pub fn social_media_preview() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.33,
            codec: "h264".to_string(),
            bitrate: 3_000_000,
            audio_codec: "aac".to_string(),
            audio_bitrate: 128_000,
            preserve_frame_rate: true,
            preserve_timecode: false, // Not critical for preview
            preserve_metadata: false,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "veryfast".to_string(),
        }
    }

    /// Archive proxy preset - high quality preservation.
    ///
    /// - 75% scale factor
    /// - H.264 High Profile
    /// - 20 Mbps video bitrate
    /// - AAC 320 kbps audio
    /// - MP4 container
    ///
    /// Best for: Long-term archival with smaller files.
    #[must_use]
    pub fn archive_high_quality() -> ProxyGenerationSettings {
        ProxyGenerationSettings {
            scale_factor: 0.75,
            codec: "h264".to_string(),
            bitrate: 20_000_000,
            audio_codec: "aac".to_string(),
            audio_bitrate: 320_000,
            preserve_frame_rate: true,
            preserve_timecode: true,
            preserve_metadata: true,
            container: "mp4".to_string(),
            use_hw_accel: true,
            threads: 0,
            quality_preset: "slow".to_string(),
        }
    }

    /// Get all available presets with descriptions.
    #[must_use]
    pub fn all_presets() -> Vec<PresetInfo> {
        vec![
            PresetInfo {
                name: "Quarter Res H.264 (Remote)".to_string(),
                description: "Optimized for remote editing over slow connections".to_string(),
                settings: Self::quarter_res_h264_remote(),
                typical_size_reduction: 0.95, // 95% size reduction
            },
            PresetInfo {
                name: "Quarter Res H.264 (Balanced)".to_string(),
                description: "Standard offline editing workflow".to_string(),
                settings: Self::quarter_res_h264_balanced(),
                typical_size_reduction: 0.94,
            },
            PresetInfo {
                name: "Half Res H.264 (High Quality)".to_string(),
                description: "High-quality offline edit and color grading preview".to_string(),
                settings: Self::half_res_h264_high_quality(),
                typical_size_reduction: 0.85,
            },
            PresetInfo {
                name: "Full Res H.264 (Editing)".to_string(),
                description: "Maintain resolution while reducing file size".to_string(),
                settings: Self::full_res_h264_editing(),
                typical_size_reduction: 0.70,
            },
            PresetInfo {
                name: "Quarter Res VP9 (Compressed)".to_string(),
                description: "Maximum compression for cloud storage".to_string(),
                settings: Self::quarter_res_vp9_compressed(),
                typical_size_reduction: 0.96,
            },
            PresetInfo {
                name: "Half Res VP9 (Balanced)".to_string(),
                description: "Good quality with small files".to_string(),
                settings: Self::half_res_vp9_balanced(),
                typical_size_reduction: 0.88,
            },
            PresetInfo {
                name: "DNxHD Proxy".to_string(),
                description: "Avid Media Composer workflow".to_string(),
                settings: Self::dnxhd_proxy(),
                typical_size_reduction: 0.60,
            },
            PresetInfo {
                name: "ProRes Proxy".to_string(),
                description: "Final Cut Pro / DaVinci Resolve workflow".to_string(),
                settings: Self::prores_proxy(),
                typical_size_reduction: 0.65,
            },
            PresetInfo {
                name: "Social Media Preview".to_string(),
                description: "Quick previews for social media review".to_string(),
                settings: Self::social_media_preview(),
                typical_size_reduction: 0.92,
            },
            PresetInfo {
                name: "Archive (High Quality)".to_string(),
                description: "Long-term archival with smaller files".to_string(),
                settings: Self::archive_high_quality(),
                typical_size_reduction: 0.75,
            },
        ]
    }

    /// Find a preset by name (case-insensitive partial match).
    #[must_use]
    pub fn find_preset(name: &str) -> Option<ProxyGenerationSettings> {
        let name_lower = name.to_lowercase();
        Self::all_presets()
            .into_iter()
            .find(|p| p.name.to_lowercase().contains(&name_lower))
            .map(|p| p.settings)
    }
}

/// Information about a proxy preset.
#[derive(Debug, Clone)]
pub struct PresetInfo {
    /// Preset name.
    pub name: String,

    /// Preset description.
    pub description: String,

    /// Proxy generation settings.
    pub settings: ProxyGenerationSettings,

    /// Typical file size reduction (0.0 to 1.0, where 0.95 = 95% smaller).
    pub typical_size_reduction: f64,
}

impl PresetInfo {
    /// Get the estimated output size for a given input size.
    #[must_use]
    pub fn estimated_output_size(&self, input_size: u64) -> u64 {
        (input_size as f64 * (1.0 - self.typical_size_reduction)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets() {
        let presets = ProxyPresets::all_presets();
        assert_eq!(presets.len(), 10);

        for preset in &presets {
            assert!(!preset.name.is_empty());
            assert!(!preset.description.is_empty());
            assert!(preset.typical_size_reduction >= 0.0);
            assert!(preset.typical_size_reduction <= 1.0);
        }
    }

    #[test]
    fn test_quarter_res_remote() {
        let settings = ProxyPresets::quarter_res_h264_remote();
        assert_eq!(settings.scale_factor, 0.25);
        assert_eq!(settings.bitrate, 1_500_000);
        assert_eq!(settings.quality_preset, "veryfast");
    }

    #[test]
    fn test_half_res_high_quality() {
        let settings = ProxyPresets::half_res_h264_high_quality();
        assert_eq!(settings.scale_factor, 0.5);
        assert_eq!(settings.bitrate, 5_000_000);
        assert_eq!(settings.audio_bitrate, 192_000);
    }

    #[test]
    fn test_dnxhd_proxy() {
        let settings = ProxyPresets::dnxhd_proxy();
        assert_eq!(settings.codec, "dnxhd");
        assert_eq!(settings.container, "mov");
        assert_eq!(settings.audio_codec, "pcm");
    }

    #[test]
    fn test_find_preset() {
        let settings = ProxyPresets::find_preset("quarter");
        assert!(settings.is_some());

        let settings = ProxyPresets::find_preset("prores");
        assert!(settings.is_some());
        assert_eq!(settings.expect("should succeed in test").codec, "prores");

        let settings = ProxyPresets::find_preset("nonexistent");
        assert!(settings.is_none());
    }

    #[test]
    fn test_preset_info() {
        let presets = ProxyPresets::all_presets();
        let preset = &presets[0];

        let input_size = 1_000_000_000; // 1 GB
        let output_size = preset.estimated_output_size(input_size);

        assert!(output_size < input_size);
        assert!(output_size > 0);
    }

    #[test]
    fn test_vp9_presets() {
        let quarter = ProxyPresets::quarter_res_vp9_compressed();
        assert_eq!(quarter.codec, "vp9");
        assert_eq!(quarter.container, "webm");
        assert_eq!(quarter.audio_codec, "opus");

        let half = ProxyPresets::half_res_vp9_balanced();
        assert_eq!(half.codec, "vp9");
        assert_eq!(half.scale_factor, 0.5);
    }

    #[test]
    fn test_social_media_preset() {
        let settings = ProxyPresets::social_media_preview();
        assert_eq!(settings.scale_factor, 0.33);
        assert!(!settings.preserve_timecode);
        assert!(!settings.preserve_metadata);
    }

    #[test]
    fn test_archive_preset() {
        let settings = ProxyPresets::archive_high_quality();
        assert_eq!(settings.scale_factor, 0.75);
        assert_eq!(settings.bitrate, 20_000_000);
        assert_eq!(settings.audio_bitrate, 320_000);
        assert_eq!(settings.quality_preset, "slow");
    }
}
