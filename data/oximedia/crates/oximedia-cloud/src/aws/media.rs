//! AWS Media Services integration

use aws_sdk_cloudwatch::Client as CloudWatchClient;
use aws_sdk_mediaconvert::Client as MediaConvertClient;
use aws_sdk_medialive::Client as MediaLiveClient;
use aws_sdk_mediapackage::Client as MediaPackageClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{CloudError, Result};

/// AWS Media Services wrapper
pub struct AwsMediaServices {
    media_convert: MediaConvertClient,
    media_live: MediaLiveClient,
    media_package: MediaPackageClient,
    cloud_watch: CloudWatchClient,
}

impl AwsMediaServices {
    /// Create new AWS Media Services client
    ///
    /// # Errors
    ///
    /// Returns an error if AWS SDK initialization fails
    pub async fn new(region: String) -> Result<Self> {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new(region))
            .load()
            .await;

        let media_convert = MediaConvertClient::new(&config);
        let media_live = MediaLiveClient::new(&config);
        let media_package = MediaPackageClient::new(&config);
        let cloud_watch = CloudWatchClient::new(&config);

        Ok(Self {
            media_convert,
            media_live,
            media_package,
            cloud_watch,
        })
    }

    /// Submit an AWS `MediaConvert` transcoding job.
    ///
    /// Builds a complete `JobSettings` structure including:
    /// - Input from the provided S3 URI with audio/video selectors
    /// - Output group using `FILE_GROUP_SETTINGS` targeting the output S3 prefix
    /// - H.264/AAC encoding parameters derived from `job_config.settings`
    ///
    /// Optional keys in `job_config.settings`:
    /// - `"width"` – video width in pixels (default: 1920)
    /// - `"height"` – video height in pixels (default: 1080)
    /// - `"video_bitrate"` – video bitrate in bits/s (default: 5_000_000)
    /// - `"audio_bitrate"` – audio bitrate in bits/s (default: 192_000)
    /// - `"framerate_numerator"` – framerate numerator (default: 30)
    /// - `"framerate_denominator"` – framerate denominator (default: 1)
    /// - `"output_name_modifier"` – suffix appended to output filenames (default: `"_output"`)
    ///
    /// # Errors
    ///
    /// Returns `CloudError::MediaService` if the `MediaConvert` API call fails.
    pub async fn submit_transcode_job(&self, job_config: MediaConvertJobConfig) -> Result<String> {
        use aws_sdk_mediaconvert::types::{
            AacCodingMode, AacRateControlMode, AacSettings, AacSpecification, AudioCodec,
            AudioCodecSettings, AudioDefaultSelection, AudioDescription, AudioSelector,
            ContainerSettings, ContainerType, FileGroupSettings, H264CodecLevel, H264CodecProfile,
            H264EntropyEncoding, H264FramerateControl, H264ParControl, H264QualityTuningLevel,
            H264RateControlMode, H264SceneChangeDetect, H264Settings, Input, JobSettings, Output,
            OutputGroup, OutputGroupSettings, OutputGroupType, VideoCodec, VideoCodecSettings,
            VideoDescription, VideoSelector,
        };

        // Parse optional encoding parameters from job_config.settings
        let width = job_config
            .settings
            .get("width")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1920);

        let height = job_config
            .settings
            .get("height")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1080);

        let video_bitrate = job_config
            .settings
            .get("video_bitrate")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(5_000_000);

        let audio_bitrate = job_config
            .settings
            .get("audio_bitrate")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(192_000);

        let framerate_numerator = job_config
            .settings
            .get("framerate_numerator")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(30);

        let framerate_denominator = job_config
            .settings
            .get("framerate_denominator")
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(1);

        let output_name_modifier = job_config
            .settings
            .get("output_name_modifier")
            .map(String::as_str)
            .unwrap_or("_output");

        // Build H.264 video codec settings
        let h264_settings = H264Settings::builder()
            .codec_level(H264CodecLevel::Auto)
            .codec_profile(H264CodecProfile::High)
            .entropy_encoding(H264EntropyEncoding::Cabac)
            .framerate_control(H264FramerateControl::Specified)
            .framerate_numerator(framerate_numerator)
            .framerate_denominator(framerate_denominator)
            .par_control(H264ParControl::Specified)
            .par_numerator(1)
            .par_denominator(1)
            .quality_tuning_level(H264QualityTuningLevel::SinglePassHq)
            .rate_control_mode(H264RateControlMode::Cbr)
            .bitrate(video_bitrate)
            .scene_change_detect(H264SceneChangeDetect::Enabled)
            .build();

        let video_description = VideoDescription::builder()
            .width(width)
            .height(height)
            .codec_settings(
                VideoCodecSettings::builder()
                    .codec(VideoCodec::H264)
                    .h264_settings(h264_settings)
                    .build(),
            )
            .build();

        // Build AAC audio codec settings
        let aac_settings = AacSettings::builder()
            .bitrate(audio_bitrate)
            .coding_mode(AacCodingMode::CodingMode20)
            .rate_control_mode(AacRateControlMode::Cbr)
            .specification(AacSpecification::Mpeg4)
            .build();

        // audio_source_name links this AudioDescription to the named AudioSelector on the input
        let audio_description = AudioDescription::builder()
            .audio_source_name("Audio Selector 1")
            .codec_settings(
                AudioCodecSettings::builder()
                    .codec(AudioCodec::Aac)
                    .aac_settings(aac_settings)
                    .build(),
            )
            .build();

        // Build output with the MP4 container
        let output = Output::builder()
            .container_settings(
                ContainerSettings::builder()
                    .container(ContainerType::Mp4)
                    .build(),
            )
            .video_description(video_description)
            .audio_descriptions(audio_description)
            .name_modifier(output_name_modifier)
            .build();

        // Build output group targeting the S3 output URI prefix
        let file_group_settings = FileGroupSettings::builder()
            .destination(&job_config.output_uri)
            .build();

        let output_group = OutputGroup::builder()
            .name("File Group")
            .output_group_settings(
                OutputGroupSettings::builder()
                    .r#type(OutputGroupType::FileGroupSettings)
                    .file_group_settings(file_group_settings)
                    .build(),
            )
            .outputs(output)
            .build();

        // Build input with an audio selector and video selector
        let audio_selector = AudioSelector::builder()
            .default_selection(AudioDefaultSelection::Default)
            .build();

        let input = Input::builder()
            .file_input(&job_config.input_uri)
            .audio_selectors("Audio Selector 1", audio_selector)
            .video_selector(VideoSelector::builder().build())
            .build();

        // Assemble job settings
        let job_settings = JobSettings::builder()
            .inputs(input)
            .output_groups(output_group)
            .build();

        // Submit the job to AWS MediaConvert
        let response = self
            .media_convert
            .create_job()
            .role(&job_config.role_arn)
            .settings(job_settings)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to submit MediaConvert job: {e}"))
            })?;

        let job_id = response.job().and_then(|j| j.id()).ok_or_else(|| {
            CloudError::MediaService("No job ID in MediaConvert response".to_string())
        })?;

        tracing::info!("Submitted MediaConvert job: {job_id}");
        Ok(job_id.to_string())
    }

    /// Get `MediaConvert` job status
    ///
    /// # Errors
    ///
    /// Returns an error if retrieving job status fails
    pub async fn get_job_status(&self, job_id: &str) -> Result<JobStatus> {
        let output = self
            .media_convert
            .get_job()
            .id(job_id)
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to get job status: {e}")))?;

        let job = output
            .job()
            .ok_or_else(|| CloudError::MediaService("No job data returned".to_string()))?;

        let status = match job.status() {
            Some(aws_sdk_mediaconvert::types::JobStatus::Submitted) => JobStatus::Submitted,
            Some(aws_sdk_mediaconvert::types::JobStatus::Progressing) => JobStatus::InProgress,
            Some(aws_sdk_mediaconvert::types::JobStatus::Complete) => JobStatus::Completed,
            Some(aws_sdk_mediaconvert::types::JobStatus::Error) => JobStatus::Failed,
            Some(aws_sdk_mediaconvert::types::JobStatus::Canceled) => JobStatus::Cancelled,
            _ => JobStatus::Unknown,
        };

        Ok(status)
    }

    /// Create `MediaLive` channel
    ///
    /// # Errors
    ///
    /// Returns an error if channel creation fails
    pub async fn create_live_channel(&self, config: LiveChannelConfig) -> Result<String> {
        use aws_sdk_medialive::types::{ChannelClass, InputSpecification};

        let input_spec = InputSpecification::builder()
            .codec(aws_sdk_medialive::types::InputCodec::Avc)
            .resolution(aws_sdk_medialive::types::InputResolution::Hd)
            .maximum_bitrate(aws_sdk_medialive::types::InputMaximumBitrate::Max20Mbps)
            .build();

        let output = self
            .media_live
            .create_channel()
            .name(&config.channel_name)
            .channel_class(ChannelClass::SinglePipeline)
            .input_specification(input_spec)
            .role_arn(&config.role_arn)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to create MediaLive channel: {e}"))
            })?;

        Ok(output
            .channel()
            .and_then(|c| c.id())
            .unwrap_or_default()
            .to_string())
    }

    /// Start `MediaLive` channel
    ///
    /// # Errors
    ///
    /// Returns an error if starting the channel fails
    pub async fn start_channel(&self, channel_id: &str) -> Result<()> {
        self.media_live
            .start_channel()
            .channel_id(channel_id)
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to start channel: {e}")))?;

        Ok(())
    }

    /// Stop `MediaLive` channel
    ///
    /// # Errors
    ///
    /// Returns an error if stopping the channel fails
    pub async fn stop_channel(&self, channel_id: &str) -> Result<()> {
        self.media_live
            .stop_channel()
            .channel_id(channel_id)
            .send()
            .await
            .map_err(|e| CloudError::MediaService(format!("Failed to stop channel: {e}")))?;

        Ok(())
    }

    /// Create `MediaPackage` endpoint
    ///
    /// # Errors
    ///
    /// Returns an error if endpoint creation fails
    pub async fn create_packaging_endpoint(
        &self,
        config: PackagingEndpointConfig,
    ) -> Result<String> {
        let output = self
            .media_package
            .create_origin_endpoint()
            .channel_id(&config.channel_id)
            .id(&config.endpoint_id)
            .send()
            .await
            .map_err(|e| {
                CloudError::MediaService(format!("Failed to create packaging endpoint: {e}"))
            })?;

        Ok(output.url().unwrap_or_default().to_string())
    }

    /// Get `CloudWatch` metrics for a `MediaConvert` job.
    ///
    /// Queries the CloudWatch `GetMetricStatistics` API for the metrics:
    /// - `JobDuration` (returned as `duration_seconds`)
    /// - `InputFileSize` (returned as `input_size_bytes`)
    /// - `OutputFileSize` (returned as `output_size_bytes`)
    ///
    /// Uses a 24-hour lookback window with a 86400-second period and `Sum` statistic.
    ///
    /// # Errors
    ///
    /// Returns `CloudError::MediaService` if the CloudWatch API call fails.
    pub async fn get_metrics(&self, job_id: &str) -> Result<HashMap<String, f64>> {
        use aws_sdk_cloudwatch::types::{Dimension, Statistic};
        use aws_smithy_types::DateTime as SmithyDateTime;
        use chrono::Utc;

        let now = Utc::now();
        let start = now - chrono::Duration::seconds(86_400);

        let end_time = SmithyDateTime::from_secs(now.timestamp());
        let start_time = SmithyDateTime::from_secs(start.timestamp());

        // Map of (CW metric name, output key)
        let metric_defs: &[(&str, &str)] = &[
            ("JobDuration", "duration_seconds"),
            ("InputFileSize", "input_size_bytes"),
            ("OutputFileSize", "output_size_bytes"),
        ];

        let mut result = HashMap::new();

        for (cw_metric, output_key) in metric_defs {
            let dimension = Dimension::builder().name("Job").value(job_id).build();

            let resp = self
                .cloud_watch
                .get_metric_statistics()
                .namespace("AWS/MediaConvert")
                .metric_name(*cw_metric)
                .dimensions(dimension)
                .start_time(start_time)
                .end_time(end_time)
                .period(86_400)
                .statistics(Statistic::Sum)
                .send()
                .await
                .map_err(|e| {
                    CloudError::MediaService(format!(
                        "CloudWatch GetMetricStatistics failed for {cw_metric}: {e}"
                    ))
                })?;

            let value = resp
                .datapoints()
                .iter()
                .filter_map(|dp| dp.sum())
                .fold(0.0_f64, |acc, v| acc + v);

            result.insert((*output_key).to_string(), value);
        }

        Ok(result)
    }
}

/// `MediaConvert` job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaConvertJobConfig {
    /// Input URI (s3://bucket/path/input.mp4)
    pub input_uri: String,
    /// Output URI prefix (s3://bucket/output/)
    pub output_uri: String,
    /// IAM role ARN that MediaConvert will assume
    pub role_arn: String,
    /// Optional encoding overrides (see `submit_transcode_job` for supported keys)
    pub settings: HashMap<String, String>,
}

/// Job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job submitted
    Submitted,
    /// Job in progress
    InProgress,
    /// Job completed
    Completed,
    /// Job failed
    Failed,
    /// Job cancelled
    Cancelled,
    /// Unknown status
    Unknown,
}

/// Live channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveChannelConfig {
    /// Channel name
    pub channel_name: String,
    /// IAM role ARN
    pub role_arn: String,
    /// Input attachments
    pub input_ids: Vec<String>,
    /// Encoder settings
    pub encoder_settings: HashMap<String, String>,
}

/// Packaging endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagingEndpointConfig {
    /// Channel ID
    pub channel_id: String,
    /// Endpoint ID
    pub endpoint_id: String,
    /// Manifest settings
    pub manifest_settings: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status() {
        assert_eq!(JobStatus::Submitted, JobStatus::Submitted);
        assert_ne!(JobStatus::InProgress, JobStatus::Completed);
    }

    #[test]
    fn test_media_convert_job_config_defaults() {
        let config = MediaConvertJobConfig {
            input_uri: "s3://bucket/input.mp4".to_string(),
            output_uri: "s3://bucket/output/".to_string(),
            role_arn: "arn:aws:iam::123456789012:role/MediaConvert".to_string(),
            settings: HashMap::new(),
        };

        assert!(!config.input_uri.is_empty());
        assert!(config.input_uri.starts_with("s3://"));
        assert!(config.output_uri.ends_with('/'));
    }

    #[test]
    fn test_media_convert_job_config_with_overrides() {
        let mut settings = HashMap::new();
        settings.insert("width".to_string(), "1280".to_string());
        settings.insert("height".to_string(), "720".to_string());
        settings.insert("video_bitrate".to_string(), "3000000".to_string());
        settings.insert("audio_bitrate".to_string(), "128000".to_string());
        settings.insert("framerate_numerator".to_string(), "25".to_string());
        settings.insert("framerate_denominator".to_string(), "1".to_string());

        let config = MediaConvertJobConfig {
            input_uri: "s3://my-bucket/input/source.mov".to_string(),
            output_uri: "s3://my-bucket/output/hd/".to_string(),
            role_arn: "arn:aws:iam::999999999999:role/MediaConvertRole".to_string(),
            settings,
        };

        assert_eq!(config.settings.get("width"), Some(&"1280".to_string()));
        assert_eq!(
            config.settings.get("video_bitrate"),
            Some(&"3000000".to_string())
        );
        assert_eq!(
            config.settings.get("framerate_numerator"),
            Some(&"25".to_string())
        );
    }

    #[test]
    fn test_live_channel_config() {
        let config = LiveChannelConfig {
            channel_name: "live-sports-1".to_string(),
            role_arn: "arn:aws:iam::123456789012:role/MediaLiveRole".to_string(),
            input_ids: vec!["input-abc123".to_string()],
            encoder_settings: HashMap::new(),
        };

        assert_eq!(config.channel_name, "live-sports-1");
        assert_eq!(config.input_ids.len(), 1);
    }

    #[test]
    fn test_packaging_endpoint_config() {
        let config = PackagingEndpointConfig {
            channel_id: "channel-abc".to_string(),
            endpoint_id: "hls-endpoint-1".to_string(),
            manifest_settings: HashMap::new(),
        };

        assert_eq!(config.channel_id, "channel-abc");
        assert_eq!(config.endpoint_id, "hls-endpoint-1");
    }
}
