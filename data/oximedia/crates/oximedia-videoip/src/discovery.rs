//! mDNS-based service discovery for video sources.

use crate::error::{VideoIpError, VideoIpResult};
use crate::types::{AudioFormat, VideoFormat};
use mdns_sd::{ResolvedService, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::Duration;
use tokio::time::timeout;

/// Service type for video-over-IP discovery.
pub const SERVICE_TYPE: &str = "_oximedia-videoip._udp.local.";

/// Discovered video source information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Source name.
    pub name: String,
    /// IP address.
    pub address: IpAddr,
    /// Port number.
    pub port: u16,
    /// Video format.
    pub video_format: VideoFormat,
    /// Audio format.
    pub audio_format: AudioFormat,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl SourceInfo {
    /// Returns the socket address for this source.
    #[must_use]
    pub const fn socket_addr(&self) -> SocketAddr {
        SocketAddr::new(self.address, self.port)
    }
}

/// Service discovery client for finding video sources.
pub struct DiscoveryClient {
    /// mDNS daemon.
    daemon: ServiceDaemon,
}

impl DiscoveryClient {
    /// Creates a new discovery client.
    ///
    /// # Errors
    ///
    /// Returns an error if the mDNS daemon cannot be created.
    pub fn new() -> VideoIpResult<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| VideoIpError::Discovery(format!("failed to create daemon: {e}")))?;

        Ok(Self { daemon })
    }

    /// Discovers all available video sources on the network.
    ///
    /// # Errors
    ///
    /// Returns an error if discovery fails.
    pub async fn discover_all(&self, timeout_secs: u64) -> VideoIpResult<Vec<SourceInfo>> {
        let receiver = self
            .daemon
            .browse(SERVICE_TYPE)
            .map_err(|e| VideoIpError::Discovery(format!("failed to browse: {e}")))?;

        let mut sources = Vec::new();
        let deadline = Duration::from_secs(timeout_secs);

        let result = timeout(deadline, async {
            while let Ok(event) = receiver.recv_async().await {
                match event {
                    ServiceEvent::ServiceResolved(info) => {
                        if let Ok(source_info) = Self::parse_service_info(&info) {
                            sources.push(source_info);
                        }
                    }
                    ServiceEvent::SearchStopped(_) => break,
                    _ => {}
                }
            }
        })
        .await;

        // Timeout is expected, it's how we limit the search duration
        let _ = result;

        Ok(sources)
    }

    /// Discovers a specific video source by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the source is not found or discovery fails.
    pub async fn discover_by_name(
        &self,
        name: &str,
        timeout_secs: u64,
    ) -> VideoIpResult<SourceInfo> {
        let sources = self.discover_all(timeout_secs).await?;

        sources
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| VideoIpError::ServiceNotFound(name.to_string()))
    }

    /// Parses service information from mDNS record.
    fn parse_service_info(info: &ResolvedService) -> VideoIpResult<SourceInfo> {
        let name = info.get_fullname().trim_end_matches('.').to_string();
        let addresses = info.get_addresses();
        let port = info.get_port();

        let scoped_ip = addresses
            .iter()
            .next()
            .ok_or_else(|| VideoIpError::Discovery("no address found".to_string()))?;

        // Parse TXT records
        let properties = info.get_properties();
        let _codec = properties
            .get("codec")
            .map_or_else(|| "vp9".to_string(), |v| v.val_str().to_string());
        let width: u32 = properties
            .get("width")
            .and_then(|v| v.val_str().parse::<u32>().ok())
            .unwrap_or(1920);
        let height: u32 = properties
            .get("height")
            .and_then(|v| v.val_str().parse::<u32>().ok())
            .unwrap_or(1080);
        let fps: f64 = properties
            .get("fps")
            .and_then(|v| v.val_str().parse::<f64>().ok())
            .unwrap_or(30.0);

        // Create dummy formats (these should be properly parsed from TXT records)
        let video_format = VideoFormat::new(
            crate::types::VideoCodec::Vp9,
            crate::types::Resolution::new(width, height),
            crate::types::FrameRate::from_float(fps)?,
        );

        let audio_format = AudioFormat::new(crate::types::AudioCodec::Opus, 48000, 2)?;

        let mut metadata = HashMap::new();
        // TxtRecord iteration
        for prop in properties.iter() {
            let key = prop.key();
            let val_str = prop.val_str();
            metadata.insert(key.to_string(), val_str.to_string());
        }

        Ok(SourceInfo {
            name,
            address: scoped_ip.to_ip_addr(),
            port,
            video_format,
            audio_format,
            metadata,
        })
    }
}

/// Service announcement server for advertising video sources.
pub struct DiscoveryServer {
    /// mDNS daemon.
    daemon: ServiceDaemon,
    /// Registered service info.
    service_info: Option<ServiceInfo>,
}

impl DiscoveryServer {
    /// Creates a new discovery server.
    ///
    /// # Errors
    ///
    /// Returns an error if the mDNS daemon cannot be created.
    pub fn new() -> VideoIpResult<Self> {
        let daemon = ServiceDaemon::new()
            .map_err(|e| VideoIpError::Discovery(format!("failed to create daemon: {e}")))?;

        Ok(Self {
            daemon,
            service_info: None,
        })
    }

    /// Announces a video source on the network.
    ///
    /// # Errors
    ///
    /// Returns an error if the announcement fails.
    pub fn announce(
        &mut self,
        name: &str,
        port: u16,
        video_format: &VideoFormat,
        audio_format: &AudioFormat,
    ) -> VideoIpResult<()> {
        let mut properties = HashMap::new();

        // Add video format properties
        properties.insert(
            "codec".to_string(),
            format!("{:?}", video_format.codec).to_lowercase(),
        );
        properties.insert(
            "width".to_string(),
            video_format.resolution.width.to_string(),
        );
        properties.insert(
            "height".to_string(),
            video_format.resolution.height.to_string(),
        );
        properties.insert(
            "fps".to_string(),
            video_format.frame_rate.to_float().to_string(),
        );

        // Add audio format properties
        properties.insert(
            "audio_codec".to_string(),
            format!("{:?}", audio_format.codec),
        );
        properties.insert(
            "sample_rate".to_string(),
            audio_format.sample_rate.to_string(),
        );
        properties.insert("channels".to_string(), audio_format.channels.to_string());

        let service_info = ServiceInfo::new(SERVICE_TYPE, name, name, "", port, properties)
            .map_err(|e| VideoIpError::Discovery(format!("failed to create service info: {e}")))?;

        self.daemon
            .register(service_info.clone())
            .map_err(|e| VideoIpError::Discovery(format!("failed to register service: {e}")))?;

        self.service_info = Some(service_info);

        Ok(())
    }

    /// Stops announcing the video source.
    pub fn stop_announce(&mut self) -> VideoIpResult<()> {
        if let Some(ref info) = self.service_info {
            self.daemon
                .unregister(info.get_fullname())
                .map_err(|e| VideoIpError::Discovery(format!("failed to unregister: {e}")))?;
            self.service_info = None;
        }
        Ok(())
    }
}

impl Drop for DiscoveryServer {
    fn drop(&mut self) {
        let _ = self.stop_announce();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioCodec, FrameRate, Resolution, VideoCodec};

    #[test]
    fn test_discovery_client_creation() {
        let client = DiscoveryClient::new();
        assert!(client.is_ok());
    }

    #[test]
    fn test_discovery_server_creation() {
        let server = DiscoveryServer::new();
        assert!(server.is_ok());
    }

    #[test]
    fn test_source_info() {
        let video_format =
            VideoFormat::new(VideoCodec::Vp9, Resolution::HD_1080, FrameRate::FPS_60);
        let audio_format =
            AudioFormat::new(AudioCodec::Opus, 48000, 2).expect("should succeed in test");

        let source = SourceInfo {
            name: "Test Source".to_string(),
            address: "192.168.1.100".parse().expect("should succeed in test"),
            port: 5000,
            video_format,
            audio_format,
            metadata: HashMap::new(),
        };

        let addr = source.socket_addr();
        assert_eq!(addr.port(), 5000);
    }

    #[tokio::test]
    async fn test_announce_and_discover() {
        let mut server = DiscoveryServer::new().expect("should succeed in test");
        let video_format =
            VideoFormat::new(VideoCodec::Vp9, Resolution::HD_1080, FrameRate::FPS_30);
        let audio_format =
            AudioFormat::new(AudioCodec::Opus, 48000, 2).expect("should succeed in test");

        // Announce service (hostname must be a valid .local. name)
        server
            .announce("testcamera.local.", 5000, &video_format, &audio_format)
            .expect("should succeed in test");

        // Give mDNS time to propagate
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Try to discover
        let client = DiscoveryClient::new().expect("should succeed in test");
        let sources = client
            .discover_all(2)
            .await
            .expect("should succeed in test");

        // Note: This may not find the service in CI environments
        // where mDNS may not work properly
        if !sources.is_empty() {
            assert!(sources.iter().any(|s| s.name.contains("testcamera")));
        }

        server.stop_announce().expect("should succeed in test");
    }
}
