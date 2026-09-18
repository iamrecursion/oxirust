//! Platform integration metadata.

pub mod facebook;
pub mod twitch;
pub mod youtube;

pub use facebook::{FacebookConfig, FacebookIntegration};
pub use twitch::{
    StreamHealthMonitor, TwitchChatMessage, TwitchChatParser, TwitchConfig, TwitchEventSub,
    TwitchEventType, TwitchIntegration, TwitchStreamInfo,
};
pub use youtube::{
    PrivacyStatus, VideoQuality, YouTubeConfig, YouTubeIntegration, YoutubeAdaptiveBitrate,
    YoutubeChatFilter, YoutubeLiveChat, YoutubeStreamConfig, YoutubeStreamStats,
};
