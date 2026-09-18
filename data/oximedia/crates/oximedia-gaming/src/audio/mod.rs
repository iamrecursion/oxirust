//! Multi-source audio capture and mixing.

pub mod game;
pub mod mic;
pub mod mix;
pub mod music;

pub use game::{AudioDevice, GameAudioCapture};
pub use mic::{MicConfig, MicrophoneCapture};
pub use mix::{AudioMixer, AudioSource, MixerConfig};
pub use music::{MusicPlayer, MusicTrack};
