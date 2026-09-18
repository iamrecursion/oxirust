use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// Recording session management
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a recording session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingStatus {
    /// Currently writing data.
    Recording,
    /// Paused (waiting for keyframe or manual pause).
    Paused,
    /// Finished successfully.
    Finished,
    /// Aborted due to an error.
    Failed,
}

/// A single recording session.
#[derive(Debug, Clone)]
pub struct RecordingSession {
    /// Unique session identifier.
    pub id: u64,
    /// Stream key being recorded.
    pub stream_key: String,
    /// Destination file path (or object-store key).
    pub destination: String,
    /// Current status.
    pub status: RecordingStatus,
    /// Bytes written so far.
    pub bytes_written: u64,
    /// Packet count.
    pub packet_count: u64,
    /// Wall-clock start time (seconds since UNIX epoch).
    pub started_at: u64,
    /// Wall-clock end time (seconds since UNIX epoch, 0 if still running).
    pub ended_at: u64,
    /// First PTS seen (milliseconds).
    pub first_pts: Option<u32>,
    /// Last PTS seen (milliseconds).
    pub last_pts: Option<u32>,
}

impl RecordingSession {
    /// Creates a new recording session.
    #[must_use]
    pub fn new(
        id: u64,
        stream_key: impl Into<String>,
        destination: impl Into<String>,
        started_at: u64,
    ) -> Self {
        Self {
            id,
            stream_key: stream_key.into(),
            destination: destination.into(),
            status: RecordingStatus::Recording,
            bytes_written: 0,
            packet_count: 0,
            started_at,
            ended_at: 0,
            first_pts: None,
            last_pts: None,
        }
    }

    /// Duration of the recording in milliseconds (based on PTS range).
    #[must_use]
    pub fn duration_ms(&self) -> Option<u32> {
        match (self.first_pts, self.last_pts) {
            (Some(first), Some(last)) => Some(last.wrapping_sub(first)),
            _ => None,
        }
    }

    /// Records a media packet and updates session statistics.
    pub fn ingest(&mut self, packet: &MediaPacket) {
        if self.status != RecordingStatus::Recording {
            return;
        }
        self.bytes_written += packet.data.len() as u64;
        self.packet_count += 1;
        if self.first_pts.is_none() {
            self.first_pts = Some(packet.timestamp);
        }
        self.last_pts = Some(packet.timestamp);
    }

    /// Finishes the session.
    pub fn finish(&mut self, ended_at: u64) {
        self.status = RecordingStatus::Finished;
        self.ended_at = ended_at;
    }

    /// Marks the session as failed.
    pub fn fail(&mut self, ended_at: u64) {
        self.status = RecordingStatus::Failed;
        self.ended_at = ended_at;
    }

    /// Pauses the session.
    pub fn pause(&mut self) {
        if self.status == RecordingStatus::Recording {
            self.status = RecordingStatus::Paused;
        }
    }

    /// Resumes the session.
    pub fn resume(&mut self) {
        if self.status == RecordingStatus::Paused {
            self.status = RecordingStatus::Recording;
        }
    }
}

/// Registry of all recording sessions.
pub struct RecordingRegistry {
    /// Active and completed sessions.
    sessions: Arc<RwLock<HashMap<u64, RecordingSession>>>,
    /// Next session ID.
    next_id: Arc<RwLock<u64>>,
}

impl RecordingRegistry {
    /// Creates a new registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Starts a new recording session.  Returns the session ID.
    pub async fn start_session(
        &self,
        stream_key: impl Into<String>,
        destination: impl Into<String>,
        now_secs: u64,
    ) -> u64 {
        let id = {
            let mut next = self.next_id.write().await;
            let id = *next;
            *next += 1;
            id
        };
        let session = RecordingSession::new(id, stream_key, destination, now_secs);
        let mut sessions = self.sessions.write().await;
        sessions.insert(id, session);
        id
    }

    /// Feeds a packet into a session.
    pub async fn ingest(&self, session_id: u64, packet: &MediaPacket) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.ingest(packet);
        }
    }

    /// Finishes a session.
    pub async fn finish_session(&self, session_id: u64, now_secs: u64) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.finish(now_secs);
        }
    }

    /// Marks a session as failed.
    pub async fn fail_session(&self, session_id: u64, now_secs: u64) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.fail(now_secs);
        }
    }

    /// Pauses a session.
    pub async fn pause_session(&self, session_id: u64) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.pause();
        }
    }

    /// Resumes a session.
    pub async fn resume_session(&self, session_id: u64) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(&session_id) {
            session.resume();
        }
    }

    /// Returns a snapshot of a session.
    pub async fn get_session(&self, session_id: u64) -> Option<RecordingSession> {
        let sessions = self.sessions.read().await;
        sessions.get(&session_id).cloned()
    }

    /// Returns all sessions for a given stream key.
    pub async fn sessions_for_stream(&self, stream_key: &str) -> Vec<RecordingSession> {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|s| s.stream_key == stream_key)
            .cloned()
            .collect()
    }

    /// Returns the count of active (Recording or Paused) sessions.
    pub async fn active_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions
            .values()
            .filter(|s| {
                s.status == RecordingStatus::Recording || s.status == RecordingStatus::Paused
            })
            .count()
    }

    /// Removes all finished and failed sessions from memory.
    pub async fn prune_completed(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.retain(|_, s| {
            s.status == RecordingStatus::Recording || s.status == RecordingStatus::Paused
        });
    }
}

impl Default for RecordingRegistry {
    fn default() -> Self {
        Self::new()
    }
}
