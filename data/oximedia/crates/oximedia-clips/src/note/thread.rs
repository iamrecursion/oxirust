//! Threaded note/comment system.

use super::{Note, NoteId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a note thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ThreadId(Uuid);

impl ThreadId {
    /// Creates a new random thread ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Creates a thread ID from a UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Returns the inner UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for ThreadId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ThreadId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A threaded conversation of notes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteThread {
    /// Thread ID.
    pub id: ThreadId,

    /// Root note.
    pub root: Note,

    /// Replies organized by parent note ID.
    replies: HashMap<NoteId, Vec<Note>>,
}

impl NoteThread {
    /// Creates a new thread with a root note.
    #[must_use]
    pub fn new(root: Note) -> Self {
        Self {
            id: ThreadId::new(),
            root,
            replies: HashMap::new(),
        }
    }

    /// Adds a reply to the thread.
    pub fn add_reply(&mut self, reply: Note) {
        let parent_id = reply.reply_to.unwrap_or(self.root.id);
        self.replies.entry(parent_id).or_default().push(reply);
    }

    /// Gets direct replies to a note.
    #[must_use]
    pub fn get_replies(&self, note_id: &NoteId) -> Vec<&Note> {
        self.replies
            .get(note_id)
            .map_or_else(Vec::new, |replies| replies.iter().collect())
    }

    /// Gets all notes in the thread.
    #[must_use]
    pub fn all_notes(&self) -> Vec<&Note> {
        let mut notes = vec![&self.root];
        for replies in self.replies.values() {
            notes.extend(replies.iter());
        }
        notes
    }

    /// Gets the total number of notes in the thread.
    #[must_use]
    pub fn total_notes(&self) -> usize {
        1 + self.replies.values().map(Vec::len).sum::<usize>()
    }

    /// Gets the depth of the thread (maximum nesting level).
    #[must_use]
    pub fn depth(&self) -> usize {
        self.calculate_depth(&self.root.id, 0)
    }

    fn calculate_depth(&self, note_id: &NoteId, current_depth: usize) -> usize {
        let replies = self.get_replies(note_id);
        if replies.is_empty() {
            return current_depth;
        }

        replies
            .iter()
            .map(|reply| self.calculate_depth(&reply.id, current_depth + 1))
            .max()
            .unwrap_or(current_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_creation() {
        let root = Note::new("Root note");
        let thread = NoteThread::new(root.clone());
        assert_eq!(thread.total_notes(), 1);
        assert_eq!(thread.root.id, root.id);
    }

    #[test]
    fn test_thread_replies() {
        let root = Note::new("Root note");
        let root_id = root.id;
        let mut thread = NoteThread::new(root);

        let reply1 = Note::reply_to("Reply 1", root_id);
        let reply2 = Note::reply_to("Reply 2", root_id);
        thread.add_reply(reply1);
        thread.add_reply(reply2);

        assert_eq!(thread.total_notes(), 3);
        assert_eq!(thread.get_replies(&root_id).len(), 2);
    }

    #[test]
    fn test_thread_depth() {
        let root = Note::new("Root");
        let root_id = root.id;
        let mut thread = NoteThread::new(root);

        let reply1 = Note::reply_to("Reply 1", root_id);
        let reply1_id = reply1.id;
        thread.add_reply(reply1);

        let reply2 = Note::reply_to("Reply 2", reply1_id);
        thread.add_reply(reply2);

        assert_eq!(thread.depth(), 2);
    }
}
