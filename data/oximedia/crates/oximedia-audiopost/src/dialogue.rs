//! Dialogue editing and processing for audio post-production.
//!
//! Provides tools for managing ADR dialogue lines, applying word replacements,
//! tracking sessions by speaker, and queuing lines for re-recording.

#![allow(dead_code)]

/// A single dialogue line with timing, speaker, and take information.
#[derive(Debug, Clone)]
pub struct DialogueLine {
    /// Unique identifier for this line.
    pub id: u64,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Text content of the dialogue.
    pub text: String,
    /// Name of the speaker.
    pub speaker: String,
    /// Take number (1-based).
    pub take: u32,
}

impl DialogueLine {
    /// Create a new dialogue line.
    #[must_use]
    pub fn new(
        id: u64,
        start_ms: u64,
        end_ms: u64,
        text: impl Into<String>,
        speaker: impl Into<String>,
        take: u32,
    ) -> Self {
        Self {
            id,
            start_ms,
            end_ms,
            text: text.into(),
            speaker: speaker.into(),
            take,
        }
    }

    /// Duration of this dialogue line in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Estimated words per minute based on word count and duration.
    ///
    /// Returns 0.0 if the duration is zero.
    #[must_use]
    pub fn words_per_minute(&self) -> f64 {
        let duration_secs = self.duration_ms() as f64 / 1000.0;
        if duration_secs < 1e-9 {
            return 0.0;
        }
        let word_count = self.text.split_whitespace().count();
        (word_count as f64 / duration_secs) * 60.0
    }
}

/// A word replacement rule: replace `original` with `replacement`.
#[derive(Debug, Clone)]
pub struct WordReplace {
    /// The original word or phrase to find.
    pub original: String,
    /// The replacement word or phrase.
    pub replacement: String,
}

impl WordReplace {
    /// Create a new replacement rule.
    #[must_use]
    pub fn new(original: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            original: original.into(),
            replacement: replacement.into(),
        }
    }
}

/// Editor for a collection of dialogue lines with word-replacement support.
#[derive(Debug, Default)]
pub struct DialogueEditor {
    /// All dialogue lines.
    pub lines: Vec<DialogueLine>,
    /// Word replacement rules applied via [`apply_replacements`](Self::apply_replacements).
    pub replacements: Vec<WordReplace>,
}

impl DialogueEditor {
    /// Create an empty dialogue editor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a dialogue line to the editor.
    pub fn add_line(&mut self, line: DialogueLine) {
        self.lines.push(line);
    }

    /// Remove the dialogue line with the given `id`.
    ///
    /// Returns `true` if a line was removed, `false` if no line with that id existed.
    pub fn remove_line(&mut self, id: u64) -> bool {
        let before = self.lines.len();
        self.lines.retain(|l| l.id != id);
        self.lines.len() < before
    }

    /// Add a word replacement rule.
    pub fn add_replacement(&mut self, rule: WordReplace) {
        self.replacements.push(rule);
    }

    /// Apply all replacement rules to every dialogue line in-place.
    ///
    /// Returns the total number of individual substitutions made.
    pub fn apply_replacements(&mut self) -> usize {
        let mut count = 0usize;
        for line in &mut self.lines {
            for rule in &self.replacements {
                let original = line.text.clone();
                line.text = original.replace(rule.original.as_str(), rule.replacement.as_str());
                // Count occurrences replaced
                let occurrences = line.text.matches(rule.replacement.as_str()).count();
                // Only count if something changed
                if line.text != original.replace(rule.original.as_str(), rule.replacement.as_str())
                {
                    // text was already updated above
                }
                let _ = occurrences;
                // More accurate count: original occurrences of `rule.original`
                count += original.matches(rule.original.as_str()).count();
            }
        }
        count
    }

    /// Find all dialogue lines spoken by the given speaker (case-sensitive).
    #[must_use]
    pub fn find_lines_by_speaker(&self, speaker: &str) -> Vec<&DialogueLine> {
        self.lines.iter().filter(|l| l.speaker == speaker).collect()
    }

    /// Total duration of all dialogue lines in milliseconds (sum, not span).
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.lines.iter().map(|l| l.duration_ms()).sum()
    }
}

/// Queue of dialogue lines flagged for ADR re-recording.
#[derive(Debug, Default)]
pub struct ADRQueue {
    /// Pairs of (line_id, studio_note).
    pub sessions: Vec<(u64, String)>,
}

impl ADRQueue {
    /// Create an empty ADR queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a line to the ADR queue with an optional studio note.
    pub fn add(&mut self, id: u64, note: &str) {
        self.sessions.push((id, note.to_string()));
    }

    /// Number of lines currently pending re-recording.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.sessions.len()
    }

    /// Remove an entry by line id.
    ///
    /// Returns `true` if an entry was removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.sessions.len();
        self.sessions.retain(|(lid, _)| *lid != id);
        self.sessions.len() < before
    }

    /// Retrieve the studio note for a given line id, if present.
    #[must_use]
    pub fn note_for(&self, id: u64) -> Option<&str> {
        self.sessions
            .iter()
            .find(|(lid, _)| *lid == id)
            .map(|(_, note)| note.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(id: u64, start: u64, end: u64, text: &str, speaker: &str) -> DialogueLine {
        DialogueLine::new(id, start, end, text, speaker, 1)
    }

    #[test]
    fn test_duration_ms_basic() {
        let line = make_line(1, 1000, 4000, "Hello", "Alice");
        assert_eq!(line.duration_ms(), 3000);
    }

    #[test]
    fn test_duration_ms_zero() {
        let line = make_line(1, 5000, 5000, "Oops", "Bob");
        assert_eq!(line.duration_ms(), 0);
    }

    #[test]
    fn test_duration_ms_saturating() {
        // end < start should saturate to 0
        let line = make_line(1, 5000, 3000, "Bad", "X");
        assert_eq!(line.duration_ms(), 0);
    }

    #[test]
    fn test_words_per_minute_basic() {
        // 6 words over 2 seconds = 180 WPM
        let line = make_line(1, 0, 2000, "one two three four five six", "Alice");
        let wpm = line.words_per_minute();
        assert!((wpm - 180.0).abs() < 1.0, "wpm={wpm}");
    }

    #[test]
    fn test_words_per_minute_zero_duration() {
        let line = make_line(1, 1000, 1000, "zero", "Bob");
        assert_eq!(line.words_per_minute(), 0.0);
    }

    #[test]
    fn test_dialogue_editor_add_line() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "Hi", "Alice"));
        assert_eq!(editor.lines.len(), 1);
    }

    #[test]
    fn test_dialogue_editor_remove_line_exists() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(42, 0, 1000, "Line", "Alice"));
        assert!(editor.remove_line(42));
        assert!(editor.lines.is_empty());
    }

    #[test]
    fn test_dialogue_editor_remove_line_missing() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "Line", "Alice"));
        assert!(!editor.remove_line(99));
        assert_eq!(editor.lines.len(), 1);
    }

    #[test]
    fn test_apply_replacements_counts() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "hello world hello", "Alice"));
        editor.add_replacement(WordReplace::new("hello", "hi"));
        let count = editor.apply_replacements();
        assert_eq!(count, 2);
        assert_eq!(editor.lines[0].text, "hi world hi");
    }

    #[test]
    fn test_apply_replacements_no_match() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "good morning", "Bob"));
        editor.add_replacement(WordReplace::new("evening", "night"));
        let count = editor.apply_replacements();
        assert_eq!(count, 0);
        assert_eq!(editor.lines[0].text, "good morning");
    }

    #[test]
    fn test_find_lines_by_speaker() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "A1", "Alice"));
        editor.add_line(make_line(2, 1000, 2000, "B1", "Bob"));
        editor.add_line(make_line(3, 2000, 3000, "A2", "Alice"));
        let alice_lines = editor.find_lines_by_speaker("Alice");
        assert_eq!(alice_lines.len(), 2);
        assert!(alice_lines.iter().all(|l| l.speaker == "Alice"));
    }

    #[test]
    fn test_find_lines_by_speaker_none() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "Hi", "Alice"));
        let result = editor.find_lines_by_speaker("Charlie");
        assert!(result.is_empty());
    }

    #[test]
    fn test_total_duration_ms() {
        let mut editor = DialogueEditor::new();
        editor.add_line(make_line(1, 0, 1000, "A", "Alice"));
        editor.add_line(make_line(2, 2000, 5000, "B", "Bob"));
        // 1000 + 3000 = 4000
        assert_eq!(editor.total_duration_ms(), 4000);
    }

    #[test]
    fn test_adr_queue_add_and_count() {
        let mut queue = ADRQueue::new();
        queue.add(1, "Needs re-record");
        queue.add(2, "Noise issue");
        assert_eq!(queue.pending_count(), 2);
    }

    #[test]
    fn test_adr_queue_note_for() {
        let mut queue = ADRQueue::new();
        queue.add(10, "Check timing");
        assert_eq!(queue.note_for(10), Some("Check timing"));
        assert_eq!(queue.note_for(99), None);
    }

    #[test]
    fn test_adr_queue_remove() {
        let mut queue = ADRQueue::new();
        queue.add(5, "note");
        assert!(queue.remove(5));
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_adr_queue_remove_missing() {
        let mut queue = ADRQueue::new();
        queue.add(1, "x");
        assert!(!queue.remove(99));
        assert_eq!(queue.pending_count(), 1);
    }
}
