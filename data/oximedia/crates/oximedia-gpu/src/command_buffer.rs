//! GPU command buffer recording and submission.
#![allow(dead_code)]

use std::collections::VecDeque;

/// Type of a GPU command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandType {
    /// Draw call
    Draw,
    /// Compute dispatch
    Compute,
    /// Copy / transfer
    Copy,
    /// Resource barrier / transition
    Barrier,
    /// Clear a render target
    Clear,
    /// Begin / end render pass markers
    RenderPassMarker,
}

impl CommandType {
    /// Returns `true` if the command represents a draw call.
    #[must_use]
    pub fn is_draw(&self) -> bool {
        matches!(self, Self::Draw)
    }

    /// Returns `true` if the command is a compute dispatch.
    #[must_use]
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::Compute)
    }

    /// Returns `true` if the command transfers data between buffers/textures.
    #[must_use]
    pub fn is_copy(&self) -> bool {
        matches!(self, Self::Copy)
    }
}

/// A single recorded GPU command with metadata.
#[derive(Debug, Clone)]
pub struct CommandEntry {
    /// Type of the command.
    pub command_type: CommandType,
    /// Opaque payload (e.g., serialised draw parameters).
    pub payload: Vec<u8>,
    /// Human-readable label for debugging.
    pub label: String,
}

impl CommandEntry {
    /// Create a new command entry.
    #[must_use]
    pub fn new(command_type: CommandType, label: impl Into<String>) -> Self {
        Self {
            command_type,
            payload: Vec::new(),
            label: label.into(),
        }
    }

    /// Create a new entry with a raw payload.
    #[must_use]
    pub fn with_payload(
        command_type: CommandType,
        label: impl Into<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            command_type,
            payload,
            label: label.into(),
        }
    }

    /// Estimate the GPU cost (in arbitrary units) of executing this command.
    ///
    /// Draw calls are assumed to be more expensive than compute dispatches,
    /// which in turn are more expensive than copies.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_cost(&self) -> f32 {
        let base: f32 = match self.command_type {
            CommandType::Draw => 10.0,
            CommandType::Compute => 8.0,
            CommandType::Copy => 3.0,
            CommandType::Barrier => 1.0,
            CommandType::Clear => 2.0,
            CommandType::RenderPassMarker => 0.1,
        };
        // Payload size adds a small overhead proportional to data moved.
        base + self.payload.len() as f32 * 0.001
    }
}

/// State of a [`CommandBuffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferState {
    /// Ready to record commands.
    Recording,
    /// Recording finished; ready to submit.
    Executable,
    /// Submitted to GPU queue; cannot be re-used until reset.
    Pending,
    /// Buffer has been reset and can start recording again.
    Reset,
}

/// A GPU command buffer that records and submits work to the GPU.
pub struct CommandBuffer {
    commands: VecDeque<CommandEntry>,
    state: CommandBufferState,
    label: String,
}

impl CommandBuffer {
    /// Create a new, empty command buffer in the `Recording` state.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            commands: VecDeque::new(),
            state: CommandBufferState::Recording,
            label: label.into(),
        }
    }

    /// Record a new command into the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not in the `Recording` state.
    pub fn record(&mut self, entry: CommandEntry) {
        assert_eq!(
            self.state,
            CommandBufferState::Recording,
            "CommandBuffer '{}' must be in Recording state to accept new commands",
            self.label
        );
        self.commands.push_back(entry);
    }

    /// Finish recording and transition the buffer to `Executable`.
    ///
    /// Returns `false` if the buffer was not in `Recording` state.
    pub fn finish(&mut self) -> bool {
        if self.state == CommandBufferState::Recording {
            self.state = CommandBufferState::Executable;
            true
        } else {
            false
        }
    }

    /// Simulate submission to the GPU queue.
    ///
    /// Returns the list of submitted commands (for testing / inspection) and
    /// transitions the buffer to `Pending`.
    ///
    /// Returns `None` if the buffer is not `Executable`.
    pub fn submit(&mut self) -> Option<Vec<CommandEntry>> {
        if self.state != CommandBufferState::Executable {
            return None;
        }
        self.state = CommandBufferState::Pending;
        Some(self.commands.iter().cloned().collect())
    }

    /// Reset the buffer, clearing all recorded commands.
    pub fn reset(&mut self) {
        self.commands.clear();
        self.state = CommandBufferState::Reset;
    }

    /// Begin a fresh recording pass after a reset.
    ///
    /// Returns `false` if the buffer was not in `Reset` state.
    pub fn begin(&mut self) -> bool {
        if self.state == CommandBufferState::Reset {
            self.state = CommandBufferState::Recording;
            true
        } else {
            false
        }
    }

    /// Number of recorded commands.
    #[must_use]
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Current state of the buffer.
    #[must_use]
    pub fn state(&self) -> CommandBufferState {
        self.state
    }

    /// Label of this buffer.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Total estimated GPU cost of all recorded commands.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_estimated_cost(&self) -> f32 {
        self.commands.iter().map(CommandEntry::estimated_cost).sum()
    }
}

// ---------------------------------------------------------------------------
// Double-buffered command submission
// ---------------------------------------------------------------------------

/// Identifies one of the two command-buffer slots used in double buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSlot {
    /// First slot (index 0).
    A,
    /// Second slot (index 1).
    B,
}

impl BufferSlot {
    /// Return the other slot.
    #[must_use]
    pub fn flip(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }

    /// Numeric index of this slot.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

/// Lifecycle state of a slot in [`DoubleBufferedSubmitter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// No work has been recorded; slot is free.
    Idle,
    /// CPU is currently recording commands into this slot.
    Recording,
    /// Recorded and ready to send to the GPU queue.
    ReadyToSubmit,
    /// Submitted to the GPU queue; GPU may still be executing.
    Inflight,
    /// GPU execution finished; results are available.
    Retired,
}

/// Double-buffered GPU command submission.
///
/// Maintains two independent [`CommandBuffer`] slots (A and B).  While the GPU
/// executes the commands in slot B, the CPU records new work into slot A, and
/// vice-versa.  This overlaps CPU recording with GPU execution to maximise
/// throughput.
///
/// # Typical cycle (one frame)
///
/// ```text
/// 1. begin_record(recording_slot)   → SlotState::Recording
/// 2. … record commands …
/// 3. finish_record(recording_slot)  → SlotState::ReadyToSubmit
/// 4. submit(recording_slot)         → SlotState::Inflight
/// 5. mark_retired(other_slot)       → SlotState::Retired (after GPU fence)
/// 6. reset(retired_slot)            → SlotState::Idle
/// ```
pub struct DoubleBufferedSubmitter {
    slots: [CommandBuffer; 2],
    slot_states: [SlotState; 2],
    /// The slot currently being recorded by the CPU.
    active_slot: BufferSlot,
    /// Number of full A→B→A cycles completed.
    frame_count: u64,
}

impl DoubleBufferedSubmitter {
    /// Create a new double-buffered submitter with both slots idle.
    #[must_use]
    pub fn new() -> Self {
        let mut slot_a = CommandBuffer::new("DoubleBuffer-SlotA");
        let mut slot_b = CommandBuffer::new("DoubleBuffer-SlotB");
        // Reset both so they start in `Reset` state and can `begin()`.
        slot_a.reset();
        slot_b.reset();
        Self {
            slots: [slot_a, slot_b],
            slot_states: [SlotState::Idle, SlotState::Idle],
            active_slot: BufferSlot::A,
            frame_count: 0,
        }
    }

    /// Current state of `slot`.
    #[must_use]
    pub fn state(&self, slot: BufferSlot) -> SlotState {
        self.slot_states[slot.index()]
    }

    /// The slot the CPU is currently (or will next) record into.
    #[must_use]
    pub fn active_slot(&self) -> BufferSlot {
        self.active_slot
    }

    /// Total number of complete double-buffer cycles (frames) submitted.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Begin recording commands into `slot`.
    ///
    /// Returns `false` if `slot` is not currently `Idle`.
    pub fn begin_record(&mut self, slot: BufferSlot) -> bool {
        if self.slot_states[slot.index()] != SlotState::Idle {
            return false;
        }
        self.slots[slot.index()].begin();
        self.slot_states[slot.index()] = SlotState::Recording;
        self.active_slot = slot;
        true
    }

    /// Record a command entry into the active recording slot.
    ///
    /// Returns `false` if `slot` is not in `Recording` state.
    pub fn record(&mut self, slot: BufferSlot, entry: CommandEntry) -> bool {
        if self.slot_states[slot.index()] != SlotState::Recording {
            return false;
        }
        self.slots[slot.index()].record(entry);
        true
    }

    /// Finish recording for `slot`.
    ///
    /// Returns `false` if `slot` is not in `Recording` state.
    pub fn finish_record(&mut self, slot: BufferSlot) -> bool {
        if self.slot_states[slot.index()] != SlotState::Recording {
            return false;
        }
        let ok = self.slots[slot.index()].finish();
        if ok {
            self.slot_states[slot.index()] = SlotState::ReadyToSubmit;
        }
        ok
    }

    /// Submit the commands in `slot` to the (simulated) GPU queue.
    ///
    /// Returns the submitted commands on success, or `None` if `slot` is not
    /// `ReadyToSubmit`.
    pub fn submit(&mut self, slot: BufferSlot) -> Option<Vec<CommandEntry>> {
        if self.slot_states[slot.index()] != SlotState::ReadyToSubmit {
            return None;
        }
        let cmds = self.slots[slot.index()].submit()?;
        self.slot_states[slot.index()] = SlotState::Inflight;
        self.frame_count += 1;
        Some(cmds)
    }

    /// Mark `slot` as retired (GPU execution complete).
    ///
    /// Returns `false` if `slot` is not `Inflight`.
    pub fn mark_retired(&mut self, slot: BufferSlot) -> bool {
        if self.slot_states[slot.index()] != SlotState::Inflight {
            return false;
        }
        self.slot_states[slot.index()] = SlotState::Retired;
        true
    }

    /// Reset `slot` to `Idle` so it can be recorded into again.
    ///
    /// Returns `false` if `slot` is not `Retired`.
    pub fn reset_slot(&mut self, slot: BufferSlot) -> bool {
        if self.slot_states[slot.index()] != SlotState::Retired {
            return false;
        }
        self.slots[slot.index()].reset();
        self.slot_states[slot.index()] = SlotState::Idle;
        true
    }

    /// Number of commands recorded in `slot`.
    #[must_use]
    pub fn command_count(&self, slot: BufferSlot) -> usize {
        self.slots[slot.index()].command_count()
    }
}

impl Default for DoubleBufferedSubmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_draw() -> CommandEntry {
        CommandEntry::new(CommandType::Draw, "draw_quad")
    }

    fn make_compute() -> CommandEntry {
        CommandEntry::new(CommandType::Compute, "dispatch_cs")
    }

    fn make_copy() -> CommandEntry {
        CommandEntry::new(CommandType::Copy, "copy_buffer")
    }

    // --- CommandType tests ---

    #[test]
    fn test_is_draw_true() {
        assert!(CommandType::Draw.is_draw());
    }

    #[test]
    fn test_is_draw_false_for_compute() {
        assert!(!CommandType::Compute.is_draw());
    }

    #[test]
    fn test_is_compute_true() {
        assert!(CommandType::Compute.is_compute());
    }

    #[test]
    fn test_is_copy_true() {
        assert!(CommandType::Copy.is_copy());
    }

    #[test]
    fn test_is_copy_false_for_barrier() {
        assert!(!CommandType::Barrier.is_copy());
    }

    // --- CommandEntry tests ---

    #[test]
    fn test_entry_estimated_cost_draw_greater_than_copy() {
        let draw = make_draw();
        let copy = make_copy();
        assert!(draw.estimated_cost() > copy.estimated_cost());
    }

    #[test]
    fn test_entry_estimated_cost_payload_increases_cost() {
        let small = CommandEntry::with_payload(CommandType::Copy, "s", vec![0u8; 10]);
        let large = CommandEntry::with_payload(CommandType::Copy, "l", vec![0u8; 1000]);
        assert!(large.estimated_cost() > small.estimated_cost());
    }

    #[test]
    fn test_entry_label_stored() {
        let e = CommandEntry::new(CommandType::Draw, "my_draw");
        assert_eq!(e.label, "my_draw");
    }

    // --- CommandBuffer tests ---

    #[test]
    fn test_new_buffer_is_recording() {
        let buf = CommandBuffer::new("test");
        assert_eq!(buf.state(), CommandBufferState::Recording);
    }

    #[test]
    fn test_record_increments_count() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        buf.record(make_compute());
        assert_eq!(buf.command_count(), 2);
    }

    #[test]
    fn test_finish_transitions_to_executable() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        assert!(buf.finish());
        assert_eq!(buf.state(), CommandBufferState::Executable);
    }

    #[test]
    fn test_submit_returns_commands() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        buf.record(make_copy());
        buf.finish();
        let cmds = buf.submit().expect("submit should succeed");
        assert_eq!(cmds.len(), 2);
    }

    #[test]
    fn test_submit_transitions_to_pending() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_compute());
        buf.finish();
        buf.submit();
        assert_eq!(buf.state(), CommandBufferState::Pending);
    }

    #[test]
    fn test_reset_clears_commands_and_sets_reset_state() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        buf.finish();
        buf.submit();
        buf.reset();
        assert_eq!(buf.command_count(), 0);
        assert_eq!(buf.state(), CommandBufferState::Reset);
    }

    #[test]
    fn test_begin_after_reset_allows_recording() {
        let mut buf = CommandBuffer::new("test");
        buf.reset();
        assert!(buf.begin());
        buf.record(make_draw());
        assert_eq!(buf.command_count(), 1);
    }

    #[test]
    fn test_total_estimated_cost_sums_entries() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        buf.record(make_copy());
        let expected = make_draw().estimated_cost() + make_copy().estimated_cost();
        assert!((buf.total_estimated_cost() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_label_stored() {
        let buf = CommandBuffer::new("my_buf");
        assert_eq!(buf.label(), "my_buf");
    }

    #[test]
    fn test_submit_fails_when_not_executable() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        // Not finished yet — still Recording
        assert!(buf.submit().is_none());
    }

    #[test]
    fn test_finish_fails_when_already_executable() {
        let mut buf = CommandBuffer::new("test");
        buf.record(make_draw());
        buf.finish();
        // Calling finish again should return false
        assert!(!buf.finish());
    }

    // --- BufferSlot tests ---

    #[test]
    fn test_buffer_slot_flip() {
        assert_eq!(BufferSlot::A.flip(), BufferSlot::B);
        assert_eq!(BufferSlot::B.flip(), BufferSlot::A);
    }

    #[test]
    fn test_buffer_slot_index() {
        assert_eq!(BufferSlot::A.index(), 0);
        assert_eq!(BufferSlot::B.index(), 1);
    }

    // --- DoubleBufferedSubmitter tests ---

    #[test]
    fn test_double_buffer_initial_state() {
        let db = DoubleBufferedSubmitter::new();
        assert_eq!(db.state(BufferSlot::A), SlotState::Idle);
        assert_eq!(db.state(BufferSlot::B), SlotState::Idle);
        assert_eq!(db.frame_count(), 0);
    }

    #[test]
    fn test_double_buffer_begin_record() {
        let mut db = DoubleBufferedSubmitter::new();
        assert!(db.begin_record(BufferSlot::A));
        assert_eq!(db.state(BufferSlot::A), SlotState::Recording);
    }

    #[test]
    fn test_double_buffer_begin_record_fails_when_not_idle() {
        let mut db = DoubleBufferedSubmitter::new();
        db.begin_record(BufferSlot::A);
        // Already Recording — cannot begin again
        assert!(!db.begin_record(BufferSlot::A));
    }

    #[test]
    fn test_double_buffer_full_cycle_slot_a() {
        let mut db = DoubleBufferedSubmitter::new();
        assert!(db.begin_record(BufferSlot::A));
        db.record(BufferSlot::A, make_draw());
        assert!(db.finish_record(BufferSlot::A));
        assert_eq!(db.state(BufferSlot::A), SlotState::ReadyToSubmit);
        let cmds = db.submit(BufferSlot::A).expect("submit should succeed");
        assert_eq!(cmds.len(), 1);
        assert_eq!(db.state(BufferSlot::A), SlotState::Inflight);
        assert_eq!(db.frame_count(), 1);
        assert!(db.mark_retired(BufferSlot::A));
        assert_eq!(db.state(BufferSlot::A), SlotState::Retired);
        assert!(db.reset_slot(BufferSlot::A));
        assert_eq!(db.state(BufferSlot::A), SlotState::Idle);
    }

    #[test]
    fn test_double_buffer_interleaved_slots() {
        let mut db = DoubleBufferedSubmitter::new();
        // Record slot A
        db.begin_record(BufferSlot::A);
        db.record(BufferSlot::A, make_compute());
        db.finish_record(BufferSlot::A);
        // While A is ready-to-submit, record slot B
        db.begin_record(BufferSlot::B);
        db.record(BufferSlot::B, make_copy());
        db.finish_record(BufferSlot::B);
        // Submit both
        assert!(db.submit(BufferSlot::A).is_some());
        assert!(db.submit(BufferSlot::B).is_some());
        assert_eq!(db.frame_count(), 2);
    }

    #[test]
    fn test_double_buffer_submit_fails_when_not_ready() {
        let mut db = DoubleBufferedSubmitter::new();
        db.begin_record(BufferSlot::A);
        // Not finished yet — submit should fail
        assert!(db.submit(BufferSlot::A).is_none());
    }

    #[test]
    fn test_double_buffer_mark_retired_fails_when_not_inflight() {
        let mut db = DoubleBufferedSubmitter::new();
        assert!(!db.mark_retired(BufferSlot::A));
    }

    #[test]
    fn test_double_buffer_reset_slot_fails_when_not_retired() {
        let mut db = DoubleBufferedSubmitter::new();
        assert!(!db.reset_slot(BufferSlot::A));
    }

    #[test]
    fn test_double_buffer_command_count() {
        let mut db = DoubleBufferedSubmitter::new();
        db.begin_record(BufferSlot::B);
        db.record(BufferSlot::B, make_draw());
        db.record(BufferSlot::B, make_draw());
        assert_eq!(db.command_count(BufferSlot::B), 2);
    }
}
