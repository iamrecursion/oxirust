#![allow(dead_code)]
//! Work item scheduling for hardware acceleration dispatch.

/// Kind of work an acceleration unit can perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkItemKind {
    /// General-purpose compute shader dispatch.
    Compute,
    /// Image scaling operation.
    Scale,
    /// Color space conversion.
    ColorConvert,
    /// Motion estimation.
    MotionEstimate,
    /// Data transfer between host and device.
    Transfer,
}

impl WorkItemKind {
    /// Returns `true` if this kind represents a compute operation
    /// (as opposed to a pure data transfer).
    #[must_use]
    pub fn is_compute(&self) -> bool {
        !matches!(self, Self::Transfer)
    }

    /// Returns a human-readable label for the kind.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Scale => "scale",
            Self::ColorConvert => "color_convert",
            Self::MotionEstimate => "motion_estimate",
            Self::Transfer => "transfer",
        }
    }
}

/// A single unit of work submitted to the accelerator.
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique identifier for this work item.
    pub id: u64,
    /// Kind of work to perform.
    pub kind: WorkItemKind,
    /// Width of the data being processed (pixels or elements).
    pub width: u32,
    /// Height of the data being processed.
    pub height: u32,
    /// Optional priority (higher = more urgent).
    pub priority: u8,
}

impl WorkItem {
    /// Creates a new `WorkItem`.
    #[must_use]
    pub fn new(id: u64, kind: WorkItemKind, width: u32, height: u32) -> Self {
        Self {
            id,
            kind,
            width,
            height,
            priority: 0,
        }
    }

    /// Creates a new `WorkItem` with an explicit priority.
    #[must_use]
    pub fn with_priority(
        id: u64,
        kind: WorkItemKind,
        width: u32,
        height: u32,
        priority: u8,
    ) -> Self {
        Self {
            id,
            kind,
            width,
            height,
            priority,
        }
    }

    /// Estimates the number of cycles required to process this work item.
    ///
    /// The model is simplistic but useful for relative scheduling decisions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_cycles(&self) -> u64 {
        let pixels = u64::from(self.width) * u64::from(self.height);
        let cost_per_pixel: u64 = match self.kind {
            WorkItemKind::Compute => 8,
            WorkItemKind::Scale => 12,
            WorkItemKind::ColorConvert => 6,
            WorkItemKind::MotionEstimate => 32,
            WorkItemKind::Transfer => 2,
        };
        pixels * cost_per_pixel
    }
}

/// A batch of work items submitted together.
#[derive(Debug, Default)]
pub struct WorkItemBatch {
    items: Vec<WorkItem>,
}

impl WorkItemBatch {
    /// Creates an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a work item to the batch.
    pub fn add(&mut self, item: WorkItem) {
        self.items.push(item);
    }

    /// Returns the total estimated cycles for all items in the batch.
    #[must_use]
    pub fn total_cycles(&self) -> u64 {
        self.items.iter().map(WorkItem::estimated_cycles).sum()
    }

    /// Returns the number of items in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the batch contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns an iterator over the work items.
    pub fn iter(&self) -> impl Iterator<Item = &WorkItem> {
        self.items.iter()
    }
}

/// Scheduler that orders work items for optimal throughput.
#[derive(Debug, Default)]
pub struct WorkItemScheduler {
    queue: Vec<WorkItem>,
}

impl WorkItemScheduler {
    /// Creates a new scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedules a batch of work items, ordering by priority (descending)
    /// and then by estimated cycles (ascending, shortest job first).
    pub fn schedule(&mut self, batch: WorkItemBatch) -> Vec<WorkItem> {
        let mut items: Vec<WorkItem> = batch.items;
        // Highest priority first; break ties by shortest estimated cycles.
        items.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then_with(|| a.estimated_cycles().cmp(&b.estimated_cycles()))
        });
        self.queue.extend(items.iter().cloned());
        items
    }

    /// Returns the total number of items that have passed through the scheduler.
    #[must_use]
    pub fn total_scheduled(&self) -> usize {
        self.queue.len()
    }

    /// Clears the internal history queue.
    pub fn reset(&mut self) {
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: u64, kind: WorkItemKind, w: u32, h: u32) -> WorkItem {
        WorkItem::new(id, kind, w, h)
    }

    #[test]
    fn test_kind_is_compute_true() {
        assert!(WorkItemKind::Compute.is_compute());
    }

    #[test]
    fn test_kind_is_compute_scale() {
        assert!(WorkItemKind::Scale.is_compute());
    }

    #[test]
    fn test_kind_is_compute_transfer_false() {
        assert!(!WorkItemKind::Transfer.is_compute());
    }

    #[test]
    fn test_kind_label() {
        assert_eq!(WorkItemKind::ColorConvert.label(), "color_convert");
        assert_eq!(WorkItemKind::MotionEstimate.label(), "motion_estimate");
        assert_eq!(WorkItemKind::Transfer.label(), "transfer");
    }

    #[test]
    fn test_work_item_estimated_cycles_compute() {
        let item = make_item(1, WorkItemKind::Compute, 1920, 1080);
        assert_eq!(item.estimated_cycles(), 1920 * 1080 * 8);
    }

    #[test]
    fn test_work_item_estimated_cycles_motion() {
        let item = make_item(2, WorkItemKind::MotionEstimate, 10, 10);
        assert_eq!(item.estimated_cycles(), 10 * 10 * 32);
    }

    #[test]
    fn test_work_item_estimated_cycles_transfer_cheapest() {
        let transfer = make_item(3, WorkItemKind::Transfer, 100, 100);
        let compute = make_item(4, WorkItemKind::Compute, 100, 100);
        assert!(transfer.estimated_cycles() < compute.estimated_cycles());
    }

    #[test]
    fn test_work_item_with_priority() {
        let item = WorkItem::with_priority(5, WorkItemKind::Scale, 640, 480, 10);
        assert_eq!(item.priority, 10);
    }

    #[test]
    fn test_batch_add_and_len() {
        let mut batch = WorkItemBatch::new();
        batch.add(make_item(1, WorkItemKind::Compute, 100, 100));
        batch.add(make_item(2, WorkItemKind::Scale, 200, 200));
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_empty() {
        let batch = WorkItemBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.total_cycles(), 0);
    }

    #[test]
    fn test_batch_total_cycles() {
        let mut batch = WorkItemBatch::new();
        batch.add(make_item(1, WorkItemKind::Transfer, 10, 10)); // 200
        batch.add(make_item(2, WorkItemKind::Compute, 10, 10)); // 800
        assert_eq!(batch.total_cycles(), 1000);
    }

    #[test]
    fn test_scheduler_orders_by_priority() {
        let mut sched = WorkItemScheduler::new();
        let mut batch = WorkItemBatch::new();
        batch.add(WorkItem::with_priority(1, WorkItemKind::Compute, 10, 10, 1));
        batch.add(WorkItem::with_priority(2, WorkItemKind::Scale, 10, 10, 5));
        let ordered = sched.schedule(batch);
        assert_eq!(ordered[0].id, 2); // higher priority first
        assert_eq!(ordered[1].id, 1);
    }

    #[test]
    fn test_scheduler_total_scheduled() {
        let mut sched = WorkItemScheduler::new();
        let mut batch = WorkItemBatch::new();
        batch.add(make_item(1, WorkItemKind::Compute, 10, 10));
        batch.add(make_item(2, WorkItemKind::Scale, 10, 10));
        sched.schedule(batch);
        assert_eq!(sched.total_scheduled(), 2);
    }

    #[test]
    fn test_scheduler_reset() {
        let mut sched = WorkItemScheduler::new();
        let mut batch = WorkItemBatch::new();
        batch.add(make_item(1, WorkItemKind::Transfer, 8, 8));
        sched.schedule(batch);
        sched.reset();
        assert_eq!(sched.total_scheduled(), 0);
    }

    #[test]
    fn test_batch_iter() {
        let mut batch = WorkItemBatch::new();
        batch.add(make_item(42, WorkItemKind::ColorConvert, 1, 1));
        let ids: Vec<u64> = batch.iter().map(|i| i.id).collect();
        assert_eq!(ids, vec![42]);
    }
}
