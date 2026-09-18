//! Command handler functions for monitor, restore, captions, preset, probe, and info.
//!
//! The handlers are grouped by domain into submodules and re-exported here so that
//! callers can keep using `crate::handlers::Foo` paths unchanged.

pub(crate) mod dispatch;
pub(crate) mod inspect;
pub(crate) mod logging;
pub(crate) mod preset_ui;
pub(crate) mod reference;

pub(crate) use dispatch::{
    handle_captions_command, handle_monitor_command, handle_restore_command,
};
pub(crate) use inspect::probe_file;
pub(crate) use logging::{init_color, init_logging, LogFormat};
pub(crate) use preset_ui::handle_preset_command;
pub(crate) use reference::{show_info, show_version};
