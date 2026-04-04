//! Native tracing helpers and shared guest-trace payload parsing.

use std::collections::BTreeMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde::Deserialize;
use vzglyd_kernel::trace::TraceRecorder;

static ACTIVE_TRACE_RECORDER: Lazy<Mutex<Option<TraceRecorder>>> = Lazy::new(|| Mutex::new(None));

/// JSON payload emitted by guest span starts and instant events.
#[derive(Debug, Deserialize)]
pub struct GuestTracePayload {
    /// Event or span name.
    pub name: String,
    /// String attributes attached to the event.
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
}

/// JSON payload emitted when a guest span ends.
#[derive(Debug, Deserialize, Default)]
pub struct GuestTraceEndPayload {
    /// Optional final status value.
    pub status: Option<String>,
    /// String attributes attached when the span ends.
    #[serde(default)]
    pub attrs: BTreeMap<String, String>,
}

/// Replace the active process-wide trace recorder.
pub fn set_active_trace_recorder(recorder: Option<TraceRecorder>) {
    let mut slot = ACTIVE_TRACE_RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *slot = recorder;
}

/// Return a clone of the active process-wide trace recorder, if tracing is enabled.
pub fn active_trace_recorder() -> Option<TraceRecorder> {
    ACTIVE_TRACE_RECORDER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Decode a guest trace start or instant-event payload.
pub fn parse_guest_trace_payload(message: &str) -> Option<GuestTracePayload> {
    serde_json::from_str(message).ok()
}

/// Decode a guest trace end payload.
pub fn parse_guest_trace_end_payload(message: &str) -> GuestTraceEndPayload {
    serde_json::from_str(message).unwrap_or_default()
}
