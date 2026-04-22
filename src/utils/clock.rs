//! Clock and time utilities.

use chrono::{Local, Timelike};
use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current time in seconds since UNIX epoch.
pub fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Returns the current time as seconds since midnight (UTC).
///
/// Used to populate the `clock_seconds` shader uniform. The value ranges from
/// 0.0 (midnight) to 86400.0 (end of day). UTC is used here because native
/// Rust has no stable cross-platform way to query the system local timezone
/// offset without an external dependency. Slides that need local wall-clock
/// time in a specific timezone should derive it from `clock_seconds` plus an
/// offset passed via the params system.
pub fn local_clock_seconds() -> f32 {
    let now = Local::now();
    (now.hour() * 3600 + now.minute() * 60 + now.second()) as f32
}
