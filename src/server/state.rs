//! Shared state types for the management HTTP server.

use std::path::PathBuf;
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, RwLock};

use vzglyd_kernel::SecretsStore;

/// Snapshot of the display engine status read by the `/api/status` endpoint.
#[derive(Debug, Clone, Default)]
pub struct AppStatus {
    /// Path of the slide currently on screen, relative to the slides directory.
    pub current_slide: Option<String>,
    /// Rolling-average frames per second.
    pub fps: f32,
}

/// Axum shared state passed to all route handlers.
#[derive(Clone)]
pub struct ServerState {
    /// Absolute path to the slides directory (contains `playlist.json`).
    pub slides_dir: PathBuf,
    /// Channel for sending a new playlist JSON string to the main app thread.
    pub playlist_tx: SyncSender<String>,
    /// Shared secrets store — read by all routes, written only by POST /api/secrets.
    pub secrets: Arc<RwLock<SecretsStore>>,
    /// Current display engine status — written by the main thread, read by routes.
    pub app_status: Arc<RwLock<AppStatus>>,
}
