//! Embedded HTTP management server for the native VRX-64 host.
//!
//! The server runs in a dedicated OS thread that owns a Tokio runtime, keeping
//! it completely separate from the winit event loop on the main thread.
//!
//! Communication with the main thread uses two shared objects:
//! - [`std::sync::mpsc::SyncSender<String>`] — sends new playlist JSON to the
//!   main thread for hot-reload (bounded channel, capacity 1).
//! - [`Arc<RwLock<AppStatus>>`] — lets the server read the current slide and FPS.

pub mod routes;
pub mod secrets;
pub mod state;

pub use secrets::{load_secrets, save_secrets};
pub use state::{AppStatus, ServerState};

use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, RwLock};
use std::thread;

use axum::Router;
use axum::routing::{get, post};
use tower_http::cors::{Any, CorsLayer};
use vzglyd_kernel::SecretsStore;

/// Opaque handle that keeps the server thread alive.
///
/// When dropped the thread is detached (the server continues until the process exits).
/// There is no clean shutdown mechanism since the native app terminates the process.
pub struct ServerHandle {
    _thread: thread::JoinHandle<()>,
}

/// Start the management server in a background thread.
///
/// Returns:
/// - A [`ServerHandle`] whose lifetime ties to the server thread.
/// - A [`Receiver<String>`] that yields new playlist JSON strings sent by the server
///   whenever the user saves a playlist via the management UI.
/// - An [`Arc<RwLock<AppStatus>>`] that the caller (main thread) should update each
///   frame with the current slide path and FPS.
pub fn start_server(
    slides_dir: std::path::PathBuf,
    secrets: Arc<RwLock<SecretsStore>>,
) -> (ServerHandle, Receiver<String>, Arc<RwLock<AppStatus>>) {
    let (playlist_tx, playlist_rx): (SyncSender<String>, Receiver<String>) =
        mpsc::sync_channel(1);
    let app_status: Arc<RwLock<AppStatus>> = Arc::new(RwLock::new(AppStatus::default()));
    let app_status_clone = Arc::clone(&app_status);

    let thread = thread::Builder::new()
        .name("vzglyd-management".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("build tokio runtime for management server");

            rt.block_on(async move {
                let state = ServerState {
                    slides_dir,
                    playlist_tx,
                    secrets,
                    app_status: app_status_clone,
                };

                let cors = CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any);

                let app = Router::new()
                    // Static assets
                    .route("/", get(routes::get_index))
                    .route(
                        "/assets/pkg/vzglyd_web_bg.wasm",
                        get(routes::get_vzglyd_wasm),
                    )
                    .route(
                        "/assets/pkg/vzglyd_web.js",
                        get(routes::get_vzglyd_js),
                    )
                    .route(
                        "/assets/pkg/vzglyd_web_bg.js",
                        get(routes::get_vzglyd_bg_js),
                    )
                    // Playlist
                    .route("/api/playlist", get(routes::get_playlist))
                    .route("/api/playlist", post(routes::post_playlist))
                    // Slides
                    .route("/api/slides", get(routes::get_slides))
                    .route("/api/slides/upload", post(routes::upload_slide))
                    .route(
                        "/api/slides/:path/manifest",
                        get(routes::get_slide_manifest),
                    )
                    .route(
                        "/api/slides/:path/bundle",
                        get(routes::get_slide_bundle),
                    )
                    // Secrets
                    .route("/api/secrets", get(routes::get_secrets))
                    .route("/api/secrets", post(routes::post_secrets))
                    .route("/api/secrets/export", get(routes::export_secrets))
                    // Status
                    .route("/api/status", get(routes::get_status))
                    .with_state(state)
                    .layer(cors);

                let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
                    .await
                    .expect("bind management server to 0.0.0.0:8080");

                eprintln!("[vzglyd] management server listening on http://0.0.0.0:8080");
                axum::serve(listener, app)
                    .await
                    .expect("management server error");
            });
        })
        .expect("spawn management server thread");

    (ServerHandle { _thread: thread }, playlist_rx, app_status)
}
