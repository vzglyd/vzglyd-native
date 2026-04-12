//! Axum route handlers for the management HTTP server.

use std::io::Read;
use std::path::Path;

use axum::Json;
use axum::body::Body;
use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::Value;
use vzglyd_kernel::SecretsStore;
use vzglyd_kernel::SlideLibraryEntry;
use vzglyd_kernel::manifest::SlideManifest;
use vzglyd_kernel::schedule::{PLAYLIST_FILENAME, parse_playlist};

use super::secrets::save_secrets;
use super::state::ServerState;

// ── Embedded static assets ────────────────────────────────────────────────────

const MANAGEMENT_HTML: &str = include_str!("../ui/management.html");

// ── Static asset routes ───────────────────────────────────────────────────────

/// Serve the management SPA.
pub async fn get_index() -> impl IntoResponse {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(MANAGEMENT_HTML))
        .unwrap()
}

// ── Playlist routes ───────────────────────────────────────────────────────────

/// GET /api/playlist — return current playlist.json contents.
pub async fn get_playlist(State(state): State<ServerState>) -> impl IntoResponse {
    let path = state.slides_dir.join(PLAYLIST_FILENAME);
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => match serde_json::from_str::<Value>(&content) {
            Ok(json) => (StatusCode::OK, Json(json)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("playlist is not valid JSON: {e}")})),
            )
                .into_response(),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "playlist.json not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("read error: {e}")})),
        )
            .into_response(),
    }
}

/// POST /api/playlist — validate and write playlist.json, then signal hot-reload.
pub async fn post_playlist(
    State(state): State<ServerState>,
    body: String,
) -> impl IntoResponse {
    // Validate with the kernel parser.
    if let Err(e) = parse_playlist(body.as_bytes()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid playlist: {e}")})),
        )
            .into_response();
    }

    // Pretty-print and write atomically.
    let json: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("JSON parse error: {e}")})),
            )
                .into_response();
        }
    };
    let pretty =
        serde_json::to_string_pretty(&json).expect("serialization of valid JSON is infallible");

    let path = state.slides_dir.join(PLAYLIST_FILENAME);
    let tmp_path = state.slides_dir.join("playlist.json.tmp");
    if let Err(e) = tokio::fs::write(&tmp_path, pretty.as_bytes()).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("write error: {e}")})),
        )
            .into_response();
    }
    if let Err(e) = tokio::fs::rename(&tmp_path, &path).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("rename error: {e}")})),
        )
            .into_response();
    }

    // Signal the main thread. Ignore errors (e.g., channel full — main thread
    // will pick up the file on the next poll anyway).
    let _ = state.playlist_tx.try_send(pretty);

    (StatusCode::OK, Json(json)).into_response()
}

// ── Slide library routes ──────────────────────────────────────────────────────

/// GET /api/slides — list .vzglyd files in the slides directory.
pub async fn get_slides(State(state): State<ServerState>) -> impl IntoResponse {
    let dir = state.slides_dir.clone();
    let result = tokio::task::spawn_blocking(move || list_slide_bundles(&dir)).await;
    match result {
        Ok(Ok(entries)) => (StatusCode::OK, Json(entries)).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("task error: {e}")})),
        )
            .into_response(),
    }
}

fn list_slide_bundles(slides_dir: &Path) -> Result<Vec<SlideLibraryEntry>, String> {
    let rd = std::fs::read_dir(slides_dir).map_err(|e| format!("read dir: {e}"))?;
    let mut entries: Vec<SlideLibraryEntry> = rd
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                == Some("vzglyd")
        })
        .map(|e| {
            let path = e.path();
            let size_bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let manifest = extract_manifest_from_archive(&path).ok();
            SlideLibraryEntry {
                path: filename,
                size_bytes,
                manifest,
            }
        })
        .collect();
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
}

/// POST /api/slides/upload — receive and save a .vzglyd file.
pub async fn upload_slide(
    State(state): State<ServerState>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let mut filename: Option<String> = None;
    let mut file_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            let raw_name = field
                .file_name()
                .unwrap_or("upload.vzglyd")
                .to_string();
            // Sanitize filename — strip any path components.
            let safe_name = std::path::Path::new(&raw_name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("upload.vzglyd")
                .to_string();
            if !safe_name.ends_with(".vzglyd") {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "file must have .vzglyd extension"})),
                )
                    .into_response();
            }
            match field.bytes().await {
                Ok(bytes) => {
                    filename = Some(safe_name);
                    file_bytes = Some(bytes.to_vec());
                }
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({"error": format!("read upload: {e}")})),
                    )
                        .into_response();
                }
            }
            break;
        }
    }

    let (filename, bytes) = match (filename, file_bytes) {
        (Some(f), Some(b)) => (f, b),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "missing 'file' field in multipart"})),
            )
                .into_response();
        }
    };

    let dest = state.slides_dir.join(&filename);
    let tmp = state.slides_dir.join(format!("{filename}.tmp"));
    if let Err(e) = tokio::fs::write(&tmp, &bytes).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("write: {e}")})),
        )
            .into_response();
    }
    if let Err(e) = tokio::fs::rename(&tmp, &dest).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("rename: {e}")})),
        )
            .into_response();
    }

    (
        StatusCode::CREATED,
        Json(serde_json::json!({"path": filename})),
    )
        .into_response()
}

/// GET /api/slides/:path/manifest — extract and return manifest from a .vzglyd bundle.
pub async fn get_slide_manifest(
    State(state): State<ServerState>,
    AxumPath(bundle_path): AxumPath<String>,
) -> impl IntoResponse {
    if let Err(e) = validate_bundle_path(&bundle_path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response();
    }

    let full_path = state.slides_dir.join(&bundle_path);
    let result = tokio::task::spawn_blocking(move || {
        extract_manifest_from_archive(&full_path)
    })
    .await;

    match result {
        Ok(Ok(manifest)) => (StatusCode::OK, Json(manifest)).into_response(),
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("task error: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/slides/:path/bundle — serve the raw .vzglyd bytes for browser preview.
pub async fn get_slide_bundle(
    State(state): State<ServerState>,
    AxumPath(bundle_path): AxumPath<String>,
) -> impl IntoResponse {
    if let Err(e) = validate_bundle_path(&bundle_path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response();
    }

    let full_path = state.slides_dir.join(&bundle_path);
    match tokio::fs::read(&full_path).await {
        Ok(bytes) => Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(Body::from(bytes))
            .unwrap()
            .into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("bundle not found: {bundle_path}")})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("read error: {e}")})),
        )
            .into_response(),
    }
}

/// GET /api/slides/:path/art/:kind — serve declared cassette artwork from a bundle.
pub async fn get_slide_art(
    State(state): State<ServerState>,
    AxumPath((bundle_path, kind)): AxumPath<(String, String)>,
) -> impl IntoResponse {
    if let Err(e) = validate_bundle_path(&bundle_path) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response();
    }
    if let Err(e) = validate_art_kind(&kind) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": e})),
        )
            .into_response();
    }

    let full_path = state.slides_dir.join(&bundle_path);
    let result = tokio::task::spawn_blocking(move || {
        extract_art_from_archive(&full_path, &kind)
    })
    .await;

    match result {
        Ok(Ok((content_type, bytes))) => Response::builder()
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(bytes))
            .unwrap()
            .into_response(),
        Ok(Err(e)) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("task error: {e}")})),
        )
            .into_response(),
    }
}

// ── Secrets routes ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct SecretsListResponse {
    keys: Vec<String>,
}

/// GET /api/secrets — return key names only (no values).
pub async fn get_secrets(State(state): State<ServerState>) -> impl IntoResponse {
    let keys = state
        .secrets
        .read()
        .map(|s| s.keys().iter().map(|k| k.to_string()).collect::<Vec<_>>())
        .unwrap_or_default();
    (StatusCode::OK, Json(SecretsListResponse { keys })).into_response()
}

/// POST /api/secrets — merge new secrets, persist to disk, update shared Arc.
pub async fn post_secrets(
    State(state): State<ServerState>,
    Json(body): Json<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    // Validate: no empty keys, no NUL bytes.
    for key in body.keys() {
        if key.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "key must not be empty"})),
            )
                .into_response();
        }
        if key.contains('\0') || body[key].contains('\0') {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "key/value must not contain NUL bytes"})),
            )
                .into_response();
        }
    }

    let patch = SecretsStore(body);

    // Update in-memory store.
    {
        let Ok(mut guard) = state.secrets.write() else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "secrets lock poisoned"})),
            )
                .into_response();
        };
        guard.merge(patch);
    }

    // Persist to disk.
    let secrets_snapshot = state
        .secrets
        .read()
        .map(|s| s.clone())
        .unwrap_or_default();
    let slides_dir = state.slides_dir.clone();
    if let Err(e) = tokio::task::spawn_blocking(move || save_secrets(&slides_dir, &secrets_snapshot))
        .await
        .unwrap_or_else(|e| Err(format!("task error: {e}")))
    {
        log::warn!("failed to persist secrets: {e}");
    }

    let keys = state
        .secrets
        .read()
        .map(|s| s.keys().iter().map(|k| k.to_string()).collect::<Vec<_>>())
        .unwrap_or_default();
    (StatusCode::OK, Json(SecretsListResponse { keys })).into_response()
}

/// GET /api/secrets/export — return FULL secrets JSON (values included).
///
/// Only expose over localhost. The management UI uses this for "Export secrets.json"
/// so users can place the file next to playlist.json for the web editor to discover.
pub async fn export_secrets(State(state): State<ServerState>) -> impl IntoResponse {
    let json = state
        .secrets
        .read()
        .ok()
        .and_then(|s| s.to_json().ok())
        .unwrap_or_else(|| "{}".to_string());
    Response::builder()
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::CONTENT_DISPOSITION,
            "attachment; filename=\"secrets.json\"",
        )
        .body(Body::from(json))
        .unwrap()
        .into_response()
}

// ── Status route ──────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct StatusResponse {
    current_slide: Option<String>,
    fps: f32,
}

/// GET /api/status — current slide and FPS.
pub async fn get_status(State(state): State<ServerState>) -> impl IntoResponse {
    let status = state
        .app_status
        .read()
        .map(|s| StatusResponse {
            current_slide: s.current_slide.clone(),
            fps: s.fps,
        })
        .unwrap_or_else(|_| StatusResponse {
            current_slide: None,
            fps: 0.0,
        });
    (StatusCode::OK, Json(status)).into_response()
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Validate a bundle path — must end in `.vzglyd` and must not escape the dir.
fn validate_bundle_path(path: &str) -> Result<(), String> {
    if !path.ends_with(".vzglyd") {
        return Err("path must end with .vzglyd".into());
    }
    let p = std::path::Path::new(path);
    for component in p.components() {
        match component {
            std::path::Component::Prefix(_)
            | std::path::Component::RootDir
            | std::path::Component::ParentDir => {
                return Err(format!("path '{path}' attempts to escape the slides directory"));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_art_kind(kind: &str) -> Result<(), String> {
    match kind {
        "j-card" | "side-a" | "side-b" => Ok(()),
        _ => Err("art kind must be one of: j-card, side-a, side-b".into()),
    }
}

/// Extract and parse `manifest.json` from a `.vzglyd` ZIP archive.
fn extract_manifest_from_archive(path: &std::path::Path) -> Result<SlideManifest, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("open '{}': {e}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("read zip '{}': {e}", path.display()))?;

    // Look for manifest.json at root or any path ending with manifest.json.
    let manifest_idx = (0..zip.len()).find(|&i| {
        zip.by_index(i)
            .map(|f| {
                let name = f.name();
                name == "manifest.json" || name.ends_with("/manifest.json")
            })
            .unwrap_or(false)
    });

    let Some(idx) = manifest_idx else {
        return Err(format!(
            "manifest.json not found in '{}'",
            path.display()
        ));
    };

    let mut entry = zip.by_index(idx).map_err(|e| format!("read manifest entry: {e}"))?;
    let mut content = String::new();
    entry
        .read_to_string(&mut content)
        .map_err(|e| format!("read manifest content: {e}"))?;

    let manifest = vzglyd_kernel::manifest::parse_manifest(content.as_bytes())
        .map_err(|e| format!("parse manifest: {e}"))?;
    manifest
        .validate(crate::slide_loader::ABI_VERSION)
        .map_err(|e| format!("validate manifest: {e}"))?;
    Ok(manifest)
}

fn extract_art_from_archive(path: &Path, kind: &str) -> Result<(&'static str, Vec<u8>), String> {
    let manifest = extract_manifest_from_archive(path)?;
    let art_path = cassette_art_path(&manifest, kind)?;

    let file = std::fs::File::open(path)
        .map_err(|e| format!("open '{}': {e}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|e| format!("read zip '{}': {e}", path.display()))?;
    let mut entry = zip.by_name(art_path).map_err(|e| {
        format!(
            "art asset '{art_path}' not found in '{}': {e}",
            path.display()
        )
    })?;
    let mut bytes = Vec::new();
    entry
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read art asset '{art_path}': {e}"))?;
    Ok((image_content_type(art_path), bytes))
}

fn cassette_art_path<'a>(manifest: &'a SlideManifest, kind: &str) -> Result<&'a str, String> {
    let art = manifest
        .assets
        .as_ref()
        .and_then(|assets| assets.art.as_ref())
        .ok_or_else(|| "manifest.assets.art is required".to_string())?;
    let path = match kind {
        "j-card" => &art.j_card.path,
        "side-a" => &art.side_a_label.path,
        "side-b" => &art.side_b_label.path,
        _ => return Err("art kind must be one of: j-card, side-a, side-b".into()),
    };
    Ok(path)
}

fn image_content_type(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use zip::write::SimpleFileOptions;
    use zip::{CompressionMethod, ZipWriter};

    fn temp_archive_path(label: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "vzglyd_routes_{label}_{}_{}.vzglyd",
            std::process::id(),
            unique
        ))
    }

    fn write_archive(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).expect("create test archive");
        let mut zip = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for (name, bytes) in entries {
            zip.start_file(*name, options).expect("start zip entry");
            zip.write_all(bytes).expect("write zip entry");
        }
        zip.finish().expect("finish zip");
    }

    fn manifest_with_art() -> Vec<u8> {
        br#"{
            "name":"Art Test",
            "abi_version":3,
            "scene_space":"screen_2d",
            "assets":{
                "art":{
                    "j_card":{"path":"art/j-card.png"},
                    "side_a_label":{"path":"art/side-a.png"},
                    "side_b_label":{"path":"art/side-b.png"}
                }
            }
        }"#
        .to_vec()
    }

    #[test]
    fn extract_art_from_archive_reads_manifest_declared_asset() {
        let archive_path = temp_archive_path("art_ok");
        let manifest = manifest_with_art();
        write_archive(
            &archive_path,
            &[
                ("manifest.json", manifest.as_slice()),
                ("art/j-card.png", b"cover"),
                ("art/side-a.png", b"side-a"),
                ("art/side-b.png", b"side-b"),
            ],
        );

        let (content_type, bytes) =
            extract_art_from_archive(&archive_path, "side-a").expect("extract side A art");
        assert_eq!(content_type, "image/png");
        assert_eq!(bytes, b"side-a");

        let _ = std::fs::remove_file(archive_path);
    }

    #[test]
    fn extract_manifest_from_archive_rejects_missing_cassette_art() {
        let archive_path = temp_archive_path("missing_art");
        write_archive(
            &archive_path,
            &[(
                "manifest.json",
                br#"{"name":"Old Bundle","abi_version":3,"scene_space":"screen_2d"}"#,
            )],
        );

        let error =
            extract_manifest_from_archive(&archive_path).expect_err("missing art should fail");
        assert!(error.contains("manifest.assets.art is required"));

        let _ = std::fs::remove_file(archive_path);
    }
}
