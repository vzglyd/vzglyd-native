//! File I/O for `secrets.json` in the slides directory.

use std::path::Path;

use vzglyd_kernel::{SECRETS_FILENAME, SecretsStore};

/// Load `secrets.json` from `slides_dir`.
///
/// Returns an empty [`SecretsStore`] if the file does not exist.
/// Returns an error if the file exists but cannot be read or parsed.
pub fn load_secrets(slides_dir: &Path) -> Result<SecretsStore, String> {
    let path = slides_dir.join(SECRETS_FILENAME);
    if !path.exists() {
        return Ok(SecretsStore::default());
    }
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("read secrets.json: {e}"))?;
    SecretsStore::from_json(&content).map_err(|e| format!("parse secrets.json: {e}"))
}

/// Write `secrets.json` to `slides_dir`, creating or overwriting it.
pub fn save_secrets(slides_dir: &Path, secrets: &SecretsStore) -> Result<(), String> {
    let path = slides_dir.join(SECRETS_FILENAME);
    let json = secrets.to_json().map_err(|e| format!("serialize secrets: {e}"))?;
    // Atomic write: write to a temp file then rename.
    let tmp_path = path.with_extension("json.tmp");
    std::fs::write(&tmp_path, json.as_bytes())
        .map_err(|e| format!("write secrets.json.tmp: {e}"))?;
    std::fs::rename(&tmp_path, &path).map_err(|e| format!("rename secrets.json: {e}"))?;
    Ok(())
}
