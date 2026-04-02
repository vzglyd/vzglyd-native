//! Archive handling for .vzglyd packages.

use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

/// Extracts a .vzglyd archive to a temporary directory.
pub fn extract_archive(archive_path: &Path) -> Result<TempPackage, String> {
    let file =
        std::fs::File::open(archive_path).map_err(|e| format!("Failed to open archive: {}", e))?;

    let mut archive = ZipArchive::new(file).map_err(|e| format!("Invalid archive: {}", e))?;

    // Create a temp directory unique to this archive (per-process + path hash).
    // Using path hash ensures multiple slides extract to distinct dirs.
    let path_hash = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        archive_path.hash(&mut h);
        h.finish()
    };
    let temp_dir =
        std::env::temp_dir().join(format!("vzglyd-{}-{:x}", std::process::id(), path_hash));

    std::fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {}", e))?;

    // Extract files
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("Failed to read file in archive: {}", e))?;

        let outpath = temp_dir.join(file.mangled_name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        } else {
            if let Some(parent) = outpath.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create parent directory: {}", e))?;
            }

            let mut contents = Vec::new();
            file.read_to_end(&mut contents)
                .map_err(|e| format!("Failed to read file: {}", e))?;

            std::fs::write(&outpath, &contents)
                .map_err(|e| format!("Failed to write file: {}", e))?;
        }
    }

    Ok(TempPackage { path: temp_dir })
}

/// Temporary package directory.
pub struct TempPackage {
    pub path: std::path::PathBuf,
}

impl Drop for TempPackage {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
