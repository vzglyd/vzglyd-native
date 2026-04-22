use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ── 1. Compile loading-slide to wasm32-wasip1 ─────────────────────────────
    let loading_slide_dir = manifest_dir.join("loading-slide");
    println!("cargo:rerun-if-changed=loading-slide/src");
    println!("cargo:rerun-if-changed=loading-slide/assets");
    println!("cargo:rerun-if-changed=loading-slide/art");
    println!("cargo:rerun-if-changed=loading-slide/manifest.json");
    println!("cargo:rerun-if-changed=loading-slide/Cargo.toml");

    compile_wasm_slide(&manifest_dir, &loading_slide_dir, "loading_slide")
        .expect("loading-slide wasm build failed");

    let loading_wasm = resolve_slide_wasm(&manifest_dir, &loading_slide_dir, "loading_slide")
        .unwrap_or_else(|| panic!("failed to locate compiled loading slide wasm after build"));

    // Pack loading.vzglyd
    let loading_vzglyd = out_dir.join("loading.vzglyd");
    pack_vzglyd(
        &loading_vzglyd,
        &loading_wasm,
        &loading_slide_dir,
        &["assets", "art"],
    )
    .expect("failed to pack loading.vzglyd");

    // ── 2. Compile information-slide to wasm32-wasip1 ────────────────────────
    let info_slide_dir = manifest_dir.join("information-slide");
    println!("cargo:rerun-if-changed=information-slide/src");
    println!("cargo:rerun-if-changed=information-slide/manifest.json");
    println!("cargo:rerun-if-changed=information-slide/Cargo.toml");

    compile_wasm_slide(&manifest_dir, &info_slide_dir, "information_slide")
        .expect("information-slide wasm build failed");

    let info_wasm = resolve_slide_wasm(&manifest_dir, &info_slide_dir, "information_slide")
        .unwrap_or_else(|| panic!("failed to locate compiled information slide wasm after build"));

    // Pack information.vzglyd
    let info_vzglyd = out_dir.join("information.vzglyd");
    pack_vzglyd(&info_vzglyd, &info_wasm, &info_slide_dir, &[])
        .expect("failed to pack information.vzglyd");

    println!("cargo:rerun-if-changed=build.rs");
}

fn compile_wasm_slide(
    _manifest_dir: &Path,
    slide_dir: &Path,
    package_name: &str,
) -> Result<(), String> {
    let status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args([
            "build",
            "--manifest-path",
            slide_dir.join("Cargo.toml").to_str().unwrap(),
            "--target",
            "wasm32-wasip1",
            "--release",
            "--quiet",
        ])
        .status()
        .map_err(|e| format!("failed to invoke cargo for {}: {}", package_name, e))?;

    if !status.success() {
        return Err(format!("{} wasm build failed", package_name));
    }

    // Rebuild whenever slide source changes.
    println!(
        "cargo:rerun-if-changed={}/src",
        package_name.replace('-', "_")
    );

    Ok(())
}

fn resolve_slide_wasm(manifest_dir: &Path, slide_dir: &Path, slide_name: &str) -> Option<PathBuf> {
    let wasm_file = format!("{}.wasm", slide_name);
    let candidates = [
        manifest_dir.join(format!("target/wasm32-wasip1/release/{}", wasm_file)),
        manifest_dir.join(format!("target/wasm32-wasip1/release/deps/{}", wasm_file)),
        slide_dir.join(format!("target/wasm32-wasip1/release/{}", wasm_file)),
        slide_dir.join(format!("target/wasm32-wasip1/release/deps/{}", wasm_file)),
    ];

    candidates.into_iter().find(|path| path.exists())
}

fn pack_vzglyd(
    out: &Path,
    wasm: &Path,
    slide_dir: &Path,
    subdirs: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::BufWriter;

    let file = fs::File::create(out)?;
    let mut zip = zip::ZipWriter::new(BufWriter::new(file));
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // manifest.json
    zip.start_file("manifest.json", stored)?;
    zip.write_all(&fs::read(slide_dir.join("manifest.json"))?)?;

    // slide.wasm
    zip.start_file("slide.wasm", stored)?;
    zip.write_all(&fs::read(wasm)?)?;

    for dir in subdirs {
        pack_child_dir(&mut zip, stored, slide_dir, dir)?;
    }

    zip.finish()?;
    Ok(())
}

fn pack_child_dir<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    options: zip::write::SimpleFileOptions,
    slide_dir: &Path,
    child: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = slide_dir.join(child);
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let name = entry.file_name();
            zip.start_file(format!("{child}/{}", name.to_string_lossy()), options)?;
            zip.write_all(&fs::read(entry.path())?)?;
        }
    }
    Ok(())
}
