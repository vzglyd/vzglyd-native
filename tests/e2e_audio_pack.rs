//! End-to-end tests for audio in .vzglyd bundles.
//!
//! These tests exercise the pack pipeline:
//! 1. Create a slide directory with manifest.json referencing sound files
//! 2. Pack the directory into a .vzglyd archive
//! 3. Verify the archive contains all declared assets
//! 4. Load sounds from the archive and play through the AudioEngine

use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use vzglyd_native::audio::AudioEngine;
use vzglyd_native::slide_loader::pack_slide_directory;

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Generate a minimal valid WAV file (16-bit PCM, 44100 Hz, mono, 100ms silence).
fn make_wav_bytes() -> Vec<u8> {
    let sample_rate: u32 = 44_100;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let num_samples = sample_rate as usize / 10;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as usize * (bits_per_sample as usize / 8);
    let chunk_size: u32 = 36 + data_size as u32;

    let mut buf = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_size as u32).to_le_bytes());
    buf.resize(44 + data_size, 0u8);
    buf
}

/// Create a unique temp directory for a test slide package.
fn temp_slide_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vzglyd_e2e_audio_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_millis()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

/// Write a minimal manifest.json + dummy slide.wasm to a slide directory.
fn write_minimal_slide(dir: &PathBuf, manifest_json: &str) {
    write_required_art_assets(dir);
    fs::write(dir.join("manifest.json"), manifest_with_required_art(manifest_json))
        .expect("write manifest");
    fs::write(dir.join("slide.wasm"), b"\0asm\x01\0\0\0").expect("write dummy wasm");
}

fn write_required_art_assets(dir: &PathBuf) {
    let art_dir = dir.join("art");
    fs::create_dir_all(&art_dir).expect("create art dir");
    fs::write(art_dir.join("j-card.png"), b"j-card").expect("write j-card art");
    fs::write(art_dir.join("side-a.png"), b"side-a").expect("write side A art");
    fs::write(art_dir.join("side-b.png"), b"side-b").expect("write side B art");
}

fn manifest_with_required_art(manifest: &str) -> String {
    let mut value: serde_json::Value =
        serde_json::from_str(manifest).expect("test manifest should be valid JSON");
    let object = value.as_object_mut().expect("test manifest should be an object");
    let assets = object
        .entry("assets")
        .or_insert_with(|| serde_json::json!({}));
    let assets_object = assets
        .as_object_mut()
        .expect("test manifest assets should be an object");
    assets_object.insert(
        "art".into(),
        serde_json::json!({
            "j_card": { "path": "art/j-card.png" },
            "side_a_label": { "path": "art/side-a.png" },
            "side_b_label": { "path": "art/side-b.png" }
        }),
    );
    serde_json::to_string(&value).expect("serialize test manifest")
}

// ── End-to-end tests ─────────────────────────────────────────────────────────

#[test]
fn e2e_pack_vzglyd_contains_declared_sound() {
    // 1. Create slide directory with sound asset
    let dir = temp_slide_dir("pack_sound");
    let wav = make_wav_bytes();
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("click.wav"), &wav).expect("write wav");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "Audio Test Slide",
            "abi_version": 3,
            "scene_space": "screen_2d",
            "assets": {
                "sounds": [
                    { "id": "click", "path": "assets/click.wav" }
                ]
            }
        }"#,
    );

    // 2. Pack into .vzglyd
    let archive_path = dir.join("test.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // 3. Verify the sound file is in the archive
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");
    let mut sound_entry = archive
        .by_name("assets/click.wav")
        .expect("sound file should be in archive");
    let mut archived_sound = Vec::new();
    sound_entry
        .read_to_end(&mut archived_sound)
        .expect("read sound");
    assert_eq!(archived_sound, wav);
}

#[test]
fn e2e_pack_vzglyd_contains_multiple_sounds() {
    // 1. Create slide with two sounds
    let dir = temp_slide_dir("multi_sound");
    let wav1 = make_wav_bytes();
    let wav2 = make_wav_bytes();
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("sound1.wav"), &wav1).expect("write sound1");
    fs::write(dir.join("assets").join("sound2.wav"), &wav2).expect("write sound2");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "Multi Sound Slide",
            "abi_version": 3,
            "scene_space": "screen_2d",
            "assets": {
                "sounds": [
                    { "id": "sound1", "path": "assets/sound1.wav" },
                    { "id": "sound2", "path": "assets/sound2.wav" }
                ]
            }
        }"#,
    );

    // 2. Pack
    let archive_path = dir.join("multi.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // 3. Verify both sounds are in the archive
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");

    let mut s1 = Vec::new();
    archive
        .by_name("assets/sound1.wav")
        .expect("sound1 in archive")
        .read_to_end(&mut s1)
        .unwrap();
    assert_eq!(s1, wav1);

    let mut s2 = Vec::new();
    archive
        .by_name("assets/sound2.wav")
        .expect("sound2 in archive")
        .read_to_end(&mut s2)
        .unwrap();
    assert_eq!(s2, wav2);
}

#[test]
fn e2e_pack_vzglyd_contains_textures_and_sounds() {
    // 1. Create slide with texture + sound
    let dir = temp_slide_dir("all_assets");
    let wav = make_wav_bytes();
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("click.wav"), &wav).expect("write wav");
    let png = create_minimal_png();
    fs::write(dir.join("assets").join("pixel.png"), &png).expect("write png");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "All Assets",
            "abi_version": 3,
            "scene_space": "screen_2d",
            "assets": {
                "textures": [{ "path": "assets/pixel.png" }],
                "sounds": [{ "id": "click", "path": "assets/click.wav" }]
            }
        }"#,
    );

    // 2. Pack
    let archive_path = dir.join("all.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // 3. Verify archive contains everything
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");

    assert!(archive.by_name("manifest.json").is_ok());
    assert!(archive.by_name("slide.wasm").is_ok());
    assert!(archive.by_name("assets/click.wav").is_ok());
    assert!(archive.by_name("assets/pixel.png").is_ok());
}

#[test]
fn e2e_packed_sound_plays_through_engine() {
    // 1. Create slide with sound
    let dir = temp_slide_dir("play_sound");
    let wav = make_wav_bytes();
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("tone.wav"), &wav).expect("write wav");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "Play Test",
            "abi_version": 3,
            "scene_space": "screen_2d",
            "assets": {
                "sounds": [{ "id": "tone", "path": "assets/tone.wav" }]
            }
        }"#,
    );

    // 2. Pack
    let archive_path = dir.join("play.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // 3. Extract sound from archive and play it
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");
    let mut sound_data = Vec::new();
    archive
        .by_name("assets/tone.wav")
        .expect("tone in archive")
        .read_to_end(&mut sound_data)
        .unwrap();

    // 4. Play through the audio engine
    let engine = AudioEngine::global().expect("audio engine available");
    let handle = engine
        .play(&sound_data, 0.5, false)
        .expect("packed sound should play");
    std::thread::sleep(std::time::Duration::from_millis(30));
    handle.stop();
}

#[test]
fn e2e_pack_vzglyd_preserves_sound_file_integrity() {
    // Create slide with sound
    let dir = temp_slide_dir("integrity");
    let wav = make_wav_bytes();
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("check.wav"), &wav).expect("write wav");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "Integrity Test",
            "abi_version": 3,
            "scene_space": "screen_2d",
            "assets": {
                "sounds": [{ "id": "check", "path": "assets/check.wav" }]
            }
        }"#,
    );

    // Pack
    let archive_path = dir.join("integrity.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // Extract and verify byte-for-byte match
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");
    let mut extracted = Vec::new();
    archive
        .by_name("assets/check.wav")
        .expect("check.wav in archive")
        .read_to_end(&mut extracted)
        .unwrap();

    assert_eq!(
        extracted.len(),
        wav.len(),
        "extracted sound should have same size"
    );
    assert_eq!(extracted, wav, "extracted sound should match original");
}

/// Create a minimal valid 1x1 PNG file.
fn create_minimal_png() -> Vec<u8> {
    vec![
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
        0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41,
        0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
        0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
        0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
        0x42, 0x60, 0x82,
    ]
}
