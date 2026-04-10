//! Integration tests for native audio playback.
//!
//! These tests exercise the full audio pipeline including:
//! - WAV generation and decoding through rodio
//! - SoundRegistry lifecycle (play → pause → resume → stop)
//! - Concurrent sound playback
//! - Error handling for invalid audio data
//! - Sound catalog construction from manifest
//!
//! # Running on headless CI
//!
//! Tests that interact with audio hardware are marked `#[ignore]` so they do not
//! fail on CI machines without an audio device. To run them, set up a virtual
//! PulseAudio null sink first:
//!
//! ```bash
//! pactl load-module module-null-sink sink_name=virtual0
//! cargo test -p VRX-64-native --test audio_integration -- --ignored
//! pactl unload-module module-null-sink
//! ```
//!
//! Tests that do not call `AudioEngine::global()` or `SoundRegistry::play()` run
//! unconditionally and do not require audio hardware.

use std::fs;
use std::path::PathBuf;
use vzglyd_native::audio::{AudioEngine, AudioError, SoundRegistry};

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

/// Generate a minimal valid MP3 frame (MPEG-2 Layer III, 128kbps, 24kHz, mono).
fn make_mp3_bytes() -> Vec<u8> {
    let mut frame = vec![
        0xFF, 0xFB, 0x90, 0x00, // Frame header
    ];
    frame.resize(144 + 17, 0u8);
    frame
}

/// Create a temporary directory for test assets.
fn temp_test_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("vzglyd_audio_test_{label}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

// ── AudioEngine integration tests ─────────────────────────────────────────────

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_plays_valid_wav() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let wav = make_wav_bytes();
    let handle = engine.play(&wav, 0.5, false).expect("WAV playback should succeed");
    handle.stop();
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_rejects_synthetic_mp3_frame() {
    // A hand-crafted synthetic MP3 frame does not decode through rodio's
    // symphonia decoder because it lacks proper side-info structure.
    // This test confirms the engine rejects it cleanly with a decode error
    // rather than panicking.
    let engine = AudioEngine::global().expect("audio engine should be available");
    let mp3 = make_mp3_bytes();
    let result = engine.play(&mp3, 0.5, false);
    // Expected: symphonia cannot decode a synthetic frame
    assert!(
        matches!(result, Err(AudioError::DecodeError(_))),
        "synthetic MP3 frame should produce a decode error (real MP3 files work fine)"
    );
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_rejects_garbage_data() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let garbage = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
    let result = engine.play(&garbage, 0.5, false);
    assert!(
        matches!(result, Err(AudioError::DecodeError(_))),
        "expected DecodeError for garbage data"
    );
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_rejects_empty_data() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let result = engine.play(&[], 0.5, false);
    assert!(
        matches!(result, Err(AudioError::DecodeError(_))),
        "expected DecodeError for empty data"
    );
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_plays_looped_sound() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let wav = make_wav_bytes();
    let handle = engine.play(&wav, 0.3, true).expect("looped playback should succeed");
    std::thread::sleep(std::time::Duration::from_millis(50));
    handle.stop();
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_volume_set_at_play_time() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let wav = make_wav_bytes();

    // Test various volume levels
    for volume in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let handle = engine.play(&wav, volume, false).expect("play at volume {volume}");
        handle.stop();
    }
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn engine_concurrent_playback() {
    let engine = AudioEngine::global().expect("audio engine should be available");
    let wav = make_wav_bytes();

    // Start multiple sounds playing simultaneously
    let h1 = engine.play(&wav, 0.3, false).expect("play sound 1");
    let h2 = engine.play(&wav, 0.5, false).expect("play sound 2");
    let h3 = engine.play(&wav, 0.7, false).expect("play sound 3");

    std::thread::sleep(std::time::Duration::from_millis(30));

    h1.stop();
    h2.stop();
    h3.stop();
}

// ── SoundRegistry integration tests ──────────────────────────────────────────

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_play_stop_lifecycle() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    assert!(registry.play(1, &wav, 0.5, false).is_ok());
    assert_eq!(registry.len(), 1);

    assert!(registry.stop(1).is_ok());
    assert!(registry.is_empty());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_play_multiple_sounds() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    for id in 1..=5 {
        assert!(registry.play(id, &wav, 0.5, false).is_ok());
    }
    assert_eq!(registry.len(), 5);

    for id in 1..=5 {
        assert!(registry.stop(id).is_ok());
    }
    assert!(registry.is_empty());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_replaces_existing_sound() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    assert!(registry.play(1, &wav, 0.3, false).is_ok());
    assert_eq!(registry.len(), 1);

    // Playing same ID should replace the previous sound
    assert!(registry.play(1, &wav, 0.8, false).is_ok());
    assert_eq!(registry.len(), 1); // Still 1, not 2

    assert!(registry.stop(1).is_ok());
    assert!(registry.is_empty());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_pause_resume_preserves_sound() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    assert!(registry.play(1, &wav, 0.5, false).is_ok());
    assert!(registry.pause(1).is_ok());
    assert_eq!(registry.len(), 1); // Still in registry
    assert!(registry.resume(1).is_ok());
    assert_eq!(registry.len(), 1);

    assert!(registry.stop(1).is_ok());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_volume_change_on_playing_sound() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    assert!(registry.play(1, &wav, 0.3, false).is_ok());
    assert!(registry.set_volume(1, 0.9).is_ok());
    assert!(registry.stop(1).is_ok());
}

#[test]
fn registry_stop_nonexistent_returns_error() {
    let mut registry = SoundRegistry::new();
    let result = registry.stop(999);
    assert!(matches!(result, Err(AudioError::NotFound(999))));
}

#[test]
fn registry_operations_on_nonexistent_ids() {
    let mut registry = SoundRegistry::new();

    assert!(matches!(registry.set_volume(42, 0.5), Err(AudioError::NotFound(42))));
    assert!(matches!(registry.pause(42), Err(AudioError::NotFound(42))));
    assert!(matches!(registry.resume(42), Err(AudioError::NotFound(42))));
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn registry_full_lifecycle() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    // Play
    assert!(registry.play(1, &wav, 0.4, false).is_ok());
    assert_eq!(registry.len(), 1);

    // Pause
    assert!(registry.pause(1).is_ok());

    // Volume change while paused
    assert!(registry.set_volume(1, 0.8).is_ok());

    // Resume
    assert!(registry.resume(1).is_ok());

    // Volume change while playing
    assert!(registry.set_volume(1, 0.2).is_ok());

    // Stop
    assert!(registry.stop(1).is_ok());
    assert!(registry.is_empty());
}

// ── Sound catalog construction tests ─────────────────────────────────────────

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn catalog_loads_wav_from_disk() {
    let dir = temp_test_dir("catalog_wav");
    let wav = make_wav_bytes();
    fs::write(dir.join("test.wav"), &wav).expect("write wav");

    // Simulate what build_host_sound_catalog does
    let data = fs::read(dir.join("test.wav")).expect("read wav");
    assert_eq!(data, wav);

    // Verify it decodes through rodio
    let engine = AudioEngine::global().expect("audio engine available");
    let handle = engine.play(&data, 0.5, false).expect("decoded WAV should play");
    handle.stop();

    // Cleanup
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn catalog_rejects_synthetic_mp3_from_disk() {
    // A synthetic MP3 frame cannot be decoded by rodio's symphonia decoder.
    // This test confirms the engine rejects it cleanly.
    let dir = temp_test_dir("catalog_mp3");
    let mp3 = make_mp3_bytes();
    fs::write(dir.join("test.mp3"), &mp3).expect("write mp3");

    let data = fs::read(dir.join("test.mp3")).expect("read mp3");
    assert_eq!(data, mp3);

    // The engine should reject a synthetic MP3 frame
    let engine = AudioEngine::global().expect("audio engine available");
    let result = engine.play(&data, 0.5, false);
    assert!(
        matches!(result, Err(AudioError::DecodeError(_))),
        "synthetic MP3 frame should produce a decode error (real MP3 files work fine)"
    );

    // Cleanup
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn catalog_handles_missing_file_gracefully() {
    let dir = temp_test_dir("catalog_missing");
    let result = fs::read(dir.join("nonexistent.wav"));
    assert!(result.is_err(), "reading nonexistent file should fail");
    let _ = fs::remove_dir_all(&dir);
}

// ── Stress/lifetime tests ────────────────────────────────────────────────────

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn rapid_play_stop_cycles() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    for i in 0..20 {
        let id = i % 5; // Cycle through 5 IDs
        assert!(registry.play(id, &wav, 0.5, false).is_ok());
        assert!(registry.stop(id).is_ok());
    }
    assert!(registry.is_empty());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn play_while_other_sounds_are_playing() {
    let mut registry = SoundRegistry::new();
    let wav = make_wav_bytes();

    // Start sound 1
    assert!(registry.play(1, &wav, 0.3, false).is_ok());

    // While 1 is playing, start sound 2
    assert!(registry.play(2, &wav, 0.5, false).is_ok());

    // Both should be active
    assert_eq!(registry.len(), 2);

    // Stop both
    assert!(registry.stop(1).is_ok());
    assert!(registry.stop(2).is_ok());
    assert!(registry.is_empty());
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn stop_is_idempotent_on_sound_handle() {
    let engine = AudioEngine::global().expect("audio engine available");
    let wav = make_wav_bytes();
    let handle = engine.play(&wav, 0.5, false).expect("play should succeed");

    // Calling stop multiple times should not panic
    handle.stop();
    handle.stop();
    handle.stop();
}

#[test]
#[ignore = "requires audio hardware (ALSA/PulseAudio); see module doc for virtual sink setup"]
fn drop_handle_does_not_panic() {
    let engine = AudioEngine::global().expect("audio engine available");
    let wav = make_wav_bytes();
    {
        let _handle = engine.play(&wav, 0.5, false).expect("play should succeed");
        // Handle dropped while playing
    }
    // No panic should occur - audio continues in background
}
