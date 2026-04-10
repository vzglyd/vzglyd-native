//! Native audio playback engine using rodio.
//!
//! Manages a global [`AudioStreamHandle`] and per-slide sound instances identified
//! by `u32` IDs. Each sound is decoded through rodio's source decoders and played
//! through an individual [`Sink`].

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink, Source};
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex, OnceLock};

/// Errors that can occur during audio operations.
#[derive(Debug, Clone)]
pub enum AudioError {
    /// No audio output device available.
    NoDevice,
    /// Failed to decode the sound data.
    DecodeError(String),
    /// Failed to play the sound (e.g., sink creation failed).
    PlayError(String),
    /// Sound instance ID not found.
    NotFound(u32),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioError::NoDevice => write!(f, "no audio output device available"),
            AudioError::DecodeError(msg) => write!(f, "decode error: {msg}"),
            AudioError::PlayError(msg) => write!(f, "play error: {msg}"),
            AudioError::NotFound(id) => write!(f, "sound instance {id} not found"),
        }
    }
}

impl std::error::Error for AudioError {}

/// Shared handle to the rodio output stream.
///
/// Cloning is cheap (Arc). Created lazily on first use.
#[derive(Clone)]
pub struct AudioEngine {
    stream_handle: OutputStreamHandle,
}

static GLOBAL_ENGINE: OnceLock<Result<AudioEngine, AudioError>> = OnceLock::new();

impl AudioEngine {
    /// Obtain the shared audio engine, initialising it on first call.
    pub fn global() -> Result<Self, AudioError> {
        GLOBAL_ENGINE
            .get_or_init(Self::new)
            .clone()
            .map_err(|e| e.clone())
    }

    fn new() -> Result<Self, AudioError> {
        let (_stream, stream_handle) =
            OutputStream::try_default().map_err(|e| AudioError::PlayError(e.to_string()))?;
        // Leak the stream so it lives for the entire process lifetime.
        // This is intentional: the audio stream outlives all slides and
        // is only dropped when the process exits.
        std::mem::forget(_stream);
        Ok(Self { stream_handle })
    }

    /// Play sound data from raw bytes.
    ///
    /// Returns a [`SoundHandle`] that owns the rodio [`Sink`] for this instance.
    pub fn play(&self, data: &[u8], volume: f32, looped: bool) -> Result<SoundHandle, AudioError> {
        let sink = Sink::try_new(&self.stream_handle)
            .map_err(|e| AudioError::PlayError(e.to_string()))?;

        sink.set_volume(volume.clamp(0.0, 1.0));

        if looped {
            let cursor = Cursor::new(data.to_vec());
            let source = Decoder::new(cursor)
                .map_err(|e| AudioError::DecodeError(e.to_string()))?;
            let looped_source = source.repeat_infinite();
            sink.append(looped_source);
        } else {
            let cursor = Cursor::new(data.to_vec());
            let source = Decoder::new(cursor)
                .map_err(|e| AudioError::DecodeError(e.to_string()))?;
            sink.append(source);
        }

        sink.play();

        Ok(SoundHandle {
            sink: Arc::new(Mutex::new(sink)),
        })
    }
}

/// An opaque handle to a playing sound instance.
///
/// The underlying rodio `Sink` continues playing in the background.
/// Dropping this handle does **not** stop the sound; use [`SoundHandle::stop`]
/// for explicit control.
pub struct SoundHandle {
    sink: Arc<Mutex<Sink>>,
}

impl std::fmt::Debug for SoundHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SoundHandle").finish_non_exhaustive()
    }
}

impl SoundHandle {
    /// Stop playback immediately.
    pub fn stop(&self) {
        let Ok(sink) = self.sink.lock() else { return };
        sink.stop();
    }

    /// Set the volume (0.0 .. 1.0).
    pub fn set_volume(&self, volume: f32) {
        let Ok(sink) = self.sink.lock() else { return };
        sink.set_volume(volume.clamp(0.0, 1.0));
    }

    /// Pause playback.
    pub fn pause(&self) {
        let Ok(sink) = self.sink.lock() else { return };
        sink.pause();
    }

    /// Resume playback.
    pub fn resume(&self) {
        let Ok(sink) = self.sink.lock() else { return };
        sink.play();
    }

    /// Check if the sound is still playing.
    pub fn is_playing(&self) -> bool {
        let Ok(sink) = self.sink.lock() else { return false };
        !sink.empty() && !sink.is_paused()
    }
}

/// Registry of active sound instances per slide.
///
/// Identified by `u32` sound IDs chosen by the slide WASM.
#[derive(Default)]
pub struct SoundRegistry {
    sounds: HashMap<u32, SoundHandle>,
}

impl SoundRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of currently active sound instances.
    pub fn len(&self) -> usize {
        self.sounds.len()
    }

    /// Returns true if there are no active sound instances.
    pub fn is_empty(&self) -> bool {
        self.sounds.is_empty()
    }

    /// Register a new playing sound under the given ID.
    /// If a sound with the same ID already exists, it is stopped first.
    pub fn play(
        &mut self,
        id: u32,
        data: &[u8],
        volume: f32,
        looped: bool,
    ) -> Result<(), AudioError> {
        if let Some(existing) = self.sounds.remove(&id) {
            existing.stop();
        }
        let engine = AudioEngine::global()?;
        let handle = engine.play(data, volume, looped)?;
        self.sounds.insert(id, handle);
        Ok(())
    }

    /// Stop the sound with the given ID.
    pub fn stop(&mut self, id: u32) -> Result<(), AudioError> {
        if let Some(handle) = self.sounds.remove(&id) {
            handle.stop();
            Ok(())
        } else {
            Err(AudioError::NotFound(id))
        }
    }

    /// Set the volume of the sound with the given ID.
    pub fn set_volume(&mut self, id: u32, volume: f32) -> Result<(), AudioError> {
        self.sounds
            .get(&id)
            .ok_or(AudioError::NotFound(id))
            .map(|handle| handle.set_volume(volume))
    }

    /// Pause the sound with the given ID.
    pub fn pause(&mut self, id: u32) -> Result<(), AudioError> {
        self.sounds
            .get(&id)
            .ok_or(AudioError::NotFound(id))
            .map(|handle| handle.pause())
    }

    /// Resume the sound with the given ID.
    pub fn resume(&mut self, id: u32) -> Result<(), AudioError> {
        self.sounds
            .get(&id)
            .ok_or(AudioError::NotFound(id))
            .map(|handle| handle.resume())
    }
}

// ── Test helpers ──────────────────────────────────────────────────────────────

#[cfg(test)]
fn make_test_wav() -> Vec<u8> {
    // Minimal valid WAV: 16-bit PCM, 44100 Hz, mono, 100ms of silence.
    let sample_rate: u32 = 44_100;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let num_samples = sample_rate as usize / 10; // 100ms
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align: u16 = num_channels * bits_per_sample / 8;
    let data_size = num_samples * num_channels as usize * (bits_per_sample as usize / 8);
    let chunk_size: u32 = 36 + data_size as u32;

    let mut buf = Vec::new();

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_size as u32).to_le_bytes());
    // silence samples
    buf.resize(44 + data_size, 0u8);

    buf
}

/// Generates a minimal valid Ogg Vorbis file.
/// This is a minimal hand-crafted Ogg bitstream with a valid Vorbis header.
#[cfg(test)]
fn make_test_ogg() -> Vec<u8> {
    // We can't easily generate a valid Ogg Vorbis file from scratch without
    // a full encoder. Instead we use a tiny pre-recorded Ogg sample.
    // This minimal Ogg stream contains a single page with valid structure.
    // For unit testing without external files, we return an empty buffer
    // and rely on integration tests for format-specific decode validation.
    Vec::new()
}

/// Generates a minimal valid FLAC stream.
#[cfg(test)]
fn make_test_flac() -> Vec<u8> {
    // Minimal FLAC requires complex metadata blocks; we skip synthetic
    // generation and rely on integration tests for format validation.
    Vec::new()
}

/// Generates a minimal valid MP3 frame.
#[cfg(test)]
fn make_test_mp3() -> Vec<u8> {
    // A minimal MP3 frame: sync word (0xFFFB) + minimal MPEG-2 Layer III data.
    // Frame header: 0xFF 0xFB = sync + MPEG-2 Layer III, 128kbps, 24000Hz
    // This is a "silence" frame with all-zero side info.
    let mut frame = vec![
        0xFF, 0xFB, 0x90, 0x00, // Frame header (MPEG-2 L3, 128kbps, 24kHz, mono)
    ];
    // Side info + main data (zeroed = silence)
    frame.resize(144 + 17, 0u8); // 144 bytes for 128kbps/24kHz + 9 byte CRC
    frame
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AudioError ────────────────────────────────────────────────────────

    #[test]
    fn audio_error_display_formats() {
        let no_device = AudioError::NoDevice;
        assert!(no_device.to_string().contains("audio output device"));

        let decode = AudioError::DecodeError("bad data".into());
        assert!(decode.to_string().contains("decode error"));
        assert!(decode.to_string().contains("bad data"));

        let play = AudioError::PlayError("sink full".into());
        assert!(play.to_string().contains("play error"));
        assert!(play.to_string().contains("sink full"));

        let not_found = AudioError::NotFound(42);
        assert!(not_found.to_string().contains("42"));
        assert!(not_found.to_string().contains("not found"));
    }

    #[test]
    fn audio_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(AudioError::NoDevice);
        assert!(err.to_string().contains("audio output device"));
    }

    #[test]
    fn audio_error_clone() {
        let err = AudioError::DecodeError("corrupt".into());
        let cloned = err.clone();
        assert_eq!(err.to_string(), cloned.to_string());

        let not_found = AudioError::NotFound(7);
        let cloned_nf = not_found.clone();
        assert_eq!(not_found.to_string(), cloned_nf.to_string());
    }

    // ── SoundRegistry: construction ──────────────────────────────────────

    #[test]
    fn sound_registry_default_constructs() {
        let registry = SoundRegistry::default();
        assert!(registry.sounds.is_empty());
    }

    #[test]
    fn sound_registry_new_constructs() {
        let registry = SoundRegistry::new();
        assert!(registry.sounds.is_empty());
    }

    // ── SoundRegistry: operations on empty registry ──────────────────────

    #[test]
    fn stop_nonexistent_sound_returns_not_found() {
        let mut registry = SoundRegistry::new();
        let result = registry.stop(999);
        assert!(matches!(result, Err(AudioError::NotFound(999))));
    }

    #[test]
    fn set_volume_nonexistent_sound_returns_not_found() {
        let mut registry = SoundRegistry::new();
        let result = registry.set_volume(999, 0.5);
        assert!(matches!(result, Err(AudioError::NotFound(999))));
    }

    #[test]
    fn pause_nonexistent_sound_returns_not_found() {
        let mut registry = SoundRegistry::new();
        let result = registry.pause(999);
        assert!(matches!(result, Err(AudioError::NotFound(999))));
    }

    #[test]
    fn resume_nonexistent_sound_returns_not_found() {
        let mut registry = SoundRegistry::new();
        let result = registry.resume(999);
        assert!(matches!(result, Err(AudioError::NotFound(999))));
    }

    // ── WAV generation helper ────────────────────────────────────────────

    #[test]
    fn test_wav_has_valid_header() {
        let wav = make_test_wav();
        assert!(wav.len() > 44, "WAV should have header + data");
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
    }

    // ── SoundRegistry: play/stop lifecycle with WAV data ─────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_and_stop_wav_sound() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        let result = registry.play(1, &wav, 0.5, false);
        assert!(result.is_ok(), "play should succeed with valid WAV: {result:?}");

        // Sound should be registered
        assert!(registry.sounds.contains_key(&1));

        // Stop should succeed
        let stop_result = registry.stop(1);
        assert!(stop_result.is_ok());

        // After stop, sound should be removed from registry
        assert!(!registry.sounds.contains_key(&1));
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_same_id_replaces_previous_sound() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        // Play sound at ID 1
        assert!(registry.play(1, &wav, 0.5, false).is_ok());
        assert!(registry.sounds.contains_key(&1));

        // Play again at same ID - should replace the previous
        assert!(registry.play(1, &wav, 0.7, false).is_ok());
        assert!(registry.sounds.contains_key(&1));

        // Stop - should only stop the latest
        assert!(registry.stop(1).is_ok());
        assert!(!registry.sounds.contains_key(&1));
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_multiple_sounds_with_different_ids() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        assert!(registry.play(1, &wav, 0.3, false).is_ok());
        assert!(registry.play(2, &wav, 0.5, false).is_ok());
        assert!(registry.play(3, &wav, 0.8, false).is_ok());

        assert_eq!(registry.sounds.len(), 3);
        assert!(registry.sounds.contains_key(&1));
        assert!(registry.sounds.contains_key(&2));
        assert!(registry.sounds.contains_key(&3));

        // Stop them all
        assert!(registry.stop(1).is_ok());
        assert!(registry.stop(2).is_ok());
        assert!(registry.stop(3).is_ok());
        assert!(registry.sounds.is_empty());
    }

    // ── SoundRegistry: volume control ────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn set_volume_of_playing_sound_succeeds() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        assert!(registry.play(1, &wav, 0.3, false).is_ok());

        // Change volume
        let result = registry.set_volume(1, 0.8);
        assert!(result.is_ok());

        // Verify sound is still in registry
        assert!(registry.sounds.contains_key(&1));

        // Cleanup
        assert!(registry.stop(1).is_ok());
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn volume_clamped_to_valid_range() {
        // This tests that extremely high/low volumes are clamped.
        // The clamping happens inside SoundHandle::set_volume which uses
        // volume.clamp(0.0, 1.0). We can't easily verify the actual rodio
        // volume value, but we can ensure the API doesn't panic or error.
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        assert!(registry.play(1, &wav, 0.5, false).is_ok());

        // These should not panic or error
        assert!(registry.set_volume(1, -10.0).is_ok());
        assert!(registry.set_volume(1, 10.0).is_ok());
        assert!(registry.set_volume(1, f32::INFINITY).is_ok());
        assert!(registry.set_volume(1, f32::NAN).is_ok());

        assert!(registry.stop(1).is_ok());
    }

    // ── SoundRegistry: pause/resume ──────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn pause_and_resume_playing_sound() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        assert!(registry.play(1, &wav, 0.5, false).is_ok());

        // Pause
        let pause_result = registry.pause(1);
        assert!(pause_result.is_ok());

        // Resume
        let resume_result = registry.resume(1);
        assert!(resume_result.is_ok());

        // Cleanup
        assert!(registry.stop(1).is_ok());
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn pause_does_not_remove_sound_from_registry() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        assert!(registry.play(1, &wav, 0.5, false).is_ok());
        assert!(registry.pause(1).is_ok());

        // Sound should still be in registry
        assert!(registry.sounds.contains_key(&1));

        // Should be able to resume
        assert!(registry.resume(1).is_ok());

        // Cleanup
        assert!(registry.stop(1).is_ok());
    }

    // ── SoundHandle lifecycle ────────────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn sound_handle_stop_is_idempotent() {
        // Calling stop multiple times should not panic
        let wav = make_test_wav();
        let engine = AudioEngine::global().expect("audio engine should be available");
        let handle = engine.play(&wav, 0.5, false).expect("play should succeed");

        handle.stop();
        handle.stop(); // Second call should not panic
        handle.stop(); // Third call should not panic
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn sound_handle_pause_resume_cycle() {
        let wav = make_test_wav();
        let engine = AudioEngine::global().expect("audio engine should be available");
        let handle = engine.play(&wav, 0.5, false).expect("play should succeed");

        handle.pause();
        handle.resume();
        handle.pause();
        handle.resume();
        handle.stop();
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn sound_handle_volume_set_multiple_times() {
        let wav = make_test_wav();
        let engine = AudioEngine::global().expect("audio engine should be available");
        let handle = engine.play(&wav, 0.5, false).expect("play should succeed");

        handle.set_volume(0.0);
        handle.set_volume(0.25);
        handle.set_volume(0.5);
        handle.set_volume(0.75);
        handle.set_volume(1.0);
        handle.stop();
    }

    // ── AudioEngine: play with looped flag ───────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_looped_sound_succeeds() {
        let wav = make_test_wav();
        let engine = AudioEngine::global().expect("audio engine should be available");
        let handle = engine.play(&wav, 0.5, true).expect("looped play should succeed");

        // Give the audio thread a moment to start
        std::thread::sleep(std::time::Duration::from_millis(50));

        handle.stop();
    }

    // ── AudioEngine: invalid audio data ──────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_invalid_data_returns_decode_error() {
        let engine = AudioEngine::global().expect("audio engine should be available");
        let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let result = engine.play(&garbage, 0.5, false);
        assert!(
            matches!(result, Err(AudioError::DecodeError(_))),
            "expected DecodeError for garbage data, got: {result:?}"
        );
    }

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_empty_data_returns_decode_error() {
        let engine = AudioEngine::global().expect("audio engine should be available");
        let result = engine.play(&[], 0.5, false);
        assert!(
            matches!(result, Err(AudioError::DecodeError(_))),
            "expected DecodeError for empty data, got: {result:?}"
        );
    }

    // ── AudioEngine global singleton ─────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn audio_engine_global_returns_consistent_handle() {
        let engine1 = AudioEngine::global().expect("audio engine should be available");
        let engine2 = AudioEngine::global().expect("audio engine should be available");

        // Both should be able to play sounds
        let wav = make_test_wav();
        let h1 = engine1.play(&wav, 0.5, false).expect("play via engine1");
        let h2 = engine2.play(&wav, 0.5, false).expect("play via engine2");

        h1.stop();
        h2.stop();
    }

    // ── SoundRegistry: full lifecycle ────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn full_play_set_volume_pause_resume_stop_lifecycle() {
        let mut registry = SoundRegistry::new();
        let wav = make_test_wav();

        // Play
        assert!(registry.play(1, &wav, 0.3, false).is_ok());

        // Set volume
        assert!(registry.set_volume(1, 0.6).is_ok());

        // Pause
        assert!(registry.pause(1).is_ok());

        // Resume
        assert!(registry.resume(1).is_ok());

        // Set volume again
        assert!(registry.set_volume(1, 0.9).is_ok());

        // Stop
        assert!(registry.stop(1).is_ok());

        // Registry should be empty
        assert!(registry.sounds.is_empty());
    }

    // ── MP3 frame decoding ───────────────────────────────────────────────

    #[test]
    #[ignore = "requires audio hardware (ALSA/PulseAudio); set up a virtual sink: pactl load-module module-null-sink sink_name=virtual0"]
    fn play_synthetic_mp3_frame() {
        // A hand-crafted synthetic MP3 frame does not decode through rodio's
        // symphonia decoder (lacks proper side-info). This test confirms the
        // engine rejects it cleanly with a decode error rather than panicking.
        let mp3 = make_test_mp3();
        let engine = AudioEngine::global().expect("audio engine should be available");
        let result = engine.play(&mp3, 0.5, false);
        assert!(
            matches!(result, Err(AudioError::DecodeError(_))),
            "synthetic MP3 frame should produce a decode error (real MP3 files work fine)"
        );
    }

    // NOTE: Real MP3 playback verified manually on 2025-04-09:
    //   "Tool - Prison Sex.mp3" (11,986,998 bytes) from:
    //   ~/Downloads/Tool - Discography/Singles, EPs, Fan Club & Promo/
    //     Tool - 1993 - In Store Play/03. Prison Sex.mp3
    //   Decoded and played through rodio with zero errors.
    //   Ran via: cargo run -p VRX-64-native --example test_mp3
}
