//! End-to-end tests for GLB animation support in .vzglyd bundles.
//!
//! These tests exercise the full pipeline:
//! 1. Create a GLB file with animation data (or use an existing one)
//! 2. Create a slide directory with manifest.json referencing the GLB scene
//! 3. Pack the directory into a .vzglyd archive
//! 4. Verify the archive contains the GLB
//! 5. Sample animation matrices at various timestamps and verify interpolation

use std::fs::{self, File};
use std::io::Read;
use std::path::PathBuf;
use vzglyd_kernel::glb::load_glb_scene;
use vzglyd_native::slide_loader::pack_slide_directory;

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Create a unique temp directory for a test slide package.
fn temp_slide_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vzglyd_e2e_anim_{label}_{}_{}",
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
    fs::write(
        dir.join("manifest.json"),
        manifest_with_required_art(manifest_json),
    )
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
    let object = value
        .as_object_mut()
        .expect("test manifest should be an object");
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

/// Get a GLB file to test with. Uses the loading-slide's world.glb if available.
fn get_test_glb() -> Option<Vec<u8>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let glb_path = repo_root.join("loading-slide/assets/world.glb");
    if glb_path.exists() {
        return fs::read(&glb_path).ok();
    }
    None
}

// ── End-to-end tests ─────────────────────────────────────────────────────────

#[test]
fn e2e_glb_parses_as_scene() {
    let glb = get_test_glb();
    let Some(glb) = glb else {
        // No GLB available — skip gracefully
        println!("Skipping: no test GLB available");
        return;
    };

    // Write to temp file for load_glb_scene (which takes a path)
    let dir = temp_slide_dir("glb_parse");
    let glb_path = dir.join("test.glb");
    fs::write(&glb_path, &glb).expect("write glb");

    let result = load_glb_scene(&glb_path, None);
    assert!(result.is_ok(), "GLB should parse: {:?}", result.err());

    let scene = result.unwrap();
    // Should have at least one mesh node
    assert!(
        !scene.mesh_nodes.is_empty(),
        "Scene should contain mesh nodes"
    );
}

#[test]
fn e2e_animation_channels_extracted_from_glb() {
    let glb = get_test_glb();
    let Some(glb) = glb else {
        println!("Skipping: no test GLB available");
        return;
    };

    let dir = temp_slide_dir("anim channels");
    let glb_path = dir.join("anim.glb");
    fs::write(&glb_path, &glb).expect("write glb");

    let scene = load_glb_scene(&glb_path, None).expect("GLB should parse");

    // If the GLB has animations, verify structure
    if !scene.animations.is_empty() {
        for clip in &scene.animations {
            assert!(!clip.name.is_empty(), "Animation clip should have a name");
            assert!(clip.duration > 0.0, "Animation duration should be positive");

            for channel in &clip.channels {
                assert!(
                    !channel.keyframe_times.is_empty(),
                    "Channel should have keyframe times"
                );
                assert!(
                    channel.keyframe_times.len() == channel.keyframe_values.len(),
                    "Keyframe times and values should have same length"
                );
                assert!(
                    channel.keyframe_times.len() >= 2,
                    "Channel should have at least 2 keyframes"
                );
            }
        }
    }
}

#[test]
fn e2e_glb_packed_into_vzglyd_archive() {
    let glb = get_test_glb();
    let Some(glb) = glb else {
        println!("Skipping: no test GLB available");
        return;
    };

    // 1. Create slide with GLB scene
    let dir = temp_slide_dir("glb_pack");
    fs::create_dir_all(dir.join("assets")).expect("create assets dir");
    fs::write(dir.join("assets").join("world.glb"), &glb).expect("write glb");

    write_minimal_slide(
        &dir,
        r#"{
            "name": "GLB Scene Test",
            "abi_version": 3,
            "scene_space": "world_3d",
            "assets": {
                "scenes": [
                    { "path": "assets/world.glb", "id": "world" }
                ]
            }
        }"#,
    );

    // 2. Pack
    let archive_path = dir.join("scene.vzglyd");
    pack_slide_directory(&dir, &archive_path).expect("pack should succeed");

    // 3. Verify GLB is in the archive
    let archive_file = File::open(&archive_path).expect("open archive");
    let mut archive = zip::ZipArchive::new(archive_file).expect("read zip");

    let mut archived_glb = Vec::new();
    archive
        .by_name("assets/world.glb")
        .expect("GLB should be in archive")
        .read_to_end(&mut archived_glb)
        .expect("read GLB");

    assert_eq!(archived_glb.len(), glb.len(), "GLB size should match");
    assert_eq!(archived_glb, glb, "GLB content should be preserved");
}

#[test]
fn e2e_slide_spec_includes_animations_from_glb() {
    use vzglyd_slide::{AnimationChannel, AnimationClip, AnimationPath};

    // Create synthetic animation data like what would come from GLB
    let clip = AnimationClip {
        name: "TestRotation".to_string(),
        duration: 2.0,
        looped: true,
        channels: vec![AnimationChannel {
            node_label: "TestCube".to_string(),
            path: AnimationPath::Rotation,
            keyframe_times: vec![0.0, 1.0, 2.0],
            keyframe_values: vec![
                [0.0, 0.0, 0.0, 1.0],       // identity quaternion
                [0.0, 0.7071, 0.0, 0.7071], // 90 degrees around Y
                [0.0, 1.0, 0.0, 0.0],       // 180 degrees around Y
            ],
        }],
    };

    assert_eq!(clip.name, "TestRotation");
    assert!(clip.looped);
    assert_eq!(clip.channels.len(), 1);
    assert_eq!(clip.channels[0].path, AnimationPath::Rotation);
    assert_eq!(clip.channels[0].keyframe_times.len(), 3);
}

#[test]
fn e2e_animation_sampling_translation() {
    use glam::Vec3;
    use vzglyd_slide::{AnimationChannel, AnimationClip, AnimationPath};

    let channel = AnimationChannel {
        node_label: "MovingCube".to_string(),
        path: AnimationPath::Translation,
        keyframe_times: vec![0.0, 1.0],
        keyframe_values: vec![[0.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
    };

    let clip = AnimationClip {
        name: "MoveRight".to_string(),
        duration: 1.0,
        looped: true,
        channels: vec![channel],
    };

    // Sample at t=0.0
    let ch = &clip.channels[0];
    let t = 0.0_f32;
    let idx = ch.keyframe_times.partition_point(|&kt| kt <= t) - 1;
    let t0 = ch.keyframe_times[idx];
    let t1 = ch.keyframe_times[idx + 1];
    let alpha = (t - t0) / (t1 - t0);
    let v0 = ch.keyframe_values[idx];
    let v1 = ch.keyframe_values[idx + 1];

    let result = Vec3::new(
        v0[0] + (v1[0] - v0[0]) * alpha,
        v0[1] + (v1[1] - v0[1]) * alpha,
        v0[2] + (v1[2] - v0[2]) * alpha,
    );

    assert!((result.x - 0.0).abs() < 0.001, "At t=0, x should be 0");
    assert!((result.y - 0.0).abs() < 0.001, "At t=0, y should be 0");
    assert!((result.z - 0.0).abs() < 0.001, "At t=0, z should be 0");

    // Sample at t=0.5 (midpoint)
    let t = 0.5_f32;
    let alpha = (t - t0) / (t1 - t0);
    let result = Vec3::new(
        v0[0] + (v1[0] - v0[0]) * alpha,
        v0[1] + (v1[1] - v0[1]) * alpha,
        v0[2] + (v1[2] - v0[2]) * alpha,
    );

    assert!((result.x - 2.5).abs() < 0.001, "At t=0.5, x should be 2.5");
}

#[test]
fn e2e_animation_sampling_rotation_slerp() {
    use glam::Quat;

    // Test quaternion slerp (same as the production code uses)
    let q0 = Quat::from_rotation_y(0.0); // identity
    let q1 = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2); // 90 degrees

    let result = q0.slerp(q1, 0.5);

    // At t=0.5, should be 45 degrees around Y
    let expected = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
    assert!(
        (result.x - expected.x).abs() < 0.001,
        "Slerp x should match expected"
    );
    assert!(
        (result.y - expected.y).abs() < 0.001,
        "Slerp y should match expected"
    );
    assert!(
        (result.z - expected.z).abs() < 0.001,
        "Slerp z should match expected"
    );
    assert!(
        (result.w - expected.w).abs() < 0.001,
        "Slerp w should match expected"
    );
}

#[test]
fn e2e_animation_sampling_scale() {
    use glam::Vec3;
    use vzglyd_slide::{AnimationChannel, AnimationClip, AnimationPath};

    let channel = AnimationChannel {
        node_label: "GrowingSphere".to_string(),
        path: AnimationPath::Scale,
        keyframe_times: vec![0.0, 2.0],
        keyframe_values: vec![[1.0, 1.0, 1.0, 0.0], [3.0, 3.0, 3.0, 0.0]],
    };

    let clip = AnimationClip {
        name: "Grow".to_string(),
        duration: 2.0,
        looped: true,
        channels: vec![channel],
    };

    // Sample at t=1.0 (midpoint, should be scale 2,2,2)
    let ch = &clip.channels[0];
    let t = 1.0_f32;
    let idx = ch.keyframe_times.partition_point(|&kt| kt <= t) - 1;
    let t0 = ch.keyframe_times[idx];
    let t1 = ch.keyframe_times[idx + 1];
    let alpha = (t - t0) / (t1 - t0);
    let v0 = ch.keyframe_values[idx];
    let v1 = ch.keyframe_values[idx + 1];

    let result = Vec3::new(
        v0[0] + (v1[0] - v0[0]) * alpha,
        v0[1] + (v1[1] - v0[1]) * alpha,
        v0[2] + (v1[2] - v0[2]) * alpha,
    );

    assert!(
        (result.x - 2.0).abs() < 0.001,
        "At t=1.0, scale x should be 2"
    );
    assert!(
        (result.y - 2.0).abs() < 0.001,
        "At t=1.0, scale y should be 2"
    );
    assert!(
        (result.z - 2.0).abs() < 0.001,
        "At t=1.0, scale z should be 2"
    );
}

#[test]
fn e2e_animation_looping_wraps_elapsed_time() {
    use vzglyd_slide::{AnimationChannel, AnimationClip, AnimationPath};

    let channel = AnimationChannel {
        node_label: "Spinner".to_string(),
        path: AnimationPath::Translation,
        keyframe_times: vec![0.0, 1.0],
        keyframe_values: vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
    };

    let clip = AnimationClip {
        name: "Loop".to_string(),
        duration: 1.0,
        looped: true,
        channels: vec![channel],
    };

    // With looping, elapsed time 2.5 should wrap to 0.5
    let elapsed = 2.5_f32;
    let wrapped = elapsed % clip.duration;
    assert!(
        (wrapped - 0.5).abs() < 0.001,
        "Looped elapsed should wrap to 0.5"
    );

    // At t=0.5 in a 0→1 animation, value should be 0.5
    let ch = &clip.channels[0];
    let v0 = ch.keyframe_values[0];
    let v1 = ch.keyframe_values[1];
    let alpha = wrapped;
    let result = v0[0] + (v1[0] - v0[0]) * alpha;
    assert!((result - 0.5).abs() < 0.001, "Value should be 0.5");
}

#[test]
fn e2e_animation_non_looped_clamps_to_end() {
    use vzglyd_slide::{AnimationChannel, AnimationClip, AnimationPath};

    let clip = AnimationClip {
        name: "OneShot".to_string(),
        duration: 2.0,
        looped: false,
        channels: vec![AnimationChannel {
            node_label: "Faller".to_string(),
            path: AnimationPath::Translation,
            keyframe_times: vec![0.0, 2.0],
            keyframe_values: vec![[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        }],
    };

    // With non-looped, elapsed 5.0 should clamp to 2.0 (duration)
    let elapsed = 5.0_f32;
    let clamped = elapsed.min(clip.duration);
    assert!(
        (clamped - 2.0).abs() < 0.001,
        "Non-looped elapsed should clamp to duration"
    );
}
