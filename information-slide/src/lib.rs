//! Information slide — shown when manual intervention is required.
//!
//! This is a minimal screen-space slide with a dark background. The actual
//! text (error message, management URL, etc.) is rendered by the host's
//! overlay system, which injects the message from the kernel's InfoState.
//!
//! The slide accepts JSON parameters with optional custom message lines,
//! but normally the host drives the content via the kernel's InfoReason.

use std::ops::Range;
use std::sync::LazyLock;

use lume_slide::{
    DrawSource, DrawSpec, Limits, PipelineKind, ScreenVertex, SceneSpace, SlideSpec, StaticMesh,
};

const WIRE_VERSION: u8 = 1;

/// Full-screen quad vertices (dark background).
fn background_vertices() -> Vec<ScreenVertex> {
    // Two triangles forming a full-screen quad.
    // Coordinates are in screen space (0..320, 0..240).
    vec![
        // Triangle 1
        ScreenVertex {
            position: [0.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
        ScreenVertex {
            position: [320.0, 0.0, 0.0],
            tex_coords: [1.0, 0.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
        ScreenVertex {
            position: [0.0, 240.0, 0.0],
            tex_coords: [0.0, 1.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
        // Triangle 2
        ScreenVertex {
            position: [320.0, 0.0, 0.0],
            tex_coords: [1.0, 0.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
        ScreenVertex {
            position: [320.0, 240.0, 0.0],
            tex_coords: [1.0, 1.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
        ScreenVertex {
            position: [0.0, 240.0, 0.0],
            tex_coords: [0.0, 1.0],
            color: [0.08, 0.08, 0.12, 1.0],
            mode: 0.0,
        },
    ]
}

fn info_slide_spec() -> SlideSpec<ScreenVertex> {
    let index_count = 6u32;
    SlideSpec {
        name: "information_slide".into(),
        limits: Limits::pi4(),
        scene_space: SceneSpace::Screen2D,
        camera_path: None,
        shaders: None,
        overlay: None,
        font: None,
        textures_used: 0,
        textures: vec![],
        sounds: vec![],
        animations: vec![],
        static_meshes: vec![StaticMesh {
            label: "bg_quad".into(),
            vertices: background_vertices(),
            indices: (0..6).collect(),
        }],
        dynamic_meshes: vec![],
        draws: vec![DrawSpec {
            label: "bg_draw".into(),
            source: DrawSource::Static(0),
            pipeline: PipelineKind::Opaque,
            index_range: Range { start: 0, end: index_count },
        }],
        lighting: None,
    }
}

static SPEC_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let mut bytes = vec![WIRE_VERSION];
    bytes.extend(postcard::to_stdvec(&info_slide_spec()).expect("serialize info slide spec"));
    bytes
});

pub fn serialized_spec() -> &'static [u8] {
    &SPEC_BYTES
}

#[cfg(target_arch = "wasm32")]
lume_slide::params_buf!(64);

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_configure(_len: i32) -> i32 {
    // Information slide has no configurable parameters.
    0
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_spec_ptr() -> *const u8 {
    SPEC_BYTES.as_ptr()
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_spec_len() -> u32 {
    SPEC_BYTES.len() as u32
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_abi_version() -> u32 {
    3
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_init() -> i32 {
    0
}

#[cfg(target_arch = "wasm32")]
#[unsafe(no_mangle)]
pub extern "C" fn vzglyd_update(_dt: f32) -> i32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_valid() {
        info_slide_spec().validate().unwrap();
    }

    #[test]
    fn info_slide_is_minimal_screen() {
        let spec = info_slide_spec();
        assert_eq!(spec.scene_space, SceneSpace::Screen2D);
        assert_eq!(spec.static_meshes.len(), 1);
        assert_eq!(spec.draws.len(), 1);
        assert!(spec.overlay.is_none());
        assert!(spec.textures.is_empty());
        assert!(spec.lighting.is_none());
    }
}
