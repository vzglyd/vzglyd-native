//! ABI 1 slide wire-format decoding.

use bytemuck::Pod;
use serde::Serialize;
use serde::de::DeserializeOwned;
use vzglyd_slide::{SceneSpace, ScreenVertex, SlideSpec, WorldVertex};

/// Current wire-format version used by packaged `lume` slides.
pub const WIRE_VERSION: u8 = 1;

/// Typed slide spec decoded from the ABI 1 wire blob.
pub enum DecodedSlideSpec {
    Screen(SlideSpec<ScreenVertex>),
    World(SlideSpec<WorldVertex>),
}

/// Errors that can occur while decoding a versioned slide spec blob.
#[derive(Debug)]
pub enum WireError {
    MissingVersion,
    UnsupportedVersion(u8),
    Deserialize(postcard::Error),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVersion => write!(f, "missing version byte in slide wire format"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported slide wire version {version}")
            }
            Self::Deserialize(error) => write!(f, "failed to decode slide spec: {error}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Deserialize a versioned ABI 1 wire blob into a typed `SlideSpec`.
pub fn deserialize_spec<V>(bytes: &[u8]) -> Result<SlideSpec<V>, WireError>
where
    V: DeserializeOwned + Serialize + Pod,
{
    let (version, payload) = bytes.split_first().ok_or(WireError::MissingVersion)?;
    if *version != WIRE_VERSION {
        return Err(WireError::UnsupportedVersion(*version));
    }

    postcard::from_bytes(payload).map_err(WireError::Deserialize)
}

/// Decode a versioned ABI 1 blob into the concrete slide type expected by the renderer.
pub fn decode_slide_spec(bytes: &[u8]) -> Result<DecodedSlideSpec, String> {
    let world_result = deserialize_spec::<WorldVertex>(bytes);
    if let Ok(spec) = &world_result {
        if spec.scene_space == SceneSpace::World3D {
            return Ok(DecodedSlideSpec::World(spec.clone()));
        }
    }

    let screen_result = deserialize_spec::<ScreenVertex>(bytes);
    if let Ok(spec) = &screen_result {
        if spec.scene_space == SceneSpace::Screen2D {
            return Ok(DecodedSlideSpec::Screen(spec.clone()));
        }
    }

    if let Ok(spec) = world_result {
        return Err(format!(
            "world slide spec declared unsupported scene space {:?}",
            spec.scene_space
        ));
    }

    if let Ok(spec) = screen_result {
        return Err(format!(
            "screen slide spec declared unsupported scene space {:?}",
            spec.scene_space
        ));
    }

    let world_error = world_result.expect_err("checked above");
    let screen_error = screen_result.expect_err("checked above");
    Err(format!(
        "failed to decode slide spec as world ({world_error}) or screen ({screen_error})"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vzglyd_slide::{Limits, SceneSpace};

    fn encode_spec<V>(spec: &SlideSpec<V>) -> Vec<u8>
    where
        V: DeserializeOwned + Serialize + Pod,
    {
        let mut bytes = vec![WIRE_VERSION];
        bytes.extend(postcard::to_stdvec(spec).expect("serialize spec"));
        bytes
    }

    fn minimal_world_spec() -> SlideSpec<WorldVertex> {
        SlideSpec {
            name: "world".into(),
            limits: Limits::pi4(),
            scene_space: SceneSpace::World3D,
            camera_path: None,
            shaders: None,
            overlay: None,
            font: None,
            textures_used: 0,
            textures: vec![],
            static_meshes: vec![],
            dynamic_meshes: vec![],
            draws: vec![],
            lighting: None,
        }
    }

    fn minimal_screen_spec() -> SlideSpec<ScreenVertex> {
        SlideSpec {
            name: "screen".into(),
            limits: Limits::pi4(),
            scene_space: SceneSpace::Screen2D,
            camera_path: None,
            shaders: None,
            overlay: None,
            font: None,
            textures_used: 0,
            textures: vec![],
            static_meshes: vec![],
            dynamic_meshes: vec![],
            draws: vec![],
            lighting: None,
        }
    }

    #[test]
    fn deserialize_world_wire_blob() {
        let spec = minimal_world_spec();
        let decoded = deserialize_spec::<WorldVertex>(&encode_spec(&spec)).expect("decode");
        assert_eq!(decoded.name, spec.name);
        assert_eq!(decoded.scene_space, SceneSpace::World3D);
    }

    #[test]
    fn decode_world_and_screen_specs() {
        let world = decode_slide_spec(&encode_spec(&minimal_world_spec())).expect("world decode");
        assert!(matches!(world, DecodedSlideSpec::World(_)));

        let screen =
            decode_slide_spec(&encode_spec(&minimal_screen_spec())).expect("screen decode");
        assert!(matches!(screen, DecodedSlideSpec::Screen(_)));
    }

    #[test]
    fn deserialize_rejects_missing_version_byte() {
        let error = deserialize_spec::<WorldVertex>(&[]).expect_err("missing version");
        assert!(matches!(error, WireError::MissingVersion));
    }

    #[test]
    fn deserialize_rejects_unsupported_version() {
        let error = deserialize_spec::<WorldVertex>(&[WIRE_VERSION + 1]).expect_err("unsupported");
        assert!(
            matches!(error, WireError::UnsupportedVersion(version) if version == WIRE_VERSION + 1)
        );
    }
}
