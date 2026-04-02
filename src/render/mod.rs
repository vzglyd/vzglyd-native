//! Rendering module for slides and transitions.

pub(crate) mod shader_contract;
pub mod slide;
pub mod transition;

pub use slide::{
    DynamicMeshBuffers, MeshBuffers, ScreenBindGroup, ScreenSlideRenderer, ScreenUniforms,
    SlidePipelines, SlideRenderer, SlideTexture, WorldBindGroup, WorldSlideRenderer, WorldUniforms,
    create_slide_renderer,
};

pub use transition::{
    ActiveTransition, TransitionKind, TransitionRenderer, TransitionState, TransitionUniforms,
};
