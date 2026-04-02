//! Slide module for WASM slide instantiation and rendering.

pub mod instance;
pub mod wire;

pub use instance::SlideInstance;
pub use wire::{DecodedSlideSpec, decode_slide_spec};
