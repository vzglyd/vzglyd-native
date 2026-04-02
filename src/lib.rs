//! VZGLYD Native Host
//!
//! This is the native (Linux/Raspberry Pi) host implementation for the VZGLYD display engine.
//! It integrates the platform-agnostic kernel with:
//! - winit for windowing and event handling
//! - wgpu for GPU rendering
//! - wasmtime for WASM slide instantiation
//! - std::fs for asset loading

pub mod app;
pub mod assets;
pub mod gpu;
pub mod render;
pub mod slide;
pub mod utils;
pub mod wasm;

pub use app::NativeApp;
pub use gpu::GpuContext;
pub use wasm::WasmRuntime;
