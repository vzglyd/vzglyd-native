//! WASM runtime for slide instantiation.
//!
//! Handles wasmtime engine, store, and host function exports.

use std::path::Path;
use wasmtime::{Config, Engine, Module};

/// WASM runtime managing the wasmtime engine.
pub struct WasmRuntime {
    pub engine: Engine,
}

impl WasmRuntime {
    /// Creates a new WASM runtime.
    pub fn new() -> Result<Self, String> {
        let mut config = Config::new();
        config.epoch_interruption(true);

        let engine =
            Engine::new(&config).map_err(|e| format!("Failed to create wasmtime engine: {}", e))?;

        Ok(Self { engine })
    }

    /// Compiles a WASM module from a file path.
    pub fn compile(&self, path: &str) -> Result<Module, String> {
        let path = Path::new(path);
        Module::from_file(&self.engine, path).map_err(|e| {
            format!(
                "Failed to compile WASM module from '{}': {}",
                path.display(),
                e
            )
        })
    }

    /// Compiles a WASM module from bytes.
    pub fn compile_bytes(&self, wasm_bytes: &[u8]) -> Result<Module, String> {
        Module::from_binary(&self.engine, wasm_bytes)
            .map_err(|e| format!("Failed to compile WASM module: {}", e))
    }
}

impl Default for WasmRuntime {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
