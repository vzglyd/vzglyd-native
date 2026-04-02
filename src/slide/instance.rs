//! WASM slide instance management.
//!
//! Handles instantiation, spec reading, and per-frame updates.

use log::{debug, info, warn};
use std::time::SystemTime;
use vzglyd_slide::ABI_VERSION as SLIDE_ABI_VERSION;
use wasmtime::{Instance, Linker, Memory, Module, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

/// ABI version understood by packaged `lume` slides.
pub const ABI_VERSION: u32 = SLIDE_ABI_VERSION;

/// WASI state wrapper.
pub struct WasiState {
    pub wasi: WasiCtx,
    pub start_time: f64,
}

/// Slide instance holding the WASM instance and state.
pub struct SlideInstance {
    store: Store<WasiState>,
    instance: Instance,
    memory: Memory,
}

impl SlideInstance {
    /// Creates a new slide instance from a compiled module.
    pub fn new(module: &Module) -> Result<Self, String> {
        // Debug: log expected imports
        log::info!("WASM module expects {} imports:", module.imports().len());
        for (i, imp) in module.imports().enumerate() {
            log::info!("  {}. {}::{}", i, imp.module(), imp.name());
        }

        let start_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        // Build WASI context
        let wasi = WasiCtxBuilder::new()
            .inherit_stdout()
            .inherit_stderr()
            .build();

        let state = WasiState { wasi, start_time };

        let mut store: Store<WasiState> = Store::new(module.engine(), state);
        store.set_epoch_deadline(1);

        // Create linker and add WASI
        let mut linker = Linker::new(module.engine());
        wasmtime_wasi::add_to_linker(&mut linker, |s: &mut WasiState| &mut s.wasi)
            .map_err(|e| format!("Failed to add WASI to linker: {}", e))?;

        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|e| format!("Failed to instantiate WASM: {}", e))?;

        // WASI command-style modules may initialise their spec buffer in `_start`.
        if let Ok(start_fn) = instance.get_typed_func::<(), ()>(&mut store, "_start") {
            match start_fn.call(&mut store, ()) {
                Ok(()) => {}
                Err(error) => {
                    let clean_exit = error
                        .downcast_ref::<wasmtime_wasi::I32Exit>()
                        .is_some_and(|exit| exit.0 == 0);
                    if !clean_exit {
                        return Err(format!("Failed to run _start: {}", error));
                    }
                }
            }
        }

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| "WASM module missing 'memory' export".to_string())?;

        let abi_version_fn = instance
            .get_typed_func::<(), u32>(&mut store, "vzglyd_abi_version")
            .map_err(|_| "Missing vzglyd_abi_version export".to_string())?;
        let version = abi_version_fn
            .call(&mut store, ())
            .map_err(|e| format!("Failed to call vzglyd_abi_version: {}", e))?;
        if version != ABI_VERSION {
            return Err(format!(
                "ABI version mismatch: expected {}, got {}",
                ABI_VERSION, version
            ));
        }
        info!("Slide ABI version: {}", version);

        // Run vzglyd_init if exported (this sets up the slide state)
        if let Ok(init_fn) = instance.get_typed_func::<(), i32>(&mut store, "vzglyd_init") {
            match init_fn.call(&mut store, ()) {
                Ok(status) => {
                    debug!("Slide vzglyd_init -> {}", status);
                }
                Err(error) => {
                    return Err(format!("Failed to call vzglyd_init: {}", error));
                }
            }
        }

        Ok(Self {
            store,
            instance,
            memory,
        })
    }

    /// Reads the slide spec bytes from WASM memory.
    pub fn read_spec_bytes(&mut self) -> Result<Vec<u8>, String> {
        let ptr_fn = self
            .instance
            .get_typed_func::<(), i32>(&mut self.store, "vzglyd_spec_ptr")
            .map_err(|_| "Missing vzglyd_spec_ptr export".to_string())?;
        let len_fn = self
            .instance
            .get_typed_func::<(), i32>(&mut self.store, "vzglyd_spec_len")
            .map_err(|_| "Missing vzglyd_spec_len export".to_string())?;

        let ptr = ptr_fn
            .call(&mut self.store, ())
            .map_err(|e| format!("Failed to call vzglyd_spec_ptr: {}", e))?
            as usize;

        let len = len_fn
            .call(&mut self.store, ())
            .map_err(|e| format!("Failed to call vzglyd_spec_len: {}", e))?
            as usize;

        if len == 0 {
            return Err("vzglyd_spec_len returned 0".to_string());
        }

        let end = ptr
            .checked_add(len)
            .ok_or_else(|| format!("Spec pointer/length overflow: {}+{}", ptr, len))?;
        let memory_view = self.memory.data(&self.store);

        if end > memory_view.len() {
            return Err(format!(
                "Spec pointer/length out of bounds: {}+{}",
                ptr, len
            ));
        }

        Ok(memory_view[ptr..end].to_vec())
    }

    /// Calls vzglyd_update(dt) and returns whether the slide changed.
    pub fn update(&mut self, dt: f32) -> bool {
        if let Some(update_fn) = self.instance.get_func(&mut self.store, "vzglyd_update") {
            match update_fn.typed::<f32, i32>(&self.store) {
                Ok(typed_fn) => match typed_fn.call(&mut self.store, dt) {
                    Ok(val) => return val != 0,
                    Err(e) => debug!("vzglyd_update error: {}", e),
                },
                Err(e) => debug!("vzglyd_update type error: {}", e),
            }
        }
        false
    }

    /// Reads dynamic mesh bytes if available.
    pub fn read_dynamic_mesh_bytes(&mut self) -> Option<Vec<u8>> {
        let ptr_fn = self
            .instance
            .get_func(&mut self.store, "vzglyd_dynamic_meshes_ptr")?;
        let len_fn = self
            .instance
            .get_func(&mut self.store, "vzglyd_dynamic_meshes_len")?;

        let ptr = ptr_fn
            .typed::<(), i32>(&self.store)
            .ok()?
            .call(&mut self.store, ())
            .ok()? as u32;
        let len = len_fn
            .typed::<(), i32>(&self.store)
            .ok()?
            .call(&mut self.store, ())
            .ok()? as u32;

        if len == 0 {
            return None;
        }

        let memory_view = self.memory.data(&mut self.store);
        let start = ptr as usize;
        let end = start + len as usize;

        if end > memory_view.len() {
            warn!("Dynamic mesh pointer/length out of bounds");
            return None;
        }

        Some(memory_view[start..end].to_vec())
    }

    /// Reads overlay bytes if available.
    pub fn read_overlay_bytes(&mut self) -> Option<Vec<u8>> {
        let ptr_fn = self
            .instance
            .get_func(&mut self.store, "vzglyd_overlay_ptr")?;
        let len_fn = self
            .instance
            .get_func(&mut self.store, "vzglyd_overlay_len")?;

        let ptr = ptr_fn
            .typed::<(), i32>(&self.store)
            .ok()?
            .call(&mut self.store, ())
            .ok()? as u32;
        let len = len_fn
            .typed::<(), i32>(&self.store)
            .ok()?
            .call(&mut self.store, ())
            .ok()? as u32;

        if len == 0 {
            return None;
        }

        let memory_view = self.memory.data(&mut self.store);
        let start = ptr as usize;
        let end = start + len as usize;

        if end > memory_view.len() {
            warn!("Overlay pointer/length out of bounds");
            return None;
        }

        Some(memory_view[start..end].to_vec())
    }

    /// Returns a reference to the store.
    pub fn store(&self) -> &Store<WasiState> {
        &self.store
    }

    /// Returns a mutable reference to the store.
    pub fn store_mut(&mut self) -> &mut Store<WasiState> {
        &mut self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use vzglyd_slide::WorldVertex;

    #[test]
    fn packaged_clock_slide_spec_can_be_read() {
        let archive_path = Path::new("slides/clock.vzglyd");
        if !archive_path.exists() {
            return;
        }

        let extracted = crate::assets::archive::extract_archive(archive_path).expect("extract");
        let runtime = crate::wasm::WasmRuntime::new().expect("runtime");
        let module_path = extracted.path.join("slide.wasm");
        let module = Module::from_file(&runtime.engine, &module_path).expect("module");

        let mut instance = SlideInstance::new(&module).expect("instance");
        let bytes = instance.read_spec_bytes().expect("spec bytes");
        assert_eq!(
            bytes.first().copied(),
            Some(crate::slide::wire::WIRE_VERSION)
        );
        let spec =
            crate::slide::wire::deserialize_spec::<WorldVertex>(&bytes).expect("decode spec");
        assert_eq!(spec.name, "clock_world");
    }
}
