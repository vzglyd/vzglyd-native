//! Native application and event loop.
//!
//! Integrates the kernel with winit event loop and wgpu rendering.

use std::io::{Cursor, Read};
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;
use vzglyd_kernel::TransitionKind as KernelTransitionKind;
use vzglyd_kernel::schedule::{build_schedule_from_playlist, parse_playlist};
use vzglyd_kernel::{Engine, EngineInput, FrameRenderState, Host, LogLevel, RenderCommand};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

#[cfg(target_os = "linux")]
use winit::platform::x11::{
    ActiveEventLoopExtX11, WindowAttributesExtX11, WindowType as X11WindowType,
};

use crate::assets::archive::{TempPackage, extract_archive};
use crate::gpu::context::{GpuContext, HEIGHT, OffscreenTarget, WIDTH};
use crate::render::{SlideRenderer, TransitionKind, TransitionRenderer, create_slide_renderer};
use crate::slide::instance::SlideInstance;
use crate::slide::wire::WIRE_VERSION;
use crate::wasm::WasmRuntime;
use vzglyd_slide::{Limits, SceneSpace, ShaderSources, SlideSpec, WorldLighting, WorldVertex};

const LOADING_SCENE_PATH: &str = "$loading";
const LOADING_SLIDE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/loading.vzglyd"));
#[cfg(target_os = "linux")]
const X11_BORDERLESS_INSET: u32 = 1;

/// A slide compiled and spec-read on a background thread.
/// Sent to the main thread for GPU renderer creation.
struct PendingSlide {
    idx: usize,
    path: String,
    /// Serialised slide spec bytes, ready for `create_slide_renderer`.
    spec_bytes: Vec<u8>,
    /// Keeps the extracted archive temp dir alive until the renderer is stored.
    _extracted: Option<TempPackage>,
}

/// Native application state.
pub struct NativeApp {
    context: Option<GpuContext>,
    engine: Option<Engine>,
    wasm_runtime: WasmRuntime,
    window: Option<Arc<Window>>,
    last_frame: Option<Instant>,
    running: bool,

    // Slide management
    slides_dir: Option<String>,
    /// Resolved, ordered list of absolute slide paths (from playlist or discovery).
    slide_paths: Vec<String>,
    bootstrap_renderer: Option<LoadedSlideRenderer>,
    bootstrap_target: Option<OffscreenTarget>,
    slide_renderers: Vec<Option<LoadedSlideRenderer>>,
    transition_renderer: Option<TransitionRenderer>,
    window_title: Option<String>,

    // Per-slide offscreen targets (indexed by schedule index).
    offscreen_targets: Vec<Option<OffscreenTarget>>,
    // Composite target used during transitions.
    composite_target: Option<OffscreenTarget>,

    /// Receives compiled slides from the background loader thread.
    pending_rx: Option<Receiver<Result<PendingSlide, (usize, String, String)>>>,
}

/// Loaded slide renderer with metadata and the optional extracted archive directory.
pub struct LoadedSlideRenderer {
    pub renderer: SlideRenderer,
    pub duration_secs: f32,
    /// Keeps the extracted temp dir alive for the lifetime of the renderer.
    _extracted: Option<TempPackage>,
}

/// Temporary host wrapper that avoids borrow issues.
struct HostWrapper<'a> {
    app: &'a mut NativeApp,
}

impl<'a> Host for HostWrapper<'a> {
    fn request_data(&mut self, key: &str) -> Option<Vec<u8>> {
        std::fs::read(key).ok()
    }

    fn submit_render_commands(&mut self, cmds: &[RenderCommand]) {
        // Frame-level kernel commands (BeginFrame/Clear/EndFrame) — slide drawing
        // is driven by frame_state() in render_frame(), not here.
        for cmd in cmds {
            match cmd {
                RenderCommand::BeginFrame
                | RenderCommand::EndFrame
                | RenderCommand::Clear { .. } => {}
                _ => {}
            }
        }
    }

    fn log(&mut self, level: LogLevel, msg: &str) {
        match level {
            LogLevel::Debug => log::debug!("{}", msg),
            LogLevel::Info => log::info!("{}", msg),
            LogLevel::Warn => log::warn!("{}", msg),
            LogLevel::Error => log::error!("{}", msg),
        }
    }

    fn now(&self) -> f32 {
        self.app
            .last_frame
            .map(|i| i.elapsed().as_secs_f32())
            .unwrap_or(0.0)
    }
}

fn loading_slide_spec() -> SlideSpec<WorldVertex> {
    SlideSpec {
        name: "loading_scene".into(),
        limits: Limits::pi4(),
        scene_space: SceneSpace::World3D,
        camera_path: None,
        shaders: Some(ShaderSources {
            vertex_wgsl: None,
            fragment_wgsl: Some(include_str!("loading_shader.wgsl").to_string()),
        }),
        overlay: None,
        font: None,
        textures_used: 0,
        textures: vec![],
        static_meshes: vec![],
        dynamic_meshes: vec![],
        draws: vec![],
        lighting: Some(WorldLighting::new([1.0, 1.0, 1.0], 0.08, None)),
    }
}

static LOADING_SPEC_BYTES: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let mut bytes = vec![WIRE_VERSION];
    bytes.extend(postcard::to_stdvec(&loading_slide_spec()).expect("serialize loading slide spec"));
    bytes
});

fn packaged_loading_slide_spec(engine: &wasmtime::Engine) -> Result<Vec<u8>, String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(LOADING_SLIDE))
        .map_err(|error| format!("loading slide archive: {error}"))?;
    let mut wasm_file = archive
        .by_name("slide.wasm")
        .map_err(|error| format!("loading slide archive missing slide.wasm: {error}"))?;

    let mut wasm_bytes = Vec::new();
    wasm_file
        .read_to_end(&mut wasm_bytes)
        .map_err(|error| format!("loading slide wasm read failed: {error}"))?;

    let module = wasmtime::Module::from_binary(engine, &wasm_bytes)
        .map_err(|error| format!("loading slide wasm compile failed: {error}"))?;
    let mut instance = SlideInstance::new(&module)
        .map_err(|error| format!("loading slide instantiate failed: {error}"))?;
    instance
        .read_spec_bytes()
        .map_err(|error| format!("loading slide spec read failed: {error}"))
}

impl NativeApp {
    /// Creates a new native application.
    pub fn new(slides_dir: Option<String>) -> Result<Self, String> {
        Ok(Self {
            context: None,
            engine: Some(Engine::new()),
            wasm_runtime: WasmRuntime::new()?,
            window: None,
            last_frame: None,
            running: false,
            slides_dir,
            slide_paths: Vec::new(),
            bootstrap_renderer: None,
            bootstrap_target: None,
            slide_renderers: Vec::new(),
            transition_renderer: None,
            window_title: None,
            offscreen_targets: Vec::new(),
            composite_target: None,
            pending_rx: None,
        })
    }

    /// Runs the application.
    pub fn run(slides_dir: Option<String>) -> Result<(), String> {
        let event_loop =
            EventLoop::new().map_err(|e| format!("Failed to create event loop: {}", e))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = Self::new(slides_dir)?;

        // Resolve slide paths and build kernel schedule (fast — no WASM compilation here).
        if let Some(mut engine) = app.engine.take() {
            let mut host = HostWrapper { app: &mut app };
            engine.init(&mut host);

            match &app.slides_dir {
                None => {
                    eprintln!("[vzglyd] no --slides-dir provided — nothing to show");
                    log::warn!("No slides directory provided. Pass --slides-dir <path>.");
                }
                Some(dir) => {
                    eprintln!("[vzglyd] building schedule from: {}", dir);
                    let dir = dir.clone();
                    app.slide_paths = load_schedule(&mut engine, &dir);
                    if app.slide_paths.is_empty() {
                        eprintln!(
                            "[vzglyd] WARNING: no slides found in '{}' — check the path and playlist.json",
                            dir
                        );
                        log::warn!(
                            "No slides found in '{}'. Expected .vzglyd files or playlist.json.",
                            dir
                        );
                    } else {
                        eprintln!(
                            "[vzglyd] schedule: {} slide(s) — {:?}",
                            app.slide_paths.len(),
                            app.slide_paths
                        );
                    }
                }
            }

            app.engine = Some(engine);
        }

        event_loop
            .run_app(&mut app)
            .map_err(|e| format!("Event loop error: {}", e))?;

        Ok(())
    }

    fn is_bootstrapping(&self) -> bool {
        self.bootstrap_renderer.is_some() && self.pending_rx.is_some()
    }

    fn set_window_title(&mut self, title: String) {
        if self
            .window_title
            .as_ref()
            .is_some_and(|current| current == &title)
        {
            return;
        }

        if let Some(window) = &self.window {
            window.set_title(&title);
        }
        self.window_title = Some(title);
    }

    fn sync_content_title(&mut self, frame_state: &FrameRenderState) {
        let title_idx = frame_state
            .next_slide_idx
            .unwrap_or(frame_state.current_slide_idx);
        if let Some(path) = self.slide_paths.get(title_idx) {
            self.set_window_title(scene_title(path));
        }
    }

    fn handle_surface_error(&mut self, error: wgpu::SurfaceError) {
        match error {
            wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                if let Some(ctx) = &mut self.context {
                    ctx.reconfigure();
                }
            }
            wgpu::SurfaceError::OutOfMemory => {
                log::error!("Surface out of memory; stopping render loop");
                self.running = false;
            }
            wgpu::SurfaceError::Timeout => {
                log::warn!("Surface acquisition timed out");
            }
        }
    }

    fn try_init_bootstrap_renderer(&mut self) {
        if self.slide_paths.is_empty() || self.context.is_none() {
            return;
        }

        let ctx = self.context.as_ref().expect("checked above");
        let spec_bytes = match packaged_loading_slide_spec(&self.wasm_runtime.engine) {
            Ok(bytes) => {
                log::info!("loaded packaged bootstrap slide from embedded loading.vzglyd");
                bytes
            }
            Err(error) => {
                log::warn!(
                    "failed to load packaged bootstrap slide: {error}; falling back to synthetic loading spec"
                );
                LOADING_SPEC_BYTES.clone()
            }
        };

        match create_slide_renderer(ctx, &spec_bytes) {
            Ok(renderer) => {
                self.bootstrap_renderer = Some(LoadedSlideRenderer {
                    renderer,
                    duration_secs: 0.0,
                    _extracted: None,
                });
                self.set_window_title(scene_title(LOADING_SCENE_PATH));
                log::info!("startup bootstrap loading renderer initialized");
            }
            Err(error) => {
                log::error!("failed to initialize bootstrap loading renderer: {error}");
                if let Some(path) = self.slide_paths.first() {
                    self.set_window_title(scene_title(path));
                }
            }
        }
    }

    fn ensure_bootstrap_target(&mut self) {
        if self.bootstrap_target.is_none() {
            if let Some(ctx) = &self.context {
                self.bootstrap_target = Some(ctx.create_offscreen_target());
            }
        }
    }

    /// Spawns the background slide-loader thread.
    ///
    /// Compiling WASM via Cranelift is CPU-bound and slow in debug builds. Running
    /// it off the event loop thread lets the window appear and stay responsive.
    fn start_background_load(&mut self) {
        if self.slide_paths.is_empty() {
            eprintln!("[vzglyd] no slides to load — window will stay blank");
            log::warn!(
                "No slides to load — pass --slides-dir <path> pointing at a directory with .vzglyd files or playlist.json"
            );
            return;
        }

        let (tx, rx) = mpsc::channel();
        self.pending_rx = Some(rx);

        let paths = self.slide_paths.clone();
        // Engine is Clone + Send + Sync — safe to pass to a background thread.
        let engine = self.wasm_runtime.engine.clone();

        eprintln!(
            "[vzglyd] spawning background loader for {} slide(s)",
            paths.len()
        );
        eprintln!(
            "[vzglyd] TIP: `cargo run --release -- --slides-dir <dir>` compiles WASM ~10× faster"
        );

        std::thread::spawn(move || {
            for (idx, path) in paths.iter().enumerate() {
                eprintln!("[loader] {}/{}: {}", idx + 1, paths.len(), path);

                let result = compile_slide_spec(&engine, path, idx);
                let send_result = tx.send(result);
                if send_result.is_err() {
                    // Main thread dropped the receiver — exit early.
                    break;
                }
            }
            log::info!("[loader] all slides compiled");
        });
    }

    /// Drains the background loader channel and creates GPU renderers from any
    /// newly-ready spec bytes. Called once per frame.
    fn poll_pending_slides(&mut self) {
        let Some(rx) = &self.pending_rx else { return };
        let Some(ctx) = &self.context else { return };

        let mut any = false;
        loop {
            match rx.try_recv() {
                Ok(Ok(pending)) => {
                    log::info!(
                        "[main] creating GPU renderer for slide {}: {}",
                        pending.idx,
                        pending.path
                    );
                    while self.slide_renderers.len() <= pending.idx {
                        self.slide_renderers.push(None);
                    }
                    match create_slide_renderer(ctx, &pending.spec_bytes) {
                        Ok(renderer) => {
                            self.slide_renderers[pending.idx] = Some(LoadedSlideRenderer {
                                renderer,
                                duration_secs: 7.0,
                                _extracted: pending._extracted,
                            });
                            log::info!("[main] slide {} ready", pending.idx);
                            any = true;
                        }
                        Err(e) => log::error!(
                            "[main] renderer creation failed for slide {}: {}",
                            pending.idx,
                            e
                        ),
                    }
                }
                Ok(Err((idx, path, e))) => {
                    log::error!("[main] slide {} ({}) failed to compile: {}", idx, path, e);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    log::info!("[main] background slide loading complete");
                    self.pending_rx = None;
                    self.bootstrap_renderer = None;
                    self.bootstrap_target = None;
                    break;
                }
            }
        }
        if any && self.pending_rx.is_none() {
            self.bootstrap_renderer = None;
            self.bootstrap_target = None;
        }
    }

    /// Ensures an offscreen target exists at `idx`, creating it if needed.
    fn ensure_offscreen(&mut self, idx: usize) {
        while self.offscreen_targets.len() <= idx {
            self.offscreen_targets.push(None);
        }
        if self.offscreen_targets[idx].is_none() {
            if let Some(ctx) = &self.context {
                self.offscreen_targets[idx] = Some(ctx.create_offscreen_target());
            }
        }
    }

    fn update_and_render_renderer(
        renderer: &mut LoadedSlideRenderer,
        ctx: &GpuContext,
        target: &OffscreenTarget,
        dt: f32,
    ) {
        match &mut renderer.renderer {
            SlideRenderer::Screen(screen) => {
                screen.update(dt);
                screen.render(ctx, target);
            }
            SlideRenderer::World(world) => {
                world.update(dt);
                world.render(ctx, target);
            }
        }
    }

    /// Updates GPU uniforms then renders slide `idx` to its offscreen target.
    fn render_slide_to_target(&mut self, idx: usize, dt: f32) {
        let ctx = match &self.context {
            Some(c) => c,
            None => return,
        };
        let offscreen = match self.offscreen_targets.get(idx).and_then(|t| t.as_ref()) {
            Some(t) => t,
            None => return,
        };
        if let Some(Some(renderer)) = self.slide_renderers.get_mut(idx) {
            Self::update_and_render_renderer(renderer, ctx, offscreen, dt);
        } else {
            clear_target(ctx, offscreen, wgpu::Color::BLACK);
        }
    }

    fn render_bootstrap(&mut self, dt: f32) {
        self.ensure_bootstrap_target();

        let result = {
            let Some(ctx) = &self.context else { return };
            let Some(target) = self.bootstrap_target.as_ref() else {
                return;
            };

            if let Some(renderer) = self.bootstrap_renderer.as_mut() {
                Self::update_and_render_renderer(renderer, ctx, target, dt);
            } else {
                clear_target(ctx, target, wgpu::Color::BLACK);
            }

            let bind_group = ctx.create_blit_bind_group(target);
            ctx.blit_to_surface(target, &bind_group)
        };

        if let Err(error) = result {
            self.handle_surface_error(error);
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Renders the current frame, driven entirely by kernel `frame_state()`.
    fn render_frame(&mut self, dt: f32) {
        if self.is_bootstrapping() {
            self.render_bootstrap(dt);
            return;
        }

        let frame_state: FrameRenderState = match &self.engine {
            Some(e) => e.frame_state(),
            None => return,
        };

        self.sync_content_title(&frame_state);

        if frame_state.total_slides == 0 {
            self.render_blank();
            return;
        }

        let current_idx = frame_state.current_slide_idx;

        self.ensure_offscreen(current_idx);
        if let Some(next_idx) = frame_state.next_slide_idx {
            self.ensure_offscreen(next_idx);
            if self.composite_target.is_none() {
                if let Some(ctx) = &self.context {
                    self.composite_target = Some(ctx.create_offscreen_target());
                }
            }
        }

        self.render_slide_to_target(current_idx, dt);
        if let Some(next_idx) = frame_state.next_slide_idx {
            self.render_slide_to_target(next_idx, dt);
        }

        let ctx = match &self.context {
            Some(c) => c,
            None => return,
        };
        let window = match &self.window {
            Some(w) => w.clone(),
            None => return,
        };

        let result = match frame_state.next_slide_idx {
            None => {
                if let Some(o) = self
                    .offscreen_targets
                    .get(current_idx)
                    .and_then(|t| t.as_ref())
                {
                    let bg = ctx.create_blit_bind_group(o);
                    ctx.blit_to_surface(o, &bg)
                } else {
                    Ok(())
                }
            }
            Some(next_idx) => {
                let kind = kernel_to_native_transition(
                    frame_state
                        .transition_kind
                        .unwrap_or(KernelTransitionKind::Crossfade),
                );
                if kind == TransitionKind::Cut {
                    if let Some(o) = self
                        .offscreen_targets
                        .get(next_idx)
                        .and_then(|t| t.as_ref())
                    {
                        let bg = ctx.create_blit_bind_group(o);
                        ctx.blit_to_surface(o, &bg)
                    } else {
                        Ok(())
                    }
                } else if let (Some(tr), Some(out), Some(inc), Some(comp)) = (
                    self.transition_renderer.as_ref(),
                    self.offscreen_targets
                        .get(current_idx)
                        .and_then(|t| t.as_ref()),
                    self.offscreen_targets
                        .get(next_idx)
                        .and_then(|t| t.as_ref()),
                    self.composite_target.as_ref(),
                ) {
                    tr.render(ctx, frame_state.transition_progress, kind, out, inc, comp);
                    let bg = ctx.create_blit_bind_group(comp);
                    ctx.blit_to_surface(comp, &bg)
                } else {
                    // Transition renderer not ready — show incoming slide.
                    if let Some(o) = self
                        .offscreen_targets
                        .get(next_idx)
                        .and_then(|t| t.as_ref())
                    {
                        let bg = ctx.create_blit_bind_group(o);
                        ctx.blit_to_surface(o, &bg)
                    } else {
                        Ok(())
                    }
                }
            }
        };

        if let Err(error) = result {
            self.handle_surface_error(error);
        }

        window.request_redraw();
    }

    /// Renders a blank (dark) frame — shown while slides are still loading.
    fn render_blank(&mut self) {
        let ctx = match &self.context {
            Some(c) => c,
            None => return,
        };
        let window = match &self.window {
            Some(w) => w.clone(),
            None => return,
        };

        let offscreen = ctx.create_offscreen_target();
        clear_target(
            ctx,
            &offscreen,
            wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.1,
                a: 1.0,
            },
        );

        let result = {
            let bg = ctx.create_blit_bind_group(&offscreen);
            ctx.blit_to_surface(&offscreen, &bg)
        };
        if let Err(error) = result {
            self.handle_surface_error(error);
        }
        window.request_redraw();
    }
}

// ---------------------------------------------------------------------------
// Background loading helpers
// ---------------------------------------------------------------------------

/// Compiles WASM and reads spec bytes for a single slide path.
///
/// This is the slow, CPU-bound step (Cranelift JIT) — run it off the main thread.
fn compile_slide_spec(
    engine: &wasmtime::Engine,
    path: &str,
    idx: usize,
) -> Result<PendingSlide, (usize, String, String)> {
    macro_rules! bail {
        ($msg:expr) => {
            return Err((idx, path.to_string(), $msg))
        };
    }

    let p = std::path::Path::new(path);

    let (wasm_path, extracted) = if p.extension().and_then(|e| e.to_str()) == Some("vzglyd") {
        log::info!("[loader] extracting archive: {}", path);
        let pkg = match extract_archive(p) {
            Ok(p) => p,
            Err(e) => bail!(format!("extract: {}", e)),
        };
        let wasm = pkg.path.join("slide.wasm");
        if !wasm.exists() {
            bail!(format!("no slide.wasm inside {}", path));
        }
        (wasm.to_string_lossy().to_string(), Some(pkg))
    } else if p.is_dir() {
        let wasm = p.join("slide.wasm");
        if !wasm.exists() {
            bail!(format!("no slide.wasm in directory {}", path));
        }
        (wasm.to_string_lossy().to_string(), None)
    } else {
        (path.to_string(), None)
    };

    eprintln!("[loader] compiling WASM: {}", wasm_path);
    eprintln!("[loader]   (Cranelift JIT — takes ~10s in release, ~5min in debug)");
    let t0 = std::time::Instant::now();
    let module = match wasmtime::Module::from_file(engine, &wasm_path) {
        Ok(m) => m,
        Err(e) => bail!(format!("WASM compile: {}", e)),
    };
    eprintln!(
        "[loader] compiled in {:.1}s; instantiating…",
        t0.elapsed().as_secs_f32()
    );

    let mut instance = match SlideInstance::new(&module) {
        Ok(i) => i,
        Err(e) => bail!(format!("instantiate: {}", e)),
    };
    eprintln!("[loader] reading spec bytes from vzglyd_init()…");

    let spec_bytes = match instance.read_spec_bytes() {
        Ok(b) => b,
        Err(e) => bail!(format!("spec read: {}", e)),
    };
    eprintln!(
        "[loader] slide {} spec ready ({} bytes)",
        idx,
        spec_bytes.len()
    );

    Ok(PendingSlide {
        idx,
        path: path.to_string(),
        spec_bytes,
        _extracted: extracted,
    })
}

// ---------------------------------------------------------------------------
// Schedule helpers
// ---------------------------------------------------------------------------

/// Builds the kernel schedule and returns the ordered absolute slide paths.
fn load_schedule(engine: &mut Engine, slides_dir: &str) -> Vec<String> {
    let playlist_path = std::path::Path::new(slides_dir).join("playlist.json");
    eprintln!(
        "[vzglyd] looking for playlist: {} (exists={})",
        playlist_path.display(),
        playlist_path.exists()
    );

    if playlist_path.exists() {
        match std::fs::read(&playlist_path) {
            Ok(bytes) => match parse_playlist(&bytes) {
                Ok(playlist) => {
                    let paths = build_schedule_from_playlist(&playlist, slides_dir);
                    eprintln!(
                        "[vzglyd] playlist.json: {} total entries, {} enabled → {:?}",
                        playlist.slides.len(),
                        paths.len(),
                        paths
                    );
                    engine.set_schedule_from_playlist(&playlist, slides_dir);
                    return paths;
                }
                Err(e) => eprintln!(
                    "[vzglyd] WARNING: playlist.json parse failed: {} — falling back to discovery",
                    e
                ),
            },
            Err(e) => eprintln!(
                "[vzglyd] WARNING: playlist.json read failed: {} — falling back to discovery",
                e
            ),
        }
    } else {
        eprintln!(
            "[vzglyd] no playlist.json — scanning '{}' for .vzglyd archives",
            slides_dir
        );
    }

    let paths = discover_slide_paths(slides_dir);
    eprintln!("[vzglyd] discovered: {:?}", paths);
    engine.set_schedule(paths.clone());
    paths
}

/// Scans `dir` (non-recursively) for `.vzglyd` archives and slide directories.
fn discover_slide_paths(dir: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return paths;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_slide_archive(&path) {
            if let Some(s) = path.to_str() {
                paths.push(s.to_string());
            }
        } else if is_slide_directory(&path) {
            if let Some(s) = path.to_str() {
                paths.push(s.to_string());
            }
        }
    }
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

fn scene_title(path: &str) -> String {
    let label = std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path);
    format!("VZGLYD — {label}")
}

fn initial_window_title(paths: &[String]) -> String {
    if paths.is_empty() {
        "VZGLYD".to_string()
    } else {
        scene_title(LOADING_SCENE_PATH)
    }
}

fn build_window_attributes(event_loop: &ActiveEventLoop, title: &str) -> WindowAttributes {
    let desired_size = PhysicalSize::new(WIDTH, HEIGHT);
    let mut level = WindowLevel::AlwaysOnTop;
    let mut attributes = Window::default_attributes()
        .with_title(title)
        .with_inner_size(desired_size)
        .with_resizable(false)
        .with_decorations(false);

    #[cfg(target_os = "linux")]
    {
        if event_loop.is_x11() {
            attributes = attributes
                .with_override_redirect(false)
                .with_x11_window_type(vec![X11WindowType::Normal]);

            if let Some((window_size, position)) =
                x11_borderless_managed_geometry(event_loop, desired_size)
            {
                log::info!(
                    "X11 borderless window matched the monitor size; using {:?} at {:?} to avoid fullscreen heuristics",
                    window_size,
                    position
                );
                attributes = attributes
                    .with_inner_size(window_size)
                    .with_position(position);
                level = WindowLevel::Normal;
            }
        }
    }

    attributes.with_window_level(level)
}

#[cfg(target_os = "linux")]
fn x11_borderless_managed_geometry(
    event_loop: &ActiveEventLoop,
    desired_size: PhysicalSize<u32>,
) -> Option<(PhysicalSize<u32>, PhysicalPosition<i32>)> {
    let monitor = event_loop.primary_monitor()?;
    let monitor_size = monitor.size();
    if desired_size.width < monitor_size.width || desired_size.height < monitor_size.height {
        return None;
    }

    let inset = X11_BORDERLESS_INSET;
    let inset_span = inset.saturating_mul(2);
    let window_size = PhysicalSize::new(
        monitor_size.width.saturating_sub(inset_span).max(1),
        monitor_size.height.saturating_sub(inset_span).max(1),
    );
    let monitor_position = monitor.position();
    let position = PhysicalPosition::new(
        monitor_position.x + inset as i32,
        monitor_position.y + inset as i32,
    );

    Some((window_size, position))
}

fn is_slide_archive(path: &std::path::Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("vzglyd"))
}

fn is_slide_directory(path: &std::path::Path) -> bool {
    if !path.is_dir() {
        return false;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };

    let mut has_manifest = false;
    let mut has_wasm = false;
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            continue;
        };
        if file_name == "manifest.json" || file_name.ends_with("_slide.json") {
            has_manifest = true;
        }
        if file_name == "slide.wasm" || file_name.ends_with("_slide.wasm") {
            has_wasm = true;
        }
    }

    has_manifest && has_wasm
}

fn kernel_to_native_transition(kind: KernelTransitionKind) -> TransitionKind {
    match kind {
        KernelTransitionKind::Crossfade => TransitionKind::Crossfade,
        KernelTransitionKind::WipeLeft => TransitionKind::WipeLeft,
        KernelTransitionKind::WipeDown => TransitionKind::WipeDown,
        KernelTransitionKind::Dissolve => TransitionKind::Dissolve,
        KernelTransitionKind::Cut => TransitionKind::Cut,
    }
}

fn clear_target(ctx: &GpuContext, target: &OffscreenTarget, color: wgpu::Color) {
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clear"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("clear_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
    }
    ctx.queue.submit(Some(encoder.finish()));
}

// ---------------------------------------------------------------------------
// winit ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for NativeApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let initial_title = initial_window_title(&self.slide_paths);
        let window = event_loop
            .create_window(build_window_attributes(event_loop, &initial_title))
            .expect("Failed to create window");
        let window = Arc::new(window);

        let context = pollster::block_on(GpuContext::new(window.clone()))
            .expect("Failed to create GPU context");
        let transition_renderer = TransitionRenderer::new(&context);

        self.context = Some(context);
        self.transition_renderer = Some(transition_renderer);
        self.window = Some(window);
        self.window_title = Some(initial_title);
        self.running = true;

        self.try_init_bootstrap_renderer();
        if self.bootstrap_renderer.is_none() {
            if let Some(path) = self.slide_paths.first() {
                self.set_window_title(scene_title(path));
            }
        }

        // Window is up — start loading slides in the background.
        self.start_background_load();

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.running = false;
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(ctx) = &mut self.context {
                    ctx.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if !self.running {
                    return;
                }

                let now = Instant::now();
                let dt = self
                    .last_frame
                    .map(|last| now.duration_since(last).as_secs_f32())
                    .unwrap_or(1.0 / 60.0);
                self.last_frame = Some(now);

                // Collect any newly-compiled slides.
                self.poll_pending_slides();

                // While the bootstrap loading scene is visible, keep the kernel's
                // schedule frozen so the first real slide starts at t=0 once the
                // background loader hands control back to content rendering.
                if !self.is_bootstrapping() {
                    if let Some(mut engine) = self.engine.take() {
                        let mut host = HostWrapper { app: self };
                        let _output = engine.update(&mut host, EngineInput { dt, events: vec![] });
                        self.engine = Some(engine);
                    }
                }

                self.render_frame(dt);
            }
            _ => {}
        }
    }
}
