//! Native application and event loop.
//!
//! Integrates the kernel with winit event loop and wgpu rendering.

use signal_hook::consts::signal::{SIGINT, SIGTERM};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use vzglyd_kernel::SecretsStore;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use vzglyd_kernel::TransitionKind as KernelTransitionKind;
use vzglyd_kernel::schedule::{PLAYLIST_FILENAME, Playlist, parse_playlist};
use vzglyd_kernel::{
    Engine, EngineConfig, EngineInput, FrameRenderState, Host, LogLevel, RenderCommand, SlideEntry,
    SlideManifestMetadata,
};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId, WindowLevel};

#[cfg(target_os = "linux")]
use winit::platform::x11::{
    ActiveEventLoopExtX11, WindowAttributesExtX11, WindowType as X11WindowType,
};

use crate::gpu::context::{GpuContext, HEIGHT, OffscreenTarget, WIDTH};
use crate::render::{
    LoadedSlide, OverlayRenderer, SlideRenderer, TransitionKind, TransitionRenderer,
    create_loaded_slide_renderer, load_wasm_slide, load_wasm_slide_from_bytes,
};
use crate::server::{AppStatus, start_server};
use vzglyd_kernel::manifest::SlideManifest;
use crate::trace::set_active_trace_recorder;

const LOADING_SCENE_PATH: &str = "$loading";
const LOADING_SLIDE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/loading.vzglyd"));
#[cfg(target_os = "linux")]
const X11_BORDERLESS_INSET: u32 = 1;
static SHUTDOWN_SIGNAL_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SHUTDOWN_SIGNAL_HANDLERS: OnceLock<()> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub slides_dir: Option<String>,
    pub scene_path: Option<String>,
    pub trace: bool,
    pub trace_out: Option<String>,
}

#[derive(Clone, Debug)]
struct ScheduledSlide {
    path: String,
    params: Option<serde_json::Value>,
}

struct PendingSlide {
    idx: usize,
    package: LoadedSlidePackage,
}

struct LoadedSlidePackage {
    slide: LoadedSlide,
    manifest: Option<SlideManifest>,
    path: String,
}

/// Native application state.
pub struct NativeApp {
    context: Option<GpuContext>,
    engine: Option<Engine>,
    window: Option<Arc<Window>>,
    last_frame: Option<Instant>,
    running: bool,

    // Slide management
    run_config: RunConfig,
    /// Display scale from `playlist.json` (default 1.0). Applied to the blit
    /// shader so that scale < 1.0 adds a black border for CRT overscan safety.
    display_scale: f32,
    /// Resolved, ordered schedule metadata cloned from the kernel after setup.
    slides: Vec<ScheduledSlide>,
    bootstrap_renderer: Option<LoadedSlideRenderer>,
    bootstrap_target: Option<OffscreenTarget>,
    slide_renderers: Vec<Option<LoadedSlideRenderer>>,
    transition_renderer: Option<TransitionRenderer>,
    overlay_renderer: Option<OverlayRenderer>,
    window_title: Option<String>,

    // Per-slide offscreen targets (indexed by schedule index).
    offscreen_targets: Vec<Option<OffscreenTarget>>,
    // Composite target used during transitions.
    composite_target: Option<OffscreenTarget>,

    /// Receives compiled slides from the background loader thread.
    pending_rx: Option<Receiver<Result<PendingSlide, (usize, String, String)>>>,
    trace_recorder: Option<vzglyd_kernel::trace::TraceRecorder>,
    shutdown_requested: Arc<AtomicBool>,
    /// Secrets (API keys etc.) injected into sidecar WASI environments.
    pub secrets: Arc<RwLock<SecretsStore>>,
    /// Receives new playlist JSON from the management server for hot-reload.
    playlist_reload_rx: Option<Receiver<String>>,
    /// Shared status written each frame; read by the management server.
    app_status: Arc<RwLock<AppStatus>>,
    /// Frame counter and time for rolling FPS calculation.
    fps_frame_count: u32,
    fps_window_start: Option<Instant>,
}

/// Loaded slide renderer with metadata and the optional extracted archive directory.
pub struct LoadedSlideRenderer {
    pub renderer: SlideRenderer,
    pub manifest: Option<SlideManifest>,
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

impl NativeApp {
    /// Creates a new native application.
    pub fn new(run_config: RunConfig) -> Result<Self, String> {
        let shutdown_requested = install_shutdown_signal_handlers()?;
        let trace_recorder = if run_config.trace {
            let trace_path = resolve_trace_output_path(&run_config);
            let recorder = vzglyd_kernel::trace::TraceRecorder::new(
                &trace_path,
                "native",
                trace_label(&run_config),
            )
            .map_err(|error| format!("Failed to create trace recorder: {error}"))?;
            if let Some(scene_path) = run_config.scene_path.as_deref() {
                recorder.set_metadata("scene_path", scene_path);
            }
            if let Some(slides_dir) = run_config.slides_dir.as_deref() {
                recorder.set_metadata("slides_dir", slides_dir);
            }
            log::info!(
                "native tracing enabled: {}",
                recorder.trace_path().display()
            );
            Some(recorder)
        } else {
            None
        };
        set_active_trace_recorder(trace_recorder.clone());

        Ok(Self {
            context: None,
            engine: Some(Engine::with_config(EngineConfig::default())),
            window: None,
            last_frame: None,
            running: false,
            run_config,
            display_scale: 1.0,
            slides: Vec::new(),
            bootstrap_renderer: None,
            bootstrap_target: None,
            slide_renderers: Vec::new(),
            transition_renderer: None,
            overlay_renderer: None,
            window_title: None,
            offscreen_targets: Vec::new(),
            composite_target: None,
            pending_rx: None,
            trace_recorder,
            shutdown_requested,
            secrets: Arc::new(RwLock::new(SecretsStore::default())),
            playlist_reload_rx: None,
            app_status: Arc::new(RwLock::new(AppStatus::default())),
            fps_frame_count: 0,
            fps_window_start: None,
        })
    }

    /// Runs the application.
    pub fn run(run_config: RunConfig) -> Result<(), String> {
        let event_loop =
            EventLoop::new().map_err(|e| format!("Failed to create event loop: {}", e))?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = Self::new(run_config)?;

        // Load secrets from slides_dir/secrets.json (single-threaded, safe to use set_var here).
        if let Some(dir) = app.run_config.slides_dir.as_deref() {
            match crate::server::load_secrets(std::path::Path::new(dir)) {
                Ok(secrets) => {
                    if !secrets.is_empty() {
                        eprintln!("[vzglyd] loaded {} secret(s) from secrets.json", secrets.len());
                    }
                    // Safety: single-threaded at this point (before EventLoop::run_app).
                    for (key, val) in &secrets.0 {
                        #[allow(unused_unsafe)]
                        unsafe { std::env::set_var(key, val) };
                    }
                    if let Ok(mut guard) = app.secrets.write() {
                        *guard = secrets;
                    }
                }
                Err(e) => log::warn!("failed to load secrets.json: {e}"),
            }
        }

        // Start the management HTTP server on port 8080 (only when slides_dir is set).
        if let Some(dir) = app.run_config.slides_dir.as_deref() {
            let slides_dir = std::path::PathBuf::from(dir);
            let (_handle, playlist_rx, app_status) =
                start_server(slides_dir, Arc::clone(&app.secrets));
            app.playlist_reload_rx = Some(playlist_rx);
            app.app_status = app_status;
        }

        // Resolve slide paths and build kernel schedule before the event loop starts.
        if let Some(mut engine) = app.engine.take() {
            let mut host = HostWrapper { app: &mut app };
            engine.init(&mut host);
            let (slides, display_scale) = load_schedule(&mut engine, &app.run_config)?;
            app.slides = slides;
            app.display_scale = display_scale;
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

    fn exit_for_signal(&mut self, event_loop: &ActiveEventLoop) -> bool {
        if !self.shutdown_requested.swap(false, Ordering::SeqCst) {
            return false;
        }

        if let Some(recorder) = &self.trace_recorder {
            recorder.instant(
                "native.main".to_string(),
                "lifecycle",
                "signal_exit",
                BTreeMap::from([("signal".to_string(), "interrupt".to_string())]),
            );
        }
        log::info!("shutdown signal received; exiting cleanly");
        self.running = false;
        event_loop.exit();
        true
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
        if let Some(slide) = self.slides.get(title_idx) {
            self.set_window_title(scene_title(&slide.path));
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
        if self.slides.is_empty() || self.context.is_none() {
            return;
        }

        let ctx = self.context.as_ref().expect("checked above");
        match load_wasm_slide_from_bytes(LOADING_SLIDE)
            .map_err(|error| format!("loading slide: {error}"))
            .and_then(|(slide, _manifest)| create_loaded_slide_renderer(ctx, slide))
        {
            Ok(renderer) => {
                self.bootstrap_renderer = Some(LoadedSlideRenderer {
                    renderer,
                    manifest: None,
                });
                self.set_window_title(scene_title(LOADING_SCENE_PATH));
                log::info!("startup bootstrap loading renderer initialized");
            }
            Err(error) => {
                log::error!("failed to initialize bootstrap loading renderer: {error}");
                if let Some(slide) = self.slides.first() {
                    self.set_window_title(scene_title(&slide.path));
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
        if self.slides.is_empty() {
            eprintln!("[vzglyd] no slides to load — window will stay blank");
            log::warn!("No slides to load — check the configured slide source");
            return;
        }

        let (tx, rx) = mpsc::channel();
        self.pending_rx = Some(rx);

        let slides = self.slides.clone();
        let secrets = Arc::clone(&self.secrets);

        eprintln!(
            "[vzglyd] spawning background loader for {} slide(s)",
            slides.len()
        );

        std::thread::spawn(move || {
            // Snapshot secrets once for this loader batch.
            let extra_env: Vec<(String, String)> = secrets
                .read()
                .map(|s| s.0.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default();
            for (idx, slide) in slides.iter().enumerate() {
                eprintln!("[loader] {}/{}: {}", idx + 1, slides.len(), slide.path);
                let result = load_slide_package(idx, slide, &extra_env);
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
                        pending.package.path
                    );
                    while self.slide_renderers.len() <= pending.idx {
                        self.slide_renderers.push(None);
                    }
                    let manifest = pending.package.manifest;
                    match create_loaded_slide_renderer(ctx, pending.package.slide) {
                        Ok(renderer) => {
                            self.slide_renderers[pending.idx] = Some(LoadedSlideRenderer {
                                renderer,
                                manifest: manifest.clone(),
                            });
                            if let Some(engine) = &mut self.engine {
                                engine.apply_manifest_metadata(
                                    pending.idx,
                                    manifest_schedule_metadata(manifest.as_ref()),
                                );
                            }
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

    /// Drain the playlist hot-reload channel and apply any pending new playlist.
    ///
    /// Called once per frame. If a new playlist JSON string is received from
    /// the management server, the schedule is re-resolved and the background
    /// WASM compilation is restarted.
    fn poll_playlist_reload(&mut self) {
        let Some(rx) = &self.playlist_reload_rx else {
            return;
        };

        // Drain all pending messages, keep only the latest.
        let mut latest: Option<String> = None;
        loop {
            match rx.try_recv() {
                Ok(json) => {
                    latest = Some(json);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.playlist_reload_rx = None;
                    break;
                }
            }
        }

        let Some(json) = latest else { return };

        let Some(slides_dir) = self.run_config.slides_dir.as_deref() else {
            return;
        };

        match vzglyd_kernel::schedule::parse_playlist(json.as_bytes()) {
            Ok(playlist) => {
                log::info!("[hot-reload] applying new playlist ({} slides)", playlist.slides.len());

                if let Some(mut engine) = self.engine.take() {
                    engine.set_schedule_from_playlist(&playlist, slides_dir);
                    self.display_scale = playlist.display_scale;
                    self.slides = schedule_snapshot(&engine);
                    self.engine = Some(engine);
                }

                // Clear all existing renderers and restart background loading.
                self.slide_renderers.clear();
                self.offscreen_targets.clear();
                self.pending_rx = None;
                self.bootstrap_renderer = None;
                self.bootstrap_target = None;
                self.start_background_load();
                self.try_init_bootstrap_renderer();
            }
            Err(e) => {
                log::warn!("[hot-reload] received invalid playlist JSON: {e}");
            }
        }
    }

    /// Update the shared AppStatus with the current FPS (rolling 1-second window).
    fn update_fps_status(&mut self, dt: f32) {
        self.fps_frame_count += 1;
        let now = Instant::now();
        let window_start = self.fps_window_start.get_or_insert(now);
        let elapsed = now.duration_since(*window_start).as_secs_f32();

        if elapsed >= 1.0 {
            let fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.fps_window_start = Some(now);

            let current_slide = self
                .slides
                .first()
                .map(|s| {
                    std::path::Path::new(&s.path)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(&s.path)
                        .to_string()
                });

            if let Ok(mut status) = self.app_status.write() {
                status.fps = fps;
                status.current_slide = current_slide;
            }
        }
        let _ = dt;
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

    fn sync_renderer_activity(&mut self, frame_state: &FrameRenderState) {
        let current_idx = frame_state.current_slide_idx;
        let next_idx = frame_state.next_slide_idx;

        for (idx, renderer) in self.slide_renderers.iter_mut().enumerate() {
            if let Some(renderer) = renderer {
                let active = idx == current_idx || Some(idx) == next_idx;
                renderer.renderer.set_active(active);
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
                screen.update(ctx, dt);
                screen.render(ctx, target);
            }
            SlideRenderer::World(world) => {
                world.update(ctx, dt);
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
                renderer.renderer.set_active(true);
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
        let _frame_trace = self
            .trace_recorder
            .as_ref()
            .map(|recorder| recorder.scoped("native.main", "frame", "render_frame"));
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

        // Screensaver mode: suppress the normal slide and HUD border.
        if let Some(ss_state) = frame_state.screensaver.clone() {
            self.render_screensaver(&ss_state);
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            return;
        }

        let current_idx = frame_state.current_slide_idx;
        self.sync_renderer_activity(&frame_state);

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

        // Determine which slide's name to show in the overlay footer.
        // During transitions we show the incoming (next) slide's name.
        let display_idx = frame_state.next_slide_idx.unwrap_or(current_idx);
        let slide_name: Option<String> = self
            .slide_renderers
            .get(display_idx)
            .and_then(|r| r.as_ref())
            .and_then(|r| r.manifest.as_ref())
            .and_then(|m| m.name.clone());

        let result = match frame_state.next_slide_idx {
            None => {
                if let Some(o) = self
                    .offscreen_targets
                    .get(current_idx)
                    .and_then(|t| t.as_ref())
                {
                    let bg = ctx.create_blit_bind_group(o);
                    if let Some(overlay) = self.overlay_renderer.as_mut() {
                        ctx.blit_and_overlay_to_surface(o, &bg, |view, encoder| {
                            overlay.record_pass(ctx, view, encoder, slide_name.as_deref(), ctx.surface_display_rect());
                        })
                    } else {
                        ctx.blit_to_surface(o, &bg)
                    }
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
                        if let Some(overlay) = self.overlay_renderer.as_mut() {
                            ctx.blit_and_overlay_to_surface(o, &bg, |view, encoder| {
                                overlay.record_pass(ctx, view, encoder, slide_name.as_deref(), ctx.surface_display_rect());
                            })
                        } else {
                            ctx.blit_to_surface(o, &bg)
                        }
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
                    if let Some(overlay) = self.overlay_renderer.as_mut() {
                        ctx.blit_and_overlay_to_surface(comp, &bg, |view, encoder| {
                            overlay.record_pass(ctx, view, encoder, slide_name.as_deref(), ctx.surface_display_rect());
                        })
                    } else {
                        ctx.blit_to_surface(comp, &bg)
                    }
                } else {
                    // Transition renderer not ready — show incoming slide.
                    if let Some(o) = self
                        .offscreen_targets
                        .get(next_idx)
                        .and_then(|t| t.as_ref())
                    {
                        let bg = ctx.create_blit_bind_group(o);
                        if let Some(overlay) = self.overlay_renderer.as_mut() {
                            ctx.blit_and_overlay_to_surface(o, &bg, |view, encoder| {
                                overlay.record_pass(ctx, view, encoder, slide_name.as_deref(), ctx.surface_display_rect());
                            })
                        } else {
                            ctx.blit_to_surface(o, &bg)
                        }
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

    /// Renders the screensaver frame — suppresses the slide and HUD border,
    /// showing the full-screen intermission scene instead.
    fn render_screensaver(&mut self, state: &vzglyd_kernel::ScreensaverFrameState) {
        let ctx = match &self.context {
            Some(c) => c,
            None => return,
        };
        let Some(overlay) = self.overlay_renderer.as_mut() else {
            return;
        };
        let blit_rect = ctx.surface_display_rect();
        let result = ctx.clear_and_overlay_to_surface(|view, encoder| {
            overlay.record_screensaver_pass(ctx, view, encoder, state, blit_rect);
        });
        if let Err(error) = result {
            self.handle_surface_error(error);
        }
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

impl Drop for NativeApp {
    fn drop(&mut self) {
        if let Some(recorder) = self.trace_recorder.take() {
            match recorder.flush() {
                Ok(path) => {
                    let size_bytes = std::fs::metadata(&path).ok().map(|meta| meta.len());
                    match size_bytes {
                        Some(size) => eprintln!(
                            "[vzglyd] trace written: {} ({} bytes)",
                            path.display(),
                            size
                        ),
                        None => eprintln!("[vzglyd] trace written: {}", path.display()),
                    }
                    eprintln!("[vzglyd] open the trace in Perfetto: https://ui.perfetto.dev/");
                }
                Err(error) => {
                    eprintln!("[vzglyd] failed to flush trace: {error}");
                }
            }
        }
        set_active_trace_recorder(None);
    }
}

// ---------------------------------------------------------------------------
// Background loading helpers
// ---------------------------------------------------------------------------

fn load_slide_package(
    idx: usize,
    slide: &ScheduledSlide,
    extra_env: &[(String, String)],
) -> Result<PendingSlide, (usize, String, String)> {
    macro_rules! bail {
        ($msg:expr) => {
            return Err((idx, slide.path.clone(), $msg))
        };
    }

    let params_bytes = slide
        .params
        .as_ref()
        .map(|value| serde_json::to_vec(value).expect("params serialization is infallible"));
    let (loaded, manifest) = match load_wasm_slide(&slide.path, params_bytes.as_deref(), extra_env) {
        Ok(loaded) => loaded,
        Err(error) => bail!(format!("slide load failed: {error}")),
    };
    if let Err(error) = loaded.validate() {
        bail!(format!("slide validation failed: {error}"));
    }

    Ok(PendingSlide {
        idx,
        package: LoadedSlidePackage {
            slide: loaded,
            manifest,
            path: slide.path.clone(),
        },
    })
}

fn resolve_trace_output_path(run_config: &RunConfig) -> PathBuf {
    if let Some(path) = run_config.trace_out.as_deref() {
        return PathBuf::from(path);
    }

    let source = run_config
        .scene_path
        .as_deref()
        .and_then(|path| Path::new(path).file_stem())
        .and_then(|stem| stem.to_str())
        .or_else(|| {
            run_config
                .slides_dir
                .as_deref()
                .and_then(|path| Path::new(path).file_name())
                .and_then(|name| name.to_str())
        })
        .unwrap_or("session");
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from("traces").join(format!(
        "vzglyd-native-{}-{timestamp}.perfetto.json",
        sanitize_trace_component(source)
    ))
}

fn install_shutdown_signal_handlers() -> Result<Arc<AtomicBool>, String> {
    let flag = SHUTDOWN_SIGNAL_FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone();
    flag.store(false, Ordering::SeqCst);

    if SHUTDOWN_SIGNAL_HANDLERS.get().is_none() {
        signal_hook::flag::register(SIGINT, Arc::clone(&flag))
            .map_err(|error| format!("Failed to register SIGINT handler: {error}"))?;
        signal_hook::flag::register(SIGTERM, Arc::clone(&flag))
            .map_err(|error| format!("Failed to register SIGTERM handler: {error}"))?;
        let _ = SHUTDOWN_SIGNAL_HANDLERS.set(());
    }

    Ok(flag)
}

fn sanitize_trace_component(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "session".to_string()
    } else {
        sanitized
    }
}

fn trace_label(run_config: &RunConfig) -> String {
    let raw = run_config
        .scene_path
        .as_deref()
        .or(run_config.slides_dir.as_deref())
        .unwrap_or("native-session");
    raw.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

// ---------------------------------------------------------------------------
// Schedule helpers
// ---------------------------------------------------------------------------

fn manifest_schedule_metadata(manifest: Option<&SlideManifest>) -> SlideManifestMetadata {
    SlideManifestMetadata {
        duration_secs: manifest
            .and_then(|manifest| manifest.display_duration_seconds())
            .map(|seconds| seconds as f32),
        transition_in: manifest.and_then(SlideManifest::transition_in_kind),
        transition_out: manifest.and_then(SlideManifest::transition_out_kind),
    }
}

fn schedule_snapshot(engine: &Engine) -> Vec<ScheduledSlide> {
    engine
        .schedule_entries()
        .iter()
        .map(|entry: &SlideEntry| ScheduledSlide {
            path: entry.path.clone(),
            params: entry.params.clone(),
        })
        .collect()
}

/// Builds the kernel schedule and returns the ordered slide metadata plus the
/// display scale from the playlist (defaults to `1.0` if not specified or when
/// running in single-scene mode).
fn load_schedule(
    engine: &mut Engine,
    run_config: &RunConfig,
) -> Result<(Vec<ScheduledSlide>, f32), String> {
    if let Some(scene_path) = run_config.scene_path.as_ref() {
        eprintln!("[vzglyd] single-scene mode: {scene_path}");
        engine.set_schedule(vec![scene_path.clone()]);
        return Ok((schedule_snapshot(engine), 1.0));
    }

    let Some(slides_dir) = run_config.slides_dir.as_deref() else {
        eprintln!("[vzglyd] no slide source configured");
        return Ok((Vec::new(), 1.0));
    };

    let playlist_path = Path::new(slides_dir).join(PLAYLIST_FILENAME);
    eprintln!("[vzglyd] using shared slides repo root: {}", slides_dir);

    if !playlist_path.exists() {
        return Err(format!(
            "shared slides repo '{}' is missing required {}",
            slides_dir, PLAYLIST_FILENAME
        ));
    }

    let bytes = std::fs::read(&playlist_path)
        .map_err(|error| format!("failed to read {}: {}", playlist_path.display(), error))?;
    let playlist = parse_playlist(&bytes)
        .map_err(|error| format!("invalid {}: {}", playlist_path.display(), error))?;
    validate_shared_repo_playlist(&playlist)
        .map_err(|error| format!("invalid {}: {}", playlist_path.display(), error))?;

    let display_scale = playlist.display_scale;
    engine.set_schedule_from_playlist(&playlist, slides_dir);
    let slides = schedule_snapshot(engine);
    let paths: Vec<&str> = slides.iter().map(|slide| slide.path.as_str()).collect();
    eprintln!(
        "[vzglyd] {}: {} total entries, {} enabled, display_scale={} → {:?}",
        PLAYLIST_FILENAME,
        playlist.slides.len(),
        slides.len(),
        display_scale,
        paths
    );

    Ok((slides, display_scale))
}

fn validate_shared_repo_playlist(playlist: &Playlist) -> Result<(), String> {
    for (index, entry) in playlist.slides.iter().enumerate() {
        validate_shared_repo_bundle_path(&entry.path)
            .map_err(|error| format!("slides[{index}].path {}", error))?;
    }
    Ok(())
}

fn validate_shared_repo_bundle_path(path: &str) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("must be a non-empty string".into());
    }
    if path.starts_with('/') {
        return Err("must be relative to the repo root".into());
    }
    if path.contains('\\') {
        return Err("must use forward slashes".into());
    }
    if path
        .split('/')
        .any(|segment| segment == "." || segment == "..")
    {
        return Err("must not contain '.' or '..' path segments".into());
    }
    if !path.ends_with(".vzglyd") {
        return Err("must point to a .vzglyd bundle".into());
    }
    Ok(())
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

fn initial_window_title(slides: &[ScheduledSlide]) -> String {
    if slides.is_empty() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_repo_relative_bundle_paths() {
        assert!(validate_shared_repo_bundle_path("clock.vzglyd").is_ok());
        assert!(validate_shared_repo_bundle_path("daily/headlines.vzglyd").is_ok());
    }

    #[test]
    fn rejects_non_bundle_paths() {
        let error = validate_shared_repo_bundle_path("daily/headlines/slide.wasm")
            .expect_err("expected validation error");
        assert!(error.contains(".vzglyd"));
    }

    #[test]
    fn rejects_escape_segments() {
        let error = validate_shared_repo_bundle_path("../clock.vzglyd")
            .expect_err("expected validation error");
        assert!(error.contains("'.' or '..'"));
    }

    #[test]
    fn default_trace_output_uses_perfetto_extension() {
        let path = resolve_trace_output_path(&RunConfig {
            slides_dir: None,
            scene_path: Some("/slides/air_quality.vzglyd".to_string()),
            trace: true,
            trace_out: None,
        });
        let path_str = path.to_string_lossy();
        assert!(path_str.starts_with("traces/"));
        assert!(path_str.ends_with(".perfetto.json"));
        assert!(path_str.contains("air-quality"));
    }

    #[test]
    fn explicit_trace_output_path_wins() {
        let path = resolve_trace_output_path(&RunConfig {
            slides_dir: Some("slides".to_string()),
            scene_path: None,
            trace: true,
            trace_out: Some("/tmp/custom.perfetto.json".to_string()),
        });
        assert_eq!(path, PathBuf::from("/tmp/custom.perfetto.json"));
    }
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
        if self.exit_for_signal(event_loop) {
            return;
        }
        if self.window.is_some() {
            return;
        }

        let initial_title = initial_window_title(&self.slides);
        let window = event_loop
            .create_window(build_window_attributes(event_loop, &initial_title))
            .expect("Failed to create window");
        let window = Arc::new(window);

        let mut context = pollster::block_on(GpuContext::new(window.clone()))
            .expect("Failed to create GPU context");
        context.set_display_scale(self.display_scale);
        let transition_renderer = TransitionRenderer::new(&context);
        let overlay_renderer = OverlayRenderer::new(&context);

        self.context = Some(context);
        self.transition_renderer = Some(transition_renderer);
        self.overlay_renderer = Some(overlay_renderer);
        self.window = Some(window);
        self.window_title = Some(initial_title);
        self.running = true;

        self.try_init_bootstrap_renderer();
        if self.bootstrap_renderer.is_none() {
            if let Some(slide) = self.slides.first() {
                self.set_window_title(scene_title(&slide.path));
            }
        }

        // Window is up — start loading slides in the background.
        self.start_background_load();

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.exit_for_signal(event_loop) {
            return;
        }
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

                // Check for playlist hot-reload from management server.
                self.poll_playlist_reload();

                // Update FPS in shared AppStatus.
                self.update_fps_status(dt);

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
