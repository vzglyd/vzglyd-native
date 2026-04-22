//! GPU context management for wgpu.
//!
//! Handles device, queue, surface, offscreen targets, and resource management.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{
    BindGroupLayout, Buffer, Device, Queue, RenderPipeline, Sampler, Surface, SurfaceConfiguration,
    Texture, TextureView,
};
use winit::{dpi::PhysicalSize, window::Window};

use vzglyd_kernel::{BufferHandle, BufferUsage, SamplerHandle, TextureFormat, TextureHandle};

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Resource registry tracking GPU resources by handle.
pub struct ResourceRegistry {
    pub textures: HashMap<u32, Arc<Texture>>,
    pub samplers: HashMap<u32, Arc<Sampler>>,
    pub buffers: HashMap<u32, Arc<Buffer>>,
    pub bind_group_layouts: HashMap<String, Arc<BindGroupLayout>>,
    pub pipelines: HashMap<String, Arc<RenderPipeline>>,
    next_texture_id: u32,
    next_sampler_id: u32,
    next_buffer_id: u32,
}

impl ResourceRegistry {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            samplers: HashMap::new(),
            buffers: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            pipelines: HashMap::new(),
            next_texture_id: 1,
            next_sampler_id: 1,
            next_buffer_id: 1,
        }
    }

    pub fn alloc_texture_handle(&mut self) -> TextureHandle {
        let id = self.next_texture_id;
        self.next_texture_id += 1;
        TextureHandle(id)
    }

    pub fn alloc_sampler_handle(&mut self) -> SamplerHandle {
        let id = self.next_sampler_id;
        self.next_sampler_id += 1;
        SamplerHandle(id)
    }

    pub fn alloc_buffer_handle(&mut self) -> BufferHandle {
        let id = self.next_buffer_id;
        self.next_buffer_id += 1;
        BufferHandle(id)
    }

    pub fn insert_texture(&mut self, handle: TextureHandle, texture: Texture) {
        self.textures.insert(handle.0, Arc::new(texture));
    }

    pub fn insert_sampler(&mut self, handle: SamplerHandle, sampler: Sampler) {
        self.samplers.insert(handle.0, Arc::new(sampler));
    }

    pub fn insert_buffer(&mut self, handle: BufferHandle, buffer: Buffer) {
        self.buffers.insert(handle.0, Arc::new(buffer));
    }

    pub fn get_texture(&self, handle: TextureHandle) -> Option<Arc<Texture>> {
        self.textures.get(&handle.0).cloned()
    }

    pub fn get_sampler(&self, handle: SamplerHandle) -> Option<Arc<Sampler>> {
        self.samplers.get(&handle.0).cloned()
    }

    pub fn get_buffer(&self, handle: BufferHandle) -> Option<Arc<Buffer>> {
        self.buffers.get(&handle.0).cloned()
    }

    pub fn remove_texture(&mut self, handle: TextureHandle) {
        self.textures.remove(&handle.0);
    }

    pub fn remove_buffer(&mut self, handle: BufferHandle) {
        self.buffers.remove(&handle.0);
    }

    pub fn insert_bind_group_layout(&mut self, key: String, layout: BindGroupLayout) {
        self.bind_group_layouts.insert(key, Arc::new(layout));
    }

    pub fn get_bind_group_layout(&self, key: &str) -> Option<Arc<BindGroupLayout>> {
        self.bind_group_layouts.get(key).cloned()
    }

    pub fn insert_pipeline(&mut self, key: String, pipeline: RenderPipeline) {
        self.pipelines.insert(key, Arc::new(pipeline));
    }

    pub fn get_pipeline(&self, key: &str) -> Option<Arc<RenderPipeline>> {
        self.pipelines.get(key).cloned()
    }
}

impl Default for ResourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU context holding device, queue, surface, and configuration.
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
    pub depth_view: TextureView,
    pub window: Arc<Window>,
    pub registry: ResourceRegistry,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub blit_bind_group_layout: Arc<BindGroupLayout>,
    pub blit_sampler: Arc<Sampler>,
    /// Uniform buffer holding the current display scale for the blit shader.
    pub blit_scale_buf: Arc<wgpu::Buffer>,
    /// The active display scale (default 1.0).
    pub display_scale: f32,
}

impl GpuContext {
    /// Creates a new GPU context.
    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("Failed to create surface: {}", e))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or_else(|| "Failed to find appropriate GPU adapter".to_string())?;

        // Desktop default limits can over-request on GLES-backed systems. Match
        // the selected adapter's supported limits like `lume` does.
        let required_limits = adapter.limits();

        // PUSH_CONSTANTS is required for World3D slide shaders (var<push_constant> vzglyd_push).
        let required_features = wgpu::Features::PUSH_CONSTANTS;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vzglyd device"),
                    required_features,
                    required_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|format| !format.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_size = Self::initial_surface_size(&window);

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: surface_size.width,
            height: surface_size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let depth_view = Self::create_depth_texture(&device, config.width, config.height);
        let (blit_pipeline, blit_bind_group_layout, blit_sampler) =
            Self::create_blit_pipeline(&device, format);

        // Uniform buffer for blit display scale (16-byte minimum for wgpu uniforms).
        let blit_scale_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blit_scale_uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        // Write default scale of 1.0.
        queue.write_buffer(&blit_scale_buf, 0, bytemuck::bytes_of(&1.0f32));

        Ok(Self {
            device,
            queue,
            surface,
            config,
            depth_view,
            window,
            registry: ResourceRegistry::new(),
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            blit_scale_buf,
            display_scale: 1.0,
        })
    }

    /// Creates a depth texture view.
    fn create_depth_texture(device: &Device, width: u32, height: u32) -> TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn usable_surface_size(size: PhysicalSize<u32>) -> Option<PhysicalSize<u32>> {
        (size.width > 0 && size.height > 0).then_some(size)
    }

    fn initial_surface_size(window: &Window) -> PhysicalSize<u32> {
        Self::usable_surface_size(window.inner_size()).unwrap_or(PhysicalSize::new(WIDTH, HEIGHT))
    }

    fn sync_surface_size_to_window(&mut self) {
        if let Some(size) = Self::usable_surface_size(self.window.inner_size()) {
            self.config.width = size.width;
            self.config.height = size.height;
        }
    }

    /// Creates the blit pipeline for displaying offscreen targets.
    fn create_blit_pipeline(
        device: &Device,
        format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>, Arc<Sampler>) {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/blit.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        (
            Arc::new(pipeline),
            Arc::new(bind_group_layout),
            Arc::new(sampler),
        )
    }

    /// Creates an offscreen render target.
    pub fn create_offscreen_target(&self) -> OffscreenTarget {
        let device = &self.device;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen color"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen depth"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        OffscreenTarget {
            color_texture,
            color_view,
            depth_view,
        }
    }

    /// Resizes the surface configuration.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.config.width == width && self.config.height == height {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.reconfigure();
    }

    /// Reconfigures the surface against the current window size.
    pub fn reconfigure(&mut self) {
        self.sync_surface_size_to_window();
        self.surface.configure(&self.device, &self.config);
        self.depth_view =
            Self::create_depth_texture(&self.device, self.config.width, self.config.height);
    }

    /// Creates a bind group for blitting an offscreen target.
    pub fn create_blit_bind_group(&self, target: &OffscreenTarget) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit_bind_group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&target.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blit_scale_buf.as_entire_binding(),
                },
            ],
        })
    }

    /// Update the display scale written into the blit uniform buffer.
    ///
    /// The change takes effect on the next frame. A scale of `1.0` fills the
    /// letterbox rect exactly; values below `1.0` add a black border (useful for
    /// CRT overscan); values above `1.0` zoom in and crop the edges.
    pub fn set_display_scale(&mut self, scale: f32) {
        self.display_scale = scale;
        self.queue
            .write_buffer(&self.blit_scale_buf, 0, bytemuck::bytes_of(&scale));
    }

    /// Returns the pixel rect actually occupied by the visible slide image,
    /// accounting for the display scale.
    ///
    /// For `scale >= 1.0` this equals [`surface_blit_rect`] (the image fills or
    /// exceeds the letterbox area).  For `scale < 1.0` this is the centred,
    /// scaled-down sub-rect.  Use this rect for the HUD overlay viewport so the
    /// border aligns with the visible image edges.
    pub fn surface_display_rect(&self) -> (u32, u32, u32, u32) {
        let (bx, by, bw, bh) = self.surface_blit_rect();
        let s = self.display_scale.min(1.0); // clamp: scale > 1.0 still fills blit rect
        let sw = (bw as f32 * s).round() as u32;
        let sh = (bh as f32 * s).round() as u32;
        let sx = bx + bw.saturating_sub(sw) / 2;
        let sy = by + bh.saturating_sub(sh) / 2;
        (sx, sy, sw.max(1), sh.max(1))
    }

    /// Blits an offscreen target to the surface with letterboxing.
    pub fn blit_to_surface(
        &self,
        _target: &OffscreenTarget,
        bind_group: &wgpu::BindGroup,
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit_encoder"),
            });

        {
            let (x, y, width, height) = self.surface_blit_rect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, width, height);
            pass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Blits an offscreen target to the surface and then records an overlay pass
    /// into the same encoder before submitting and presenting.
    ///
    /// The closure receives the surface `TextureView` and a mutable encoder so
    /// the overlay can append its own render pass sharing one submit/present.
    pub fn blit_and_overlay_to_surface<F>(
        &self,
        _target: &OffscreenTarget,
        bind_group: &wgpu::BindGroup,
        record_overlay: F,
    ) -> Result<(), wgpu::SurfaceError>
    where
        F: FnOnce(&wgpu::TextureView, &mut wgpu::CommandEncoder),
    {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit_and_overlay_encoder"),
            });

        {
            let (x, y, width, height) = self.surface_blit_rect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, width, height);
            pass.draw(0..3, 0..1);
        }

        record_overlay(&view, &mut encoder);

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Clears the surface to black and records an overlay pass into the same encoder.
    ///
    /// Used for the screensaver path where no slide is blitted — the overlay
    /// geometry provides its own full-screen background.
    pub fn clear_and_overlay_to_surface<F>(
        &self,
        record_overlay: F,
    ) -> Result<(), wgpu::SurfaceError>
    where
        F: FnOnce(&wgpu::TextureView, &mut wgpu::CommandEncoder),
    {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screensaver_encoder"),
            });

        {
            // Clear to black. The screensaver geometry draws on top with LoadOp::Load.
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screensaver_clear_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
        }

        record_overlay(&view, &mut encoder);

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    /// Calculates the blit rect for letterbox/pillarbox scaling.
    pub fn surface_blit_rect(&self) -> (u32, u32, u32, u32) {
        let surface_width = self.config.width.max(1);
        let surface_height = self.config.height.max(1);
        let render_width = WIDTH as u64;
        let render_height = HEIGHT as u64;
        let surface_width_u64 = surface_width as u64;
        let surface_height_u64 = surface_height as u64;

        if surface_width_u64 * render_height > surface_height_u64 * render_width {
            let width = ((surface_height_u64 * render_width) / render_height).max(1) as u32;
            let x = surface_width.saturating_sub(width) / 2;
            (x, 0, width, surface_height)
        } else {
            let height = ((surface_width_u64 * render_height) / render_width).max(1) as u32;
            let y = surface_height.saturating_sub(height) / 2;
            (0, y, surface_width, height)
        }
    }

    /// Converts kernel TextureFormat to wgpu TextureFormat.
    pub fn texture_format_to_wgpu(format: TextureFormat) -> wgpu::TextureFormat {
        match format {
            TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        }
    }

    /// Converts kernel BufferUsage to wgpu BufferUsages.
    pub fn buffer_usage_to_wgpu(usage: BufferUsage) -> wgpu::BufferUsages {
        let mut wgpu_usage = wgpu::BufferUsages::empty();
        if usage.vertex {
            wgpu_usage |= wgpu::BufferUsages::VERTEX;
        }
        if usage.index {
            wgpu_usage |= wgpu::BufferUsages::INDEX;
        }
        if usage.uniform {
            wgpu_usage |= wgpu::BufferUsages::UNIFORM;
        }
        if usage.storage {
            wgpu_usage |= wgpu::BufferUsages::STORAGE;
        }
        wgpu_usage
    }
}

/// Offscreen render target for compositing.
pub struct OffscreenTarget {
    pub color_texture: Texture,
    pub color_view: TextureView,
    pub depth_view: TextureView,
}
