//! Host-level HUD overlay rendered on top of every slide.
//!
//! Thin GPU wrapper around the platform-agnostic geometry in
//! `vzglyd_kernel::overlay`. Builds the font atlas texture once, then records a
//! render pass each frame that composites the border, footer, slide title, and
//! wall-clock time on top of slide content.

use bytemuck::cast_slice;
use chrono::Local;
use std::sync::Arc;
use vzglyd_kernel::{
    OverlayVertex, ScreensaverFrameState, build_font_atlas_pixels, build_hud_geometry_with_update,
    build_screensaver_geometry,
};

use crate::gpu::context::GpuContext;

// ── OverlayRenderer ───────────────────────────────────────────────────────────

/// GPU renderer for the host HUD overlay.
///
/// Create once with [`OverlayRenderer::new`] after the surface is configured,
/// then call [`OverlayRenderer::record_pass`] every frame.
pub struct OverlayRenderer {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    font_atlas_texture: Arc<wgpu::Texture>,
    glyph_map: std::collections::HashMap<char, [f32; 4]>,
    vertex_buffer: Arc<wgpu::Buffer>,
    index_buffer: Arc<wgpu::Buffer>,
    vertex_capacity: usize,
    index_capacity: usize,
}

impl OverlayRenderer {
    pub fn new(ctx: &GpuContext) -> Self {
        let (atlas_pixels, atlas_w, atlas_h, glyph_map) = build_font_atlas_pixels();

        let atlas_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("overlay_font_atlas"),
            size: wgpu::Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &atlas_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w * 4),
                rows_per_image: Some(atlas_h),
            },
            wgpu::Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
        );

        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("overlay_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("overlay_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../../shaders/overlay.wgsl"));

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("overlay_pipeline_layout"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let vertex_stride = std::mem::size_of::<OverlayVertex>() as u64;
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: vertex_stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 32,
                                shader_location: 3,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let initial_verts = 2048usize;
        let initial_idxs = 3072usize;

        let vertex_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_vertex_buffer"),
            size: (initial_verts * std::mem::size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_index_buffer"),
            size: (initial_idxs * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline: Arc::new(pipeline),
            bind_group,
            font_atlas_texture: Arc::new(atlas_texture),
            glyph_map,
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
            vertex_capacity: initial_verts,
            index_capacity: initial_idxs,
        }
    }

    /// Records the overlay render pass into `encoder`, targeting `view`.
    ///
    /// `blit_rect` is `(x, y, width, height)` in surface pixels — the letterbox
    /// rect the slide was blitted into. The overlay is constrained to that rect
    /// so that the border and footer align with the slide edges, not the screen edges.
    ///
    /// Must be called after the slide blit pass so that `LoadOp::Load` preserves
    /// the slide content underneath the overlay geometry.
    pub fn record_pass(
        &mut self,
        ctx: &GpuContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        slide_name: Option<&str>,
        updated_str: Option<&str>,
        blit_rect: (u32, u32, u32, u32),
    ) {
        let clock_str = Local::now().format("%H:%M:%S").to_string();
        let (vp_x, vp_y, sw, sh) = blit_rect;

        let (vertices, indices): (Vec<OverlayVertex>, Vec<u16>) =
            build_hud_geometry_with_update(
                &self.glyph_map,
                sw,
                sh,
                slide_name,
                &clock_str,
                updated_str,
            );

        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len().next_power_of_two();
            self.vertex_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_vertex_buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<OverlayVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if indices.len() > self.index_capacity {
            self.index_capacity = indices.len().next_power_of_two();
            self.index_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_index_buffer"),
                size: (self.index_capacity * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        ctx.queue
            .write_buffer(&self.vertex_buffer, 0, cast_slice(&vertices));
        ctx.queue
            .write_buffer(&self.index_buffer, 0, cast_slice(&indices));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_viewport(vp_x as f32, vp_y as f32, sw as f32, sh as f32, 0.0, 1.0);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }

    /// Records the screensaver overlay pass into `encoder`, targeting `view`.
    ///
    /// Renders the full-screen intermission geometry (black background + drifting
    /// "Intermission" text + countdown) using the same pipeline as [`record_pass`].
    /// Call this instead of [`record_pass`] when the screensaver is active; no
    /// slide blit is needed beforehand since the geometry includes a full-screen
    /// black background quad.
    pub fn record_screensaver_pass(
        &mut self,
        ctx: &GpuContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        state: &ScreensaverFrameState,
        blit_rect: (u32, u32, u32, u32),
    ) {
        let (vp_x, vp_y, sw, sh) = blit_rect;

        let (vertices, indices) = build_screensaver_geometry(
            &self.glyph_map,
            sw,
            sh,
            state.elapsed_secs,
            state.remaining_secs,
        );

        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len().next_power_of_two();
            self.vertex_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_vertex_buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<OverlayVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if indices.len() > self.index_capacity {
            self.index_capacity = indices.len().next_power_of_two();
            self.index_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay_index_buffer"),
                size: (self.index_capacity * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        ctx.queue.write_buffer(&self.vertex_buffer, 0, cast_slice(&vertices));
        ctx.queue.write_buffer(&self.index_buffer, 0, cast_slice(&indices));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("screensaver_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_viewport(vp_x as f32, vp_y as f32, sw as f32, sh as f32, 0.0, 1.0);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
    }
}
