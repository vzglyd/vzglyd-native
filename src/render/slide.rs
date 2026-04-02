//! Slide rendering for Screen and World slides.
//!
//! Handles slide specification parsing, pipeline creation, and rendering.

use bytemuck::Pod;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use vzglyd_slide::{
    DrawSource, DrawSpec, FilterMode, PipelineKind, ScreenVertex, ShaderSources, SlideSpec,
    StaticMesh, TextureDesc, WorldLighting, WorldVertex,
};

use crate::gpu::context::{GpuContext, HEIGHT, OffscreenTarget, WIDTH};
use crate::render::shader_contract::{ShaderContract, assemble_slide_shader_source};
use crate::slide::{DecodedSlideSpec, decode_slide_spec};
use crate::utils::clock::melbourne_clock_seconds;

/// Uniforms for screen-space slides.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniforms {
    pub time: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

/// Uniforms for world-space slides.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 3],
    pub time: f32,
    pub fog_color: [f32; 4],
    pub fog_start: f32,
    pub fog_end: f32,
    pub clock_seconds: f32,
    pub _pad: f32,
    pub ambient_light: [f32; 4],
    pub main_light_dir: [f32; 4],
    pub main_light_color: [f32; 4],
}

/// Static mesh buffers for rendering.
pub struct MeshBuffers {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
}

/// Dynamic mesh buffers for animated geometry.
pub struct DynamicMeshBuffers {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub vertex_capacity: u32,
    pub index_capacity: u32,
    pub current_index_count: u32,
}

/// Texture resource with sampler.
pub struct SlideTexture {
    pub texture: Arc<wgpu::Texture>,
    pub view: wgpu::TextureView,
    pub sampler: Arc<wgpu::Sampler>,
}

/// Bind group for screen slides.
pub struct ScreenBindGroup {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: Arc<wgpu::Buffer>,
}

/// Bind group for world slides.
pub struct WorldBindGroup {
    pub bind_group: wgpu::BindGroup,
    pub uniform_buffer: Arc<wgpu::Buffer>,
}

/// Render pipeline pair (opaque + transparent).
pub struct SlidePipelines {
    pub opaque: Option<Arc<wgpu::RenderPipeline>>,
    pub transparent: Option<Arc<wgpu::RenderPipeline>>,
}

impl SlidePipelines {
    pub fn get(&self, kind: PipelineKind) -> Option<&wgpu::RenderPipeline> {
        match kind {
            PipelineKind::Opaque => self.opaque.as_ref().map(|p| p.as_ref()),
            PipelineKind::Transparent => self.transparent.as_ref().map(|p| p.as_ref()),
        }
    }
}

/// Screen slide renderer.
pub struct ScreenSlideRenderer {
    pub spec: SlideSpec<ScreenVertex>,
    pub pipelines: SlidePipelines,
    pub bind_group: ScreenBindGroup,
    pub textures: Vec<SlideTexture>,
    pub font_texture: Option<SlideTexture>,
    pub static_meshes: Vec<MeshBuffers>,
    pub elapsed: f32,
}

/// World slide renderer.
pub struct WorldSlideRenderer {
    pub spec: SlideSpec<WorldVertex>,
    pub pipelines: SlidePipelines,
    pub bind_group: WorldBindGroup,
    pub textures: Vec<SlideTexture>,
    pub static_meshes: Vec<MeshBuffers>,
    pub dynamic_meshes: Vec<DynamicMeshBuffers>,
    pub lighting: WorldLighting,
    pub elapsed: f32,
}

/// Slide renderer enum.
pub enum SlideRenderer {
    Screen(ScreenSlideRenderer),
    World(WorldSlideRenderer),
}

impl ScreenSlideRenderer {
    /// Creates a new screen slide renderer.
    pub fn new(ctx: &GpuContext, spec: SlideSpec<ScreenVertex>) -> Result<Self, String> {
        let mut textures = load_slide_textures(&ctx.device, &ctx.queue, &spec.textures)?;
        if textures.is_empty() {
            textures.push(create_solid_texture(
                &ctx.device,
                &ctx.queue,
                "screen_fallback",
                255,
                255,
                255,
                255,
            )?);
        }

        let font_texture = spec
            .font
            .as_ref()
            .map(|font| create_font_texture(&ctx.device, &ctx.queue, font))
            .transpose()?;

        let tex_view = &textures[0].view;
        let detail_view = textures
            .get(1)
            .map(|texture| &texture.view)
            .unwrap_or(tex_view);
        let lookup_view = textures
            .get(2)
            .map(|texture| &texture.view)
            .unwrap_or(detail_view);
        let font_view = font_texture
            .as_ref()
            .map(|texture| &texture.view)
            .unwrap_or(tex_view);
        let sampler = &textures[0].sampler;
        let font_sampler = font_texture
            .as_ref()
            .map(|texture| &texture.sampler)
            .unwrap_or(sampler);

        let static_meshes: Vec<MeshBuffers> = spec
            .static_meshes
            .iter()
            .map(|mesh| create_static_mesh_buffers(&ctx.device, mesh))
            .collect();

        // Create uniform buffer
        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screen_slide_uniforms"),
            size: std::mem::size_of::<ScreenUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (bind_group_layout, bind_group) = make_screen_bind_group(
            &ctx.device,
            tex_view,
            font_view,
            detail_view,
            lookup_view,
            sampler,
            font_sampler,
            &uniform_buffer,
        );

        let shader_source = resolve_slide_shader_source(
            ShaderContract::Screen2D,
            spec.shaders.as_ref(),
            include_str!("../../shaders/default_screen.wgsl"),
        );

        let pipelines = create_screen_slide_pipelines(
            ctx,
            &spec.draws,
            &bind_group_layout,
            &shader_source,
            "screen_slide",
        )?;

        Ok(Self {
            spec,
            pipelines,
            bind_group: ScreenBindGroup {
                bind_group,
                uniform_buffer: Arc::new(uniform_buffer),
            },
            textures,
            font_texture,
            static_meshes,
            elapsed: 0.0,
        })
    }

    /// Updates the slide and returns whether it changed.
    pub fn update(&mut self, dt: f32) -> bool {
        self.elapsed += dt;
        true
    }

    /// Renders the slide to an offscreen target.
    pub fn render(&self, ctx: &GpuContext, target: &OffscreenTarget) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screen_slide_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("screen_slide_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &self.bind_group.bind_group, &[]);

            // Update uniforms
            let uniforms = ScreenUniforms {
                time: self.elapsed,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            };
            ctx.queue.write_buffer(
                &self.bind_group.uniform_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );

            for draw in &self.spec.draws {
                if let DrawSource::Static(mesh_idx) = draw.source {
                    if let Some(mesh) = self.static_meshes.get(mesh_idx) {
                        if let Some(pipeline) = self.pipelines.get(draw.pipeline) {
                            let draw_range_end = draw.index_range.end.min(mesh.index_count);
                            if draw.index_range.start < draw_range_end {
                                pass.set_pipeline(pipeline);
                                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint16,
                                );
                                pass.draw_indexed(draw.index_range.start..draw_range_end, 0, 0..1);
                            }
                        }
                    }
                }
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
    }
}

impl WorldSlideRenderer {
    /// Creates a new world slide renderer.
    pub fn new(ctx: &GpuContext, spec: SlideSpec<WorldVertex>) -> Result<Self, String> {
        let mut textures = load_slide_textures(&ctx.device, &ctx.queue, &spec.textures)?;
        if textures.is_empty() {
            textures.push(create_solid_texture(
                &ctx.device,
                &ctx.queue,
                "world_fallback",
                255,
                255,
                255,
                255,
            )?);
        }

        // Create static meshes
        let static_meshes: Vec<MeshBuffers> = spec
            .static_meshes
            .iter()
            .map(|mesh| create_static_mesh_buffers(&ctx.device, mesh))
            .collect();

        // Create dynamic mesh buffers
        let dynamic_meshes: Vec<DynamicMeshBuffers> = spec
            .dynamic_meshes
            .iter()
            .map(|mesh| create_dynamic_mesh_buffers(&ctx.device, mesh))
            .collect();

        // Create uniform buffer
        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world_slide_uniforms"),
            size: std::mem::size_of::<WorldUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lighting = spec.lighting.clone().unwrap_or_default();

        let font_view = &textures[0].view;
        let secondary_view = textures
            .get(1)
            .map(|texture| &texture.view)
            .unwrap_or(font_view);
        let material_a_view = textures
            .get(2)
            .map(|texture| &texture.view)
            .unwrap_or(secondary_view);
        let material_b_view = textures
            .get(3)
            .map(|texture| &texture.view)
            .unwrap_or(material_a_view);
        let font_sampler = &textures[0].sampler;
        let secondary_sampler = textures
            .get(1)
            .map(|texture| &texture.sampler)
            .unwrap_or(font_sampler);

        let (bind_group_layout, bind_group) = make_world_bind_group(
            &ctx.device,
            &uniform_buffer,
            font_view,
            secondary_view,
            material_a_view,
            material_b_view,
            font_sampler,
            secondary_sampler,
        );

        let shader_source = resolve_slide_shader_source(
            ShaderContract::World3D,
            spec.shaders.as_ref(),
            include_str!("../../shaders/default_world.wgsl"),
        );

        let pipelines = create_slide_pipelines(
            ctx,
            &spec.draws,
            &bind_group_layout,
            &shader_source,
            "world_slide",
        )?;

        Ok(Self {
            spec,
            pipelines,
            bind_group: WorldBindGroup {
                bind_group,
                uniform_buffer: Arc::new(uniform_buffer),
            },
            textures,
            static_meshes,
            dynamic_meshes,
            lighting,
            elapsed: 0.0,
        })
    }

    /// Updates the slide and returns whether it changed.
    pub fn update(&mut self, dt: f32) -> bool {
        self.elapsed += dt;
        true
    }

    /// Renders the slide to an offscreen target.
    pub fn render(&self, ctx: &GpuContext, target: &OffscreenTarget) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("world_slide_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world_slide_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            pass.set_bind_group(0, &self.bind_group.bind_group, &[]);

            // Update uniforms
            let uniforms = self.build_uniforms();
            ctx.queue.write_buffer(
                &self.bind_group.uniform_buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );

            // Draw static meshes
            for draw in &self.spec.draws {
                if let DrawSource::Static(mesh_idx) = draw.source {
                    if let Some(mesh) = self.static_meshes.get(mesh_idx) {
                        if let Some(pipeline) = self.pipelines.get(draw.pipeline) {
                            let draw_range_end = draw.index_range.end.min(mesh.index_count);
                            if draw.index_range.start < draw_range_end {
                                pass.set_pipeline(pipeline);
                                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                                pass.set_index_buffer(
                                    mesh.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint16,
                                );
                                pass.draw_indexed(draw.index_range.start..draw_range_end, 0, 0..1);
                            }
                        }
                    }
                }
            }
        }

        ctx.queue.submit(Some(encoder.finish()));
    }

    fn build_uniforms(&self) -> WorldUniforms {
        let (eye, target, up, fov_y_deg) = sample_camera(&self.spec.camera_path, self.elapsed);
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh(
            fov_y_deg.to_radians(),
            WIDTH as f32 / HEIGHT as f32,
            0.15,
            180.0,
        );
        let view_proj_mat = proj * view;
        let view_proj = view_proj_mat.to_cols_array_2d();

        WorldUniforms {
            view_proj,
            cam_pos: eye.to_array(),
            time: self.elapsed,
            fog_color: [0.0, 0.0, 0.0, 1.0],
            fog_start: 18.0,
            fog_end: 75.0,
            clock_seconds: melbourne_clock_seconds(),
            _pad: 0.0,
            ambient_light: pack_ambient_light(&self.lighting),
            main_light_dir: pack_main_light_dir(&self.lighting),
            main_light_color: pack_main_light_color(&self.lighting),
        }
    }
}

/// Creates texture and sampler from descriptor.
fn create_texture_from_desc(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    desc: &TextureDesc,
) -> Result<SlideTexture, String> {
    let format = match desc.format {
        vzglyd_slide::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&desc.label),
        size: wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &desc.data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(desc.width * 4),
            rows_per_image: Some(desc.height),
        },
        wgpu::Extent3d {
            width: desc.width,
            height: desc.height,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mag_filter = match desc.mag_filter {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    };
    let min_filter = match desc.min_filter {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    };
    let address_mode_u = match desc.wrap_u {
        vzglyd_slide::WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        vzglyd_slide::WrapMode::Repeat => wgpu::AddressMode::Repeat,
    };
    let address_mode_v = match desc.wrap_v {
        vzglyd_slide::WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        vzglyd_slide::WrapMode::Repeat => wgpu::AddressMode::Repeat,
    };
    let address_mode_w = match desc.wrap_w {
        vzglyd_slide::WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        vzglyd_slide::WrapMode::Repeat => wgpu::AddressMode::Repeat,
    };
    let mipmap_filter = match desc.mip_filter {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    };

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(&desc.label),
        address_mode_u,
        address_mode_v,
        address_mode_w,
        mag_filter,
        min_filter,
        mipmap_filter,
        ..Default::default()
    });

    Ok(SlideTexture {
        texture: Arc::new(texture),
        view,
        sampler: Arc::new(sampler),
    })
}

fn create_solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> Result<SlideTexture, String> {
    create_texture_from_desc(
        device,
        queue,
        &TextureDesc {
            label: label.to_string(),
            width: 1,
            height: 1,
            format: vzglyd_slide::TextureFormat::Rgba8Unorm,
            wrap_u: vzglyd_slide::WrapMode::ClampToEdge,
            wrap_v: vzglyd_slide::WrapMode::ClampToEdge,
            wrap_w: vzglyd_slide::WrapMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            data: vec![r, g, b, a],
        },
    )
}

fn create_font_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    font: &vzglyd_slide::FontAtlas,
) -> Result<SlideTexture, String> {
    create_texture_from_desc(
        device,
        queue,
        &TextureDesc {
            label: "font_atlas".into(),
            width: font.width,
            height: font.height,
            format: vzglyd_slide::TextureFormat::Rgba8Unorm,
            wrap_u: vzglyd_slide::WrapMode::ClampToEdge,
            wrap_v: vzglyd_slide::WrapMode::ClampToEdge,
            wrap_w: vzglyd_slide::WrapMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            data: font.pixels.clone(),
        },
    )
}

fn load_slide_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures: &[TextureDesc],
) -> Result<Vec<SlideTexture>, String> {
    textures
        .iter()
        .map(|desc| create_texture_from_desc(device, queue, desc))
        .collect()
}

/// Creates vertex and index buffers for a static mesh.
fn create_static_mesh_buffers<V: Pod>(device: &wgpu::Device, mesh: &StaticMesh<V>) -> MeshBuffers {
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("static_vertex_buffer"),
        size: mesh.vertices.len() as u64 * std::mem::size_of::<V>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    vertex_buffer.slice(..).get_mapped_range_mut()
        [..mesh.vertices.len() * std::mem::size_of::<V>()]
        .copy_from_slice(bytemuck::cast_slice(&mesh.vertices));
    vertex_buffer.unmap();

    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("static_index_buffer"),
        size: mesh.indices.len() as u64 * 2,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    index_buffer.slice(..).get_mapped_range_mut()[..mesh.indices.len() * 2]
        .copy_from_slice(bytemuck::cast_slice(&mesh.indices));
    index_buffer.unmap();

    MeshBuffers {
        vertex_buffer: Arc::new(vertex_buffer),
        index_buffer: Arc::new(index_buffer),
        index_count: mesh.indices.len() as u32,
    }
}

/// Creates dynamic mesh buffers.
fn create_dynamic_mesh_buffers(
    device: &wgpu::Device,
    mesh: &vzglyd_slide::DynamicMesh,
) -> DynamicMeshBuffers {
    let vertex_capacity = mesh.max_vertices;
    let index_capacity = mesh.indices.len() as u32;

    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dynamic_vertex_buffer"),
        size: vertex_capacity as u64 * std::mem::size_of::<WorldVertex>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dynamic_index_buffer"),
        size: index_capacity as u64 * 2,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    DynamicMeshBuffers {
        vertex_buffer: Arc::new(vertex_buffer),
        index_buffer: Arc::new(index_buffer),
        vertex_capacity,
        index_capacity,
        current_index_count: 0,
    }
}

fn bgl_texture(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn bgl_sampler(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn bgl_uniform(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn make_screen_bind_group(
    device: &wgpu::Device,
    tex_view: &wgpu::TextureView,
    font_view: &wgpu::TextureView,
    detail_view: &wgpu::TextureView,
    lookup_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    font_sampler: &wgpu::Sampler,
    uniform_buffer: &wgpu::Buffer,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("screen_slide_bgl"),
        entries: &[
            bgl_texture(0, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(1, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(2, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(3, wgpu::ShaderStages::FRAGMENT),
            bgl_sampler(4, wgpu::ShaderStages::FRAGMENT),
            bgl_sampler(5, wgpu::ShaderStages::FRAGMENT),
            bgl_uniform(6, wgpu::ShaderStages::FRAGMENT),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("screen_slide_bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(font_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(detail_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(lookup_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(font_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });
    (layout, bind_group)
}

fn make_world_bind_group(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
    font_view: &wgpu::TextureView,
    secondary_view: &wgpu::TextureView,
    material_a_view: &wgpu::TextureView,
    material_b_view: &wgpu::TextureView,
    font_sampler: &wgpu::Sampler,
    secondary_sampler: &wgpu::Sampler,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("world_slide_bgl"),
        entries: &[
            bgl_uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            bgl_texture(1, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(2, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(3, wgpu::ShaderStages::FRAGMENT),
            bgl_texture(4, wgpu::ShaderStages::FRAGMENT),
            bgl_sampler(5, wgpu::ShaderStages::FRAGMENT),
            bgl_sampler(6, wgpu::ShaderStages::FRAGMENT),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("world_slide_bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(font_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(secondary_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(material_a_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(material_b_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(font_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Sampler(secondary_sampler),
            },
        ],
    });
    (layout, bind_group)
}

fn resolve_slide_shader_source(
    contract: ShaderContract,
    shaders: Option<&ShaderSources>,
    default_shader_body: &str,
) -> String {
    let shader_body = shaders
        .and_then(|sources| {
            sources
                .fragment_wgsl
                .as_deref()
                .or(sources.vertex_wgsl.as_deref())
        })
        .unwrap_or(default_shader_body);
    assemble_slide_shader_source(contract, shader_body)
}

fn pack_ambient_light(lighting: &WorldLighting) -> [f32; 4] {
    let intensity = lighting.ambient_intensity.max(0.0);
    [
        lighting.ambient_color[0] * intensity,
        lighting.ambient_color[1] * intensity,
        lighting.ambient_color[2] * intensity,
        0.0,
    ]
}

fn pack_main_light_dir(lighting: &WorldLighting) -> [f32; 4] {
    let Some(light) = lighting.directional_light else {
        return [0.0, 1.0, 0.0, 0.0];
    };

    let dir = Vec3::from_array(light.direction).normalize_or_zero();
    if dir.length_squared() == 0.0 {
        [0.0, 1.0, 0.0, 0.0]
    } else {
        [dir.x, dir.y, dir.z, 1.0]
    }
}

fn pack_main_light_color(lighting: &WorldLighting) -> [f32; 4] {
    let Some(light) = lighting.directional_light else {
        return [0.0, 0.0, 0.0, 0.0];
    };

    let intensity = light.intensity.max(0.0);
    [
        light.color[0] * intensity,
        light.color[1] * intensity,
        light.color[2] * intensity,
        0.0,
    ]
}

fn sample_camera(path: &Option<vzglyd_slide::CameraPath>, elapsed: f32) -> (Vec3, Vec3, Vec3, f32) {
    if let Some(path) = path {
        if path.keyframes.len() >= 2 {
            let duration = path
                .keyframes
                .last()
                .expect("camera path should not be empty")
                .time;
            let t = if duration > 0.0 && path.looped {
                elapsed % duration
            } else {
                elapsed.min(duration)
            };
            let mut index = 0;
            while index + 1 < path.keyframes.len() && path.keyframes[index + 1].time < t {
                index += 1;
            }

            let a = &path.keyframes[index];
            let b = &path.keyframes[(index + 1).min(path.keyframes.len() - 1)];
            let span = (b.time - a.time).max(0.0001);
            let lerp_t = ((t - a.time) / span).clamp(0.0, 1.0);
            let eye = Vec3::new(
                lerp(a.position[0], b.position[0], lerp_t),
                lerp(a.position[1], b.position[1], lerp_t),
                lerp(a.position[2], b.position[2], lerp_t),
            );
            let target = Vec3::new(
                lerp(a.target[0], b.target[0], lerp_t),
                lerp(a.target[1], b.target[1], lerp_t),
                lerp(a.target[2], b.target[2], lerp_t),
            );
            let up = Vec3::new(
                lerp(a.up[0], b.up[0], lerp_t),
                lerp(a.up[1], b.up[1], lerp_t),
                lerp(a.up[2], b.up[2], lerp_t),
            )
            .normalize_or_zero();
            let fov_y = lerp(a.fov_y_deg, b.fov_y_deg, lerp_t);
            return (eye, target, up, fov_y);
        }
    }

    let t = smoothstep((elapsed % 24.0) / 24.0);
    let eye = Vec3::new(lerp(14.0, -3.0, t), 4.5, lerp(50.0, 24.0, t));
    let target = Vec3::new(0.0, 2.5, 0.0);
    (eye, target, Vec3::Y, 60.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Creates render pipelines for a slide.
fn create_slide_pipelines(
    ctx: &GpuContext,
    draw_plan: &[DrawSpec],
    bind_group_layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label_prefix: &str,
) -> Result<SlidePipelines, String> {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label_prefix)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_pipeline_layout", label_prefix)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

    let has_opaque = draw_plan.iter().any(|d| d.pipeline == PipelineKind::Opaque);
    let has_transparent = draw_plan
        .iter()
        .any(|d| d.pipeline == PipelineKind::Transparent);

    let opaque = has_opaque.then(|| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}_pipeline_opaque", label_prefix)),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[WorldVertex::desc()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::gpu::context::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
    });

    let transparent = has_transparent.then(|| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}_pipeline_transparent", label_prefix)),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[WorldVertex::desc()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::gpu::context::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
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
                multiview: None,
                cache: None,
            })
    });

    Ok(SlidePipelines {
        opaque: opaque.map(|p| Arc::new(p)),
        transparent: transparent.map(|p| Arc::new(p)),
    })
}

/// Creates render pipelines for screen slides.
fn create_screen_slide_pipelines(
    ctx: &GpuContext,
    draw_plan: &[DrawSpec],
    bind_group_layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label_prefix: &str,
) -> Result<SlidePipelines, String> {
    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}_shader", label_prefix)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_pipeline_layout", label_prefix)),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

    let has_opaque = draw_plan.iter().any(|d| d.pipeline == PipelineKind::Opaque);
    let has_transparent = draw_plan
        .iter()
        .any(|d| d.pipeline == PipelineKind::Transparent);

    let opaque = has_opaque.then(|| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}_pipeline_opaque", label_prefix)),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ScreenVertex::desc()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::gpu::context::DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            })
    });

    let transparent = has_transparent.then(|| {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}_pipeline_transparent", label_prefix)),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[ScreenVertex::desc()],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::gpu::context::DEPTH_FORMAT,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
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
                multiview: None,
                cache: None,
            })
    });

    Ok(SlidePipelines {
        opaque: opaque.map(|p| Arc::new(p)),
        transparent: transparent.map(|p| Arc::new(p)),
    })
}

/// Creates a slide renderer based on the scene space.
pub fn create_slide_renderer(ctx: &GpuContext, spec_bytes: &[u8]) -> Result<SlideRenderer, String> {
    match decode_slide_spec(spec_bytes)? {
        DecodedSlideSpec::World(spec) => {
            Ok(SlideRenderer::World(WorldSlideRenderer::new(ctx, spec)?))
        }
        DecodedSlideSpec::Screen(spec) => {
            Ok(SlideRenderer::Screen(ScreenSlideRenderer::new(ctx, spec)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use wasmtime::Module;

    #[test]
    fn packaged_clock_world_shader_parses_with_contract_prelude() {
        let archive_path = Path::new("slides/clock.vzglyd");
        if !archive_path.exists() {
            return;
        }

        let extracted = crate::assets::archive::extract_archive(archive_path).expect("extract");
        let runtime = crate::wasm::WasmRuntime::new().expect("runtime");
        let module_path = extracted.path.join("slide.wasm");
        let module = Module::from_file(&runtime.engine, &module_path).expect("module");
        let mut instance = crate::slide::SlideInstance::new(&module).expect("instance");
        let spec_bytes = instance.read_spec_bytes().expect("spec bytes");

        let spec = match crate::slide::decode_slide_spec(&spec_bytes).expect("decode spec") {
            DecodedSlideSpec::World(spec) => spec,
            DecodedSlideSpec::Screen(_) => panic!("clock slide should decode as world"),
        };

        let shader_source = resolve_slide_shader_source(
            ShaderContract::World3D,
            spec.shaders.as_ref(),
            include_str!("../../shaders/default_world.wgsl"),
        );

        naga::front::wgsl::parse_str(&shader_source).expect("assembled clock shader parses");
    }
}
