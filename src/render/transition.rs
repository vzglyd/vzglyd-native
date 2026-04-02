//! Transition rendering and compositing.
//!
//! Handles crossfade, wipe, and dissolve transitions between slides.

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::gpu::context::{GpuContext, OffscreenTarget};

/// Uniforms for transition rendering.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TransitionUniforms {
    pub blend_factor: f32,
    pub transition_kind: u32,
    pub padding: [u32; 2],
}

/// Transition kind matching the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TransitionKind {
    Crossfade = 0,
    WipeLeft = 1,
    WipeDown = 2,
    Dissolve = 3,
    Cut = 4,
}

impl TransitionKind {
    /// Returns whether this transition uses the compositor (two offscreen targets).
    pub const fn uses_compositor(self) -> bool {
        !matches!(self, Self::Cut)
    }

    /// Returns the shader tag for this transition kind.
    const fn shader_tag(self) -> u32 {
        match self {
            Self::Crossfade => 0,
            Self::WipeLeft => 1,
            Self::WipeDown => 2,
            Self::Dissolve => 3,
            Self::Cut => 0,
        }
    }
}

impl Default for TransitionKind {
    fn default() -> Self {
        Self::Crossfade
    }
}

/// Transition renderer with pipeline and resources.
pub struct TransitionRenderer {
    pub pipeline: Arc<wgpu::RenderPipeline>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub sampler: Arc<wgpu::Sampler>,
    pub uniform_buffer: Arc<wgpu::Buffer>,
}

/// Active transition state.
pub struct ActiveTransition {
    pub kind: TransitionKind,
    pub start_time: Instant,
    pub duration: Duration,
}

/// Transition state machine.
pub enum TransitionState {
    Idle,
    Blending(ActiveTransition),
}

impl Default for TransitionState {
    fn default() -> Self {
        Self::Idle
    }
}

impl TransitionState {
    /// Returns whether the transition is idle.
    pub fn is_idle(&self) -> bool {
        matches!(self, Self::Idle)
    }

    /// Returns the blend factor (0.0 to 1.0) based on elapsed time.
    pub fn blend_factor(&self) -> f32 {
        match self {
            TransitionState::Idle => 0.0,
            TransitionState::Blending(active) => {
                let elapsed = active.start_time.elapsed().as_secs_f32();
                let duration = active.duration.as_secs_f32();
                (elapsed / duration).min(1.0)
            }
        }
    }

    /// Returns whether the transition is complete.
    pub fn is_complete(&self) -> bool {
        match self {
            TransitionState::Idle => true,
            TransitionState::Blending(active) => active.start_time.elapsed() >= active.duration,
        }
    }
}

impl TransitionRenderer {
    /// Creates a new transition renderer.
    pub fn new(ctx: &GpuContext) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../../shaders/transition.wgsl"));

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("transition_bgl"),
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
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("transition_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transition_uniforms"),
            size: std::mem::size_of::<TransitionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("transition_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("transition_pipeline"),
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
                        format: ctx.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });

        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            sampler: Arc::new(sampler),
            uniform_buffer: Arc::new(uniform_buffer),
        }
    }

    /// Starts a new transition.
    pub fn start_transition(&self, kind: TransitionKind, duration: Duration) -> ActiveTransition {
        ActiveTransition {
            kind,
            start_time: Instant::now(),
            duration,
        }
    }

    /// Renders the transition by compositing two offscreen targets.
    ///
    /// `blend_factor` should be in [0.0, 1.0] — pass `FrameRenderState::transition_progress`
    /// directly from the kernel.
    pub fn render(
        &self,
        ctx: &GpuContext,
        blend_factor: f32,
        kind: TransitionKind,
        outgoing_target: &OffscreenTarget,
        incoming_target: &OffscreenTarget,
        output_target: &OffscreenTarget,
    ) {
        // Update uniforms
        let uniforms = TransitionUniforms {
            blend_factor: blend_factor.clamp(0.0, 1.0),
            transition_kind: kind.shader_tag(),
            padding: [0; 2],
        };
        ctx.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Create bind group for this frame
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transition_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&outgoing_target.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&incoming_target.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(
                        self.uniform_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("transition_encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("transition_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_target.color_view,
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        ctx.queue.submit(Some(encoder.finish()));
    }
}
