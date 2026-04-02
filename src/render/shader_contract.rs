//! WGSL shader contract assembly for ABI-1 slides.

/// Supported shader contracts.
#[derive(Clone, Copy)]
pub enum ShaderContract {
    Screen2D,
    World3D,
}

const SCREEN2D_SHADER_PRELUDE: &str = r#"// VZGLYD shader contract v1: Screen2D
const VZGLYD_SHADER_CONTRACT_VERSION: u32 = 1u;

struct VzglydVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: f32,
};

struct VzglydVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) mode: f32,
};

struct VzglydUniforms {
    time: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var t_font: texture_2d<f32>;
@group(0) @binding(2) var t_detail: texture_2d<f32>;
@group(0) @binding(3) var t_lookup: texture_2d<f32>;
@group(0) @binding(4) var s_diffuse: sampler;
@group(0) @binding(5) var s_font: sampler;
@group(0) @binding(6) var<uniform> u: VzglydUniforms;
"#;

const WORLD3D_SHADER_PRELUDE: &str = r#"// VZGLYD shader contract v1: World3D
const VZGLYD_SHADER_CONTRACT_VERSION: u32 = 1u;

struct VzglydVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: f32,
};

struct VzglydVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: f32,
};

struct VzglydUniforms {
    view_proj: mat4x4<f32>,
    cam_pos: vec3<f32>,
    time: f32,
    fog_color: vec4<f32>,
    fog_start: f32,
    fog_end: f32,
    clock_seconds: f32,
    _pad: f32,
    ambient_light: vec4<f32>,
    main_light_dir: vec4<f32>,
    main_light_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: VzglydUniforms;
@group(0) @binding(1) var t_font: texture_2d<f32>;
@group(0) @binding(2) var t_noise: texture_2d<f32>;
@group(0) @binding(3) var t_material_a: texture_2d<f32>;
@group(0) @binding(4) var t_material_b: texture_2d<f32>;
@group(0) @binding(5) var s_clamp: sampler;
@group(0) @binding(6) var s_repeat: sampler;

fn vzglyd_ambient_light() -> vec3<f32> {
    return u.ambient_light.rgb;
}

fn vzglyd_main_light_dir() -> vec3<f32> {
    let dir = u.main_light_dir.xyz;
    let len_sq = dot(dir, dir);
    if len_sq <= 0.000001 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return normalize(dir);
}

fn vzglyd_main_light_rgb() -> vec3<f32> {
    return u.main_light_color.rgb;
}

fn vzglyd_main_light_strength() -> f32 {
    return max(max(u.main_light_color.r, u.main_light_color.g), u.main_light_color.b);
}

fn vzglyd_direct_light_scale() -> f32 {
    let ambient = vzglyd_ambient_light();
    return max(1.0 - max(max(ambient.r, ambient.g), ambient.b), 0.0);
}

fn vzglyd_main_light_screen_uv() -> vec2<f32> {
    let dir = vzglyd_main_light_dir();
    return clamp(
        vec2<f32>(0.5 + dir.x * 0.22, 0.5 - dir.y * 0.30),
        vec2<f32>(0.05, 0.05),
        vec2<f32>(0.95, 0.95),
    );
}
"#;

/// Assemble a complete WGSL module from the fixed prelude plus the slide shader body.
pub fn assemble_slide_shader_source(contract: ShaderContract, shader_body: &str) -> String {
    let prelude = match contract {
        ShaderContract::Screen2D => SCREEN2D_SHADER_PRELUDE,
        ShaderContract::World3D => WORLD3D_SHADER_PRELUDE,
    };
    format!("{prelude}\n{shader_body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_shader_prelude_exposes_uniforms_and_interfaces() {
        let shader = assemble_slide_shader_source(
            ShaderContract::World3D,
            r#"
fn vs_main(in: VzglydVertexInput) -> VzglydVertexOutput {
    var out: VzglydVertexOutput;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.normal = in.normal;
    out.color = in.color;
    out.mode = in.mode;
    return out;
}

fn fs_main(in: VzglydVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(u.time, in.color.yzw);
}
"#,
        );

        naga::front::wgsl::parse_str(&shader).expect("world contract shader parses");
    }

    #[test]
    fn screen_shader_prelude_exposes_uniforms_and_interfaces() {
        let shader = assemble_slide_shader_source(
            ShaderContract::Screen2D,
            r#"
fn vs_main(in: VzglydVertexInput) -> VzglydVertexOutput {
    var out: VzglydVertexOutput;
    out.clip_pos = vec4<f32>(in.position, 1.0);
    out.tex_coords = in.tex_coords;
    out.color = in.color;
    out.mode = in.mode;
    return out;
}

fn fs_main(in: VzglydVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(u.time, in.color.yzw);
}
"#,
        );

        naga::front::wgsl::parse_str(&shader).expect("screen contract shader parses");
    }
}
