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
    let normal = normalize(in.normal);
    let light_dir = vzglyd_main_light_dir();
    let diffuse = max(dot(normal, light_dir), 0.0) * vzglyd_direct_light_scale();
    let lit = in.color.rgb * (vzglyd_ambient_light() + diffuse * vzglyd_main_light_rgb());
    return vec4<f32>(lit, in.color.a);
}
