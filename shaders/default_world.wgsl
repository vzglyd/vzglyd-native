fn vs_main(in: VzglydVertexInput) -> VzglydVertexOutput {
    var out: VzglydVertexOutput;
    let world_pos = (vzglyd_push.model_matrix * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    let normal_mat = mat3x3<f32>(
        vzglyd_push.model_matrix[0].xyz,
        vzglyd_push.model_matrix[1].xyz,
        vzglyd_push.model_matrix[2].xyz,
    );
    out.normal = normalize(normal_mat * in.normal);
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
