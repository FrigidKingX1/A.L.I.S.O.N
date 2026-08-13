// A.L.I.S.O.N. Brain Radar -- 6-axis homeostatic drive visualizer.
// QML ShaderEffect supplies qt_TexCoord0 (0..1) and the uniform values.
varying vec2 qt_TexCoord0;

uniform float u_gamma;
uniform float u_drives[6];
uniform float u_time;

void main() {
    vec2 uv = qt_TexCoord0 * 2.0 - 1.0;
    float ang = atan(uv.y, uv.x);          // -PI .. PI
    float r = length(uv);

    // Snap to one of 6 sectors (60 deg each).
    float sector = floor(((ang + 3.14159265) / (2.0 * 3.14159265)) * 6.0 + 0.5);
    sector = mod(sector, 6.0);
    int idx = int(sector);

    float val = 0.0;
    if (idx == 0) val = u_drives[0];
    else if (idx == 1) val = u_drives[1];
    else if (idx == 2) val = u_drives[2];
    else if (idx == 3) val = u_drives[3];
    else if (idx == 4) val = u_drives[4];
    else val = u_drives[5];

    float ringR = val * 0.85 + 0.04;
    float ring = smoothstep(0.025, 0.0, abs(r - ringR));

    // sector spokes
    float half = 3.14159265 / 6.0;
    float spoke = smoothstep(0.012, 0.0,
        abs(mod(ang + half, 2.0 * half) - half)) * 0.25;

    // concentric grid
    float grid = smoothstep(0.015, 0.0, abs(r - 0.3))
               + smoothstep(0.015, 0.0, abs(r - 0.6))
               + smoothstep(0.015, 0.0, abs(r - 0.9));

    vec3 cyan   = vec3(0.0, 0.898, 1.0);
    vec3 violet = vec3(0.486, 0.302, 1.0);
    vec3 col = mix(cyan, violet, clamp(val, 0.0, 1.0));

    float energy = 0.55 + 0.45 * (u_gamma / 3.0);
    float glow = (ring + spoke + grid * 0.18) * energy;
    float vignette = smoothstep(1.06, 0.92, r);

    gl_FragColor = vec4(col * glow, clamp(glow, 0.0, 1.0) * vignette);
}
