// A.L.I.S.O.N. Ambient Overlay -- glowing arc visualizer.
// Adapted for QML ShaderEffect: uv derived from qt_TexCoord0 (0..1).
// u_audioRMS is a synthetic "activity" pulse until Module 1 (Voice Core)
// feeds real audio RMS.
varying vec2 qt_TexCoord0;

uniform float u_time;       // seconds
uniform float u_audioRMS;   // 0..1 activity pulse
uniform float u_anxiety;    // 0..1
uniform float u_curiosity;  // 0..1

void main() {
    vec2 uv = qt_TexCoord0 * 2.0 - 1.0;
    float r = length(uv);
    float ang = atan(uv.y, uv.x);       // -PI .. PI

    // --- base ring ---
    float ringR = 0.78 + 0.03 * sin(u_time * 1.5);
    float ring = smoothstep(0.03, 0.0, abs(r - ringR));

    // --- reactive deformation: anxiety roughens, curiosity ripples ---
    float wave = sin(ang * 6.0 + u_time * 4.0) * (0.02 + 0.05 * u_anxiety)
               + sin(ang * 3.0 - u_time * 2.0) * 0.03 * u_curiosity;
    float ring2 = smoothstep(0.025, 0.0, abs(r - (ringR + wave)));

    // --- audio pulse glow ---
    float pulse = u_audioRMS * (0.6 + 0.4 * sin(u_time * 12.0));
    float inner = smoothstep(ringR - 0.12 - pulse * 0.1, ringR - 0.02, r)
                * smoothstep(ringR + 0.02, ringR - 0.02, r);

    // --- faint orbiting motes ---
    float mote = 0.0;
    for (int i = 0; i < 3; i++) {
        float a = u_time * (0.6 + 0.2 * float(i)) + float(i) * 2.094;
        float mr = ringR + 0.10 + 0.05 * sin(u_time * 3.0 + float(i));
        vec2 mp = vec2(cos(a), sin(a)) * mr;
        mote += smoothstep(0.05, 0.0, length(uv - mp));
    }

    vec3 cyan   = vec3(0.0, 0.898, 1.0);
    vec3 violet = vec3(0.486, 0.302, 1.0);
    vec3 col = mix(cyan, violet, clamp(u_anxiety * 0.6 + u_curiosity * 0.4, 0.0, 1.0));

    float glow = (ring + ring2 * 1.2 + inner * pulse + mote * 0.6);
    float vignette = smoothstep(1.05, 0.9, r);

    gl_FragColor = vec4(col * glow, clamp(glow, 0.0, 1.0) * vignette);
}
