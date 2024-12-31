/**
 * 
 * @returns (r, theta, phi); 0 <= theta <= PI
 */
fn cart2sph(d: vec3f) -> vec3f {
    var r = sqrt(dot(d, d)) + eps;
    var theta = acos(d.z / r);
    var phi: f32;
    if (abs(d.x) + abs(d.y) < eps) {
        phi = 0;
    }
    else if (abs(d.x) < eps) {
        phi = sign(d.y) * PI * 0.5;
    }
    else if (d.x > 0) {
        phi = atan(d.y / d.x);
    }
    else {
        phi = atan(d.y / d.x) + PI;
    }
    return vec3f(r, theta, phi);
}