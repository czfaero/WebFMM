import { TreeBuilder } from "./TreeBuilder";

function debug_l2p(core, box_id, tree: TreeBuilder){

}



// // function f32(x) { return x; }
// // function i32(x) { return Math.floor(x); }

// // function vec3f(x, y, z) { return { x: x, y: y, z: z }; }
// // function vec2f(x, y) { return { x: x, y: y }; }
// // function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
// // function inverseSqrt(x) { return 1 / Math.sqrt(x); }
// // const sqrt = Math.sqrt;
// // const abs = Math.abs;
// // const pow = Math.pow;
// let acos = Math.acos;
// let atan = Math.atan;
// let cos = Math.cos;
// let sin = Math.sin;

// function vec3_add(arr) {
//     let x = 0, y = 0, z = 0;
//     for (const v of arr) {
//         x += v.x;
//         y += v.y;
//         z += v.z;
//     }
//     return { x: x, y: y, z: z }
// }
// function vec3_minus(v) {
//     return { x: -v.x, y: -v.y, z: -v.z }
// }

// // @group(0) @binding(0) var<uniform> uniforms : Uniforms;
// // @group(0) @binding(1) var<storage, read_write> particleBuffer: array<f32>;
// // @group(0) @binding(2) var<storage, read_write> resultBuffer: array<f32>;
// // @group(0) @binding(3) var<storage, read_write> command: array<i32>;
// // @group(0) @binding(4) var<storage, read_write> factorial: array<f32>;
// // @group(0) @binding(5) var<storage, read_write> Lnm: array<f32>;










function debug_l2p_shader(box, index, buffers) {
    function u32(x) { return Math.floor(x); }
    function f32(x) { return x; }
    function i32(x) { return Math.floor(x); }
    function vec3f(x, y, z) { return { x: x, y: y, z: z }; }
    function vec4f(x, y, z, w) { return { x: x, y: y, z: z, w: w }; }
    function vec2f(x, y) { return { x: x, y: y }; }
    function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    const sqrt = Math.sqrt;
    function inverseSqrt(x) { return 1 / Math.sqrt(x); }
    const abs = Math.abs;
    const pow = Math.pow;
    const acos = Math.acos;
    const atan = Math.atan;
    const cos = Math.cos;
    const sin = Math.sin;
    function vec3_add(arr) {
        let x = 0, y = 0, z = 0;
        for (const v of arr) { x += v.x; y += v.y; z += v.z; }
        return { x: x, y: y, z: z }
    }
    function vec3_minus(v) { return { x: -v.x, y: -v.y, z: -v.z } }
    const PI = Math.PI;
    const eps = 1e-6;

    function unmorton(boxIndex) {
        var mortonIndex3D = [0, 0, 0];

        var n = boxIndex;
        var k = 0;
        var i = 0;
        while (n != 0) {
            let j = 2 - k;
            mortonIndex3D[j] += (n % 2) * (1 << i);
            n >>= 1;
            k = (k + 1) % 3;
            if (k == 0) { i++; }
        }
        return {
            x: mortonIndex3D[1],
            y: mortonIndex3D[2],
            z: mortonIndex3D[0]
        };
    }
    function cart2sph(d) {
        var r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + eps;
        var theta = acos(d.z / r);
        var phi;
        if (abs(d.x) + abs(d.y) < eps) {
            phi = 0;
        }
        else if (abs(d.x) < eps) {
            phi = d.y / abs(d.y) * PI * 0.5;
        }
        else if (d.x > 0) {
            phi = atan(d.y / d.x);
        }
        else {
            phi = atan(d.y / d.x) + PI;
        }
        return vec3f(r, theta, phi);
    }


function thread_l2p(particleIndex, boxCenter, LnmOffset, buffers) {
    const Lnm = buffers.Lnm;
    const particleBuffer = buffers.particleBuffer;
    const factorial = buffers.factorial;
    function getParticle(i) {
        return vec4f(particleBuffer[i * 4], particleBuffer[i * 4 + 1], particleBuffer[i * 4 + 2], particleBuffer[i * 4 + 3]);
    }

    let particle = getParticle(particleIndex);
    let dist = vec3f(particle.x - boxCenter.x, particle.y - boxCenter.y, particle.z - boxCenter.z);
    let c = cart2sph(dist);
    let r = c.x; let theta = c.y; let phi = c.z;
    var accelR = 0; var accelTheta = 0; var accelPhi = 0;
    var xx = cos(theta);
    var yy = sin(theta);
    if (abs(yy) < eps) { yy = 1 / eps; }
    var s2 = sqrt((1 - xx) * (1 + xx));
    var fact = 1; var pn = 1; var rhom = 1;

    for (var m = 0; m < numExpansions; m++) {
        var p = pn;
        var nms = m * (m + 1) / 2 + m;
        var ere = cos(f32(m) * phi);
        if (m == 0) { ere = 0.5; }
        let eim = sin(f32(m) * phi);
        var anm = rhom * inverseSqrt(factorial[2 * m]);
        var YnmReal = anm * p;
        var p1 = p;
        p = xx * (2 * f32(m) + 1) * p;
        var YnmRealTheta = anm * (p - (f32(m) + 1) * xx * p1) / yy;
        var realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
        var imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
        accelR += 2 * f32(m) / r * YnmReal * realj;
        accelTheta += 2 * YnmRealTheta * realj;
        accelPhi -= 2 * f32(m) * YnmReal * imagj;
        rhom *= r;
        var rhon = rhom;
        for (var n = m + 1; n < numExpansions; n++) {
            nms = n * (n + 1) / 2 + m;
            anm = rhon * inverseSqrt(factorial[n + m] / factorial[n - m]);
            YnmReal = anm * p;
            var p2 = p1;
            p1 = p;
            p = (xx * f32(2 * n + 1) * p1 - f32(n + m) * p2) / f32(n - m + 1);
            YnmRealTheta = anm * (f32(n - m + 1) * p - f32(n + 1) * xx * p1) / yy;
            realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
            imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
            accelR += 2 * f32(n) / r * YnmReal * realj;
            accelTheta += 2 * YnmRealTheta * realj;
            accelPhi -= 2 * f32(m) * YnmReal * imagj;
            rhon *= r;
        }
        pn = -pn * fact * s2;
        fact = fact + 2;
    }
    let accelX = sin(theta) * cos(phi) * accelR + cos(theta) * cos(phi) / r * accelTheta - sin(phi) / r / yy * accelPhi;
    let accelY = sin(theta) * sin(phi) * accelR + cos(theta) * sin(phi) / r * accelTheta + cos(phi) / r / yy * accelPhi;
    let accelZ = cos(theta) * accelR - sin(theta) / r * accelTheta;
}



    const uniforms = buffers.uniforms;
    const numExpansions = uniforms.numExpansions;
    const numCoefficients = numExpansions * (numExpansions + 1) / 2;

    const boxSize = buffers.uniforms.boxSize;
    //let particleStart = command[group_id.x * commandLength + 1];
    //let particleCount = command[group_id.x * commandLength + 2];
    //let index = command[group_id.x * commandLength + 3];
    //let threadId = i32(local_id.x);
    let LnmOffset = box * numCoefficients * 2;
    let index3D = unmorton(u32(index));
    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let boxCenter = vec3_add([index3D, vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);
    // (vec3f(index3D) + vec3f(0.5, 0.5, 0.5)) * uniforms.boxSize + boxMin;
    const start = buffers.particleOffset[0][box];

    const end = buffers.particleOffset[1][box];
    for (let i = start; i <= end; i++) {
        thread_l2p(i, boxCenter, LnmOffset, buffers);
    }

}
