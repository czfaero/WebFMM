import { FMMSolver } from "./FMMSolver";
import { TreeBuilder } from "./TreeBuilder";

export function debug_l2p(core: FMMSolver, debug_Lnm, box_id) {

    let fact = 1.0;
    let factorial = new Float32Array(2 * core.numExpansions);
    for (let m = 0; m < factorial.length; m++) {
        factorial[m] = fact;
        fact = fact * (m + 1);
    }
    const tree = core.tree;
    const boxSize = core.tree.rootBoxSize / (1 << tree.maxLevel);
    const buffers = {
        particleOffset: tree.particleOffset,
        particleBuffer: tree.nodeBuffer,
        factorial: factorial,
        uniforms: {
            boxSize: boxSize,
            boxMinX: tree.boxMinX,
            boxMinY: tree.boxMinY,
            boxMinZ: tree.boxMinZ,
        }

    };
    const index = core.tree.boxIndexFull[box_id];
    const r = debug_l2p_shader(debug_Lnm, box_id, index, buffers);
    return r;
}




function debug_l2p_shader(debug_Lnm, box, index, buffers) {
    const PI = 3.14159265358979323846;
    const inv4PI = 0.25 / PI;
    const eps = 1e-6;
    const numExpansions = 10;
    const numExpansion2 = numExpansions * numExpansions;
    const numCoefficients = numExpansions * (numExpansions + 1) / 2; //55
    const DnmSize = (4 * numExpansion2 * numExpansions - numExpansions) / 3;
    const numRelativeBox = 512;
    const maxM2LInteraction = 189;
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


    function thread_l2p(particleIndex) {
        const LnmOffset = 0;
        const Lnm = debug_Lnm;
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
        return [accelX, accelY, accelZ]
    }



    const uniforms = buffers.uniforms;
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
    const result = new Float32Array((end - start + 1) * 3);
    for (let i = start; i <= end; i++) {
        const tempAccel = thread_l2p(i);
        result.set(tempAccel, (i - start) * 3);
    }
    return result;
}
