import { CalcALP_R } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "./FMMSolver";
import { TreeBuilder } from "./TreeBuilder";
import { cart2sph, GetIndex3D } from "./utils";

export function debug_l2p(core: FMMSolver, debug_Lnm, box_id) {

    let fact = 1.0;
    let factorial = new Float64Array(2 * core.numExpansions);
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
    const r = debug_l2p_shader2(debug_Lnm, box_id, index, buffers);
    return r;
}




// function debug_l2p_shader(debug_Lnm, box, index, buffers) {
//     const PI = 3.14159265358979323846;
//     const inv4PI = 0.25 / PI;
//     const eps = 1e-6;
//     const numExpansions = 10;
//     const numExpansion2 = numExpansions * numExpansions;
//     const numCoefficients = numExpansions * (numExpansions + 1) / 2; //55
//     const DnmSize = (4 * numExpansion2 * numExpansions - numExpansions) / 3;
//     const numRelativeBox = 512;
//     const maxM2LInteraction = 189;
//     function u32(x) { return Math.floor(x); }
//     function f32(x) { return x; }
//     function i32(x) { return Math.floor(x); }
//     function vec3f(x, y, z) { return { x: x, y: y, z: z }; }
//     function vec4f(x, y, z, w) { return { x: x, y: y, z: z, w: w }; }
//     function vec2f(x, y) { return { x: x, y: y }; }
//     function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
//     const sqrt = Math.sqrt;
//     function inverseSqrt(x) { return 1 / Math.sqrt(x); }
//     const abs = Math.abs;
//     const pow = Math.pow;
//     const acos = Math.acos;
//     const atan = Math.atan;
//     const cos = Math.cos;
//     const sin = Math.sin;
//     function vec3_add(arr) {
//         let x = 0, y = 0, z = 0;
//         for (const v of arr) { x += v.x; y += v.y; z += v.z; }
//         return { x: x, y: y, z: z }
//     }
//     function vec3_minus(v) { return { x: -v.x, y: -v.y, z: -v.z } }

//     function thread_l2p(particleIndex) {
//         const LnmOffset = 0;
//         const Lnm = debug_Lnm;
//         const particleBuffer = buffers.particleBuffer;
//         const factorial = buffers.factorial;
//         function getParticle(i) {
//             return vec4f(particleBuffer[i * 4], particleBuffer[i * 4 + 1], particleBuffer[i * 4 + 2], particleBuffer[i * 4 + 3]);
//         }

//         let particle = getParticle(particleIndex);
//         let dist = vec3f(particle.x - boxCenter.x, particle.y - boxCenter.y, particle.z - boxCenter.z);
//         let c = cart2sph(dist);
//         let r = c.x; let theta = c.y; let phi = c.z;
//         var accelR = 0; var accelTheta = 0; var accelPhi = 0;
//         var xx = cos(theta);
//         var yy = sin(theta);
//         if (abs(yy) < eps) { yy = 1 / eps; }
//         var s2 = sqrt((1 - xx) * (1 + xx));
//         var fact = 1; var pn = 1; var rhom = 1;

//         for (var m = 0; m < numExpansions; m++) {
//             var p = pn;
//             var nms = m * (m + 1) / 2 + m;
//             var ere = cos(f32(m) * phi);
//             if (m == 0) { ere = 0.5; }
//             let eim = sin(f32(m) * phi);
//             var anm = rhom * inverseSqrt(factorial[2 * m]);
//             var YnmReal = anm * p;
//             var p1 = p;
//             p = xx * (2 * f32(m) + 1) * p;
//             var YnmRealTheta = anm * (p - (f32(m) + 1) * xx * p1) / yy;
//             var realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
//             var imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
//             accelR += 2 * f32(m) / r * YnmReal * realj;
//             accelTheta += 2 * YnmRealTheta * realj;
//             accelPhi -= 2 * f32(m) * YnmReal * imagj;
//             rhom *= r;
//             var rhon = rhom;
//             for (var n = m + 1; n < numExpansions; n++) {
//                 nms = n * (n + 1) / 2 + m;
//                 anm = rhon * inverseSqrt(factorial[n + m] / factorial[n - m]);
//                 YnmReal = anm * p;
//                 var p2 = p1;
//                 p1 = p;
//                 p = (xx * f32(2 * n + 1) * p1 - f32(n + m) * p2) / f32(n - m + 1);
//                 YnmRealTheta = anm * (f32(n - m + 1) * p - f32(n + 1) * xx * p1) / yy;
//                 realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
//                 imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
//                 accelR += 2 * f32(n) / r * YnmReal * realj;
//                 accelTheta += 2 * YnmRealTheta * realj;
//                 accelPhi -= 2 * f32(m) * YnmReal * imagj;
//                 rhon *= r;
//             }
//             pn = -pn * fact * s2;
//             fact = fact + 2;
//         }
//         let accelX = sin(theta) * cos(phi) * accelR + cos(theta) * cos(phi) / r * accelTheta - sin(phi) / r / yy * accelPhi;
//         let accelY = sin(theta) * sin(phi) * accelR + cos(theta) * sin(phi) / r * accelTheta + cos(phi) / r / yy * accelPhi;
//         let accelZ = cos(theta) * accelR - sin(theta) / r * accelTheta;
//         return [accelX, accelY, accelZ]
//     }
//     const uniforms = buffers.uniforms;
//     const boxSize = buffers.uniforms.boxSize;
//     //let particleStart = command[group_id.x * commandLength + 1];
//     //let particleCount = command[group_id.x * commandLength + 2];
//     //let index = command[group_id.x * commandLength + 3];
//     //let threadId = i32(local_id.x);
//     let LnmOffset = box * numCoefficients * 2;
//     let index3D = GetIndex3D(u32(index));
//     let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
//     let boxCenter = vec3_add([index3D, vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);
//     // (vec3f(index3D) + vec3f(0.5, 0.5, 0.5)) * uniforms.boxSize + boxMin;
//     const start = buffers.particleOffset[0][box];

//     const end = buffers.particleOffset[1][box];
//     const result = new Float32Array((end - start + 1) * 3);
//     for (let i = start; i <= end; i++) {
//         const tempAccel = thread_l2p(i);
//         debugger;
//         result.set(tempAccel, (i - start) * 3);
//     }
//     return result;
// }


function debug_l2p_shader2(debug_Lnm, box_id, index, buffers) {
    console.log("-- debug l2p --")

    const PI = Math.PI;
    const inv4PI = 0.25 / PI;
    const eps = 1e-6;
    const numExpansions = 10;
    const numExpansion2 = numExpansions * numExpansions;
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
    function vec3_scale(v, a) { return { x: v.x * a, y: v.y * a, z: v.z * a }; }

    const uniforms = buffers.uniforms;

    let boxSize = uniforms.boxSize;
    let index3D = GetIndex3D(u32(index));
    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let boxCenter = vec3_add([vec3_scale(index3D, boxSize), vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);


    console.log(`box ${index}`, index3D, " center: ", boxCenter, "\nboxSize: ", boxSize);
    const start = buffers.particleOffset[0][box_id];
    const end = buffers.particleOffset[1][box_id];

    const count = end - start + 1;
    const result = new Float32Array(3 * count);
    const nodeBuffer = buffers.particleBuffer;
    const factorial = buffers.factorial;


    const Lnm = debug_Lnm;
    function getParticle(i) {
        return vec4f(nodeBuffer[i * 4], nodeBuffer[i * 4 + 1], nodeBuffer[i * 4 + 2], nodeBuffer[i * 4 + 3]);
    }



    function thread(thread_id) {
        const node_id = start + thread_id;
        const node = getParticle(node_id);
        let dist = vec3f(node.x - boxCenter.x, node.y - boxCenter.y, node.z - boxCenter.z);

        if (thread_id == 0) {
            console.log("l2p node0", node);
            console.log("l2p dist0", dist);
        }
        let c = cart2sph(dist);
        let r = c.x; let theta = c.y; let phi = c.z;
        let accelR = 0, accelTheta = 0, accelPhi = 0;
        let sinTheta = sin(theta), cosTheta = cos(theta), sinPhi = sin(phi), cosPhi = cos(phi);
        if (abs(sinTheta) < eps) { sinTheta = eps; }
        let debug_count = 0;
        function proc(n, m, abs_m, r_n, Pnm, Pnm_d) {

            let i_Lnm = n * n + n + m;
            let Lnm_real = Lnm[i_Lnm * 2], Lnm_imag = Lnm[i_Lnm * 2 + 1];
            let Ynm_fact = sqrt(factorial[n - abs_m] / factorial[n + abs_m]);
            let same_real = r_n / r * Ynm_fact;
            let same_real_Pnm = same_real * Pnm;
            let angle = m * phi;
            let real = Lnm_real * cos(angle) - Lnm_imag * sin(angle);
            let imag = Lnm_real * sin(angle) + Lnm_imag * cos(angle);

            let d_r = n * same_real_Pnm * real;
            let d_theta = same_real * Pnm_d * real;
            let d_phi = -m / sinTheta * same_real_Pnm * imag;
            accelR += d_r;
            accelTheta += d_theta;
            accelPhi += d_phi;
            //debugger;
        }
        CalcALP_R(numExpansions, theta, r, proc);

        let accelX = sinTheta * cosPhi * accelR
            + cosTheta * cosPhi * accelTheta
            - sinPhi * accelPhi;
        let accelY = sinTheta * sinPhi * accelR
            + cosTheta * sinPhi * accelTheta
            + cosPhi * accelPhi;
        let accelZ = cosTheta * accelR - sinTheta * accelTheta;
        result[thread_id * 3] = accelX;
        result[thread_id * 3 + 1] = accelY;
        result[thread_id * 3 + 2] = accelZ;
        //debugger;
    }
    const maxNodeCount = count;// to-do
    for (let i = 0; i <= maxNodeCount; i++) {
        thread(i);
    }
    console.log("-- debug l2p end--")
    return result;
}
