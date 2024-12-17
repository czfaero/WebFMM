import { CalcALP_R } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "./FMMSolver";
import { TreeBuilder } from "./TreeBuilder";
import { cart2sph, GetIndex3D } from "./utils";

export function debug_m2p(core: FMMSolver, debug_Mnm, src_box_id, dst_box_id) {

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
            numExpansions: core.numExpansions
        }

    };
    const src_index = core.tree.boxIndexFull[src_box_id];
    const r = debug_m2p_shader(debug_Mnm, dst_box_id, src_index, buffers);
    return r;
}


function debug_m2p_shader(debug_Mnm, dst_box_id, src_index, buffers) {
    const PI = Math.PI;
    const inv4PI = 0.25 / PI;
    const eps = 1e-6;
    const numExpansions = buffers.uniforms.numExpansions;
    const numExpansion2 = numExpansions * numExpansions;
    const DnmSize = (4 * numExpansion2 * numExpansions - numExpansions) / 3;
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
    let index3D = GetIndex3D(u32(src_index));
    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let boxCenter = vec3_add([vec3_scale(index3D, boxSize), vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);

    console.log("-- debug m2p --");
    console.log(`src`)
    console.log(`box ${src_index}`, index3D, " center: ", boxCenter, "\nboxSize: ", boxSize);

    const start = buffers.particleOffset[0][dst_box_id];
    const end = buffers.particleOffset[1][dst_box_id];

    const count = end - start + 1;
    const result = new Float32Array(3 * count);
    const nodeBuffer = buffers.particleBuffer;
    const factorial = buffers.factorial;


    const Mnm = debug_Mnm;
    function getParticle(i) {
        return vec4f(nodeBuffer[i * 4], nodeBuffer[i * 4 + 1], nodeBuffer[i * 4 + 2], nodeBuffer[i * 4 + 3]);
    }



    function thread(thread_id) {
        const node_id = start + thread_id;
        const node = getParticle(node_id);
        let dist = vec3f(node.x - boxCenter.x, node.y - boxCenter.y, node.z - boxCenter.z);
        let c = cart2sph(dist);
        let r = c.x; let theta = c.y; let phi = c.z;

        if (thread_id == 0) {
            console.log("node0", node, dist, { r: r, theta: theta, phi: phi });
        }
        let accelR = 0, accelTheta = 0, accelPhi = 0;
        let sinTheta = sin(theta), cosTheta = cos(theta), sinPhi = sin(phi), cosPhi = cos(phi);
        if (abs(sinTheta) < eps) { sinTheta = eps; }

        function proc(n, m, abs_m, r_n, Pnm, Pnm_d) {

            let i_Mnm = n * n + n + m;
            let Mnm_real = Mnm[i_Mnm * 2], Mnm_imag = Mnm[i_Mnm * 2 + 1];
            let Ynm_fact = sqrt(factorial[n - abs_m] / factorial[n + abs_m]);
            let angle = m * phi;
            let same_real = Ynm_fact / r_n / r / r;
            let same_real_Pnm = same_real * Pnm;
            let real = Mnm_real * cos(angle) - Mnm_imag * sin(angle);
            let imag = Mnm_real * sin(angle) + Mnm_imag * cos(angle);
            let d_r = (-n - 1) * real * same_real_Pnm;
            let d_theta = real * same_real * Pnm_d;
            let d_phi = -m / sinTheta * imag * same_real_Pnm;
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
    }
    const maxNodeCount = count;// to-do
    for (let i = 0; i <= maxNodeCount; i++) {
        thread(i);
    }
    console.log("-- debug m2p end --");
    return result;
}
