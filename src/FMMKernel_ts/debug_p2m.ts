import { CalcALP } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "../FMMSolver";
import { cart2sph, GetIndex3D } from "../utils";
const PI = 3.14159265358979323846;
export function debug_p2m(core: FMMSolver, box_id) {
    const tree = core.tree;
    const numExpansions = core.numExpansions
    let fact = 1.0;
    let factorial = new Float32Array(2 * numExpansions);
    for (let m = 0; m < factorial.length; m++) {
        factorial[m] = fact;
        fact = fact * (m + 1);
    }

    let numM = numExpansions * (numExpansions + 1) / 2;

    let ng = new Int32Array(numM);
    let mg = new Int32Array(numM);
    for (let n = 0; n < numExpansions; n++) {
        for (let m = 0; m <= n; m++) {
            let nms = n * (n + 1) / 2 + m;
            ng[nms] = n;
            mg[nms] = m;
        }
    }



    const boxSize = core.tree.rootBoxSize / (1 << core.tree.maxLevel);

    let maxParticlePerBox = 0;
    const numBoxIndex = Math.pow(1 << core.tree.maxLevel, 3);
    for (let jj = 0; jj < numBoxIndex; jj++) {
        let c = tree.nodeEndOffset[jj] - tree.nodeStartOffset[jj] + 1;
        if (c > maxParticlePerBox) { maxParticlePerBox = c; }
    }
    const uniforms = {
        numExpansions: core.numExpansions,
        boxSize: boxSize,
        boxMinX: core.tree.boxMinX,
        boxMinY: core.tree.boxMinY,
        boxMinZ: core.tree.boxMinZ,
    }
    const r = debug_p2m_shader(box_id, tree.boxIndexFull[box_id],
        {
            factorial: factorial,
            uniforms: uniforms,
            nodeStartOffset: tree.nodeStartOffset,
            nodeEndOffset: tree.nodeEndOffset,
            particleBuffer: tree.nodeBuffer,
            ng: ng,
            mg: mg

        }
    );

    return r;
}


function debug_p2m_shader(box_id: number, index: number, buffers: any) {
    const log = 0;
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

    if (log) console.log("-- debug p2m --");

    const uniforms = buffers.uniforms;
    const numExpansions = uniforms.numExpansions;
    const numExpansions2 = numExpansions * numExpansions;
    // const numCoefficients = numExpansions * (numExpansions + 1) / 2;
    const resultBuffer = new Float32Array(numExpansions2 * 2);
    const boxSize = uniforms.boxSize;
    const factorial = buffers.factorial;

    let start = buffers.nodeStartOffset[box_id];
    let end = buffers.nodeEndOffset[box_id];

    const particleBuffer: Float32Array = buffers.particleBuffer;
    function getParticle(i) {
        return vec4f(particleBuffer[i * 4], particleBuffer[i * 4 + 1], particleBuffer[i * 4 + 2], particleBuffer[i * 4 + 3]);
    }

    let index3D = GetIndex3D(u32(index));
    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let boxCenter = vec3_add([vec3_scale(index3D, boxSize), vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);
    if (log) console.log(`box ${index}`, index3D, " center: ", boxCenter, "\nboxSize: ", boxSize);
    if (log) console.log("node count:", end - start + 1)

    let ng = buffers.ng, mg = buffers.mg;
    let numM = numExpansions * (numExpansions + 1) / 2;
    const threadsPerGroup = numM;

    function thread(thread_id: number) {
        if (thread_id >= numM) { return }
        let M_real = 0, M_imag = 0;
        let n = ng[thread_id], m = mg[thread_id];

        // maxParticlePerBox let particleIndex = start + i;
        for (let particleIndex = start; particleIndex <= end; particleIndex++) {
            let particle = getParticle(particleIndex);
            let dist = vec3f(particle.x - boxCenter.x, particle.y - boxCenter.y, particle.z - boxCenter.z);
            let r = cart2sph(dist);
            let rho = r.x; let alpha = r.y; let beta = r.z;

            if (thread_id == 0 && particleIndex == start) {
                if (log) console.log("thread0node0:", particle, dist, { rho: rho, alpha: alpha, beta: beta });
            }

            // only thread_0_0
            const Pnm = CalcALP(numExpansions, cos(alpha));// shared

            // sync
            let i = n * (n + 1) / 2 + m;
            let C = particle.w * sqrt(factorial[n - m] / factorial[n + m]) * Pnm[i];
            let re = C * cos(-m * beta), im = C * sin(-m * beta);
            for (let i = 0; i < n; i++) {
                re *= rho;
                im *= rho;
            }

            M_real += re;
            M_imag += im;
            // sync
        }
        let i = n * n + n + m;
        resultBuffer[i * 2] = M_real;
        resultBuffer[i * 2 + 1] = M_imag;
        i = n * n + n - m;//共轭
        resultBuffer[i * 2] = M_real;
        resultBuffer[i * 2 + 1] = -M_imag;
    }

    for (let thread_id = 0; thread_id < threadsPerGroup; thread_id++) {
        thread(thread_id);
    }

    if (log) console.log("-- debug p2m end --");
    return resultBuffer;
}