import { FMMSolver } from "./FMMSolver";
import { TreeBuilder } from "./TreeBuilder";
const PI = 3.14159265358979323846;
const inv4PI = 0.25 / PI;
const eps = 1e-6;
export function debug_p2m(core: FMMSolver, box_id) {
    const tree = core.tree;
    let fact = 1.0;
    let factorial = new Float32Array(2 * core.numExpansions);
    for (let m = 0; m < factorial.length; m++) {
        factorial[m] = fact;
        fact = fact * (m + 1);
    }

    const boxSize = core.tree.rootBoxSize / (1 << core.tree.maxLevel);
    const particleOffset = core.tree.particleOffset;
    let maxParticlePerBox = 0;
    for (let jj = 0; jj < numBoxIndex; jj++) {
        let c = particleOffset[1][jj] - particleOffset[0][jj] + 1;
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
            particleOffset: tree.particleOffset,
            particleBuffer: tree.nodeBuffer

        }
    );

    return r;
}



const numBoxIndex = 32;//to-do
const threadsPerGroup = 64;
//var<workgroup> YnmReal: array<f32, YnmLength>; // to-do: runtime replace 
//var<workgroup> beta: f32;
//var<workgroup> particle: vec4f;
// numCoefficients= this.numExpansions * (this.numExpansions + 1) / 2;
function debug_p2m_shader(box: number, index, buffers) {
    console.log(buffers)
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

    const uniforms = buffers.uniforms;
    const numExpansions = uniforms.numExpansions;
    const numCoefficients = numExpansions * (numExpansions + 1) / 2;
    const YnmLength = numExpansions * numExpansions;
    const resultBuffer = new Float32Array(numCoefficients * 2);
    let YnmReal = new Float32Array(YnmLength);
    const boxSize = uniforms.boxSize;

    const factorial = buffers.factorial;
    let index3D = unmorton(u32(index));
    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let boxCenter = vec3_add([index3D, vec3f(0.5 * boxSize, 0.5 * boxSize, 0.5 * boxSize), boxMin]);
    const particleOffset = buffers.particleOffset;
    const particleBuffer: Float32Array = buffers.particleBuffer;
    function getParticle(i) {
        return vec4f(particleBuffer[i * 4], particleBuffer[i * 4 + 1], particleBuffer[i * 4 + 2], particleBuffer[i * 4 + 3]);
    }

    var ng = new Int32Array(threadsPerGroup);
    var mg = new Int32Array(threadsPerGroup);
    for (let n = 0; n < numExpansions; n++) {
        for (let m = 0; m <= n; m++) {
            let nms = n * (n + 1) / 2 + m;
            ng[nms] = n;
            mg[nms] = m;
        }
    }
    // let start = particleOffset[box * 2];
    // let end = particleOffset[box * 2 + 1];
    let start = particleOffset[0][box];
    let end = particleOffset[1][box];
    console.log("nodes of box " + box)
    console.log(particleBuffer.subarray(start * 4, end * 4))
    for (let particleIndex = start; particleIndex <= end; particleIndex++) {
        let particle = getParticle(particleIndex);
        let dist = vec3f(particle.x - boxCenter.x, particle.y - boxCenter.y, particle.z - boxCenter.z);
        let r = cart2sph(dist);
        let rho = r.x; let alpha = r.y; let beta = r.z;
        let xx = Math.cos(alpha);
        let s2 = Math.sqrt((1 - xx) * (1 + xx));
        for (let thread = 0; thread < numCoefficients; thread++) {
            const mm = mg[thread], nn = ng[thread];
            var fact = 1;
            var pn = 1;
            var rhom = 1;
            for (var m = 0; m <= mm; m++) {
                var p = pn;
                var nm = m * m + 2 * m;
                YnmReal[nm] = rhom * p * inverseSqrt(factorial[2 * m]);
                var p1 = p;
                p = xx * (2 * m + 1) * p;
                rhom *= rho;
                var rhon = rhom;
                for (var n = m + 1; n <= nn; n++) {
                    nm = n * n + n + m;
                    YnmReal[nm] = rhon * p * inverseSqrt(factorial[n + m] / factorial[n - m]);
                    let p2 = p1;
                    p1 = p;
                    p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
                    rhon *= rho;
                }
                pn = -pn * fact * s2;
                fact += 2;
            }
            nm = nn * nn + nn + mm;
            let ere = cos(-mm * beta);
            let eim = sin(-mm * beta);
            resultBuffer[thread * 2] += particle.w * YnmReal[nm] * ere;
            resultBuffer[thread * 2 + 1] += particle.w * YnmReal[nm] * eim;
        }


    }
    return resultBuffer;

}
// one thread for a complex number of coef
