import { CalcALP } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "./FMMSolver";
import { TreeBuilder } from "./TreeBuilder";
import { KernelWgpu } from "./kernels/kernel_wgpu";
import { cart2sph, GetIndex3D } from "./utils";

var uniforms = null;

export function debug_m2l(core: FMMSolver, numLevel, debug_Mnm, src_box_id, dst_box_id) {
    const tree = core.tree;
    let indexi = GetIndex3D(core.tree.boxIndexFull[src_box_id]);
    let ix = indexi.x,
        iy = indexi.y,
        iz = indexi.z;
    let jbd = dst_box_id + core.tree.levelOffset[numLevel - 1];
    let indexj = GetIndex3D(core.tree.boxIndexFull[jbd]);
    let jx = indexj.x, jy = indexj.y, jz = indexj.z;
    let je = core.morton1({ x: ix - jx + 3, y: iy - jy + 3, z: iz - jz + 3 }, 3) + 1;

    const boxSize = core.tree.rootBoxSize / (1 << numLevel);
    const buffers = {
        Dnm: (core.kernel as KernelWgpu).Dnm,
        particleOffset: tree.particleOffset,
        particleBuffer: tree.nodeBuffer,
        uniforms: { boxSize: boxSize }

    };
    const mnmSource = src_box_id;// core.interactionList[ii][ij] + core.tree.levelOffset[numLevel - 1]
    const r = debug_m2l_shader(debug_Mnm, mnmSource, je, buffers);

    return r;
    // todo: je
}

function debug_m2l_shader(debug_Mnm, mnmSource: number, je: number, buffers) {
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
    function oddeven(n) {
        if ((n & 1) == 1) { return -1; } else { return 1; }
    }
    function unmorton1(je) {
        let boxSize = uniforms.boxSize;
        var nb = je - 1;
        var mortonIndex3D = [0, 0, 0];
        var k = 0;
        var i = 1;
        while (nb != 0) {
            var j = 2 - k;
            mortonIndex3D[j] = mortonIndex3D[j] + nb % 2 * i;
            nb = nb / 2;
            j = k + 1;
            k = j % 3;
            if (k == 0) { i = i * 2; }
        }
        let nd = mortonIndex3D[0];
        mortonIndex3D[0] = mortonIndex3D[1];
        mortonIndex3D[1] = mortonIndex3D[2];
        mortonIndex3D[2] = nd;
        return vec3f(f32(mortonIndex3D[0] - 3) * boxSize,
            f32(mortonIndex3D[1] - 3) * boxSize,
            f32(mortonIndex3D[2] - 3) * boxSize);
    }

    // const threadsPerGroup = 64;
    const threadsPerGroup = numCoefficients;
    const Mnm = buffers.Mnm;
    const Dnm = buffers.Dnm;

    uniforms = buffers.uniforms;
    var ng = new Int32Array(threadsPerGroup);
    var mg = new Int32Array(threadsPerGroup);
    for (let n = 0; n < numExpansions; n++) {
        for (let m = 0; m <= n; m++) {
            let nms = n * (n + 1) / 2 + m;
            ng[nms] = n;
            mg[nms] = m;
        }
    }

    let sharedMnmSource = new Float32Array(2 * threadsPerGroup);
    let tempTargetBuffer = new Float32Array(2 * threadsPerGroup);// every thread need a tempTarget
    // let MnmSourceOffset = mnmSource * numCoefficients;
    // every thread copy
    // for (let i = 0; i < numCoefficients; i++) {
    //     sharedMnmSource[2 * i] = Mnm[2 * (MnmSourceOffset + i)];
    //     sharedMnmSource[2 * i + 1] = Mnm[2 * (MnmSourceOffset + i) + 1];
    // }
    for (let i = 0; i < numCoefficients * 2; i++) {
        sharedMnmSource[i] = debug_Mnm[i];
    }



    let dist = unmorton1(je);
    let rho = sqrt(dot(dist, dist)) + eps;
    let jbase = (je - 1) * DnmSize;

    // every thread
    for (let index = 0; index < numCoefficients; index++) {
        //debug03
        let tempTarget = { x: 0, y: 0 };
        let n = ng[index];
        let m = mg[index];
        let nms = n * (n + 1) / 2 + m;
        for (var k = -n; k < 0; k++) {
            let nks = n * (n + 1) / 2 - k;
            let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
            let DnmReal = Dnm[2 * nmk + 0];
            let DnmImag = Dnm[2 * nmk + 1];
            tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
            tempTarget.x += DnmImag * sharedMnmSource[2 * nks + 1];
            tempTarget.y -= DnmReal * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
        }

        for (var k = 0; k <= n; k++) {
            let nks = n * (n + 1) / 2 + k;
            let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
            let DnmReal = Dnm[2 * nmk + 0];
            let DnmImag = Dnm[2 * nmk + 1];
            tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
            tempTarget.x -= DnmImag * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmReal * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
        }
        tempTargetBuffer[2 * nms] = tempTarget.x;
        tempTargetBuffer[2 * nms + 1] = tempTarget.y;
    }
    sharedMnmSource = tempTargetBuffer;
    tempTargetBuffer = new Float32Array(2 * threadsPerGroup);
    //debug02
    // every thread
    for (let index = 0; index < numCoefficients; index++) {
        let j = ng[index];
        let k = mg[index];
        let jks = j * (j + 1) / 2 + k;
        let tempTarget = vec2f(0, 0);
        var fnmm = 1.0;
        for (var i = 0; i < j - k; i++) { fnmm = fnmm * f32(i + 1); }
        var fnpm = 1.0;
        for (var i = 0; i < j + k; i++) { fnpm = fnpm * f32(i + 1); }
        let ajk = oddeven(j) * inverseSqrt(fnmm * fnpm);
        var rhon = 1.0 / pow(rho, f32(j + k + 1));

        // debug06
        //if(ij==0){debugTemp=vec2f(rhon,ajk);}
        // debug07
        //if(ij==0){debugTemp=vec2f(fnmm,fnpm);}

        for (var n = abs(k); n < numExpansions; n++) {
            let nks = n * (n + 1) / 2 + k;
            let jnk = (j + n) * (j + n) + j + n;
            fnmm = 1.0;
            for (var i = 0; i < n - k; i++) { fnmm = fnmm * f32(i + 1); }
            fnpm = 1.0;
            for (var i = 0; i < n + k; i++) { fnpm = fnpm * f32(i + 1); }
            let ank = oddeven(n) * inverseSqrt(fnmm * fnpm);
            fnpm = 1.0;
            for (var i = 0; i < j + n; i++) { fnpm = fnpm * f32(i + 1); }
            let ajn = oddeven(j + n) / fnpm;
            let sr = oddeven(j + k) * ank * ajk / ajn;
            let CnmReal = sr * rhon;//* Ynm
            let CnmImag = 0;//sr * rhon *Ynm;

            tempTarget.x += sharedMnmSource[2 * nks + 0] * CnmReal;
            tempTarget.x -= sharedMnmSource[2 * nks + 1] * CnmImag;
            tempTarget.y += sharedMnmSource[2 * nks + 0] * CnmImag;
            tempTarget.y += sharedMnmSource[2 * nks + 1] * CnmReal;
            rhon /= rho;

        }
        tempTargetBuffer[2 * jks] = tempTarget.x;
        tempTargetBuffer[2 * jks + 1] = tempTarget.y;

    }
    sharedMnmSource = tempTargetBuffer;
    tempTargetBuffer = new Float32Array(2 * threadsPerGroup);

    //debug04
    // thread
    for (let index = 0; index < numCoefficients; index++) {

        let jbase = (je + numRelativeBox - 1) * DnmSize;
        let n = ng[index];
        let m = mg[index];
        let nms = n * (n + 1) / 2 + m;
        let tempTarget = vec2f(0, 0);
        for (var k = -n; k < 0; k++) {
            let nks = n * (n + 1) / 2 - k;
            let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
            let DnmReal = Dnm[2 * nmk + 0];
            let DnmImag = Dnm[2 * nmk + 1];
            tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
            tempTarget.x += DnmImag * sharedMnmSource[2 * nks + 1];
            tempTarget.y -= DnmReal * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
        }
        for (var k = 0; k <= n; k++) {
            let nks = n * (n + 1) / 2 + k;
            let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
            let DnmReal = Dnm[2 * nmk + 0];
            let DnmImag = Dnm[2 * nmk + 1];
            tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
            tempTarget.x -= DnmImag * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmReal * sharedMnmSource[2 * nks + 1];
            tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
        }
        tempTargetBuffer[2 * index] = tempTarget.x;
        tempTargetBuffer[2 * index + 1] = tempTarget.y;
    }
    return tempTargetBuffer;
}


export function debug_m2l_p4(core: FMMSolver, numLevel, debug_Mnm, src_box_id, dst_box_id) {
    function oddeven(n) {
        if ((n & 1) == 1) { return -1; } else { return 1; }
    }
    const sqrt = Math.sqrt;
    function inverseSqrt(x) { return 1 / Math.sqrt(x); }
    const abs = Math.abs;
    const pow = Math.pow;
    const acos = Math.acos;
    const atan = Math.atan;
    const cos = Math.cos;
    const sin = Math.sin;
    const floor = Math.floor;
    const rsqrt = inverseSqrt;
    function vec3f(x, y, z) { return { x: x, y: y, z: z }; }
    const eps = 1e-6;
    const PI = 3.14159265358979323846;
    const inv4PI = 0.25 / PI;
    const numExpansions = 10;
    const numExpansion2 = numExpansions * numExpansions;

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
    function CalcAnm(n: number, m: number, factorial: Float32Array) {
        return oddeven(n) * inverseSqrt(factorial[n - m] * factorial[n + m]);
    }
    function CalcYnmFact(n: number, m: number, factorial: Float32Array) {
        m = abs(m);
        return sqrt(factorial[n - m] / factorial[n + m]);
    }



    const tree = core.tree;
    const threadsPerGroup = numExpansion2;
    const debug_Lnm = new Float32Array(2 * threadsPerGroup);

    const MnmSource = new Float32Array(2 * threadsPerGroup);
    for (let i = 0; i < numExpansion2 * 2; i++) {
        MnmSource[i] = debug_Mnm[i];
    }
    const factorial = new Float32Array(2 * numExpansions); // accessed up to 2*(numExpansions-1)
    for (let i = 0, fact = 1; i < factorial.length; i++) {
        factorial[i] = fact;
        fact = fact * (i + 1);
    }

    const boxSize = core.tree.rootBoxSize / (1 << numLevel);

    const src_index3D = GetIndex3D(tree.boxIndexFull[src_box_id]);
    const dst_index3D = GetIndex3D(tree.boxIndexFull[dst_box_id]);
    const distX = dst_index3D.x - src_index3D.x,
        distY = dst_index3D.y - src_index3D.y,
        distZ = dst_index3D.z - src_index3D.z;

    const sph = cart2sph({ x: distX * boxSize, y: distY * boxSize, z: distZ * boxSize });
    const rho = sph.x, alpha = sph.y, beta = sph.z;
    const Pnm = CalcALP(numExpansions, cos(alpha));
    const rho_n = new Float32Array(numExpansions);
    for (let i = 0, v = 1; i < rho_n.length; i++) {
        rho_n[i] = v;
        v = v * rho;
    }

    let ng = new Int32Array(threadsPerGroup);
    let mg = new Int32Array(threadsPerGroup);
    for (let n = 0; n < numExpansions; n++) {
        for (let m = -n; m <= n; m++) {
            let i = n * n + n + m;
            ng[i] = n;
            mg[i] = m;
        }
    }
    function thread(thread_id: number) {
        let L_real = 0, L_imag = 0;
        const j = ng[thread_id];
        const k = mg[thread_id]; // -j<=k<=j
        const Ajk = CalcAnm(j, k, factorial);
        for (let n = 0; n < numExpansions; n++) {
            for (let m = -n; m <= n; m++) {
                let i_Pnm = (j + n) * (j + n + 1) / 2 + abs(m - k);
                const Anm = CalcAnm(n, m, factorial);
                const A_under = CalcAnm(j + n, m - k, factorial);
                const YnmFact = CalcYnmFact(j + n, m - k, factorial);
                const C = oddeven((abs(k - m) - abs(k) - abs(m)) / 2) * Anm * Ajk * YnmFact * Pnm[i_Pnm]
                    * oddeven(n) / A_under / rho_n[j] / rho_n[n] / rho;

                let i_src = n * n + n + m;
                const O_real = MnmSource[2 * i_src + 0];
                const O_imag = MnmSource[2 * i_src + 1];

                const e_cos = cos((k - m) * beta);
                const e_sin = sin((k - m) * beta);

                const xcos = O_real * e_cos,
                    ysin = O_imag * e_sin,
                    xsin = O_real * e_sin,
                    ycos = O_imag * e_cos;
                L_real += xcos - ysin;
                L_imag += xsin + ycos;
            }
        }
        debug_Lnm[thread_id * 2] += L_real;
        debug_Lnm[thread_id * 2 + 1] += L_imag;
        // end of thread
    }
    for (let t = 0; t < threadsPerGroup; t++) {
        thread(t);
    }
    return debug_Lnm;
}