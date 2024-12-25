import { CalcALP } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "../FMMSolver";
import { cart2sph, GetIndex3D, GetIndexFrom3D } from "../utils";

/**
 * calc tV1! * tV2! / (bV0! * bV0! * bV1! * bV2!)  
 * @param readyValues factorial buffer[i]= i! 
 * @returns 
 */
export function factorialCombineL2L(tV1, tV2, bV0, bV1, bV2, readyValues: ArrayLike<number>) {
    const readyV = readyValues.length - 1;
    if (bV0 >= readyValues.length) {
        debugger;
    }
    return readyValues[tV1] / readyValues[bV0] / readyValues[bV1]
        * readyValues[tV2] / readyValues[bV0] / readyValues[bV2];

}



export function debug_l2l_p4(core: FMMSolver, numLevel, debug_src_Lnm, src_box_id, dst_box_id) {

    function oddeven(n) {
        if ((n % 2) != 0) { return -1; } else { return 1; }
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
    function vec3_add(arr) {
        let x = 0, y = 0, z = 0;
        for (const v of arr) { x += v.x; y += v.y; z += v.z; }
        return { x: x, y: y, z: z }
    }
    function vec3_minus(v) { return { x: -v.x, y: -v.y, z: -v.z } }
    function vec3_scale(v, a) { return { x: v.x * a, y: v.y * a, z: v.z * a }; }


    const eps = 1e-6;
    const PI = 3.14159265358979323846;
    const inv4PI = 0.25 / PI;
    const numExpansions = core.numExpansions;
    const numExpansion2 = numExpansions * numExpansions;


    const tree = core.tree;
    const debug_dst_Lnm = new Float32Array(2 * numExpansion2);

    const LnmSource = new Float32Array(2 * numExpansion2);
    for (let i = 0; i < numExpansion2 * 2; i++) {
        LnmSource[i] = debug_src_Lnm[i];
    }
    const factorial = new Float64Array(2 * numExpansions);
    for (let i = 0, fact = 1; i < factorial.length; i++) {
        factorial[i] = fact;
        fact = fact * (i + 1);
    }

    console.log("-- debug l2l --");
    const src_boxSize = core.tree.rootBoxSize / (2 << numLevel);//large one
    const uniforms = {
        boxSize: src_boxSize,
        boxMinX: tree.boxMinX,
        boxMinY: tree.boxMinY,
        boxMinZ: tree.boxMinZ,
    }

    const dst_boxSize = src_boxSize / 2;

    const src_index3D = GetIndex3D(tree.boxIndexFull[src_box_id]);

    let boxMin = vec3f(uniforms.boxMinX, uniforms.boxMinY, uniforms.boxMinZ);
    let src_boxCenter = vec3_add(
        [vec3_scale(src_index3D, src_boxSize),
        vec3f(0.5 * src_boxSize, 0.5 * src_boxSize, 0.5 * src_boxSize),
            boxMin]);

    const dst_index3D = GetIndex3D(tree.boxIndexFull[dst_box_id]);
    let dst_boxCenter = vec3_add(
        [vec3_scale(dst_index3D, dst_boxSize),
        vec3f(0.5 * dst_boxSize, 0.5 * dst_boxSize, 0.5 * dst_boxSize),
            boxMin]);

    const dist = vec3_add([src_boxCenter, vec3_minus(dst_boxCenter)]);

    const sph = cart2sph(dist);
    const rho = sph.x, alpha = sph.y, beta = sph.z;

    console.log("boxSize:", src_boxSize)
    console.log(`src box ${src_box_id}@lv${numLevel}`, src_index3D);
    console.log(`dst box ${dst_box_id}@lv${numLevel + 1}`, dst_index3D);
    console.log("dist", dist)
    console.log("sph", { rho: rho, alpha: alpha, beta: beta })

    const Pnm = CalcALP(2 * numExpansions, cos(alpha));
    const rho_n = new Float32Array(numExpansions);
    for (let i = 0, v = 1; i < rho_n.length; i++) {
        rho_n[i] = v;
        v = v * rho;
    }

    let ng = new Int32Array(numExpansion2);
    let mg = new Int32Array(numExpansion2);
    for (let n = 0; n < numExpansions; n++) {
        for (let m = -n; m <= n; m++) {
            let i = n * n + n + m;
            ng[i] = n;
            mg[i] = m;
        }
    }

    /** thread for one dst Mnm */
    function thread(thread_id: number) {
        const j = ng[thread_id];
        const k = mg[thread_id]; // -j<=k<=j

        /** loop for one intercation*/
        function loop() {
            let L_real = 0, L_imag = 0;

            for (let n = j; n < numExpansions; n++) {
                for (let m = -n; m <= n; m++) {
                    if (n - j < abs(m - k)) {
                        continue; // Pnm not exist
                    }
                    let i_src = n * n + n + m;
                    const O_real = LnmSource[2 * i_src + 0];
                    const O_imag = LnmSource[2 * i_src + 1];

                    let i_Pnm = (n - j) * (n - j + 1) / 2 + abs(m - k);

                    const factorialStuff =
                        factorialCombineL2L(n - m, n + m,
                            n - j + abs(m - k), j - k, j + k, factorial);

                    const imag_i = oddeven((abs(m) - abs(m - k) - abs(k)) / 2);
                    const C = oddeven(n + j) * imag_i * Pnm[i_Pnm] * rho_n[n - j] * sqrt(factorialStuff);

                    const angle = (m - k) * beta;
                    const e_cos = cos(angle);
                    const e_sin = sin(angle);

                    L_real += C * (O_real * e_cos - O_imag * e_sin);
                    L_imag += C * (O_real * e_sin + O_imag * e_cos);
                    if (isNaN(C)) { debugger; }
                }
            }
            debug_dst_Lnm[thread_id * 2] += L_real;
            debug_dst_Lnm[thread_id * 2 + 1] += L_imag;
        }
        let interaction_count = 1; // max 8 for 3d l2l
        for (let i = 0; i < interaction_count; i++) {
            loop();
        }

    }

    for (let t = 0; t < core.MnmSize; t++) {
        thread(t);
    }
    console.log("-- debug l2l end--");
    return debug_dst_Lnm;
}