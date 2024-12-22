import { CalcALP } from "./AssociatedLegendrePolyn";
import { FMMSolver } from "../FMMSolver";
import { cart2sph, GetIndex3D, GetIndexFrom3D } from "../utils";

/**
 * tV! * tV! / (bV1! * ... * bV4!)  
 * bV1 > bV2, bV3 > bV4, all < readyValues.length
 * @param readyValues factorial buffer[i]= i! 
 * @returns 
 */
export function factorialCombineM2L(tV: number, bV1, bV2, bV3, bV4, readyValues: ArrayLike<number>) {
    const readyV = readyValues.length - 1;
    if (tV <= readyV) {
        return readyValues[tV] / readyValues[bV1] / readyValues[bV2]
            * readyValues[tV] / readyValues[bV3] / readyValues[bV4];
    }
    let part1 = readyValues[tV] / readyValues[bV1] / readyValues[bV2];
    let part2 = readyValues[tV] / readyValues[bV3] / readyValues[bV4]
    for (let v = readyV + 1; v <= tV; v++) {
        part1 *= v;
        part2 *= v;
    }
    return part1 * part2;


}



export function debug_m2l_p4(core: FMMSolver, numLevel, debug_Mnm, src_box_id, dst_box_id) {
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
    const debug_Lnm = new Float32Array(2 * numExpansion2);

    const MnmSource = new Float32Array(2 * numExpansion2);
    for (let i = 0; i < numExpansion2 * 2; i++) {
        MnmSource[i] = debug_Mnm[i];
    }
    const factorial = new Float64Array(2 * numExpansions);
    for (let i = 0, fact = 1; i < factorial.length; i++) {
        factorial[i] = fact;
        fact = fact * (i + 1);
    }

    console.log("-- debug m2l --");
    const boxSize = core.tree.rootBoxSize / (1 << numLevel);

    const src_index3D = GetIndex3D(tree.boxIndexFull[src_box_id]);
    const dst_index3D = GetIndex3D(tree.boxIndexFull[dst_box_id]);
    const dist_index3D = vec3_add([src_index3D, vec3_minus(dst_index3D)]);
    const dist = vec3_scale(dist_index3D, boxSize)
    const sph = cart2sph(dist);
    const rho = sph.x, alpha = sph.y, beta = sph.z;

    console.log("boxSize:", boxSize)
    console.log(`src box ${src_box_id}`, src_index3D);
    console.log(`dst box ${dst_box_id}`, dst_index3D);
    console.log("dist", dist, dist_index3D)
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

    /** thread for one Lnm */
    function thread(thread_id: number) {

        /** loop for one intercation*/
        function loop() {
            let L_real = 0, L_imag = 0;
            const j = ng[thread_id];
            const k = mg[thread_id]; // -j<=k<=j

            for (let n = 0; n < numExpansions; n++) {
                for (let m = -n; m <= n; m++) {
                    let i_Pnm = (j + n) * (j + n + 1) / 2 + abs(m - k);
                    const factorialStuff =
                        factorialCombineM2L(j + n - abs(m - k), n - m, n + m, j - k, j + k, factorial);
                    const C = Pnm[i_Pnm] * oddeven(n) * oddeven((abs(k - m) - abs(k) - abs(m)) / 2) * sqrt(factorialStuff) / rho_n[j] / rho_n[n] / rho;

                    let i_src = n * n + n + m;
                    const O_real = MnmSource[2 * i_src + 0];
                    const O_imag = MnmSource[2 * i_src + 1];

                    const angle = (m - k) * beta;
                    const e_cos = cos(angle);
                    const e_sin = sin(angle);

                    L_real += C * (O_real * e_cos - O_imag * e_sin);
                    L_imag += C * (O_real * e_sin + O_imag * e_cos);
                    if (isNaN(C)) { debugger; }
                }
            }
            debug_Lnm[thread_id * 2] += L_real;
            debug_Lnm[thread_id * 2 + 1] += L_imag;
        }
        let interaction_count = 1;
        for (let i = 0; i < interaction_count; i++) {
            loop();
        }

    }

    for (let t = 0; t < core.MnmSize; t++) {
        thread(t);
    }
    console.log("-- debug m2l end--");
    return debug_Lnm;
}