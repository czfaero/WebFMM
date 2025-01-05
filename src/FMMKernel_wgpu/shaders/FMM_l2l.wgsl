#include contants;

#include uniforms_l2l_def;

#include CalcALP;
#include GetIndex3D;
#include cart2sph;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(2) var<storage, read> boxIndexMask: array<i32>;
@group(0) @binding(3) var<storage, read> factorial: array<f32>;
@group(0) @binding(4) var<storage, read> i2nm: array<i32>;
@group(0) @binding(5) var<storage, read_write> Lnm: array<f32>;

var<workgroup> rho: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
var<workgroup> Pnm: array<f32, PnmSize2>;
var<workgroup> rho_n: array<f32, numExpansions>;


fn factorialCombineL2L(tV1: i32, tV2: i32, bV0: i32, bV1: i32, bV2: i32) -> f32 {
    return factorial[tV1] / factorial[bV0] / factorial[bV1]
        * factorial[tV2] / factorial[bV0] / factorial[bV2];
}
fn oddeven(n : i32) -> f32 {
    if ((n % 2) != 0) { return -1; } else { return 1; }
}

@compute @workgroup_size(MnmSize)
fn l2l(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id: vec3<u32>)
{
#include uniforms_l2l_expand;
    let i = local_id.x;
    let dst_box_id = group_id.x;
    let j = i2nm[i * 2];
    let k = i2nm[i * 2 + 1];

    let dst_boxSize = src_boxSize / 2;
    let boxMin = vec3f(boxMinX, boxMinY, boxMinZ);
    let dst_index = u32(boxIndexFull[dst_box_id + offset_lower]);
    let dst_index3D = GetIndex3D(dst_index);
    let dst_boxCenter = boxMin + (vec3f(dst_index3D) + vec3f(0.5, 0.5, 0.5)) * dst_boxSize;
    
    var L_real = 0.0;
    var L_imag = 0.0;

    let src_index = dst_index / 8;
    let src_box_id = boxIndexMask[src_index];

    if(i == 0){
        let src_index3D = GetIndex3D(src_index);
        let src_boxCenter = boxMin + (vec3f(src_index3D) + vec3f(0.5, 0.5, 0.5)) * src_boxSize;
        let dist = src_boxCenter - dst_boxCenter;
        let sph = cart2sph(dist);
        rho = sph.x; alpha = sph.y; beta = sph.z;
        CalcALP(2 * numExpansions, cos(alpha));
        var v = 1.0;
        for (var ii: u32 = 0; ii < numExpansions; ii++) {
            rho_n[ii] = v;
            v = v * rho;
        }
    }
    
    workgroupBarrier();

    let O_offset = ((u32(src_box_id) + offset) * MnmSize) * 2;

    for (var n = j; n < i32(numExpansions); n++) {
        for (var m = -n; m <= n; m++) {
            if (n - j < abs(m - k)) {
                continue; // Pnm not exist
            }
            let i_src = u32(n * n + n + m);
            let O_real = Lnm[O_offset + 2 * i_src + 0];
            let O_imag = Lnm[O_offset + 2 * i_src + 1];

            let i_Pnm = (n - j) * (n - j + 1) / 2 + abs(m - k);

            let factorialStuff =
                factorialCombineL2L(n - m, n + m,
                    n - j + abs(m - k), j - k, j + k);

            let imag_i = oddeven((abs(m) - abs(m - k) - abs(k)) / 2);
            let C = oddeven(n + j) * imag_i * Pnm[i_Pnm] * rho_n[n - j] * sqrt(factorialStuff);

            let angle = f32(m - k) * beta;
            let e_cos = cos(angle);
            let e_sin = sin(angle);

            L_real += C * (O_real * e_cos - O_imag * e_sin);
            L_imag += C * (O_real * e_sin + O_imag * e_cos);
        }
    }

    
    let L_offset = ((dst_box_id + offset_lower) * MnmSize) * 2;
    Lnm[L_offset + 2 * i + 0] += L_real;
    Lnm[L_offset + 2 * i + 1] += L_imag;
 
}