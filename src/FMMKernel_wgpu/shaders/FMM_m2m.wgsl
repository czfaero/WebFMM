#include contants;

#include uniforms_m2m_def;

#include CalcALP;
#include GetIndex3D;
#include cart2sph;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(2) var<storage, read> boxIndexMask: array<i32>;
@group(0) @binding(3) var<storage, read> factorial: array<f32>;
@group(0) @binding(4) var<storage, read> i2nm: array<i32>;
@group(0) @binding(5) var<storage, read_write> Mnm: array<f32>;

var<workgroup> rho: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
var<workgroup> Pnm: array<f32, PnmSize2>;
var<workgroup> rho_n: array<f32, numExpansions>;


fn factorialCombineM2M(tV1: i32, tV2: i32, bV0: i32, bV1: i32, bV2: i32) -> f32 {
    return factorial[tV1] / factorial[bV0] / factorial[bV1]
        * factorial[tV2] / factorial[bV0] / factorial[bV2];
}
fn oddeven(n : i32) -> f32 {
    if ((n % 2) != 0) { return -1; } else { return 1; }
}


@compute @workgroup_size(MnmSize)
fn m2m(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id: vec3<u32>)
{
#include uniforms_m2m_expand;
    let i = local_id.x;
    let dst_box_id = group_id.x;
    let j = i2nm[i * 2];
    let k = i2nm[i * 2 + 1];

    let boxMin = vec3f(boxMinX, boxMinY, boxMinZ);
    let dst_index = u32(boxIndexFull[dst_box_id + offset]);
    let dst_index3D = GetIndex3D(dst_index);
    let dst_boxCenter = boxMin + (vec3f(dst_index3D) + vec3f(0.5, 0.5, 0.5)) * dst_boxSize;
    
    let src_boxSize = dst_boxSize / 2;
    var M_real = 0.0;
    var M_imag = 0.0;
    for (var b: u32 = 0; b < 8; b++) {
        let src_index = dst_index * 8 + b;
        let src_box_id = boxIndexMask[src_index];
        if (src_box_id != -1) {
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
        if (src_box_id != -1) {
            let O_offset = ((u32(src_box_id) + offset_lower) * MnmSize) * 2;
 
            for (var n = 0; n <= j; n++) {
                for (var m = -n; m <= n; m++) {
                    let n_ = j - n; let m_ = k - m;
                    let i_src = u32(n_ * n_ + n_ + m_);

                    let O_real = Mnm[O_offset + 2 * i_src + 0];
                    let O_imag = Mnm[O_offset + 2 * i_src + 1];

                    let i_Pnm = (n) * (n + 1) / 2 + abs(m);
                    let factorialStuff =
                        factorialCombineM2M(j - k, j + k,
                            n + abs(m), j - n - abs(k - m), j - n + abs(k - m));

                    let imag_i = oddeven((abs(k) - abs(m) - abs(k - m)) / 2);
                    let C = imag_i * Pnm[i_Pnm] * rho_n[n] * sqrt(factorialStuff);

                    let angle = f32(-m) * beta;
                    let e_cos = cos(angle);
                    let e_sin = sin(angle);

                    M_real += C * (O_real * e_cos - O_imag * e_sin);
                    M_imag += C * (O_real * e_sin + O_imag * e_cos);

                }
            }
        
        }
        workgroupBarrier();
    }

    let M_offset = ((dst_box_id + offset) * MnmSize) * 2;
    Mnm[M_offset + 2 * i + 0] = M_real;
    Mnm[M_offset + 2 * i + 1] = M_imag;
 
}