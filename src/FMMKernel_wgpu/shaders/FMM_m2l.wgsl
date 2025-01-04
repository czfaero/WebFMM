#include contants;

#include uniforms_m2l_def;

#include CalcALP;
#include GetIndex3D;
#include cart2sph;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(2) var<storage, read> factorial: array<f32>;
@group(0) @binding(3) var<storage, read> i2nm: array<i32>;
@group(0) @binding(4) var<storage, read> interactionList: array<u32>;
@group(0) @binding(5) var<storage, read_write> Mnm: array<f32>;
@group(0) @binding(6) var<storage, read_write> Lnm: array<f32>;

var<workgroup> rho: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
var<workgroup> Pnm: array<f32, PnmSize2>;
var<workgroup> rho_n: array<f32, numExpansions>;

var<workgroup> src_box_id: u32;


fn factorialCombineM2L(tV: i32, bV1: i32, bV2: i32, bV3: i32, bV4: i32) -> f32 {
   return factorial[tV] / factorial[bV1] / factorial[bV2]
            * factorial[tV] / factorial[bV3] / factorial[bV4];

}
fn oddeven(n : i32) -> f32 {
    if ((n % 2) != 0) { return -1; } else { return 1; }
}


@compute @workgroup_size(MnmSize)
fn m2l(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id: vec3<u32>)
{
#include uniforms_m2l_expand;
    let i = local_id.x;
    let dst_box_id = group_id.x;
    let j = i2nm[i * 2];
    let k = i2nm[i * 2 + 1];
    var debug: f32; var debug2: f32;

    let dst_index = u32(boxIndexFull[dst_box_id + offset]);
    let dst_index3D = vec3<i32>(GetIndex3D(dst_index));

    let list_offset = dst_box_id * (1 + maxM2LInteraction);
    let interactionCount = interactionList[list_offset];

    var L_real = 0.0; var L_imag = 0.0;
    for(var ii: u32 = 0; ii < maxM2LInteraction; ii++){
        if(ii < interactionCount){
            if(i == 0){
                src_box_id = interactionList[list_offset + 1 + ii];
                let src_index = u32(boxIndexFull[src_box_id + offset]);
                let src_index3D = vec3<i32>(GetIndex3D(src_index));
                let dist = vec3f(src_index3D - dst_index3D) * boxSize;
                let sph = cart2sph(dist);
                rho = sph.x; alpha = sph.y; beta = sph.z;
                CalcALP(2 * numExpansions, cos(alpha));
                var v = 1.0;
                for (var ii: u32 = 0; ii < numExpansions; ii++) {
                    rho_n[ii] = v;
                    v = v * rho;
                }
            }
        }
        workgroupBarrier();
        if(ii < interactionCount){
            let O_offset = (u32(src_box_id) + offset) * MnmSize * 2;
            for (var n = 0; n < i32(numExpansions); n++) {
                for (var m = -n; m <= n; m++) {
                    let i_Pnm = (j + n) * (j + n + 1) / 2 + abs(m - k);
                    let factorialStuff =
                        factorialCombineM2L(j + n - abs(m - k), n - m, n + m, j - k, j + k);
                    let imag_i = oddeven((abs(k - m) - abs(k) - abs(m)) / 2);
                    let C = Pnm[i_Pnm] * oddeven(n) * imag_i * sqrt(factorialStuff) / rho_n[j] / rho_n[n] / rho;

                    let i_src = u32(n * n + n + m);
                    let O_real = Mnm[O_offset + 2 * i_src + 0];
                    let O_imag = Mnm[O_offset + 2 * i_src + 1];

                    let angle = f32(m - k) * beta;
                    let e_cos = cos(angle);
                    let e_sin = sin(angle);
                    
                    L_real += C * (O_real * e_cos - O_imag * e_sin);
                    L_imag += C * (O_real * e_sin + O_imag * e_cos);
                }
                if(ii==0){debug=f32(rho);debug2=f32(alpha);}
            }
            workgroupBarrier();
        }

    } // end of interation loop
    let L_offset = ((dst_box_id + offset) * MnmSize) * 2;
    Lnm[L_offset + 2 * i + 0] += L_real;
    Lnm[L_offset + 2 * i + 1] += L_imag;
    // Lnm[L_offset + 2 * i + 0] = debug;
    // Lnm[L_offset + 2 * i + 1] = debug2;
}