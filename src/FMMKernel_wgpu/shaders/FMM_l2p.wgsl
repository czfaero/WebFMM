#include contants;

#include uniforms_l2p_def;
#include CalcALP_R;
#include GetIndex3D;
#include cart2sph;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> nodeBuffer: array<f32>;
@group(0) @binding(2) var<storage, read> nodeOffsetBuffer: array<u32>;
@group(0) @binding(3) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(4) var<storage, read> factorial: array<f32>;
@group(0) @binding(5) var<storage, read> Lnm: array<f32>;
@group(0) @binding(6) var<storage, read_write> accelBuffer: array<f32>;

#include getNode;

var<private> r: f32;
var<private> theta: f32;
var<private> phi: f32;
var<private> sinTheta: f32;

var<private> L_offset: u32;

var<private> accelR: f32 = 0;
var<private> accelTheta: f32 = 0;
var<private> accelPhi: f32 = 0;

// var<private> debug: f32;
// var<private> debug1: f32;

fn CalcALP_R_callback(n: i32,
                      m: i32,
                      m_abs: i32,
                      r_n: f32,
                      p: f32,
                      p_d: f32){
    let i_Lnm = u32(n * n + n + m);
    let Lnm_real = Lnm[L_offset + i_Lnm * 2];
    let Lnm_imag = Lnm[L_offset + i_Lnm * 2 + 1];
    let Ynm_fact = sqrt(factorial[n - m_abs] / factorial[n + m_abs]);
    let same_real = r_n / r * Ynm_fact;
    let same_real_Pnm = same_real * p;
    let angle = f32(m) * phi;
    let real = Lnm_real * cos(angle) - Lnm_imag * sin(angle);
    let imag = Lnm_real * sin(angle) + Lnm_imag * cos(angle);

    let d_r = f32(n) * same_real_Pnm * real;
    let d_theta = same_real * p_d * real;
    let d_phi = -f32(m) / sinTheta * same_real_Pnm * imag;

    // if(i_Lnm==1){debug=d_r;debug1=d_theta;}
    accelR += d_r;
    accelTheta += d_theta;
    accelPhi += d_phi;

}

@compute @workgroup_size(maxThreadPerGroup) 
fn l2p(@builtin(global_invocation_id) id : vec3<u32>) {
// global_invocation_id is equal to workgroup_id * workgroup_size + local_invocation_id.

#include uniforms_l2p_expand;

    let node_id = id.x;
    if(node_id >= nodeCount){return;}

    var box_id: u32 = 0;
    for(var i: u32 = 0; i < boxCount; i++){
        if(node_id >= nodeOffsetBuffer[i]){
            box_id = i;
        }
    }
    let node = getNode(node_id);

    let box_index = boxIndexFull[box_id]; //offset is 0 for max level
    let index3D = GetIndex3D(u32(box_index));
    let boxMin = vec3f(boxMinX, boxMinY, boxMinZ);
    let boxCenter = boxMin + (vec3f(index3D) + vec3f(0.5, 0.5, 0.5)) * boxSize;
    
    let dist = node.xyz - boxCenter;

    let c = cart2sph(dist);
    r = c.x; theta = c.y; phi = c.z;

    let cosTheta = cos(theta);
    let sinPhi = sin(phi); 
    let cosPhi = cos(phi);

    sinTheta = sin(theta); 
    if (abs(sinTheta) < eps) { sinTheta = eps; }
    L_offset =  ((box_id) * MnmSize) * 2;

    CalcALP_R(numExpansions, theta, r);


    let accelX = sinTheta * cosPhi * accelR
        + cosTheta * cosPhi * accelTheta
        - sinPhi * accelPhi;
    let accelY = sinTheta * sinPhi * accelR
        + cosTheta * sinPhi * accelTheta
        + cosPhi * accelPhi;
    let accelZ = cosTheta * accelR - sinTheta * accelTheta;

    accelBuffer[node_id * 3] -= accelX * node.w;
    accelBuffer[node_id * 3 + 1] -= accelY * node.w;
    accelBuffer[node_id * 3 + 2] -= accelZ * node.w;
}