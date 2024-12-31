// v2误差似乎更大

#include contants;

#include uniforms_p2m_def;

#include CalcALP_R;
#include GetIndex3D;
#include cart2sph;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> nodeBuffer: array<f32>;
@group(0) @binding(2) var<storage, read> nodeOffsetBuffer: array<u32>;
@group(0) @binding(3) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(4) var<storage, read> factorial: array<f32>;
@group(0) @binding(5) var<storage, read> i2nm: array<i32>;
@group(0) @binding(6) var<storage, read_write> Mnm: array<f32>;

#include getNode;

var<private> n: i32;
var<private> m: i32;
var<private> M: vec2f = vec2f(0, 0);
var<workgroup> rho: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
var<workgroup> node: vec4f;

fn CalcALP_R_callback(n_: i32,
                      m_: i32,
                      m_abs: i32,
                      r_n: f32,
                      p: f32,
                      p_d: f32){
    if(n_ != n || m_ != m) { return; }
    let C = node.w * sqrt(factorial[n - m_abs] / factorial[n + m_abs]) * p * r_n;
    let angle = -f32(m) * beta;
    let re = C * cos(angle); 
    let im = C * sin(angle);
    M += vec2f(re, im);
}

// one group for one box, with
// could use 共轭, but not use. keep it same with others.
@compute @workgroup_size(MnmSize) 
fn p2m(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id : vec3<u32>) {

#include uniforms_p2m_expand;

    let i = local_id.x;
    let box_id = group_id.x;
    let box_index = boxIndexFull[box_id]; //offset is 0 for max level
    let index3D = GetIndex3D(u32(box_index));

    let boxMin = vec3f(boxMinX, boxMinY, boxMinZ);
    let boxCenter = boxMin + (vec3f(index3D) + vec3f(0.5, 0.5, 0.5)) * boxSize;
    
    let start = nodeOffsetBuffer[box_id];
    let end = nodeOffsetBuffer[box_id + boxCount];

    n = i2nm[i * 2];
    m = i2nm[i * 2 + 1];

    for(var node_id: u32 = start; node_id <= end; node_id++){
        node = getNode(node_id);
        let dist = node.xyz - boxCenter;
        let t = cart2sph(dist);
        rho = t.x; alpha = t.y; beta = t.z;
        CalcALP_R(numExpansions, alpha, rho);
    }

    let boxMnmOffset = box_id * MnmSize * 2;
    Mnm[boxMnmOffset + i * 2] = M.x;
    Mnm[boxMnmOffset + i * 2 + 1] = M.y;
}