#include contants;

#include uniforms_p2m_def;

#include CalcALP;
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
var<workgroup> Pnm: array<f32, PnmSize>;


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

    for(var ii: u32 = 0; ii < maxBoxNodeCount; ii++){
        let node_id = start + ii;
        if(node_id <= end){
            node = getNode(node_id);
            let dist = node.xyz - boxCenter;
            let t = cart2sph(dist);
            rho = t.x; alpha = t.y; beta = t.z;
            CalcALP(numExpansions, cos(alpha));
        }
        workgroupBarrier();
        if(node_id <= end){
            let m_abs = abs(m);
            let i = n * (n + 1) / 2 + m_abs ;
            let C = node.w * sqrt(factorial[n - m_abs] / factorial[n + m_abs]) * Pnm[i];
            let angle = -f32(m) * beta;
            var re = C * cos(angle); var im = C * sin(angle);
            for (var iii = 0; iii < n; iii++) {
                re *= rho;
                im *= rho;
            }
            M += vec2f(re, im);

        }
        workgroupBarrier();
    }

    let boxMnmOffset = box_id * MnmSize * 2;
    Mnm[boxMnmOffset + i * 2] = M.x;
    Mnm[boxMnmOffset + i * 2 + 1] = M.y;
}