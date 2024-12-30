#include contants;

#include uniforms_p2m_def;

#include CalcALP_R;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> nodeBuffer: array<f32>;
@group(0) @binding(2) var<storage, read> nodeOffsetBuffer: array<u32>;
@group(0) @binding(3) var<storage, read> boxIndexFull: array<i32>;
@group(0) @binding(4) var<storage, read> factorial: array<f32>;
@group(0) @binding(5) var<storage, read> i2nm: array<i32>;
@group(0) @binding(5) var<storage, read_write> Mnm: array<f32>;



// one group for one box, with
// could use 共轭, but not use. keep it simple
@compute @workgroup_size(MnmSize) 
fn p2m(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id : vec3<u32>) {

#include uniforms_p2m_expand;

    let box_id = group_id.x;
    let box_index = boxIndexFull[box_id];//offset is 0 for max level
    let index3D = GetIndex3D(u32(box_index));

    let boxMin = vec3f(boxMinX, boxMinY, boxMinZ);
    let boxCenter = boxMin + (vec3f(index3D) + vec3f(0.5, 0.5, 0.5)) * boxSize 
    
    let start = nodeOffsetBuffer[box_id];
    let end = nodeOffsetBuffer[box_id + boxCount];

    let n = i2nm[local_id.x];
    let m = i2nm[local_id.x];
   
    CalcALP_R();
  
}