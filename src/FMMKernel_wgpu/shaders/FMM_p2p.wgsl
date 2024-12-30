#include contants;

#include uniforms_p2p_def;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> nodeBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> accelBuffer: array<f32>;
@group(0) @binding(3) var<storage, read> nodeOffsetBuffer: array<u32>;
@group(0) @binding(4) var<storage, read> interactionList: array<u32>;


fn getNode(i : u32) -> vec4f{
    return vec4f(
        nodeBuffer[i * 4],
        nodeBuffer[i * 4 + 1],
        nodeBuffer[i * 4 + 2],
        nodeBuffer[i * 4 + 3]
    );
}

@compute @workgroup_size(maxThreadPerGroup) 
fn p2p(@builtin(global_invocation_id) id : vec3<u32>) {
// global_invocation_id is equal to workgroup_id * workgroup_size + local_invocation_id.

#include uniforms_p2p_expand;
    var debug_temp:f32=0;
    
    let dst_node_id = id.x;
    if(dst_node_id >= nodeCount){return;}

    var dst_box_id: u32 = 0;
    for(var i: u32 = 0; i < boxCount; i++){
        if(dst_node_id >= nodeOffsetBuffer[i]){
            dst_box_id = i;
        }
    }
    let list_offset = dst_box_id * (1 + maxM2LInteraction);
    let interactionCount = interactionList[list_offset];
    
    let dst_node = getNode(dst_node_id);
    var accel: vec3f = vec3f(0, 0, 0);
    for(var i: u32 = 0; i < interactionCount; i++){
        let src_box_id = interactionList[i + list_offset + 1];
        let src_start = nodeOffsetBuffer[src_box_id];
        let src_end = nodeOffsetBuffer[src_box_id + boxCount];
        for(var src_node_id = src_start; src_node_id <= src_end; src_node_id++){
            if(src_node_id ==  dst_node_id){ continue; }
            let src_node = getNode(src_node_id);
            let dist = dst_node.xyz - src_node.xyz;
            let invDist = inverseSqrt(dot(dist, dist) + eps); 
            let invDistCube = invDist * invDist * invDist;
            let r = src_node.w * dst_node.w * invDistCube * dist;
            accel += r;
        }
        debug_temp+=f32(src_end-src_start+1);

    }

    accelBuffer[dst_node_id * 3] = accel.x;
    accelBuffer[dst_node_id * 3 + 1] = accel.y;
    accelBuffer[dst_node_id * 3 + 2] = accel.z;

    // accelBuffer[dst_node_id * 3] = debug_temp;
    // accelBuffer[dst_node_id * 3 + 1] = f32(dst_box_id);
    // accelBuffer[dst_node_id * 3 + 2] = f32(1111111);
}