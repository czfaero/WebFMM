const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;


struct Uniforms {
  unused:f32,
  particleCount:u32
}




@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> particleBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<f32>;


fn p2p_core(a : vec4<f32>, b : vec4<f32>) -> vec3f
{
    let dist = a.xyz - b.xyz;
    let invDist = inverseSqrt(dot(dist, dist) + eps); 
    let invDistCube = invDist * invDist * invDist;
    let s = b.w * invDistCube;
    return -s * inv4PI * dist;
}

fn getParticle(i:u32) -> vec4f{
    return vec4f(particleBuffer[i*4],particleBuffer[i*4+1],particleBuffer[i*4+2],particleBuffer[i*4+3]);
}

const group_size:u32 = 128;
@compute @workgroup_size(group_size) 
fn direct(@builtin(global_invocation_id) id : vec3<u32>) {
    // global_invocation_id is equal to workgroup_id * workgroup_size + local_invocation_id.
    let i = id.x;
    if(i >= uniforms.particleCount){return;}
    
    let p1 = getParticle(i);
    var accel : vec3f = vec3f(0,0,0);
    for(var j:u32 = 0; j < uniforms.particleCount; j++){
        let p2 = getParticle(j);
        let r = p2p_core(p1,p2);
        accel+=r;
    }
    resultBuffer[i*3] = accel.x;
    resultBuffer[i*3+1] = accel.y;
    resultBuffer[i*3+2] = accel.z;

    //resultBuffer[i*3] = f32(i);
}