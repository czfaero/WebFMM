
const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;

const eps = 1e-6;

struct Uniforms {
  commandCount: u32
}


@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> particleBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> command: array<u32>;
@group(0) @binding(4) var<storage, read_write> particleOffset: array<u32>;

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

// it says: The total number of workgroup invocations (1024) exceeds the maximum allowed (256).
// cannot find limitation '256'
@compute @workgroup_size(256, 1)
fn p2p(@builtin(global_invocation_id) id : vec3<u32>) {

  let thread = id.x;

  if(thread >= uniforms.commandCount){return;}

  resultBuffer[thread*3] = 0;
  resultBuffer[thread*3+1] = 0;
  resultBuffer[thread*3+2] = 0;

  let i = command[thread*2]; //  index of particle
  let jj = command[thread*2+1]; // index of box
  let start = particleOffset[jj*2];
  let end = particleOffset[jj*2+1];
  for(var j = start; j <= end; j++){
    let a = getParticle(i);
    let b = getParticle(j);
    let r = p2p_core(a,b);
    resultBuffer[thread*3] += r.x;
    resultBuffer[thread*3+1] += r.y;
    resultBuffer[thread*3+2] += r.z;
  }

}
