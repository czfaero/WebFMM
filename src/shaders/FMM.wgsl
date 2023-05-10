
const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;

const eps = 1e-6;


@group(0) @binding(0) var<storage, read_write> particleBuffer: array<f32>;
@group(0) @binding(1) var<storage, read_write> accelBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> cmd: array<u32>;


fn cart2sph(d : vec3f) -> vec3f
{
  var r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z) + eps;
  var theta = acos(d.z / r);
  var phi:f32;
  if (abs(d.x) + abs(d.y) < eps)
  {
    phi = 0;
  }
  else if (abs(d.x) < eps)
  {
    phi = d.y / abs(d.y) * PI * 0.5;
  }
  else if (d.x > 0)
  {
    phi = atan(d.y / d.x);
  }
  else
  {
    phi = atan(d.y / d.x) + PI;
  }
  return vec3f(r,theta,phi);
}


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
fn p2p(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let thread = global_id.x;
  let total = cmd[0];
  let cmdLength = 2u;
  let cmdPerThread = cmd[0] / 256 + 1;
  let start = cmdPerThread * thread * cmdLength + 2;
  for(var c = 0u; c<cmdPerThread;c++){
    if(start+c*2>=total){ break;}
    let i = cmd[start+c*2];
    let j = cmd[start+c*2+1];
    let a = getParticle(i);
    let b = getParticle(j);
    let r = p2p_core(a,b);

    accelBuffer[i*3]+=r.x;
    accelBuffer[i*3+1]+=r.y;
    accelBuffer[i*3+2]+=r.z;

  }
  //accelBuffer[0]=f32(total);
}