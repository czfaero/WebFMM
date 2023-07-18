const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;

const numExpansions=10u;


struct Uniforms {
  boxSize:f32,
  boxMinX:f32,
  boxMinY:f32,
  boxMinZ:f32,
  numBoxIndex: u32,
  numExpansions: u32,
  maxParticlePerBox:u32
}



@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> particleBuffer: array<f32>;
@group(0) @binding(2) var<storage, read_write> resultBuffer: array<f32>;
@group(0) @binding(3) var<storage, read_write> command: array<u32>;
@group(0) @binding(4) var<storage, read_write> particleOffset: array<u32>;
@group(0) @binding(5) var<storage, read_write> factorial: array<f32>;




fn unmorton(boxIndex: u32) -> vec3<u32>{
  var mortonIndex3D :array<u32,3>;

  var n = boxIndex;
  var k:u32 = 0;
  var i:u32 = 0;
  while (n != 0) {
    let j = 2 - k;
    mortonIndex3D[j] += (n % 2) * (1u << i);
    n >>= 1;
    k = (k + 1) % 3;
    if (k == 0) {i++;}
    }
    return vec3<u32>(
            mortonIndex3D[1],
            mortonIndex3D[2],
            mortonIndex3D[0]
        );
}


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

fn getParticle(i:u32) -> vec4f{
   return vec4f(particleBuffer[i*4],particleBuffer[i*4+1],particleBuffer[i*4+2],particleBuffer[i*4+3]);
}

const numBoxIndex=32;//to-do
const YnmLength=numExpansions*numExpansions;
var<workgroup> YnmReal: array<f32, YnmLength>; // to-do: runtime replace 
var<workgroup> beta: f32;
var<workgroup> particle: vec4f;

@compute @workgroup_size(numExpansions, numExpansions) //to-do: runtime replace 
fn p2m(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) group_id:vec3<u32>) {
  let box = command[group_id.x];
  let thread_m = local_id.x;
  let thread_n = local_id.y;
  let index3D = unmorton(box);
  let boxCenter = vec3f(uniforms.boxMinX, uniforms.boxMinX, uniforms.boxMinX) 
                 + (vec3f(index3D) + vec3f(0.5,0.5,0.5)) * uniforms.boxSize;
  
  let numCoefficients = uniforms.numExpansions * (uniforms.numExpansions + 1) / 2;
  let start = particleOffset[box*2];
  let end = particleOffset[box*2+1];
  var real = 0f; var imag = 0f;
  var debug:f32;var debug1:f32;var debug_i=0u;
  for(var i = 0u; i < uniforms.maxParticlePerBox; i++){
    let particleIndex = start + i;
    if(thread_m==0 && thread_n==0){
      if(particleIndex <= end){
        particle = getParticle(particleIndex);
        let r = cart2sph(particle.xyz - boxCenter);
        let rho = r.x; let alpha = r.y; beta= r.z;
        let xx = cos(alpha);
        let s2 = sqrt((1 - xx) * (1 + xx));
        var fact = 1f;
        var pn = 1f;
        var rhom = 1f;
        for (var m = 0u; m < uniforms.numExpansions; m++) {
          var p = pn;
          var nm = m * m + 2 * m;
          YnmReal[nm] = rhom * p * inverseSqrt(factorial[2*m]);
          var p1 = p;
          p = xx * f32(2 * m + 1) * p;
          rhom *= rho;
          var rhon = rhom;
          for (var n = m + 1; n < uniforms.numExpansions; n++) {
            nm = n * n + n + m;
            YnmReal[nm] = rhon * p * inverseSqrt(factorial[n+m]/factorial[n-m]);
            let p2 = p1;
            p1 = p;
            p = (xx * f32(2 * n + 1) * p1 - f32(n + m) * p2) / f32(n - m + 1);
            rhon *= rho;
          }
          pn = -pn * fact * s2;
          fact += 2;
        }
      }// end of particle
    }// end of only thread 0 part


    workgroupBarrier();
    if(particleIndex <= end)
    {

      let m = thread_m;
      let n = thread_n;
      if(m <= n){
        let nm = n * n + n + m;
        let nms = n * (n + 1) / 2 + m;
        let write_index = group_id.x * numCoefficients + nms;
        real += particle.w * YnmReal[nm] * cos(-f32(m) * beta);
        imag += particle.w * YnmReal[nm] * sin(-f32(m) * beta);

        // // debug 
        // if(nms==1&&group_id.x==0){
        //   resultBuffer[group_id.x * numCoefficients *2 + debug_i] = real;
        //   debug_i++;
        // }
      }
    }

    //workgroupBarrier();
  }//end of particle loop

  {
    let m = thread_m;
    let n = thread_n;
    if(m <= n){
      let nms = n * (n + 1) / 2 + m;
      let write_index = group_id.x * numCoefficients + nms;
      resultBuffer[write_index * 2] = real;
      resultBuffer[write_index * 2 + 1] = imag;
    }
  }
}




