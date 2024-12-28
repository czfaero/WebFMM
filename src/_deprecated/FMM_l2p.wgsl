const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;

const numExpansions = 10;
const numCoefficients = numExpansions * (numExpansions + 1) / 2; 

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
@group(0) @binding(3) var<storage, read_write> command: array<i32>;
@group(0) @binding(4) var<storage, read_write> factorial: array<f32>;
@group(0) @binding(5) var<storage, read_write> Lnm: array<f32>;




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

fn getParticle(i:i32) -> vec4f{
   return vec4f(particleBuffer[i*4],particleBuffer[i*4+1],particleBuffer[i*4+2],particleBuffer[i*4+3]);
}

const threadsPerGroup = 256;
const commandLength = 4; 

@compute @workgroup_size(threadsPerGroup,1) 
fn l2p(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) group_id:vec3<u32>) {
  let box = command[group_id.x * commandLength + 0] ;
  let particleStart = command[group_id.x * commandLength + 1];
  let particleCount = command[group_id.x * commandLength + 2];
  let index = command[group_id.x * commandLength + 3];
  let threadId=i32(local_id.x);
  let LnmOffset = box * numCoefficients * 2;
  let index3D = unmorton(u32(index));
  let boxMin = vec3f(uniforms.boxMinX,uniforms.boxMinY,uniforms.boxMinZ) ;
  let boxCenter = (vec3f(index3D) + vec3f(0.5,0.5,0.5)) * uniforms.boxSize + boxMin;
  let particleIndex = particleStart + threadId;
  let particle = getParticle(particleIndex);
  let dist = particle.xyz - boxCenter; 
  let c = cart2sph(dist);
  let r = c.x; let theta = c.y; let phi = c.z;
  var accelR = 0f; var accelTheta = 0f; var accelPhi = 0f;
  var xx = cos(theta);
  var yy = sin(theta);
  if (abs(yy) < eps)  {yy = 1 / eps;}
  var s2 = sqrt((1 - xx) * (1 + xx));
  var fact :f32= 1; var pn:f32 = 1; var rhom:f32 = 1;
  for (var m = 0; m < numExpansions; m++)
  {
    var p = pn;
    var nms = m * (m + 1) / 2 + m;
    var ere = cos(f32(m) * phi);
    if (m == 0) {ere = 0.5;}
    let eim = sin(f32(m) * phi);
    var anm = rhom * inverseSqrt(factorial[2 * m]);
    var YnmReal = anm * p;
    var p1 = p;
    p = xx * (2 * f32(m) + 1) * p;
    var YnmRealTheta = anm * (p - (f32(m) + 1) * xx * p1) / yy;
    var realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
    var imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
    accelR += 2 * f32(m) / r * YnmReal * realj;
    accelTheta += 2 * YnmRealTheta * realj;
    accelPhi -= 2 * f32(m) * YnmReal * imagj;
    rhom *= r;
    var rhon = rhom;
    for (var n = m + 1; n < numExpansions; n++)
    {
      nms = n * (n + 1) / 2 + m;
      anm = rhon * inverseSqrt(factorial[n + m] / factorial[n - m]);
      YnmReal = anm * p;
      var p2 = p1;
      p1 = p;
      p = (xx * f32(2 * n + 1) * p1 - f32(n + m) * p2) / f32(n - m + 1);
      YnmRealTheta = anm * (f32(n - m + 1) * p - f32(n + 1) * xx * p1) / yy;
      realj = ere * Lnm[LnmOffset + 2 * nms + 0] - eim * Lnm[LnmOffset + 2 * nms + 1];
      imagj = eim * Lnm[LnmOffset + 2 * nms + 0] + ere * Lnm[LnmOffset + 2 * nms + 1];
      accelR += 2 * f32(n) / r * YnmReal * realj;
      accelTheta += 2 * YnmRealTheta * realj;
      accelPhi -= 2 * f32(m) * YnmReal * imagj;
      rhon *= r;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
  let accelX = sin(theta) * cos(phi) * accelR + cos(theta) * cos(phi) / r * accelTheta - sin(phi) / r / yy * accelPhi;
  let accelY = sin(theta) * sin(phi) * accelR + cos(theta) * sin(phi) / r * accelTheta + cos(phi) / r / yy * accelPhi;
  let accelZ = cos(theta) * accelR - sin(theta) / r * accelTheta;

  resultBuffer[particleIndex*3] = accelX;
  resultBuffer[particleIndex*3+1] = accelY;
  resultBuffer[particleIndex*3+2] = accelZ;
}




