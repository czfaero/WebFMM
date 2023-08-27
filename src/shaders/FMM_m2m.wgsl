const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;

const numExpansions = 10u;
const numExpansion2 = numExpansions * numExpansions;

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

fn oddeven(n:i32){
    return (n&1==1) ? -1:1;
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

var<workgroup> sharedYnm: array<f32, YnmLength>; 

fn calculate_ynm(rho:f32, alpha:f32)
{
  var xx = cos(alpha);
  var s2 = sqrt((1 - xx) * (1 + xx));
  var fact : f32 = 1;
  var pn : f32 = 1.0;
  var rhom : f32 = 1;
  for(var m : i32 = 0; m < numExpansions; m++){
    var p = pn;
    var npn = m * m + 2 * m;
    var nmn = m * m;
    sharedYnm[npn] = rhom * p / factorial[2 * m];
    sharedYnm[nmn] = sharedYnm[npn];
    var p1 = p;
    p = xx * (2 * m + 1) * p;
    rhom *= -rho;
    var rhon = rhom;
    for(var n = m + 1; n < numExpansions; n++){
      npm = n * n + n + m;
      nmm = n * n + n - m;
      sharedYnm[npm] = rhon * p / factorial[n+m];
      sharedYnm[nmm] = sharedYnm[npm];
      var p2 = p1;
      p1 = p;
      p = (xx * (2 * n + 1) * p1 - (n + m) * p2) / (n - m + 1);
      rhon *= -rho;
    }
    pn = -pn * fact * s2;
    fact = fact + 2;
  }
}

fn core(j:i32,  beta:f32, )
{
  int   nm, jnkms;
  float ere, eim, ajk, ajnkm, cnm, CnmReal, CnmImag;
  var k = 0;
  for(var i = 0; i <= j; i++ ){k += i;}
  k = threadIdx.x - k; 
  ajk = oddeven(j) * rsqrtf(factorial[j - k] * factorial[j + k]);
  var MnmTarget : vec2f;
  for(var n = 0; n <= j; n++){
    for(var m = -n; m <= min(k-1,n); m++){
      if(j - n >= k - m){
        let nm = n * n + n + m;
        let jnkms = (j - n) * (j - n + 1) / 2 + k - m;
        let ere = cosf(-m * beta);
        let eim = sinf(-m * beta);
        let ajnkm = rsqrtf(factorial[j - n - k + m] * factorial[j - n + k - m]);
        var cnm = ODDEVEN(m + j);
        cnm *= ajnkm / ajk * sharedYnm[nm];
        let CnmReal = cnm * ere;
        let CnmImag = cnm * eim;
        MnmTarget[0] += MnmSource[2 * jnkms + 0] * CnmReal;
        MnmTarget[0] -= MnmSource[2 * jnkms + 1] * CnmImag;
        MnmTarget[1] += MnmSource[2 * jnkms + 0] * CnmImag;
        MnmTarget[1] += MnmSource[2 * jnkms + 1] * CnmReal;
      }
    }
    for(var m = k; m <= n; m++){
      if(var j - n >= m - k){
        let nm = n * n + n + m;
        let jnkms = (j - n) * (j - n + 1) / 2 - k + m;
        let ere = cosf(-m * beta);
        let eim = sinf(-m * beta);
        let ajnkm = rsqrtf(factorial[j - n - k + m]
                     * factorial[j - n + k - m]);
        var cnm = oddeven(k + j + m);
        cnm *= ajnkm / ajk * sharedYnm[nm];
        CnmReal = cnm * ere;
        CnmImag = cnm * eim;
        MnmTarget[0] += sharedMnmSource[2 * jnkms + 0] * CnmReal;
        MnmTarget[0] += sharedMnmSource[2 * jnkms + 1] * CnmImag;
        MnmTarget[1] += sharedMnmSource[2 * jnkms + 0] * CnmImag;
        MnmTarget[1] -= sharedMnmSource[2 * jnkms + 1] * CnmReal;
      }
    }
  }
}

fn m2m(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id:vec3<u32>)
{
  numInteraction = 1;
  var j=0;
  for(j = 0;j < numExpansions; j++){
    for(k = 0; k <= j; k++){
      i = j * (j + 1) / 2 + k;
      if(i==local_id.x){break;}
    }
  }
  

  for(var ij = 0; ij < numInteraction; ij++){
   // todo: get MnmSource

    let r = cart2sph(dist);
    let rho = r.x; let alpha = r.y; beta= r.z;

    calculate_ynm(sharedYnm, rho, alpha);

    let  MnmResult = core(j, beta);
  }
  // todo: write output
}