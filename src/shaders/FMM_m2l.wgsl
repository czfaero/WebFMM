const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;

const numExpansions = 10;
const numExpansion2 = numExpansions * numExpansions;
const numCoefficients = numExpansions * (numExpansions + 1) / 2; //55
const DnmSize = (4 * numExpansion2 * numExpansions - numExpansions) / 3;
const numRelativeBox = 512; 
const maxM2LInteraction    = 189; 

struct Uniforms {
  boxSize:f32
}




@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> Mnm: array<f32>;
@group(0) @binding(2) var<storage, read_write> command: array<i32>;
@group(0) @binding(4) var<storage, read_write> Dnm: array<f32>;
@group(0) @binding(5) var<storage, read_write> Lnm: array<f32>;


fn unmorton1(je:i32)-> vec3f
{
  let boxSize= uniforms.boxSize;
  var nb=je-1;
  var mortonIndex3D :array<i32,3>;
  var k = 0;
  var i = 1;
  while (nb != 0)
  {
    var j = 2 - k;
    mortonIndex3D[j] = mortonIndex3D[j] + nb % 2 * i;
    nb = nb / 2;
    j = k + 1;
    k = j % 3;
    if (k == 0)
       {i = i * 2;}
  }
  let nd = mortonIndex3D[0];
  mortonIndex3D[0] = mortonIndex3D[1];
  mortonIndex3D[1] = mortonIndex3D[2];
  mortonIndex3D[2] = nd;
  return vec3f(f32(mortonIndex3D[0] - 3),
               f32(mortonIndex3D[1] - 3),
               f32(mortonIndex3D[2] - 3)) * boxSize;
}



fn oddeven(n:i32) ->f32 {
   if((n & 1) == 1) {return -1;} else {return 1;}
}

const boxPerGroup = 1u;
const threadsPerGroup = 64u;
var<workgroup> sharedMnmSource: array<f32, 2 * threadsPerGroup>; 

@compute @workgroup_size(threadsPerGroup, boxPerGroup) 
fn m2l(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id:vec3<u32>)
{
  const commandLength = 2 * maxM2LInteraction + 1;
  let threadId = i32(local_id.x);
  let groupId = i32(group_id.x);
  var debugTemp:vec2f;

  var ng:array<i32,threadsPerGroup>;
  var mg:array<i32,threadsPerGroup>;
  for (var n = 0; n < numExpansions; n++) {
    for (var m = 0; m <= n; m++) {
      let nms = n * (n + 1) / 2 + m;
      ng[nms] = n;
      mg[nms] = m;
    }
  }
  
  let maxInteraction=i32(maxM2LInteraction);

  var tempTarget : vec2f;
  let numInteraction = command[groupId*commandLength];
  var LnmResult: vec2f=vec2f(0,0);



  for(var ij = 0; ij < maxInteraction; ij++){
    let mnmSource = command[groupId * commandLength +1 + ij*2];
    let je        = command[groupId * commandLength +1 + ij*2 + 1];
    let MnmSourceOffset = mnmSource * numCoefficients;
    sharedMnmSource[2 * threadId] = Mnm[2 * (MnmSourceOffset + threadId) ]; 
    sharedMnmSource[2 * threadId + 1] = Mnm[2 * (MnmSourceOffset + threadId) + 1];
    workgroupBarrier();
    
    // debug01
    //if(ij==0){debugTemp=vec2f(sharedMnmSource[2 * threadId],sharedMnmSource[2 * threadId+1]);}
  
    let dist = unmorton1(je);
    let rho = sqrt(dot(dist,dist)) + eps;

    // debug03
    //if(ij==0){debugTemp=vec2f(rho,f32(je));}
    {
      let jbase = (je - 1) * DnmSize;
      let n = ng[local_id.x];
      let m = mg[local_id.x];
      let nms = n * (n + 1) / 2 + m;

      for (var k = -n; k < 0; k++)
      {
        let nks = n * (n + 1) / 2 - k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
        tempTarget.x += DnmImag * sharedMnmSource[2 * nks + 1];
        tempTarget.y -= DnmReal * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
      }

      for (var k = 0; k <= n; k++)
      {
        let nks = n * (n + 1) / 2 + k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
        tempTarget.x -= DnmImag * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmReal * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
      }
      workgroupBarrier();

      
      sharedMnmSource[2 * nms ] = tempTarget.x;
      sharedMnmSource[2 * nms + 1] = tempTarget.y;

    }
    // debug02
    //if(ij==0){debugTemp=tempTarget;}
  
    workgroupBarrier();
    {
      let j = ng[local_id.x];
      let k = mg[local_id.x];
      let jks = j * (j + 1) / 2 + k;
      tempTarget = vec2f(0,0);
      var fnmm = 1.0;
      for (var i = 0; i < j - k; i++){fnmm = fnmm * f32(i + 1);}
      var fnpm = 1.0;
      for (var i = 0; i < j + k; i++){fnpm = fnpm * f32(i + 1);}
      let ajk = oddeven(j) * inverseSqrt(fnmm * fnpm);
      var rhon = 1.0 / pow(rho,f32(j+k+1));

      // debug06
      //if(ij==0){debugTemp=vec2f(rhon,ajk);}
      // debug07
      //if(ij==0){debugTemp=vec2f(fnmm,fnpm);}

      for (var n = abs(k); n <numExpansions; n++)
      {
        let nks = n * (n + 1) / 2 + k;
        let jnk = (j + n) * (j + n) + j + n;
        fnmm = 1.0;
        for (var i = 0; i < n - k; i++){fnmm = fnmm * f32(i + 1);}
        fnpm = 1.0;
        for (var i = 0; i < n + k; i++){fnpm = fnpm * f32(i + 1);}
        let ank = oddeven(n) * inverseSqrt(fnmm * fnpm);
        fnpm = 1.0;
        for (var i = 0; i < j + n; i++){fnpm = fnpm * f32(i + 1);}
        let ajn = oddeven(j + n) / fnpm;
        let sr = oddeven(j + k) * ank * ajk / ajn;
        let CnmReal = sr * rhon;
        let CnmImag = sr * rhon;

        tempTarget.x += sharedMnmSource[2 * nks + 0] * CnmReal;
        tempTarget.x -= sharedMnmSource[2 * nks + 1] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 0] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 1] * CnmReal;
        rhon /= rho;
      // debug08 OK
      //if(ij==0&&n == abs(k)){debugTemp=vec2f(rhon,sr);}
      // debug09 OK
      //if(ij==0&&n == abs(k)){debugTemp=vec2f(CnmReal,CnmImag);}
      // debug10 OK?
      //if(ij==0&&n == abs(k)){debugTemp=tempTarget; }
      // debug11
      //if(ij==0&&n == abs(k)+1){debugTemp=tempTarget;}
       // debug12
      //if(ij==0&&n == abs(k)+2){debugTemp=tempTarget;}
      // debug13 OK
      //if(ij==0&&n == abs(k)+2){debugTemp=vec2f(rhon,sr);}
      // debug14 
      //if(ij==0&&n == abs(k)){debugTemp=vec2f(sharedMnmSource[2 * nks + 0] ,sharedMnmSource[2 * nks + 1] );}
       // debug15
      //if(ij==0&&n == abs(k)){debugTemp=vec2f(Ynm[jnk*2] ,Ynm[jnk*2+1] );}
      }
      workgroupBarrier();
      sharedMnmSource[2 * jks ] = tempTarget.x;
      sharedMnmSource[2 * jks + 1] = tempTarget.y;
    }
    // debug04
    //if(ij==0){debugTemp=tempTarget;}
    workgroupBarrier();
    {
      let jbase = (je + numRelativeBox - 1) * DnmSize;
      let n = ng[local_id.x];
      let m = mg[local_id.x];
      let nms = n * (n + 1) / 2 + m;
      tempTarget = vec2f(0,0);
      for (var k = -n; k < 0; k++)
      {
        let nks = n * (n + 1) / 2 - k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
        tempTarget.x += DnmImag * sharedMnmSource[2 * nks + 1];
        tempTarget.y -= DnmReal * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
      }
      for (var k = 0; k <= n; k++)
      {
        let nks = n * (n + 1) / 2 + k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedMnmSource[2 * nks + 0];
        tempTarget.x -= DnmImag * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmReal * sharedMnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedMnmSource[2 * nks + 0];
      }
    }
    // debug05
    //if(ij==0){debugTemp=tempTarget;}
    if(ij<numInteraction){
      LnmResult+=tempTarget;
    }
    workgroupBarrier();
  }// end of loop


  if(threadId<numCoefficients){
    let targetIndex = i32(numCoefficients) * groupId + threadId;
    Lnm[targetIndex*2] = LnmResult.x;
    Lnm[targetIndex*2 +1] = LnmResult.y;
    // Lnm[targetIndex*2] = debugTemp.x;
    // Lnm[targetIndex*2 +1] = debugTemp.y;
  }
  
}