const PI = 3.14159265358979323846;
const inv4PI = 0.25/PI;
const eps = 1e-6;

const numExpansions = 10;
const numExpansion2 = numExpansions * numExpansions;
const numCoefficients = numExpansions * (numExpansions + 1) / 2; //55
const DnmSize = (4 * numExpansion2 * numExpansions - numExpansions) / 3;
const numRelativeBox = 512; 

struct Uniforms {
  boxSize:f32
}




@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> Mnm: array<f32>;
@group(0) @binding(2) var<storage, read_write> command: array<i32>;
@group(0) @binding(3) var<storage, read_write> Ynm: array<f32>;
@group(0) @binding(4) var<storage, read_write> Dnm: array<f32>;
@group(0) @binding(5) var<storage, read_write> ng: array<i32>;
@group(0) @binding(6) var<storage, read_write> mg: array<i32>;




fn oddeven(n:i32) ->f32 {
   if((n & 1) == 1) {return -1;} else {return 1;}
}

const boxPerGroup = 1u;
const threadsPerGroup = 64u;
var<workgroup> sharedMnmSource: array<f32, 2 * threadsPerGroup>; 

@compute @workgroup_size(threadsPerGroup, boxPerGroup) 
fn m2m(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id:vec3<u32>)
{
  let threadId = i32(local_id.x);
  let boxIndex = group_id.x;
  let mnmSource = command[boxIndex * 3 + 0];//Mnm index
  let je =  command[boxIndex * 3 + 1];// result of morton1
  let mnmTarget =  command[boxIndex * 3 + 2];

  const numInteraction = 1u;

  
  var MnmResult : vec2f;
  var tempTarget : vec2f;

  for(var ij = 0u; ij < numInteraction; ij++){
    let MnmSourceOffset = mnmSource * numCoefficients * 2;
    sharedMnmSource[2 * threadId] = Mnm[2 * (MnmSourceOffset + threadId) ]; 
    sharedMnmSource[2 * threadId + 1] = Mnm[2 * (MnmSourceOffset + threadId) + 1];
    workgroupBarrier();
    let rho = uniforms.boxSize * sqrt(3.0) / 4;
    {
      let jbase = (je - 1) * DnmSize;
      let n = ng[local_id.x];
      let m = mg[local_id.x];
      let nms = n * (n + 1) / 2 + m;
      // for (i = 0; i < 2; i++)
      //   tempTarget[i] = 0;
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
      var rhon = 1.0;
      for (var n = 0; n <= j - abs(k); n++)
      {
        let nks = (j - n) * (j - n + 1) / 2 + k;
        let jnk = n * n + n;
        fnmm = 1.0;
        for (var i = 0; i < j - n - k; i++){fnmm = fnmm * f32(i + 1);}
        fnpm = 1.0;
        for (var i = 0; i < j - n + k; i++){fnpm = fnpm * f32(i + 1);}
        let ank = oddeven(j - n) * inverseSqrt(fnmm * fnpm);
        fnpm = 1.0;
        for (var i = 0; i < n; i++){fnpm = fnpm * f32(i + 1);}
        let ajn = oddeven(n) / fnpm;
        let sr = oddeven(n) * ank * ajn / ajk;
        let CnmReal = sr * Ynm[jnk*2] * rhon; // carefully
        let CnmImag = sr * Ynm[jnk*2+1] * rhon;
        tempTarget.x += sharedMnmSource[2 * nks + 0] * CnmReal;
        tempTarget.x -= sharedMnmSource[2 * nks + 1] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 0] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 1] * CnmReal;
        rhon *= rho;
      }
      workgroupBarrier();
      sharedMnmSource[2 * jks ] = tempTarget.x;
      sharedMnmSource[2 * jks + 1] = tempTarget.y;
    }
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
   MnmResult += tempTarget;
   workgroupBarrier();
  }
  for(var x=0;x<=8;x++){
    if(threadId%8==x){
      let mnmTargetOffset = mnmTarget * numCoefficients * 2 + threadId;
      Mnm[mnmTargetOffset*2] += MnmResult.x;
      Mnm[mnmTargetOffset*2+1] += MnmResult.y;
    }
    workgroupBarrier();
  }

  // debug
  //Mnm[boxIndex]=f32(mnmTarget);
}