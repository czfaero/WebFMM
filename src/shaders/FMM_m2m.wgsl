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
@group(0) @binding(5) var<storage, read_write> resultBuffer: array<f32>;





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
  const commandLength = 3;
  let threadId = i32(local_id.x);
  let groupId = group_id.x;
  let mnmSource = command[groupId * commandLength + 0];//Mnm index
  let je =  command[groupId * commandLength + 1];// result of morton1
  let plainBoxIndex =  command[groupId * commandLength + 2];// 0 - 
  var debugTemp:vec2f;

  const numInteraction = 1u;
  var ng:array<i32,threadsPerGroup>;
  var mg:array<i32,threadsPerGroup>;
  for (var n = 0; n < numExpansions; n++) {
    for (var m = 0; m <= n; m++) {
      let nms = n * (n + 1) / 2 + m;
      ng[nms] = n;
      mg[nms] = m;
    }
  }
  
  var MnmResult : vec2f;
  var tempTarget : vec2f;

  // debug 14
  //debugTemp=vec2f(f32(mnmSource),f32(je));

  for(var ij = 0u; ij < numInteraction; ij++){
    let MnmSourceOffset = mnmSource * numCoefficients;
    sharedMnmSource[2 * threadId] = Mnm[2 * (MnmSourceOffset + threadId) ]; 
    sharedMnmSource[2 * threadId + 1] = Mnm[2 * (MnmSourceOffset + threadId) + 1];
    workgroupBarrier();
    let rho = uniforms.boxSize * sqrt(3.0) / 4;
    {
      let jbase = (je - 1) * DnmSize;
      let n = ng[local_id.x];
      let m = mg[local_id.x];
      let nms = n * (n + 1) / 2 + m;
      // debug 00 nms // OK!
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
        // debug 02 
        //if(k==-n){debugTemp=vec2f(DnmReal,DnmImag);}
        // debug 12
        //if(k==-n){debugTemp=tempTarget;}
        // debug 13
        //if(k==-n){debugTemp=vec2f(sharedMnmSource[2 * nks + 0],sharedMnmSource[2 * nks + 1]);}
      }
      // debug 01 tempTarget //OK!
      //debugTemp=tempTarget;
      // debug03 
      //debugTemp=vec2f(f32(je),f32(n));
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
      //debug 04 tempTarget // OK?
      //debugTemp=tempTarget;
      
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
        let CnmReal = sr * Ynm[jnk*2] * rhon;
        let CnmImag = sr * Ynm[jnk*2+1] * rhon;
        // debug06 //OK!
        // if(n==0){debugTemp=vec2f(CnmReal,CnmImag);}
        // debug07 //OK!
        // if(n==0){debugTemp=vec2f(f32(jnk),Ynm[jnk*2]);}
        tempTarget.x += sharedMnmSource[2 * nks + 0] * CnmReal;
        tempTarget.x -= sharedMnmSource[2 * nks + 1] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 0] * CnmImag;
        tempTarget.y += sharedMnmSource[2 * nks + 1] * CnmReal;
        rhon *= rho;
        // debug08 //OK?
        //if(n==0){debugTemp=tempTarget;}
        //debug09 //OK?  2* 0.001error
        //if(n==1){debugTemp=tempTarget;}
      }
      workgroupBarrier();
      //debug 05 tempTarget //OK  8* 0.001 error
      // debugTemp=tempTarget;
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
    //debug 10 tempTarget //OK 2* 0.001 error
    debugTemp=tempTarget;
   MnmResult += tempTarget;
   workgroupBarrier();
  }
  //debug 11 
  //debugTemp=MnmResult;
  if(threadId<numCoefficients){
    let smallBoxIndex = plainBoxIndex % 8;
    let targetBoxIndex = plainBoxIndex / 8;
    let targetIndex = targetBoxIndex * numCoefficients *2 *8
                    + threadId * 8 *2
                    + smallBoxIndex;
    resultBuffer[targetIndex]=MnmResult.x;
    resultBuffer[targetIndex + 8]=MnmResult.y;
  }
}