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
@group(0) @binding(1) var<storage, read_write> Lnm: array<f32>;
@group(0) @binding(2) var<storage, read_write> command: array<i32>;
@group(0) @binding(3) var<storage, read_write> Dnm: array<f32>;
@group(0) @binding(4) var<storage, read_write> LnmOld: array<f32>;





fn oddeven(n:i32) ->f32 {
   if((n & 1) == 1) {return -1;} else {return 1;}
}

const boxPerGroup = 1u;
const threadsPerGroup = 64u;
var<workgroup> sharedLnmSource: array<f32, 2 * threadsPerGroup>; 

@compute @workgroup_size(threadsPerGroup, boxPerGroup) 
fn l2l(@builtin(local_invocation_id) local_id : vec3<u32>, 
       @builtin(workgroup_id) group_id:vec3<u32>)
{
  const commandLength = 2;
  let threadId = i32(local_id.x);
  let groupId = i32(group_id.x);
  let lnmSource = command[groupId * commandLength + 0];//
  let je =  command[groupId * commandLength + 1];// 

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
  
  var LnmResult : vec2f;
  var tempTarget : vec2f;

   

  for(var ij = 0u; ij < numInteraction; ij++){
    let LnmSourceOffset=lnmSource*numCoefficients;
    sharedLnmSource[2 * threadId] = LnmOld[2 * (LnmSourceOffset + threadId) ]; 
    sharedLnmSource[2 * threadId + 1] = LnmOld[2 * (LnmSourceOffset + threadId) + 1];
    
   
    workgroupBarrier();
    let rho = uniforms.boxSize * sqrt(3.0) / 4;
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
        tempTarget.x += DnmReal * sharedLnmSource[2 * nks + 0];
        tempTarget.x += DnmImag * sharedLnmSource[2 * nks + 1];
        tempTarget.y -= DnmReal * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedLnmSource[2 * nks + 0];
      }

      for (var k = 0; k <= n; k++)
      {
        let nks = n * (n + 1) / 2 + k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedLnmSource[2 * nks + 0];
        tempTarget.x -= DnmImag * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmReal * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedLnmSource[2 * nks + 0];
      }
      workgroupBarrier();

      
      sharedLnmSource[2 * nms ] = tempTarget.x;
      sharedLnmSource[2 * nms + 1] = tempTarget.y;

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
        let jnk = (n - j) * (n - j) + n - j;
        fnmm = 1.0;
        for (var i = 0; i < j - n - k; i++){fnmm = fnmm * f32(i + 1);}
        fnpm = 1.0;
        for (var i = 0; i < j - n + k; i++){fnpm = fnpm * f32(i + 1);}
        let ank = oddeven(n) * inverseSqrt(fnmm * fnpm);
        fnpm = 1.0;
        for (var i = 0; i < n-j; i++){fnpm = fnpm * f32(i + 1);}
        let ajn = oddeven(n-j) / fnpm;
        let sr = ank * ajn / ajk;
        let CnmReal = sr * rhon; // *Ynm
        let CnmImag = 0f; //sr * rhon;

        tempTarget.x += sharedLnmSource[2 * nks + 0] * CnmReal;
        tempTarget.x -= sharedLnmSource[2 * nks + 1] * CnmImag;
        tempTarget.y += sharedLnmSource[2 * nks + 0] * CnmImag;
        tempTarget.y += sharedLnmSource[2 * nks + 1] * CnmReal;
        rhon *= rho;

      }
      workgroupBarrier();

      sharedLnmSource[2 * jks ] = tempTarget.x;
      sharedLnmSource[2 * jks + 1] = tempTarget.y;
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
        tempTarget.x += DnmReal * sharedLnmSource[2 * nks + 0];
        tempTarget.x += DnmImag * sharedLnmSource[2 * nks + 1];
        tempTarget.y -= DnmReal * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedLnmSource[2 * nks + 0];
      }
      for (var k = 0; k <= n; k++)
      {
        let nks = n * (n + 1) / 2 + k;
        let nmk = jbase + (4 * n * n * n + 6 * n * n + 5 * n) / 3 + m * (2 * n + 1) + k;
        let DnmReal = Dnm[2 * nmk + 0];
        let DnmImag = Dnm[2 * nmk + 1];
        tempTarget.x += DnmReal * sharedLnmSource[2 * nks + 0];
        tempTarget.x -= DnmImag * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmReal * sharedLnmSource[2 * nks + 1];
        tempTarget.y += DnmImag * sharedLnmSource[2 * nks + 0];
      }
    }

   workgroupBarrier();
  }

  if(threadId<numCoefficients){

    //debugTemp=vec2f(sharedLnmSource[2 * threadId],tempTarget.x);

    let targetIndex = i32(numCoefficients) * groupId + threadId;
    Lnm[targetIndex*2] = tempTarget.x;
    Lnm[targetIndex*2 +1] = tempTarget.y;
    // Lnm[targetIndex*2] = debugTemp.x;
    // Lnm[targetIndex*2 +1] = debugTemp.y;
  }

}