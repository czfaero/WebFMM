struct Uniforms {
 srcPackSize : i32, //8
 numThread: i32, 
 commandLength : i32, 
 commandTarget : i32,
 vectorLength : i32, // 55*2
}


@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read_write> command: array<i32>;

const group_size = 256u;
@compute @workgroup_size(group_size) 
fn sum(@builtin(local_invocation_id) local_id : vec3<u32>, 
              @builtin(workgroup_id) group_id:vec3<u32>) {
  let id =i32(group_id.x * group_size + local_id.x);
  if(id >= uniforms.numThread) {return;}
  var r = 0f;
  for(var i = 0; i < uniforms.srcPackSize; i++){
    r += src[id * uniforms.srcPackSize + i];
  }
  let vectorIndex = id /  uniforms.vectorLength;
  let innerIndex = id %  uniforms.vectorLength;
  let targetVectorIndex = command [vectorIndex * uniforms.commandLength + uniforms.commandTarget];
  let offset = targetVectorIndex * uniforms.vectorLength + innerIndex;
  dst[offset] = r;
    
  //dst[512*110+vectorIndex]=f32(targetVectorIndex);
  
}
