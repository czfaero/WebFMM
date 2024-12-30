// import wgsl_p2p from './shaders/FMM_p2p.wgsl';
// import wgsl_p2m from './shaders/FMM_p2m.wgsl';
// import wgsl_m2m from './shaders/FMM_m2m.wgsl';
// import wgsl_m2l from './shaders/FMM_m2l.wgsl';
// import wgsl_l2l from './shaders/FMM_l2l.wgsl';
// import wgsl_l2p from './shaders/FMM_l2p.wgsl';
// //import wgsl_buffer_sum from '../shaders/buffer_sum.wgsl';

// import { cart2sph, GetIndex3D, GetIndexFrom3D } from "../utils";
// import { IFMMKernel } from '../IFMMKernel';
// import { FMMSolver } from '../FMMSolver';


// const eps = 1e-6;
// const inv4PI = 0.25 / Math.PI;

// const SIZEOF_32 = 4;


// const maxM2LInteraction = 189;


// class Complex {
//   re: number;
//   im: number;
//   multiply(cn2: Complex): Complex {
//     const cn1 = this;
//     return new Complex(
//       cn1.re * cn2.re - cn1.im * cn2.im,
//       cn1.re * cn2.im + cn1.im * cn2.re);
//   }
//   multiplyReal(x: number): Complex {
//     const cn1 = this;
//     return new Complex(
//       cn1.re * x,
//       cn1.im * x);
//   }
//   conj() {
//     return new Complex(this.re, -this.im);
//   }
//   exp() {
//     const tmp = Math.exp(this.re);
//     return new Complex(
//       Math.cos(this.im) * tmp,
//       Math.sin(this.im) * tmp
//     );
//   }

//   static fromBuffer(b: Float64Array, i: number): Complex {
//     return new Complex(b[i * 2], b[i * 2 + 1]);
//   };
//   constructor(re: number, im = 0) {
//     this.re = re;
//     this.im = im;
//   }
// }



// export class KernelWgpuxxxxxxxxxx implements IFMMKernel {
//   core: FMMSolver;
//   debug: boolean;
//   constructor(core: FMMSolver) {
//     this.debug = false;
//     this.core = core;
//   }
//   adapter: GPUAdapter;
//   device: GPUDevice;
//   nodeBufferGPU: GPUBuffer;
//   commandBufferGPU: GPUBuffer;
//   readBufferGPU: GPUBuffer;
//   maxThreadCount: number;
//   maxWorkgroupCount: number;
//   accelBuffer: Float32Array;
//   accelBufferGPU: GPUBuffer;
//   uniformBufferSize: number;
//   uniformBufferGPU: GPUBuffer;
//   particleOffsetGPU: GPUBuffer;
//   factorialGPU: GPUBuffer;
//   mnmBufferGPU: GPUBuffer;
//   lnmBufferGPU: GPUBuffer;
//   i2nmBufferGPU: GPUBuffer;
//   debug_info: any;
//   shaders: any;
//   /** only for debug. complex [numBoxIndexTotal][numCoefficients] */
//   Mnm: Array<Float32Array>;
//   /** only for debug. complex [numBoxIndexLeaf][numCoefficients] */
//   Lnm: Array<Float32Array>;
//   factorial: Float32Array;

//   async Init() {
//     const core = this.core;
//     const tree = core.tree;
//     this.accelBuffer = new Float32Array(tree.nodeCount * 3);
//     this.adapter = await navigator.gpu.requestAdapter();
//     this.device = await this.adapter.requestDevice();
//     // to-do: check limit
//     console.log(this.adapter);
//     this.maxThreadCount = 256;
//     this.maxWorkgroupCount = 256;
//     // this.cmdBufferLength = this.maxThreadCount * this.maxWorkgroupCount * 2;// to-do: set a good value
//     // this.cmdBufferSize = this.cmdBufferLength * SIZEOF_32;
//     this.nodeBufferGPU = this.device.createBuffer({
//       size: tree.nodeBuffer.byteLength,
//       usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
//     });
//     this.uniformBufferSize = 16 * SIZEOF_32;// see shader

//     this.uniformBufferGPU = this.device.createBuffer({
//       size: this.uniformBufferSize,
//       usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
//     });
//     this.accelBufferGPU = this.device.createBuffer({
//       size: this.core.numExpansions * 2 * SIZEOF_32,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//     });
//     this.factorialGPU = this.device.createBuffer({
//       size: this.core.numExpansions * 2 * SIZEOF_32,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//     });
//     this.mnmBufferGPU = this.device.createBuffer({
//       size: core.tree.numBoxIndexTotal * core.MnmSize * 2 * SIZEOF_32,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//     });
//     this.lnmBufferGPU = this.device.createBuffer({
//       size: core.tree.numBoxIndexTotal * core.MnmSize * 2 * SIZEOF_32,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//     });
//     this.readBufferGPU = this.device.createBuffer({
//       size: Math.max(this.accelBufferGPU.size, this.lnmBufferGPU.size),
//       usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
//     });

//     this.commandBufferGPU = this.device.createBuffer({
//       size: 776192,// TO-DO
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
//     });


//     this.device.queue.writeBuffer(this.nodeBufferGPU, 0, tree.nodeBuffer);

//     this.shaders = {
//       p2p: wgsl_p2p,
//       p2m: wgsl_p2m,
//       m2m: wgsl_m2m,
//       m2l: wgsl_m2l,
//       l2l: wgsl_l2l,
//       l2p: wgsl_l2p,
//     }
//     const nameList = "p2p p2m m2m m2l l2l l2p".split(" ");
//     for (const n of nameList) {
//       this.shaders[n] = this.device.createShaderModule({
//         code: this.shaders[n],
//       })
//     }

//     if (this.debug) {
//       this.debug_info = {};
//       this.debug_info["particleBufferGPU"] = this.nodeBufferGPU.size;
//       this.debug_info["uniformBufferGPU"] = this.uniformBufferGPU.size;
//     }

//     await this.precalc();
//   }

//   async precalc() {
//     const core = this.core;
//     let i2nm = new Int32Array(core.MnmSize * 4);
//     this.i2nmBufferGPU = this.device.createBuffer({
//       size: i2nm.byteLength,
//       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//     });
//     for (let n = 0; n < core.numExpansions; n++) {
//       for (let m = 0; m <= n; m++) {
//         let i = n * (n + 1) / 2 + m;
//         i2nm[i * 4 + 0] = n;
//         i2nm[i * 4 + 1] = m;
//       }
//     }

//     for (let n = 0; n < core.numExpansions; n++) {
//       for (let m = -n; m <= n; m++) {
//         let i = n * n + n + m;
//         i2nm[i * 4 + 2] = n;
//         i2nm[i * 4 + 3] = m;
//       }
//     }

//     this.device.queue.writeBuffer(this.i2nmBufferGPU, 0, i2nm);
//   }
//   async p2p() {
//     if (this.debug) {
//       this.debug_p2p_call_count = 0;
//       this.debug_info["events"] = [];
//       this.debug_info["events"].push({ time: performance.now(), tag: "start" });
//     }
//     const tree = this.core.tree;
//     const numBoxIndex = this.core.tree.numBoxIndexLeaf;


//     const particleOffsetBuffer = new Uint32Array(numBoxIndex * 2);
//     for (let i = 0; i < numBoxIndex; i++) {
//       particleOffsetBuffer[i * 2] = tree.nodeStartOffset[0][i];
//       particleOffsetBuffer[i * 2 + 1] = tree.nodeEndOffset[1][i];
//     }
//     this.particleOffsetGPU = this.device.createBuffer({
//       size: particleOffsetBuffer.byteLength,
//       usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
//     });
//     this.device.queue.writeBuffer(this.particleOffsetGPU, 0, particleOffsetBuffer);

//     let commandCount = 0;
//     const commandSize = 2;
//     const maxCommandCount = this.maxThreadCount * this.maxWorkgroupCount;
//     if (maxCommandCount > this.maxThreadCount * this.maxWorkgroupCount) {
//       throw `Thread Count (${maxCommandCount}) > MaxThread(${this.maxThreadCount * this.maxWorkgroupCount})`;
//     }
//     const command = new Uint32Array(maxCommandCount * commandSize);




//     if (this.debug) {
//       this.debug_info["commandBufferGPU"] = this.commandBufferGPU.size;
//       this.debug_info["particleOffsetGPU"] = this.particleOffsetGPU.size;
//       this.debug_info["readBufferGPU"] = this.readBufferGPU.size;
//     }

//     for (let ii = 0; ii < numBoxIndex; ii++) {
//       for (let i = tree.nodeStartOffset[0][ii]; i <= tree.nodeEndOffset[0][ii]; i++) {
//         for (let ij = 0; ij < this.core.interactionCounts[ii]; ij++) {
//           const jj = this.core.interactionList[ii][ij];
//           command[commandCount * 2] = i;
//           command[commandCount * 2 + 1] = jj;
//           commandCount++;
//           if (commandCount == maxCommandCount) {
//             this.device.queue.writeBuffer(this.commandBufferGPU, 0, command);
//             await this.p2p_ApplyAccel(command, commandCount);
//             commandCount = 0;
//             command.fill(0);
//           }
//         }
//       }
//     }

//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, commandCount * 2);
//     await this.p2p_ApplyAccel(command, commandCount);
//     if (this.debug) {
//       this.debug_info["events"].push({ time: performance.now(), tag: "end" });
//     }

//     if (this.debug) {
//       await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//       const handle = this.readBufferGPU.getMappedRange();
//       this.accelBuffer = new Float32Array(handle);
//       // this.readBufferGPU.unmap();
//       console.log(`debug_p2p_call_count: ${this.debug_p2p_call_count}`);
//       console.log(this.accelBuffer);
//       const startTime = this.debug_info["events"][0].time;
//       this.debug_info["events"].forEach(o => o.time -= startTime);
//       //console.log(JSON.stringify(this.debug_info))
//       console.log(this.debug_info);
//       this.readBufferGPU.unmap();
//     }
//   }

//   debug_p2p_call_count: number;
//   async p2p_ApplyAccel(cmdBuffer: Uint32Array, commandCount: number) {
//     if (this.debug) {
//       this.debug_info["events"].push({ time: performance.now(), tag: "prepare submit", i: this.debug_p2p_call_count, thread_count: commandCount });
//       this.debug_p2p_call_count++;
//     }
//     const uniformBuffer = new Uint32Array(4);
//     uniformBuffer.set([commandCount]);
//     this.device.queue.writeBuffer(
//       this.uniformBufferGPU,
//       0,
//       uniformBuffer
//     );

//     //console.log(cmdBuffer); throw "pause";
//     await this.RunCompute("p2p",
//       [this.uniformBufferGPU, this.nodeBufferGPU, this.resultBufferGPU, this.commandBufferGPU, this.particleOffsetGPU],
//       this.maxWorkgroupCount, this.resultBufferGPU, [this.resultBufferGPU]
//     );

//     //console.log("applyaccel");
//     await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//     const handle = this.readBufferGPU.getMappedRange();
//     let tempAccelBuffer = new Float32Array(handle);
//     // console.log(cmdBuffer);
//     // console.log(tempAccelBuffer);
//     for (let i = 0; i < commandCount; i++) {
//       let index = cmdBuffer[i * 2];
//       this.accelBuffer[index * 3] += tempAccelBuffer[i * 3];
//       this.accelBuffer[index * 3 + 1] += tempAccelBuffer[i * 3 + 1];
//       this.accelBuffer[index * 3 + 2] += tempAccelBuffer[i * 3 + 2];
//     }
//     this.readBufferGPU.unmap();

//   }
//   async p2m() {
//     const core = this.core;
//     const tree = core.tree;

//     let fact = 1.0;
//     let factorial = new Float32Array(2 * this.core.numExpansions);
//     for (let m = 0; m < factorial.length; m++) {
//       factorial[m] = fact;
//       fact = fact * (m + 1);
//     }
//     this.device.queue.writeBuffer(this.factorialGPU, 0, factorial);
//     this.factorial = factorial;
//     const numBoxIndex = this.core.tree.numBoxIndexLeaf;

//     // command (boxId)
//     let command = new Uint32Array(numBoxIndex);
//     for (let i = 0; i < numBoxIndex; i++) {
//       command[i] = core.tree.boxIndexFull[i];
//     }
//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, numBoxIndex);

//     const boxSize = core.tree.rootBoxSize / (1 << core.tree.maxLevel);

//     let maxParticlePerBox = 0;
//     for (let jj = 0; jj < numBoxIndex; jj++) {
//       let c = tree.nodeEndOffset[jj] - tree.nodeStartOffset[0][jj] + 1;
//       if (c > maxParticlePerBox) { maxParticlePerBox = c; }
//     }

//     const uniformBuffer = new Float32Array(4);
//     uniformBuffer[0] = boxSize;
//     uniformBuffer[1] = core.tree.boxMinX;
//     uniformBuffer[2] = core.tree.boxMinY;
//     uniformBuffer[3] = core.tree.boxMinZ;
//     const uniformBuffer2 = new Uint32Array(3);
//     uniformBuffer2[0] = numBoxIndex;
//     uniformBuffer2[1] = this.core.numExpansions;
//     uniformBuffer2[2] = maxParticlePerBox;
//     this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
//     this.device.queue.writeBuffer(this.uniformBufferGPU, uniformBuffer.byteLength, uniformBuffer2);

//     await this.RunCompute("p2m",
//       [this.uniformBufferGPU, this.nodeBufferGPU, this.mnmBufferGPU, this.commandBufferGPU, this.particleOffsetGPU, this.factorialGPU],
//       numBoxIndex, this.mnmBufferGPU
//     );
//     if (this.debug) {
//       console.log("debug p2m");
//       await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//       const handle = this.readBufferGPU.getMappedRange();
//       let tempReadBuffer = new Float32Array(handle);
//       this.Mnm = new Array(core.tree.numBoxIndexTotal);
//       for (let i = 0; i < numBoxIndex; i++) {
//         const MnmVec = new Float32Array(this.core.numCoefficients * 2);
//         for (let j = 0; j < this.core.numCoefficients * 2; j++) {
//           MnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
//         }
//         this.Mnm[i] = MnmVec;
//       }
//       this.readBufferGPU.unmap();
//       //console.log(this.Mnm)
//     }
//     //throw "pause"
//   }

//   async m2m(numLevel: number) {
//     const core = this.core;

//     //?
//     const numBoxIndexOld = core.tree.levelOffset[numLevel - 1] - core.tree.levelOffset[numLevel];
//     console.log("old", numBoxIndexOld)
//     let command = new Int32Array(numBoxIndexOld * 3);
//     const boxPerGroup = 1;
//     const commandLength = 3;

//     let command_sum = new Int32Array(1 << 3 * numLevel);

//     for (let jj = 0; jj < numBoxIndexOld; jj++) {
//       let jb = jj + core.tree.levelOffset[numLevel];
//       let nfjp = Math.trunc(core.tree.boxIndexFull[jb] / 8);
//       let nfjc = core.tree.boxIndexFull[jb] % 8;
//       let ib = core.tree.boxIndexMaskBuffers[numLevel][nfjp] + core.tree.levelOffset[numLevel - 1];// MnmIndex
//       let boxIndex3D = GetIndex3D(nfjc);
//       boxIndex3D.x = 4 - boxIndex3D.x * 2;
//       boxIndex3D.y = 4 - boxIndex3D.y * 2;
//       boxIndex3D.z = 4 - boxIndex3D.z * 2;
//       let je = GetIndexFrom3D(boxIndex3D);
//       command[jj * commandLength + 0] = jb;//Mnm index
//       command[jj * commandLength + 1] = je + 1;
//       command[jj * commandLength + 2] = core.tree.boxIndexFull[jb];//to-do: check for empty box or more level 
//       //command[jj * commandLength + 3] = ib;
//       //console.log(command[jj * 3 + 2])
//       command_sum[nfjp] = ib;
//     }
//     //console.log(command)
//     //console.log(command_sum)

//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, numBoxIndexOld * 3);
//     const boxSize = core.tree.rootBoxSize / (1 << numLevel);

//     const uniformBuffer = new Float32Array(1);
//     uniformBuffer[0] = boxSize;
//     this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);

//     if (this.resultBufferGPU.size < (1 << 3 * numLevel) * 8 * this.core.numCoefficients * 2 * SIZEOF_32) {
//       throw "resultBufferGPU < numBoxIndexOld * numCoefficients * 2";
//     }
//     this.RunCompute("m2m",
//       [this.uniformBufferGPU, this.mnmBufferGPU, this.commandBufferGPU
//         , this.dnmBufferGPU, this.resultBufferGPU],
//       numBoxIndexOld / boxPerGroup, this.resultBufferGPU, [this.resultBufferGPU]
//     );

//     if (this.debug) {


//     }
//     const numThread = numBoxIndexOld / 8 * core.numCoefficients * 2;
//     const uniformBuffer_sum = new Int32Array(5);
//     uniformBuffer_sum[0] = 8; //srcPackSize 
//     uniformBuffer_sum[1] = numThread;
//     uniformBuffer_sum[2] = 1;//commandLength
//     uniformBuffer_sum[3] = 0;//commandTarget
//     uniformBuffer_sum[4] = core.numCoefficients * 2;//vectorLength 
//     this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer_sum);
//     const numGroup = Math.ceil(numThread / 256);
//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command_sum);
//     await this.RunCompute("sum",
//       [this.uniformBufferGPU, this.resultBufferGPU, this.mnmBufferGPU, this.commandBufferGPU],
//       numGroup,
//       this.mnmBufferGPU
//     );

//     if (this.debug) {
//       //console.log("debug m2m")
//       await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//       const handle = this.readBufferGPU.getMappedRange();
//       let tempReadBuffer = new Float32Array(handle);
//       this.Mnm = new Array(core.tree.numBoxIndexTotal);
//       for (let i = 0; i < core.tree.numBoxIndexTotal; i++) {
//         const MnmVec = new Float32Array(this.core.numCoefficients * 2);
//         for (let j = 0; j < this.core.numCoefficients * 2; j++) {
//           MnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
//         }
//         this.Mnm[i] = MnmVec;
//       }
//       this.readBufferGPU.unmap();
//       //console.log("Mnm:")
//       //console.log(this.Mnm);
//       //throw "pause after m2m";
//     }
//   }
//   async m2l(numLevel: number) {

//     const core = this.core;
//     const boxCount = core.tree.levelBoxCounts[numLevel];
//     const commandLength = 2 * maxM2LInteraction + 1;//count, pairs
//     let command = new Int32Array(commandLength * boxCount);

//     for (let ii = 0; ii < boxCount; ii++) {
//       let ib = ii + core.tree.levelOffset[numLevel - 1];
//       let indexi = GetIndex3D(core.tree.boxIndexFull[ib]);
//       let ix = indexi.x,
//         iy = indexi.y,
//         iz = indexi.z;

//       command[ii * commandLength] = core.interactionCounts[ii];
//       for (let ij = 0; ij < core.interactionCounts[ii]; ij++) {
//         let jj = core.interactionList[ii][ij];
//         let jbd = jj + core.tree.levelOffset[numLevel - 1];
//         let indexj = GetIndex3D(core.tree.boxIndexFull[jbd]);
//         let jx = indexj.x, jy = indexj.y, jz = indexj.z;

//         let je = GetIndexFrom3D({ x: ix - jx + 3, y: iy - jy + 3, z: iz - jz + 3 });
//         let jb = jj + core.tree.levelOffset[numLevel - 1];
//         command[ii * commandLength + 1 + ij * 2] = jb;//Mnm index
//         command[ii * commandLength + 1 + ij * 2 + 1] = je + 1;

//       }
//     }
//     // console.log("m2l command:");
//     // console.log(command);
//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);

//     const boxSize = core.tree.rootBoxSize / (1 << numLevel);

//     const uniformBuffer = new Float32Array(1);
//     uniformBuffer[0] = boxSize;
//     this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
//     this.RunCompute("m2l",
//       [this.uniformBufferGPU, this.mnmBufferGPU, this.commandBufferGPU
//         , this.dnmBufferGPU, this.lnmBufferGPU],
//       boxCount, this.lnmBufferGPU
//     );
//     if (this.debug) {
//       //console.log("debug m2l")
//       {
//         await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//         const handle = this.readBufferGPU.getMappedRange();
//         let tempReadBuffer = new Float32Array(handle);
//         {
//           this.Lnm = new Array(core.tree.numBoxIndexLeaf);
//           for (let i = 0; i < core.tree.numBoxIndexTotal; i++) {
//             const LnmVec = new Float32Array(this.core.numCoefficients * 2);
//             for (let j = 0; j < this.core.numCoefficients * 2; j++) {
//               LnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
//             }
//             this.Lnm[i] = LnmVec;
//           }
//         }
//         //console.log("Lnm:")
//         //console.log(this.Lnm);
//         this.readBufferGPU.unmap();
//       }
//       //throw "pause after m2l";
//     }



//   }
//   async l2l(numLevel: number) {
//     //console.log("l2l")
//     const core = this.core;
//     const boxCount = core.tree.levelBoxCounts[numLevel];
//     const commandLength = 2;
//     let command = new Int32Array(commandLength * boxCount);

//     let nbc = -1, neo = new Array(core.tree.numBoxIndexFull);
//     let numBoxIndexOld = 0;

//     for (let i = 0; i < core.tree.numBoxIndexFull; i++) { neo[i] = -1; }
//     for (let ii = 0; ii < boxCount; ii++) {
//       let ib = ii + core.tree.levelOffset[numLevel - 1];
//       if (nbc != Math.floor(core.tree.boxIndexFull[ib] / 8)) {
//         nbc = Math.floor(core.tree.boxIndexFull[ib] / 8);
//         neo[nbc] = numBoxIndexOld;
//         numBoxIndexOld++;
//       }
//     }
//     //console.log(neo);


//     numBoxIndexOld = boxCount;
//     if (numBoxIndexOld < 8) { numBoxIndexOld = 8; }
//     // for (let ii = 0; ii < numBoxIndexOld; ii++) {
//     //   for (let i = 0; i < core.numCoefficients; i++) {
//     //     //LnmOld[ii][i] = Lnm[ii][i];
//     //   }
//     // }

//     for (let ii = 0; ii < boxCount; ii++) {
//       let ib = ii + core.tree.levelOffset[numLevel - 1];
//       let nfip = Math.floor(core.tree.boxIndexFull[ib] / 8);
//       let nfic = core.tree.boxIndexFull[ib] % 8;
//       let boxIndex3D = GetIndex3D(nfic);
//       boxIndex3D.x = boxIndex3D.x * 2 + 2;
//       boxIndex3D.y = boxIndex3D.y * 2 + 2;
//       boxIndex3D.z = boxIndex3D.z * 2 + 2;
//       let je = GetIndexFrom3D(boxIndex3D);
//       ib = neo[nfip];//source
//       //console.log(`${ib}=>${ii}`)
//       command[ii * commandLength] = ib;
//       command[ii * commandLength + 1] = je + 1;
//     }
//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);
//     const boxSize = core.tree.rootBoxSize / (1 << numLevel);
//     const uniformBuffer = new Float32Array(1);
//     uniformBuffer[0] = boxSize;
//     this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);

//     const preFunc = (commandEncoder: GPUCommandEncoder) => {
//       commandEncoder.copyBufferToBuffer(this.lnmBufferGPU, 0, this.lnmOldBufferGPU, 0, this.lnmBufferGPU.size);
//     }

//     this.RunCompute("l2l",
//       [this.uniformBufferGPU, this.lnmBufferGPU, this.commandBufferGPU,
//       this.dnmBufferGPU, this.lnmOldBufferGPU],
//       boxCount, this.lnmBufferGPU, [], preFunc
//     );

//     if (this.debug) {
//       //console.log("debug l2l")
//       {
//         await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//         const handle = this.readBufferGPU.getMappedRange();
//         let tempReadBuffer = new Float32Array(handle);
//         {
//           this.Lnm = new Array(core.tree.numBoxIndexLeaf);
//           for (let i = 0; i < core.tree.numBoxIndexTotal; i++) {
//             const LnmVec = new Float32Array(this.core.numCoefficients * 2);
//             for (let j = 0; j < this.core.numCoefficients * 2; j++) {
//               LnmVec[j] = tempReadBuffer[i * this.core.numCoefficients * 2 + j];
//             }
//             this.Lnm[i] = LnmVec;
//           }
//         }
//         //console.log("Lnm:")
//         //console.log(this.Lnm);
//         this.readBufferGPU.unmap();
//       }
//       //throw "pause after l2l";
//     }

//   }
//   async l2p() {
//     const core = this.core;
//     const commandLength = 4;//
//     const boxCount = core.tree.levelBoxCounts[core.tree.maxLevel - 1];
//     let command = new Int32Array(commandLength * boxCount);
//     let boxSize = core.tree.rootBoxSize / (1 << core.tree.maxLevel);
//     const threadsPerGroup = 256;
//     // loop foreach box set group
//     let groupCount = 0;
//     for (let ii = 0; ii < boxCount; ii++) {
//       let nParticle = core.tree.nodeEndOffset[ii] - core.tree.nodeStartOffset[ii] + 1;
//       let nGroup = (nParticle + threadsPerGroup) / threadsPerGroup;
//       nGroup = Math.floor(nGroup);
//       for (let n = 0; n < nGroup; n++) {
//         command[groupCount * commandLength + 0] = ii;
//         command[groupCount * commandLength + 1] = core.tree.nodeStartOffset[ii] + n * threadsPerGroup;
//         command[groupCount * commandLength + 2] = (n == nGroup - 1) ? threadsPerGroup : (nParticle - (nGroup - 1) * threadsPerGroup);
//         command[groupCount * commandLength + 2] = core.tree.boxIndexFull[ii];
//         groupCount++;
//       }

//     }
//     this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0);

//     await this.RunCompute("l2p",
//       [this.uniformBufferGPU, this.nodeBufferGPU, this.resultBufferGPU, this.commandBufferGPU,
//       this.factorialGPU, this.lnmBufferGPU,],
//       groupCount, this.resultBufferGPU, [this.resultBufferGPU]
//     );
//     await this.readBufferGPU.mapAsync(GPUMapMode.READ);
//     const handle = this.readBufferGPU.getMappedRange();
//     let tempAccelBuffer = new Float32Array(handle);
//     // console.log(cmdBuffer);
//     //console.log(tempAccelBuffer);
//     //console.log(this.accelBuffer);
//     for (let i = 0; i < this.particleCount; i++) {

//       this.accelBuffer[i * 3] += tempAccelBuffer[i * 3];
//       this.accelBuffer[i * 3 + 1] += tempAccelBuffer[i * 3 + 1];
//       this.accelBuffer[i * 3 + 2] += tempAccelBuffer[i * 3 + 2];
//     }
//     this.readBufferGPU.unmap();


//   }

//   async RunCompute(entryPoint: string, buffers: Array<GPUBuffer>, workgroupCount = 1, readBuffer: GPUBuffer = null, buffersToClear: Array<GPUBuffer> = null, preFunc: any = null) {
//     const shaderModule = this.shaders[entryPoint];
//     const computePipeline = this.device.createComputePipeline({
//       layout: 'auto', // infer from shader code.
//       compute: {
//         module: shaderModule,
//         entryPoint: entryPoint
//       }
//     });

//     const bindGroupLayout = computePipeline.getBindGroupLayout(0);

//     const entries = buffers.map((b, i) => {
//       return { binding: i, resource: { buffer: b } };
//     });
//     const bindGroup = this.device.createBindGroup({
//       layout: bindGroupLayout,
//       entries: entries
//     });
//     const commandEncoder = this.device.createCommandEncoder();
//     if (preFunc) {
//       preFunc(commandEncoder);
//     }
//     if (buffersToClear) {
//       for (const b of buffersToClear) {
//         commandEncoder.clearBuffer(b);
//       }
//     }

//     const computePassEncoder = commandEncoder.beginComputePass();
//     computePassEncoder.setPipeline(computePipeline);
//     computePassEncoder.setBindGroup(0, bindGroup);
//     computePassEncoder.dispatchWorkgroups(workgroupCount);
//     computePassEncoder.end();

//     if (readBuffer) {
//       commandEncoder.copyBufferToBuffer(readBuffer, 0, this.readBufferGPU, 0, readBuffer.size);
//     }
//     const gpuCommands = commandEncoder.finish();
//     this.device.queue.submit([gpuCommands]);
//     if (!readBuffer) {
//       await this.device.queue.onSubmittedWorkDone();
//     }
//   }

//   Release() {
//     this.uniformBufferGPU.destroy();
//     this.nodeBufferGPU.destroy();

//     this.uniformBufferGPU.destroy();

//     this.factorialGPU.destroy();
//     this.mnmBufferGPU.destroy();
//     this.lnmBufferGPU.destroy();
//     this.readBufferGPU.destroy();

//     this.commandBufferGPU.destroy();
//   }
// }


// function CompareNumber(a: number, b: number, delta = 0.002) {
//   return Math.abs(a - b) < delta
// }