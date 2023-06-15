import wgsl from '../shaders/FMM.wgsl';

import { IKernel } from './kernel';

const SIZEOF_32 = 4;

export class KernelWgpu implements IKernel {
  debug: boolean;
  particleCount: number;
  constructor() {
    this.debug = false;
  }
  adapter: GPUAdapter;
  device: GPUDevice;
  particleBufferGPU: GPUBuffer;
  accelBufferGPU: GPUBuffer;
  cmdBufferGPU: GPUBuffer;
  readBufferGPU: GPUBuffer;
  maxThreadCount: number;
  maxWorkgroupCount: number;
  accelBuffer: Float32Array;
  accelBufferSize: number;
  uniformBufferSize: number;
  uniformBuffer: Uint32Array;
  uniformBufferGPU: GPUBuffer;
  async Init(particleBuffer: Float32Array) {
    this.particleCount = particleBuffer.length / 4;
    this.accelBuffer = new Float32Array(this.particleCount * 3);
    this.accelBufferSize = this.accelBuffer.byteLength;
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    // to-do: check limit
    console.log(this.adapter);
    this.maxThreadCount = 256;
    this.maxWorkgroupCount = 128;
    // this.cmdBufferLength = this.maxThreadCount * this.maxWorkgroupCount * 2;// to-do: set a good value
    // this.cmdBufferSize = this.cmdBufferLength * SIZEOF_32;
    this.particleBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.accelBufferGPU = this.device.createBuffer({
      size: this.accelBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    this.readBufferGPU = this.device.createBuffer({
      size: this.accelBufferSize,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.uniformBuffer = new Uint32Array(1);
    this.uniformBufferSize = this.uniformBuffer.byteLength;

    this.uniformBufferGPU = this.device.createBuffer({
      size: this.uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(this.particleBufferGPU, 0, particleBuffer);
  }
  async p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) {
    this.debug_p2p_call_count = 0;
    let cmdCount = 0;
    const maxCmdCount = this.accelBuffer.length / 3;
    const cmd = new Uint32Array(maxCmdCount * 2);
    this.cmdBufferGPU = this.device.createBuffer({
      size: cmd.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    for (let ii = 0; ii < numBoxIndex; ii++) {
      for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
        for (let ij = 0; ij < numInteraction[ii]; ij++) {
          const jj = interactionList[ii][ij];
          for (let j = particleOffset[0][jj]; j <= particleOffset[1][jj]; j++) {
            //calc
            cmd[cmdCount * 2] = i;
            cmd[cmdCount * 2 + 1] = j;
            cmdCount++;
            if (cmdCount == maxCmdCount) {
              this.device.queue.writeBuffer(this.cmdBufferGPU, 0, cmd);
              await this.p2p_ApplyAccel(cmd, cmdCount);
              cmdCount = 0;
              cmd.fill(0);
            }
          }
        }
      }
    }

    this.device.queue.writeBuffer(this.cmdBufferGPU, 0, cmd, 0, cmdCount * 2);
    await this.p2p_ApplyAccel(cmd, cmdCount);
    if (this.debug) {
      // await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      // const handle = this.readBufferGPU.getMappedRange();
      // this.accelBuffer = new Float32Array(handle);
      // // this.readBufferGPU.unmap();
      console.log(`debug_p2p_call_count: ${this.debug_p2p_call_count}`);
    }
    console.log(`debug_p2p_call_count: ${this.debug_p2p_call_count}`);
  }

  debug_p2p_call_count: number;
  async p2p_ApplyAccel(cmdBuffer: Uint32Array, length: number) {
    this.debug_p2p_call_count++;
    this.uniformBuffer.set([length]);
    this.device.queue.writeBuffer(
      this.uniformBufferGPU,
      0,
      this.uniformBuffer
    );
    await this.RunCompute("p2p",
      [this.uniformBufferGPU, this.particleBufferGPU, this.accelBufferGPU, this.cmdBufferGPU],
      this.maxWorkgroupCount
    );
    //console.log("applyaccel");
    await this.readBufferGPU.mapAsync(GPUMapMode.READ);
    const handle = this.readBufferGPU.getMappedRange();
    let tempAccelBuffer = new Float32Array(handle);
    // console.log(cmdBuffer);
    // console.log(tempAccelBuffer);
    for (let i = 0; i < length; i++) {
      let index = cmdBuffer[i * 2];
      this.accelBuffer[index * 3] += tempAccelBuffer[i * 3];
      this.accelBuffer[index * 3 + 1] += tempAccelBuffer[i * 3 + 1];
      this.accelBuffer[index * 3 + 2] += tempAccelBuffer[i * 3 + 2];
    }
    this.readBufferGPU.unmap();
  }



  async RunCompute(entryPoint: string, buffers: Array<GPUBuffer>, workgroupCount = 1, readBuffer = true) {
    const shaderModule = this.device.createShaderModule({
      code: wgsl,
    })
    const computePipeline = this.device.createComputePipeline({
      layout: 'auto', // infer from shader code.
      compute: {
        module: shaderModule,
        entryPoint: entryPoint
      }
    });

    const bindGroupLayout = computePipeline.getBindGroupLayout(0);

    const entries = buffers.map((b, i) => {
      return { binding: i, resource: { buffer: b } };
    });
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: entries
    });
    const commandEncoder = this.device.createCommandEncoder();
    const computePassEncoder = commandEncoder.beginComputePass();
    computePassEncoder.setPipeline(computePipeline);
    computePassEncoder.setBindGroup(0, bindGroup);
    computePassEncoder.dispatchWorkgroups(workgroupCount);
    computePassEncoder.end();

    if (readBuffer) {
      commandEncoder.copyBufferToBuffer(this.accelBufferGPU, 0, this.readBufferGPU, 0, this.particleCount * 4 * 3);
    }
    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
    if (!readBuffer) {
      await this.device.queue.onSubmittedWorkDone();
    }
  }
}