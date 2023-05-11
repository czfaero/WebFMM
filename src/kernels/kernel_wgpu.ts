import wgsl from '../shaders/FMM.wgsl';

import { IKernel } from './kernel';


export class KernelWgpu implements IKernel {

  particleCount: number;
  constructor() {

  }
  adapter: GPUAdapter;
  device: GPUDevice;
  particleBufferGPU: GPUBuffer;
  accelBufferGPU: GPUBuffer;
  cmdBufferGPU: GPUBuffer;
  readBufferGPU: GPUBuffer;
  maxGPUThread: number;
  async Init(particleBuffer: Float32Array) {
    this.particleCount = particleBuffer.length / 4;
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    // to-do: check limit
    console.log(this.adapter);
    this.maxGPUThread = 256;
    this.particleBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.accelBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength / 4 * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    this.cmdBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength, // to-do: set a good value
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.readBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength / 4 * 3,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.particleBufferGPU, 0, particleBuffer);
  }
  p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any): void {
    const cmd = new Uint32Array(40000);
    let cmdSize = 0;
    for (let ii = 0; ii < numBoxIndex; ii++) {
      for (let ij = 0; ij < numInteraction[ii]; ij++) {
        const jj = interactionList[ii][ij];
        for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
          for (let j = particleOffset[0][jj]; j <= particleOffset[1][jj]; j++) {
            //calc
            cmd[2 + cmdSize * 2] = i;
            cmd[2 + cmdSize * 2 + 1] = j;
            cmdSize++;
          }
        }
      }
    }
    cmd[0] = cmdSize;
    this.device.queue.writeBuffer(this.cmdBufferGPU, 0, cmd);


    this.RunCompute("p2p", [this.particleBufferGPU, this.accelBufferGPU, this.cmdBufferGPU]);


  }
  async RunCompute(entryPoint: string, buffers: Array<GPUBuffer>) {
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

    console.log(bindGroupLayout);
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
    computePassEncoder.dispatchWorkgroups(256, 1);
    computePassEncoder.end();
    commandEncoder.copyBufferToBuffer(this.accelBufferGPU, 0, this.readBufferGPU, 0, this.particleCount * 4 * 3);

    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
    await this.device.queue.onSubmittedWorkDone();

    await this.readBufferGPU.mapAsync(GPUMapMode.READ);
    const handle = this.readBufferGPU.getMappedRange();
    const accels = new Float32Array(handle);
    console.log(accels);

    this.readBufferGPU.unmap();
  }
}