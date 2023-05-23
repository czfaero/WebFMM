import wgsl from '../shaders/FMM.wgsl';

import { IKernel } from './kernel';

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
  maxGPUThread: number;
  cmdBufferSize: number;

  accelBuffer: Float32Array;
  async Init(particleBuffer: Float32Array) {
    this.particleCount = particleBuffer.length / 4;
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    // to-do: check limit
    console.log(this.adapter);
    this.maxGPUThread = 256;
    this.cmdBufferSize = 8000000;// to-do: set a good value
    this.particleBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.accelBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength / 4 * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    this.cmdBufferGPU = this.device.createBuffer({
      size: this.cmdBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.readBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength / 4 * 3,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.particleBufferGPU, 0, particleBuffer);
  }
  async p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) {
    const cmd = new Uint32Array(this.cmdBufferSize / 4);
    let cmdSize = 0;
    const threadOffset = 2;
    const cmdOffset = threadOffset + 2 * this.maxGPUThread;
    let cmdCounts = new Array();
    const maxCmdCount = (cmd.length - cmdOffset) / 2;
    for (let ii = 0; ii < numBoxIndex; ii++) {
      for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
        let cmdCount = 0;
        for (let ij = 0; ij < numInteraction[ii]; ij++) {
          const jj = interactionList[ii][ij];
          for (let j = particleOffset[0][jj]; j <= particleOffset[1][jj]; j++) {
            //calc
            cmd[cmdOffset + cmdSize * 2] = i;
            cmd[cmdOffset + cmdSize * 2 + 1] = j;
            cmdSize++;
            cmdCount++;
            if (cmdSize == maxCmdCount) {
              cmdCounts.push(cmdCount);
              cmd[0] = cmdSize;
              this.CmdThread(cmd, cmdSize, cmdCounts, threadOffset, cmdOffset);
              this.device.queue.writeBuffer(this.cmdBufferGPU, 0, cmd);
              await this.RunCompute("p2p", [this.particleBufferGPU, this.accelBufferGPU, this.cmdBufferGPU]);
              cmdSize = 0;
              cmdCount = 0;
              cmdCounts = new Array();
              cmd.fill(0);
            }
          }
        }
        cmdCounts.push(cmdCount);
      }
    }

    cmd[0] = cmdSize;
    this.CmdThread(cmd, cmdSize, cmdCounts, threadOffset, cmdOffset);
    this.device.queue.writeBuffer(this.cmdBufferGPU, 0, cmd, 0, cmdOffset + cmdSize * 2);
    await this.RunCompute("p2p", [this.particleBufferGPU, this.accelBufferGPU, this.cmdBufferGPU]);


    if (this.debug) {
      await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      const handle = this.readBufferGPU.getMappedRange();
      this.accelBuffer = new Float32Array(handle);
      // this.readBufferGPU.unmap();
    }
  }

  CmdThread(cmd: Uint32Array, cmdSize: number, cmdCounts: Array<number>, threadOffset: number, cmdOffset) {

    const targetCmdPerThread = cmdSize / this.maxGPUThread;
    let thread = 0;
    let currentCount = 0;
    let totalCount = 0;
    for (const cmdCount of cmdCounts) {
      let test = cmd[cmdOffset + totalCount * 2];
      for (let i = 0; i < cmdCount; i++) {
        let v = cmd[cmdOffset + totalCount * 2 + i * 2];
        if (test != v) { throw `[0]${test} [${i}]${v}`; }
      }
      currentCount += cmdCount;
      if (currentCount > targetCmdPerThread) {
        cmd[threadOffset + thread * 2] = cmdOffset + totalCount * 2;
        cmd[threadOffset + thread * 2 + 1] = currentCount;
        totalCount += currentCount;
        currentCount = 0;
        thread++;
      }

    }
    console.log(targetCmdPerThread);
    console.log(cmdCounts);
    console.log(cmd)
    console.log(cmdSize);
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
    if (this.debug) {
      commandEncoder.copyBufferToBuffer(this.accelBufferGPU, 0, this.readBufferGPU, 0, this.particleCount * 4 * 3);
    }

    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
    await this.device.queue.onSubmittedWorkDone();
  }
}