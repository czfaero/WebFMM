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
  resultBufferGPU: GPUBuffer;
  commandBufferGPU: GPUBuffer;
  readBufferGPU: GPUBuffer;
  maxThreadCount: number;
  maxWorkgroupCount: number;
  accelBuffer: Float32Array;
  uniformBufferSize: number;
  uniformBuffer: Uint32Array;
  uniformBufferGPU: GPUBuffer;
  particleOffsetGPU: GPUBuffer;

  debug_info: any;

  async Init(particleBuffer: Float32Array) {
    this.particleCount = particleBuffer.length / 4;
    this.accelBuffer = new Float32Array(this.particleCount * 3);
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    // to-do: check limit
    console.log(this.adapter);
    this.maxThreadCount = 256;
    this.maxWorkgroupCount = 256;
    // this.cmdBufferLength = this.maxThreadCount * this.maxWorkgroupCount * 2;// to-do: set a good value
    // this.cmdBufferSize = this.cmdBufferLength * SIZEOF_32;
    this.particleBufferGPU = this.device.createBuffer({
      size: particleBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.uniformBuffer = new Uint32Array(1);
    this.uniformBufferSize = this.uniformBuffer.byteLength;

    this.uniformBufferGPU = this.device.createBuffer({
      size: this.uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(this.particleBufferGPU, 0, particleBuffer);

    if (this.debug) {
      this.debug_info = {};
      this.debug_info["particleBufferGPU"] = this.particleBufferGPU.size;
      this.debug_info["uniformBufferGPU"] = this.uniformBufferGPU.size;
    }
  }
  async p2p(numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) {
    if (this.debug) {
      this.debug_p2p_call_count = 0;
      this.debug_info["events"] = [];
      this.debug_info["events"].push({ time: performance.now(), tag: "start" });
    }

    const particleOffsetBuffer = new Uint32Array(numBoxIndex * 2);
    for (let i = 0; i < numBoxIndex; i++) {
      particleOffsetBuffer[i * 2] = particleOffset[0][i];
      particleOffsetBuffer[i * 2 + 1] = particleOffset[1][i];
    }
    this.particleOffsetGPU = this.device.createBuffer({
      size: particleOffsetBuffer.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.device.queue.writeBuffer(this.particleOffsetGPU, 0, particleOffsetBuffer);

    let commandCount = 0;
    const commandSize = 2;
    const maxCommandCount = this.maxThreadCount * this.maxWorkgroupCount;
    if (maxCommandCount > this.maxThreadCount * this.maxWorkgroupCount) {
      throw `Thread Count (${maxCommandCount}) > MaxThread(${this.maxThreadCount * this.maxWorkgroupCount})`;
    }
    const command = new Uint32Array(maxCommandCount * commandSize);
    this.commandBufferGPU = this.device.createBuffer({
      size: command.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    this.resultBufferGPU = this.device.createBuffer({
      size: maxCommandCount * 3 * SIZEOF_32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    this.readBufferGPU = this.device.createBuffer({
      size: this.resultBufferGPU.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    if (this.debug) {
      this.debug_info["commandBufferGPU"] = this.commandBufferGPU.size;
      this.debug_info["particleOffsetGPU"] = this.particleOffsetGPU.size;
      this.debug_info["resultBufferGPU"] = this.resultBufferGPU.size;
      this.debug_info["readBufferGPU"] = this.readBufferGPU.size;
    }

    for (let ii = 0; ii < numBoxIndex; ii++) {
      for (let i = particleOffset[0][ii]; i <= particleOffset[1][ii]; i++) {
        for (let ij = 0; ij < numInteraction[ii]; ij++) {
          const jj = interactionList[ii][ij];
          command[commandCount * 2] = i;
          command[commandCount * 2 + 1] = jj;
          commandCount++;
          if (commandCount == maxCommandCount) {
            this.device.queue.writeBuffer(this.commandBufferGPU, 0, command);
            await this.p2p_ApplyAccel(command, commandCount);
            commandCount = 0;
            command.fill(0);
          }
        }
      }
    }

    this.device.queue.writeBuffer(this.commandBufferGPU, 0, command, 0, commandCount * 2);
    await this.p2p_ApplyAccel(command, commandCount);
    this.debug_info["events"].push({ time: performance.now(), tag: "end" });

    if (this.debug) {
      // await this.readBufferGPU.mapAsync(GPUMapMode.READ);
      // const handle = this.readBufferGPU.getMappedRange();
      // this.accelBuffer = new Float32Array(handle);
      // // this.readBufferGPU.unmap();
      console.log(`debug_p2p_call_count: ${this.debug_p2p_call_count}`);
      const startTime = this.debug_info["events"][0].time;
      this.debug_info["events"].forEach(o => o.time -= startTime);
      //console.log(JSON.stringify(this.debug_info))
      console.log(this.debug_info);
    }
  }

  debug_p2p_call_count: number;
  async p2p_ApplyAccel(cmdBuffer: Uint32Array, commandCount: number) {
    if (this.debug) {
      this.debug_info["events"].push({ time: performance.now(), tag: "prepare submit", i: this.debug_p2p_call_count, thread_count: commandCount });
      this.debug_p2p_call_count++;
    }
    this.uniformBuffer.set([commandCount]);
    this.device.queue.writeBuffer(
      this.uniformBufferGPU,
      0,
      this.uniformBuffer
    );
    //console.log(cmdBuffer); throw "pause";
    await this.RunCompute("p2p",
      [this.uniformBufferGPU, this.particleBufferGPU, this.resultBufferGPU, this.commandBufferGPU, this.particleOffsetGPU],
      this.maxWorkgroupCount, true
    );
    //console.log("applyaccel");
    await this.readBufferGPU.mapAsync(GPUMapMode.READ);
    const handle = this.readBufferGPU.getMappedRange();
    let tempAccelBuffer = new Float32Array(handle);
    // console.log(cmdBuffer);
    // console.log(tempAccelBuffer);
    for (let i = 0; i < commandCount; i++) {
      let index = cmdBuffer[i * 2];
      this.accelBuffer[index * 3] += tempAccelBuffer[i * 3];
      this.accelBuffer[index * 3 + 1] += tempAccelBuffer[i * 3 + 1];
      this.accelBuffer[index * 3 + 2] += tempAccelBuffer[i * 3 + 2];
    }
    this.readBufferGPU.unmap();
  }

  Mnm: Array<Float32Array>;
  async p2m(numBoxIndex: number, particleOffset: any) { }

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
      commandEncoder.copyBufferToBuffer(this.resultBufferGPU, 0, this.readBufferGPU, 0, this.readBufferGPU.size);
    }
    const gpuCommands = commandEncoder.finish();
    this.device.queue.submit([gpuCommands]);
    if (!readBuffer) {
      await this.device.queue.onSubmittedWorkDone();
    }
  }
}