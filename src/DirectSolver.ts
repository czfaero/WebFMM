import { INBodySolver } from './INBodySolver';
import wgsl from './shaders/Direct.wgsl';
import { TreeBuilder } from './TreeBuilder';
const SIZEOF_32 = 4;
const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;
export class DirectSolver implements INBodySolver {
    tree: TreeBuilder;
    adapter: GPUAdapter;
    device: GPUDevice;
    shaderModule: GPUShaderModule;
    particleBufferGPU: GPUBuffer;
    accelBufferGPU: GPUBuffer;
    readBufferGPU: GPUBuffer;
    uniformBufferGPU: GPUBuffer;
    nodeBuffer: Float32Array
    nodeCount: number;
    dataReady: boolean;
    accelBuffer: Float32Array;
    useWgpu: boolean;

    debug_info: any;


    isDataReady() { return this.dataReady; }
    getAccelBuffer() { return this.accelBuffer; }

    constructor(tree: TreeBuilder, useWgpu = true) {
        this.nodeBuffer = tree.nodeBuffer;
        this.tree = tree;
        this.nodeCount = this.nodeBuffer.length / 4;
        this.useWgpu = useWgpu;
    }


    async main() {
        const time = performance.now();
        if (this.useWgpu) {
            await this.Init_wgpu();
            await this.Calc_wgpu();
            this.Release_wgpu();
        } else {
            this.Calc();
        }
        this.dataReady = true;
        this.debug_info = [{ time: performance.now() - time }];
    }



    getNode(i: number) {
        return {
            x: this.nodeBuffer[i * 4],
            y: this.nodeBuffer[i * 4 + 1],
            z: this.nodeBuffer[i * 4 + 2],
            w: this.nodeBuffer[i * 4 + 3]
        }
    }

    async Init_wgpu() {
        this.adapter = await navigator.gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();

        this.shaderModule = this.device.createShaderModule({
            code: wgsl,
        })
        this.uniformBufferGPU = this.device.createBuffer({
            size: 2 * SIZEOF_32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.particleBufferGPU = this.device.createBuffer({
            size: this.nodeBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.accelBufferGPU = this.device.createBuffer({
            size: this.particleBufferGPU.size / 4 * 3,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.readBufferGPU = this.device.createBuffer({
            size: this.accelBufferGPU.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(this.particleBufferGPU, 0, this.nodeBuffer);
    }
    Release_wgpu() {
        this.device.destroy();
    }
    async Calc_wgpu() {

        const threadPerGroup = 128;

        let workgroupCount = Math.ceil(this.nodeCount / threadPerGroup);
        if (workgroupCount > 10) {
            //debugger;
        }

        const uniformBuffer = new Float32Array(1);
        uniformBuffer[0] = 0;

        const uniformBuffer2 = new Uint32Array(1);
        uniformBuffer2[0] = this.nodeCount;

        this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
        this.device.queue.writeBuffer(this.uniformBufferGPU, uniformBuffer.byteLength, uniformBuffer2);



        const computePipeline = this.device.createComputePipeline({
            layout: 'auto', // infer from shader code.
            compute: {
                module: this.shaderModule,
                entryPoint: "direct"
            }
        });

        const bindGroupLayout = computePipeline.getBindGroupLayout(0);

        const entries = [this.uniformBufferGPU, this.particleBufferGPU, this.accelBufferGPU].map((b, i) => {
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

        commandEncoder.copyBufferToBuffer(this.accelBufferGPU, 0, this.readBufferGPU, 0, this.readBufferGPU.size);

        const gpuCommands = commandEncoder.finish();
        this.device.queue.submit([gpuCommands]);
        await this.readBufferGPU.mapAsync(GPUMapMode.READ);
        const handle = this.readBufferGPU.getMappedRange();
        const temp = new Float32Array(handle);
        this.accelBuffer = new Float32Array(temp);


        this.readBufferGPU.unmap();

    }

    Calc() {
        const accelBuffer = new Float32Array(this.nodeCount * 3);
        for (let i = 0; i < this.nodeCount; i++) {
            let ax = 0, ay = 0, az = 0;
            const dst = this.getNode(i);
            for (let j = 0; j < this.nodeCount; j++) {
                if (i == j) continue;
                const src = this.getNode(j);
                let dx = dst.x - src.x,
                    dy = dst.y - src.y,
                    dz = dst.z - src.z;
                const invDist = 1.0 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps);
                const invDistCube = invDist * invDist * invDist;
                const s = dst.w * src.w * invDistCube;
                ax += dx * s;
                ay += dy * s;
                az += dz * s;
            }
            accelBuffer[i * 3] = ax;
            accelBuffer[i * 3 + 1] = ay;
            accelBuffer[i * 3 + 2] = az;
        }
        this.accelBuffer = accelBuffer;
    }
}