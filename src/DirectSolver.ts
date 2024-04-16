import wgsl from './shaders/Direct.wgsl';
import { TreeBuilder } from './TreeBuilder';
const SIZEOF_32 = 4;
const eps = 1e-6;
const inv4PI = 0.25 / Math.PI;
export class DirectSolver {
    tree:TreeBuilder;
    adapter: GPUAdapter;
    device: GPUDevice;
    shaderModule: GPUShaderModule;
    particleBufferGPU: GPUBuffer;
    accelBufferGPU: GPUBuffer;
    readBufferGPU: GPUBuffer;
    uniformBufferGPU: GPUBuffer;
    nodeBuffer: Float32Array
    particleCount: number;
    dataReady: boolean;
    accelBuffer: Float32Array;
    isDataReady() { return this.dataReady; }
    getAccelBuffer() { return this.accelBuffer; }

    constructor(tree: TreeBuilder) {
        this.nodeBuffer = tree.nodeBuffer;
        this.tree=tree;
    }

    async Init() {

        this.particleCount = this.nodeBuffer.length / 4
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

    async main() {
        await this.Init();
        const threadPerGroup = 128;

        let workgroupCount = Math.ceil(this.particleCount / threadPerGroup);

        const uniformBuffer = new Float32Array(1);
        uniformBuffer[0] = 0;

        const uniformBuffer2 = new Uint32Array(1);
        uniformBuffer2[0] = this.particleCount;

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
        this.dataReady = true;
        this.Release();
    }


    Release() {
        this.device.destroy();
    }
    getParticle(i: number) {
        return {
            x: this.nodeBuffer[i * 4],
            y: this.nodeBuffer[i * 4 + 1],
            z: this.nodeBuffer[i * 4 + 2],
            w: this.nodeBuffer[i * 4 + 3]
        }
    }

    direct() {
        const accelBuffer = new Float32Array(this.particleCount * 3);
        for (let i = 0; i < this.particleCount; i++) {
            let ax = 0, ay = 0, az = 0;
            const p1 = this.getParticle(i);
            for (let j = 0; j < this.particleCount; j++) {
                if (i == j) continue;
                const p2 = this.getParticle(j);
                let dx = p1.x - p2.x,
                    dy = p1.y - p2.y,
                    dz = p1.z - p2.z;
                const invDist = 1.0 / Math.sqrt(dx * dx + dy * dy + dz * dz + eps);
                const invDistCube = p2.w * invDist * invDist * invDist;
                ax -= dx * invDistCube;
                ay -= dy * invDistCube;
                az -= dz * invDistCube;
            }
            accelBuffer[i * 3] = inv4PI * ax;
            accelBuffer[i * 3 + 1] = inv4PI * ay;
            accelBuffer[i * 3 + 2] = inv4PI * az;
        }
    }
}