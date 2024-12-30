import wgsl_p2p from './shaders/FMM_p2p.wgsl';
import wgsl_p2m from './shaders/FMM_p2m.wgsl';
import wgsl_m2m from './shaders/FMM_m2m.wgsl';
import wgsl_m2l from './shaders/FMM_m2l.wgsl';
import wgsl_l2l from './shaders/FMM_l2l.wgsl';
import wgsl_l2p from './shaders/FMM_l2p.wgsl';

import { FMMSolver } from "../FMMSolver";
import { IFMMKernel } from "../IFMMKernel";

const SIZEOF_32 = 4;
const maxM2LInteraction = 189;

// like enum
const u32 = "u32", f32 = "f32";

const uniforms_p2p = {
    nodeCount: u32,
    boxCount: u32,
};

const uniform_structs =
    { uniforms_p2p };

export class FMMKernel_wgsl implements IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    debug_info: any;
    accelBuffer: Float32Array;
    /**
     * CPU buffer only for debug.
     * index same as tree.boxIndexFull
     * Access: levelOffset[numLevel]+i
     */
    Mnm: Float32Array;
    Lnm: Float32Array;


    adapter: GPUAdapter;
    device: GPUDevice;
    shaders: any;

    nodeBufferGPU: GPUBuffer;
    /** [...tree.nodeStartOffset, ...tree.nodeEndOffset] */
    nodeOffsetBufferGPU: GPUBuffer;

    accelBufferGPU: GPUBuffer;
    /** Need this due to WebGPU restriction */
    readBufferGPU: GPUBuffer;

    uniformBufferGPU: GPUBuffer;
    uniformBufferMaxMember: number;

    mnmBufferGPU: GPUBuffer;
    lnmBufferGPU: GPUBuffer;

    /** [count, ...Array(maxM2L), ...] */
    interactionListBuffer: Uint32Array;
    /** [count, ...Array(maxM2L), ...] */
    interactionListGPU: GPUBuffer;

    i2nmBufferGPU: GPUBuffer;
    factorialGPU: GPUBuffer;

    maxThreadPerGroup: number;

    constructor(core: FMMSolver) {
        this.core = core;
        this.debug_info = [];
        this.debug = true;
    }
    async Init() {
        const core = this.core;
        const tree = core.tree;
        this.accelBuffer = new Float32Array(tree.nodeCount * 3);
        this.adapter = await navigator.gpu.requestAdapter();
        this.device = await this.adapter.requestDevice();
        this.maxThreadPerGroup = this.device.limits.maxComputeInvocationsPerWorkgroup;
        this.InitShaders();

        // GPU buffers
        this.nodeBufferGPU = this.device.createBuffer({
            size: tree.nodeBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.nodeBufferGPU, 0, tree.nodeBuffer);

        this.nodeOffsetBufferGPU = this.device.createBuffer({
            size: tree.nodeStartOffset.byteLength * 2,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.nodeOffsetBufferGPU, 0, tree.nodeStartOffset);
        this.device.queue.writeBuffer(this.nodeOffsetBufferGPU, tree.nodeStartOffset.byteLength, tree.nodeEndOffset);

        this.uniformBufferMaxMember = Math.max(...Object.keys(uniform_structs).map(x => Object.keys(uniform_structs[x]).length))
        this.uniformBufferGPU = this.device.createBuffer({
            size: this.uniformBufferMaxMember * SIZEOF_32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.accelBufferGPU = this.device.createBuffer({
            size: tree.nodeCount * 3 * SIZEOF_32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        //If a buffer usage contains BufferUsage::MapRead the only other allowed usage is BufferUsage::CopyDst
        this.readBufferGPU = this.device.createBuffer({
            size: this.accelBufferGPU.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.mnmBufferGPU = this.device.createBuffer({
            size: core.tree.numBoxIndexTotal * core.MnmSize * 2 * SIZEOF_32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.lnmBufferGPU = this.device.createBuffer({
            size: core.tree.numBoxIndexTotal * core.MnmSize * 2 * SIZEOF_32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        this.interactionListBuffer = new Uint32Array((maxM2LInteraction + 1) * tree.numBoxIndexLeaf);
        this.interactionListGPU = this.device.createBuffer({
            size: this.interactionListBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        await this.precalc();
    }

    async precalc() {
        const core = this.core;

        let i2nm = new Int32Array(core.MnmSize * 4);
        this.i2nmBufferGPU = this.device.createBuffer({
            size: i2nm.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        for (let n = 0; n < core.numExpansions; n++) {
            for (let m = 0; m <= n; m++) {
                let i = n * (n + 1) / 2 + m;
                i2nm[i * 4 + 0] = n;
                i2nm[i * 4 + 1] = m;
            }
        }
        for (let n = 0; n < core.numExpansions; n++) {
            for (let m = -n; m <= n; m++) {
                let i = n * n + n + m;
                i2nm[i * 4 + 2] = n;
                i2nm[i * 4 + 3] = m;
            }
        }
        this.device.queue.writeBuffer(this.i2nmBufferGPU, 0, i2nm);

        let factorial = new Float32Array(2 * core.numExpansions);
        for (let m = 0, fact = 1.0; m < factorial.length; m++) {
            factorial[m] = fact;
            fact = fact * (m + 1);
        }
        this.factorialGPU = this.device.createBuffer({
            size: factorial.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.factorialGPU, 0, factorial);
    }

    setInteractionList() {
        const core = this.core;

        // GPUBuffer size: (maxM2LInteraction + 1) * tree.numBoxIndexLeaf * SIZEOF_32
        this.interactionListBuffer.fill(0);
        for (let i = 0; i < core.interactionCounts.length; i++) {
            const offset = (maxM2LInteraction + 1) * i;
            this.interactionListBuffer[offset] = core.interactionCounts[i];
            this.interactionListBuffer.set(core.interactionList[i], offset + 1);
        }
        console.log(this.interactionListBuffer)

        this.device.queue.writeBuffer(this.interactionListGPU, 0, this.interactionListBuffer);
    }
    async p2p() {
        const waitDone = this.debug;
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; // tree.numBoxIndexLeaf
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        this.setInteractionList();
        let workgroupCount = Math.ceil(tree.nodeCount / this.maxThreadPerGroup);
        this.UniformTransfer(
            {
                nodeCount: tree.nodeCount,
                boxCount: boxCount,
            }
            , uniforms_p2p);
        await this.RunCompute("p2p",
            [this.uniformBufferGPU, this.nodeBufferGPU, this.accelBufferGPU, this.nodeOffsetBufferGPU, this.interactionListGPU],
            workgroupCount,
            waitDone,
            this.accelBufferGPU, // debug  to Read
        );
        this.debug_info.push({ step: "P2P", time: performance.now() - time });

        await this.GetReadBufferContent(this.accelBuffer);//dbug
    }

    async p2m() {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {

        }
        this.debug_info.push({ step: "P2M", time: performance.now() - time });
    }
    async m2m(numLevel: number) {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        const src_mask = tree.boxIndexMaskBuffers[numLevel + 1];
        for (let i = 0; i < boxCount; i++) {


        }
        this.debug_info.push({ step: `M2M@${numLevel + 1}->${numLevel}`, time: performance.now() - time });
    }
    async m2l(numLevel: number) {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        for (let i = 0; i < boxCount; i++) {

        }
        this.debug_info.push({
            step: `M2L@${numLevel}`,
            time: performance.now() - time
        });
    }
    async l2l(numLevel: number) {
        const time = performance.now();

        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        const dst_mask = tree.boxIndexMaskBuffers[numLevel + 1];
        for (let i = 0; i < boxCount; i++) {

        }
        this.debug_info.push({ step: `L2L@${numLevel}->${numLevel + 1}`, time: performance.now() - time });

    }
    async l2p() {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
        }
        this.debug_info.push({ step: `L2P`, time: performance.now() - time });

        await this.GetReadBufferContent(this.accelBuffer);
    }

    Release() { }

    InitShaders() {
        const contants = {
            maxThreadPerGroup: { v: this.maxThreadPerGroup, t: u32 },
            PI: { v: Math.PI, t: f32 },
            eps: { v: 1e-6, t: f32 },
            maxM2LInteraction: { v: maxM2LInteraction, t: u32 },
        };

        const contants_wsgl = Object.keys(contants)
            .map(key => {
                const o = contants[key]
                return `const ${key} : ${o.t} = ${o.v};`
            }).join("\n");

        const includeSrc = {
            contants: contants_wsgl,
        };

        const uniform_names = Object.keys(uniform_structs);
        uniform_names.forEach(name => {
            const obj = uniform_structs[name];
            const keys = Object.keys(obj);
            const def = `struct Uniforms {\n${keys.map(k => `${k}:${obj[k]}`).join(",\n  ")}\n}\n`;
            const expand = keys.map(k => `let ${k} = uniforms.${k};`).join("\n");
            includeSrc[name + "_def"] = def;
            includeSrc[name + "_expand"] = expand;
        });



        this.shaders = {
            p2p: wgsl_p2p,
            p2m: wgsl_p2m,
            m2m: wgsl_m2m,
            m2l: wgsl_m2l,
            l2l: wgsl_l2l,
            l2p: wgsl_l2p,
        }
        Object.keys(this.shaders)
            .forEach(key => {
                let shader: string = this.shaders[key];
                shader = shader.replaceAll(/[ ]*#include[ ]+([a-zA-Z_][a-zA-Z_0-9]*)[ ]*;/g,
                    (_, name) => {
                        const src = includeSrc[name];
                        if (!src) { debugger; throw "src not exist"; }
                        return src
                    }
                );
                this.shaders[key] = shader;
            });
        const nameList = "p2p p2m m2m m2l l2l l2p".split(" ");
        debugger;
        for (const n of nameList) {
            this.shaders[n] = this.device.createShaderModule({
                code: this.shaders[n],
            })
        }
    }

    async RunCompute(
        entryPoint: string,
        buffers: Array<GPUBuffer>,
        workgroupCount = 1,
        waitDone = false,
        bufferToRead: GPUBuffer = null,
        buffersToClear: Array<GPUBuffer> = null,
        preFunc: any = null) {
        const shaderModule = this.shaders[entryPoint];
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
        if (preFunc) {
            preFunc(commandEncoder);
        }
        if (buffersToClear) {
            for (const b of buffersToClear) {
                commandEncoder.clearBuffer(b);
            }
        }

        const computePassEncoder = commandEncoder.beginComputePass();
        computePassEncoder.setPipeline(computePipeline);
        computePassEncoder.setBindGroup(0, bindGroup);
        computePassEncoder.dispatchWorkgroups(workgroupCount);
        computePassEncoder.end();

        if (bufferToRead) {
            commandEncoder.copyBufferToBuffer(bufferToRead, 0, this.readBufferGPU, 0, bufferToRead.size);
        }

        const gpuCommands = commandEncoder.finish();
        this.device.queue.submit([gpuCommands]);

        if (waitDone) {
            await this.device.queue.onSubmittedWorkDone();
        }
    }

    UniformTransfer(data: any, struct: any) {
        const keys = Object.keys(struct);
        const uniformBuffer = new ArrayBuffer(keys.length * SIZEOF_32);
        const view = new DataView(uniformBuffer);
        // https://www.w3.org/TR/WGSL/
        // "numeric values in host-shared buffers are stored in little-endian format."
        const littleEndian = true;
        keys.forEach((key, i) => {
            switch (struct[key]) {
                case "u32":
                    view.setUint32(i * SIZEOF_32, data[key], littleEndian);
                    break;
                case "f32":
                    view.setFloat32(i * SIZEOF_32, data[key], littleEndian);
                    break;
            }
        })
        this.device.queue.writeBuffer(this.uniformBufferGPU, 0, uniformBuffer);
    }

    async GetReadBufferContent(dst: Float32Array) {
        await this.readBufferGPU.mapAsync(GPUMapMode.READ);
        const handle = this.readBufferGPU.getMappedRange();
        let temp = new Float32Array(handle);
        dst.set(temp);
        this.readBufferGPU.unmap();
    }

}