/// <reference types="@webgpu/types" />
import { mat4, quat, vec3 } from 'gl-matrix';
import wgsl from './shaders/NodeLinkRenderer.wgsl';
import { FPSController } from './compenents/FPSController';
import { NodeLinkSimulator } from './NodeLinkSimulator';


export class NodeLinkRenderer {
    simulator: NodeLinkSimulator;

    log_func: any;

    Log(str: string) {
        if (this.log_func) {
            this.log_func(str);
        }
    }

    destroyFlag: boolean;
    Destroy() {
        this.destroyFlag = true;
    }

    async init(canvasElement: HTMLCanvasElement) {
        const _ = this;
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter.requestDevice();

        const context = canvasElement.getContext('webgpu') as unknown as GPUCanvasContext;
        const simulator = _.simulator;
        await simulator.ResetData();
        _.destroyFlag = false;
        const devicePixelRatio = window.devicePixelRatio || 1;
        // const presentationSize = [
        //   canvasElement.clientWidth * devicePixelRatio,
        //   canvasElement.clientHeight * devicePixelRatio,
        // ];
        //Will cause: Attachment [TextureView] size does not match the size of the other attachments. 
        //            - While validating depthStencilAttachment.

        const presentationSize = [
            canvasElement.width,
            canvasElement.height,
        ];
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        // console.log(presentationFormat);
        context.configure({
            device,
            format: presentationFormat,
            alphaMode: 'opaque',
        });


        // For node
        const quadVertexData = new Float32Array([
            -1.0, -1.0, +1.0, -1.0, -1.0, +1.0,
            -1.0, +1.0, +1.0, -1.0, +1.0, +1.0,
        ]);
        const quadVertexBuffer = device.createBuffer({
            size: quadVertexData.byteLength,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(quadVertexBuffer.getMappedRange()).set(quadVertexData);
        quadVertexBuffer.unmap();

        // The node pos buffers for both computing and rendering
        const nodeBufferGPU = device.createBuffer({
            size: simulator.nodeBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(nodeBufferGPU, 0, simulator.nodeBuffer);



        // Link
        const linkBufferGPU = device.createBuffer({
            size: simulator.linkBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(linkBufferGPU, 0, simulator.linkBuffer);

        const nodeColorBufferGPU = device.createBuffer({
            size: simulator.nodeColorBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(nodeColorBufferGPU, 0, simulator.nodeColorBuffer);

        const shaderModule = device.createShaderModule({
            code: wgsl,
        })
        // Compute 

        // const computePipeline = device.createComputePipeline({
        //   layout: 'auto',
        //   compute: {
        //     module: shaderModule,
        //     entryPoint: "comp"
        //   }
        // });
        // const bindGroupLayout = computePipeline.getBindGroupLayout(0);
        // const bindGroup0 = device.createBindGroup({
        //   layout: bindGroupLayout,
        //   entries: [
        //     { binding: 0, resource: { buffer: posBuffer0 } },
        //     { binding: 1, resource: { buffer: posBuffer1 } },
        //   ]
        // });
        // const bindGroup1 = device.createBindGroup({
        //   layout: bindGroupLayout,
        //   entries: [
        //     { binding: 0, resource: { buffer: posBuffer1 } },
        //     { binding: 1, resource: { buffer: posBuffer0 } },
        //   ]
        // });

        // Render - nodes
        const nodePipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'main_node_vert',
                buffers: [
                    {
                        arrayStride: 2 * 4, // vec2 float
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // vertex positions
                                shaderLocation: 0, offset: 0, format: 'float32x2',
                            }
                        ],
                    } as GPUVertexBufferLayout,
                    {
                        // instanced particles buffer
                        arrayStride: 4 * 4,
                        stepMode: 'instance',
                        attributes: [
                            {
                                // instance position
                                shaderLocation: 1, offset: 0, format: 'float32x4',
                            },
                        ],
                    } as GPUVertexBufferLayout,
                    {
                        // instanced particles buffer
                        arrayStride: 3 * 4,
                        stepMode: 'instance',
                        attributes: [
                            {
                                // instance color
                                shaderLocation: 2, offset: 0, format: 'float32x3',
                            },
                        ],
                    } as GPUVertexBufferLayout,
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'main_node_frag',
                targets: [
                    {
                        format: presentationFormat,
                        blend: {
                            // The source color is the value written by the fragment shader. 
                            // The destination color is the color from the image in the framebuffer.
                            // https://www.khronos.org/opengl/wiki/Blending
                            color: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            }
                        },
                    } as GPUColorTargetState,
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });


        // Render - links
        const linkPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'main_link_vert',
                buffers: [
                    {
                        arrayStride: 2 * 4, // vec2 float
                        stepMode: 'vertex',
                        attributes: [
                            {
                                // vertex positions
                                shaderLocation: 0, offset: 0, format: 'float32x2',
                            }
                        ],
                    } as GPUVertexBufferLayout,
                    {
                        arrayStride: 2 * 4, // vec2 u32
                        stepMode: 'instance',
                        attributes: [
                            {
                                shaderLocation: 1, offset: 0, format: 'uint32x2',
                            }
                        ],
                    } as GPUVertexBufferLayout
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'main_link_frag',
                targets: [
                    {
                        format: presentationFormat,
                        blend: {
                            color: {
                                srcFactor: 'src',
                                dstFactor: 'zero',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'zero',
                                dstFactor: 'one',
                                operation: 'add',
                            }
                        },
                    } as GPUColorTargetState,
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });

        const depthTexture = device.createTexture({
            size: presentationSize,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // uniform
        const aspect = presentationSize[0] / presentationSize[1];
        const projectionMatrix = mat4.create();

        const uniformBufferSize = Math.max(4 * 4 * 4 + 4 * 3, 80); //  float32 4x4 matrix, vec3<f32>;
        const uniformBufferData = new Float32Array(uniformBufferSize / 4);

        const uniformBuffer = device.createBuffer({
            size: uniformBufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const nodeBindGroup = device.createBindGroup({
            layout: nodePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer,
                    },
                },
            ],
        });
        const linkBindGroup = device.createBindGroup({
            layout: linkPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: nodeBufferGPU,
                    },
                }
            ],
        });


        mat4.perspective(projectionMatrix, (2 * Math.PI) / 5, aspect, 1, 300.0);

        // the world
        const worldUp = vec3.fromValues(0, 1, 0);
        const worldOrigin = vec3.fromValues(0, 0, 0);

        let lastTime = performance.now();
        let frameCount = 0;
        const camera = new FPSController(vec3.fromValues(0, 0, 20), quat.create());
        function UpdateView(time: DOMHighResTimeStamp) {
            const deltaTime = (time - lastTime) / 1000; //ms -> second double
            lastTime = time;
            camera.Update(deltaTime);

            const viewMatrix = mat4.create();
            const watchCenter = vec3.create();
            vec3.add(watchCenter, camera.position, camera.front);

            mat4.lookAt(
                viewMatrix,
                camera.position,
                watchCenter,
                worldUp
            );
            // x: right y: up z: out screen
            // view matrix = inverse of model matrix of camera
            //mat4.invert(viewMatrix, viewMatrix);
            const viewProjectionMatrix = mat4.create();
            mat4.multiply(viewProjectionMatrix, projectionMatrix, viewMatrix);
            uniformBufferData.set(viewProjectionMatrix, 0);
            uniformBufferData.set(camera.front, 4 * 4);
            device.queue.writeBuffer(
                uniformBuffer,
                0,
                uniformBufferData
            );

            frameCount = (frameCount + 1);
            if (frameCount >= 60) {
                frameCount = 0;
            }
            const fps = 1 / deltaTime;
            if (fps < 0.1) {
                // debugger; // too slow
            }
            _.Log(`FPS: ${(fps).toFixed(1)}`);
        }
        let runSimulatorUpdate = true;
        function Update(time: DOMHighResTimeStamp) {
            UpdateView(time);
            if (runSimulatorUpdate) {
                try {
                    let hasUpdate = simulator.Update();
                    if (hasUpdate) {
                        device.queue.writeBuffer(linkBufferGPU, 0, simulator.linkBuffer);
                        device.queue.writeBuffer(nodeColorBufferGPU, 0, simulator.nodeColorBuffer);
                        device.queue.writeBuffer(nodeBufferGPU, 0, simulator.nodeBuffer);
                    }
                } catch (e) {
                    if (e.type == "retry") {
                        console.log("Retry at: ", e.info);
                        simulator.ResetData()
                        console.log("Restart");
                    } else if (e.type == "abort") {
                        runSimulatorUpdate = false;
                    }
                    else {
                        throw e;
                    }
                }
            }
            const commandEncoder = device.createCommandEncoder();
            commandEncoder.pushDebugGroup("NodeLinkRenderer"); // require HTTPS
            // Must create every time, or there would be 'Destroyed texture [Texture] used in a submit.'
            const textureView = context.getCurrentTexture().createView();

            const renderPassDescriptor = {
                colorAttachments: [
                    {
                        view: textureView,

                        clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    },
                ],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            } as GPURenderPassDescriptor;

            // const computePassEncoder = commandEncoder.beginComputePass();
            // computePassEncoder.setPipeline(computePipeline);
            // computePassEncoder.setBindGroup(0, frameCount % 2 === 0 ? bindGroup0 : bindGroup1);
            // computePassEncoder.dispatchWorkgroups(3, 1);
            // computePassEncoder.end();

            const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
            renderPassEncoder.setPipeline(nodePipeline);
            renderPassEncoder.setBindGroup(0, nodeBindGroup);
            renderPassEncoder.setVertexBuffer(0, quadVertexBuffer);
            renderPassEncoder.setVertexBuffer(1, nodeBufferGPU);
            renderPassEncoder.setVertexBuffer(2, nodeColorBufferGPU);
            renderPassEncoder.draw(quadVertexData.length / 2, simulator.nodeBuffer.length / 4, 0, 0);
            renderPassEncoder.setPipeline(linkPipeline);
            renderPassEncoder.setBindGroup(0, linkBindGroup);
            renderPassEncoder.setVertexBuffer(1, linkBufferGPU);
            renderPassEncoder.draw(quadVertexData.length / 2, simulator.linkBuffer.length / 2, 0, 0);

            renderPassEncoder.end();
            commandEncoder.popDebugGroup();
            device.queue.submit([commandEncoder.finish()]);
            if (_.destroyFlag) {
                device.destroy();
                return;
            }
            requestAnimationFrame(Update);
        }

        requestAnimationFrame(Update);
    };

}