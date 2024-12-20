import wgsl from './shaders/FMM.wgsl';

import { IFMMKernel } from './FMMKernel';
import { KernelWgpu } from './FMMKernel_wgpu/kernel_wgpu';
//import { KernelTs } from './kernels/kernel_ts';
import { TreeBuilder } from './TreeBuilder';

import { cart2sph, GetIndex3D, GetIndexFrom3D } from "./utils";


import { Debug_Id_Pair } from './Force';
import { debug_p2m } from './FMMKernel_ts/debug_p2m';
import { debug_m2l_p4 } from './FMMKernel_ts/debug_m2l';
import { debug_l2p } from './FMMKernel_ts/debug_l2p';
import { debug_m2p } from './FMMKernel_ts/debug_m2p';
import { FMMKernel_ts } from './FMMKernel_ts/kernel_ts';

/**max of M2L interacting boxes */
const maxM2LInteraction = 189;

export class FMMSolver {
    // Basic data and helper

    debug_watch_box_id_pairs: Array<Debug_Id_Pair>;//non-empty id

    debug_results;


    getNode(i: number) {
        return this.tree.getNode(i);
    }




    interactionList: any;
    /** int[numBoxIndexLeaf]; reused for InteractionList */
    interactionCounts: Int32Array;

    getBoxRange(offset: number, count: number) {
        const tree = this.tree;
        let xmin = 1000000,
            xmax = -1000000,
            ymin = 1000000,
            ymax = -1000000,
            zmin = 1000000,
            zmax = -1000000;

        for (let i = 0; i < count; i++) {
            let box_id = i + offset;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[box_id]);
            xmin = Math.min(xmin, boxIndex3D.x);
            xmax = Math.max(xmax, boxIndex3D.x);
            ymin = Math.min(ymin, boxIndex3D.y);
            ymax = Math.max(ymax, boxIndex3D.y);
            zmin = Math.min(zmin, boxIndex3D.z);
            zmax = Math.max(zmax, boxIndex3D.z);
        }
        return { xmin, xmax, ymin, ymax, zmin, zmax };
    }

    setInteractionListP2P() {
        const tree = this.tree;

        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        const mask = tree.boxIndexMaskBuffers[tree.maxLevel - 1];
        let { xmin, xmax, ymin, ymax, zmin, zmax } = this.getBoxRange(offset, boxCount);

        //p2p
        for (let i = 0; i < boxCount; i++) {
            this.interactionList[i].fill(0);
            this.interactionCounts[i] = 0;
            let box_id = i + offset;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[box_id]);
            let ix = boxIndex3D.x;
            let iy = boxIndex3D.y;
            let iz = boxIndex3D.z;
            for (let jx = Math.max(ix - 1, xmin); jx <= Math.min(ix + 1, xmax); jx++) {
                for (let jy = Math.max(iy - 1, ymin); jy <= Math.min(iy + 1, ymax); jy++) {
                    for (let jz = Math.max(iz - 1, zmin); jz <= Math.min(iz + 1, zmax); jz++) {
                        boxIndex3D.x = jx;
                        boxIndex3D.y = jy;
                        boxIndex3D.z = jz;
                        let boxIndex = GetIndexFrom3D(boxIndex3D);
                        let target_box_id = mask[boxIndex];
                        if (target_box_id != -1) {
                            this.interactionList[i][this.interactionCounts[i]] = target_box_id;
                            this.interactionCounts[i]++;
                        }
                    }
                }
            }
        }
    }
    setInteractionListM2L(numLevel: number) {
        const tree = this.tree;

        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        const mask = tree.boxIndexMaskBuffers[numLevel];
        let { xmin, xmax, ymin, ymax, zmin, zmax } = this.getBoxRange(offset, boxCount);

        for (let i = 0; i < boxCount; i++) {
            this.interactionList[i].fill(0);
            this.interactionCounts[i] = 0;
            let box_id = i + offset;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[box_id]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            let ixp = Math.floor(ix / 2),
                iyp = Math.floor(iy / 2),
                izp = Math.floor(iz / 2);// parent box index

            let xstart = Math.max(ixp * 2 - 2, xmin),
                ystart = Math.max(iyp * 2 - 2, ymin),
                zstart = Math.max(izp * 2 - 2, zmin);
            let xend = Math.min(ixp * 2 + 3, xmax),
                yend = Math.min(iyp * 2 + 3, ymax),
                zend = Math.min(izp * 2 + 3, zmax);
            for (let jx = xstart; jx <= xend; jx++)
                for (let jy = ystart; jy <= yend; jy++)
                    for (let jz = zstart; jz <= zend; jz++) {
                        if (jx < ix - 1 || ix + 1 < jx || jy < iy - 1 || iy + 1 < jy || jz < iz - 1 || iz + 1 < jz) {
                            let target_box_index = GetIndexFrom3D({ x: jx, y: jy, z: jz });
                            let target_box_id = mask[target_box_index];
                            
                            if (target_box_id != -1) {
                                this.interactionList[i][this.interactionCounts[i]] = target_box_id;
                                this.interactionCounts[i]++;
                            }

                        }
                    }
        }

    }

    kernel: IFMMKernel;
    async main() {
        const tree = this.tree;

        this.setInteractionListP2P();

        await this.kernel.Init();

        // debug
        if (this.debug_watch_box_id_pairs) {
            this.debug_results = this.debug_watch_box_id_pairs.map(pair => {

                const p2m_result = debug_p2m(this, pair.src);

                const m2l_result = (() => {
                    const numLevel = tree.maxLevel;
                    return debug_m2l_p4(this, numLevel, p2m_result, pair.src, pair.dst);
                })();

                const l2p_result = debug_l2p(this, m2l_result, pair.dst);

                const m2p_result = debug_m2p(this, p2m_result, pair.src, pair.dst);
                let direct_result;
                // p2p
                {
                    const dst_start = tree.nodeStartOffset[pair.dst];
                    const dst_count = tree.nodeEndOffset[pair.dst] - dst_start + 1;
                    const src_start = tree.nodeStartOffset[pair.src];
                    const src_count = tree.nodeEndOffset[pair.src] - src_start + 1;

                    direct_result = new Float32Array(dst_count * 3);
                    function dot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
                    function inverseSqrt(x) { return 1 / Math.sqrt(x); }
                    const eps = 1e-6;
                    for (let dst_i = 0; dst_i < dst_count; dst_i++) {

                        let accel = { x: 0, y: 0, z: 0 };
                        let dst = tree.getNode(dst_start + dst_i);
                        for (let src_i = 0; src_i < src_count; src_i++) {
                            let src = tree.getNode(src_start + src_i);
                            let dist = {
                                x: dst.x - src.x,
                                y: dst.y - src.y,
                                z: dst.z - src.z
                            };
                            let invDist = inverseSqrt(dot(dist, dist) + eps);
                            let s = invDist * invDist * invDist;
                            accel.x += -s * dist.x;
                            accel.y += -s * dist.y;
                            accel.z += -s * dist.z;
                        }
                        direct_result[dst_i * 3] = accel.x;
                        direct_result[dst_i * 3 + 1] = accel.y;
                        direct_result[dst_i * 3 + 2] = accel.z;
                    }
                }


                return [
                    pair,
                    { step: "p2m", result: p2m_result },
                    { step: "m2l", result: m2l_result },
                    { step: "l2p", result: l2p_result },
                    { step: "direct", result: direct_result },
                    { step: "m2p", result: m2p_result }
                ];

            });

            console.log(this.debug_results)
           // debugger;

        }

        //     kernel.p2p(numBoxIndex);
        await this.kernel.p2p();
        await this.kernel.p2m();


        for (let numLevel = tree.maxLevel - 1; numLevel >= 2; numLevel--) {
            await this.kernel.m2m(numLevel);
        }


        this.setInteractionListM2L(1);
        await this.kernel.m2l(1);
        debugger;
        for (let numLevel = 2; numLevel < tree.maxLevel; numLevel++) {
            await this.kernel.l2l(numLevel - 1);
            this.setInteractionListM2L(numLevel);
            await this.kernel.m2l(numLevel);

        }

        await this.kernel.l2p();

        this.kernel.Release();
    }

    numExpansions: number;
    numExpansion2: number;
    numExpansion4: number;
    numCoefficients: number;
    DnmSize: number;

    tree: TreeBuilder;

    constructor(tree: TreeBuilder, kernelName: string = "ts") {
        const TKernel = {
            "wgpu": KernelWgpu,
            "ts": FMMKernel_ts
        }[kernelName];
        if (!TKernel) throw "Unknown Kernel: " + kernelName;
        console.log("Create with kernel: " + kernelName);
        this.kernel = new TKernel(this);
        this.tree = tree;

        // constants
        this.numExpansions = 16;
        this.numExpansion2 = this.numExpansions * this.numExpansions;
        this.numExpansion4 = this.numExpansion2 * this.numExpansion2;
        this.numCoefficients = this.numExpansions * (this.numExpansions + 1) / 2;
        this.DnmSize = (4 * this.numExpansion2 * this.numExpansions - this.numExpansions) / 3;

        this.interactionCounts = new Int32Array(tree.numBoxIndexLeaf);
        this.interactionList = new Array(tree.numBoxIndexLeaf).fill(0).map(_ => new Int32Array(maxM2LInteraction));
    }

    isDataReady() {
        return this.kernel.dataReady;
    }
    getAccelBuffer() {
        throw "pause";
    }

}