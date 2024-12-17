import wgsl from './shaders/FMM.wgsl';

import { IFMMKernel } from './FMMKernel';
import { KernelWgpu } from './kernel_wgpu/kernel_wgpu';
//import { KernelTs } from './kernels/kernel_ts';
import { TreeBuilder } from './TreeBuilder';

import { cart2sph, GetIndex3D, GetIndexFrom3D } from "./utils";


import { Debug_Id_Pair } from './Force';
import { debug_p2m } from './kernel_ts/debug_p2m';
import { debug_m2l, debug_m2l_p4 } from './kernel_ts/debug_m2l';
import { debug_l2p } from './kernel_ts/debug_l2p';
import { debug_m2p } from './kernel_ts/debug_m2p';

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






    getInteractionListP2P(numBoxIndex: number, numLevel: number) {
        const tree = this.tree;
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + tree.levelOffset[numLevel - 1];
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }

        //p2p
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + tree.levelOffset[numLevel - 1];
            tree.numInteraction[ii] = 0;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[ib]);
            let ix = boxIndex3D.x;
            let iy = boxIndex3D.y;
            let iz = boxIndex3D.z;
            for (let jx = Math.max(ix - 1, jxmin); jx <= Math.min(ix + 1, jxmax); jx++) {
                for (let jy = Math.max(iy - 1, jymin); jy <= Math.min(iy + 1, jymax); jy++) {
                    for (let jz = Math.max(iz - 1, jzmin); jz <= Math.min(iz + 1, jzmax); jz++) {
                        boxIndex3D.x = jx;
                        boxIndex3D.y = jy;
                        boxIndex3D.z = jz;
                        let boxIndex = GetIndexFrom3D(boxIndex3D, numLevel);
                        let jj = tree.boxIndexMask[boxIndex];
                        if (jj != -1) {
                            this.interactionList[ii][tree.numInteraction[ii]] = jj;
                            tree.numInteraction[ii]++;
                        }
                    }
                }
            }
        }
    }
    getInteractionListM2L(numBoxIndex: number, numLevel: number) {
        const tree = this.tree;
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + tree.levelOffset[numLevel - 1];
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }

        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + tree.levelOffset[numLevel - 1];
            tree.numInteraction[ii] = 0;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            for (let jj = 0; jj < numBoxIndex; jj++) {
                let jb = jj + tree.levelOffset[numLevel - 1];
                boxIndex3D = GetIndex3D(tree.boxIndexFull[jb]);
                let jx = boxIndex3D.x,
                    jy = boxIndex3D.y,
                    jz = boxIndex3D.z;
                if (jx < ix - 1 || ix + 1 < jx || jy < iy - 1 || iy + 1 < jy || jz < iz - 1 || iz + 1 < jz) {
                    this.interactionList[ii][tree.numInteraction[ii]] = jj;
                    tree.numInteraction[ii]++;
                }
            }
        }

    }
    getInteractionListM2LLower(numBoxIndex: number, numLevel: number) {
        const tree = this.tree;
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + tree.levelOffset[numLevel - 1];
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + tree.levelOffset[numLevel - 1];
            tree.numInteraction[ii] = 0;
            let boxIndex3D = GetIndex3D(tree.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            let ixp = Math.floor((ix + 2) / 2),
                iyp = Math.floor((iy + 2) / 2),
                izp = Math.floor((iz + 2) / 2);
            for (let jxp = ixp - 1; jxp <= ixp + 1; jxp++) {
                for (let jyp = iyp - 1; jyp <= iyp + 1; jyp++) {
                    for (let jzp = izp - 1; jzp <= izp + 1; jzp++) {
                        for (let jx = Math.max(2 * jxp - 2, jxmin); jx <= Math.min(2 * jxp - 1, jxmax); jx++) {
                            for (let jy = Math.max(2 * jyp - 2, jymin); jy <= Math.min(2 * jyp - 1, jymax); jy++) {
                                for (let jz = Math.max(2 * jzp - 2, jzmin); jz <= Math.min(2 * jzp - 1, jzmax); jz++) {
                                    if (jx < ix - 1 || ix + 1 < jx || jy < iy - 1 || iy + 1 < jy || jz < iz - 1 || iz + 1 < jz) {
                                        boxIndex3D.x = jx;
                                        boxIndex3D.y = jy;
                                        boxIndex3D.z = jz;
                                        let boxIndex = GetIndexFrom3D(boxIndex3D, numLevel);
                                        let jj = tree.boxIndexMask[boxIndex];
                                        if (jj != -1) {
                                            this.interactionList[ii][tree.numInteraction[ii]] = jj;
                                            tree.numInteraction[ii]++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


    }
    kernel: IFMMKernel;
    async main() {
        // this.setBoxSize();
        // this.setOptimumLevel();
        // this.sortParticles();
        // this.countNonEmptyBoxes();
        // this.allocate();
        //this.interactionList = new Array(this.numBoxIndexLeaf).fill(0).map(_ => new Int32Array(maxM2LInteraction));
        const tree = this.tree;

        //     kernel.precalc();
        let numBoxIndex = 0;
        //   // P2P
        this.getInteractionListP2P(numBoxIndex, tree.maxLevel);
        //     bodyAccel.fill(0);



        await this.kernel.Init();

        // debug
        if (this.debug_watch_box_id_pairs) {
            this.debug_results = this.debug_watch_box_id_pairs.map(pair => {

                const p2m_result = debug_p2m(this, pair.src);

                const m2l_result = (() => {
                    const numLevel = tree.maxLevel;
                    return debug_m2l_p4(this, numLevel, p2m_result, pair.src, pair.dst);

                    return debug_m2l(this, numLevel, p2m_result, pair.src, pair.dst);
                })();

                const l2p_result = debug_l2p(this, m2l_result, pair.dst);

                const m2p_result = debug_m2p(this, p2m_result, pair.src, pair.dst);
                let direct_result;
                // p2p
                {
                    const dst_start = tree.particleOffset[0][pair.dst];
                    const dst_count = tree.particleOffset[1][pair.dst] - dst_start + 1;
                    const src_start = tree.particleOffset[0][pair.src];
                    const src_count = tree.particleOffset[1][pair.src] - src_start + 1;

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
                    {
                        step: "l2p", result: l2p_result
                    },
                    { step: "direct", result: direct_result }
                    , { step: "m2p", result: m2p_result }
                ];

            });

            console.log(this.debug_results)
            debugger;

        }

        //     kernel.p2p(numBoxIndex);
        await this.kernel.p2p(tree.numInteraction, this.interactionList);

        await this.kernel.p2m();

        if (tree.maxLevel > 2) {
            for (let numLevel = tree.maxLevel - 1; numLevel >= 2; numLevel--) {
                let numBoxIndexOld = numBoxIndex;
                //numBoxIndex = tree.getBoxDataOfParent(numBoxIndex, numLevel);
                // todo
                await this.kernel.m2m(numLevel);
            }
            //numLevel = 2;
        }
        else {
            tree.getBoxIndexMask(numBoxIndex, tree.maxLevel);
        }
        console.log(numBoxIndex)
        this.getInteractionListM2L(numBoxIndex, 2);
        throw "pause before m2l"
        // await this.kernel.m2l(numBoxIndex, numLevel);
        await this.kernel.m2l(numBoxIndex, 2);
        throw "pause"

        if (tree.maxLevel > 2) {

            for (let numLevel = 3; numLevel <= tree.maxLevel; numLevel++) {

                numBoxIndex = tree.levelOffset[numLevel - 2] - tree.levelOffset[numLevel - 1];

                await this.kernel.l2l(numBoxIndex, numLevel);

                tree.getBoxIndexMask(numBoxIndex, numLevel);

                this.getInteractionListM2LLower(numBoxIndex, numLevel);

                await this.kernel.m2l(numBoxIndex, numLevel);
            }
            // numLevel = tree.maxLevel;
        }

        await this.kernel.l2p(numBoxIndex);

        this.kernel.Release();

    }
    numExpansions: number;
    numExpansion2: number;
    numExpansion4: number;
    numCoefficients: number;
    DnmSize: number;

    tree: TreeBuilder;

    constructor(tree: TreeBuilder, kernelName: string = "wgpu") {
        const TKernel = {
            "wgpu": KernelWgpu,
            // "ts": KernelTs
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
    }

    isDataReady() {
        return this.kernel.dataReady;
    }
    getAccelBuffer() {
        throw "pause";
    }

}