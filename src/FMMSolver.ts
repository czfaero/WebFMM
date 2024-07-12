import wgsl from './shaders/FMM.wgsl';

import { IKernel } from './kernels/kernel';
import { KernelWgpu } from './kernels/kernel_wgpu';
//import { KernelTs } from './kernels/kernel_ts';
import { TreeBuilder } from './TreeBuilder';

import { debug_p2m } from './debug_p2m';

/**max of M2L interacting boxes */
const maxM2LInteraction = 189;

export class FMMSolver {
    // Basic data and helper


    getNode(i: number) {
        return this.tree.getNode(i);
    }




    /**@return Object with x,y,z */
    unmorton(boxIndex: number) {
        const mortonIndex3D = new Int32Array(3);

        mortonIndex3D.fill(0);
        let n = boxIndex;
        let k = 0;
        let i = 0;
        while (n != 0) {
            let j = 2 - k;
            mortonIndex3D[j] += (n % 2) * (1 << i);
            n >>= 1;
            k = (k + 1) % 3;
            if (k == 0) i++;
        }
        return {
            x: mortonIndex3D[1],
            y: mortonIndex3D[2],
            z: mortonIndex3D[0]
        }
    }
    // Generate Morton index for a box center to use in M2L translation
    morton1(boxIndex3D, numLevel: number) {

        let boxIndex = 0;
        for (let i = 0; i < numLevel; i++) {
            let nx = boxIndex3D.x % 2;
            boxIndex3D.x >>= 1;
            boxIndex += nx * (1 << (3 * i + 1));

            let ny = boxIndex3D.y % 2;
            boxIndex3D.y >>= 1;
            boxIndex += ny * (1 << (3 * i));

            let nz = boxIndex3D.z % 2;
            boxIndex3D.z >>= 1;
            boxIndex += nz * (1 << (3 * i + 2));
        }
        return boxIndex
    }



    interactionList: any;


    // Propagate non-empty/full link list to parent boxes
    getBoxDataOfParent(_numBoxIndex: number, numLevel: number) {
        const tree = this.tree;
        tree.levelOffset[numLevel - 1] = tree.levelOffset[numLevel] + _numBoxIndex;
        let numBoxIndexOld = _numBoxIndex;
        let numBoxIndex = 0;
        let currentIndex = -1;
        for (let i = 0; i < tree.numBoxIndexFull; i++)
            tree.boxIndexMask[i] = -1;
        for (let i = 0; i < numBoxIndexOld; i++) {
            let boxIndex = i + tree.levelOffset[numLevel];
            if (currentIndex != Math.floor(tree.boxIndexFull[boxIndex] / 8)) {
                currentIndex = Math.floor(tree.boxIndexFull[boxIndex] / 8);
                tree.boxIndexMask[currentIndex] = numBoxIndex;
                tree.boxIndexFull[numBoxIndex + tree.levelOffset[numLevel - 1]] = currentIndex;
                numBoxIndex++;
            }
        }
        return numBoxIndex;
    }

    // Recalculate non-empty box index for current level
    getBoxIndexMask(numBoxIndex: number, numLevel: number) {
        const tree = this.tree;
        for (let i = 0; i < tree.numBoxIndexFull; i++)
            tree.boxIndexMask[i] = -1;
        for (let i = 0; i < numBoxIndex; i++) {
            let boxIndex = i + tree.levelOffset[numLevel - 1];
            tree.boxIndexMask[tree.boxIndexFull[boxIndex]] = i;
        }
    }

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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[jb]);
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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[ib]);
            let ix = boxIndex3D.x;
            let iy = boxIndex3D.y;
            let iz = boxIndex3D.z;
            for (let jx = Math.max(ix - 1, jxmin); jx <= Math.min(ix + 1, jxmax); jx++) {
                for (let jy = Math.max(iy - 1, jymin); jy <= Math.min(iy + 1, jymax); jy++) {
                    for (let jz = Math.max(iz - 1, jzmin); jz <= Math.min(iz + 1, jzmax); jz++) {
                        boxIndex3D.x = jx;
                        boxIndex3D.y = jy;
                        boxIndex3D.z = jz;
                        let boxIndex = this.morton1(boxIndex3D, numLevel);
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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[jb]);
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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            for (let jj = 0; jj < numBoxIndex; jj++) {
                let jb = jj + tree.levelOffset[numLevel - 1];
                boxIndex3D = this.unmorton(tree.boxIndexFull[jb]);
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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[jb]);
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
            let boxIndex3D = this.unmorton(tree.boxIndexFull[ib]);
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
                                        let boxIndex = this.morton1(boxIndex3D, numLevel);
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
    kernel: IKernel;
    async main() {
        // this.setBoxSize();
        // this.setOptimumLevel();
        // this.sortParticles();
        // this.countNonEmptyBoxes();
        // this.allocate();
        //this.interactionList = new Array(this.numBoxIndexLeaf).fill(0).map(_ => new Int32Array(maxM2LInteraction));
        const tree = this.tree;
        let numLevel = tree.maxLevel;
        tree.levelOffset[numLevel - 1] = 0;
        //     kernel.precalc();
        let numBoxIndex = 0;
        //   // P2P
        this.getInteractionListP2P(numBoxIndex, numLevel);
        //     bodyAccel.fill(0);



        await this.kernel.Init(tree.nodeBuffer);

        // debug
        {
            const p2m_result = tree.debug_watch.map(info => {
                return debug_p2m(this, info.box, tree)
            })
            console.log(p2m_result)

        }

        //     kernel.p2p(numBoxIndex);
        await this.kernel.p2p(tree.numInteraction, this.interactionList);

        await this.kernel.p2m();

        if (tree.maxLevel > 2) {
            for (numLevel = tree.maxLevel - 1; numLevel >= 2; numLevel--) {
                let numBoxIndexOld = numBoxIndex;
                numBoxIndex = this.getBoxDataOfParent(numBoxIndex, numLevel);
                await this.kernel.m2m(numBoxIndex, numBoxIndexOld, numLevel);
            }
            numLevel = 2;
        }
        else {
            this.getBoxIndexMask(numBoxIndex, numLevel);
        }
        console.log(numBoxIndex)
        this.getInteractionListM2L(numBoxIndex, numLevel);
        await this.kernel.m2l(numBoxIndex, numLevel);

        if (tree.maxLevel > 2) {

            for (numLevel = 3; numLevel <= tree.maxLevel; numLevel++) {

                numBoxIndex = tree.levelOffset[numLevel - 2] - tree.levelOffset[numLevel - 1];

                await this.kernel.l2l(numBoxIndex, numLevel);

                this.getBoxIndexMask(numBoxIndex, numLevel);

                this.getInteractionListM2LLower(numBoxIndex, numLevel);

                await this.kernel.m2l(numBoxIndex, numLevel);
            }
            numLevel = tree.maxLevel;
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
        this.numExpansions = 10;
        this.numExpansion2 = this.numExpansions * this.numExpansions;
        this.numExpansion4 = this.numExpansion2 * this.numExpansion2;
        this.numCoefficients = this.numExpansions * (this.numExpansions + 1) / 2;
        this.DnmSize = (4 * this.numExpansion2 * this.numExpansions - this.numExpansions) / 3;
    }

    isDataReady() {
        return this.kernel.dataReady;
    }

}