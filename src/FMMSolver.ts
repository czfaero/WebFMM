import wgsl from './shaders/FMM.wgsl';

import { IKernel } from './kernels/kernel';
import { KernelWgpu } from './kernels/kernel_wgpu';
import { KernelTs } from './kernels/kernel_ts';

/**max of M2L interacting boxes */
const maxM2LInteraction = 189;

export class FMMSolver {
    // Basic data and helper
    particleBuffer: Float32Array; // vec4
    particleCount: number;
    getParticle(i: number) {
        return {
            x: this.particleBuffer[i * 4],
            y: this.particleBuffer[i * 4 + 1],
            z: this.particleBuffer[i * 4 + 2],
            w: this.particleBuffer[i * 4 + 3]
        }
    }

    // Init box sizes
    boxMinX: number;
    boxMinY: number;
    boxMinZ: number;
    rootBoxSize: number;
    setBoxSize() {
        // FmmSystem::setDomainSize(int numParticles)
        // 也许可以配合随机生成初始位置简单设定范围。

        let xmin = 1000000,
            xmax = -1000000,
            ymin = 1000000,
            ymax = -1000000,
            zmin = 1000000,
            zmax = -1000000;

        for (let i = 0; i < this.particleCount; i++) {
            const { x, y, z } = this.getParticle(i);
            xmin = Math.min(xmin, x);
            xmax = Math.max(xmax, x);
            ymin = Math.min(ymin, y);
            ymax = Math.max(ymax, y);
            zmin = Math.min(zmin, z);
            zmax = Math.max(zmax, z);
        }

        this.boxMinX = xmin;
        this.boxMinY = ymin;
        this.boxMinZ = zmin;

        this.rootBoxSize = 0;
        this.rootBoxSize = Math.max(this.rootBoxSize, xmax - xmin, ymax - ymin, zmax - zmin);
        this.rootBoxSize *= 1.00001; // Keep particle on the edge from falling out

    };

    //setOptimumLevel
    maxLevel: number;
    /** All boxes at maxLevel */
    numBoxIndexFull: number;
    setOptimumLevel() {
        // 按照点的数量区间定级别
        const level_switch = [1e5, 7e5, 7e6, 5e7, 3e8, 2e9]; // gpu-fmm

        this.maxLevel = 2;
        for (const level of level_switch) {
            if (this.particleCount >= level) {
                this.maxLevel++;
            } else {
                break;
            }
        }
        console.log("maxLevel: " + this.maxLevel);

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
    };
    /**@return Array, box id for every partical*/
    morton(): Int32Array {
        const resultIndex = new Int32Array(this.particleCount);
        const boxSize = this.rootBoxSize / (1 << this.maxLevel);
        for (let nodeIndex = 0; nodeIndex < this.particleCount; nodeIndex++) {
            const { x, y, z } = this.getParticle(nodeIndex);
            let nx = Math.floor((x - this.boxMinX) / boxSize),
                ny = Math.floor((y - this.boxMinY) / boxSize),
                nz = Math.floor((z - this.boxMinZ) / boxSize);

            if (nx >= (1 << this.maxLevel)) nx--;
            if (ny >= (1 << this.maxLevel)) ny--;
            if (nz >= (1 << this.maxLevel)) nz--;
            let boxIndex = 0;
            for (let level = 0; level < this.maxLevel; level++) {
                boxIndex += nx % 2 << (3 * level + 1);
                nx >>= 1;

                boxIndex += ny % 2 << (3 * level);
                ny >>= 1;

                boxIndex += nz % 2 << (3 * level + 2);
                nz >>= 1;
            }
            resultIndex[nodeIndex] = boxIndex;
        }
        return resultIndex;
    };
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

    /** sort the Morton index
     * @return sortValue (boxid), sortIndex (i) 
    */
    sort(mortonIndex: Int32Array) {
        const tempSortIndex = new Int32Array(this.numBoxIndexFull);
        const sortValue = new Int32Array(this.particleCount);
        const sortIndex = new Int32Array(this.particleCount);
        for (let i = 0; i < this.particleCount; i++) {
            sortIndex[i] = i;
        }
        tempSortIndex.fill(0);
        for (const i in mortonIndex) {
            tempSortIndex[mortonIndex[i]]++;
        }
        for (let i = 1; i < this.numBoxIndexFull; i++) {
            tempSortIndex[i] += tempSortIndex[i - 1];
        }
        for (let i = this.particleCount - 1; i >= 0; i--) {
            tempSortIndex[mortonIndex[i]]--;
            sortValue[tempSortIndex[mortonIndex[i]]] = mortonIndex[i];
            sortIndex[tempSortIndex[mortonIndex[i]]] = i;
        }
        return { sortValue, sortIndex }
    }
    sortParticles() {
        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);

        const tempParticle = new Float32Array(this.particleBuffer.length);
        for (let i = 0; i < this.particleCount; i++) {
            const offset = sortIndex[i] * 4;
            tempParticle.set(this.particleBuffer.subarray(offset, offset + 4), i * 4);
        }
        this.particleBuffer = tempParticle;
    };
    /** non-empty FMM boxes @ maxLevel */
    numBoxIndexLeaf: number;
    /** numBoxIndexLeaf for all levels */
    numBoxIndexTotal: number;
    countNonEmptyBoxes() {
        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);
        this.numBoxIndexLeaf = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.particleCount; i++) {
            if (sortValue[i] != currentIndex) {
                this.numBoxIndexLeaf++;
                currentIndex = sortValue[i];
            }
        }

        this.numBoxIndexTotal = this.numBoxIndexLeaf;
        for (let numLevel = this.maxLevel - 1; numLevel >= 2; numLevel--) {
            currentIndex = -1;
            for (let i = 0; i < this.particleCount; i++) {
                const temp = Math.floor(sortValue[i] / (1 << 3 * (this.maxLevel - numLevel)));
                if (temp != currentIndex) {
                    this.numBoxIndexTotal++;
                    currentIndex = temp;
                }
            }
        }
    }

    levelOffset: Int32Array;
    /**
     * first and last particle in each box
     * int[2][numBoxIndexLeaf]
     */
    particleOffset: any;
    /** int[numBoxIndexFull]; link list for box index : Full -> NonEmpty */
    boxIndexMask: Int32Array;
    /** int[numBoxIndexTotal]; link list for box index : NonEmpty -> Full */
    boxIndexFull: Int32Array;
    /** int[numBoxIndexLeaf] */
    numInteraction: Int32Array;
    /** int[numBoxIndexLeaf][maxM2LInteraction] */
    interactionList: any;
    /** int[numBoxIndexLeaf] */
    boxOffsetStart: Int32Array;
    /** int[numBoxIndexLeaf] */
    boxOffsetEnd: Int32Array;
    allocate() {

        this.particleOffset = [0, 0].map(_ => new Array(this.numBoxIndexLeaf));
        this.boxIndexMask = new Int32Array(this.numBoxIndexFull);
        this.boxIndexFull = new Int32Array(this.numBoxIndexTotal);
        this.levelOffset = new Int32Array(this.maxLevel);

        this.numInteraction = new Int32Array(this.numBoxIndexLeaf);
        this.interactionList = new Array(this.numBoxIndexLeaf).fill(0).map(_ => new Int32Array(maxM2LInteraction));
        this.boxOffsetStart = new Int32Array(this.numBoxIndexLeaf);
        this.boxOffsetEnd = new Int32Array(this.numBoxIndexLeaf);
    }

    getBoxData() {
        const mortonIndex = this.morton();

        let numBoxIndex = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.numBoxIndexFull; i++) this.boxIndexMask[i] = -1;
        for (let i = 0; i < this.particleCount; i++) {
            if (mortonIndex[i] != currentIndex) {
                this.boxIndexMask[mortonIndex[i]] = numBoxIndex;
                this.boxIndexFull[numBoxIndex] = mortonIndex[i];
                this.particleOffset[0][numBoxIndex] = i;
                if (numBoxIndex > 0) this.particleOffset[1][numBoxIndex - 1] = i - 1;
                currentIndex = mortonIndex[i];
                numBoxIndex++;
            }
        }
        this.particleOffset[1][numBoxIndex - 1] = this.particleCount - 1;
        return numBoxIndex;
    }
    // Propagate non-empty/full link list to parent boxes
    getBoxDataOfParent(_numBoxIndex: number, numLevel: number) {
        this.levelOffset[numLevel - 1] = this.levelOffset[numLevel] + _numBoxIndex;
        let numBoxIndexOld = _numBoxIndex;
        let numBoxIndex = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.numBoxIndexFull; i++)
            this.boxIndexMask[i] = -1;
        for (let i = 0; i < numBoxIndexOld; i++) {
            let boxIndex = i + this.levelOffset[numLevel];
            if (currentIndex != Math.floor(this.boxIndexFull[boxIndex] / 8)) {
                currentIndex = Math.floor(this.boxIndexFull[boxIndex] / 8);
                this.boxIndexMask[currentIndex] = numBoxIndex;
                this.boxIndexFull[numBoxIndex + this.levelOffset[numLevel - 1]] = currentIndex;
                numBoxIndex++;
            }
        }
        return numBoxIndex;
    }

    // Recalculate non-empty box index for current level
    getBoxIndexMask(numBoxIndex: number, numLevel: number) {
        for (let i = 0; i < this.numBoxIndexFull; i++)
            this.boxIndexMask[i] = -1;
        for (let i = 0; i < numBoxIndex; i++) {
            let boxIndex = i + this.levelOffset[numLevel - 1];
            this.boxIndexMask[this.boxIndexFull[boxIndex]] = i;
        }
    }

    getInteractionListP2P(numBoxIndex: number, numLevel: number) {
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + this.levelOffset[numLevel - 1];
            let boxIndex3D = this.unmorton(this.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }

        //p2p
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + this.levelOffset[numLevel - 1];
            this.numInteraction[ii] = 0;
            let boxIndex3D = this.unmorton(this.boxIndexFull[ib]);
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
                        let jj = this.boxIndexMask[boxIndex];
                        if (jj != -1) {
                            this.interactionList[ii][this.numInteraction[ii]] = jj;
                            this.numInteraction[ii]++;
                        }
                    }
                }
            }
        }
    }
    getInteractionListM2L(numBoxIndex: number, numLevel: number) {
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + this.levelOffset[numLevel - 1];
            let boxIndex3D = this.unmorton(this.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }

        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + this.levelOffset[numLevel - 1];
            this.numInteraction[ii] = 0;
            let boxIndex3D = this.unmorton(this.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            for (let jj = 0; jj < numBoxIndex; jj++) {
                let jb = jj + this.levelOffset[numLevel - 1];
                boxIndex3D = this.unmorton(this.boxIndexFull[jb]);
                let jx = boxIndex3D.x,
                    jy = boxIndex3D.y,
                    jz = boxIndex3D.z;
                if (jx < ix - 1 || ix + 1 < jx || jy < iy - 1 || iy + 1 < jy || jz < iz - 1 || iz + 1 < jz) {
                    this.interactionList[ii][this.numInteraction[ii]] = jj;
                    this.numInteraction[ii]++;
                }
            }
        }

    }
    getInteractionListM2LLower(numBoxIndex: number, numLevel: number) {
        // Initialize the minimum and maximum values
        let jxmin = 1000000,
            jxmax = -1000000,
            jymin = 1000000,
            jymax = -1000000,
            jzmin = 1000000,
            jzmax = -1000000;
        // Calculate the minimum and maximum of boxIndex3D
        for (let jj = 0; jj < numBoxIndex; jj++) {
            let jb = jj + this.levelOffset[numLevel - 1];
            let boxIndex3D = this.unmorton(this.boxIndexFull[jb]);
            jxmin = Math.min(jxmin, boxIndex3D.x);
            jxmax = Math.max(jxmax, boxIndex3D.x);
            jymin = Math.min(jymin, boxIndex3D.y);
            jymax = Math.max(jymax, boxIndex3D.y);
            jzmin = Math.min(jzmin, boxIndex3D.z);
            jzmax = Math.max(jzmax, boxIndex3D.z);
        }
        for (let ii = 0; ii < numBoxIndex; ii++) {
            let ib = ii + this.levelOffset[numLevel - 1];
            this.numInteraction[ii] = 0;
            let boxIndex3D = this.unmorton(this.boxIndexFull[ib]);
            let ix = boxIndex3D.x,
                iy = boxIndex3D.y,
                iz = boxIndex3D.z;
            let ixp = (ix + 2) / 2,
                iyp = (iy + 2) / 2,
                izp = (iz + 2) / 2;
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
                                        let jj = this.boxIndexMask[boxIndex];
                                        if (jj != -1) {
                                            this.interactionList[ii][this.numInteraction[ii]] = jj;
                                            this.numInteraction[ii]++;
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
        this.setBoxSize();
        this.setOptimumLevel();
        this.sortParticles();
        this.countNonEmptyBoxes();
        this.allocate();
        let numLevel = this.maxLevel;
        this.levelOffset[numLevel - 1] = 0;
        //     kernel.precalc();
        let numBoxIndex = this.getBoxData();
        //   // P2P
        this.getInteractionListP2P(numBoxIndex, numLevel);
        //     bodyAccel.fill(0);

        await this.kernel.Init(this.particleBuffer);
        //     kernel.p2p(numBoxIndex);
        await this.kernel.p2p(numBoxIndex, this.interactionList, this.numInteraction, this.particleOffset);

        await this.kernel.p2m(numBoxIndex, this.particleOffset);

        if (this.maxLevel > 2) {
            for (numLevel = this.maxLevel - 1; numLevel >= 2; numLevel--) {
                let numBoxIndexOld = numBoxIndex;
                numBoxIndex = this.getBoxDataOfParent(numBoxIndex, numLevel);
                await this.kernel.m2m(numBoxIndex, numBoxIndexOld, numLevel);
            }
            numLevel = 2;
        }
        else {
            this.getBoxIndexMask(numBoxIndex, numLevel);
        }
        this.getInteractionListM2L(numBoxIndex, numLevel);
        await this.kernel.m2l(numBoxIndex, numLevel);

        if (this.maxLevel > 2) {

            for (numLevel = 3; numLevel <= this.maxLevel; numLevel++) {

                numBoxIndex = this.levelOffset[numLevel - 2] - this.levelOffset[numLevel - 1];

                await this.kernel.l2l(numBoxIndex, numLevel);

                this.getBoxIndexMask(numBoxIndex, numLevel);

                this.getInteractionListM2LLower(numBoxIndex, numLevel);

                await this.kernel.m2l(numBoxIndex, numLevel);
            }
            numLevel = this.maxLevel;
        }

        await this.kernel.l2p(numBoxIndex);

        this.kernel.Release();

    }
    numExpansions: number;
    numExpansion2: number;
    numExpansion4: number;
    numCoefficients: number;
    DnmSize: number;

    constructor(particleBuffer: Float32Array, kernelName: string) {
        const TKernel = { "wgpu": KernelWgpu, "ts": KernelTs }[kernelName];
        if (!TKernel) throw "Unknown Kernel: " + kernelName;
        console.log("Create with kernel: " + kernelName);
        this.kernel = new TKernel(this);
        this.particleBuffer = particleBuffer;
        this.particleCount = particleBuffer.length / 4;


        // constants
        this.numExpansions = 10;
        this.numExpansion2 = this.numExpansions * this.numExpansions;
        this.numExpansion4 = this.numExpansion2 * this.numExpansion2;
        this.numCoefficients = this.numExpansions * (this.numExpansions + 1) / 2;
        this.DnmSize = (4 * this.numExpansion2 * this.numExpansions - this.numExpansions) / 3;
    }

}