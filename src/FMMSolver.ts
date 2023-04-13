const maxM2LInteraction = 189;        // max of M2L interacting boxes

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
    numBoxIndexFull: number;
    setOptimumLevel() {
        // 按照点的数量区间定级别
        const level_switch = [1e5, 7e5, 7e6, 5e7, 3e8, 2e9]; // gpu-fmm

        this.maxLevel = 1;
        for (const level of level_switch) {
            if (this.particleCount >= level) {
                this.maxLevel++;
            } else {
                break;
            }
        }

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
    };
    morton() {
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
            tempParticle.set(this.particalBuffer.subarray(offset, offset + 4), i * 4);
        }
        this.particleBuffer = tempParticle;
    };
    numBoxIndexLeaf: number;
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
                if (sortValue[i] / (1 << 3 * (this.maxLevel - numLevel)) != currentIndex) {
                    this.numBoxIndexTotal++;
                    currentIndex = sortValue[i] / (1 << 3 * (this.maxLevel - numLevel));
                }
            }
        }


    }
    levelOffset: Int32Array;
    particleOffset: any;
    boxIndexMask: Int32Array;
    boxIndexFull: Int32Array;
    numInteraction: Int32Array;
    interactionList: any;
    boxOffsetStart: Int32Array;
    boxOffsetEnd: Int32Array;
    allocate() {

        this.particleOffset = [0, 0].map(_ => new Array(this.numBoxIndexLeaf));
        this.boxIndexMask = new Int32Array(this.numBoxIndexFull);
        this.boxIndexFull = new Int32Array(this.numBoxIndexTotal);
        this.levelOffset = new Int32Array(this.maxLevel);

        this.numInteraction = new Int32Array(this.numBoxIndexLeaf);
        this.interactionList = new Array(this.numBoxIndexLeaf).map(_ => new Int32Array(maxM2LInteraction));
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
    main() {
        this.setBoxSize();
        this.setOptimumLevel();
        this.sortParticles();
        this.countNonEmptyBoxes();
        //     allocate();
        let numLevel = this.maxLevel;
        //     levelOffset[numLevel-1] = 0;
        //     kernel.precalc();
        let numBoxIndex = this.getBoxData();
        //   // P2P
        this.getInteractionListP2P(numBoxIndex, numLevel);
        //     bodyAccel.fill(0);
        //     kernel.p2p(numBoxIndex);
    }

    constructor() {
    }

}