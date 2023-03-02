export class FMMSolver {
    // Basic data and helper
    particalBuffer: Float32Array; // vec4
    particalCount: number;
    getNodePosition(i: number) {
        return {
            x: this.particalBuffer[i * 4],
            y: this.particalBuffer[i * 4 + 1],
            z: this.particalBuffer[i * 4 + 2],
            w: this.particalBuffer[i * 4 + 3]
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

        for (let i = 0; i < this.particalCount; i++) {
            const { x, y, z } = this.getNodePosition(i);
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
            if (this.particalCount >= level) {
                this.maxLevel++;
            } else {
                break;
            }
        }

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
    };
    morton() {
        const resultIndex = new Uint32Array(this.particalCount);
        const boxSize = this.rootBoxSize / (1 << this.maxLevel);
        for (let nodeIndex = 0; nodeIndex < this.particalCount; nodeIndex++) {
            const { x, y, z } = this.getNodePosition(nodeIndex);
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

    sort(mortonIndex: Uint32Array) {
        const tempSortIndex = new Uint32Array(this.numBoxIndexFull);
        const sortValue = new Uint32Array(this.particalCount);
        const sortIndex = new Uint32Array(this.particalCount);
        for (let i = 0; i < this.particalCount; i++) {
            sortIndex[i] = i;
        }
        tempSortIndex.fill(0);
        for (const i in mortonIndex) {
            tempSortIndex[mortonIndex[i]]++;
        }
        for (let i = 1; i < this.numBoxIndexFull; i++) {
            tempSortIndex[i] += tempSortIndex[i - 1];
        }
        for (let i = this.particalCount - 1; i >= 0; i--) {
            tempSortIndex[mortonIndex[i]]--;
            sortValue[tempSortIndex[mortonIndex[i]]] = mortonIndex[i];
            sortIndex[tempSortIndex[mortonIndex[i]]] = i;
        }
        return { sortValue, sortIndex }
    }
    sortParticles() {
        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);

        const tempParticle = new Float32Array(this.particalBuffer.length);
        for (let i = 0; i < this.particalCount; i++) {
            const offset = sortIndex[i] * 4;
            tempParticle.set(this.particalBuffer.subarray(offset, offset + 4), i * 4);
        }
        this.particalBuffer = tempParticle;
    };
    numBoxIndexLeaf: number;
    numBoxIndexTotal: number;
    countNonEmptyBoxes() {
        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);
        this.numBoxIndexLeaf = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.particalCount; i++) {
            if (sortValue[i] != currentIndex) {
                this.numBoxIndexLeaf++;
                currentIndex = sortValue[i];
            }
        }

        this.numBoxIndexTotal = this.numBoxIndexLeaf;
        for (let numLevel = this.maxLevel - 1; numLevel >= 2; numLevel--) {
            currentIndex = -1;
            for (let i = 0; i < this.particalCount; i++) {
                if (sortValue[i] / (1 << 3 * (this.maxLevel - numLevel)) != currentIndex) {
                    this.numBoxIndexTotal++;
                    currentIndex = sortValue[i] / (1 << 3 * (this.maxLevel - numLevel));
                }
            }
        }


    }
    main() {
        this.setBoxSize();
        this.setOptimumLevel();
        this.sortParticles();
        //     countNonEmptyBoxes(numParticles);
        //     allocate();
        //     numLevel = maxLevel;
        //     levelOffset[numLevel-1] = 0;
        //     kernel.precalc();
        //     getBoxData(numParticles,numBoxIndex);
        //   // P2P
        //     getInteractionListP2P(numBoxIndex,numLevel,0);
        //     bodyAccel.fill(0);
        //     kernel.p2p(numBoxIndex);
    }

    constructor() {
    }

}