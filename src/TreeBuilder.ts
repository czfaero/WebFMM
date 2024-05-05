

export class TreeBuilder {
    nodeBuffer: Float32Array;
    colorBuffer: Float32Array;
    linkBuffer: Uint32Array;
    particleCount: number;
    getNode(i: number) {
        const particleBuffer = this.nodeBuffer;
        return {
            x: particleBuffer[i * 4],
            y: particleBuffer[i * 4 + 1],
            z: particleBuffer[i * 4 + 2],
            w: particleBuffer[i * 4 + 3]
        }
    }
    boxMinX: number;
    boxMinY: number;
    boxMinZ: number;
    rootBoxSize: number;
    setBoxSize() {
        let xmin = 1000000,
            xmax = -1000000,
            ymin = 1000000,
            ymax = -1000000,
            zmin = 1000000,
            zmax = -1000000;

        for (let i = 0; i < this.particleCount; i++) {
            const { x, y, z } = this.getNode(i);
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
    maxLevel: number;
    /** All boxes at maxLevel. 1 << 3 * maxLevel */
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

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
        console.log(`maxLevel:${this.maxLevel} | BoxFull: ${this.numBoxIndexFull}`);
    };

    /**@return Array, box id for every partical*/
    morton(): Int32Array {
        const particleBuffer = this.nodeBuffer;
        const maxLevel = this.maxLevel;
        const particleCount = particleBuffer.length / 4;
        const resultIndex = new Int32Array(particleCount);
        const boxSize = this.rootBoxSize / (1 << maxLevel);
        for (let nodeIndex = 0; nodeIndex < particleCount; nodeIndex++) {
            const { x, y, z } = this.getNode(nodeIndex);
            let nx = Math.floor((x - this.boxMinX) / boxSize),
                ny = Math.floor((y - this.boxMinY) / boxSize),
                nz = Math.floor((z - this.boxMinZ) / boxSize);

            if (nx >= (1 << maxLevel)) nx--;
            if (ny >= (1 << maxLevel)) ny--;
            if (nz >= (1 << maxLevel)) nz--;
            let boxIndex = 0;
            for (let level = 0; level < maxLevel; level++) {
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
        //console.log(sortValue)

        const tempNodes = new Float32Array(this.nodeBuffer.length);
        const tempColor = new Float32Array(this.colorBuffer.length);
        const inverseSortIndex = new Uint32Array(sortIndex.length);
        for (let i = 0; i < this.particleCount; i++) {
            const offset = sortIndex[i] * 4;
            tempNodes.set(this.nodeBuffer.subarray(offset, offset + 4), i * 4);
            const offset3 = sortIndex[i] * 3;
            tempColor.set(this.colorBuffer.subarray(offset3, offset3 + 3), i * 3);

            // debug: viusalize box 
            //tempColor.set(sortValue[i] == 7 ? [1, 1, 1] : [0, 0, 0], i * 3);
            tempColor.set( [1, 1, 0], i * 3);


            inverseSortIndex[sortIndex[i]] = i;
        }

        for (let i = 0; i < this.linkBuffer.length; i++) {
            this.linkBuffer[i] = inverseSortIndex[this.linkBuffer[i]];
        }
        this.nodeBuffer.set(tempNodes);
        this.colorBuffer.set(tempColor);
    }
    /** non-empty FMM boxes @ maxLevel */
    numBoxIndexLeaf: number;
    /** numBoxIndexLeaf for all levels */
    numBoxIndexTotal: number;
    countNonEmptyBoxes(sortValue) {
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
    // interactionList: any;
    // /** int[numBoxIndexLeaf] */
    // boxOffsetStart: Int32Array;
    // /** int[numBoxIndexLeaf] */
    // boxOffsetEnd: Int32Array;
    allocate() {

        this.particleOffset = [0, 0].map(_ => new Array(this.numBoxIndexLeaf));
        this.boxIndexMask = new Int32Array(this.numBoxIndexFull);
        this.boxIndexFull = new Int32Array(this.numBoxIndexTotal);
        this.levelOffset = new Int32Array(this.maxLevel);

        this.numInteraction = new Int32Array(this.numBoxIndexLeaf);
        //this.interactionList
        // this.boxOffsetStart = new Int32Array(this.numBoxIndexLeaf);
        // this.boxOffsetEnd = new Int32Array(this.numBoxIndexLeaf);
    }


    getBoxData(mortonIndex) {
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

    constructor(particleBuffer: Float32Array, edgeBuffer: Uint32Array, colorBuffer: Float32Array) {
        this.nodeBuffer = particleBuffer;
        this.particleCount = particleBuffer.length / 4;
        this.linkBuffer = edgeBuffer;
        this.colorBuffer = colorBuffer;
        // if (colorBuffer.length / 3 != this.particleCount) { throw "Color buffer length not match: " + this.colorBuffer.length }

        this.setBoxSize();
        this.setOptimumLevel();
        this.sortParticles();

        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);

        this.countNonEmptyBoxes(sortValue);
        this.allocate();

        //     kernel.precalc();
        let numBoxIndex = this.getBoxData(mortonIndex);
        //console.log(this.particleOffset)
    }


}