

export class TreeBuilder {
    particleBuffer: Float32Array;
    edgeBuffer: Uint32Array;
    particleCount: number;
    getParticle(i: number) {
        const particleBuffer = this.particleBuffer;
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
        console.log("maxLevel: " + this.maxLevel);

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
    };

    /**@return Array, box id for every partical*/
    morton(): Int32Array {
        const particleBuffer = this.particleBuffer;
        const maxLevel = this.maxLevel;
        const particleCount = particleBuffer.length / 4;
        const resultIndex = new Int32Array(particleCount);
        const boxSize = this.rootBoxSize / (1 << maxLevel);
        for (let nodeIndex = 0; nodeIndex < particleCount; nodeIndex++) {
            const { x, y, z } = this.getParticle(nodeIndex);
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

        const tempParticle = new Float32Array(this.particleBuffer.length);
        for (let i = 0; i < this.particleCount; i++) {
            const offset = sortIndex[i] * 4;
            tempParticle.set(this.particleBuffer.subarray(offset, offset + 4), i * 4);
        }
        this.particleBuffer = tempParticle;
    }

    constructor(particleBuffer: Float32Array, edgeBuffer: Uint32Array) {
        this.particleBuffer = particleBuffer;
        this.particleCount = particleBuffer.length / 4;
        this.edgeBuffer = edgeBuffer;
    }


    main() {
        this.setBoxSize();
        this.setOptimumLevel();
        this.sortParticles();
    }

}