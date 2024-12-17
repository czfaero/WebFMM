

export class TreeBuilder {
    nodeBuffer: Float32Array;
    colorBuffer: Float32Array;
    linkBuffer: Uint32Array;
    nodeCount: number;
    getNode(i: number) {
        const nodeBuffer = this.nodeBuffer;
        return {
            x: nodeBuffer[i * 4],
            y: nodeBuffer[i * 4 + 1],
            z: nodeBuffer[i * 4 + 2],
            w: nodeBuffer[i * 4 + 3]
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

        for (let i = 0; i < this.nodeCount; i++) {
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
        this.rootBoxSize *= 1.00001; // Keep node on the edge from falling out

    };
    maxLevel: number;
    /** All boxes at maxLevel. 1 << 3 * maxLevel */
    numBoxIndexFull: number;
    setOptimumLevel() {
        // 按照点的数量区间定级别
        const level_switch = [1e5, 7e5, 7e6, 5e7, 3e8, 2e9]; // gpu-fmm

        this.maxLevel = 2;
        for (const level of level_switch) {
            if (this.nodeCount >= level) {
                this.maxLevel++;
            } else {
                break;
            }
        }

        this.numBoxIndexFull = 1 << 3 * this.maxLevel;
    };

    /**@return Array, box id for every partical*/
    morton(): Int32Array {
        const nodeBuffer = this.nodeBuffer;
        const maxLevel = this.maxLevel;
        const nodeCount = nodeBuffer.length / 4;
        const resultIndex = new Int32Array(nodeCount);
        const boxSize = this.rootBoxSize / (1 << maxLevel);
        for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
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
        const sortValue = new Int32Array(this.nodeCount);
        const sortIndex = new Int32Array(this.nodeCount);
        for (let i = 0; i < this.nodeCount; i++) {
            sortIndex[i] = i;
        }
        tempSortIndex.fill(0);
        for (const i in mortonIndex) {
            tempSortIndex[mortonIndex[i]]++;
        }
        for (let i = 1; i < this.numBoxIndexFull; i++) {
            tempSortIndex[i] += tempSortIndex[i - 1];
        }
        for (let i = this.nodeCount - 1; i >= 0; i--) {
            tempSortIndex[mortonIndex[i]]--;
            sortValue[tempSortIndex[mortonIndex[i]]] = mortonIndex[i];
            sortIndex[tempSortIndex[mortonIndex[i]]] = i;
        }
        return { sortValue, sortIndex }
    }
    sortNodes() {
        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);
        //console.log(sortValue)

        const tempNodes = new Float32Array(this.nodeBuffer.length);
        const tempColor = new Float32Array(this.colorBuffer.length);
        const inverseSortIndex = new Uint32Array(sortIndex.length);
        for (let i = 0; i < this.nodeCount; i++) {
            const offset = sortIndex[i] * 4;
            tempNodes.set(this.nodeBuffer.subarray(offset, offset + 4), i * 4);
            const offset3 = sortIndex[i] * 3;
            tempColor.set(this.colorBuffer.subarray(offset3, offset3 + 3), i * 3);

            // debug: viusalize box 
            //tempColor.set(sortValue[i] == 7 ? [1, 1, 1] : [0, 0, 0], i * 3);
            tempColor.set([1, 1, 0], i * 3);


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
    /** non-empty FMM boxes (numBoxIndexLeaf) for all levels */
    numBoxIndexTotal: number;
    countNonEmptyBoxes(sortValue) {
        this.numBoxIndexLeaf = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.nodeCount; i++) {
            if (sortValue[i] != currentIndex) {
                this.numBoxIndexLeaf++;
                currentIndex = sortValue[i];
            }
        }

        this.numBoxIndexTotal = this.numBoxIndexLeaf;
        for (let numLevel = this.maxLevel - 1; numLevel >= 2; numLevel--) {
            currentIndex = -1;
            for (let i = 0; i < this.nodeCount; i++) {
                const temp = Math.floor(sortValue[i] / (1 << 3 * (this.maxLevel - numLevel)));
                //console.log("temp-index",temp-currentIndex)
                if (temp != currentIndex) {

                    this.numBoxIndexTotal++;
                    currentIndex = temp;
                }
            }
        }
    }
    /** int[maxLevel]. 
     * From large box to small box (maxLevel).   
     * 1<<numLevel*3 = boxCount
     */
    levelOffset: Int32Array;
    /**
     * first and last node in each box, by non-empty id
     * int[2][numBoxIndexLeaf]
     */
    nodeOffset: any;
    /** int[maxLevel][numBoxIndexFull]; link list for box index : Full -> NonEmpty */
    boxIndexMaskBuffers: Array<Int32Array>;
    /** int[numBoxIndexTotal];  
     *  link list for box index : NonEmpty -> Full 
     *  Access: levelOffset[numLevel]+i
     *  Size: non-empty FMM boxes @ all levels
     */
    boxIndexFull: Int32Array;

    allocate() {

        this.nodeOffset = [0, 0].map(_ => new Array(this.numBoxIndexLeaf));
        this.boxIndexMaskBuffers = new Array(this.maxLevel);
        this.boxIndexFull = new Int32Array(this.numBoxIndexTotal);
        this.levelOffset = new Int32Array(this.maxLevel);
        // this.boxOffsetStart = new Int32Array(this.numBoxIndexLeaf);
        // this.boxOffsetEnd = new Int32Array(this.numBoxIndexLeaf);
    }


    /**
     * Set nodeOffset boxIndexFull boxIndexMask for numLevel=maxLevel-1
     */
    initBoxData(mortonIndex) {
        let boxIndexMask = new Int32Array(this.numBoxIndexFull);
        boxIndexMask.fill(-1);
        let numBoxIndex = 0;
        let currentIndex = -1;
        for (let i = 0; i < this.nodeCount; i++) {
            if (mortonIndex[i] != currentIndex) {
                boxIndexMask[mortonIndex[i]] = numBoxIndex;
                this.boxIndexFull[numBoxIndex] = mortonIndex[i];
                this.nodeOffset[0][numBoxIndex] = i;
                if (numBoxIndex > 0) this.nodeOffset[1][numBoxIndex - 1] = i - 1;
                currentIndex = mortonIndex[i];
                numBoxIndex++;
            }
        }
        this.nodeOffset[1][numBoxIndex - 1] = this.nodeCount - 1;
        this.boxIndexMaskBuffers[this.maxLevel - 1] = boxIndexMask;
        this.levelOffset[this.maxLevel - 2] = numBoxIndex;// next level
    }

    // Set nodeOffset boxIndexFull boxIndexMask for numLevel<maxLevel-1
    // numLevel: max-2 => 1
    initBoxDataOfParent(numLevel: number) {
        //console.log(`getBoxDataOfParent ${_numBoxIndex} ${numLevel}`)
        const tree = this;
        let boxIndexMask = new Int32Array(1 << numLevel * 3);
        boxIndexMask.fill(-1);

        let numBoxIndex = 0;
        let currentIndex = -1;
        const end = tree.levelOffset[numLevel];
        for (let boxIndex = tree.levelOffset[numLevel + 1]; boxIndex < end; boxIndex++) {
            if (currentIndex != Math.floor(tree.boxIndexFull[boxIndex] / 8)) {
                currentIndex = Math.floor(tree.boxIndexFull[boxIndex] / 8);
                boxIndexMask[currentIndex] = numBoxIndex;
                tree.boxIndexFull[numBoxIndex + tree.levelOffset[numLevel]] = currentIndex;
                numBoxIndex++;
            }
        }
        tree.levelOffset[numLevel - 1] = tree.levelOffset[numLevel] + numBoxIndex; // next level
        this.boxIndexMaskBuffers[numLevel] = boxIndexMask;
    }
    constructor(nodeBuffer: Float32Array, linkBuffer: Uint32Array, colorBuffer: Float32Array) {
        this.nodeBuffer = nodeBuffer;
        this.nodeCount = nodeBuffer.length / 4;
        this.linkBuffer = linkBuffer;
        this.colorBuffer = colorBuffer;

        this.setBoxSize();
        this.setOptimumLevel();
        this.sortNodes();

        const mortonIndex = this.morton();
        const { sortValue, sortIndex } = this.sort(mortonIndex);

        this.countNonEmptyBoxes(sortValue);
        this.allocate();

        this.levelOffset[this.maxLevel - 1] = 0;
        this.initBoxData(mortonIndex);

        for (let numLevel = this.maxLevel - 2; numLevel >= 1; numLevel--) {
            this.initBoxDataOfParent(numLevel);
        }

        console.log(`-- Tree info --
nodeCount                     : ${this.nodeCount}
maxLevel                      : ${this.maxLevel} 
BoxIndexFull=1<<3*maxLevel    : ${this.numBoxIndexFull}
BoxIndexLeaf={non-empty@max}  : ${this.numBoxIndexLeaf}
BoxIndexTotal={non-empty@all} : ${this.numBoxIndexTotal}
levelOffset: ${this.levelOffset}
  numLevel -> numBoxIndex of max
  0 -> ? of 8 (fixed)
${Array.from(this.levelOffset)
                .map((x, i) => { return i < this.maxLevel - 1 ? `  ${i + 1} -> ${x - this.levelOffset[i + 1]} of ${1 << 3 * (i + 2)}` : "" })
                .join("\n")}
`);
        console.log(this);

    }
    debug_watch: any;

    // for wgpu debug
    // box id by non-empty
    debug_restrict_nodes(box_ids: Array<number>, inbox_indexs: Array<number> = []) {
        console.log("debug: restrict nodes");
        this.debug_watch = [];

        for (let i = 0; i < this.numBoxIndexLeaf; i++) {
            if (box_ids.includes(i)) {
                if (inbox_indexs != null && inbox_indexs.length > 0) {
                    for (let j = this.nodeOffset[0][i]; j <= this.nodeOffset[1][i]; j++) {
                        if (inbox_indexs.includes(j - this.nodeOffset[0][i])) {
                            this.nodeBuffer[j * 4 + 3] = 1;
                            this.debug_watch.push({ index: j, box: i, value: this.getNode(j) });
                        } else {
                            this.nodeBuffer[j * 4 + 3] = 0;
                        }
                    }
                } else {
                    for (let j = this.nodeOffset[0][i]; j <= this.nodeOffset[1][i]; j++) {
                        this.nodeBuffer[j * 4 + 3] = 1;
                    }
                }

                continue;
            }
            for (let j = this.nodeOffset[0][i]; j <= this.nodeOffset[1][i]; j++) {
                this.nodeBuffer[j * 4 + 3] = 0;
            }
        }
        console.log(this.debug_watch);

    }


}