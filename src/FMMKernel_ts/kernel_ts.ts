import { FMMSolver } from "../FMMSolver";
import { IFMMKernel } from "../FMMKernel";
import { debug_p2m } from "./debug_p2m";
import { debug_m2l_p4 } from "./debug_m2l";
import { debug_l2p } from "./debug_l2p";

/**
 * debug kernel for valiating FMM
 * TODO: Web Worker
 */
export class FMMKernel_ts implements IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    /**
     * index same as tree.boxIndexFull
     * Access: levelOffset[numLevel]+i
     */
    Mnm: Float32Array;
    Lnm: Float32Array;
    getMnmOffset(numLevel: number, index: number) {
        let i = this.core.tree.levelOffset[numLevel] + index;
        return i * 2;
    }
    async Init() {
        const core = this.core;
        const tree = core.tree;
        tree.boxIndexFull;
        this.Mnm = new Float32Array(core.MnmSize * 2 * tree.numBoxIndexTotal);
        this.Lnm = new Float32Array(core.MnmSize * 2 * tree.numBoxIndexTotal);

    }
    async p2p() {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            const target_count = core.interactionCounts[i];
            for (let j = 0; j < target_count; j++) {
                let target_box_id = core.interactionList[j];
                // To-do:
            }

        }
    }

    async p2m() {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            let boxMnm = debug_p2m(core, box_id);
            this.Mnm.set(boxMnm, this.getMnmOffset(tree.maxLevel - 1, i));
        }
    }
    async m2m(numLevel: number) {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
    }
    async m2l(numLevel: number) {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            const target_count = core.interactionCounts[i];
            const target_list = core.interactionList[i];
            let LnmOffset = this.getMnmOffset(numLevel, i);
            for (let j = 0; j < target_count; j++) {
                let target_box_id = target_list[j];
                let MnmOffset = this.getMnmOffset(numLevel, j);
                const boxMnm = this.Mnm.subarray(MnmOffset, MnmOffset + core.MnmSize * 2);
                const boxLnm = debug_m2l_p4(core, numLevel, boxMnm, target_box_id, box_id);
                boxLnm.forEach((v, k) => this.Lnm[LnmOffset + k] += v);
            }
        }
    }
    async l2l(numLevel: number) {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
    }
    async l2p() {
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            let LnmOffset = this.getMnmOffset(tree.maxLevel - 1, i);
            let boxLnm = this.Lnm.subarray(LnmOffset, LnmOffset + core.MnmSize * 2);
            const boxAccel = debug_l2p(core, boxLnm, box_id);
            let accelOffset = i * 3;
            boxAccel.forEach((v, k) => this.accelBuffer[accelOffset + k] += v);
        }
    }
    dataReady: boolean;
    Release: () => void;

    constructor(core: FMMSolver) {
        this.core = core;
    }
}