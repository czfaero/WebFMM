import { FMMSolver } from "../FMMSolver";
import { IFMMKernel } from "../IFMMKernel";
import { debug_p2m } from "./debug_p2m";
import { debug_m2l_p4 } from "./debug_m2l";
import { debug_l2p } from "./debug_l2p";
import { debug_m2m_p4 } from "./debug_m2m";
import { debug_l2l_p4 } from "./debug_l2l";
import { debug_p2p } from "./debug_p2p";
import { DebugMode } from "../Debug";

/**
 * debug kernel for valiating FMM
 * TODO: Web Worker
 */
export class FMMKernel_ts implements IFMMKernel {
    core: FMMSolver;
    debugMode:DebugMode;
    accelBuffer: Float32Array;
    /**
     * index same as tree.boxIndexFull
     * Access: levelOffset[numLevel]+i
     */
    Mnm: Float32Array;
    Lnm: Float32Array;
    debug_info: any;
    getMnmOffset(numLevel: number, index: number) {
        let i = this.core.tree.levelOffset[numLevel] + index;
        return i * this.core.MnmSize * 2;// complex number
    }
    constructor(core: FMMSolver) {
        this.core = core;
        this.debug_info = [];
    }
    async Init() {
        const core = this.core;
        const tree = core.tree;
        this.accelBuffer = new Float32Array(tree.nodeCount * 3);
        this.Mnm = new Float32Array(core.MnmSize * 2 * tree.numBoxIndexTotal);
        this.Lnm = new Float32Array(core.MnmSize * 2 * tree.numBoxIndexTotal);

    }
    async p2p() {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let dst_box_id = i + offset;
            const src_count = core.interactionCounts[i];
            const src_list = core.interactionList[i];
            for (let j = 0; j < src_count; j++) {
                let src_box_id = src_list[j];
                const boxAccel = debug_p2p(core, src_box_id, dst_box_id);
                let accelOffset = tree.nodeStartOffset[dst_box_id] * 3;
                boxAccel.forEach((v, k) => this.accelBuffer[accelOffset + k] += v);
            }
        }
        this.debug_info.push({ step: "P2P", time: performance.now() - time });
    }

    async p2m() {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            let boxMnm = debug_p2m(core, box_id);
            let MnmOffset = this.getMnmOffset(tree.maxLevel - 1, i);
            this.Mnm.set(boxMnm, MnmOffset);
        }
        this.debug_info.push({ step: "P2M", time: performance.now() - time });
    }
    async m2m(numLevel: number) {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        const src_mask = tree.boxIndexMaskBuffers[numLevel + 1];
        for (let i = 0; i < boxCount; i++) {
            let dst_box_id = i + offset;
            let dst_index = core.tree.boxIndexFull[dst_box_id];
            let dst_boxMnm_total = new Float32Array(core.MnmSize * 2);
            for (let j = 0; j < 8; j++) {
                let src_index = dst_index * 8 + j;
                let src_box_id = src_mask[src_index];
                if (src_box_id != -1) {
                    let MnmOffset = this.getMnmOffset(numLevel + 1, src_box_id);
                    const src_boxMnm = this.Mnm.subarray(MnmOffset, MnmOffset + core.MnmSize * 2);
                    let dst_boxMnm = debug_m2m_p4(core, numLevel, src_boxMnm, src_box_id, dst_box_id);
                    dst_boxMnm.forEach((v, i) => dst_boxMnm_total[i] += v);
                }
            }
            this.Mnm.set(dst_boxMnm_total, this.getMnmOffset(numLevel, i));

        }
        this.debug_info.push({ step: `M2M@${numLevel + 1}->${numLevel}`, time: performance.now() - time });
    }
    async m2l(numLevel: number) {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        let debug_interactionCount = 0;
        for (let i = 0; i < boxCount; i++) {
            let dst_box_id = i + offset;
            const target_count = core.interactionCounts[i];
            const src_list = core.interactionList[i];
            let LnmOffset = this.getMnmOffset(numLevel, i);
            debug_interactionCount += target_count;
            for (let j = 0; j < target_count; j++) {
                let src_box_id = src_list[j];
                let MnmOffset = this.getMnmOffset(numLevel, src_box_id);
                const boxMnm = this.Mnm.subarray(MnmOffset, MnmOffset + core.MnmSize * 2);
                const boxLnm = debug_m2l_p4(core, numLevel, boxMnm, src_box_id, dst_box_id);
                boxLnm.forEach((v, k) => this.Lnm[LnmOffset + k] += v);
            }

        }
        this.debug_info.push({
            step: `M2L@${numLevel}`,
            time: performance.now() - time,
            interactionCount: debug_interactionCount
        });
    }
    async l2l(numLevel: number) {
        const time = performance.now();

        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[numLevel]; //non-empty
        const offset = tree.levelOffset[numLevel];
        const dst_mask = tree.boxIndexMaskBuffers[numLevel + 1];
        for (let i = 0; i < boxCount; i++) {
            let src_box_id = i + offset;
            let src_index = core.tree.boxIndexFull[src_box_id];
            let src_LnmOffset = this.getMnmOffset(numLevel, i);
            const src_boxLnm = this.Lnm.subarray(src_LnmOffset, src_LnmOffset + core.MnmSize * 2);

            for (let j = 0; j < 8; j++) {
                let dst_index = src_index * 8 + j;
                let dst_box_id = dst_mask[dst_index];
                if (dst_box_id != -1) {
                    let dst_LnmOffset = this.getMnmOffset(numLevel + 1, dst_box_id);
                    let dst_boxLnm = debug_l2l_p4(core, numLevel, src_boxLnm, src_box_id, dst_box_id);
                    dst_boxLnm.forEach((v, i) => this.Lnm[dst_LnmOffset + i] += v);
                }
            }
        }
        this.debug_info.push({ step: `L2L@${numLevel}->${numLevel + 1}`, time: performance.now() - time });
    }
    async l2p() {
        const time = performance.now();
        const core = this.core;
        const tree = core.tree;
        const boxCount = tree.levelBoxCounts[tree.maxLevel - 1]; //non-empty
        const offset = tree.levelOffset[tree.maxLevel - 1]; // should be 0
        for (let i = 0; i < boxCount; i++) {
            let box_id = i + offset;
            let LnmOffset = this.getMnmOffset(tree.maxLevel - 1, i);
            let boxLnm = this.Lnm.subarray(LnmOffset, LnmOffset + core.MnmSize * 2);
            const boxAccel = debug_l2p(core, boxLnm, box_id);
            let accelOffset = tree.nodeStartOffset[box_id] * 3;
            boxAccel.forEach((v, k) => this.accelBuffer[accelOffset + k] += v);
        }
        this.debug_info.push({ step: `L2P`, time: performance.now() - time });
    }

    Release() { }
}