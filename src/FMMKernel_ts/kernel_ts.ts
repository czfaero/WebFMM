import { FMMSolver } from "../FMMSolver";
import { IFMMKernel } from "../FMMKernel";

/**
 * debug kernel for valiating FMM
 * TODO: Web Worker
 */
export class FMMKernel_ts implements IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    async Init() {}
    async p2p(numInteraction, interactionList){

    }
    p2m: () => Promise<void>;
    m2m: (numLevel: number) => Promise<void>;
    m2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2p: (numBoxIndex: number) => Promise<void>;
    Mnm: Array<Float32Array>;
    Lnm: Array<Float32Array>;
    dataReady: boolean;
    Release: () => void;
}