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
    async Init() { }
    async p2p() {

    }
    async p2m() {

    }
    async m2m(numLevel: number) {

    }
    async m2l(numLevel: number) {

    }
    async l2l(numLevel: number) {

    }
    async l2p() {

    }
    Mnm: Array<Float32Array>;
    Lnm: Array<Float32Array>;
    dataReady: boolean;
    Release: () => void;
}