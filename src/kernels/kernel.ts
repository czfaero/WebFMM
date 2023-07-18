import { FMMSolver } from "../FMMSolver";

export interface IKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    Init: (particleBuffer: Float32Array) => Promise<void>;
    p2p: (numBoxIndex: number, interactionList: any, numInteraction: any, particleOffset: any) => Promise<void>;
    p2m: (numBoxIndex: number, particleOffset: any) => Promise<void>;
    Mnm: Array<Float32Array>;
}
