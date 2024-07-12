import { FMMSolver } from "../FMMSolver";

export interface IKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    Init: (particleBuffer: Float32Array) => Promise<void>;
    p2p: (numInteraction,interactionList) => Promise<void>;
    p2m: () => Promise<void>;
    m2m: (numBoxIndex: number, numBoxIndexOld: number, numLevel: number) => Promise<void>;
    m2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2p: (numBoxIndex: number) => Promise<void>;
    Mnm: Array<Float32Array>;
    Lnm: Array<Float32Array>;
    dataReady: boolean;
    Release: () => void;
}
