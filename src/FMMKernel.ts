import { FMMSolver } from "./FMMSolver";

export interface IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    Init: () => Promise<void>;
    p2p: (numInteraction, interactionList) => Promise<void>;
    p2m: () => Promise<void>;
    m2m: (numLevel: number) => Promise<void>;
    m2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2l: (numBoxIndex: number, numLevel: number) => Promise<void>;
    l2p: (numBoxIndex: number) => Promise<void>;
    dataReady: boolean;
    Release: () => void;
}
