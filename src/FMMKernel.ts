import { FMMSolver } from "./FMMSolver";

export interface IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    Init: () => Promise<void>;
    p2p: () => Promise<void>;
    p2m: () => Promise<void>;
    m2m: (numLevel: number) => Promise<void>;
    m2l: (numLevel: number) => Promise<void>;
    l2l: (numLevel: number) => Promise<void>;
    l2p: () => Promise<void>;
    dataReady: boolean;
    Release: () => void;
}
