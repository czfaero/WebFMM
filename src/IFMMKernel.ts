import { FMMSolver } from "./FMMSolver";

export interface IFMMKernel {
    core: FMMSolver;
    debug: boolean;
    accelBuffer: Float32Array;
    Init: () => Promise<void>;
    p2p: () => Promise<void>;
    p2m: () => Promise<void>;
    /**
     * 
     * @param numLevel the higher (smaller) level, dst level
     * @returns 
     */
    m2m: (numLevel: number) => Promise<void>;
    /**
     * 
     * @param numLevel level
     * @returns 
     */
    m2l: (numLevel: number) => Promise<void>;
    /**
     * 
     * @param numLevel the higher (smaller) level, src level
     * @returns 
     */
    l2l: (numLevel: number) => Promise<void>;
    l2p: () => Promise<void>;
    Release: () => void;
    // constructor(core: FMMSolver) {this.core = core;} // how to define this?
}
