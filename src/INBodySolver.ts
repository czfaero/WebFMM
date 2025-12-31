import { DebugMode } from "./Debug";
import { IFMMKernel } from "./IFMMKernel";


export interface INBodySolver {
    isDataReady: () => boolean;
    getAccelBuffer: () => Float32Array;
    main: () => Promise<void>;
    Destroy: () => void;
    // constructor(tree) {} // how to define this?


    // for debug
    debug_watch_box_id_pairs?: any;
    debugMode?: DebugMode;
    debugInfo?: any;
    kernel?: IFMMKernel;
}
