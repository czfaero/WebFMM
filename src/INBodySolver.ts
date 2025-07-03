import { DebugMode } from "./Debug";


export interface INBodySolver {
    isDataReady: () => boolean;
    getAccelBuffer: () => Float32Array;
    main: () => Promise<void>;
    // constructor(tree) {} // how to define this?


    // for debug
    debug_watch_box_id_pairs?: any;
    debugMode?: DebugMode;
    debug_info?: any;
    kernel?: any;
}
