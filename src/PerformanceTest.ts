import { DirectSolver } from "./DirectSolver";
import { FMMSolver } from "./FMMSolver";
import { INBodySolver } from "./INBodySolver";
import { TreeBuilder } from "./TreeBuilder";

export class PerformanceTestTask {
    type: string;
    data: TreeBuilder;
    onTaskStart: Function;
    onTaskEnd: Function;
    constructor(type: string, data: TreeBuilder, onTaskStart = null, onTaskEnd = null) {
        this.type = type;
        this.data = data;
        this.onTaskStart = onTaskStart;
        this.onTaskEnd = onTaskEnd;
    }
    accelBuffer: Float32Array;
    debugInfo: any;
    time: number;
    async Proc() {
        if (this.onTaskStart) { this.onTaskStart(this) }
        let solver: INBodySolver;
        switch (this.type) {
            case "direct-cpu":
                solver = new DirectSolver(this.data, false); break;
            case "direct-gpu":
                solver = new DirectSolver(this.data, true); break;
            case "fmm-gpu":
                solver = new FMMSolver(this.data, "wgsl"); break;
            case "fmm-cpu":
                solver = new FMMSolver(this.data, "ts"); break;
            default: throw "Unknown Task: " + this.type;
        }
        await solver.main();
        if (!solver.isDataReady()) { throw "?"; }
        this.accelBuffer = solver.getAccelBuffer();
        this.debugInfo = solver.debugInfo;
        this.time = solver.debugInfo.find(x => x.step == "total").time;
        if (this.onTaskEnd) { this.onTaskEnd(this) }
    }
}

// const E2 = debug_getError(accelBuffer4, accelBuffer3);
// console.log("Error: FMM GPU / Direct CPU", E2);

// function debug_getError(baseBuffer, testBuffer) {
//     let upper = 0;
//     let under = 0;
//     for (let i = 0; i < baseBuffer.length; i++) {
//         const bv = baseBuffer[i], tv = testBuffer[i];
//         const e = bv - tv;
//         upper += e * e;
//         under += bv * bv;
//     }
//     return Math.sqrt(upper / under)
// }
