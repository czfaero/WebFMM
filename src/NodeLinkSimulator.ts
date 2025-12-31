import { FMMSolver } from './FMMSolver';
import { DirectSolver } from './DirectSolver';
import { TreeBuilder } from './TreeBuilder';
import { INBodySolver } from './INBodySolver';
import { debug_FindNaN, DebugMode } from './Debug';
import { INodeLinkDataProvider } from './INodeLinkDataProvider';


const k = 5;// spring force coef
const k_distance = 0.1;// distance when spring 0 force
const delta = 0.04;//F = accel * delta; 
const maxAccel = 15;

const stepMode = 0;
const maxIter = 100;

export class NodeLinkSimulator {
    debugMode: DebugMode;

    dataProvider: INodeLinkDataProvider;

    nodeBuffer: Float32Array;
    linkBuffer: Uint32Array;
    nodeColorBuffer: Float32Array;

    iterCount: number;
    stepNextFlag: boolean;
    solverErr: any;
    tree: TreeBuilder;
    solver: INBodySolver;
    pause: boolean;
    private ResetStatus() {
        this.iterCount = 0;
        this.stepNextFlag = false;
        this.solverErr = false;
        this.tree = null;
        this.solver = null;
        this.solverErr = null;
    }
    constructor(dataProvider: INodeLinkDataProvider) {
        this.dataProvider = dataProvider;
        this.ResetStatus();
    }
    ResetData() {
        const dataProvider = this.dataProvider;
        this.ResetStatus();
        this.nodeBuffer = dataProvider.GetNodes();
        this.linkBuffer = dataProvider.GetLinks();
        this.nodeColorBuffer = dataProvider.GetNodeColors();
    }
    SetDataProvider(dataProvider: INodeLinkDataProvider) {
        this.dataProvider = dataProvider;
        this.ResetData();
        this.ResetStatus();
    }

    Pause() {
        this.pause = true;
    }
    PauseToggle() {
        this.pause = !this.pause;
    }

    /**
     * 
     * @returns true if buffers has update
     */
    Update() {
        let hasUpdate = false;
        const _ = this;
        const nodeBuffer = _.nodeBuffer;
        const linkBuffer = _.linkBuffer;
        const nodeColorBuffer = _.nodeColorBuffer;

        // throw error from solver promise
        if (_.solverErr) {
            throw _.solverErr;
        }

        // start new solver
        if (
            (!stepMode || _.stepNextFlag)
            && (!_.pause)
        ) {
            if (_.solver == null) {
                _.stepNextFlag = false;
                if (_.iterCount > maxIter) { _.solver = null; return; }
                _.tree = new TreeBuilder(nodeBuffer, linkBuffer, nodeColorBuffer);

                // solver = new DirectSolver(tree);
                _.solver = new FMMSolver(_.tree, "wgsl");
                _.solver.debugMode = _.debugMode;
                _.solver.kernel.debugMode = _.debugMode;
                (_.solver as FMMSolver).iterCount = _.iterCount;
                _.solver.main().catch(e => {
                    _.solverErr = e;
                });
                _.iterCount++;
            }
        }
        // process accel
        if (_.solver && _.solver.isDataReady()) {
            const accelBuffer = _.solver.getAccelBuffer();
            //console.log(accelBuffer)
            if (accelBuffer.length == 0) { throw "accelbuffer error" }
            for (let i = 0; i < linkBuffer.length; i += 2) {
                const i1 = linkBuffer[i], i2 = linkBuffer[i + 1];
                const p1 = GetPoint(i1, nodeBuffer),
                    p2 = GetPoint(i2, nodeBuffer);
                const dist = GetDist(p1, p2);
                const l_sqr = GetSquareLength(dist);
                const l = Math.sqrt(l_sqr);
                const l_ = l - k_distance;
                const normalized = { x: dist.x / l, y: dist.y / l, z: dist.z / l };
                const dist2 = { x: normalized.x * l_, y: normalized.y * l_, z: normalized.z * l_ }

                if (isNaN(dist2.x) || isNaN(accelBuffer[i1 * 3])) {
                    if (_.debugMode == DebugMode.debugger) {
                        debugger;
                        _.Log("Aborted at iter " + _.iterCount);
                        const err = { type: "abort"};
                        throw err;
                    }
                    else if (_.debugMode == DebugMode.retry_at_end) {
                        const err = { type: "retry", info: "retry at simulator" };
                        throw err;
                    }
                    else {
                        _.Log("Aborted at iter " + _.iterCount);
                        const err = { type: "abort" };
                        throw err;
                    }

                }
                accelBuffer[i1 * 3] += dist2.x * k;
                accelBuffer[i1 * 3 + 1] += dist2.y * k;
                accelBuffer[i1 * 3 + 2] += dist2.z * k;
                accelBuffer[i2 * 3] -= dist2.x * k;
                accelBuffer[i2 * 3 + 1] -= dist2.y * k;
                accelBuffer[i2 * 3 + 2] -= dist2.z * k;
            }

            for (let i = 0; i < nodeBuffer.length / 4; i++) {
                let x = accelBuffer[i * 3 + 0];
                let y = accelBuffer[i * 3 + 1];
                let z = accelBuffer[i * 3 + 2];
                let l = Math.sqrt(x * x + y * y + z * z);
                if (l > maxAccel) {
                    x = x / l * maxAccel;
                    y = y / l * maxAccel;
                    z = z / l * maxAccel;
                }

                nodeBuffer[i * 4 + 0] += x * delta;
                nodeBuffer[i * 4 + 1] += y * delta;
                nodeBuffer[i * 4 + 2] += z * delta;
            }

            hasUpdate = true;
            _.solver.Destroy();
            _.solver = null;
            _.Log("iter: " + _.iterCount);
            // if (iterCount == 1) { RecordVideo() }
        }

        return hasUpdate;
    }
    Step() {
        this.stepNextFlag = false;
    }
    log_func: Function;
    Log(str: string) {
        if (this.log_func) {
            this.log_func(str);
        }
    }

}







function GetPoint(i: number, buffer: Float32Array) {
    return {
        x: buffer[i * 4],
        y: buffer[i * 4 + 1],
        z: buffer[i * 4 + 2],
        w: buffer[i * 4 + 3]
    }
}
function GetDist(p1, p2) {
    return {
        x: p2.x - p1.x,
        y: p2.y - p1.y,
        z: p2.z - p1.z
    };
}
function GetSquareLength(vec) {
    return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}
