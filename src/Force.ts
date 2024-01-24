import { FMMSolver } from './FMMSolver';
import { DirectSolver } from './DirectSolver';

const k = 2;// spring
const delta = 0.1;//delta time ^2

var solver: any;
var next = false;
const stepMode = false;
const maxIter = 1000;
const msg = document.querySelector("#msg") as HTMLSpanElement;

export function DataStart(nodeBuffer: Float32Array,
    linkBuffer: Uint32Array) {
    let button = document.querySelector("#button_next") as HTMLButtonElement;
    button.onclick = function () {
        next = true;
    }


    //solver = new FMMSolver(nodeBuffer, "wgpu");
    // solver = new DirectSolver(nodeBuffer);
    // solver.main();

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
var iterCount = 0;
export function DataUpdate(
    nodeBuffer: Float32Array,
    linkBuffer: Uint32Array,
    nodeBufferGPU: GPUBuffer,
    linkBufferGPU: GPUBuffer,
    device: GPUDevice
) {


    if (solver && solver.isDataReady()) {
        const accelBuffer = solver.getAccelBuffer();
        console.log(accelBuffer)
        if (accelBuffer.length == 0) { throw "accelbuffer error" }
        for (let i = 0; i < linkBuffer.length; i += 2) {
            const i1 = linkBuffer[i], i2 = linkBuffer[i + 1];
            const p1 = GetPoint(i1, nodeBuffer),
                p2 = GetPoint(i2, nodeBuffer);
            const dist = GetDist(p1, p2);
            accelBuffer[i1 * 3] += dist.x * k;
            accelBuffer[i1 * 3 + 1] += dist.y * k;
            accelBuffer[i1 * 3 + 2] += dist.z * k;
            accelBuffer[i2 * 3] -= dist.x * k;
            accelBuffer[i2 * 3 + 1] -= dist.y * k;
            accelBuffer[i2 * 3 + 2] -= dist.z * k;
        }

        for (let i = 0; i < nodeBuffer.length / 4; i++) {
            nodeBuffer[i * 4 + 0] += accelBuffer[i * 3 + 0] * delta;
            nodeBuffer[i * 4 + 1] += accelBuffer[i * 3 + 1] * delta;
            nodeBuffer[i * 4 + 2] += accelBuffer[i * 3 + 2] * delta;
        }


        device.queue.writeBuffer(nodeBufferGPU, 0, nodeBuffer);
        solver = null;
        msg.innerHTML = "iter: " + iterCount;
    }
    if (!stepMode || next) {
        next = false;
        if (iterCount > maxIter) { solver = null; return; }
        //solver = new FMMSolver(nodeBuffer, "wgpu");
        solver = new DirectSolver(nodeBuffer);
        solver.main();
        iterCount++;
    }


    //solver.kernel.debug = true;



}