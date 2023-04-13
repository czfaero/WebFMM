import { FMMSolver } from './FMMSolver';

export class Tester {
    constructor() {
    }
    async Test(instance: FMMSolver) {
        instance.setBoxSize();
        instance.setOptimumLevel();
        console.log("maxLevel: " + instance.maxLevel);


        const mortonIndex = instance.morton();
        await VerifyIntBuffer("data-morton.bin", mortonIndex);
        const { sortValue, sortIndex } = instance.sort(mortonIndex);

        await VerifyIntBuffer("data-sort-index.bin", sortIndex);

        instance.sortParticles();

        await VerifyParticleBuffer("data-pos-pre-p2p.bin", instance.particleBuffer);

        instance.countNonEmptyBoxes();
        instance.allocate();
        let numLevel = instance.maxLevel;
        //     levelOffset[numLevel-1] = 0;
        //     kernel.precalc();
        let numBoxIndex = instance.getBoxData();
        //   // P2P
        instance.getInteractionListP2P(numBoxIndex, numLevel);
        //     bodyAccel.fill(0);
        //     kernel.p2p(numBoxIndex);
    }
}

function CompareNumber(a: number, b: number) {
    return Math.abs(a - b) < 0.00001
}
function p2str(data: Float32Array, i: number) {
    const xyzw = Array.from(data.subarray(i * 4, i * 4 + 4)).map(x => x.toFixed(2)).join(' ');
    return `[${i}] ${xyzw}`;
}
async function VerifyParticleBuffer(name: string, data: Float32Array) {
    const rawData = await (await fetch(name)).arrayBuffer();
    const expect = new Float32Array(rawData);

    console.log("bin size:" + expect.length)
    console.log(`[0] ${Array.from(expect.subarray(0, 4)).map(x => x.toFixed(2)).join(' ')}`)

    let error_count = 0;
    for (let i = 0; i < data.length; i++) {
        const r = CompareNumber(expect[i], data[i]);
        if (!r) {
            error_count++;
            const p_i = Math.floor(i / 4);
            console.log(`Expect: ${p2str(expect, p_i)}`);
            console.log(`Got   : ${p2str(data, p_i)}`)
            if (error_count > 10) break;
            i = p_i * 4 + 4;
        }
    }
    if (error_count == 0) {
        console.log("Success: " + name);
    }
    else {
        console.log(expect);
        console.log(data);
        throw "Failure: " + name;
    }
}

async function VerifyIntBuffer(name: string, data: Int32Array) {
    const rawData = await (await fetch(name)).arrayBuffer();
    const expect = new Int32Array(rawData);

    console.log("bin size:" + expect.length)
    console.log(`Sample: ${Array.from(expect.subarray(0, 4)).join(' ')}`)

    if (data.length != expect.length) {
        throw `${name} size: ${data.length}!=${expect.length}`;
    }
    let error_count = 0;
    for (let i = 0; i < data.length; i++) {
        const r = CompareNumber(expect[i], data[i]);
        if (!r) {
            error_count++;
            console.log(`[${i}]Expect: ${expect[i]} | Got: ${data[i]}`);
            if (error_count > 10) {
                break;
            }
        }
    }
    if (error_count == 0) {
        console.log("Success: " + name);
    }
    else {
        throw "Failure: " + name;
    }

}