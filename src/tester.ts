import { FMMSolver } from './FMMSolver';

export class Tester {
    constructor() {
    }
    async Test(instance: FMMSolver) {
        console.log(instance);
        instance.kernel.debug = true;
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
        console.log(`numBoxIndexLeaf: ${instance.numBoxIndexLeaf} | numBoxIndexTotal: ${instance.numBoxIndexTotal}`);
        instance.allocate();
        let numLevel = instance.maxLevel;
        instance.levelOffset[numLevel - 1] = 0;
        //     kernel.precalc();
        let numBoxIndex = instance.getBoxData();
        //   // P2P
        instance.getInteractionListP2P(numBoxIndex, numLevel);

        await VerifyIntIntBuffer("data-p2p-list.bin", instance.interactionList);

        await instance.kernel.Init(instance.particleBuffer);

        let time_p2p = performance.now();
        await instance.kernel.p2p(numBoxIndex, instance.interactionList, instance.numInteraction, instance.particleOffset);
        time_p2p = performance.now() - time_p2p;
        console.log(`time p2p: ${time_p2p.toFixed(2)} ms`);

        await VerifyFloatBuffer("data-p2p.bin", instance.kernel.accelBuffer);

        console.log("p2m numBoxIndex: " + numBoxIndex);
        let time_p2m = performance.now();
        await instance.kernel.p2m(numBoxIndex, instance.particleOffset);
        time_p2m = performance.now() - time_p2m;
        console.log(`time p2m: ${time_p2m.toFixed(2)} ms`);
        await VerifyFloatBuffer2("data-p2m.bin", instance.kernel.Mnm, numBoxIndex);

        if (instance.maxLevel > 2) {
            for (numLevel = instance.maxLevel - 1; numLevel >= 2; numLevel--) {
                let numBoxIndexOld = numBoxIndex;
                numBoxIndex = instance.getBoxDataOfParent(numBoxIndex, numLevel);
                instance.kernel.m2m(numBoxIndex, numBoxIndexOld, numLevel);
            }
            numLevel = 2;
        }
        else {
            instance.getBoxIndexMask(numBoxIndex, numLevel);
        }
        await VerifyFloatBuffer2("data-m2m.bin", instance.kernel.Mnm, instance.numBoxIndexTotal);
        instance.getInteractionListM2L(numBoxIndex, numLevel);
        await instance.kernel.m2l(numBoxIndex, numLevel);

        await VerifyFloatBuffer2("data-m2l.bin", instance.kernel.Lnm, numBoxIndex);
        if (instance.maxLevel > 2) {

            for (numLevel = 3; numLevel <= instance.maxLevel; numLevel++) {

                console.log(`level ${numLevel} : l2l m2l`);
                numBoxIndex = instance.levelOffset[numLevel - 2] - instance.levelOffset[numLevel - 1];

                instance.kernel.l2l(numBoxIndex, numLevel);

                instance.getBoxIndexMask(numBoxIndex, numLevel);

                instance.getInteractionListM2LLower(numBoxIndex, numLevel);

                instance.kernel.m2l(numBoxIndex, numLevel);
            }
            numLevel = instance.maxLevel;
            await VerifyFloatBuffer2("data-l2l.bin", instance.kernel.Lnm, numBoxIndex);
        }

        instance.kernel.l2p(numBoxIndex);
        await VerifyFloatBuffer("data-l2p.bin", instance.kernel.accelBuffer,0.05);
    }
}

function CompareNumber(a: number, b: number, delta = 0.002) {
    return Math.abs(a - b) < delta
}
function Error(a: number, b: number) {
    return Math.abs(a - b)
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

async function VerifyFloatBuffer(name: string, data: Float32Array, max_error = 0.001) {
    const rawData = await (await fetch(name)).arrayBuffer();
    const expect = new Float32Array(rawData);

    console.log("bin size:" + expect.length)
    console.log(`Sample expect: ${Array.from(expect.subarray(0, 4)).map(n => n.toFixed(3)).join(' ')}`)
    console.log(`Sample buffer: ${Array.from(data.subarray(0, 4)).map(n => n.toFixed(3)).join(' ')}`)

    if (data.length != expect.length) {
        console.log(data);
        throw `${name} size: ${data.length}!=${expect.length}`;
    }
    let error_count = 0;
    for (let i = 0; i < data.length; i++) {
        const r = CompareNumber(expect[i], data[i], max_error);
        if (!r) {
            error_count++;
            console.log(`[${i}]Expect: ${expect[i]} | Got: ${data[i]} |${expect[i] - data[i]}`);
            if (error_count > 10) {
                break;
            }
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

// p2m
async function VerifyFloatBuffer2(name: string, data: Array<Float32Array>, count: number) {
    const rawData = await (await fetch(name)).arrayBuffer();
    const expect = new Float32Array(rawData);

    console.log("bin size:" + expect.length)
    console.log(`Sample expect: ${Array.from(expect.subarray(0, 4)).map(n => n.toFixed(3)).join(' ')}`)
    console.log(`Sample buffer: ${Array.from(data[0].subarray(0, 4)).map(n => n.toFixed(3)).join(' ')}`)

    if (count * data[0].length != expect.length) {
        console.log(data);
        throw `${name} size: ${count * data[0].length}!=${expect.length}`;
    }
    let error_count = 0;
    let error_count_fatal = 0;
    let max_error = 0;
    let error_sum = 0;
    for (let c = 0; c < count; c++)
        for (let i = 0; i < data[0].length; i++) {
            const r = CompareNumber(expect[c * data[0].length + i], data[c][i]);
            if (!r) {
                error_count++;
                let error = Error(expect[c * data[0].length + i], data[c][i]);
                error_sum += error;
                if (error > max_error) { max_error = error; }
                if (!CompareNumber(expect[c * data[0].length + i], data[c][i], 0.1)) {
                    error_count_fatal++;
                    console.log(`[${c},${i}]Expect: ${expect[c * data[0].length + i]} | Got: ${data[c][i]}`);
                }
                if (error_count_fatal > 10) {
                    break;
                }
            }
        }
    if (error_count_fatal == 0) {
        if (error_count == 0) {
            console.log("Success: " + name);
        } else {
            console.log("Warn: " + name + `\n average error: ${error_sum / error_count}\n max error: ${max_error}`);
        }
    }
    else {
        // console.log(expect);
        // console.log(data);
        //throw "Failure: " + name;
        console.log("Failure: " + name)
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

async function VerifyIntIntBuffer(name: string, data: any) {
    const rawData = await (await fetch(name)).arrayBuffer();
    const expect = new Int32Array(rawData);

    console.log("bin size:" + expect.length)
    console.log(`Sample: ${Array.from(expect.subarray(0, 4)).join(' ')}`)

    if (data.length * data[0].length != expect.length) {
        throw `${name} size: ${data.length * data[0].length}!=${expect.length}`;
    }
    let error_count = 0;
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            const r = CompareNumber(expect[i * data[i].length + j], data[i][j]);
            if (!r) {
                error_count++;
                console.log(`[${i}]Expect: ${expect[i]} | Got: ${data[i]}`);
            }
        }
        if (error_count > 10) {
            break;
        }
    }
    if (error_count == 0) {
        console.log("Success: " + name);
    }
    else {
        throw "Failure: " + name;
    }
}