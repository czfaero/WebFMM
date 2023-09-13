import { NodeLinkRenderer } from './NodeLinkRenderer';

import { GetNodes, GetLinks } from './diagrams/BinaryLoader'

import { FMMSolver } from './FMMSolver';

import { Tester } from './tester'

async function main() {
    if (!navigator.gpu) {
        const msg = 'This browser does not support WebGPU.';
        document.write(msg);
        throw msg;
    }
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const nodes = await GetNodes(1e5);
    const links = await GetLinks();

    const tester = new Tester();

    const solver = new FMMSolver(nodes, "wgpu");
    // try {
        await tester.Test(solver);
    // } catch (e) {
    //    console.log("stop: " + e)
    // }

    // const solver2 = new FMMSolver(nodes, "ts");
    // try {
    //     await tester.Test(solver2);
    // } catch (e) { throw e; }


    // const renderer = new NodeLinkRenderer();
    // renderer.setData(nodes, links, null);
    // await renderer.init(canvas);

}
main();



function GenTestNodes() {
    const rc = 8;
    const testNodes = new Float32Array(rc * rc * 4);
    for (let i = 0; i < rc * rc; i++) {
        const r = Math.floor(i / rc);
        const c = i % rc;
        testNodes[i * 4] = r;
        testNodes[i * 4 + 1] = c;
        testNodes[i * 4 + 2] = 0.0;
        testNodes[i * 4 + 3] = 1.0;
    }
    return testNodes;
}