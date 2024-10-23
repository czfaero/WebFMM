import { NodeLinkRenderer } from './NodeLinkRenderer';

//import { GetNodes, GetLinks } from './diagrams/BinaryLoader'
import { GetNodes, GetLinks, GetNodeColors } from './diagrams/TestGraph'
////import { GetNodes, GetLinks,GetNodeColors } from './diagrams/MatrixMarketLoader'

import { FMMSolver } from './FMMSolver';

//import { Tester } from './tester'
import { DataStart, DataUpdate, Data_debug_SetBox } from './Force';

async function main() {
    if (!navigator.gpu) {
        const msg = 'This browser does not support WebGPU.';
        document.write(msg);
        throw msg;
    }
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const nodes = await GetNodes();
    const links = await GetLinks();
    const colors = await GetNodeColors();

    //const tester = new Tester();

    // const solver = new FMMSolver(nodes, "wgpu");
    // solver.kernel.debug = true;
    // solver.main();

    // const solver2 = new FMMSolver(nodes, "ts");
    // try {
    //     await tester.Test(solver2);
    // } catch (e) { throw e; }
    Data_debug_SetBox([1, 15]);
    DataStart();
    const renderer = new NodeLinkRenderer();
    renderer.setData(nodes, links, colors);
    renderer.setDataUpdate(DataUpdate);
    await renderer.init(canvas);




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