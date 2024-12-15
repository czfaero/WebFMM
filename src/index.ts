import { NodeLinkRenderer } from './NodeLinkRenderer';

//import { GetNodes, GetLinks } from './diagrams/BinaryLoader'
import { GetNodes, GetLinks, GetNodeColors } from './diagrams/TestGraph'
////import { GetNodes, GetLinks,GetNodeColors } from './diagrams/MatrixMarketLoader'

import { FMMSolver } from './FMMSolver';

//import { Tester } from './tester'
import { DataStart, DataUpdate, Data_debug_AddWatch } from './Force';
import { CalcALP_Test, Test_AdditionTheorem, Test_MultipoleExpansion } from './AssociatedLegendrePolyn';

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

    // CalcALP_Test(0.5);
    // CalcALP_Test(1);
    // CalcALP_Test(3.14);
    // CalcALP_Test(0);
    const v3 = (a, b, c) => {
        if (Array.isArray(a)) {
            return { x: a[0], y: a[1], z: a[2] }
        }
        return { x: a, y: b, z: c }
    }
    // Test_MultipoleExpansion(v3(-25, -18.5, -10), v3(-18.5, -6, -3.5));
    // Test_MultipoleExpansion(v3(-18.5, -6, -3.5), v3(-25, -18.5, -10));
    // Test_AdditionTheorem(v3(2, 0, 0), v3(2, 0, 2));
    // Test_AdditionTheorem(v3(-25, -18.5, -10), v3(-18.5, -6, -3.5));

    //const tester = new Tester();

    // const solver = new FMMSolver(nodes, "wgpu");
    // solver.kernel.debug = true;
    // solver.main();

    // const solver2 = new FMMSolver(nodes, "ts");
    // try {
    //     await tester.Test(solver2);
    // } catch (e) { throw e; }
    Data_debug_AddWatch(1, 15);
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