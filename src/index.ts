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
    const nodes = await GetNodes(10000);
    const links = await GetLinks();

    const solver = new FMMSolver(nodes,"wgpu");
    await solver.main();

    // const tester = new Tester();
    // await tester.Test(solver);

    //const renderer = new NodeLinkRenderer();
    //renderer.setData(nodes, links, null);
    //await renderer.init(canvas);

}
main();
