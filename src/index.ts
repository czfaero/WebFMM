import { NodeLinkRenderer } from './NodeLinkRenderer';

import { GetNodes, GetLinks } from './diagrams/BinaryLoader'

import { FMMSolver } from './FMMSolver';

import { Tester } from './tester'

async function main() {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const nodes = await GetNodes(10000);
    const links = await GetLinks();

    const solver = new FMMSolver(nodes);

    const tester = new Tester();
    await tester.Test(solver);

    //const renderer = new NodeLinkRenderer();
    //renderer.setData(nodes, links, null);
    //await renderer.init(canvas);

}
main();
