import { NodeLinkRenderer } from './NodeLinkRenderer';

import { GetNodes, GetLinks, GetNodeColors } from './diagrams/MatrixMarketLoader'

import { FMMSolver } from './FMMSolver';

async function main() {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const nodes = await GetNodes();
    const links = await GetLinks();
    const nodeColors = await GetNodeColors();

    const solver = new FMMSolver();


    const renderer = new NodeLinkRenderer();
    renderer.setData(nodes, links, nodeColors);
    await renderer.init(canvas);


}
main();
