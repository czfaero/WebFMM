import { NodeLinkRenderer } from './NodeLinkRenderer';

//import { GetNodes, GetLinks } from './diagrams/BinaryLoader'
//import { GetNodes, GetLinks, GetNodeColors } from './diagrams/TestGraph'
import { GetNodes, GetLinks, GetNodeColors } from './diagrams/Random'
////import { GetNodes, GetLinks,GetNodeColors } from './diagrams/MatrixMarketLoader'

import { FMMSolver } from './FMMSolver';

//import { Tester } from './tester'
import { DataStart, DataStep, DataUpdate, Data_debug_AddWatch } from './Force';
import { Test_CalcALP as Test_CalcALP, Test_AdditionTheorem, Test_MultipoleExpansion } from './FMMKernel_ts/AssociatedLegendrePolyn';

import 'bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import { InitMenu } from './menu';


async function main() {

    let settings = {
        autoStart: false,
        webgpuAvalible: navigator.gpu ? true : false,
        StartUp: StartUp,
        NextSetp: DataStep,
    }
    InitMenu(settings);
}
main();


async function StartUp(Log_render, Log_data) {
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    const nodes = await GetNodes();
    const links = await GetLinks();
    const colors = await GetNodeColors();
    DataStart(Log_data);
    const renderer = new NodeLinkRenderer();
    renderer.log_callback = Log_render;
    renderer.setData(nodes, links, colors);
    renderer.setDataUpdate(DataUpdate);
    await renderer.init(canvas);

}

function Test() {
    Test_CalcALP(0.5);
    Test_CalcALP(1);
    Test_CalcALP(3.14);
    Test_CalcALP(0);
    const v3 = (a, b, c) => {
        if (Array.isArray(a)) {
            return { x: a[0], y: a[1], z: a[2] }
        }
        return { x: a, y: b, z: c }
    }
    Test_MultipoleExpansion(v3(-25, -18.5, -10), v3(-18.5, -6, -3.5));
    Test_MultipoleExpansion(v3(-18.5, -6, -3.5), v3(-25, -18.5, -10));
    Test_AdditionTheorem(v3(2, 0, 0), v3(2, 0, 2));
    Test_AdditionTheorem(v3(-25, -18.5, -10), v3(-18.5, -6, -3.5));
}