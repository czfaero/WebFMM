import { NodeLinkRenderer } from './NodeLinkRenderer';
import { NodeLinkSimulator } from './NodeLinkSimulator';

import { RandomNodeLinkDataProvider } from './diagrams/Random'
import { NodeLinkDataLoader } from './diagrams/DataLoader';


import { FMMSolver } from './FMMSolver';

//import { Tester } from './tester'

import { Test_CalcALP as Test_CalcALP, Test_AdditionTheorem, Test_MultipoleExpansion } from './FMMKernel_ts/AssociatedLegendrePolyn';

import 'bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import { InitMenu } from './menu';
import { DebugMode } from './Debug';
import { INodeLinkDataProvider } from './INodeLinkDataProvider';
import { SquareNodeLinkDataProvider } from './diagrams/Square';
import { PerformanceTestTask } from './PerformanceTest';
import { TreeBuilder } from './TreeBuilder';



/**
 InitMenu() from menu.ts: setup DOM elements.
 --> dataSelectorChange() init data
 --> Click 'Start' Button to call StartUp() 
 */
const dataSelector = {
    random: {
        name: "Pseudo-random",
        Init: () => new RandomNodeLinkDataProvider()
    },
    square: {
        name: "Test Square",
        Init: () => new SquareNodeLinkDataProvider()
    },
    load: {
        name: "Load Data",
        Init: async () => {
            const dp = new NodeLinkDataLoader();
            await dp.LoadJson("testdata.json");
            return dp;
        }
    }
};
var renderer: NodeLinkRenderer;
var dataProvider: INodeLinkDataProvider;
var tree: TreeBuilder;
var treePT: TreeBuilder;
var ptTaskList: Array<PerformanceTestTask> = []
var ptRunning = false;
async function ptLoop() {
    ptRunning = true;
    while (ptTaskList.length > 0) {
        const current = ptTaskList.shift();
        await current.Proc();
    }
    ptRunning = false;
}
async function main() {

    let menuSettings = {
        autoStart: false,
        webgpuAvalible: navigator.gpu ? true : false,
        StartUp: async function (Log_slot0, Log_slot1, Log_slot2) {
            const canvas = document.querySelector("canvas") as HTMLCanvasElement;
            renderer = new NodeLinkRenderer();
            renderer.simulator = new NodeLinkSimulator(dataProvider);
            renderer.log_func = Log_slot1;
            renderer.simulator.log_func = Log_slot2;

            renderer.simulator.debugMode = DebugMode.debugger;

            await renderer.init(canvas);
        },
        NextStep: function () {
            renderer.simulator.Step();

        },
        Pause: function () {
            renderer.simulator.PauseToggle();
        },
        dataSelector: dataSelector,
        dataSelectorChange: async function (value: string) {
            const d = dataSelector[value];
            if (d) {
                dataProvider = await d.Init();
                const nodeBuffer = dataProvider.GetNodes();
                const linkBuffer = dataProvider.GetLinks();
                const nodeColorBuffer = dataProvider.GetNodeColors();
                tree = new TreeBuilder(nodeBuffer, linkBuffer, nodeColorBuffer)
                if (renderer) {
                    renderer.Destroy();
                }
                return dataProvider.GetInfo().nodeCount;
            } else {
                throw "unknown data name";
            }
        },
        addPTTask: function (value, onTaskStart, onTaskEnd) {
            if (!treePT) {
                const dataProviderPT = new RandomNodeLinkDataProvider();
                dataProviderPT.nodeCount = 200000;
                const nodeBuffer = dataProviderPT.GetNodes();
                const linkBuffer = dataProviderPT.GetLinks();
                const nodeColorBuffer = dataProviderPT.GetNodeColors();
                treePT = new TreeBuilder(nodeBuffer, linkBuffer, nodeColorBuffer)
            }
            ptTaskList.push(new PerformanceTestTask(value, treePT, onTaskStart, onTaskEnd));
            if (!ptRunning) {
                ptLoop();
            }
        }
    }
    InitMenu(menuSettings);
}
main();






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