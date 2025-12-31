import { INodeLinkDataProvider } from "../INodeLinkDataProvider";



class TestRNG {
    state: number;
    m: number;
    a: number;
    c: number;
    NextInt() {
        this.state = (this.a * this.state + this.c) % this.m;
        return this.state;
    }
    NextFloat() {
        return this.NextInt() / (this.m - 1);
    }
    constructor(seed: number = 0) {
        this.state = seed;
        this.m = 0x80000000; // 2**31;
        this.a = 1103515245;
        this.c = 12345;
    }
}


export class RandomNodeLinkDataProvider implements INodeLinkDataProvider {
    range = 5;
    nodeCount = 100000;
    seed = 12;
    constructor() {

    }
    GetNodes() {
        const range = this.range;
        const rng = new TestRNG(this.seed);
        const nodeCount = this.nodeCount;
        const iterable = (function* () {
            for (let i = 0; i < nodeCount; i++) {
                yield rng.NextFloat() * range - range / 2;
                yield rng.NextFloat() * range - range / 2;
                yield rng.NextFloat() * range - range / 2;
                yield 0.5;
            }
        })();
        const nodes = new Float32Array(iterable);
        //console.log(nodes.length);
        return nodes;
    }
    GetLinks() {
        const nodeCount = this.nodeCount;
        const rng = new TestRNG(this.seed);
        const max = Math.floor(nodeCount / 10);
        const iterable = (function* () {
            for (let i = 0; i < nodeCount; i++) {
                let len = rng.NextInt() % max;
                const pos = rng.NextInt() % (i + 1);
                let end = i + len;
                yield pos;
                yield i + 1;
                for (; i < end - 1 && i < nodeCount - 1; i++) {
                    yield i + 1;
                    yield i + 2;
                }
                i++;
            }
        })();

        const links = new Uint32Array(iterable);
        return links;
    }
    GetNodeColors() {
        const nodeCount = this.nodeCount;
        let r = 1, g = 0, b = 0, t = 0;
        const iterable = (function* () {
            for (let x = 0; x < nodeCount; x++) {
                yield r;
                yield g;
                yield b;
                t = g;
                g = r;
                r = b;
                b = t;
            }
        })();
        const colors = new Float32Array(iterable);
        return colors;
    }
    GetInfo() {
        return {
            nodeCount: this.nodeCount,
            linkCount: Math.floor(this.nodeCount / 10)
        };
    }
}


