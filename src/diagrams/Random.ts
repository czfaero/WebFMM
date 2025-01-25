const range = 5;
const nodeCount = 100000;
const seed = 333;

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

export const GetNodes = function () {
    const rng = new TestRNG(seed);
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
export const GetLinks = function () {
    const rng = new TestRNG(seed);
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
export const GetNodeColors = function () {
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