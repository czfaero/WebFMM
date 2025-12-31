import { INodeLinkDataProvider } from "../INodeLinkDataProvider";


export class SquareNodeLinkDataProvider implements INodeLinkDataProvider {
    length_a: number;
    length_b: number;
    length_c: number;
    interval_a: number;
    interval_b: number;
    interval_c: number;
    count_a: number;
    count_b: number;
    count_c: number;
    count: number;
    constructor() {
        this.SetByLengthCount(5, 5, 5, 50, 50, 50);
    }
    SetByLengthCount(
        length_a: number, length_b: number, length_c: number,
        count_a: number, count_b: number, count_c: number
    ) {
        count_a = Math.floor(count_a);
        count_b = Math.floor(count_b);
        count_c = Math.floor(count_c);
        this.length_a = length_a;
        this.length_b = length_b;
        this.length_c = length_c;
        this.count_a = count_a;
        this.count_b = count_b;
        this.count_c = count_c;
        this.interval_a = this.length_a / (this.count_a - 1);
        this.interval_b = this.length_b / (this.count_b - 1);
        this.interval_c = this.length_c / (this.count_c - 1);
        this.count = count_a * count_b * count_c;
    }
    GetNodes() {
        const baseX = -this.length_a / 2,
            baseY = -this.length_b / 2,
            baseZ = -this.length_c / 2;
        const _ = this;
        const iterable = (function* () {
            for (let x = 0; x < _.count_a; x++)
                for (let y = 0; y < _.count_b; y++)
                    for (let z = 0; z < _.count_c; z++) {
                        yield x * _.interval_a + baseX;
                        yield y * _.interval_b + baseY;
                        yield z * _.interval_c + baseZ;
                        yield 1;
                    }
        })();
        const nodes = new Float32Array(iterable);
        return nodes;
    }
    GetLinks() {
        const _ = this;
        const iterable = (function* () {
            for (let x = 0; x < _.count_a; x++)
                for (let y = 0; y < _.count_b; y++)
                    for (let z = 0; z < _.count_c - 1; z++) {
                        let i = x * _.count_b * _.count_c + y * _.count_c + z;
                        yield i;
                        yield i + 1;
                    }
            for (let x = 0; x < _.count_a; x++)
                for (let y = 0; y < _.count_b - 1; y++) {
                    let i = x * _.count_b * _.count_c + y * _.count_c;
                    yield i;
                    yield i + _.count_c;
                }
            for (let x = 0; x < _.count_a - 1; x++) {
                let i = x * _.count_b * _.count_c;
                yield i;
                yield i + _.count_b * _.count_c;
            }

        })();
        const links = new Uint32Array(iterable);
        return links;
    }
    GetNodeColors() {
        const _ = this;
        let r = 1, g = 0, b = 0, t = 0;
        const iterable = (function* () {
            for (let x = 0; x < _.count_a; x++)
                for (let y = 0; y < _.count_b; y++)
                    for (let z = 0; z < _.count_c; z++) {
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
        const linkCount =
            this.count_a * this.count_b * (this.count_c - 1)
            + this.count_a * (this.count_b - 1)
            + (this.count_a - 1)
            ;
        return {
            nodeCount: this.count,
            linkCount: linkCount
        };
    }
}