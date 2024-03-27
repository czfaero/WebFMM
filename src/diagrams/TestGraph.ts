const a = 10, b = 10, c = 4;
const interval = 0.1;
const count = a * b * c;
export const GetNodes = function () {

    const baseX = -Math.floor(a / 2) * interval,
        baseY = -Math.floor(b / 2) * interval,
        baseZ = -Math.floor(c / 2) * interval;
    const iterable = (function* () {

        for (let x = 0; x < a; x++)
            for (let y = 0; y < b; y++)
                for (let z = 0; z < c; z++) {
                    yield x + baseX;
                    yield y + baseY;
                    yield z + baseZ;
                    yield 1;
                }
    })();
    const nodes = new Float32Array(iterable);
    //console.log(nodes.length);
    return nodes;
}
export const GetLinks = function () {
    const iterable = (function* () {
        for (let x = 0; x < a; x++)
            for (let y = 0; y < b; y++)
                for (let z = 0; z < c - 1; z++) {
                    let i = x * b * c + y * c + z;
                    yield i;
                    yield i + 1;
                }
        for (let x = 0; x < a; x++)
            for (let y = 0; y < b - 1; y++) {
                let i = x * b * c + y * c;
                yield i;
                yield i + c;

            }

            for (let x = 0; x < a-1; x++){
                let i = x * b * c;
                yield i;
                yield i + b*c;
            }

    })();


    const links = new Uint32Array(iterable);

    return links;
}
export const GetNodeColors = function () {
    let r = 1, g = 0, b = 0, t;
    const iterable = (function* () {
        for (let i = 0; i < count; i++) {
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
    console.log(colors.length);
    return colors;
}